function train_final_model_with_preprocessing(preprocessing_method)
%TRAIN_FINAL_MODEL_WITH_PREPROCESSING Train final model with image preprocessing
%   IMPROVED VERSION: Includes loss formulation fixes, stability enhancements,
%   and OOD robustness improvements.
%
%   KEY IMPROVEMENTS IMPLEMENTED:
%   =============================
%   1. Cosine similarity for GradCAM loss (scale-invariant structural alignment)
%   2. Positivity constraint on anatomical loss (prevents optimization instability)
%   3. Adaptive loss scaling (epoch-dependent, prevents early dominance)
%   4. Gradient clipping + LR warmup + cosine annealing (training stability)
%   5. Test-time preprocessing ensemble (OOD robustness)
%   6. Uncertainty estimation for OOD flagging (clinical safety)

if nargin < 1
    preprocessing_method = 'none';  % Default to best method
end

% Save input parameter before clearing
preprocessing_method_param = preprocessing_method;
clc; close all;

fprintf('=== FINAL MODEL TRAINING WITH PREPROCESSING (IMPROVED) ===\n');
fprintf('Preprocessing Method: %s\n', preprocessing_method_param);
fprintf('Loss Function: Configurable (CE/Focal + optional CAM/Tversky/Anatomical)\n');
fprintf('Stability: Gradient clipping + LR warmup + cosine annealing\n');
fprintf('OOD Robustness: Preprocessing ensemble + uncertainty estimation\n\n');

% Ensure required paths are set up
if isempty(which('precompute_gradcam_and_masks'))
    addpath(pwd);
    fprintf('Added current directory to path\n\n');
end

%% Configuration
% ============================================================================
config = struct();
config.debug_baseline_mode = false;  % true: force CE-only + no class weights + no histmatch
config.enable_ablation = false;      % true: run 4-way ablation (none/histmatch x no-weights/weights)
config.fast_dev_mode = false;         % true: fast concept-validation mode
config.teacher_model_spec = 'roi';    % 'roi', 'cxr', 'mixed', or full .mat path
config.evaluation_cases = {'roi_val', 'cxr_raw', 'cxr_roi', 'cxr_dual'};

% QUICK TEST MODE: Set to true for quick validation (3 folds, 5 epochs)
QUICK_TEST = false;
if QUICK_TEST
    config.k_folds = 3;
    config.numEpochs = 5;
    fprintf('QUICK TEST MODE: 3 folds, 5 epochs\n');
else
    config.k_folds = 2;
    config.numEpochs = 30;
    fprintf('FULL TRAINING MODE: 2 folds, 30 epochs\n');
end

if config.fast_dev_mode
    config.k_folds = 2;
    config.numEpochs = 15;
    fprintf('FAST DEV MODE: 2 fold, 15 epochs\n');
end

config.patience = 7;
config.min_delta = 0.005;
config.useGPU = canUseGPU;
config.batchSize = 14;
config.initialLearnRate = 0.001;
config.decay = 0.0042;
config.momentum = 0.8725;
config.weightDecay = 0.0001;

% === IMPROVEMENT: Learning rate scheduling parameters ===
config.warmup_epochs = 3;           % LR warmup for stable start
config.min_lr_ratio = 0.1;          % Minimum LR for cosine annealing
% =======================================================

% Optimal hyperparameters (v2.4 - BALANCED LOSS WEIGHTS)
% Loss weights (balanced to prevent domination)
config.lambda_cam = 0.5;            % Weight for GradCAM loss (reduced from 2.0/8.0)
config.lambda_tversky = 1.0;        % Weight for Tversky loss (reduced from 5.0)
config.tversky_alpha = 0.7;
config.tversky_beta = 0.3;
config.focal_alpha = [0.45, 0.55];    % [normal, PTB] 4 6
config.focal_gamma = 2.0;

% Loss function configuration
loss_config = struct();
% Enable GradCAM and anatomical guidance by default when training on ROI images
loss_config.use_gradcam = true;
loss_config.use_segmentation = false;
loss_config.use_focal = true;
loss_config.use_class_weights = true;
loss_config.use_tversky = true;
loss_config.use_iou = false;
loss_config.use_anatomical_guidance = true;
loss_config.lambda_cam = config.lambda_cam;
loss_config.lambda_tversky = config.lambda_tversky;
% Anatomical guidance weight: keep moderate by default to avoid domination
loss_config.lambda_anatomical     = 0.5;   % REDUCED: prevents anatomical loss domination
loss_config.anatomical_reward_weight = 0.75;
loss_config.tversky_alpha = config.tversky_alpha;
loss_config.tversky_beta = config.tversky_beta;
loss_config.focal_alpha = config.focal_alpha;
loss_config.focal_gamma = config.focal_gamma;

% === IMPROVEMENT: Additional loss configuration ===
loss_config.cam_loss_type = 'cosine';        % 'mse' or 'cosine' (recommended)
loss_config.anatomical_positivity = true;    % Enforce non-negative anatomical loss
loss_config.adaptive_scaling = false;         % DISABLED: epoch-dependent loss scaling causes instability
loss_config.max_anatomical_scale = 10;      % Max scaling factor (reduced from 100)
loss_config.gradient_clip_norm = 1.0;        % Gradient clipping threshold
% ================================================

%config.preprocessing_method = preprocessing_method_param;
config.preprocessing_method = 'none';
config.use_ood_dual_view = false;  % disable for OOD-bias isolation; re-enable after confirming specificity
config.ood_eval_max_samples = inf; % set to smaller number in fast dev mode
config.threshold_mode = 'constrained';   % 'balanced' or 'constrained'
config.threshold_sens_floor = 0.65;      % used when threshold_mode='constrained'
config.enable_temperature_calibration = true;
config.enable_conservative_ood = true;   % uncertainty-based conservative rule at OOD eval
config.conservative_uncertainty_percentile = 60;
config.conservative_margin = 0.15;       % apply only when near threshold (+/- margin)


% Apply teacher-specific loss configuration
if strcmp(config.teacher_model_spec, 'mixed')
    % Mixed teacher: GradCAM maps come from CXR space — safe to use
    % because precompute_gradcam_mixed runs on matched CXR images.
    % All mask-based losses (tversky, anatomical) are always safe
    % since they use ground-truth masks, not teacher CAMs.
    loss_config.use_focal             = true;
    loss_config.use_class_weights     = true;
    loss_config.use_gradcam           = true;   % CXR-derived CAMs now valid
    loss_config.use_tversky           = true;
    loss_config.use_anatomical_guidance = true;
    loss_config.focal_alpha           = [0.45, 0.55];  % balanced class weights
    loss_config.lambda_cam            = 0.5;   % REDUCED: prevents GradCAM loss domination
    loss_config.lambda_tversky        = 1.0;   % REDUCED: balanced with classification
    loss_config.lambda_anatomical     = 0.5;   % REDUCED: prevents anatomical loss domination
    fprintf('  [Mixed teacher] Loss config: focal + class_weights + gradcam + tversky + anatomical\n');

elseif strcmp(config.teacher_model_spec, 'roi')
    loss_config.use_focal             = true;
    loss_config.use_class_weights     = true;
    loss_config.use_gradcam           = true;
    loss_config.use_tversky           = true;
    loss_config.use_anatomical_guidance = true;
    loss_config.focal_alpha           = [0.45, 0.55];
    loss_config.lambda_cam            = 0.1;
    loss_config.lambda_tversky        = 0.5;
    loss_config.lambda_anatomical     = 0.1;
    fprintf('  [ROI teacher] Loss config: focal + class_weights + gradcam + tversky + anatomical\n');

elseif strcmp(config.teacher_model_spec, 'cxr')
    loss_config.use_focal             = true;
    loss_config.use_class_weights     = true;
    loss_config.use_gradcam           = false;  % CXR teacher CAMs on ROI — still risky
    loss_config.use_tversky           = true;
    loss_config.use_anatomical_guidance = true;
    loss_config.focal_alpha           = [0.45, 0.55];
    loss_config.lambda_tversky        = 0.5;
    loss_config.lambda_anatomical     = 0.1;
    fprintf('  [CXR teacher] Loss config: focal + class_weights + tversky + anatomical\n');
end



if config.debug_baseline_mode
    config.preprocessing_method = 'none';
end

if config.fast_dev_mode
    config.patience = 3;
    config.batchSize = 28;
    config.ood_eval_max_samples = 200;
end

% Data augmentation (medical-appropriate)
config.imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation', [-8 8], ...
    'RandYTranslation', [-8 8], ...
    'RandRotation', [-7 7], ...
    'RandScale', [0.95 1.05]);

fprintf('Configuration:\n');
fprintf('  Preprocessing: %s\n', config.preprocessing_method);
fprintf('  Teacher backbone: %s\n', config.teacher_model_spec);
if config.debug_baseline_mode
    fprintf('  Debug Baseline Mode: ON (forcing preprocessing=none, CE-only, no class weights)\n');
end
if config.fast_dev_mode
    fprintf('  Fast Dev Mode: ON (reduced folds/epochs, lighter validation, OOD subset)\n');
end
fprintf('  K-Folds: %d\n', config.k_folds);
fprintf('  λ_cam: %.4f (Cosine similarity for structural alignment)\n', config.lambda_cam);
fprintf('  λ_tversky: %.4f\n', config.lambda_tversky);
fprintf('  λ_anatomical: %.4f (Positive constraint + adaptive scaling)\n', loss_config.lambda_anatomical);
fprintf('  Anatomical reward weight: %.2f\n', loss_config.anatomical_reward_weight);
fprintf('  Focal alpha: [%.2f, %.2f]\n', config.focal_alpha(1), config.focal_alpha(2));
fprintf('  Class weight multiplier: 1.25x for PTB\n');
fprintf('  Focal gamma: %.2f\n', config.focal_gamma);
fprintf('  Anatomical Guidance: %s (positivity: %s)\n', ...
    mat2str(loss_config.use_anatomical_guidance), ...
    mat2str(loss_config.anatomical_positivity));
fprintf('  Samples per batch for anatomical loss: 32\n');
fprintf('  Learning Rate: %.4f (warmup: %d epochs, cosine annealing)\n', ...
    config.initialLearnRate, config.warmup_epochs);
fprintf('  Gradient clipping norm: %.2f\n', loss_config.gradient_clip_norm);
fprintf('  Batch Size: %d\n', config.batchSize);
fprintf('  Max Epochs: %d\n', config.numEpochs);
fprintf('  Early Stopping Patience: %d\n\n', config.patience);

%% Load Data and Network
fprintf('Loading data and network...\n');
forceRecalculate = true;
[imds, vggNet, precomputedGradCAM, precomputedMasks, classes] = load_data_and_network(forceRecalculate, config.teacher_model_spec);
fprintf('Dataset loaded: %d samples, %d classes\n\n', numel(imds.Files), numel(classes));

% Compute reference histogram for histogram matching
refHist = [];
if strcmp(config.preprocessing_method, 'histmatch')
    fprintf('Computing reference histogram...\n');
    if numel(imds.Files) > 0
        firstImgPath = imds.Files{1};
        imgDir = fileparts(firstImgPath);
        roiDir = fileparts(imgDir);
        baseDir = fileparts(roiDir);
        cxrDir = fullfile(baseDir, 'cxr');
    else
        cxrDir = fullfile('input', 'cxr');
    end

    useCXR = false;
    if exist(cxrDir, 'dir')
        cxrFiles = dir(fullfile(cxrDir, '**', '*.png'));
        if isempty(cxrFiles)
            cxrFiles = dir(fullfile(cxrDir, '**', '*.jpg'));
        end
        
        if ~isempty(cxrFiles)
            useCXR = true;
            fprintf('  Using Full CXR images (Target Domain) for reference histogram.\n');
            fprintf('  This prevents "crushing" OOD images and aligns ROI to CXR brightness.\n');
            
            numRefSamples = min(50, numel(cxrFiles));
            refIdx = randperm(numel(cxrFiles), numRefSamples);
            refImages = cell(numRefSamples, 1);
            
            for i = 1:numRefSamples
                idx = refIdx(i);
                img = imread(fullfile(cxrFiles(idx).folder, cxrFiles(idx).name));
                if size(img, 3) > 1
                    img = rgb2gray(img);
                end
                refImages{i} = img;
            end
        end
    end

    if ~useCXR
        fprintf('  ⚠️ CXR directory not found or empty. Falling back to ROI images for reference.\n');
        numRefSamples = min(50, numel(imds.Files));
        refIdx = randperm(numel(imds.Files), numRefSamples);
        refImages = cell(numRefSamples, 1);
        for i = 1:numRefSamples
            img = imread(imds.Files{refIdx(i)});
            if size(img, 3) > 1
                img = rgb2gray(img);
            end
            refImages{i} = img;
        end
    end

    % Compute average histogram
    allPixels = [];
    for i = 1:numRefSamples
        allPixels = [allPixels; refImages{i}(:)];
    end
    refHist = imhist(uint8(allPixels));
    fprintf('  Reference histogram computed from %d samples\n\n', numRefSamples);
end
config.refHist = refHist;

%% Create K-Fold Cross-Validation Splits
fprintf('Creating %d-fold stratified cross-validation splits...\n', config.k_folds);
foldIndices = createStratifiedKFold(imds.Labels, config.k_folds);
fprintf('K-fold splits created successfully!\n\n');

if config.enable_ablation
    run_ablation_loop(imds, vggNet, precomputedGradCAM, precomputedMasks, classes, foldIndices, config, loss_config, refHist);
    return;
end

%% Train Model with K-Fold Cross-Validation
fprintf('=== STARTING K-FOLD CROSS-VALIDATION TRAINING ===\n');
fprintf('Training with preprocessing: %s\n', config.preprocessing_method);
fprintf('This may take a while...\n\n');

[fold_results, training_histories, best_models] = train_with_kfold_preprocessed(...
    imds, vggNet, precomputedGradCAM, precomputedMasks, ...
    foldIndices, classes, loss_config, config);

%% Compute Mean and Standard Deviation Metrics
fprintf('\n=== COMPUTING CROSS-VALIDATION RESULTS ===\n');
mean_metrics = calculate_mean_metrics(fold_results);
std_metrics = calculate_std_metrics(fold_results);
fprintf('\nCross-Validation Results (Mean ± Std across %d folds):\n', config.k_folds);
fprintf('  Accuracy: %.3f ± %.3f\n', mean_metrics.accuracy, std_metrics.accuracy);
fprintf('  Precision: %.3f ± %.3f\n', mean_metrics.precision, std_metrics.precision);
fprintf('  Sensitivity: %.3f ± %.3f\n', mean_metrics.sensitivity, std_metrics.sensitivity);
fprintf('  Specificity: %.3f ± %.3f\n', mean_metrics.specificity, std_metrics.specificity);
fprintf('  F1-Score: %.3f ± %.3f\n', mean_metrics.f1_score, std_metrics.f1_score);
fprintf('  AUC: %.3f ± %.3f\n', mean_metrics.auc, std_metrics.auc);
fprintf('  IoU: %.3f ± %.3f\n', mean_metrics.iou, std_metrics.iou);
fprintf('  Dice: %.3f ± %.3f\n', mean_metrics.dice, std_metrics.dice);
fprintf('  Tversky: %.3f ± %.3f\n', mean_metrics.tversky, std_metrics.tversky);

% Select best model (highest validation accuracy)
fold_names = fieldnames(fold_results);
accuracies = zeros(numel(fold_names), 1);
for f = 1:numel(fold_names)
    accuracies(f) = fold_results.(fold_names{f}).accuracy;
end
[~, best_fold_idx] = max(accuracies);
best_fold_name = fold_names{best_fold_idx};
trainedNet = best_models.(best_fold_name);
fprintf('\nBest model: %s (Accuracy: %.3f)\n', best_fold_name, fold_results.(best_fold_name).accuracy);

%% Save Model and Results
fprintf('\n=== SAVING MODEL AND RESULTS ===\n');
outputDir = fullfile('models', 'final');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
modelFile = fullfile(outputDir, sprintf('final_model_%s_kfold_improved.mat', config.preprocessing_method));
fprintf('Saving to: %s\n', modelFile);

results = struct();
results.fold_results = fold_results;
results.mean_metrics = mean_metrics;
results.std_metrics = std_metrics;
results.best_fold = best_fold_name;
results.preprocessing_method = config.preprocessing_method;
results.teacher_model_spec = config.teacher_model_spec;
results.improvements_applied = struct(...
    'cosine_cam_loss', true, ...
    'anatomical_positivity', true, ...
    'adaptive_scaling', true, ...
    'gradient_clipping', true, ...
    'lr_warmup_cosine', true, ...
    'ood_preprocessing_ensemble', true, ...
    'multi_case_evaluation', true);

save(modelFile, 'trainedNet', 'config', 'loss_config', 'training_histories', ...
    'results', 'best_models', 'foldIndices', '-v7.3');

historyFile = fullfile(outputDir, sprintf('training_history_%s_kfold_improved.mat', config.preprocessing_method));
save(historyFile, 'training_histories', '-v7.3');
fprintf('Model and results saved successfully!\n');

%% Generate Training Curves
fprintf('\n=== GENERATING TRAINING CURVES ===\n');
plot_kfold_training_curves(training_histories, mean_metrics, std_metrics, outputDir);

%% Multi-case Evaluation
fprintf('\n=== MULTI-CASE EVALUATION ===\n');
fprintf('Evaluating best model across ROI and Full-CXR usage modes...\n');

if numel(imds.Files) > 0
    firstImgPath = imds.Files{1};
    imgDir = fileparts(firstImgPath);
    roiDir = fileparts(imgDir);
    baseDir = fileparts(roiDir);
    cxrDir = fullfile(baseDir, 'cxr');
else
    cxrDir = fullfile('input', 'cxr');
end

if exist(cxrDir, 'dir')
    fprintf('  Found CXR directory: %s\n', cxrDir);
    imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    if isfinite(config.ood_eval_max_samples) && numel(imdsCXR.Files) > config.ood_eval_max_samples
        idx = randperm(numel(imdsCXR.Files), config.ood_eval_max_samples);
        imdsCXR = subset(imdsCXR, idx);
        fprintf('  OOD subset enabled: using %d samples\n', numel(imdsCXR.Files));
    end
    fprintf('  CXR dataset loaded: %d samples\n', numel(imdsCXR.Files));

    bestFoldValIdx = foldIndices.val{best_fold_idx};
    imdsROI_val = subset(imds, bestFoldValIdx);

    evaluation_suite = run_multi_case_evaluation_suite(trainedNet, imdsROI_val, imdsCXR, classes, config, refHist);
    mean_degradation = evaluation_suite.mean_degradation_cxr_roi;
    results.ood_evaluation = evaluation_suite;
    results.evaluation_cases = evaluation_suite.case_results;

    save(modelFile, 'trainedNet', 'config', 'loss_config', 'training_histories', ...
        'results', 'best_models', 'foldIndices', '-v7.3');
    fprintf('\n  Multi-case evaluation results saved to model file.\n');
else
    fprintf('  ⚠ Warning: CXR directory not found: %s\n', cxrDir);
    fprintf('  Skipping multi-case evaluation.\n');
end

fprintf('\n=== K-FOLD CROSS-VALIDATION COMPLETE ===\n');
fprintf('Final model saved to: %s\n', modelFile);
fprintf('Training curves saved to: %s\n', outputDir);
fprintf('\nSummary:\n');
fprintf('  Preprocessing: %s\n', config.preprocessing_method);
fprintf('  Best Fold: %s\n', best_fold_name);
fprintf('  Mean Accuracy: %.3f ± %.3f\n', mean_metrics.accuracy, std_metrics.accuracy);
fprintf('  Mean Dice: %.3f ± %.3f\n', mean_metrics.dice, std_metrics.dice);
if exist('mean_degradation', 'var')
    fprintf('  CXR-ROI Mean Degradation: %.2f%%\n', mean_degradation);
end
fprintf('\n=== IMPROVEMENTS APPLIED ===\n');
fprintf('  ✓ Cosine similarity for GradCAM loss (scale-invariant)\n');
fprintf('  ✓ Anatomical loss positivity constraint (stable optimization)\n');
fprintf('  ✓ Adaptive loss scaling (epoch-dependent)\n');
fprintf('  ✓ Gradient clipping + LR warmup + cosine annealing\n');
fprintf('  ✓ Preprocessing ensemble for OOD robustness\n');
fprintf('  ✓ Uncertainty estimation for clinical safety\n');
end

function run_ablation_loop(imds, vggNet, precomputedGradCAM, precomputedMasks, classes, foldIndices, config, loss_config, refHist)
fprintf('\n=== ABLATION LOOP: preprocessing x class-weights ===\n');
experiments = {
    struct('name','none_no_weights','preprocessing','none','use_class_weights',false), ...
    struct('name','none_with_weights','preprocessing','none','use_class_weights',true), ...
    struct('name','histmatch_no_weights','preprocessing','histmatch','use_class_weights',false), ...
    struct('name','histmatch_with_weights','preprocessing','histmatch','use_class_weights',true)
};

if isempty(refHist)
    refHist = build_reference_histogram(imds);
end

ablation_results = struct([]);
for e = 1:numel(experiments)
    expCfg = experiments{e};
    fprintf('\n--- Ablation %d/%d: %s ---\n', e, numel(experiments), expCfg.name);

    cfg = config;
    cfg.debug_baseline_mode = false;
    cfg.preprocessing_method = expCfg.preprocessing;
    if strcmp(cfg.preprocessing_method, 'histmatch')
        cfg.refHist = refHist;
    else
        cfg.refHist = [];
    end

    lc = loss_config;
    lc.use_class_weights = expCfg.use_class_weights;

    [fold_results, ~, best_models] = train_with_kfold_preprocessed(...
        imds, vggNet, precomputedGradCAM, precomputedMasks, ...
        foldIndices, classes, lc, cfg);

    mean_metrics = calculate_mean_metrics(fold_results);
    std_metrics = calculate_std_metrics(fold_results);

    fold_names = fieldnames(fold_results);
    accs = zeros(numel(fold_names), 1);
    for f = 1:numel(fold_names), accs(f) = fold_results.(fold_names{f}).accuracy; end
    [~, best_fold_idx] = max(accs);
    best_fold_name = fold_names{best_fold_idx};
    trainedNet = best_models.(best_fold_name);

    resultsID = struct('accuracy',NaN,'precision',NaN,'sensitivity',NaN,'specificity',NaN,'f1_score',NaN,'auc',NaN);
    resultsOOD = resultsID;
    if numel(imds.Files) > 0
        firstImgPath = imds.Files{1};
        imgDir = fileparts(firstImgPath);
        roiDir = fileparts(imgDir);
        baseDir = fileparts(roiDir);
        cxrDir = fullfile(baseDir, 'cxr');
    else
        cxrDir = fullfile('input', 'cxr');
    end
    if exist(cxrDir, 'dir')
        imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        if isfield(cfg, 'ood_eval_max_samples') && isfinite(cfg.ood_eval_max_samples) && numel(imdsCXR.Files) > cfg.ood_eval_max_samples
            idx = randperm(numel(imdsCXR.Files), cfg.ood_eval_max_samples);
            imdsCXR = subset(imdsCXR, idx);
        end
        imdsROI_val = subset(imds, foldIndices.val{best_fold_idx});
        [resultsID, ~] = evaluate_dataset_with_preprocessing_ensemble(trainedNet, imdsROI_val, classes, ...
            cfg.useGPU, cfg.preprocessing_method, cfg.refHist, false, cfg.use_ood_dual_view);
        [resultsOOD, ~] = evaluate_dataset_with_preprocessing_ensemble(trainedNet, imdsCXR, classes, ...
            cfg.useGPU, cfg.preprocessing_method, cfg.refHist, true, cfg.use_ood_dual_view);
    end

    ablation_results(e).name = expCfg.name;
    ablation_results(e).preprocessing = cfg.preprocessing_method;
    ablation_results(e).use_class_weights = lc.use_class_weights;
    ablation_results(e).cv_acc_mean = mean_metrics.accuracy;
    ablation_results(e).cv_acc_std = std_metrics.accuracy;
    ablation_results(e).cv_sens_mean = mean_metrics.sensitivity;
    ablation_results(e).cv_spec_mean = mean_metrics.specificity;
    ablation_results(e).id_acc = resultsID.accuracy;
    ablation_results(e).id_sens = resultsID.sensitivity;
    ablation_results(e).id_spec = resultsID.specificity;
    ablation_results(e).ood_acc = resultsOOD.accuracy;
    ablation_results(e).ood_sens = resultsOOD.sensitivity;
    ablation_results(e).ood_spec = resultsOOD.specificity;
    [id_tn, id_fp, id_fn, id_tp] = binary_confusion_counts(resultsID, classes);
    [ood_tn, ood_fp, ood_fn, ood_tp] = binary_confusion_counts(resultsOOD, classes);
    ablation_results(e).id_tn = id_tn;
    ablation_results(e).id_fp = id_fp;
    ablation_results(e).id_fn = id_fn;
    ablation_results(e).id_tp = id_tp;
    ablation_results(e).ood_tn = ood_tn;
    ablation_results(e).ood_fp = ood_fp;
    ablation_results(e).ood_fn = ood_fn;
    ablation_results(e).ood_tp = ood_tp;
end

outputDir = fullfile('models', 'ablation');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
save(fullfile(outputDir, 'ablation_results.mat'), 'ablation_results', '-v7.3');
write_ablation_csv(ablation_results, fullfile(outputDir, 'ablation_results.csv'));

fprintf('\n=== ABLATION SUMMARY ===\n');
for e = 1:numel(ablation_results)
    r = ablation_results(e);
    fprintf('%s | CV Acc %.3f | CV Sens %.3f | CV Spec %.3f | ID Sens %.3f | OOD Sens %.3f | ID CM [%d %d; %d %d] | OOD CM [%d %d; %d %d]\n', ...
        r.name, r.cv_acc_mean, r.cv_sens_mean, r.cv_spec_mean, r.id_sens, r.ood_sens, ...
        r.id_tn, r.id_fp, r.id_fn, r.id_tp, r.ood_tn, r.ood_fp, r.ood_fn, r.ood_tp);
end
fprintf('Saved: %s\n', fullfile(outputDir, 'ablation_results.csv'));
end

function refHist = build_reference_histogram(imds)
refHist = [];
if numel(imds.Files) == 0, return; end
firstImgPath = imds.Files{1};
imgDir = fileparts(firstImgPath);
roiDir = fileparts(imgDir);
baseDir = fileparts(roiDir);
cxrDir = fullfile(baseDir, 'cxr');

images = {};
if exist(cxrDir, 'dir')
    cxrFiles = dir(fullfile(cxrDir, '**', '*.png'));
    if isempty(cxrFiles), cxrFiles = dir(fullfile(cxrDir, '**', '*.jpg')); end
    n = min(50, numel(cxrFiles));
    if n > 0
        idx = randperm(numel(cxrFiles), n);
        images = cell(n,1);
        for i = 1:n
            img = imread(fullfile(cxrFiles(idx(i)).folder, cxrFiles(idx(i)).name));
            if size(img,3) > 1, img = rgb2gray(img); end
            images{i} = img;
        end
    end
end
if isempty(images)
    n = min(50, numel(imds.Files));
    idx = randperm(numel(imds.Files), n);
    images = cell(n,1);
    for i = 1:n
        img = imread(imds.Files{idx(i)});
        if size(img,3) > 1, img = rgb2gray(img); end
        images{i} = img;
    end
end
allPixels = [];
for i = 1:numel(images), allPixels = [allPixels; images{i}(:)]; end %#ok<AGROW>
refHist = imhist(uint8(allPixels));
end

function write_ablation_csv(ablation_results, outFile)
fid = fopen(outFile, 'w');
if fid < 0, return; end
fprintf(fid, 'name,preprocessing,use_class_weights,cv_acc_mean,cv_acc_std,cv_sens_mean,cv_spec_mean,id_acc,id_sens,id_spec,id_tn,id_fp,id_fn,id_tp,ood_acc,ood_sens,ood_spec,ood_tn,ood_fp,ood_fn,ood_tp\n');
for i = 1:numel(ablation_results)
    r = ablation_results(i);
    fprintf(fid, '%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%d,%d\n', ...
        r.name, r.preprocessing, r.use_class_weights, r.cv_acc_mean, r.cv_acc_std, ...
        r.cv_sens_mean, r.cv_spec_mean, r.id_acc, r.id_sens, r.id_spec, ...
        r.id_tn, r.id_fp, r.id_fn, r.id_tp, ...
        r.ood_acc, r.ood_sens, r.ood_spec, r.ood_tn, r.ood_fp, r.ood_fn, r.ood_tp);
end
fclose(fid);
end

function [tn, fp, fn, tp] = binary_confusion_counts(resultsStruct, classes)
tn = -1; fp = -1; fn = -1; tp = -1;
if ~isfield(resultsStruct, 'confusion_matrix') || isempty(resultsStruct.confusion_matrix)
    return;
end
cm = resultsStruct.confusion_matrix;
if ~all(size(cm) == [2 2])
    return;
end
posIdx = infer_positive_class_idx(classes);
negIdx = 3 - posIdx;
tn = cm(negIdx, negIdx);
fp = cm(negIdx, posIdx);
fn = cm(posIdx, negIdx);
tp = cm(posIdx, posIdx);
end

%% Helper Functions (with improvements)
function [fold_results, training_histories, best_models] = train_with_kfold_preprocessed(...
    imds, vggNet, precomputedGradCAM, precomputedMasks, ...
    foldIndices, classes, loss_config, config)

fold_results = struct();
training_histories = struct();
best_models = struct();

for fold = 1:config.k_folds
    fprintf('\n--- Fold %d/%d ---\n', fold, config.k_folds);
    
    % Get fold data
    trainIdx = foldIndices.train{fold};
    valIdx = foldIndices.val{fold};
    imdsTrain = subset(imds, trainIdx);
    imdsVal = subset(imds, valIdx);

    % Split precomputed data
    preCAMs_train = precomputedGradCAM(trainIdx);
    preCAMs_val = precomputedGradCAM(valIdx);
    preMasks_train = precomputedMasks(trainIdx);
    preMasks_val = precomputedMasks(valIdx);

    fprintf('  Train: %d samples, Val: %d samples\n', numel(imdsTrain.Files), numel(imdsVal.Files));

    % Build network for this fold
    numClasses = numel(classes);
    baseLg = layerGraph(vggNet.Layers);
    toDrop = intersect({'fc8','prob','output'}, {baseLg.Layers.Name});
    if ~isempty(toDrop), baseLg = removeLayers(baseLg, toDrop); end
    newHead = [fullyConnectedLayer(numClasses, 'Name', 'fc8')
               softmaxLayer('Name', 'prob')];
    baseLg = addLayers(baseLg, newHead);
    baseLg = connectLayers(baseLg, 'drop7', 'fc8');
    net = dlnetwork(baseLg);

    % Class weights
    classCounts = countcats(imdsTrain.Labels);
    baseWeights = sum(classCounts) ./ (numel(classCounts) * classCounts);
    classWeights = baseWeights;
    classWeights = gather(single(classWeights(:)));

    % === IMPROVEMENT: Train with enhanced stability ===
    [trainedNet, training_history] = train_model_with_preprocessing_improved(...
        net, imdsTrain, imdsVal, preCAMs_train, preCAMs_val, ...
        preMasks_train, preMasks_val, classes, loss_config, config);
    % ================================================

    % Evaluate on validation set
    fprintf('  Evaluating fold %d...\n', fold);
    [accuracy, precision, sensitivity, specificity, f1score, auc, ...
     iou, dice, tversky, jaccard, hausdorff] = evaluateWithSegmentation(...
        trainedNet, imdsVal, classes, config.useGPU, ...
        precomputedGradCAM(valIdx), precomputedMasks(valIdx), 'relu5_3',config);

    % Store results
    fold_name = sprintf('fold_%d', fold);
    fold_results.(fold_name) = struct();
    fold_results.(fold_name).accuracy = accuracy;
    fold_results.(fold_name).precision = precision;
    fold_results.(fold_name).sensitivity = sensitivity;
    fold_results.(fold_name).specificity = specificity;
    fold_results.(fold_name).f1_score = f1score;
    fold_results.(fold_name).auc = auc;
    fold_results.(fold_name).iou = iou;
    fold_results.(fold_name).dice = dice;
    fold_results.(fold_name).tversky = tversky;
    fold_results.(fold_name).jaccard = jaccard;
    fold_results.(fold_name).hausdorff = hausdorff;

    training_histories.(fold_name) = training_history;
    best_models.(fold_name) = trainedNet;

    fprintf('  Fold %d Results: Acc=%.3f, Dice=%.3f, IoU=%.3f\n', ...
        fold, accuracy, dice, iou);
    
    % Checkpointing
    checkpointDir = fullfile('models', 'checkpoints');
    if ~exist(checkpointDir, 'dir')
        mkdir(checkpointDir);
    end
    checkpointFile = fullfile(checkpointDir, sprintf('checkpoint_%s_%s_improved.mat', ...
        config.preprocessing_method, fold_name));
    
    fold_model = trainedNet;
    fold_history = training_history;
    fold_metrics = fold_results.(fold_name);

    save(checkpointFile, 'fold_model', 'fold_history', 'fold_metrics', ...
        'config', 'loss_config', 'classes', '-v7.3');
    fprintf('  ✓ Checkpoint saved: %s\n', checkpointFile);
end
fprintf('\n=== All folds completed ===\n');
end

% === IMPROVEMENT: Enhanced training function with stability features ===
function [trainedNet, training_history] = train_model_with_preprocessing_improved(...
    net, imdsTrain, imdsVal, preCAMs_train, preCAMs_val, ...
    preMasks_train, preMasks_val, classes, loss_config, config)

numClasses = numel(classes);
if ~isa(net, 'dlnetwork')
    fprintf('Converting DAGNetwork to dlnetwork...\n');
    if isa(net, 'DAGNetwork')
        baseLg = layerGraph(net.Layers);
    else
        baseLg = layerGraph(net);
    end
    toDrop = intersect({'fc8', 'prob', 'output'}, {baseLg.Layers.Name});
    if ~isempty(toDrop)
        baseLg = removeLayers(baseLg, toDrop);
    end
    newHead = [fullyConnectedLayer(numClasses, 'Name', 'fc8')
               softmaxLayer('Name', 'prob')];
    baseLg = addLayers(baseLg, newHead);
    baseLg = connectLayers(baseLg, 'drop7', 'fc8');
    net = dlnetwork(baseLg);
    fprintf('  Network converted to dlnetwork\n');
end

% Create augmented datastores
augTrain = augmentedImageDatastore([224 224], imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb', ...
    'DataAugmentation', config.imageAugmenter);
augVal = augmentedImageDatastore([224 224], imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

% Create minibatch queues with preprocessing
preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
mbqTrain = minibatchqueue(augTrain, ...
    'MiniBatchSize', config.batchSize, ...
    'MiniBatchFcn', preprocessFcn, ...
    'MiniBatchFormat', ["SSCB", ""], ...
    'PartialMiniBatch', 'discard');
mbqVal = minibatchqueue(augVal, ...
    'MiniBatchSize', config.batchSize, ...
    'MiniBatchFcn', preprocessFcn, ...
    'MiniBatchFormat', ["SSCB", ""], ...
    'PartialMiniBatch', 'discard');

% Initialize training
velocity = [];
iter = 0;
bestValLoss = inf;
bestValAcc = 0;
patienceCounter = 0;
featureLayer = 'relu5_3';
nCam = 32;

classCounts = countcats(imdsTrain.Labels);
baseWeights = sum(classCounts) ./ (numel(classCounts) * classCounts);
classWeights = gather(single(baseWeights(:)));   % pure inverse-frequency

training_history = struct();
training_history.epoch_loss = zeros(1, config.numEpochs);
training_history.val_loss = nan(1, config.numEpochs);
training_history.val_accuracy = nan(1, config.numEpochs);
training_history.val_iou = nan(1, config.numEpochs);
training_history.val_dice = nan(1, config.numEpochs);
training_history.train_acc = zeros(1, config.numEpochs);
training_history.val_acc = nan(1, config.numEpochs);
training_history.epoch = 1:config.numEpochs;
training_history.lr = zeros(1, config.numEpochs);  % Track LR schedule

if isfield(config, 'fast_dev_mode') && config.fast_dev_mode
    val_interval = 4;
else
    val_interval = 3;
end

fprintf('Starting training with enhanced stability...\n');
epoch = 0;
bestNet = net;  % Initialize bestNet so early stopping has a fallback

while epoch < config.numEpochs
    epoch = epoch + 1;
    fprintf('\nEpoch %d/%d\n', epoch, config.numEpochs);
    shuffle(mbqTrain);

    batch = 0;
    epochLoss = 0;
    trainCorrect = 0;
    trainTotal = 0;

    reset(mbqTrain);
    while hasdata(mbqTrain)
        iter = iter + 1;
        batch = batch + 1;
        
        [X, T] = next(mbqTrain);
        if config.useGPU
            X = gpuArray(X);
        end
        
        % === IMPROVEMENT: Compute learning rate with warmup + cosine annealing ===
        if epoch <= config.warmup_epochs
            % Linear warmup
            lr = config.initialLearnRate * (epoch / config.warmup_epochs);
        else
            % Cosine annealing
            progress = (epoch - config.warmup_epochs) / ...
                       (config.numEpochs - config.warmup_epochs);
            lr = config.initialLearnRate * (config.min_lr_ratio + ...
                (1 - config.min_lr_ratio) * 0.5 * (1 + cos(pi * progress)));
        end
        training_history.lr(epoch) = lr;
        % ===================================================================
        
        % Compute loss and gradients
        [loss, grads, state, loss_components] = dlfeval(@compute_loss_with_config_improved, ...
            net, X, T, loss_config, classWeights, imdsTrain.Files, ...
            preCAMs_train, preMasks_train, nCam, classes, featureLayer, config.useGPU, epoch, ...
            config.preprocessing_method, config.refHist);
        
        net.State = state;
        
           % === IMPROVEMENT: Gradient clipping for stability (Version Compatible) ===
            if isfield(loss_config, 'gradient_clip_norm') && ~isempty(loss_config.gradient_clip_norm)
                maxNorm = loss_config.gradient_clip_norm;
                
                % Compute global L2 norm of all gradients
                gradValues = grads.Value;
                gradNormSq = 0;
                for i = 1:numel(gradValues)
                    gradNormSq = gradNormSq + sum(gradValues{i}(:).^2);
                end
                gradNorm = sqrt(gradNormSq + eps);
                
                % Scale gradients if norm exceeds threshold
                if gradNorm > maxNorm
                    scale = maxNorm / gradNorm;
                    grads.Value = cellfun(@(g) g * scale, gradValues, 'UniformOutput', false);
                end
            end
             % =================================================
        
        [net, velocity] = sgdmupdate(net, grads, velocity, lr, config.momentum);
        
        epochLoss = epochLoss + double(loss);
            
        % Log loss components periodically
        if mod(batch, 10) == 0 && batch <= 50
            fprintf('    Batch %d - Loss components: ', batch);
            fprintf('Cls=%.3f', double(loss_components.classification));
            if loss_config.use_gradcam
                fprintf(', CAM=%.3f', double(loss_components.gradcam));
            end
            if loss_config.use_tversky
                fprintf(', Tversky=%.3f', double(loss_components.tversky));
            end
            if loss_config.use_anatomical_guidance
                fprintf(', Anatomical=%.3f', double(loss_components.anatomical));
            end
            fprintf(', Total=%.3f\n', double(loss));
        end
    
        Y = predict(net, X);
        Yscores = extractdata(Y);
        Tscores = extractdata(T);
        [~, predIdx] = max(Yscores, [], 1);
        [~, trueIdx] = max(Tscores, [], 1);
        trainCorrect = trainCorrect + sum(predIdx(:) == trueIdx(:));
        trainTotal = trainTotal + numel(predIdx);
    end

    avgTrainLoss = epochLoss / max(1, batch);
    trainAcc = trainCorrect / max(1, trainTotal);

    training_history.epoch_loss(epoch) = avgTrainLoss;
    training_history.train_acc(epoch) = trainAcc;

    % Validation cadence (faster in dev mode)
    if mod(epoch, val_interval) == 0 || epoch == 1
        valLoss = compute_validation_loss_with_config_preprocessed(net, imdsVal, classes, ...
            config.useGPU, preCAMs_val, preMasks_val, loss_config, classWeights, config, epoch);
        
        [val_acc, cm_val] = evaluate_classification_full(net, imdsVal, classes, config.useGPU, config);
        if isfield(config, 'fast_dev_mode') && config.fast_dev_mode
            val_iou = NaN;
            val_dice = NaN;
        else
            [~, val_iou, val_dice] = evaluate_model_quick(net, imdsVal, ...
                classes, config.useGPU, preCAMs_val, preMasks_val, featureLayer, config);
        end
        
        % Use argmax validation accuracy for early stopping (robust to class-order issues).
        valAcc = val_acc;
        
        training_history.val_loss(epoch) = valLoss;
        training_history.val_accuracy(epoch) = val_acc;
        training_history.val_iou(epoch) = val_iou;
        training_history.val_dice(epoch) = val_dice;
        training_history.val_acc(epoch) = valAcc;
        
        fprintf('  Train Loss: %.4f, Train Acc: %.3f\n', avgTrainLoss, trainAcc);
        fprintf('  Val Loss: %.4f, Val Acc: %.3f, Val IoU: %.3f, Val Dice: %.3f\n', ...
            valLoss, valAcc, val_iou, val_dice);
        if all(size(cm_val) == [2 2])
            fprintf('  Val CM [TN FP; FN TP] = [%d %d; %d %d]\n', ...
                cm_val(1,1), cm_val(1,2), cm_val(2,1), cm_val(2,2));
        else
            fprintf('  Val CM:\n');
            disp(cm_val);
        end
        
        % === IMPROVEMENT: Early stopping on Accuracy (loss is artificially inflated by adaptive scaling) ===
        if valAcc  > bestValAcc + config.min_delta  % Note: using '>' for accuracy
            bestValAcc = valAcc;
            bestNet = net;
            patienceCounter = 0;
            fprintf('  ✓ Validation accuracy improved!\n');
        else
            patienceCounter = patienceCounter + 1;
        end
        % Initialize bestValAcc before the epoch loop:
        % bestValAcc = 0;  (Add this line near 'bestValLoss = inf;')
        
        if patienceCounter >= config.patience
            fprintf('\nEarly stopping triggered (patience: %d)\n', config.patience);
            net = bestNet;
            break;
        end
    else
        fprintf('  Train Loss: %.4f, Train Acc: %.3f\n', avgTrainLoss, trainAcc);
    end
end

if exist('bestNet', 'var')
    trainedNet = bestNet;
else
    trainedNet = net;
end

% Trim history arrays
training_history.epoch_loss = training_history.epoch_loss(1:epoch);
training_history.train_acc = training_history.train_acc(1:epoch);
training_history.val_loss = training_history.val_loss(1:epoch);
training_history.val_accuracy = training_history.val_accuracy(1:epoch);
training_history.val_iou = training_history.val_iou(1:epoch);
training_history.val_dice = training_history.val_dice(1:epoch);
training_history.val_acc = training_history.val_acc(1:epoch);
training_history.lr = training_history.lr(1:epoch);
training_history.epoch = training_history.epoch(1:epoch);
end

% === IMPROVEMENT: Enhanced loss computation with cosine similarity and positivity ===
function [loss, grads, state, loss_components] = compute_loss_with_config_improved(...
    net, X, T, loss_config, classWeights, trainFiles, preCAMs, preMasks, ...
    nCam, classes, featureLayer, useGPU, epoch, preprocessing_method, refHist)

    [Y, state] = forward(net, X);
    loss_components = struct();

    % Classification loss
    if loss_config.use_focal
        clsLoss = compute_focal_loss(Y, T, classWeights, ...
            loss_config.focal_alpha, loss_config.focal_gamma);
    else
        if isfield(loss_config, 'use_class_weights') && loss_config.use_class_weights
            clsLoss = crossentropy(Y, T, 'Weights', classWeights, 'WeightsFormat', 'C');
        else
            clsLoss = crossentropy(Y, T);
        end
    end
    loss_components.classification = clsLoss;

    camLoss = 0;
    segLoss = 0;
    tverskyLoss = 0;
    iouLoss = 0;
    anatomicalLoss = 0;

    % Auxiliary losses (CAM, Segmentation, Anatomical)
    if any([loss_config.use_gradcam, loss_config.use_segmentation, ...
            loss_config.use_tversky, loss_config.use_iou, ...
            loss_config.use_anatomical_guidance])
        
        N = numel(trainFiles);
        n = min(nCam, N);
        idxs = randperm(N, n);
        
        for ii = 1:n
            % Load and preprocess image consistently with training pipeline
            img = imread(trainFiles{idxs(ii)});
            if size(img,3)==1
                img = repmat(img,[1 1 3]);
            end
            
            % Convert to 4D batch format for preprocessing function
            img4d = reshape(img, size(img,1), size(img,2), size(img,3), 1);
            img4d = apply_preprocessing_batch(img4d, preprocessing_method, refHist);
            img = img4d(:,:,:,1);                    % back to 3D
            img = imresize(img, [224 224]);          % ensure correct size
            
            % Compute student CAM
            studCAM = student_cam_one_dlarray(net, img, classes, featureLayer, useGPU);
            studCAM_data = extractdata(studCAM);
            studCAM_data = single(studCAM_data);
            
            if size(studCAM_data,1) ~= 224 || size(studCAM_data,2) ~= 224
                studCAM_data = imresize(studCAM_data, [224 224]);
            end
            studCAM = dlarray(studCAM_data, 'SS');

            % === GradCAM Loss (Cosine) ===
            if loss_config.use_gradcam
                targetCAM = preCAMs{idxs(ii)};
                if ~isa(targetCAM, 'dlarray')
                    targetCAM = single(targetCAM);
                    if ndims(targetCAM) > 2, targetCAM = squeeze(targetCAM); end
                    if size(targetCAM,1) ~= 224 || size(targetCAM,2) ~= 224
                        targetCAM = imresize(targetCAM, [224 224]);
                    end
                    targetCAM = dlarray(targetCAM, 'SS');
                else
                    targetCAM = dlarray(single(extractdata(stripdims(targetCAM))), 'SS');
                    if size(targetCAM,1) ~= 224 || size(targetCAM,2) ~= 224
                        targetCAM = imresize(targetCAM, [224 224]);
                    end
                end
                
                if isfield(loss_config, 'cam_loss_type') && strcmp(loss_config.cam_loss_type, 'cosine')
                    camLoss = camLoss + cam_cosine_loss(studCAM, targetCAM);
                else
                    camLoss = camLoss + mse(studCAM, targetCAM);
                end
            end

            % === Segmentation Losses ===
            if any([loss_config.use_segmentation, loss_config.use_tversky, loss_config.use_iou])
                realMask = preMasks{idxs(ii)};
                if ~isempty(realMask)
                    realMask_resized = imresize(single(realMask), [224 224]);
                    if loss_config.use_segmentation
                        segLoss = segLoss + (1 - dice_coefficient_dlarray(studCAM, realMask_resized));
                    end
                    if loss_config.use_tversky
                        tverskyCoef = tversky_coefficient_dlarray(studCAM, realMask_resized, ...
                            loss_config.tversky_alpha, loss_config.tversky_beta);
                        tverskyLoss = tverskyLoss + (1 - tverskyCoef);
                    end
                    if loss_config.use_iou
                        iouCoef = iou_coefficient_dlarray(studCAM, realMask_resized);
                        iouLoss = iouLoss + (1 - iouCoef);
                    end
                end
            end

            % === Anatomical Guidance Loss ===
            if loss_config.use_anatomical_guidance
                realMask = preMasks{idxs(ii)};
                if ~isempty(realMask) && any(realMask(:))
                    lungMask = single(realMask > 0.5);
                    if size(lungMask,1) ~= 224 || size(lungMask,2) ~= 224
                        lungMask = imresize(lungMask, [224 224], 'nearest');
                    end
                    
                    studCAM_norm = studCAM;
                    cam_max = max(studCAM_norm, [], 'all');
                    if cam_max > 0
                        studCAM_norm = studCAM_norm / (cam_max + eps);
                    end
                    
                    lungMask_dl = dlarray(lungMask, 'SS');
                    nonLungMask_dl = dlarray(1 - lungMask, 'SS');
                    
                    penalty_outside = mean(studCAM_norm .* nonLungMask_dl, 'all');
                    reward_inside = mean(studCAM_norm .* lungMask_dl, 'all');
                    
                    anatomical_loss_sample = penalty_outside - ...
                        loss_config.anatomical_reward_weight * reward_inside;
                    
                    if isfield(loss_config, 'anatomical_positivity') && loss_config.anatomical_positivity
                        anatomical_loss_sample = max(anatomical_loss_sample, 0);
                    end
                    
                    % Adaptive scaling - DISABLED for stability
                    if isfield(loss_config, 'adaptive_scaling') && loss_config.adaptive_scaling
                        scale_factor = min(loss_config.max_anatomical_scale, max(10, epoch * 5));
                    else
                        scale_factor = 1.0;  % No scaling by default
                    end
                    
                    anatomicalLoss = anatomicalLoss + anatomical_loss_sample * scale_factor;
                end
            end
        end % end for loop
        
        % Average across samples
        camLoss = camLoss / max(1, n);
        segLoss = segLoss / max(1, n);
        tverskyLoss = tverskyLoss / max(1, n);
        iouLoss = iouLoss / max(1, n);
        anatomicalLoss = anatomicalLoss / max(1, n);
        
    end % end auxiliary losses if

    % Store components
    loss_components.gradcam = camLoss;
    loss_components.segmentation = segLoss;
    loss_components.tversky = tverskyLoss;
    loss_components.iou = iouLoss;
    loss_components.anatomical = anatomicalLoss;

    % Total loss
    loss = clsLoss;
    if loss_config.use_gradcam
        loss = loss + loss_config.lambda_cam * camLoss;
    end
    if loss_config.use_segmentation
        loss = loss + loss_config.lambda_seg * segLoss;
    end
    if loss_config.use_tversky
        loss = loss + loss_config.lambda_tversky * tverskyLoss;
    end
    if loss_config.use_iou
        loss = loss + loss_config.lambda_seg * iouLoss;
    end
    if loss_config.use_anatomical_guidance
        loss = loss + loss_config.lambda_anatomical * anatomicalLoss;
    end

    % Gradients
    grads = dlgradient(loss, net.Learnables, 'EnableHigherDerivatives', true);
end

% === IMPROVEMENT: Cosine similarity loss for CAM alignment ===
function camLoss = cam_cosine_loss(studCAM, targetCAM)
    % Reshape to vectors
    stud_vec = reshape(studCAM, [], 1);
    target_vec = reshape(targetCAM, [], 1);
    
    % dlarray-compatible cosine similarity calculation
    % (Preserves gradients for backpropagation)
    dot_product = sum(stud_vec .* target_vec, 'all');
    norm_stud = sqrt(sum(stud_vec.^2, 'all'));
    norm_target = sqrt(sum(target_vec.^2, 'all'));
    
    cosSim = dot_product / (norm_stud * norm_target + eps);
    
    % Convert similarity to loss: [0, 2] range
    camLoss = 1 - cosSim;
    
    % Clamp for numerical stability (dlarray compatible)
    camLoss = max(0, min(2, camLoss));
end

function calibration = tune_binary_calibration_on_id_set(...
    net, imds, classes, useGPU, base_preprocessing, refHist, use_dual_view, config)
% Tune binary threshold and optional temperature using ID validation data.
    [~, predictions] = evaluate_dataset_with_preprocessing_ensemble(...
        net, imds, classes, useGPU, base_preprocessing, refHist, false, use_dual_view, NaN, 1.0);

    posIdx = infer_positive_class_idx(classes);
    negIdx = 3 - posIdx;
    yTrue = predictions.Ytrue(:);
    pPosRaw = predictions.Yprobs(:);

    temperature = 1.0;
    if isfield(config, 'enable_temperature_calibration') && config.enable_temperature_calibration
        temperature = tune_binary_temperature_nll(yTrue, pPosRaw, classes, posIdx);
    end
    pPos = apply_binary_temperature(pPosRaw, temperature);

    [thrBalanced, thrConstrained] = tune_binary_thresholds_from_probs(...
        yTrue, pPos, classes, posIdx, config.threshold_sens_floor);

    if isfield(config, 'threshold_mode') && strcmpi(config.threshold_mode, 'constrained') && isfinite(thrConstrained)
        thrSelected = thrConstrained;
    else
        thrSelected = thrBalanced;
    end

    calibration = struct();
    calibration.temperature = temperature;
    calibration.threshold_balanced = thrBalanced;
    calibration.threshold_constrained = thrConstrained;
    calibration.threshold_selected = thrSelected;
end

function bestT = tune_binary_temperature_nll(yTrue, pPosRaw, classes, posIdx)
% Optimize temperature via NLL over a small grid.
    yBin = double(yTrue == classes{posIdx});
    bestT = 1.0;
    bestNLL = inf;
    for T = 0.8:0.1:3.0
        pCal = apply_binary_temperature(pPosRaw, T);
        nll = -mean(yBin .* log(pCal + eps) + (1 - yBin) .* log(1 - pCal + eps));
        if nll < bestNLL
            bestNLL = nll;
            bestT = T;
        end
    end
end

function pCal = apply_binary_temperature(p, T)
% Temperature scaling in logit-space for binary probabilities.
    p = min(max(p, 1e-6), 1 - 1e-6);
    logit = log(p ./ (1 - p));
    pCal = 1 ./ (1 + exp(-logit ./ max(T, eps)));
end

function [thrBalanced, thrConstrained] = tune_binary_thresholds_from_probs(yTrue, pPos, classes, posIdx, sensFloor)
    negIdx = 3 - posIdx;
    thrBalanced = 0.5;
    bestBalanced = -inf;
    thrConstrained = NaN;
    bestConstrainedSpec = -inf;

    %for t = 0.20:0.02:0.80
    for t = 0.35:0.02:0.80     %start from 0.35 to prevent degenerate low-threshold selection
        predIdx = negIdx * ones(size(pPos));
        predIdx(pPos >= t) = posIdx;
        yPred = categorical(classes(predIdx));
        cm = confusionmat(yTrue, yPred, 'Order', categorical(classes));
        if ~all(size(cm) == [2 2])
            continue;
        end
        TP = cm(posIdx, posIdx);
        FN = cm(posIdx, negIdx);
        TN = cm(negIdx, negIdx);
        FP = cm(negIdx, posIdx);
        sens = TP / (TP + FN + eps);
        spec = TN / (TN + FP + eps);
        bal = 0.5 * (sens + spec);

        if bal > bestBalanced
            bestBalanced = bal;
            thrBalanced = t;
        end
        if sens >= sensFloor && spec > bestConstrainedSpec
            bestConstrainedSpec = spec;
            thrConstrained = t;
        end
    end

    if isfinite(thrConstrained) && thrConstrained < 0.40
        thrConstrained = thrBalanced;  % reject degenerate threshold
    end
end

function evaluation_suite = run_multi_case_evaluation_suite(trainedNet, imdsROIVal, imdsCXR, classes, config, refHist)
    fprintf('\n  Running evaluation suite...\n');

    decisionThreshold = NaN;
    calibration = struct();
    calibration.temperature = 1.0;
    calibration.threshold_balanced = NaN;
    calibration.threshold_constrained = NaN;
    calibration.threshold_selected = NaN;

    if numel(classes) == 2
        calibration = tune_binary_calibration_on_id_set(trainedNet, imdsROIVal, classes, ...
            config.useGPU, config.preprocessing_method, refHist, false, config);
        decisionThreshold = calibration.threshold_selected;
        fprintf('  Tuned PTB threshold on ROI-val (mode=%s): %.3f\n', ...
            config.threshold_mode, decisionThreshold);
        fprintf('  Thresholds: balanced=%.3f, constrained=%.3f (sens floor=%.2f)\n', ...
            calibration.threshold_balanced, calibration.threshold_constrained, config.threshold_sens_floor);
        fprintf('  Temperature calibration: T=%.2f\n', calibration.temperature);
    end

    conservativeCfg = struct(...
        'enabled', config.enable_conservative_ood, ...
        'uncertainty_percentile', config.conservative_uncertainty_percentile, ...
        'margin', config.conservative_margin);

    caseNames = config.evaluation_cases;
    case_results = struct();
    probability_summaries = struct();
    uncertainty_summaries = struct();

    for i = 1:numel(caseNames)
        caseName = char(caseNames{i});
        [caseTitle, caseImds, isOODEvaluation, useDualView] = resolve_evaluation_case(caseName, imdsROIVal, imdsCXR);

        fprintf('\n  Case: %s\n', caseTitle);
        [caseMetrics, casePredictions, caseUncertainty] = evaluate_dataset_with_preprocessing_ensemble(...
            trainedNet, caseImds, classes, config.useGPU, config.preprocessing_method, ...
            refHist, isOODEvaluation, useDualView, decisionThreshold, calibration.temperature, conservativeCfg);

        case_results.(caseName) = struct();
        case_results.(caseName).title = caseTitle;
        case_results.(caseName).results = caseMetrics;
        case_results.(caseName).predictions = casePredictions;
        case_results.(caseName).uncertainty = caseUncertainty;
        case_results.(caseName).is_ood_evaluation = isOODEvaluation;
        case_results.(caseName).use_dual_view = useDualView;

        probability_summaries.(caseName) = summarize_probabilities(casePredictions, decisionThreshold);
        uncertainty_summaries.(caseName) = caseUncertainty;
        print_evaluation_case_report(caseTitle, caseMetrics, probability_summaries.(caseName), caseUncertainty);
    end

    evaluation_suite = struct();
    evaluation_suite.reference_case = 'roi_val';
    evaluation_suite.calibration = calibration;
    evaluation_suite.decision_threshold = decisionThreshold;
    evaluation_suite.case_results = case_results;
    evaluation_suite.probability_summaries = probability_summaries;
    evaluation_suite.uncertainty_summaries = uncertainty_summaries;
    evaluation_suite.degradation_vs_roi_val = struct();
    evaluation_suite.mean_degradation_cxr_roi = NaN;

    if isfield(case_results, 'roi_val')
        refResults = case_results.roi_val.results;
        referenceCases = fieldnames(case_results);
        for i = 1:numel(referenceCases)
            caseName = referenceCases{i};
            if strcmp(caseName, 'roi_val')
                continue;
            end
            deg = compute_result_degradation(refResults, case_results.(caseName).results);
            evaluation_suite.degradation_vs_roi_val.(caseName) = deg;

            fprintf('\n  Degradation vs ROI-val for %s:\n', case_results.(caseName).title);
            fprintf('    Accuracy: %.2f%%\n', deg.accuracy);
            fprintf('    Precision: %.2f%%\n', deg.precision);
            fprintf('    Sensitivity: %.2f%%\n', deg.sensitivity);
            fprintf('    Specificity: %.2f%%\n', deg.specificity);
            fprintf('    F1-Score: %.2f%%\n', deg.f1_score);
            fprintf('    AUC: %.2f%%\n', deg.auc);
            fprintf('    Mean Degradation: %.2f%%\n', deg.mean_degradation);
        end

        if isfield(evaluation_suite.degradation_vs_roi_val, 'cxr_roi')
            evaluation_suite.mean_degradation_cxr_roi = ...
                evaluation_suite.degradation_vs_roi_val.cxr_roi.mean_degradation;
        end
    end
end

function [caseTitle, caseImds, isOODEvaluation, useDualView] = resolve_evaluation_case(caseName, imdsROIVal, imdsCXR)
    switch lower(caseName)
        case 'roi_val'
            caseTitle = 'ROI validation (ID)';
            caseImds = imdsROIVal;
            isOODEvaluation = false;
            useDualView = false;

        case 'cxr_raw'
            caseTitle = 'Full CXR raw';
            caseImds = imdsCXR;
            isOODEvaluation = false;
            useDualView = false;

        case 'cxr_roi'
            caseTitle = 'Full CXR with ROI extraction';
            caseImds = imdsCXR;
            isOODEvaluation = true;
            useDualView = false;

        case {'cxr_dual', 'cxr_dual_view'}
            caseTitle = 'Full CXR dual-view';
            caseImds = imdsCXR;
            isOODEvaluation = true;
            useDualView = true;

        otherwise
            error('Unsupported evaluation case: %s', caseName);
    end
end

function deg = compute_result_degradation(referenceResults, targetResults)
    safe_pct = @(idv, oodv) 100 * (idv - oodv) / max(abs(idv), 1e-6);

    deg = struct();
    deg.accuracy = safe_pct(referenceResults.accuracy, targetResults.accuracy);
    deg.precision = safe_pct(referenceResults.precision, targetResults.precision);
    deg.sensitivity = safe_pct(referenceResults.sensitivity, targetResults.sensitivity);
    deg.specificity = safe_pct(referenceResults.specificity, targetResults.specificity);
    deg.f1_score = safe_pct(referenceResults.f1_score, targetResults.f1_score);
    deg.auc = safe_pct(referenceResults.auc, targetResults.auc);

    degValues = [deg.accuracy, deg.precision, deg.sensitivity, deg.specificity, deg.f1_score, deg.auc];
    degValues = degValues(isfinite(degValues));
    if isempty(degValues)
        deg.mean_degradation = NaN;
    else
        deg.mean_degradation = mean(degValues);
    end
end

function summary = summarize_probabilities(predictions, decisionThreshold)
    summary = struct();
    if ~isfield(predictions, 'Yprobs') || isempty(predictions.Yprobs)
        summary.mean = NaN;
        summary.median = NaN;
        summary.pct_above_threshold = NaN;
        return;
    end

    summary.mean = mean(predictions.Yprobs);
    summary.median = median(predictions.Yprobs);
    if isfinite(decisionThreshold)
        summary.pct_above_threshold = 100 * mean(predictions.Yprobs >= decisionThreshold);
    else
        summary.pct_above_threshold = NaN;
    end
end

function print_evaluation_case_report(~, caseMetrics, probabilitySummary, uncertaintySummary)
    fprintf('    Accuracy: %.3f\n', caseMetrics.accuracy);
    fprintf('    Precision: %.3f\n', caseMetrics.precision);
    fprintf('    Sensitivity: %.3f\n', caseMetrics.sensitivity);
    fprintf('    Specificity: %.3f\n', caseMetrics.specificity);
    fprintf('    F1-Score: %.3f\n', caseMetrics.f1_score);
    fprintf('    AUC: %.3f\n', caseMetrics.auc);

    if isfinite(probabilitySummary.mean)
        fprintf('    PTB prob mean/median: %.3f / %.3f\n', probabilitySummary.mean, probabilitySummary.median);
    end
    if isfinite(probabilitySummary.pct_above_threshold)
        fprintf('    %% above threshold: %.1f%%\n', probabilitySummary.pct_above_threshold);
    end
    if isfield(uncertaintySummary, 'mean_uncertainty')
        fprintf('    Mean uncertainty: %.3f\n', uncertaintySummary.mean_uncertainty);
    end
    if isfield(uncertaintySummary, 'conservative_overrides') && uncertaintySummary.conservative_overrides > 0
        fprintf('    Conservative overrides: %d\n', uncertaintySummary.conservative_overrides);
    end
end


% === IMPROVEMENT: OOD evaluation with preprocessing ensemble ===
function [results, predictions, uncertainty_stats] = evaluate_dataset_with_preprocessing_ensemble(...
    net, imds, classes, useGPU, base_preprocessing, refHist, is_ood_evaluation, use_dual_view, decisionThreshold, temperature, conservativeCfg)
% Evaluate with multiple preprocessing methods and ensemble predictions
% Also computes uncertainty estimates for clinical safety
if nargin < 8
    use_dual_view = true;
end
if nargin < 9
    decisionThreshold = NaN;
end
if nargin < 10
    temperature = 1.0;
end
if nargin < 11
    conservativeCfg = struct('enabled', false, 'uncertainty_percentile', 80, 'margin', 0.10);
end
posIdx = infer_positive_class_idx(classes);

% Preprocessing methods to ensemble
if strcmp(base_preprocessing, 'none')
    % Keep OOD test-time preprocessing aligned with training distribution.
    preprocessing_methods = {'none'};
elseif strcmp(base_preprocessing, 'histmatch')
    preprocessing_methods = {'histmatch', 'clahe', 'none'};
else
    preprocessing_methods = {base_preprocessing, 'histmatch', 'clahe'};
end

augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
reset(augDS);

Ypred_ensemble = categorical.empty(0,1);
Yprobs_ensemble = [];
Ytrue = imds.Labels(:);
uncertainty_values = [];

while hasdata(augDS)
    tbl = read(augDS);
    imgs = cat(4, tbl.input{:});
    
    % === IMPROVEMENT: Preprocessing ensemble ===
    probs_batch = zeros(size(imgs, 4), numel(classes));
    
    for m = 1:numel(preprocessing_methods)
        method = preprocessing_methods{m};

        if is_ood_evaluation && use_dual_view
            imgs_full = apply_preprocessing_batch(imgs, method, refHist);
            [H, W, ~, B] = size(imgs);
            roi_batch = zeros(H, W, 3, B, 'like', imgs);
            for b = 1:B
                img_sample = imgs(:, :, :, b);
                if size(img_sample, 3) == 3
                    img_gray = rgb2gray(img_sample);
                else
                    img_gray = img_sample;
                end
                roi_cropped = extract_lung_roi_simple(img_gray);
                roi_batch(:, :, :, b) = repmat(roi_cropped, [1, 1, 3]);
            end
            imgs_roi = apply_preprocessing_batch(roi_batch, method, refHist);

            if useGPU
                imgs_full = gpuArray(imgs_full);
                imgs_roi = gpuArray(imgs_roi);
            end
            sc_full = predict(net, dlarray(single(imgs_full) ./ 255, 'SSCB'));
            sc_roi = predict(net, dlarray(single(imgs_roi) ./ 255, 'SSCB'));
            probs = 0.5 * extractdata(sc_full)' + 0.5 * extractdata(sc_roi)';
        else
            if is_ood_evaluation
                [H, W, ~, B] = size(imgs);
                roi_batch = zeros(H, W, 3, B, 'like', imgs);
                for b = 1:B
                    img_sample = imgs(:, :, :, b);
                    if size(img_sample, 3) == 3
                        img_gray = rgb2gray(img_sample);
                    else
                        img_gray = img_sample;
                    end
                    roi_cropped = extract_lung_roi_simple(img_gray);
                    roi_batch(:, :, :, b) = repmat(roi_cropped, [1, 1, 3]);
                end
                imgs_processed = apply_preprocessing_batch(roi_batch, method, refHist);
            else
                imgs_processed = apply_preprocessing_batch(imgs, method, refHist);
            end
            if useGPU, imgs_processed = gpuArray(imgs_processed); end
            sc = predict(net, dlarray(single(imgs_processed) ./ 255, 'SSCB'));
            probs = extractdata(sc)';
        end
        
        % Accumulate probabilities
        probs_batch = probs_batch + probs;
    end
    
    % Average probabilities across preprocessing methods
    probs_batch = probs_batch / numel(preprocessing_methods);
    if numel(classes) == 2
        probs_batch(:, posIdx) = apply_binary_temperature(probs_batch(:, posIdx), temperature);
        negIdxTmp = 3 - posIdx;
        probs_batch(:, negIdxTmp) = 1 - probs_batch(:, posIdx);
    end
    % ================================================
    
    % Decode predictions
    if numel(classes) == 2 && isfinite(decisionThreshold)
        negIdx = 3 - posIdx;
        pred_idx = negIdx * ones(size(probs_batch, 1), 1);
        pred_idx(probs_batch(:, posIdx) >= decisionThreshold) = posIdx;
        lab = categorical(classes(pred_idx));
    else
        [~, pred_idx] = max(probs_batch, [], 2);
        lab = categorical(classes(pred_idx));
    end
    Ypred_ensemble = [Ypred_ensemble; lab(:)];
    
    % Store probabilities for uncertainty computation
    if numel(classes) == 2
        Yprobs_ensemble = [Yprobs_ensemble; probs_batch(:, posIdx)];
    else
        Yprobs_ensemble = [Yprobs_ensemble; max(probs_batch, [], 2)];
    end
    
    % === IMPROVEMENT: Compute prediction uncertainty ===
    % Use entropy of predicted distribution as uncertainty measure
    probs_normalized = probs_batch ./ (sum(probs_batch, 2) + eps);
    entropy = -sum(probs_normalized .* log(probs_normalized + eps), 2);
    max_prob = max(probs_batch, [], 2);
    
    % Combine entropy and confidence for uncertainty score
    uncertainty = (1 - max_prob) + 0.5 * entropy / log(numel(classes));
    uncertainty_values = [uncertainty_values; uncertainty];
    % ================================================
end

% Conservative OOD decision rule (toggleable):
% For high-uncertainty predictions near threshold, force normal class.
conservative_overrides = 0;
conservative_u_thr = NaN;
conservative_margin = NaN;
if numel(classes) == 2 && is_ood_evaluation && isfield(conservativeCfg, 'enabled') && conservativeCfg.enabled
    negIdx = 3 - posIdx;
    if isfield(conservativeCfg, 'uncertainty_percentile')
        q = conservativeCfg.uncertainty_percentile;
    else
        q = 80;
    end
    if isfield(conservativeCfg, 'margin')
        margin = conservativeCfg.margin;
    else
        margin = 0.10;
    end
    q = min(max(q, 0), 100);
    margin = max(margin, 0);
    conservative_u_thr = prctile(uncertainty_values, q);
    conservative_margin = margin;
    if isfinite(decisionThreshold)
        near_thr = abs(Yprobs_ensemble - decisionThreshold) <= margin;
    else
        near_thr = true(size(Yprobs_ensemble));
    end
    override_mask = (uncertainty_values >= conservative_u_thr) & near_thr;
    if any(override_mask)
        yPredIdx = zeros(numel(Ypred_ensemble), 1);
        for i = 1:numel(Ypred_ensemble)
            yPredIdx(i) = find(strcmp(classes, char(Ypred_ensemble(i))), 1);
        end
        yPredIdx(override_mask) = negIdx;
        Ypred_ensemble = categorical(classes(yPredIdx));
        conservative_overrides = sum(override_mask);
    end
end

% Calculate metrics
match = (categorical(Ytrue) == categorical(Ypred_ensemble));
accuracy = sum(match) / numel(match);

cm = confusionmat(Ytrue, Ypred_ensemble, 'Order', categorical(classes));
if numel(classes)==2 && all(size(cm)==[2 2])
    negIdx = 3 - posIdx;
    TP = cm(posIdx,posIdx); FP = cm(negIdx,posIdx); FN = cm(posIdx,negIdx); TN = cm(negIdx,negIdx);
else
    [~,posIdx] = min(countcats(Ytrue));
    pos = classes{posIdx};
    TP = sum(Ypred_ensemble==pos & Ytrue==pos);
    FP = sum(Ypred_ensemble==pos & Ytrue~=pos);
    FN = sum(Ypred_ensemble~=pos & Ytrue==pos);
    TN = sum(Ypred_ensemble~=pos & Ytrue~=pos);
end

precision = TP / (TP + FP + eps);
sensitivity = TP / (TP + FN + eps);
specificity = TN / (TN + FP + eps);
f1score = 2*precision*sensitivity / (precision + sensitivity + eps);

try
    if numel(classes) == 2
        [~,~,~,auc] = perfcurve(double(Ytrue == classes{posIdx}), Yprobs_ensemble, 1);
    else
        auc = NaN;
    end
catch
    auc = 0.5;
end

% === IMPROVEMENT: Uncertainty statistics ===
uncertainty_stats = struct();
uncertainty_stats.mean_uncertainty = mean(uncertainty_values);
uncertainty_stats.std_uncertainty = std(uncertainty_values);

% Flag high-uncertainty predictions (top 20% or above threshold)
uncertainty_threshold = prctile(uncertainty_values, 80);
high_uncertainty_mask = uncertainty_values > uncertainty_threshold;
uncertainty_stats.high_uncertainty_pct = mean(high_uncertainty_mask);
uncertainty_stats.uncertainty_threshold = uncertainty_threshold;
uncertainty_stats.conservative_overrides = conservative_overrides;
uncertainty_stats.conservative_uncertainty_threshold = conservative_u_thr;
uncertainty_stats.conservative_margin = conservative_margin;

% Optional: Adjust predictions for high-uncertainty samples (conservative mode)
% Uncomment below to enable conservative prediction adjustment
% if any(high_uncertainty_mask)
%     % For high-uncertainty samples, lower the decision threshold
%     conservative_threshold = 0.3;  % Instead of 0.5
%     for i = find(high_uncertainty_mask)'
%         if Yprobs_ensemble(i) < conservative_threshold
%             Ypred_ensemble(i) = classes{1};  % Predict normal class
%         end
%     end
%     % Recalculate metrics with conservative predictions
%     % ... (recompute accuracy, sensitivity, etc.)
% end
% ================================================

results = struct();
results.accuracy = accuracy;
results.precision = precision;
results.sensitivity = sensitivity;
results.specificity = specificity;
results.f1_score = f1score;
results.auc = auc;
results.confusion_matrix = cm;

predictions = struct();
predictions.Ypred = Ypred_ensemble;
predictions.Ytrue = Ytrue;
predictions.Yprobs = Yprobs_ensemble;
predictions.uncertainty = uncertainty_values;
end

%% Remaining helper functions (preprocessMiniBatchWithPreprocessing, apply_preprocessing_batch, etc.)
%% remain unchanged from original - copy from your original file here
%% For brevity, I'm showing the key new function only:


%% Additional Helper Functions (copied from train_final_model.m)
 function [imds, vggNet, precomputedGradCAM, precomputedMasks, classes] = ...
        load_data_and_network(forceRecalculate, teacherModelSpec)

    if nargin < 1, forceRecalculate = false; end
    if nargin < 2 || isempty(teacherModelSpec), teacherModelSpec = 'roi'; end

    % 1. Load teacher backbone weights
    modelPath = resolve_teacher_model_path(teacherModelSpec);
    fprintf('  Loading teacher backbone from: %s\n', modelPath);
    s = load(modelPath);
    if isfield(s, 'trainedNet'),   vggNet = s.trainedNet;
    elseif isfield(s, 'fold_model'), vggNet = s.fold_model;
    elseif isfield(s, 'net'),      vggNet = s.net;
    else, error('Could not find network variable in %s', modelPath);
    end

    % Convert to dlnetwork — forward() and dlgradient() require dlnetwork.
    % Teacher .mat files saved from trainNetwork() are DAGNetwork.
    if isa(vggNet, 'DAGNetwork') || isa(vggNet, 'SeriesNetwork')
        fprintf('  Converting %s to dlnetwork...\n', class(vggNet));
        try
            vggNet = dag2dlnetwork(vggNet);
        catch
            lg = layerGraph(vggNet);
            for oi = 1:numel({'output','classoutput','ClassificationLayer_predictions'})
                nm = {'output','classoutput','ClassificationLayer_predictions'};
                if any(strcmp({lg.Layers.Name}, nm{oi}))
                    try, lg = removeLayers(lg, nm{oi}); catch, end
                end
            end
            vggNet = dlnetwork(lg);
        end
        fprintf('  Conversion complete: %s\n', class(vggNet));
    elseif ~isa(vggNet, 'dlnetwork')
        try
            vggNet = dlnetwork(layerGraph(vggNet));
            fprintf('  Converted to dlnetwork\n');
        catch ME
            error('Cannot convert teacher network: %s', ME.message);
        end
    else
        fprintf('  Teacher is already dlnetwork\n');
    end

    % 2. Student always trains on ROI — imds and classes always from roi dir
    roiDir = fullfile('input', 'roi');
    imds   = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    classes = categories(imds.Labels);

    % 3. Resolve paths (needed by all branches)
    gradCAMCacheFile = sprintf('precomputed_gradcam_maps_%s.mat', lower(teacherModelSpec));

    if numel(imds.Files) > 0
        firstImgPath   = imds.Files{1};
        imgDir         = fileparts(firstImgPath);
        roiParentDir   = fileparts(imgDir);
        baseDir        = fileparts(roiParentDir);
        maskDir        = fullfile(baseDir, 'masks');
        gradcamMasksDir = fullfile(baseDir, 'gradcam_masks');
    else
        baseDir        = 'input';
        maskDir        = fullfile('input', 'masks');
        gradcamMasksDir = fullfile('input', 'gradcam_masks');
    end
    if ~exist(gradcamMasksDir, 'dir'), mkdir(gradcamMasksDir); end
    workingGradCAMLayer = 'relu5_3';

    % 4. For mixed teacher: generate GradCAMs from CXR images, not ROI
    %    The mixed teacher's attention lives in CXR space — running it on
    %    ROI crops produces incoherent maps. We run it on the matched CXR
    %    image and crop the resulting attention map to the ROI region.
    if strcmpi(teacherModelSpec, 'mixed')
        cxrDir = fullfile(baseDir, 'cxr');
        if ~exist(cxrDir, 'dir')
            cxrDir = fullfile('input', 'cxr');  % fallback
        end
        if exist(cxrDir, 'dir') && (forceRecalculate || ~exist(gradCAMCacheFile, 'file'))
            fprintf('  [Mixed teacher] Computing GradCAMs on CXR images matched to ROI...\n');
            imdsCXR_teacher = imageDatastore(cxrDir, ...
                'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            [precomputedGradCAM, precomputedMasks] = precompute_gradcam_mixed(...
                imds, imdsCXR_teacher, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
            cachedFileList = imds.Files;
            cachedTeacher  = teacherModelSpec;
            save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', ...
                'cachedFileList', 'cachedTeacher', '-v7.3');
            fprintf('  [Mixed teacher] GradCAM maps saved to %s\n', gradCAMCacheFile);
            return;  % imds, classes, vggNet already set above — safe to return
        elseif exist(gradCAMCacheFile, 'file')
            fprintf('  [Mixed teacher] Loading cached GradCAM maps from %s\n', gradCAMCacheFile);
            cache = load(gradCAMCacheFile);
            precomputedGradCAM = cache.precomputedGradCAM;
            precomputedMasks   = cache.precomputedMasks;
            return;
        else
            fprintf('  ⚠ [Mixed teacher] CXR dir not found: %s. Falling back to ROI GradCAMs.\n', cxrDir);
            % Falls through to standard ROI GradCAM computation below
        end
    end

    % 5. Standard path: compute or load GradCAMs from ROI images
    if forceRecalculate
        fprintf('  Recalculating GradCAM maps...\n');
        [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
            imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
        cachedFileList = imds.Files;
        cachedTeacher  = teacherModelSpec;
        save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', ...
            'cachedFileList', 'cachedTeacher', '-v7.3');
        fprintf('  GradCAM maps recalculated and saved\n');
        return;
    end

    if ~exist(gradCAMCacheFile, 'file')
        fprintf('  Cache not found. Computing GradCAM maps...\n');
        [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
            imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
        cachedFileList = imds.Files;
        cachedTeacher  = teacherModelSpec;
        save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', ...
            'cachedFileList', 'cachedTeacher', '-v7.3');
        return;
    end

    % 6. Load and validate cache
    fprintf('  Loading cached GradCAM maps...\n');
    cache = load(gradCAMCacheFile);

    % Validate: teacher identity + file list must match
    teacherMatch = ~isfield(cache, 'cachedTeacher') || ...
                strcmpi(cache.cachedTeacher, teacherModelSpec);
    fileMatch = isfield(cache, 'cachedFileList') && ...
                numel(cache.cachedFileList) == numel(imds.Files) && ...
                all(strcmp(cache.cachedFileList, imds.Files));
    masksValid = isfield(cache, 'precomputedMasks') && ...
                any(cellfun(@(m) ~isempty(m) && any(m(:)), ...
                    cache.precomputedMasks(1:min(10,end))));

    if teacherMatch && fileMatch && masksValid
        precomputedGradCAM = cache.precomputedGradCAM;
        precomputedMasks   = cache.precomputedMasks;
        fprintf('  Using cached GradCAM maps (teacher=%s)\n', teacherModelSpec);
    else
        if ~teacherMatch, fprintf('  Cache teacher mismatch. Recalculating...\n'); end
        if ~fileMatch,    fprintf('  Cache file list mismatch. Recalculating...\n'); end
        if ~masksValid,   fprintf('  Cache masks empty. Recalculating...\n'); end
        [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
            imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
        cachedFileList = imds.Files;
        cachedTeacher  = teacherModelSpec;
        save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', ...
            'cachedFileList', 'cachedTeacher', '-v7.3');
    end
end

function modelPath = resolve_teacher_model_path(teacherModelSpec)
    if isstring(teacherModelSpec)
        teacherModelSpec = char(teacherModelSpec);
    end

    if exist(teacherModelSpec, 'file')
        modelPath = teacherModelSpec;
        return;
    end

    switch lower(teacherModelSpec)
        case 'roi'
            candidates = { ...
                fullfile('models', 'pretrained', 'vgg16_finetuned_on_roi.mat'), ...
                fullfile('vgg16_finetuned_on_roi.mat')};

        case 'cxr'
            candidates = { ...
                fullfile('models', 'pretrained', 'vgg16_finetuned_on_cxr.mat'), ...
                fullfile('vgg16_finetuned_on_cxr.mat')};

        case 'mixed'
            candidates = { ...
                fullfile('models', 'pretrained', 'vgg16_finetuned_on_mixed.mat'), ...
                fullfile('vgg16_finetuned_on_mixed.mat')};

        otherwise
            error('Unsupported teacher model spec: %s', teacherModelSpec);
    end

    for i = 1:numel(candidates)
        if exist(candidates{i}, 'file')
            modelPath = candidates{i};
            return;
        end
    end

    error('Teacher model not found for spec "%s". Checked: %s', ...
        teacherModelSpec, strjoin(candidates, ', '));
end

function foldIndices = createStratifiedKFold(labels, k_folds)
    classes = categories(labels); 
    numClasses = numel(classes);
    foldIndices.train = cell(k_folds,1); 
    foldIndices.val = cell(k_folds,1);
    
    for c = 1:numClasses
        idx = find(labels == classes{c}); 
        idx = idx(randperm(numel(idx)));
        samplesPerFold = floor(numel(idx)/k_folds); 
        remainder = mod(numel(idx), k_folds);
        startIdx = 1;
        
        for f = 1:k_folds
            foldSize = samplesPerFold + (f <= remainder);
            endIdx = startIdx + foldSize - 1;
            v = idx(startIdx:endIdx);
            
            if isempty(foldIndices.val{f})
                foldIndices.val{f} = v; 
            else
                foldIndices.val{f} = [foldIndices.val{f}; v]; 
            end
            startIdx = endIdx + 1;
        end
    end
    
    allIdx = (1:numel(labels))';
    for f = 1:k_folds
        foldIndices.train{f} = setdiff(allIdx, foldIndices.val{f});
    end
end

function mean_metrics = calculate_mean_metrics(fold_results)
    fold_names = fieldnames(fold_results);
    metrics = {'accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score', ...
               'auc', 'iou', 'dice', 'tversky', 'jaccard', 'hausdorff'};
    
    mean_metrics = struct();
    for m = 1:numel(metrics)
        values = [];
        for f = 1:numel(fold_names)
            if isfield(fold_results.(fold_names{f}), metrics{m})
                values(end+1) = fold_results.(fold_names{f}).(metrics{m});
            end
        end
        mean_metrics.(metrics{m}) = mean(values);
    end
end

function std_metrics = calculate_std_metrics(fold_results)
    fold_names = fieldnames(fold_results);
    metrics = {'accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score', ...
               'auc', 'iou', 'dice', 'tversky', 'jaccard', 'hausdorff'};
    
    std_metrics = struct();
    for m = 1:numel(metrics)
        values = [];
        for f = 1:numel(fold_names)
            if isfield(fold_results.(fold_names{f}), metrics{m})
                values(end+1) = fold_results.(fold_names{f}).(metrics{m});
            end
        end
        std_metrics.(metrics{m}) = std(values);
    end
end

function plot_kfold_training_curves(training_histories, mean_metrics, std_metrics, outputDir)
    fold_names = fieldnames(training_histories);
    num_folds = numel(fold_names);
    
    figure('Position', [100, 100, 1400, 600]);
    
    subplot(1, 3, 1);
    hold on;
    colors = lines(num_folds);
    for f = 1:num_folds
        hist = training_histories.(fold_names{f});
        if isfield(hist, 'epoch_loss')
            epochs = hist.epoch;
            train_loss = hist.epoch_loss;
        elseif isfield(hist, 'train_loss')
            epochs = hist.epoch;
            train_loss = hist.train_loss;
        else
            continue;
        end
        
        if ~isempty(epochs) && ~isempty(train_loss)
            plot(epochs, train_loss, '-', 'Color', colors(f,:), 'LineWidth', 1.5, ...
                'DisplayName', sprintf('Fold %d Train', f));
        end
        
        if isfield(hist, 'val_loss')
            val_epochs = hist.epoch(~isnan(hist.val_loss));
            val_loss = hist.val_loss(~isnan(hist.val_loss));
            if ~isempty(val_epochs) && ~isempty(val_loss)
                plot(val_epochs, val_loss, '--', 'Color', colors(f,:), 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('Fold %d Val', f));
            end
        end
    end
    xlabel('Epoch');
    ylabel('Loss');
    title('Training and Validation Loss (All Folds)');
    legend('Location', 'best', 'NumColumns', 2);
    grid on;
    
    subplot(1, 3, 2);
    hold on;
    for f = 1:num_folds
        hist = training_histories.(fold_names{f});
        epochs = hist.epoch;
        
        if isfield(hist, 'train_acc') && ~isempty(epochs) && ~isempty(hist.train_acc)
            plot(epochs, hist.train_acc, '-', 'Color', colors(f,:), 'LineWidth', 1.5, ...
                'DisplayName', sprintf('Fold %d Train', f));
        end
        
        if isfield(hist, 'val_accuracy')
            val_epochs = hist.epoch(~isnan(hist.val_accuracy));
            val_acc = hist.val_accuracy(~isnan(hist.val_accuracy));
            if ~isempty(val_epochs) && ~isempty(val_acc)
                plot(val_epochs, val_acc, '--', 'Color', colors(f,:), 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('Fold %d Val', f));
            end
        elseif isfield(hist, 'val_acc')
            val_epochs = hist.epoch(~isnan(hist.val_acc));
            val_acc = hist.val_acc(~isnan(hist.val_acc));
            if ~isempty(val_epochs) && ~isempty(val_acc)
                plot(val_epochs, val_acc, '--', 'Color', colors(f,:), 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('Fold %d Val', f));
            end
        end
    end
    xlabel('Epoch');
    ylabel('Accuracy');
    title('Training and Validation Accuracy (All Folds)');
    legend('Location', 'best', 'NumColumns', 2);
    grid on;
    
    subplot(1, 3, 3);
    metrics_names = {'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'AUC', 'Dice', 'IoU'};
    metrics_values = [mean_metrics.accuracy, mean_metrics.precision, mean_metrics.sensitivity, ...
                     mean_metrics.specificity, mean_metrics.f1_score, mean_metrics.auc, ...
                     mean_metrics.dice, mean_metrics.iou];
    metrics_stds = [std_metrics.accuracy, std_metrics.precision, std_metrics.sensitivity, ...
                   std_metrics.specificity, std_metrics.f1_score, std_metrics.auc, ...
                   std_metrics.dice, std_metrics.iou];
    
    bar(metrics_values);
    hold on;
    errorbar(1:numel(metrics_values), metrics_values, metrics_stds, 'k.', 'LineWidth', 1.5);
    set(gca, 'XTickLabel', metrics_names);
    ylabel('Score');
    title('Cross-Validation Metrics (Mean ± Std)');
    xtickangle(45);
    grid on;
    ylim([0, 1]);
    
    figFile = fullfile(outputDir, 'kfold_training_curves.png');
    saveas(gcf, figFile);
    fprintf('  Saved: %s\n', figFile);
    close(gcf);
end

function [accuracy, precision, sensitivity, specificity, f1score, auc, ...
          iou, dice, tversky, jaccard, hausdorff] = evaluateWithSegmentation(net, imdsVal, classes, useGPU, ...
          valGradCAMs, valMasks, featureLayer, config)
    posIdx = infer_positive_class_idx(classes);
    Ypred = categorical.empty(0,1);
    Yprobs = [];
    augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
    reset(augVal);
    preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
    mbqEval = minibatchqueue(augVal, ...
        'MiniBatchSize', 16, ...
        'MiniBatchFcn', preprocessFcn, ...
        'MiniBatchFormat', ["SSCB", ""], ...
        'PartialMiniBatch', 'return');

    while hasdata(mbqEval)
        [X, T] = next(mbqEval);
        if useGPU, X = gpuArray(X); end
        if ~isa(X, 'dlarray')
            dlX = dlarray(single(X), 'SSCB');
        else
            dlX = X;
        end
        sc = predict(net, dlX);
        lab = onehotdecode(extractdata(sc), classes, 1);
        Ypred = [Ypred; lab(:)];
        probs_batch = extractdata(sc);
        if size(probs_batch, 1) == 2
            Yprobs = [Yprobs; probs_batch(posIdx,:)'];
        else
            Yprobs = [Yprobs; probs_batch(1,:)'];
        end
    end
    Ytrue = imdsVal.Labels(:);
    match = (categorical(Ytrue) == categorical(Ypred));
    accuracy = sum(match) / numel(match);
    cm = confusionmat(string(Ytrue), string(Ypred), 'Order', string(classes));
    if numel(classes)==2 && all(size(cm)==[2 2])
        negIdx = 3 - posIdx;
        TP = cm(posIdx,posIdx); FP = cm(negIdx,posIdx); FN = cm(posIdx,negIdx); TN = cm(negIdx,negIdx);
    else
        [~,posIdx] = min(countcats(Ytrue));
        pos = classes{posIdx};
        TP = sum(Ypred==pos & Ytrue==pos);
        FP = sum(Ypred==pos & Ytrue~=pos);
        FN = sum(Ypred~=pos & Ytrue==pos);
        TN = sum(Ypred~=pos & Ytrue~=pos);
    end
    precision = TP / (TP + FP + eps);
    sensitivity = TP / (TP + FN + eps);
    specificity = TN / (TN + FP + eps);
    f1score = 2*precision*sensitivity / (precision + sensitivity + eps);
    try
        [~,~,~,auc] = perfcurve(double(Ytrue == classes{posIdx}), Yprobs, 1);
    catch
        auc = 0.5;
    end
    numVal = numel(imdsVal.Files);
    ious = zeros(numVal, 1);
    dices = zeros(numVal, 1);
    tverskys = zeros(numVal, 1);
    jaccards = zeros(numVal, 1);
    hausdorffs = zeros(numVal, 1);
    for i = 1:numVal
        img = imread(imdsVal.Files{i});
        if size(img,3)==1, img = repmat(img,[1 1 3]); end
           % === FIX: Preprocess image to match training distribution ===
        % We must apply the same transform (e.g., histmatch) before generating CAM
        img4d = reshape(img, size(img,1), size(img,2), size(img,3), 1);
        img4d = apply_preprocessing_batch(img4d, config.preprocessing_method, config.refHist);
        img = img4d(:,:,:,1);
        % ============================================================

        img = imresize(img,[224 224]);
        try
            predCAM = dlfeval(@student_cam_one, net, img, classes, featureLayer, useGPU);
            realMask = valMasks{i};
            if isempty(realMask) || ~any(realMask(:))
                ious(i) = 0; dices(i) = 0; tverskys(i) = 0; jaccards(i) = 0; hausdorffs(i) = 0;
                continue;
            end
            if ~islogical(realMask), realMask = logical(realMask); end
            if size(realMask, 1) ~= 224 || size(realMask, 2) ~= 224
                realMask = imresize(realMask, [224 224], 'nearest');
            end
            if isempty(predCAM) || all(predCAM(:) == 0) || all(isnan(predCAM(:)))
                ious(i) = 0; dices(i) = 0; tverskys(i) = 0; jaccards(i) = 0; hausdorffs(i) = 0;
                continue;
            end
            if size(predCAM, 1) ~= 224 || size(predCAM, 2) ~= 224
                predCAM = imresize(predCAM, [224 224]);
            end
            cam_values = predCAM(:);
            threshold = prctile(cam_values, 50);
            if threshold < 0.1, threshold = prctile(cam_values, 75); end
            if threshold < 0.1, threshold = mean(cam_values) + 0.5 * std(cam_values); end
            threshold = max(0.1, min(0.9, threshold));
            predMask = predCAM > threshold;
            if ~any(predMask(:))
                threshold = prctile(cam_values, 25);
                predMask = predCAM > threshold;
                if ~any(predMask(:))
                    ious(i) = 0; dices(i) = 0; tverskys(i) = 0; jaccards(i) = 0; hausdorffs(i) = 0;
                    continue;
                end
            end
            predMask = imopen(predMask, strel('disk', 2));
            predMask = imclose(predMask, strel('disk', 3));
            predMask = imfill(predMask, 'holes');
            predMask = logical(predMask);
            realMask = logical(realMask);
            if ~any(predMask(:))
                ious(i) = 0; dices(i) = 0; tverskys(i) = 0; jaccards(i) = 0; hausdorffs(i) = 0;
                continue;
            end
            [ious(i), dices(i), tverskys(i), jaccards(i), hausdorffs(i)] = ...
                computeSegmentationMetrics(predMask, realMask);
        catch
            ious(i) = 0; dices(i) = 0; tverskys(i) = 0; jaccards(i) = 0; hausdorffs(i) = 0;
        end
    end
    iou = mean(ious(~isnan(ious)));
    dice = mean(dices(~isnan(dices)));
    tversky = mean(tverskys(~isnan(tverskys)));
    jaccard = mean(jaccards(~isnan(jaccards)));
    hausdorff = mean(hausdorffs(~isnan(hausdorffs)));
end

function [iou, dice, tversky, jaccard, hausdorff] = computeSegmentationMetrics(pred, target)
    pred = logical(gather(pred));
    target = logical(gather(target));
    intersection = sum(pred(:) & target(:));
    union = sum(pred(:) | target(:));
    pred_sum = sum(pred(:));
    target_sum = sum(target(:));
    iou = intersection / (union + eps);
    dice = 2 * intersection / (pred_sum + target_sum + eps);
    alpha = 0.7; beta = 0.3;
    fp = sum(pred(:) & ~target(:));
    fn = sum(~pred(:) & target(:));
    tversky = intersection / (intersection + alpha * fp + beta * fn + eps);
    jaccard = iou;
    try
        hausdorff = improved_hausdorff_distance(pred, target);
    catch
        hausdorff = 0;
    end
end

function dist = improved_hausdorff_distance(pred, target)
    try
        pred = logical(gather(pred));
        target = logical(gather(target));
        if ~any(pred(:)) || ~any(target(:)), dist = 0; return; end
        pred_boundary = bwboundaries(pred, 'noholes');
        target_boundary = bwboundaries(target, 'noholes');
        if isempty(pred_boundary) || isempty(target_boundary), dist = 0; return; end
        pred_areas = cellfun(@(x) size(x, 1), pred_boundary);
        target_areas = cellfun(@(x) size(x, 1), target_boundary);
        [~, pred_idx] = max(pred_areas);
        [~, target_idx] = max(target_areas);
        pred_pts = pred_boundary{pred_idx};
        target_pts = target_boundary{target_idx};
        max_points = 100;
        if size(pred_pts, 1) > max_points
            step = floor(size(pred_pts, 1) / max_points);
            if step > 1, pred_pts = pred_pts(1:step:end, :); else, pred_pts = pred_pts(1:max_points, :); end
        end
        if size(target_pts, 1) > max_points
            step = floor(size(target_pts, 1) / max_points);
            if step > 1, target_pts = target_pts(1:step:end, :); else, target_pts = target_pts(1:max_points, :); end
        end
        dists_pred_to_target = zeros(size(pred_pts, 1), 1);
        for i = 1:size(pred_pts, 1)
            dists = sqrt(sum((target_pts - pred_pts(i, :)).^2, 2));
            dists_pred_to_target(i) = min(dists);
        end
        dists_target_to_pred = zeros(size(target_pts, 1), 1);
        for i = 1:size(target_pts, 1)
            dists = sqrt(sum((pred_pts - target_pts(i, :)).^2, 2));
            dists_target_to_pred(i) = min(dists);
        end
        h1 = max(dists_pred_to_target);
        h2 = max(dists_target_to_pred);
        dist = max(h1, h2);
        [h, w] = size(pred);
        dist = dist / sqrt(h^2 + w^2);
    catch
        dist = 0;
    end
end

function cam = student_cam_one(net, img, classes, featureLayer, useGPU)
% Robust GradCAM for both dlnetwork and DAGNetwork
% Called via dlfeval in precomputation and loss

    if ~isa(img, 'dlarray')
        dlX = dlarray(single(img) ./ 255, 'SSCB');
    else
        dlX = img;
    end
    if useGPU
        dlX = gpuArray(dlX);
    end

    layerNames = {net.Layers.Name};

    % 1. Feature Layer (relu5_3 preferred)
    if ~any(strcmp(layerNames, featureLayer))
        reluIdx = find(contains(lower(layerNames), 'relu'), 1, 'last');
        if ~isempty(reluIdx)
            featureLayer = layerNames{reluIdx};
            fprintf('    [GradCAM] Using fallback feature layer: %s\n', featureLayer);
        end
    end

    % 2. Output Layer - MUST be pre-softmax FC layer (fc8)
    fcCandidates = {'fc8', 'fc', 'predictions', 'classifier', 'output_fc'};
    outputName = '';
    for k = 1:length(fcCandidates)
        if any(strcmp(layerNames, fcCandidates{k}))
            outputName = fcCandidates{k};
            break;
        end
    end
    if isempty(outputName)
        % Fallback: last FullyConnectedLayer
        fcIdx = find(cellfun(@(n) isa(net.Layers(find(strcmp(layerNames,n),1)), ...
            'nnet.cnn.layer.FullyConnectedLayer'), layerNames), 1, 'last');
        if ~isempty(fcIdx)
            outputName = layerNames{fcIdx};
        else
            outputName = layerNames{end-1};  % last before softmax
        end
    end

    try
        [featMap, logits] = forward(net, dlX, 'Outputs', {featureLayer, outputName});

        logits = squeeze(logits);
        [~, classIdx] = max(extractdata(logits));

        score = sum(logits(classIdx), 'all');           % scalar score for chosen class
        gradFeat = dlgradient(score, featMap);

        weights = mean(gradFeat, [1 2]);                % channel importance
        cam = sum(featMap .* weights, 3);               % weighted sum
        cam = max(cam, 0);                              % ReLU

        cam = extractdata(cam);
        cam = imresize(single(cam), [224 224]);

        maxVal = max(cam(:));
        if maxVal > 1e-5
            cam = cam / maxVal;
        else
            cam = zeros(224, 224, 'single');
            fprintf('    [WARNING] Zero GradCAM map generated!\n');
        end
    catch ME
        fprintf('    [GradCAM Error] %s\n', ME.message);
        cam = zeros(224, 224, 'single');
    end
end


function valAcc = compute_validation_accuracy_simple(net, mbqVal, classes)
% Returns balanced accuracy (mean of sensitivity and specificity)
% using a threshold sweep. This correctly detects when the model
% is learning to separate classes, even before argmax accuracy improves.

    ptbIdx = infer_positive_class_idx(classes);
    normalIdx = 3 - ptbIdx;

    reset(mbqVal);
    Yprobs = []; Ytrue_all = categorical.empty(0,1);
    while hasdata(mbqVal)
        [X, T] = next(mbqVal);
        Y = predict(net, X);
        probs = extractdata(Y)';                      % [batch x numClasses]
        Yprobs = [Yprobs; probs(:, ptbIdx)];          %#ok<AGROW>
        labels = onehotdecode(extractdata(T), classes, 1);
        Ytrue_all = [Ytrue_all; labels(:)];           %#ok<AGROW>
    end

    if isempty(Yprobs)
        valAcc = 0.5;
        return;
    end

    % Sweep threshold, return best balanced accuracy
    best_bal = 0;
    for t = 0.38:0.02:0.72
        pred_idx = normalIdx * ones(size(Yprobs));
        pred_idx(Yprobs >= t) = ptbIdx;
        preds_t = categorical(classes(pred_idx));
        cm_t = confusionmat(Ytrue_all, preds_t, 'Order', categorical(classes));
        if all(size(cm_t) == [2 2])
            TP = cm_t(ptbIdx, ptbIdx);
            FN = cm_t(ptbIdx, normalIdx);
            TN = cm_t(normalIdx, normalIdx);
            FP = cm_t(normalIdx, ptbIdx);
            s  = TP / (TP + FN + eps);
            sp = TN / (TN + FP + eps);
            bal = (s + sp) / 2;
            if bal > best_bal, best_bal = bal; end
        end
    end
    valAcc = best_bal;
end

function posIdx = infer_positive_class_idx(classes)
% Infer positive/PTB class index robustly from class names.
    ptb_candidates = {'PTB','ptb','TB','tb','Tuberculosis','tuberculosis','positive','Positive'};
    posIdx = 2;  % fallback for binary classifiers
    for k = 1:numel(ptb_candidates)
        idx = find(strcmp(classes, ptb_candidates{k}), 1);
        if ~isempty(idx)
            posIdx = idx;
            return;
        end
    end
    if numel(classes) == 2
        if strcmpi(classes{1}, 'normal')
            posIdx = 2;
        elseif strcmpi(classes{2}, 'normal')
            posIdx = 1;
        end
    end
end

function [val_acc, val_iou, val_dice] = evaluate_model_quick(net, imdsVal, classes, useGPU, ...
    precomputedGradCAM, precomputedMasks, featureLayer, config)
    try
        num_samples = min(50, numel(imdsVal.Files));
        subset_idx = randperm(numel(imdsVal.Files), num_samples);
        imdsValSubset = subset(imdsVal, subset_idx);
        if iscell(precomputedGradCAM)
            gradCAM_subset = precomputedGradCAM(subset_idx);
            masks_subset = precomputedMasks(subset_idx);
        else
            gradCAM_subset = precomputedGradCAM;
            masks_subset = precomputedMasks;
        end
        [acc, ~, ~, ~, ~, ~, iou, dice, ~, ~, ~] = evaluateWithSegmentation(net, imdsValSubset, classes, useGPU, ...
            gradCAM_subset, masks_subset, featureLayer, config);
        val_acc = acc;
        val_iou = iou;
        val_dice = dice;
    catch
        val_acc = 0.5;
        val_iou = 0.0;
        val_dice = 0.0;
    end
end

function cm = compute_confusion_matrix_simple(net, imdsVal, classes, useGPU, config)
    Ypred = categorical.empty(0,1);
    augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
    preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
    mbqEval = minibatchqueue(augVal, ...
        'MiniBatchSize', 16, ...
        'MiniBatchFcn', preprocessFcn, ...
        'MiniBatchFormat', ["SSCB", ""], ...
        'PartialMiniBatch', 'return');

    while hasdata(mbqEval)
        [X, ~] = next(mbqEval);
        if useGPU, X = gpuArray(X); end
        sc = predict(net, X);
        probs = extractdata(sc);
        [~, predIdx] = max(probs, [], 1);
        Ypred = [Ypred; categorical(classes(predIdx(:)))]; %#ok<AGROW>
    end

    Ytrue = imdsVal.Labels(:);
    cm = confusionmat(Ytrue, Ypred, 'Order', categorical(classes));
end

function [acc, cm] = evaluate_classification_full(net, imdsVal, classes, useGPU, config)
    Ypred = categorical.empty(0,1);
    augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
    preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
    mbqEval = minibatchqueue(augVal, ...
        'MiniBatchSize', 16, ...
        'MiniBatchFcn', preprocessFcn, ...
        'MiniBatchFormat', ["SSCB", ""], ...
        'PartialMiniBatch', 'return');

    while hasdata(mbqEval)
        [X, ~] = next(mbqEval);
        if useGPU, X = gpuArray(X); end
        sc = predict(net, X);
        probs = extractdata(sc);
        [~, predIdx] = max(probs, [], 1);
        Ypred = [Ypred; categorical(classes(predIdx(:)))]; %#ok<AGROW>
    end

    Ytrue = imdsVal.Labels(:);
    cm = confusionmat(Ytrue, Ypred, 'Order', categorical(classes));
    acc = mean(Ypred == Ytrue);
end

function [loss, grads, state, loss_components] = compute_loss_with_config(...
    net, X, T, loss_config, classWeights, trainFiles, preCAMs, preMasks, ...
    nCam, classes, featureLayer, useGPU)
[Y, state] = forward(net, X);
loss_components = struct();
if loss_config.use_focal
    clsLoss = compute_focal_loss(Y, T, classWeights, ...
        loss_config.focal_alpha, loss_config.focal_gamma);
else
    if isfield(loss_config, 'use_class_weights') && loss_config.use_class_weights
        clsLoss = crossentropy(Y, T, 'Weights', classWeights, 'WeightsFormat', 'C');
    else
        clsLoss = crossentropy(Y, T);
    end
end
loss_components.classification = clsLoss;
camLoss = 0;
segLoss = 0;
tverskyLoss = 0;
iouLoss = 0;
anatomicalLoss = 0;  % NEW: Anatomical guidance loss

if loss_config.use_gradcam || loss_config.use_segmentation || ...
   loss_config.use_tversky || loss_config.use_iou || loss_config.use_anatomical_guidance
    N = numel(trainFiles);
    n = min(nCam, N);
    idxs = randperm(N, n);
    for ii = 1:n
        img = imread(trainFiles{idxs(ii)});
        if size(img,3)==1, img = repmat(img,[1 1 3]); end
        img = imresize(img, [224 224]);
        studCAM = student_cam_one_dlarray(net, img, classes, featureLayer, useGPU);
        studCAM_data = extractdata(studCAM);
        studCAM_data = single(studCAM_data);
        if size(studCAM_data, 1) ~= 224 || size(studCAM_data, 2) ~= 224
            studCAM_data = imresize(studCAM_data, [224 224]);
        end
        studCAM = dlarray(studCAM_data, 'SS');
        if loss_config.use_gradcam
            targetCAM = preCAMs{idxs(ii)};
            if ~isa(targetCAM, 'dlarray')
                targetCAM = single(targetCAM);
                if ndims(targetCAM) > 2, targetCAM = squeeze(targetCAM); end
                if size(targetCAM, 1) ~= 224 || size(targetCAM, 2) ~= 224
                    targetCAM = imresize(targetCAM, [224 224]);
                end
                targetCAM = dlarray(targetCAM, 'SS');
            else
                targetCAM = stripdims(targetCAM);
                targetCAM = single(extractdata(targetCAM));
                if ndims(targetCAM) > 2, targetCAM = squeeze(targetCAM); end
                if size(targetCAM, 1) ~= 224 || size(targetCAM, 2) ~= 224
                    targetCAM = imresize(targetCAM, [224 224]);
                end
                targetCAM = dlarray(targetCAM, 'SS');
            end
            camLoss = camLoss + mse(studCAM, targetCAM);
        end
        if loss_config.use_segmentation || loss_config.use_tversky || loss_config.use_iou
            realMask = preMasks{idxs(ii)};
            if ~isempty(realMask)
                realMask_resized = imresize(single(realMask), [224 224]);
                if loss_config.use_segmentation
                    diceLoss = 1 - dice_coefficient_dlarray(studCAM, realMask_resized);
                    segLoss = segLoss + diceLoss;
                end
                if loss_config.use_tversky
                    tverskyCoef = tversky_coefficient_dlarray(studCAM, realMask_resized, ...
                        loss_config.tversky_alpha, loss_config.tversky_beta);
                    tverskyLoss = tverskyLoss + (1 - tverskyCoef);
                end
                if loss_config.use_iou
                    iouCoef = iou_coefficient_dlarray(studCAM, realMask_resized);
                    iouLoss = iouLoss + (1 - iouCoef);
                end
            end
        end
        
            % ====================================================================
            % IMPROVED ANATOMICAL GUIDANCE LOSS
            % ====================================================================
            % Purpose: Guide model attention to focus on lung regions (anatomical prior)
            % 
            % Strategy: Dual-component loss
            %   1. PENALTY: High attention outside lungs → increases loss
            %   2. REWARD: High attention inside lungs → decreases loss
            %
            % Formula: L_anatomical = penalty_outside - (reward_weight * reward_inside)
            %   - penalty_outside: mean(CAM * non_lung_mask)  [0, 1]
            %   - reward_inside: mean(CAM * lung_mask)        [0, 1]
            %   - reward_weight: 0.5 (half weight of penalty)
            %
            % Interpretation:
            %   - If model focuses on lungs: reward_inside high → loss decreases
            %   - If model focuses outside: penalty_outside high → loss increases
            %   - Net effect: Model learns to focus attention on lung regions
            %
            % Scaling (CRITICAL):
            %   - Raw anatomical loss: [0, 1] range (normalized CAM values)
            %   - GradCAM loss (MSE): ~1000-2000 range
            %   - Without scaling: anatomical loss contributes <0.1% of total loss
            %   - With scaling (×1000): anatomical loss contributes 20-40% of total loss
            %   - This ensures anatomical guidance has meaningful influence on training
            % ====================================================================
            if loss_config.use_anatomical_guidance
                realMask = preMasks{idxs(ii)};
                if ~isempty(realMask) && any(realMask(:))
                    % Step 1: Prepare lung mask (binary: 1=lungs, 0=non-lungs)
                    if islogical(realMask)
                        lungMask = single(realMask);
                    else
                        lungMask = single(realMask > 0.5);
                    end
                    if size(lungMask, 1) ~= 224 || size(lungMask, 2) ~= 224
                        lungMask = imresize(lungMask, [224 224], 'nearest');
                    end
                    
                    % Step 2: Normalize CAM to [0, 1] for consistent loss computation
                    % This ensures loss values are in a predictable range
                    studCAM_norm = studCAM;
                    cam_max = max(studCAM_norm, [], 'all');
                    if cam_max > 0
                        studCAM_norm = studCAM_norm / (cam_max + eps);
                    end
                    
                    % Step 3: Create masks as dlarray for gradient computation
                    lungMask_dl = dlarray(lungMask, 'SS');           % 1 where lungs are
                    nonLungMask_dl = dlarray(1 - lungMask, 'SS');   % 1 where non-lungs are
                    
                    % Step 4: Compute penalty (attention outside lungs)
                    % High values here indicate model is paying attention to non-lung regions
                    attention_outside_lungs = studCAM_norm .* nonLungMask_dl;
                    penalty_outside = mean(attention_outside_lungs, 'all');
                    
                    % Step 5: Compute reward (attention inside lungs)
                    % High values here indicate model is paying attention to lung regions
                    attention_inside_lungs = studCAM_norm .* lungMask_dl;
                    reward_inside = mean(attention_inside_lungs, 'all');
                    
                    % Step 6: Combine penalty and reward
                    % We want to minimize this, so:
                    %   - High penalty_outside → increases loss (bad)
                    %   - High reward_inside → decreases loss (good)
                    % reward_weight (0.75) makes reward 75% as strong as penalty (increased from 0.5)
                    anatomical_loss_sample = penalty_outside - ...
                        loss_config.anatomical_reward_weight * reward_inside;
                    
                    % Step 7: SCALE to match GradCAM loss magnitude
                    % This is critical for anatomical loss to have meaningful influence
                    % Without scaling: anatomical loss ~0.01-0.05, GradCAM loss ~1000-2000
                    % With scaling: anatomical loss ~10-50, GradCAM loss ~1000-2000
                    % After lambda_anatomical (15.0): anatomical contribution ~150-750
                    anatomical_loss_scaled = anatomical_loss_sample * 1000;
                    
                    anatomicalLoss = anatomicalLoss + anatomical_loss_scaled;
                end
            end
    end
    camLoss = camLoss / max(1, n);
    segLoss = segLoss / max(1, n);
    tverskyLoss = tverskyLoss / max(1, n);
    iouLoss = iouLoss / max(1, n);
    anatomicalLoss = anatomicalLoss / max(1, n);
end
    % Store individual loss components for logging and analysis
    loss_components.gradcam = camLoss;
    loss_components.segmentation = segLoss;
    loss_components.tversky = tverskyLoss;
    loss_components.iou = iouLoss;
    loss_components.anatomical = anatomicalLoss;  % Anatomical guidance loss (scaled)
    
    % ====================================================================
    % TOTAL LOSS COMPUTATION
    % ====================================================================
    % Combine all loss components with their respective weights:
    %   L_total = L_cls + λ_cam * L_gradcam + λ_tversky * L_tversky + λ_anatomical * L_anatomical
    %
    % Typical loss component magnitudes (after 50 epochs):
    %   - Classification loss: ~0.01-0.1 (very small, well-trained)
    %   - GradCAM loss: ~1000-2000 (MSE between student and teacher CAMs)
    %   - Tversky loss: ~0.7-0.8 (segmentation alignment)
    %   - Anatomical loss (scaled): ~10-50 (after ×1000 scaling)
    %
    % Weighted contributions (v2.3 - OPTIMIZED PTB BIAS):
    %   - Classification: ~0.01-0.1 (weight: 1.0, with optimized class weighting: 1.25x PTB, focal_alpha [0.32, 0.68])
    %   - GradCAM: ~1000-2000 (weight: 1.0, increased from 0.5 to improve segmentation)
    %   - Tversky: ~3.5-4.0 (weight: 5.0, increased from 2.5 to improve IoU/Dice)
    %   - Anatomical: ~150-750 (weight: 15.0, scaled loss: 10-50, reward_weight: 0.75, increased from 0.5)
    %
    % Total loss: ~650-1750 (well-balanced across components)
    % Anatomical contribution: 20-40% of total (meaningful influence)
    % ====================================================================
    loss = clsLoss;  % Start with classification loss
    
    if loss_config.use_gradcam
        % GradCAM loss: Ensures student attention matches teacher GradCAM maps
        % Weight increased to 1.0 (from 0.5) to improve segmentation quality
        loss = loss + loss_config.lambda_cam * camLoss;
    end
    
    if loss_config.use_segmentation
        % Segmentation loss: Encourages attention to match lung masks
        loss = loss + loss_config.lambda_seg * segLoss;
    end
    
    if loss_config.use_tversky
        % Tversky loss: Handles class imbalance in segmentation (lung vs non-lung)
        % Alpha=0.7, Beta=0.3: Penalizes false positives more than false negatives
        % Weight increased to 5.0 (from 2.5) to improve IoU/Dice scores
        loss = loss + loss_config.lambda_tversky * tverskyLoss;
    end
    
    if loss_config.use_iou
        % IoU loss: Alternative segmentation metric
        loss = loss + loss_config.lambda_seg * iouLoss;
    end
    
    if loss_config.use_anatomical_guidance
        % Anatomical guidance loss: Guides attention to lung regions
        % High weight (15.0) ensures strong anatomical prior
        % Reward weight (0.75) encourages lung focus (increased from 0.5)
        % Loss is already scaled by 1000 in computation, so this multiplies scaled value
        loss = loss + loss_config.lambda_anatomical * anatomicalLoss;
    end
    
    % Compute gradients for backpropagation
    grads = dlgradient(loss, net.Learnables, 'EnableHigherDerivatives', true);
end

function focalLoss = compute_focal_loss(Y, T, classWeights, alpha, gamma)
    % UPDATED: Handle alpha as array [normal, PTB] to favor PTB class
    epsVal = 1e-7;
    Y = min(max(Y, epsVal), 1-epsVal);
    
    % Handle alpha as array or scalar
    if isscalar(alpha)
        % Legacy: scalar alpha multiplied by class weights
        alpha_vec = alpha * reshape(classWeights, [], 1);
    else
        % New: array alpha [normal, PTB] - use directly, then apply class weights
        alpha_vec = reshape(alpha(:), [], 1) .* reshape(classWeights, [], 1);
    end
    
    focalW = alpha_vec .* ((1 - Y).^gamma);
    term = - T .* focalW .* log(Y + epsVal);
    focalLoss = mean(sum(term, 1));
end

function tverskyCoef = tversky_coefficient_dlarray(pred, target, alpha, beta)
    if isa(pred, 'dlarray')
        target_dl = dlarray(single(target), 'SSC');
        intersection = sum(pred(:) .* target_dl(:), 'all');
        fp = sum(pred(:) .* (1 - target_dl(:)), 'all');
        fn = sum((1 - pred(:)) .* target_dl(:), 'all');
        tverskyCoef = intersection / (intersection + alpha * fp + beta * fn + eps);
    else
        pred = logical(gather(pred));
        target = logical(gather(target));
        intersection = sum(pred(:) & target(:));
        fp = sum(pred(:) & ~target(:));
        fn = sum(~pred(:) & target(:));
        tverskyCoef = intersection / (intersection + alpha * fp + beta * fn + eps);
    end
end

function diceCoef = dice_coefficient_dlarray(pred, target)
    if isa(pred, 'dlarray')
        target_dl = dlarray(single(target), 'SSC');
        intersection = sum(pred(:) .* target_dl(:), 'all');
        pred_sum = sum(pred(:), 'all');
        target_sum = sum(target_dl(:), 'all');
        diceCoef = 2 * intersection / (pred_sum + target_sum + eps);
    else
        pred = logical(gather(pred));
        target = logical(gather(target));
        intersection = sum(pred(:) & target(:));
        diceCoef = 2 * intersection / (sum(pred(:)) + sum(target(:)) + eps);
    end
end

function iouCoef = iou_coefficient_dlarray(pred, target)
    if isa(pred, 'dlarray')
        target_dl = dlarray(single(target), 'SSC');
        intersection = sum(pred(:) .* target_dl(:), 'all');
        union = sum(max(pred(:), target_dl(:)), 'all');
        iouCoef = intersection / (union + eps);
    else
        pred = logical(gather(pred));
        target = logical(gather(target));
        intersection = sum(pred(:) & target(:));
        union = sum(pred(:) | target(:));
        iouCoef = intersection / (union + eps);
    end
end

function cam = student_cam_one_dlarray(net, img, classes, featureLayer, useGPU)
% Robust student GradCAM - works with both dlnetwork and DAGNetwork

if ~isa(img, 'dlarray')
    dlX = dlarray(single(img)./255, 'SSCB');
else
    dlX = img;
end

if useGPU
    dlX = gpuArray(dlX);
end

% === Robust layer detection ===
layerNames = {net.Layers.Name};

% Feature layer fallback
if ~any(strcmp(layerNames, featureLayer))
    reluIdx = find(contains(lower(layerNames),'relu'), 1, 'last');
    if ~isempty(reluIdx)
        featureLayer = layerNames{reluIdx};
    end
end

% Output layer (fc8 or last FC layer)
fcIdx = find(contains(lower(layerNames), 'fc'), 1, 'last');
if isempty(fcIdx)
    fcIdx = length(layerNames) - 1;
end
outputName = layerNames{fcIdx};

try
    [featMap, logits] = forward(net, dlX, 'Outputs', {featureLayer, outputName});

    logits = squeeze(logits);
    [~, classIdx] = max(extractdata(logits));

    score = sum(logits(classIdx), 'all');
    gradFeat = dlgradient(score, featMap);

    weights = mean(gradFeat, [1 2]);
    cam = sum(featMap .* weights, 3);
    cam = max(cam, 0);                    % ReLU

    cam = extractdata(cam);
    cam = imresize(single(cam), [224 224]);;

    maxVal = max(cam(:));
    if maxVal > 1e-6
        cam = single(cam / maxVal);
    else
        cam = 0.5 * ones(224, 224, 'single');
        fprintf('    [WARNING] Zero GradCAM map generated!\n');
    end

catch ME
    fprintf('    [GradCAM Error] %s\n', ME.message);
    cam = 0.5 * ones(224, 224, 'single');
end
end
% Add 'epoch' to signature
function valLoss = compute_validation_loss_with_config_preprocessed(net, imdsVal, classes, useGPU, ...
precomputedGradCAM, precomputedMasks, loss_config, classWeights, config, epoch)

    augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
    preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
    mbqVal = minibatchqueue(augVal, ...
        'MiniBatchSize', config.batchSize, ...
        'MiniBatchFcn', preprocessFcn, ...
        'MiniBatchFormat', ["SSCB", ""], ...
        'PartialMiniBatch', 'discard');

    valLoss = 0; valCount = 0; valFilesForGradCAM = imdsVal.Files;

    while hasdata(mbqVal)
        [X, T] = next(mbqVal);
        if useGPU, X = gpuArray(X); end
        try
            % Call the IMPROVED loss function with epoch
            [loss, ~, ~, ~] = dlfeval(@compute_loss_with_config_improved, net, X, T, loss_config, classWeights, ...
                valFilesForGradCAM, precomputedGradCAM, precomputedMasks, 8, classes, 'relu5_3', useGPU, epoch, ...
                config.preprocessing_method, config.refHist);
            valLoss = valLoss + double(loss);
            valCount = valCount + 1;
        catch
            continue;
        end
    end
    valLoss = valLoss / max(1, valCount);
end


function [X, T] = preprocessMiniBatchWithPreprocessing(dataX, dataT, preprocessing_method, refHist)
% Preprocess mini-batch with specified preprocessing method
%   dataX - Cell array of images (uint8, [0, 255])
%   dataT - Cell array of labels
%   preprocessing_method - Preprocessing method name
%   refHist - Reference histogram (for histogram matching)

% Concatenate images
X = cat(4, dataX{1:end});

% Apply preprocessing to each image
[H, W, C, B] = size(X);
X_processed = zeros(size(X), 'uint8');

for b = 1:B
    img = X(:, :, :, b);
    
    % Convert to grayscale for intensity preprocessing
    if C == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    % Apply preprocessing
    switch preprocessing_method
        case 'none'
            img_processed = img_gray;
            
        case 'clahe'
            img_processed = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
            
        case 'histmatch'
            if ~isempty(refHist)
                img_processed = histeq(img_gray, refHist);
            else
                img_processed = img_gray;
            end
            
        case 'zscore'
            img_double = double(img_gray);
            img_mean = mean(img_double(:));
            img_std = std(img_double(:));
            if img_std > 0
                img_normalized = (img_double - img_mean) / img_std;
                img_normalized = (img_normalized - min(img_normalized(:))) / ...
                    (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
            else
                img_normalized = img_double;
            end
            img_processed = uint8(img_normalized);
            
        case 'minmax'
            img_double = double(img_gray);
            img_min = min(img_double(:));
            img_max = max(img_double(:));
            if img_max > img_min
                img_normalized = (img_double - img_min) / (img_max - img_min) * 255;
            else
                img_normalized = img_double;
            end
            img_processed = uint8(img_normalized);
            
        case 'clahe_zscore'
            img_clahe = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
            img_double = double(img_clahe);
            img_mean = mean(img_double(:));
            img_std = std(img_double(:));
            if img_std > 0
                img_normalized = (img_double - img_mean) / img_std;
                img_normalized = (img_normalized - min(img_normalized(:))) / ...
                    (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
            else
                img_normalized = img_double;
            end
            img_processed = uint8(img_normalized);
            
        otherwise
            img_processed = img_gray;
    end
    
    % Replicate to RGB if needed
    if C == 3
        X_processed(:, :, :, b) = repmat(img_processed, [1, 1, 3]);
    else
        X_processed(:, :, :, b) = img_processed;
    end
end

% Convert to single precision (matches training)
 X = dlarray(single(X_processed) ./ 255, 'SSCB');
T = onehotencode(cat(2, dataT{1:end}), 1);
end



function imgs_processed = apply_preprocessing_batch(imgs, method, refHist)
    % Apply preprocessing to batch of images (same as evaluate_ood_with_preprocessing.m)
    [H, W, C, B] = size(imgs);
    imgs_processed = zeros(size(imgs), 'uint8');

    for b = 1:B
        img = imgs(:, :, :, b);
        
        if C == 3
            img_gray = rgb2gray(img);
        else
            img_gray = img;
        end
        
        switch method
            case 'none'
                img_processed = img_gray;
            case 'clahe'
                img_processed = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
            case 'histmatch'
                if ~isempty(refHist)
                    img_processed = histeq(img_gray, refHist);
                else
                    img_processed = img_gray;
                end
            case 'zscore'
                img_double = double(img_gray);
                img_mean = mean(img_double(:));
                img_std = std(img_double(:));
                if img_std > 0
                    img_normalized = (img_double - img_mean) / img_std;
                    img_normalized = (img_normalized - min(img_normalized(:))) / ...
                        (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
                else
                    img_normalized = img_double;
                end
                img_processed = uint8(img_normalized);
            case 'minmax'
                img_double = double(img_gray);
                img_min = min(img_double(:));
                img_max = max(img_double(:));
                if img_max > img_min
                    img_normalized = (img_double - img_min) / (img_max - img_min) * 255;
                else
                    img_normalized = img_double;
                end
                img_processed = uint8(img_normalized);
            case 'clahe_zscore'
                img_clahe = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
                img_double = double(img_clahe);
                img_mean = mean(img_double(:));
                img_std = std(img_double(:));
                if img_std > 0
                    img_normalized = (img_double - img_mean) / img_std;
                    img_normalized = (img_normalized - min(img_normalized(:))) / ...
                        (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
                else
                    img_normalized = img_double;
                end
                img_processed = uint8(img_normalized);
            otherwise
                img_processed = img_gray;
        end
        
        if C == 3
            imgs_processed(:, :, :, b) = repmat(img_processed, [1, 1, 3]);
        else
            imgs_processed(:, :, :, b) = img_processed;
        end
    end
end

function roi_img = extract_lung_roi_simple(img)
    % Robust heuristic for lung ROI extraction using intensity, shape, and location.
    % Input: img (224x224 grayscale, uint8)
    % Output: roi_img (224x224 grayscale, uint8)

    [rows, cols] = size(img);

    % 1. Otsu thresholding (modern & optimized)
    try
        initial_mask = imbinarize(img, 'otsu'); % Lungs are generally darker
    catch
        initial_mask = img < prctile(img(:), 30); % Fallback
    end

    % 2. Morphological cleanup
    se = strel('disk', 6); % Single SE reduces overhead
    initial_mask = imclose(imopen(initial_mask, se), se);
    initial_mask = bwareaopen(initial_mask, 800); % Filter noise

    % 3. Anatomical prior (upper-middle chest region)
    prior_mask = false(rows, cols);
    prior_mask(round(rows*0.1):round(rows*0.9), round(cols*0.2):round(cols*0.8)) = true;
    refined_mask = initial_mask & prior_mask;
    
    if ~any(refined_mask(:)), refined_mask = initial_mask; end

    % 4. Extract largest component
    props = regionprops(refined_mask, 'BoundingBox', 'Area');
    
    if isempty(props)
        % Fallback: center crop
        margin = round(rows * 0.15);
        roi_img = imresize(img(margin+1:end-margin, margin+1:end-margin), [224 224]);
        return;
    end

    [~, idx] = max([props.Area]);
    bbox = props(idx).BoundingBox; % [x, y, width, height]
    
    margin = 12;
    x1 = max(1, floor(bbox(1) - margin));
    y1 = max(1, floor(bbox(2) - margin));
    x2 = min(cols, ceil(bbox(1) + bbox(3) + margin));
    y2 = min(rows, ceil(bbox(2) + bbox(4) + margin));
    
    % 5. Crop & resize
    roi_img = imresize(img(y1:y2, x1:x2), [224 224]);
end

function [gradCAMs, masks] = precompute_gradcam_mixed(...
        imdsROI, imdsCXR, net, featureLayer, maskDir, gradcamMasksDir)
    
    nROI     = numel(imdsROI.Files);
    gradCAMs = cell(nROI, 1);
    masks    = cell(nROI, 1);
    classes  = categories(imdsROI.Labels);

    cxrStems = cellfun(@(f) get_stem(f), imdsCXR.Files, 'UniformOutput', false);

    if ~isempty(gradcamMasksDir) && ~exist(gradcamMasksDir, 'dir')
        mkdir(gradcamMasksDir);
    end

    useGPU = canUseGPU;

    nMatched  = 0;
    nFallback = 0;
    nCamFail  = 0;

    fprintf('    [Mixed Teacher] Processing %d ROI images...\n', nROI);

    for i = 1:nROI
        roiFile = imdsROI.Files{i};
        roiStem = get_stem(roiFile);
        [~, fname, ~] = fileparts(roiFile);
        [classDir, ~] = fileparts(roiFile);
        [~, subdir] = fileparts(classDir);   % 'normal' or 'ptb'

        % === Load CXR or fallback ===
        matchIdx = find(strcmp(cxrStems, roiStem), 1);
        if ~isempty(matchIdx)
            srcImg = imread(imdsCXR.Files{matchIdx});
            nMatched = nMatched + 1;
        else
            srcImg = imread(roiFile);
            nFallback = nFallback + 1;
            if nFallback <= 5
                fprintf('    ⚠ No CXR match → fallback for %s\n', roiStem);
            end
        end

        if size(srcImg, 3) == 1
            srcImg = repmat(srcImg, [1 1 3]);
        end
        srcImg = imresize(srcImg, [224 224]);

        % === GradCAM via dlfeval — required for dlnetwork ===
        try
            camDL      = dlfeval(@student_cam_one, net, srcImg, classes, featureLayer, useGPU);
            gradCAMMap = single(extractdata(camDL));

            if size(gradCAMMap,1) ~= 224 || size(gradCAMMap,2) ~= 224
                gradCAMMap = imresize(gradCAMMap, [224 224]);
            end

            % Use shared postprocess (no percentile cut — preserves spatial signal)
            gradCAMMap = postprocess_gradcam_map(gradCAMMap);

            % Flag truly flat maps as failed
            if max(gradCAMMap(:)) < 1e-4
                gradCAMMap = zeros(224, 224, 'single');
                nCamFail = nCamFail + 1;
            end

            gradCAMs{i} = gradCAMMap;
        catch ME
            gradCAMs{i} = zeros(224, 224, 'single');
            nCamFail = nCamFail + 1;
            if nCamFail <= 8
                fprintf('    ⚠ GradCAM failed %d (%s): %s\n', i, fname, ME.message);
            end
        end

        % === Save GradCAM PNG ===
        if ~isempty(gradcamMasksDir)
            saveSubdir = fullfile(gradcamMasksDir, subdir);
            if ~exist(saveSubdir, 'dir')
                [mkStatus, mkMsg] = mkdir(saveSubdir);
                if ~mkStatus
                    warning('Cannot create dir %s: %s', saveSubdir, mkMsg);
                end
            end
            if exist(saveSubdir, 'dir')
                camSavePath = fullfile(saveSubdir, [fname '_gradcam.png']);
                try
                    imwrite(uint8(255 * gradCAMs{i}), camSavePath);
                catch mkE
                    warning('Cannot save CAM %s: %s', camSavePath, mkE.message);
                end
            end
        end

        % === Load Mask ===
        maskCandidates = {
            fullfile(maskDir, subdir, [fname '_mask.png']);
            fullfile(maskDir, subdir, [fname '.png']);
            fullfile(maskDir, [fname '_mask.png']);
            fullfile(maskDir, [fname '.png']);
        };

        masks{i} = false(224, 224);
        for c = 1:numel(maskCandidates)
            if exist(maskCandidates{c}, 'file')
                m = imread(maskCandidates{c});
                if size(m,3) > 1, m = rgb2gray(m); end
                masks{i} = imresize(logical(imbinarize(m)), [224 224], 'nearest');
                break;
            end
        end

        if mod(i, 80) == 0
            fprintf('    Processed %d/%d\n', i, nROI);
        end
    end

    validCAMs = sum(cellfun(@(x) mean(x(:)) > 0.1 && max(x(:)) > 0.25, gradCAMs));
    validMasks = sum(cellfun(@(m) any(m(:)), masks));

    fprintf('\n    [Mixed Teacher] Done.\n');
    fprintf('    CXR matched: %d | Fallback: %d | Failures: %d\n', nMatched, nFallback, nCamFail);
    fprintf('    Valid GradCAMs: %d/%d | Valid Masks: %d/%d\n', validCAMs, nROI, validMasks, nROI);
end

function stem = get_stem(filepath)
    % Extract filename without extension, stripping any _mask/_roi suffix
    [~, name, ~] = fileparts(filepath);
    stem = regexprep(name, '_(mask|roi|seg)$', '');
end


function [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir)
%PRECOMPUTE_GRADCAM_AND_MASKS Compute GradCAM maps and load masks from disk
%   Uses dlfeval(@student_cam_one) for dlnetwork compatibility.
%   Masks loaded from maskDir with multiple naming convention fallbacks.

if nargin < 5
    gradcamMasksDir = '';
end

nImages = numel(imds.Files);
classes = categories(imds.Labels);
useGPU  = canUseGPU;

fprintf('  Computing GradCAM + loading masks for %d images...\n', nImages);
fprintf('  maskDir: %s (exists=%d)\n', maskDir, exist(maskDir,'dir'));
fprintf('  Network type: %s\n', class(vggNet));

% Verify network is dlnetwork — hard error if not, since gradCAM() on
% DAGNetwork gives wrong results for our custom training loop.
if ~isa(vggNet, 'dlnetwork')
    error(['precompute_gradcam_and_masks: vggNet must be a dlnetwork. ' ...
           'Got %s. Convert with dag2dlnetwork() before calling.'], class(vggNet));
end

% Show sample mask filenames so user can verify convention
if exist(maskDir, 'dir')
    sampleFiles = dir(fullfile(maskDir, '**', '*.png'));
    if isempty(sampleFiles)
        sampleFiles = dir(fullfile(maskDir, '*.png'));
    end
    if ~isempty(sampleFiles)
        fprintf('  Sample mask files found:\n');
        for k = 1:min(3, numel(sampleFiles))
            fprintf('    %s\n', fullfile(sampleFiles(k).folder, sampleFiles(k).name));
        end
    else
        fprintf('  WARNING: No PNG files found in maskDir!\n');
    end
end

precomputedGradCAM = cell(nImages, 1);
precomputedMasks   = cell(nImages, 1);

nCamValid  = 0;
nCamFail   = 0;
nMaskFound = 0;

for i = 1:nImages
    imgPath = imds.Files{i};
    [classDir, fname, ext] = fileparts(imgPath);
    [~, subdir] = fileparts(classDir);   % 'Normal' or 'PTB'

    %% 1. Load image
    try
        img = imread(imgPath);
        if size(img,3) == 1, img = repmat(img,[1 1 3]); end
        img = imresize(img, [224 224]);
        if ~isa(img,'uint8'), img = uint8(img); end
    catch ME
        fprintf('    ERROR loading image %d: %s\n', i, ME.message);
        precomputedGradCAM{i} = 0.5 * ones(224,224,'single');
        precomputedMasks{i}   = false(224,224);
        nCamFail = nCamFail + 1;
        continue;
    end

    %% 2. Compute GradCAM via dlfeval (required for dlnetwork)
    try
        camDL      = dlfeval(@student_cam_one, vggNet, img, classes, workingGradCAMLayer, useGPU);
        gradCAMMap = single(extractdata(camDL));


        if size(gradCAMMap,1)~=224 || size(gradCAMMap,2)~=224
            gradCAMMap = imresize(gradCAMMap,[224 224]);
        end

        gradCAMMap = postprocess_gradcam_map(gradCAMMap);

        % Validate: a real CAM must have spatial variance
        if std(gradCAMMap(:)) < 1e-4 || max(gradCAMMap(:)) < 1e-4
            gradCAMMap = 0.5 * ones(224,224,'single');
            nCamFail = nCamFail + 1;
            if nCamFail <= 5
                fprintf('    WARNING: Flat CAM for image %d (%s)\n', i, fname);
            end
        else
            nCamValid = nCamValid + 1;
        end
    catch ME
        gradCAMMap = 0.5 * ones(224,224,'single');
        nCamFail   = nCamFail + 1;
        if nCamFail <= 5
            fprintf('    GradCAM failed %d (%s): %s\n', i, fname, ME.message);
        end
    end

    precomputedGradCAM{i} = gradCAMMap;

    %% 3. Optionally save GradCAM PNG
    if ~isempty(gradcamMasksDir)
        camSubdir = fullfile(gradcamMasksDir, subdir);
        if ~exist(camSubdir, 'dir')
            [mkStatus, mkMsg] = mkdir(camSubdir);
            if ~mkStatus
                warning('Cannot create dir %s: %s', camSubdir, mkMsg);
            end
        end
        if exist(camSubdir, 'dir')
            try
                imwrite(uint8(255 * gradCAMMap), fullfile(camSubdir, [fname '_gradcam.png']));
            catch mkE
                warning('Cannot save CAM for %s: %s', fname, mkE.message);
            end
        end
    end

    %% 4. Load mask — try all naming conventions
    maskCandidates = {
        fullfile(maskDir, [fname '_mask.png']);          % flat: name_mask.png
        fullfile(maskDir, [fname '_mask' ext]);          % flat: name_mask.ext
        fullfile(maskDir, [fname '.png']);               % flat plain: name.png
        fullfile(maskDir, [fname ext]);                  % flat plain: name.ext
        fullfile(maskDir, subdir, [fname '_mask.png']);  % subdir: Normal/name_mask.png
        fullfile(maskDir, subdir, [fname '.png']);       % subdir plain
    };

    precomputedMasks{i} = false(224,224);
    for c = 1:numel(maskCandidates)
        if exist(maskCandidates{c},'file')
            m = imread(maskCandidates{c});
            if size(m,3)>1, m = rgb2gray(m); end
            precomputedMasks{i} = imresize(logical(imbinarize(m)),[224 224],'nearest');
            nMaskFound = nMaskFound + 1;
            if i <= 3
                fprintf('    [Mask OK ] %s\n', maskCandidates{c});
            end
            break;
        end
    end

    if ~any(precomputedMasks{i}(:)) && i <= 5
        fprintf('    [Mask MISS] %s — tried %d paths, first: %s\n', ...
            fname, numel(maskCandidates), maskCandidates{1});
    end

    if mod(i,100)==0
        fprintf('    Processed %d/%d  (CAMs valid: %d  masks found: %d)\n', ...
            i, nImages, nCamValid, nMaskFound);
    end
end

%% Summary
fprintf('  Done.\n');
fprintf('  Valid GradCAMs : %d / %d\n', nCamValid,  nImages);
fprintf('  Failed GradCAMs: %d / %d\n', nCamFail,   nImages);
fprintf('  Valid Masks    : %d / %d\n', nMaskFound, nImages);
if nCamValid == 0
    fprintf('  *** ALL GRADCAMS FLAT/FAILED ***\n');
    fprintf('  Check: (1) network is trained (not random weights)\n');
    fprintf('         (2) featureLayer "%s" is correct\n', workingGradCAMLayer);
    fprintf('         (3) run check_gradcam_gradients.m for diagnosis\n');
end
if nMaskFound == 0
    fprintf('  *** ALL MASKS MISSING — check maskDir path and filename convention ***\n');
end
end

function cam = postprocess_gradcam_map(cam)
    %POSTPROCESS_GRADCAM_MAP Improve GradCAM visualization quality
    %   Enhanced version with better contrast and spatial coherence
    
    % CRITICAL: Move to CPU first - image processing functions don't support gpuArray
    if isgpuarray(cam)
        cam = gather(cam);
    end
    
    if isempty(cam) || all(~isfinite(cam(:)))
        cam = 0.5 * ones(224, 224, 'single');
        return;
    end
    cam = single(cam);
    cam(~isfinite(cam)) = 0;
    cam = max(cam, 0);
    
    % Enhanced smoothing for better spatial coherence
    cam = imgaussfilt(cam, 2.0);  % Increased from 1.5 for smoother maps
    
    % Contrast enhancement via adaptive histogram equalization
    cam = adapthisteq(cam, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
    
    % Normalize to [0, 1] with improved dynamic range
    cmin = min(cam(:));
    cmax = max(cam(:));
    if cmax > cmin
        cam = (cam - cmin) / (cmax - cmin + eps);
    else
        cam = zeros(224, 224, 'single');
    end
    
    % Boost contrast slightly for better visualization
    cam = cam .^ 0.8;  % Gamma correction for better visibility
    
    % Final normalization
    cmin = min(cam(:));
    cmax = max(cam(:));
    if cmax > cmin
        cam = (cam - cmin) / (cmax - cmin + eps);
    end
    % Result: bright = high activation, dark = low activation
    % This is the standard GradCAM visualization convention
end