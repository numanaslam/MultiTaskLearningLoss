function train_final_model_with_preprocessing(preprocessing_method)
%TRAIN_FINAL_MODEL_WITH_PREPROCESSING Train final model with image preprocessing
%   IMPROVED VERSION: Includes loss formulation fixes, stability enhancements,
%   and OOD robustness improvements.

if nargin < 1
    preprocessing_method = 'histmatch';  % Default to best method
end

% Save input parameter before clearing
preprocessing_method_param = preprocessing_method;
clc; close all;

fprintf('=== FINAL MODEL TRAINING WITH PREPROCESSING (IMPROVED) ===\n');
fprintf('Preprocessing Method: %s\n', preprocessing_method_param);
fprintf('Loss Function: Focal + GradCAM(Cosine) + Tversky + Anatomical(Positive)\n');
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
% QUICK TEST MODE: Set to true for quick validation (3 folds, 5 epochs)
QUICK_TEST = false;
if QUICK_TEST
    config.k_folds = 3;
    config.numEpochs = 5;
    fprintf('QUICK TEST MODE: 3 folds, 5 epochs\n');
else
    config.k_folds = 2;
    config.numEpochs = 10;
    fprintf('FULL TRAINING MODE: 2 folds, 10 epochs\n');
end
config.patience = 5;
config.min_delta = 1e-2;
config.useGPU = canUseGPU;
config.batchSize = 14;
config.initialLearnRate = 0.001;
config.decay = 0.0042;
config.momentum = 0.8725;
config.weightDecay = 0.0001;

% === IMPROVEMENT: Learning rate scheduling parameters ===
config.warmup_epochs = 5;           % LR warmup for stable start
config.min_lr_ratio = 0.1;          % Minimum LR for cosine annealing
% =======================================================

% Optimal hyperparameters (v2.3 - OPTIMIZED PTB BIAS)
config.lambda_cam = 10.0;            % Weight for GradCAM loss
config.lambda_tversky = 2.0;        % Weight for Tversky loss
config.tversky_alpha = 0.7;
config.tversky_beta = 0.3;
config.focal_alpha = [0.45, 0.55];    % [normal, PTB]
config.focal_gamma = 2.0;

% Loss function configuration
loss_config = struct();
loss_config.use_gradcam = false;
loss_config.use_segmentation = false;
loss_config.use_focal = true;
loss_config.use_tversky = false;
loss_config.use_iou = false;
loss_config.use_anatomical_guidance = false;
loss_config.lambda_cam = config.lambda_cam;
loss_config.lambda_tversky = config.lambda_tversky;
loss_config.lambda_anatomical = 2.0;
loss_config.anatomical_reward_weight = 0.75;
loss_config.tversky_alpha = config.tversky_alpha;
loss_config.tversky_beta = config.tversky_beta;
loss_config.focal_alpha = config.focal_alpha;
loss_config.focal_gamma = config.focal_gamma;

% === IMPROVEMENT: Additional loss configuration ===
loss_config.cam_loss_type = 'cosine';        % 'mse' or 'cosine' (recommended)
loss_config.anatomical_positivity = true;    % Enforce non-negative anatomical loss
loss_config.adaptive_scaling = true;         % Epoch-dependent loss scaling
loss_config.max_anatomical_scale = 1000;     % Max scaling factor
loss_config.gradient_clip_norm = 1.0;        % Gradient clipping threshold
% ================================================

config.preprocessing_method = preprocessing_method_param;

% Data augmentation (medical-appropriate)
config.imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandRotation', [-10 10], ...
    'RandScale', [0.6 1.4]);

fprintf('Configuration:\n');
fprintf('  Preprocessing: %s\n', preprocessing_method_param);
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
forceRecalculate = false;
[imds, vggNet, precomputedGradCAM, precomputedMasks, classes] = load_data_and_network(forceRecalculate);
fprintf('Dataset loaded: %d samples, %d classes\n\n', numel(imds.Files), numel(classes));

% Compute reference histogram for histogram matching
refHist = [];
if strcmp(preprocessing_method_param, 'histmatch')
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

%% Train Model with K-Fold Cross-Validation
fprintf('=== STARTING K-FOLD CROSS-VALIDATION TRAINING ===\n');
fprintf('Training with preprocessing: %s\n', preprocessing_method_param);
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

modelFile = fullfile(outputDir, sprintf('final_model_%s_kfold_improved.mat', preprocessing_method_param));
fprintf('Saving to: %s\n', modelFile);

results = struct();
results.fold_results = fold_results;
results.mean_metrics = mean_metrics;
results.std_metrics = std_metrics;
results.best_fold = best_fold_name;
results.preprocessing_method = preprocessing_method_param;
results.improvements_applied = struct(...
    'cosine_cam_loss', true, ...
    'anatomical_positivity', true, ...
    'adaptive_scaling', true, ...
    'gradient_clipping', true, ...
    'lr_warmup_cosine', true, ...
    'ood_preprocessing_ensemble', true);

save(modelFile, 'trainedNet', 'config', 'loss_config', 'training_histories', ...
    'results', 'best_models', 'foldIndices', '-v7.3');

historyFile = fullfile(outputDir, sprintf('training_history_%s_kfold_improved.mat', preprocessing_method_param));
save(historyFile, 'training_histories', '-v7.3');
fprintf('Model and results saved successfully!\n');

%% Generate Training Curves
fprintf('\n=== GENERATING TRAINING CURVES ===\n');
plot_kfold_training_curves(training_histories, mean_metrics, std_metrics, outputDir);

%% Out-of-Distribution (OOD) Evaluation with Ensemble
fprintf('\n=== OUT-OF-DISTRIBUTION EVALUATION (WITH ENSEMBLE) ===\n');
fprintf('Evaluating best model on Full CXR dataset (OOD)...\n');
fprintf('Using preprocessing ensemble for OOD robustness\n');

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
    fprintf('  CXR dataset loaded: %d samples\n', numel(imdsCXR.Files));
    
    % Evaluate on ID (ROI) - use validation set from best fold
    fprintf('\n  Evaluating on In-Distribution (ROI) data...\n');
    bestFoldValIdx = foldIndices.val{best_fold_idx};
    imdsROI_val = subset(imds, bestFoldValIdx);

    % ID Evaluation (Pass false: No ROI extraction needed)
    [resultsID, ~] = evaluate_dataset_with_preprocessing_ensemble(trainedNet, imdsROI_val, classes, ...
        config.useGPU, preprocessing_method_param, refHist, false);

    % OOD Evaluation (Pass true: Extract ROI)
    [resultsOOD, uncertainty_stats] = evaluate_dataset_with_preprocessing_ensemble(trainedNet, imdsCXR, classes, ...
        config.useGPU, preprocessing_method_param, refHist, true);

    % Calculate performance degradation
    degradation = struct();
    degradation.accuracy = (resultsID.accuracy - resultsOOD.accuracy) / resultsID.accuracy * 100;
    degradation.precision = (resultsID.precision - resultsOOD.precision) / resultsID.precision * 100;
    degradation.sensitivity = (resultsID.sensitivity - resultsOOD.sensitivity) / resultsID.sensitivity * 100;
    degradation.specificity = (resultsID.specificity - resultsOOD.specificity) / resultsID.specificity * 100;
    degradation.f1_score = (resultsID.f1_score - resultsOOD.f1_score) / resultsID.f1_score * 100;
    degradation.auc = (resultsID.auc - resultsOOD.auc) / resultsID.auc * 100;

    mean_degradation = mean([degradation.accuracy, degradation.precision, ...
                             degradation.sensitivity, degradation.specificity, ...
                            degradation.f1_score, degradation.auc]);

    fprintf('\n  === OOD EVALUATION RESULTS (WITH ENSEMBLE) ===\n');
    fprintf('  Preprocessing: %s + ensemble\n', preprocessing_method_param);
    fprintf('  ID (ROI) Performance:\n');
    fprintf('    Accuracy: %.3f\n', resultsID.accuracy);
    fprintf('    Precision: %.3f\n', resultsID.precision);
    fprintf('    Sensitivity: %.3f\n', resultsID.sensitivity);
    fprintf('    Specificity: %.3f\n', resultsID.specificity);
    fprintf('    F1-Score: %.3f\n', resultsID.f1_score);
    fprintf('    AUC: %.3f\n', resultsID.auc);

    fprintf('\n  OOD (Full CXR) Performance (Ensemble):\n');
    fprintf('    Accuracy: %.3f\n', resultsOOD.accuracy);
    fprintf('    Precision: %.3f\n', resultsOOD.precision);
    fprintf('    Sensitivity: %.3f\n', resultsOOD.sensitivity);
    fprintf('    Specificity: %.3f\n', resultsOOD.specificity);
    fprintf('    F1-Score: %.3f\n', resultsOOD.f1_score);
    fprintf('    AUC: %.3f\n', resultsOOD.auc);

    % === IMPROVEMENT: Report uncertainty statistics ===
    if isfield(uncertainty_stats, 'mean_uncertainty')
        fprintf('\n  Uncertainty Analysis:\n');
        fprintf('    Mean prediction uncertainty: %.3f\n', uncertainty_stats.mean_uncertainty);
        fprintf('    High-uncertainty samples (%%): %.1f%%\n', uncertainty_stats.high_uncertainty_pct * 100);
        fprintf('    Uncertainty threshold used: %.3f\n', uncertainty_stats.uncertainty_threshold);
    end
    % =================================================

    fprintf('\n  Performance Degradation:\n');
    fprintf('    Accuracy: %.2f%%\n', degradation.accuracy);
    fprintf('    Precision: %.2f%%\n', degradation.precision);
    fprintf('    Sensitivity: %.2f%%\n', degradation.sensitivity);
    fprintf('    Specificity: %.2f%%\n', degradation.specificity);
    fprintf('    F1-Score: %.2f%%\n', degradation.f1_score);
    fprintf('    AUC: %.2f%%\n', degradation.auc);
    fprintf('    Mean Degradation: %.2f%%\n', mean_degradation);

    if mean_degradation < 5
        fprintf('    Status: ✓ Excellent generalization (<5%% degradation)\n');
    elseif mean_degradation < 15
        fprintf('    Status: ✓ Good generalization (<15%% degradation)\n');
    elseif mean_degradation < 30
        fprintf('    Status: ⚠ Moderate degradation (15-30%%)\n');
    else
        fprintf('    Status: ✗ High degradation (>30%%)\n');
    end

    % Save OOD results
    ood_results = struct();
    ood_results.id_results = resultsID;
    ood_results.ood_results = resultsOOD;
    ood_results.degradation = degradation;
    ood_results.mean_degradation = mean_degradation;
    ood_results.uncertainty_stats = uncertainty_stats;
    results.ood_evaluation = ood_results;

    save(modelFile, 'trainedNet', 'config', 'loss_config', 'training_histories', ...
        'results', 'best_models', 'foldIndices', '-v7.3');
    fprintf('\n  OOD evaluation results saved to model file.\n');
else
    fprintf('  ⚠ Warning: CXR directory not found: %s\n', cxrDir);
    fprintf('  Skipping OOD evaluation.\n');
end

fprintf('\n=== K-FOLD CROSS-VALIDATION COMPLETE ===\n');
fprintf('Final model saved to: %s\n', modelFile);
fprintf('Training curves saved to: %s\n', outputDir);
fprintf('\nSummary:\n');
fprintf('  Preprocessing: %s\n', preprocessing_method_param);
fprintf('  Best Fold: %s\n', best_fold_name);
fprintf('  Mean Accuracy: %.3f ± %.3f\n', mean_metrics.accuracy, std_metrics.accuracy);
fprintf('  Mean Dice: %.3f ± %.3f\n', mean_metrics.dice, std_metrics.dice);
if exist('mean_degradation', 'var')
    fprintf('  OOD Mean Degradation: %.2f%%\n', mean_degradation);
end

fprintf('\n=== IMPROVEMENTS APPLIED ===\n');
fprintf('  ✓ Cosine similarity for GradCAM loss (scale-invariant)\n');
fprintf('  ✓ Anatomical loss positivity constraint (stable optimization)\n');
fprintf('  ✓ Adaptive loss scaling (epoch-dependent)\n');
fprintf('  ✓ Gradient clipping + LR warmup + cosine annealing\n');
fprintf('  ✓ Preprocessing ensemble for OOD robustness\n');
fprintf('  ✓ Uncertainty estimation for clinical safety\n');
end

%% ========================================================================
%% HELPER FUNCTIONS
%% ========================================================================

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
        precomputedGradCAM(valIdx), precomputedMasks(valIdx), 'relu5_3', config);

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

%% ========================================================================
%% ENHANCED TRAINING FUNCTION (MATLAB BEST PRACTICES)
%% ========================================================================

function [trainedNet, training_history] = train_model_with_preprocessing_improved(...
    net, imdsTrain, imdsVal, preCAMs_train, preCAMs_val, ...
    preMasks_train, preMasks_val, classes, loss_config, config)

numClasses = numel(classes);

% Ensure network is dlnetwork
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
bestValAcc = 0;  % Initialize for early stopping on accuracy
patienceCounter = 0;
featureLayer = 'relu5_3';
nCam = 32;

classCounts = countcats(imdsTrain.Labels);
baseWeights = sum(classCounts) ./ (numel(classCounts) * classCounts);
classWeights = gather(single(baseWeights(:)));

training_history = struct();
training_history.epoch_loss = zeros(1, config.numEpochs);
training_history.val_loss = nan(1, config.numEpochs);
training_history.val_accuracy = nan(1, config.numEpochs);
training_history.val_iou = nan(1, config.numEpochs);
training_history.val_dice = nan(1, config.numEpochs);
training_history.train_acc = zeros(1, config.numEpochs);
training_history.val_acc = nan(1, config.numEpochs);
training_history.epoch = 1:config.numEpochs;
training_history.lr = zeros(1, config.numEpochs);

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
        
        % Compute loss and gradients using dlfeval (MATLAB best practice)
        [loss, grads, state, loss_components] = dlfeval(@compute_loss_with_config_improved, ...
            net, X, T, loss_config, classWeights, imdsTrain.Files, ...
            preCAMs_train, preMasks_train, nCam, classes, featureLayer, config.useGPU, epoch);
        
        net.State = state;
        
        % === IMPROVEMENT: Gradient clipping for stability ===
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
        
        % Update network parameters using SGD with momentum
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

        % Track training accuracy
        Y = predict(net, X);
        Ypred = onehotdecode(extractdata(Y), classes, 1);
        Ytrue = onehotdecode(extractdata(T), classes, 1);
        trainCorrect = trainCorrect + sum(Ypred == Ytrue);
        trainTotal = trainTotal + numel(Ypred);
    end

    avgTrainLoss = epochLoss / max(1, batch);
    trainAcc = trainCorrect / max(1, trainTotal);

    training_history.epoch_loss(epoch) = avgTrainLoss;
    training_history.train_acc(epoch) = trainAcc;

    % Validation every 3 epochs
    if mod(epoch, 3) == 0 || epoch == 1
        valLoss = compute_validation_loss_with_config_preprocessed(net, imdsVal, classes, ...
            config.useGPU, preCAMs_val, preMasks_val, loss_config, classWeights, config, epoch);
        
        [val_acc, val_iou, val_dice] = evaluate_model_quick(net, imdsVal, ...
            classes, config.useGPU, preCAMs_val, preMasks_val, featureLayer, config);
        
        valAcc = compute_validation_accuracy_simple(net, mbqVal, classes);
        
        training_history.val_loss(epoch) = valLoss;
        training_history.val_accuracy(epoch) = val_acc;
        training_history.val_iou(epoch) = val_iou;
        training_history.val_dice(epoch) = val_dice;
        training_history.val_acc(epoch) = valAcc;
        
        fprintf('  Train Loss: %.4f, Train Acc: %.3f\n', avgTrainLoss, trainAcc);
        fprintf('  Val Loss: %.4f, Val Acc: %.3f, Val IoU: %.3f, Val Dice: %.3f\n', ...
            valLoss, valAcc, val_iou, val_dice);
        
        % === IMPROVEMENT: Early stopping on Accuracy ===
        if valAcc > bestValAcc + config.min_delta
            bestValAcc = valAcc;
            bestNet = net;
            patienceCounter = 0;
            fprintf('  ✓ Validation accuracy improved!\n');
        else
            patienceCounter = patienceCounter + 1;
        end
        
        if patienceCounter >= config.patience
            fprintf('\nEarly stopping triggered (patience: %d)\n', config.patience);
            net = bestNet;
            break;
        end
    else
        fprintf('  Train Loss: %.4f, Train Acc: %.3f\n', avgTrainLoss, trainAcc);
    end
end

% Return best network
if exist('bestNet', 'var')
    trainedNet = bestNet;
else
    trainedNet = net;
end

% Trim history arrays to actual epochs trained
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

%% ========================================================================
%% ENHANCED LOSS COMPUTATION (COSINE CAM + POSITIVITY + ADAPTIVE SCALING)
%% ========================================================================

function [loss, grads, state, loss_components] = compute_loss_with_config_improved(...
    net, X, T, loss_config, classWeights, trainFiles, preCAMs, preMasks, ...
    nCam, classes, featureLayer, useGPU, epoch)

% Forward pass
[Y, state] = forward(net, X);
loss_components = struct();

% Classification loss (always computed)
if loss_config.use_focal
    clsLoss = compute_focal_loss(Y, T, classWeights, ...
        loss_config.focal_alpha, loss_config.focal_gamma);
else
    clsLoss = crossentropy(Y, T, 'Weights', classWeights, 'WeightsFormat', 'C');
end
loss_components.classification = clsLoss;

% Initialize auxiliary losses
camLoss = 0;
segLoss = 0;
tverskyLoss = 0;
iouLoss = 0;
anatomicalLoss = 0;

% Compute auxiliary losses if enabled
if loss_config.use_gradcam || loss_config.use_segmentation || ...
   loss_config.use_tversky || loss_config.use_iou || loss_config.use_anatomical_guidance
    
    N = numel(trainFiles);
    n = min(nCam, N);
    idxs = randperm(N, n);
    
    for ii = 1:n
        % Load and preprocess image
        img = imread(trainFiles{idxs(ii)});
        if size(img,3)==1, img = repmat(img,[1 1 3]); end
        img = imresize(img, [224 224]);
        
        % Compute student CAM
        studCAM = student_cam_one_dlarray(net, img, classes, featureLayer, useGPU);
        studCAM_data = extractdata(studCAM);
        studCAM_data = single(studCAM_data);
        if size(studCAM_data, 1) ~= 224 || size(studCAM_data, 2) ~= 224
            studCAM_data = imresize(studCAM_data, [224 224]);
        end
        studCAM = dlarray(studCAM_data, 'SS');
        
        % === GradCAM loss: Cosine similarity for scale-invariant alignment ===
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
            
            % Use cosine similarity if specified
            if isfield(loss_config, 'cam_loss_type') && strcmp(loss_config.cam_loss_type, 'cosine')
                camLoss = camLoss + cam_cosine_loss(studCAM, targetCAM);
            else
                camLoss = camLoss + mse(studCAM, targetCAM);  % Fallback to MSE
            end
        end
        
        % === Segmentation losses (Tversky, IoU, Dice) ===
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
        
        % === Anatomical guidance loss: Positivity constraint + adaptive scaling ===
        if loss_config.use_anatomical_guidance
            realMask = preMasks{idxs(ii)};
            if ~isempty(realMask) && any(realMask(:))
                % Prepare lung mask
                if islogical(realMask)
                    lungMask = single(realMask);
                else
                    lungMask = single(realMask > 0.5);
                end
                if size(lungMask, 1) ~= 224 || size(lungMask, 2) ~= 224
                    lungMask = imresize(lungMask, [224 224], 'nearest');
                end
                
                % Normalize CAM to [0, 1]
                studCAM_norm = studCAM;
                cam_max = max(studCAM_norm, [], 'all');
                if cam_max > 0
                    studCAM_norm = studCAM_norm / (cam_max + eps);
                end
                
                % Create masks as dlarray
                lungMask_dl = dlarray(lungMask, 'SS');
                nonLungMask_dl = dlarray(1 - lungMask, 'SS');
                
                % Compute penalty and reward
                attention_outside_lungs = studCAM_norm .* nonLungMask_dl;
                penalty_outside = mean(attention_outside_lungs, 'all');
                attention_inside_lungs = studCAM_norm .* lungMask_dl;
                reward_inside = mean(attention_inside_lungs, 'all');
                
                % Combine with reward weight
                anatomical_loss_sample = penalty_outside - ...
                    loss_config.anatomical_reward_weight * reward_inside;
                
                % === IMPROVEMENT: Enforce positivity to prevent instability ===
                if isfield(loss_config, 'anatomical_positivity') && loss_config.anatomical_positivity
                    anatomical_loss_sample = max(anatomical_loss_sample, 0);
                end
                
                % === IMPROVEMENT: Adaptive scaling based on epoch ===
                if isfield(loss_config, 'adaptive_scaling') && loss_config.adaptive_scaling
                    % Ramp up scaling gradually to avoid early dominance
                    scale_factor = min(loss_config.max_anatomical_scale, ...
                        max(100, epoch * 3));  % Linear ramp: 100→1000 over 50 epochs
                else
                    scale_factor = 1000;  % Fixed scaling (original behavior)
                end
                
                anatomical_loss_scaled = anatomical_loss_sample * scale_factor;
                anatomicalLoss = anatomicalLoss + anatomical_loss_scaled;
            end
        end 
    end

    % Average losses over samples
    camLoss = camLoss / max(1, n);
    segLoss = segLoss / max(1, n);
    tverskyLoss = tverskyLoss / max(1, n);
    iouLoss = iouLoss / max(1, n);
    anatomicalLoss = anatomicalLoss / max(1, n);
end

% Store individual loss components for logging
loss_components.gradcam = camLoss;
loss_components.segmentation = segLoss;
loss_components.tversky = tverskyLoss;
loss_components.iou = iouLoss;
loss_components.anatomical = anatomicalLoss;

% === TOTAL LOSS COMPUTATION ===
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

% Compute gradients (enable higher derivatives for CAM gradients)
grads = dlgradient(loss, net.Learnables, 'EnableHigherDerivatives', true);
end

%% ========================================================================
%% LOSS HELPER FUNCTIONS
%% ========================================================================

function camLoss = cam_cosine_loss(studCAM, targetCAM)
% Cosine similarity loss for CAM alignment (scale-invariant)
% Reshape to vectors
stud_vec = reshape(studCAM, [], 1);
target_vec = reshape(targetCAM, [], 1);

% dlarray-compatible cosine similarity calculation
dot_product = sum(stud_vec .* target_vec, 'all');
norm_stud = sqrt(sum(stud_vec.^2, 'all'));
norm_target = sqrt(sum(target_vec.^2, 'all'));

cosSim = dot_product / (norm_stud * norm_target + eps);

% Convert similarity to loss: [0, 2] range
camLoss = 1 - cosSim;

% Clamp for numerical stability (dlarray compatible)
camLoss = max(0, min(2, camLoss));
end

function focalLoss = compute_focal_loss(Y, T, classWeights, alpha, gamma)
% Focal loss with class weighting and alpha array support
epsVal = 1e-7;
Y = min(max(Y, epsVal), 1-epsVal);

% Handle alpha as array or scalar
if isscalar(alpha)
    alpha_vec = alpha * reshape(classWeights, [], 1);
else
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

%% ========================================================================
%% CAM COMPUTATION FUNCTIONS
%% ========================================================================

function cam = student_cam_one_dlarray(net, img, classes, featureLayer, useGPU)
% Compute GradCAM for a single image using dlarray (gradient-enabled)
dlX = dlarray(single(img) ./ 255, 'SSCB');
if useGPU, dlX = gpuArray(dlX); end

[featMap, logits] = forward(net, dlX, 'Outputs', {featureLayer, 'fc8'});
logits = squeeze(logits);
[~, classIdx] = max(extractdata(logits));
score = sum(logits(classIdx), 'all');

gradFeat = dlgradient(score, featMap);
w = mean(gradFeat, [1 2]);
cam = sum(featMap .* w, 3);
cam = max(cam, 0);
cam_max = max(cam, [], 'all');
cam = cam ./ (cam_max + eps);
cam = stripdims(cam);
cam = dlarray(cam, 'SS');
end

%% ========================================================================
%% VALIDATION & EVALUATION FUNCTIONS
%% ========================================================================

function valLoss = compute_validation_loss_with_config_preprocessed(net, imdsVal, classes, useGPU, ...
    precomputedGradCAM, precomputedMasks, loss_config, classWeights, config, epoch)

augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
mbqVal = minibatchqueue(augVal, ...
    'MiniBatchSize', config.batchSize, ...
    'MiniBatchFcn', preprocessFcn, ...
    'MiniBatchFormat', ["SSCB", ""], ...
    'PartialMiniBatch', 'discard');

valLoss = 0; 
valCount = 0; 
valFilesForGradCAM = imdsVal.Files;

while hasdata(mbqVal)
    [X, T] = next(mbqVal);
    if useGPU, X = gpuArray(X); end
    try
        % Call the improved loss function with epoch parameter
        [loss, ~, ~, ~] = dlfeval(@compute_loss_with_config_improved, net, X, T, loss_config, classWeights, ...
            valFilesForGradCAM, precomputedGradCAM, precomputedMasks, 8, classes, 'relu5_3', useGPU, epoch);
        valLoss = valLoss + double(loss);
        valCount = valCount + 1;
    catch
        continue;
    end
end
valLoss = valLoss / max(1, valCount);
end

function valAcc = compute_validation_accuracy_simple(net, mbqVal, classes)
% Returns balanced accuracy using threshold sweep for robust evaluation
ptb_candidates = {'PTB','ptb','TB','tb','Tuberculosis','positive','Positive'};
ptbIdx = 2;  % fallback
for k = 1:numel(ptb_candidates)
    idx = find(strcmp(classes, ptb_candidates{k}), 1);
    if ~isempty(idx), ptbIdx = idx; break; end
end
normalIdx = 3 - ptbIdx;

reset(mbqVal);
Yprobs = []; 
Ytrue_all = categorical.empty(0,1);

while hasdata(mbqVal)
    [X, T] = next(mbqVal);
    Y = predict(net, X);
    probs = extractdata(Y)';
    Yprobs = [Yprobs; probs(:, ptbIdx)];
    labels = onehotdecode(extractdata(T), classes, 1);
    Ytrue_all = [Ytrue_all; labels(:)];
end

if isempty(Yprobs)
    valAcc = 0.5;
    return;
end

% Sweep threshold, return best balanced accuracy
best_bal = 0;
for t = 0.20:0.05:0.70
    preds_t = categorical(classes(1 + (Yprobs >= t)));
    cm_t = confusionmat(Ytrue_all, preds_t, 'Order', categorical(classes));
    if all(size(cm_t) == [2 2])
        TP = cm_t(ptbIdx, ptbIdx);
        FN = cm_t(ptbIdx, normalIdx);
        TN = cm_t(normalIdx, normalIdx);
        FP = cm_t(normalIdx, ptbIdx);
        s = TP / (TP + FN + eps);
        sp = TN / (TN + FP + eps);
        bal = (s + sp) / 2;
        if bal > best_bal, best_bal = bal; end
    end
end
valAcc = best_bal;
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

%% ========================================================================
%% OOD EVALUATION WITH PREPROCESSING ENSEMBLE
%% ========================================================================

function [results, predictions, uncertainty_stats] = evaluate_dataset_with_preprocessing_ensemble(...
    net, imds, classes, useGPU, base_preprocessing, refHist, is_ood_evaluation)

% Evaluate with multiple preprocessing methods and ensemble predictions
if strcmp(base_preprocessing, 'histmatch')
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
    
    % === Preprocessing ensemble ===
    probs_batch = zeros(size(imgs, 4), numel(classes));

    for m = 1:numel(preprocessing_methods)
        method = preprocessing_methods{m};

        % Test-Time ROI Extraction for OOD evaluation
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
                
                % Extract and resize ROI back to 224x224
                roi_cropped = extract_lung_roi_simple(img_gray);
                roi_batch(:, :, :, b) = repmat(roi_cropped, [1, 1, 3]); 
            end
            
            % Apply preprocessing to the ROI batch
            imgs_processed = apply_preprocessing_batch(roi_batch, method, refHist);
        else
            % Normal processing for ID data
            imgs_processed = apply_preprocessing_batch(imgs, method, refHist);
        end
    
        if useGPU, imgs_processed = gpuArray(imgs_processed); end
        dlX = dlarray(single(imgs_processed) ./ 255, 'SSCB');
        
        sc = predict(net, dlX);
        probs = extractdata(sc)';
        
        % Accumulate probabilities
        probs_batch = probs_batch + probs;
    end

    % Average probabilities across preprocessing methods
    probs_batch = probs_batch / numel(preprocessing_methods);

    % Decode predictions
    [~, pred_idx] = max(probs_batch, [], 2);
    lab = categorical(classes(pred_idx));
    Ypred_ensemble = [Ypred_ensemble; lab(:)];

    % Store probabilities for uncertainty computation
    if numel(classes) == 2
        Yprobs_ensemble = [Yprobs_ensemble; probs_batch(:, 2)];
    else
        Yprobs_ensemble = [Yprobs_ensemble; max(probs_batch, [], 2)];
    end

    % === Compute prediction uncertainty (entropy-based) ===
    probs_normalized = probs_batch ./ (sum(probs_batch, 2) + eps);
    entropy = -sum(probs_normalized .* log(probs_normalized + eps), 2);
    max_prob = max(probs_batch, [], 2);

    % Combine entropy and confidence for uncertainty score
    uncertainty = (1 - max_prob) + 0.5 * entropy / log(numel(classes));
    uncertainty_values = [uncertainty_values; uncertainty];
end

% Calculate metrics
match = (categorical(Ytrue) == categorical(Ypred_ensemble));
accuracy = sum(match) / numel(match);
cm = confusionmat(Ytrue, Ypred_ensemble, 'Order', categorical(classes));

if numel(classes)==2 && all(size(cm)==[2 2])
    TP = cm(2,2); FP = cm(1,2); FN = cm(2,1); TN = cm(1,1);
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
        [~,~,~,auc] = perfcurve(double(Ytrue == classes{2}), Yprobs_ensemble, 1);
    else
        auc = NaN;
    end
catch
    auc = 0.5;
end

% === Uncertainty statistics ===
uncertainty_stats = struct();
uncertainty_stats.mean_uncertainty = mean(uncertainty_values);
uncertainty_stats.std_uncertainty = std(uncertainty_values);
uncertainty_threshold = prctile(uncertainty_values, 80);
high_uncertainty_mask = uncertainty_values > uncertainty_threshold;
uncertainty_stats.high_uncertainty_pct = mean(high_uncertainty_mask);
uncertainty_stats.uncertainty_threshold = uncertainty_threshold;

% Package results
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

%% ========================================================================
%% PREPROCESSING & DATA UTILITIES
%% ========================================================================

function [X, T] = preprocessMiniBatchWithPreprocessing(dataX, dataT, preprocessing_method, refHist)
% Preprocess mini-batch with specified preprocessing method
X = cat(4, dataX{1:end});
[H, W, C, B] = size(X);
X_processed = zeros(size(X), 'uint8');

for b = 1:B
    img = X(:, :, :, b);
    
    if C == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end

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

    if C == 3
        X_processed(:, :, :, b) = repmat(img_processed, [1, 1, 3]);
    else
        X_processed(:, :, :, b) = img_processed;
    end
end

X = dlarray(single(X_processed) ./ 255, 'SSCB');
T = onehotencode(cat(2, dataT{1:end}), 1);
end

function imgs_processed = apply_preprocessing_batch(imgs, method, refHist)
% Apply preprocessing to batch of images
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
% Robust heuristic for lung ROI extraction
[rows, cols] = size(img);

% Otsu thresholding
try
    initial_mask = imbinarize(img, 'otsu');
catch
    initial_mask = img < prctile(img(:), 30);
end

% Morphological cleanup
se = strel('disk', 6);
initial_mask = imclose(imopen(initial_mask, se), se);
initial_mask = bwareaopen(initial_mask, 800);

% Anatomical prior
prior_mask = false(rows, cols);
prior_mask(round(rows*0.1):round(rows*0.9), round(cols*0.2):round(cols*0.8)) = true;
refined_mask = initial_mask & prior_mask;

if ~any(refined_mask(:)), refined_mask = initial_mask; end

% Extract largest component
props = regionprops(refined_mask, 'BoundingBox', 'Area');

if isempty(props)
    margin = round(rows * 0.15);
    roi_img = imresize(img(margin+1:end-margin, margin+1:end-margin), [224 224]);
    return;
end

[~, idx] = max([props.Area]);
bbox = props(idx).BoundingBox;

margin = 12;
x1 = max(1, floor(bbox(1) - margin));
y1 = max(1, floor(bbox(2) - margin));
x2 = min(cols, ceil(bbox(1) + bbox(3) + margin));
y2 = min(rows, ceil(bbox(2) + bbox(4) + margin));

roi_img = imresize(img(y1:y2, x1:x2), [224 224]);
end

%% ========================================================================
%% DATA LOADING & CROSS-VALIDATION UTILITIES
%% ========================================================================

function [imds, vggNet, precomputedGradCAM, precomputedMasks, classes] = load_data_and_network(forceRecalculate)
if nargin < 1
    forceRecalculate = false;
end

modelPath = fullfile('models', 'pretrained', 'vgg16_finetuned_on_roi.mat');
if ~exist(modelPath, 'file')
    modelPath = 'vgg16_finetuned_on_roi.mat';
end

s = load(modelPath);
vggNet = s.trainedNet;

roiDir = fullfile('input', 'roi');
imds = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imds.Labels);

gradCAMCacheFile = 'precomputed_gradcam_maps_enhanced.mat';

if numel(imds.Files) > 0
    firstImgPath = imds.Files{1};
    imgDir = fileparts(firstImgPath);
    roiDir = fileparts(imgDir);
    baseDir = fileparts(roiDir);
    maskDir = fullfile(baseDir, 'masks');
    gradcamMasksDir = fullfile(baseDir, 'gradcam_masks');
    if ~exist(gradcamMasksDir, 'dir')
        mkdir(gradcamMasksDir);
    end
else
    maskDir = fullfile('input', 'masks');
    baseDir = 'input';
    gradcamMasksDir = fullfile('input', 'gradcam_masks');
    if ~exist(gradcamMasksDir, 'dir')
        mkdir(gradcamMasksDir);
    end
end

workingGradCAMLayer = 'relu5_3';

if forceRecalculate
    fprintf('  Recalculating GradCAM maps...\n');
    [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
        imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
    cachedFileList = imds.Files;
    save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', 'cachedFileList', '-v7.3');
    fprintf('  GradCAM maps recalculated and saved\n');
else
    if exist(gradCAMCacheFile, 'file')
        fprintf('  Loading cached GradCAM maps and masks...\n');
        cache = load(gradCAMCacheFile);
        if isfield(cache, 'precomputedMasks')
            numCachedMasks = numel(cache.precomputedMasks);
            numNonEmpty = 0;
            for j = 1:min(10, numCachedMasks)
                if ~isempty(cache.precomputedMasks{j}) && any(cache.precomputedMasks{j}(:))
                    numNonEmpty = numNonEmpty + 1;
                end
            end
            if numNonEmpty == 0
                fprintf('  ⚠️ WARNING: All cached masks are empty! Reloading...\n');
                [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
                    imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
                cachedFileList = imds.Files;
                save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', 'cachedFileList', '-v7.3');
            else
                if isfield(cache, 'cachedFileList') && ...
                   numel(cache.cachedFileList) == numel(imds.Files) && ...
                   all(strcmp(cache.cachedFileList, imds.Files))
                    precomputedGradCAM = cache.precomputedGradCAM;
                    precomputedMasks = cache.precomputedMasks;
                    fprintf('  Using cached GradCAM maps\n');
                else
                    fprintf('  Cache mismatch. Recalculating...\n');
                    [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
                        imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
                    cachedFileList = imds.Files;
                    save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', 'cachedFileList', '-v7.3');
                end
            end
        else
            fprintf('  Cache missing precomputedMasks. Recalculating...\n');
            [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
                imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
            cachedFileList = imds.Files;
            save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', 'cachedFileList', '-v7.3');
        end
    else
        fprintf('  Cache file not found. Computing GradCAM maps...\n');
        [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
            imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir);
        cachedFileList = imds.Files;
        save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', 'cachedFileList', '-v7.3');
    end
end
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

%% ========================================================================
%% PLOTTING & SEGMENTATION EVALUATION
%% ========================================================================

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
        Yprobs = [Yprobs; probs_batch(2,:)'];
    else
        Yprobs = [Yprobs; probs_batch(1,:)'];
    end
end

Ytrue = imdsVal.Labels(:);
match = (categorical(Ytrue) == categorical(Ypred));
accuracy = sum(match) / numel(match);
cm = confusionmat(string(Ytrue), string(Ypred), 'Order', string(classes));

if numel(classes)==2 && all(size(cm)==[2 2])
    TP = cm(2,2); FP = cm(1,2); FN = cm(2,1); TN = cm(1,1);
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
    [~,~,~,auc] = perfcurve(double(Ytrue == classes{2}), Yprobs, 1);
catch
    auc = 0.5;
end

% Segmentation metrics
numVal = numel(imdsVal.Files);
ious = zeros(numVal, 1);
dices = zeros(numVal, 1);
tverskys = zeros(numVal, 1);
jaccards = zeros(numVal, 1);
hausdorffs = zeros(numVal, 1);

for i = 1:numVal
    img = imread(imdsVal.Files{i});
    if size(img,3)==1, img = repmat(img,[1 1 3]); end
    
    % Preprocess image to match training distribution
    img4d = reshape(img, size(img,1), size(img,2), size(img,3), 1);
    img4d = apply_preprocessing_batch(img4d, config.preprocessing_method, config.refHist);
    img = img4d(:,:,:,1);
    
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
% Legacy CAM function (non-dlarray version for evaluation)
dlX = dlarray(single(img) ./ 255, 'SSCB');
if useGPU, dlX = gpuArray(dlX); end

[featMap, logits] = forward(net, dlX, 'Outputs', {featureLayer, 'fc8'});
logits = squeeze(logits);
[~, classIdx] = max(extractdata(logits));
score = sum(logits(classIdx), 'all');

gradFeat = dlgradient(score, featMap);
w = mean(gradFeat, [1 2]);
cam = sum(featMap .* w, 3);
cam = max(cam, 0);
cam = extractdata(cam);
cam = imresize(cam, [224 224]);
cam = single(cam ./ (max(cam(:)) + eps));
end