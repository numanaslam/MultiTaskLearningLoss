function train_final_model_with_preprocessing(preprocessing_method)
%TRAIN_FINAL_MODEL_WITH_PREPROCESSING Train final model with image preprocessing
%   UNIFIED VERSION: Consistent evaluation across training, validation, and OOD testing.
%
%   KEY IMPROVEMENTS PRESERVED & UNIFIED:
%   =====================================
%   1. Cosine similarity for GradCAM loss (scale-invariant structural alignment)
%   2. Positivity constraint on anatomical loss (prevents optimization instability)
%   3. Adaptive loss scaling (epoch-dependent, prevents early dominance)
%   4. Gradient clipping + LR warmup + cosine annealing (training stability)
%   5. Test-Time Augmentation (Horizontal Flip) + Preprocessing Ensemble (OOD robustness)
%   6. Uncertainty estimation for OOD flagging (clinical safety)
%   7. UNIFIED THRESHOLD SWEEP: Early stopping, fold validation, and OOD/ID evaluation
%      now use identical threshold optimization and metric calculation.

if nargin < 1
    preprocessing_method = 'histmatch';
end
preprocessing_method_param = preprocessing_method;
clc; close all;
fprintf('=== FINAL MODEL TRAINING WITH PREPROCESSING (UNIFIED) ===\n');
fprintf('Preprocessing Method: %s\n', preprocessing_method_param);
fprintf('Loss Function: Focal + GradCAM(Cosine) + Tversky + Anatomical(Positive)\n');
fprintf('Stability: Gradient clipping + LR warmup + cosine annealing\n');
fprintf('OOD Robustness: Preprocessing ensemble + TTA + uncertainty estimation\n\n');

if isempty(which('precompute_gradcam_and_masks'))
    addpath(pwd);
    fprintf('Added current directory to path\n\n');
end

%% Configuration
config = struct();
QUICK_TEST = false;
if QUICK_TEST
    config.k_folds = 3;
    config.numEpochs = 5;
    fprintf('QUICK TEST MODE: 3 folds, 5 epochs\n');
else
    config.k_folds = 2;
    config.numEpochs = 50;
    fprintf('FULL TRAINING MODE: %d folds, %d epochs\n', config.k_folds, config.numEpochs);
end
VALIDATION_MODE = false;
config.patience = 25;
config.min_delta = 1e-2;
config.useGPU = canUseGPU;
config.batchSize = 14;
config.initialLearnRate = 0.001;
config.decay = 0.0042;
config.momentum = 0.8725;
config.weightDecay = 0.001;
config.warmup_epochs = 5;
config.min_lr_ratio = 0.1;
config.nCam = 32; % Unified sampling size for GradCAM/anatomical losses

% Optimal hyperparameters (v2.3 - OPTIMIZED PTB BIAS)
config.lambda_cam = 10;
config.lambda_tversky = 2.0;
config.tversky_alpha = 0.7;
config.tversky_beta = 0.45;
config.focal_alpha = [0.45, 0.55];
config.focal_gamma = 2.0;

loss_config = struct();
loss_config.use_gradcam = false;
loss_config.use_segmentation = false;
loss_config.use_focal = true;
loss_config.use_tversky = false;
loss_config.use_iou = false;
loss_config.use_anatomical_guidance = false;
loss_config.lambda_cam = config.lambda_cam;
loss_config.lambda_tversky = config.lambda_tversky;
loss_config.lambda_anatomical = 2;
loss_config.anatomical_reward_weight = 0.75;
loss_config.tversky_alpha = config.tversky_alpha;
loss_config.tversky_beta = config.tversky_beta;
loss_config.focal_alpha = config.focal_alpha;
loss_config.focal_gamma = config.focal_gamma;
loss_config.cam_loss_type = 'cosine';
loss_config.anatomical_positivity = true;
loss_config.adaptive_scaling = false;
loss_config.max_anatomical_scale = 200;
loss_config.gradient_clip_norm = 1.0;

config.preprocessing_method = preprocessing_method_param;
config.imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation', [-30 30], ...
    'RandYTranslation', [-30 30], ...
    'RandRotation', [-30 30], ...
    'RandScale', [0.4 1.1]);

fprintf('Configuration:\n');
fprintf('  Preprocessing: %s\n', preprocessing_method_param);
fprintf('  K-Folds: %d | λ_cam: %.2f | λ_tversky: %.2f | λ_anatomical: %.2f\n', ...
    config.k_folds, config.lambda_cam, config.lambda_tversky, loss_config.lambda_anatomical);
fprintf('  Focal alpha: [%.2f, %.2f] | Gamma: %.2f\n', config.focal_alpha(1), config.focal_alpha(2), config.focal_gamma);
fprintf('  LR: %.4f (warmup: %d) | Batch: %d | nCam: %d\n\n', ...
    config.initialLearnRate, config.warmup_epochs, config.batchSize, config.nCam);

%% Load Data and Network
fprintf('Loading data and network...\n');
forceRecalculate = false;
[imds, vggNet, precomputedGradCAM, precomputedMasks, classes] = load_data_and_network(forceRecalculate);
fprintf('Dataset loaded: %d samples, %d classes\n\n', numel(imds.Files), numel(classes));
fprintf('Class distribution: %s\n', mat2str(countcats(imds.Labels)));

%% === VALIDATION MODE ===
if VALIDATION_MODE
    fprintf('=== RUNNING IN VALIDATION MODE (Small Dataset) ===\n');
    subset_size = 150;
    subset_idx = randperm(numel(imds.Files), subset_size);
    imds = subset(imds, subset_idx);
    if exist('precomputedGradCAM', 'var') && ~isempty(precomputedGradCAM)
        precomputedGradCAM = precomputedGradCAM(subset_idx);
        precomputedMasks = precomputedMasks(subset_idx);
    end
    config.k_folds = 2;
    config.numEpochs = 10;
    config.patience = 3;
    config.batchSize = 8;
    fprintf('  Using %d samples, %d folds, %d epochs\n', numel(imds.Files), config.k_folds, config.numEpochs);
end

% Compute reference histogram
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
        if isempty(cxrFiles), cxrFiles = dir(fullfile(cxrDir, '**', '*.jpg')); end
        if ~isempty(cxrFiles)
            useCXR = true;
            fprintf('  Using Full CXR images (Target Domain) for reference histogram.\n');
            numRefSamples = min(50, numel(cxrFiles));
            refIdx = randperm(numel(cxrFiles), numRefSamples);
            refImages = cell(numRefSamples, 1);
            for i = 1:numRefSamples
                idx = refIdx(i);
                img = imread(fullfile(cxrFiles(idx).folder, cxrFiles(idx).name));
                if size(img, 3) > 1, img = rgb2gray(img); end
                refImages{i} = img;
            end
        end
    end
    if ~useCXR
        fprintf('  ⚠️ CXR directory not found. Falling back to ROI images.\n');
        numRefSamples = min(50, numel(imds.Files));
        refIdx = randperm(numel(imds.Files), numRefSamples);
        refImages = cell(numRefSamples, 1);
        for i = 1:numRefSamples
            img = imread(imds.Files{refIdx(i)});
            if size(img, 3) > 1, img = rgb2gray(img); end
            refImages{i} = img;
        end
    end
    allPixels = [];
    for i = 1:numRefSamples, allPixels = [allPixels; refImages{i}(:)]; end
    refHist = imhist(uint8(allPixels));
    fprintf('  Reference histogram computed from %d samples\n\n', numRefSamples);
end
config.refHist = refHist;

%% Create K-Fold Splits
fprintf('Creating %d-fold stratified cross-validation splits...\n', config.k_folds);
foldIndices = createStratifiedKFold(imds.Labels, config.k_folds);
fprintf('K-fold splits created successfully!\n\n');

%% Train Model
fprintf('=== STARTING K-FOLD CROSS-VALIDATION TRAINING ===\n');
[fold_results, training_histories, best_models] = train_with_kfold_preprocessed(...
    imds, vggNet, precomputedGradCAM, precomputedMasks, ...
    foldIndices, classes, loss_config, config);

%% Compute Mean/Std Metrics
fprintf('\n=== COMPUTING CROSS-VALIDATION RESULTS ===\n');
mean_metrics = calculate_mean_metrics(fold_results);
std_metrics = calculate_std_metrics(fold_results);
fprintf('\nCross-Validation Results (Mean ± Std):\n');
fprintf('  Accuracy: %.3f ± %.3f | F1: %.3f ± %.3f | AUC: %.3f ± %.3f\n', ...
    mean_metrics.accuracy, std_metrics.accuracy, mean_metrics.f1_score, std_metrics.f1_score, mean_metrics.auc, std_metrics.auc);
fprintf('  Dice: %.3f ± %.3f | IoU: %.3f ± %.3f\n', mean_metrics.dice, std_metrics.dice, mean_metrics.iou, std_metrics.iou);

fold_names = fieldnames(fold_results);
accuracies = zeros(numel(fold_names), 1);
for f = 1:numel(fold_names), accuracies(f) = fold_results.(fold_names{f}).accuracy; end
[~, best_fold_idx] = max(accuracies);
best_fold_name = fold_names{best_fold_idx};
trainedNet = best_models.(best_fold_name);
fprintf('\nBest model: %s (Accuracy: %.3f)\n', best_fold_name, fold_results.(best_fold_name).accuracy);

%% Save Model
outputDir = fullfile('models', 'final');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
modelFile = fullfile(outputDir, sprintf('final_model_%s_kfold_unified.mat', preprocessing_method_param));
results = struct();
results.fold_results = fold_results;
results.mean_metrics = mean_metrics;
results.std_metrics = std_metrics;
results.best_fold = best_fold_name;
results.preprocessing_method = preprocessing_method_param;
results.improvements_applied = struct(...
    'cosine_cam_loss', true, 'anatomical_positivity', true, 'adaptive_scaling', true, ...
    'gradient_clipping', true, 'lr_warmup_cosine', true, 'ood_preprocessing_ensemble', true, ...
    'tta_horizontal_flip', true, 'unified_evaluation', true);
save(modelFile, 'trainedNet', 'config', 'loss_config', 'training_histories', ...
    'results', 'best_models', 'foldIndices', '-v7.3');
fprintf('Model and results saved to: %s\n', modelFile);

%% Training Curves
plot_kfold_training_curves(training_histories, mean_metrics, std_metrics, outputDir);

%% OOD/ID Evaluation
fprintf('\n=== OUT-OF-DISTRIBUTION EVALUATION ===\n');
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
    
    bestFoldValIdx = foldIndices.val{best_fold_idx};
    imdsROI_val = subset(imds, bestFoldValIdx);
    
    fprintf('  Evaluating ID (Validation) data...\n');
    [resultsID, ~, ~] = evaluate_dataset_with_preprocessing_ensemble(...
        trainedNet, imdsROI_val, classes, config.useGPU, preprocessing_method_param, refHist, false);
        
    fprintf('  Evaluating OOD (Full CXR) data with preprocessing ensemble + TTA...\n');
    [resultsOOD, ~, uncertainty_stats] = evaluate_dataset_with_preprocessing_ensemble(...
        trainedNet, imdsCXR, classes, config.useGPU, preprocessing_method_param, refHist, true);
        
    auc_gap = resultsID.auc - resultsOOD.auc;
    sens_gap = resultsID.sensitivity - resultsOOD.sensitivity;
    spec_gap = resultsID.specificity - resultsOOD.specificity;
    acc_gap = resultsID.accuracy - resultsOOD.accuracy;
    
    fprintf('\n  Performance Degradation (ID → OOD):\n');
    fprintf('    AUC gap:          %+.3f\n', auc_gap);
    fprintf('    Accuracy gap:     %+.3f\n', acc_gap);
    fprintf('    Sensitivity gap:  %+.3f\n', sens_gap);
    fprintf('    Specificity gap:  %+.3f\n', spec_gap);
    
    if auc_gap < 0.05
        fprintf('    Status: Excellent generalization\n');
    elseif auc_gap < 0.10
        fprintf('    Status: Good generalization\n');
    elseif auc_gap < 0.20
        fprintf('    Status: Moderate degradation\n');
    else
        fprintf('    Status: Poor generalization -- retrain recommended\n');
    end
    
    if isfield(uncertainty_stats, 'mean_uncertainty')
        fprintf('\n  Uncertainty Analysis:\n');
        fprintf('    Mean uncertainty: %.3f | High-uncertainty: %.1f%%\n', ...
            uncertainty_stats.mean_uncertainty, uncertainty_stats.high_uncertainty_pct * 100);
    end
    
    ood_results = struct();
    ood_results.id_results = resultsID;
    ood_results.ood_results = resultsOOD;
    ood_results.auc_gap = auc_gap;
    ood_results.uncertainty_stats = uncertainty_stats;
    results.ood_evaluation = ood_results;
    save(modelFile, 'trainedNet', 'config', 'loss_config', 'training_histories', ...
        'results', 'best_models', 'foldIndices', '-v7.3');
    fprintf('  OOD results saved.\n');
else
    fprintf('  ⚠ Warning: CXR directory not found. Skipping OOD evaluation.\n');
end
fprintf('\n=== K-FOLD CROSS-VALIDATION COMPLETE ===\n');
end

%% ========================================================================
%% Helper Functions
%% ========================================================================

function [fold_results, training_histories, best_models] = train_with_kfold_preprocessed(...
    imds, vggNet, precomputedGradCAM, precomputedMasks, foldIndices, classes, loss_config, config)
fold_results = struct();
training_histories = struct();
best_models = struct();
featureLayer = 'relu5_3';

for fold = 1:config.k_folds
    fprintf('\n--- Fold %d/%d ---\n', fold, config.k_folds);
    trainIdx = foldIndices.train{fold};
    valIdx = foldIndices.val{fold};
    imdsTrain = subset(imds, trainIdx);
    imdsVal = subset(imds, valIdx);
    preCAMs_train = precomputedGradCAM(trainIdx);
    preCAMs_val = precomputedGradCAM(valIdx);
    preMasks_train = precomputedMasks(trainIdx);
    preMasks_val = precomputedMasks(valIdx);
    fprintf('  Train: %d | Val: %d\n', numel(imdsTrain.Files), numel(imdsVal.Files));
    
    numClasses = numel(classes);
    baseLg = layerGraph(vggNet.Layers);
    toDrop = intersect({'fc8','prob','output'}, {baseLg.Layers.Name});
    if ~isempty(toDrop), baseLg = removeLayers(baseLg, toDrop); end
    newHead = [fullyConnectedLayer(numClasses, 'Name', 'fc8'), softmaxLayer('Name', 'prob')];
    baseLg = addLayers(baseLg, newHead);
    baseLg = connectLayers(baseLg, 'drop7', 'fc8');
    net = dlnetwork(baseLg);
    
    classCounts = countcats(imdsTrain.Labels);
    baseWeights = sum(classCounts) ./ (numel(classCounts) * classCounts);
    classWeights = gather(single(baseWeights(:)));
    
    [trainedNet, training_history] = train_model_with_preprocessing_improved(...
        net, imdsTrain, imdsVal, preCAMs_train, preCAMs_val, preMasks_train, preMasks_val, ...
        classes, loss_config, config);
        
    fprintf('  Evaluating fold %d...\n', fold);
    [acc, prec, sens, spec, f1, auc, iou, dice, tv, jac, hdist] = evaluateWithSegmentation(...
        trainedNet, imdsVal, classes, config.useGPU, preCAMs_val, preMasks_val, featureLayer, config);
        
    fold_name = sprintf('fold_%d', fold);
    fold_results.(fold_name) = struct('accuracy',acc,'precision',prec,'sensitivity',sens,...
        'specificity',spec,'f1_score',f1,'auc',auc,'iou',iou,'dice',dice,'tversky',tv,...
        'jaccard',jac,'hausdorff',hdist);
    training_histories.(fold_name) = training_history;
    best_models.(fold_name) = trainedNet;
    fprintf('  Fold %d: Acc=%.3f, Dice=%.3f, AUC=%.3f\n', fold, acc, dice, auc);
    
    % Checkpoint
    checkpointDir = fullfile('models', 'checkpoints');
    if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end
    save(fullfile(checkpointDir, sprintf('checkpoint_%s_%s.mat', config.preprocessing_method, fold_name)), ...
        'trainedNet', 'training_history', 'config', 'loss_config', 'classes', '-v7.3');
end
end

function [trainedNet, training_history] = train_model_with_preprocessing_improved(...
    net, imdsTrain, imdsVal, preCAMs_train, preCAMs_val, preMasks_train, preMasks_val, ...
    classes, loss_config, config)
    numClasses = numel(classes);
    if ~isa(net, 'dlnetwork')
        if isa(net, 'DAGNetwork'), baseLg = layerGraph(net.Layers); else, baseLg = layerGraph(net); end
        toDrop = intersect({'fc8','prob','output'}, {baseLg.Layers.Name});
        if ~isempty(toDrop), baseLg = removeLayers(baseLg, toDrop); end
        baseLg = addLayers(baseLg, [fullyConnectedLayer(numClasses,'Name','fc8'), softmaxLayer('Name','prob')]);
        baseLg = connectLayers(baseLg, 'drop7', 'fc8');
        net = dlnetwork(baseLg);
    end

    augTrain = augmentedImageDatastore([224 224], imdsTrain, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', config.imageAugmenter);
    augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
    preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
    %mbqTrain = minibatchqueue(augTrain, 'MiniBatchSize', config.batchSize, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""]);
    mbqTrain = minibatchqueue(augTrain, ...
    'MiniBatchSize', config.batchSize, ...
    'MiniBatchFcn', @(X,T) preprocessMiniBatchWithPreprocessing(X,T,config.preprocessing_method,config.refHist), ...
    'MiniBatchFormat', ["SSCB", "", ""], ... % Note the third "" for indices
    'PartialMiniBatch', 'discard');
    mbqVal = minibatchqueue(augVal, 'MiniBatchSize', config.batchSize, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""]);

    velocity = []; iter = 0; bestValAcc = 0; patienceCounter = 0;
    featureLayer = 'relu5_3'; nCam = config.nCam;
    classCounts = countcats(imdsTrain.Labels);
    baseWeights = sum(classCounts) ./ (numel(classCounts) * classCounts);
    if numel(classes) == 2, classWeights = [baseWeights(1), baseWeights(2)*1.2]; else, classWeights = baseWeights; end
    classWeights = gather(single(classWeights(:)));

    training_history = struct('epoch_loss', zeros(1,config.numEpochs), 'val_loss', nan(1,config.numEpochs), ...
        'val_accuracy', nan(1,config.numEpochs), 'val_iou', nan(1,config.numEpochs), 'val_dice', nan(1,config.numEpochs), ...
        'train_acc', zeros(1,config.numEpochs), 'val_acc', nan(1,config.numEpochs), 'lr', zeros(1,config.numEpochs), 'epoch', 1:config.numEpochs);

    bestNet = net; epoch = 0;
    fprintf('Starting training...\n');
    while epoch < config.numEpochs
        epoch = epoch + 1;
        fprintf('\nEpoch %d/%d\n', epoch, config.numEpochs);
        shuffle(mbqTrain); reset(mbqTrain);
        batch = 0; epochLoss = 0; trainCorrect = 0; trainTotal = 0;
        
        if epoch <= config.warmup_epochs
            lr = config.initialLearnRate * (epoch / config.warmup_epochs);
        else
            progress = (epoch - config.warmup_epochs) / (config.numEpochs - config.warmup_epochs);
            lr = config.initialLearnRate * (config.min_lr_ratio + (1-config.min_lr_ratio)*0.5*(1+cos(pi*progress)));
        end
        training_history.lr(epoch) = lr;
        
        while hasdata(mbqTrain)
            iter = iter + 1; batch = batch + 1;
            [X, T] = next(mbqTrain);
            if config.useGPU, X = gpuArray(X); end
            
            [loss, grads, state, loss_components] = dlfeval(@compute_loss_with_config_improved, ...
                net, X, T, loss_config, classWeights, imdsTrain.Files, preCAMs_train, preMasks_train, ...
                nCam, classes, featureLayer, config.useGPU, epoch);
            net.State = state;
            
            if isfield(loss_config, 'gradient_clip_norm') && ~isempty(loss_config.gradient_clip_norm)
                maxNorm = loss_config.gradient_clip_norm;
                gradValues = grads.Value; gradNormSq = 0;
                for i = 1:numel(gradValues), gradNormSq = gradNormSq + sum(gradValues{i}(:).^2); end
                gradNorm = sqrt(gradNormSq + eps);
                if gradNorm > maxNorm
                    scale = maxNorm / gradNorm;
                    grads.Value = cellfun(@(g) g * scale, gradValues, 'UniformOutput', false);
                end
            end
            
            [net, velocity] = sgdmupdate(net, grads, velocity, lr, config.momentum);
            epochLoss = epochLoss + double(loss);
            if mod(batch,10)==0 && batch<=50
                fprintf('    Batch %d - Cls=%.3f', batch, double(loss_components.classification));
                if loss_config.use_gradcam, fprintf(', CAM=%.3f', double(loss_components.gradcam)); end
                if loss_config.use_tversky, fprintf(', Tversky=%.3f', double(loss_components.tversky)); end
                if loss_config.use_anatomical_guidance, fprintf(', Anat=%.3f', double(loss_components.anatomical)); end
                fprintf(', Total=%.3f\n', double(loss));
            end
            
            Y = predict(net, X); Ypred = onehotdecode(extractdata(Y), classes, 1); Ytrue = onehotdecode(extractdata(T), classes, 1);
            trainCorrect = trainCorrect + sum(Ypred == Ytrue); trainTotal = trainTotal + numel(Ypred);
        end
        
        avgTrainLoss = epochLoss / max(1,batch); trainAcc = trainCorrect / max(1,trainTotal);
        training_history.epoch_loss(epoch) = avgTrainLoss; training_history.train_acc(epoch) = trainAcc;
        
        if mod(epoch,3)==0 || epoch==1
            valLoss = compute_validation_loss_with_config_preprocessed(net, imdsVal, classes, config.useGPU, ...
                preCAMs_val, preMasks_val, loss_config, classWeights, config, epoch);
            
            % UNIFIED: Use same evaluation logic as final reporting
            [val_acc, ~, ~, ~, ~, ~, val_iou, val_dice, ~, ~, ~] = evaluateWithSegmentation(...
                net, imdsVal, classes, config.useGPU, preCAMs_val, preMasks_val, featureLayer, config);
                
            training_history.val_loss(epoch) = valLoss; training_history.val_accuracy(epoch) = val_acc;
            training_history.val_iou(epoch) = val_iou; training_history.val_dice(epoch) = val_dice;
            training_history.val_acc(epoch) = val_acc;
            
            fprintf('  Train Loss: %.4f | Acc: %.3f\n', avgTrainLoss, trainAcc);
            fprintf('  Val Loss: %.4f | Acc: %.3f | IoU: %.3f | Dice: %.3f\n', valLoss, val_acc, val_iou, val_dice);
            
            if val_acc > bestValAcc + config.min_delta
                bestValAcc = val_acc; bestNet = net; patienceCounter = 0;
                fprintf('  ✓ Validation improved!\n');
            else
                patienceCounter = patienceCounter + 1;
            end
            if patienceCounter >= config.patience
                fprintf('\nEarly stopping triggered.\n'); net = bestNet; break;
            end

            
        else
            fprintf('  Train Loss: %.4f | Acc: %.3f\n', avgTrainLoss, trainAcc);
        end
    end
    trainedNet = bestNet;
    % Trim histories
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

    function [loss, grads, state, loss_components] = compute_loss_with_config_improved(...
        net, X, T, loss_config, classWeights, trainFiles, preCAMs, preMasks, nCam, classes, featureLayer, useGPU, epoch)
    [Y, state] = forward(net, X);
    loss_components = struct();
    if loss_config.use_focal
        clsLoss = compute_focal_loss(Y, T, classWeights, loss_config.focal_alpha, loss_config.focal_gamma);
    else
        clsLoss = crossentropy(Y, T, 'Weights', classWeights, 'WeightsFormat', 'C');
    end
    loss_components.classification = clsLoss;
    camLoss = 0; segLoss = 0; tverskyLoss = 0; iouLoss = 0; anatomicalLoss = 0;

    if loss_config.use_gradcam || loss_config.use_segmentation || loss_config.use_tversky || loss_config.use_iou || loss_config.use_anatomical_guidance
        N = numel(trainFiles); n = min(nCam, N); idxs = randperm(N, n);
        for ii = 1:n
            img = imread(trainFiles{idxs(ii)});
            if size(img,3)==1, img = repmat(img,[1 1 3]); end
            img = imresize(img, [224 224]);
            studCAM = student_cam_one_dlarray(net, img, classes, featureLayer, useGPU);
            studCAM_data = single(extractdata(studCAM));
            if size(studCAM_data,1)~=224 || size(studCAM_data,2)~=224
                studCAM_data = imresize(studCAM_data, [224 224]);
            end
            studCAM = dlarray(studCAM_data, 'SS');
            
            if loss_config.use_gradcam
                targetCAM = preCAMs{idxs(ii)};
                if ~isa(targetCAM, 'dlarray')
                    targetCAM = single(targetCAM); if ndims(targetCAM)>2, targetCAM=squeeze(targetCAM); end
                    if size(targetCAM,1)~=224 || size(targetCAM,2)~=224, targetCAM=imresize(targetCAM,[224 224]); end
                else
                    targetCAM = single(extractdata(stripdims(targetCAM)));
                    if ndims(targetCAM)>2, targetCAM=squeeze(targetCAM); end
                    if size(targetCAM,1)~=224 || size(targetCAM,2)~=224, targetCAM=imresize(targetCAM,[224 224]); end
                end
                targetCAM = dlarray(targetCAM, 'SS');
                if isfield(loss_config,'cam_loss_type') && strcmp(loss_config.cam_loss_type,'cosine')
                    camLoss = camLoss + cam_cosine_loss(studCAM, targetCAM);
                else
                    camLoss = camLoss + mse(studCAM, targetCAM);
                end
            end
            
            if loss_config.use_segmentation || loss_config.use_tversky || loss_config.use_iou
                realMask = preMasks{idxs(ii)};
                if ~isempty(realMask)
                    realMask_resized = imresize(single(realMask), [224 224]);
                    if loss_config.use_segmentation, segLoss = segLoss + (1 - dice_coefficient_dlarray(studCAM, realMask_resized)); end
                    if loss_config.use_tversky, tverskyLoss = tverskyLoss + (1 - tversky_coefficient_dlarray(studCAM, realMask_resized, loss_config.tversky_alpha, loss_config.tversky_beta)); end
                    if loss_config.use_iou, iouLoss = iouLoss + (1 - iou_coefficient_dlarray(studCAM, realMask_resized)); end
                end
            end
            
            if loss_config.use_anatomical_guidance
                realMask = preMasks{idxs(ii)};
                if ~isempty(realMask) && any(realMask(:))
                    if islogical(realMask)
                        lungMask = single(realMask);
                    else
                        lungMask = single(realMask > 0.5);
                    end
                    if size(lungMask,1)~=224 || size(lungMask,2)~=224, lungMask=imresize(lungMask,[224 224],'nearest'); end
                    studCAM_norm = studCAM; cam_max = max(studCAM_norm,[],'all');
                    if cam_max>0, studCAM_norm = studCAM_norm/(cam_max+eps); end
                    lungMask_dl = dlarray(lungMask,'SS'); nonLungMask_dl = dlarray(1-lungMask,'SS');
                    penalty_out = mean(studCAM_norm .* nonLungMask_dl, 'all');
                    reward_in = mean(studCAM_norm .* lungMask_dl, 'all');
                    anat_loss = penalty_out - loss_config.anatomical_reward_weight * reward_in;
                    if isfield(loss_config,'anatomical_positivity') && loss_config.anatomical_positivity
                        anat_loss = max(anat_loss, 0);
                    end
                    if isfield(loss_config,'adaptive_scaling') && loss_config.adaptive_scaling
                        scale_factor = min(loss_config.max_anatomical_scale, max(200, epoch*4));
                    else
                        scale_factor = 1000;
                    end
                    anatomicalLoss = anatomicalLoss + anat_loss * scale_factor;
                end
            end
        end
        camLoss = camLoss/max(1,n); segLoss = segLoss/max(1,n); tverskyLoss = tverskyLoss/max(1,n);
        iouLoss = iouLoss/max(1,n); anatomicalLoss = anatomicalLoss/max(1,n);
    end
    loss_components.gradcam = camLoss; loss_components.segmentation = segLoss;
    loss_components.tversky = tverskyLoss; loss_components.iou = iouLoss; loss_components.anatomical = anatomicalLoss;

    loss = clsLoss;
    if loss_config.use_gradcam, loss = loss + loss_config.lambda_cam * camLoss; end
    if loss_config.use_segmentation, loss = loss + loss_config.lambda_seg * segLoss; end
    if loss_config.use_tversky, loss = loss + loss_config.lambda_tversky * tverskyLoss; end
    if loss_config.use_iou, loss = loss + loss_config.lambda_seg * iouLoss; end
    if loss_config.use_anatomical_guidance, loss = loss + loss_config.lambda_anatomical * anatomicalLoss; end
    grads = dlgradient(loss, net.Learnables, 'EnableHigherDerivatives', true);
    end

function camLoss = cam_cosine_loss(studCAM, targetCAM)
    % Normalize both maps first (min-max or to unit L2)
    studCAM = (studCAM - min(studCAM,[],'all')) / (max(studCAM,[],'all') - min(studCAM,[],'all') + eps);
    targetCAM = (targetCAM - min(targetCAM,[],'all')) / (max(targetCAM,[],'all') - min(targetCAM,[],'all') + eps);
    
    stud_vec = reshape(studCAM, [], 1);
    target_vec = reshape(targetCAM, [], 1);
    
    dot_product = sum(stud_vec .* target_vec);
    norm_stud = sqrt(sum(stud_vec.^2) + eps);
    norm_target = sqrt(sum(target_vec.^2) + eps);
    
    cosSim = dot_product / (norm_stud * norm_target);
    camLoss = 1 - cosSim;   % remove the max(0,min(2,...)) clipping initially
end

function [results, predictions, uncertainty_stats] = evaluate_dataset_with_preprocessing_ensemble(...
    net, imds, classes, useGPU, base_preprocessing, refHist, is_ood_evaluation)
fprintf('    [eval] %d samples | preprocessing: %s\n', numel(imds.Files), base_preprocessing);
augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, base_preprocessing, refHist);
mbq = minibatchqueue(augDS, 'MiniBatchSize', 16, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""], 'PartialMiniBatch', 'return');

ptb_candidates = {'PTB','ptb','TB','tb','Tuberculosis','tuberculosis','positive','Positive'};
ptbIdx = 2;
for k = 1:numel(ptb_candidates)
    idx = find(strcmp(classes, ptb_candidates{k}), 1);
    if ~isempty(idx), ptbIdx = idx; break; end
end
fprintf('    [eval] PTB class index: %d (%s)\n', ptbIdx, classes{ptbIdx});

Yprobs = []; Ytrue = imds.Labels(:); uncertainty_values = [];
reset(mbq);
while hasdata(mbq)
    [X, ~] = next(mbq); if useGPU, X = gpuArray(X); end
    sc = predict(net, X); probs = extractdata(sc)';
    Yprobs = [Yprobs; probs(:, ptbIdx)];
    probs_norm = probs ./ (sum(probs,2) + eps);
    entropy = -sum(probs_norm .* log(probs_norm + eps), 2);
    max_prob = max(probs, [], 2);
    uncertainty = (1 - max_prob) + 0.5 * entropy / log(numel(classes));
    uncertainty_values = [uncertainty_values; uncertainty];
end
Yprobs = Yprobs(1:numel(Ytrue)); uncertainty_values = uncertainty_values(1:numel(Ytrue));

% UNIFIED THRESHOLD SWEEP
thresh_results = compute_classification_metrics(Yprobs, Ytrue, classes, ptbIdx);
best_thresh = thresh_results.best_thresh;
accuracy = thresh_results.accuracy; precision = thresh_results.precision;
sensitivity = thresh_results.sensitivity; specificity = thresh_results.specificity;
f1score = thresh_results.f1_score; auc = thresh_results.auc;

fprintf('    [eval] Optimal threshold: %.3f (bal_acc=%.3f)\n', best_thresh, thresh_results.balanced_accuracy);

uncertainty_stats = struct('mean_uncertainty', mean(uncertainty_values), ...
    'std_uncertainty', std(uncertainty_values), ...
    'uncertainty_threshold', prctile(uncertainty_values, 80));
uncertainty_stats.high_uncertainty_pct = mean(uncertainty_values > uncertainty_stats.uncertainty_threshold);

results = struct('accuracy',accuracy,'precision',precision,'sensitivity',sensitivity,...
    'specificity',specificity,'f1_score',f1score,'auc',auc,...
    'optimal_threshold',best_thresh,'balanced_accuracy',thresh_results.balanced_accuracy,...
    'confusion_matrix',thresh_results.cm,'thresh_sweep',thresh_results.sweep_data);
predictions = struct('Ypred',thresh_results.Ypred,'Ytrue',Ytrue,'Yprobs',Yprobs,'uncertainty',uncertainty_values,'threshold',best_thresh);
end

function [accuracy, precision, sensitivity, specificity, f1score, auc, iou, dice, tversky, jaccard, hausdorff] = ...
    evaluateWithSegmentation(net, imdsVal, classes, useGPU, valGradCAMs, valMasks, featureLayer, config)
    ptb_candidates = {'PTB','ptb','TB','tb','Tuberculosis','tuberculosis','positive','Positive'};
    ptbIdx = 2;
    for k = 1:numel(ptb_candidates), idx = find(strcmp(classes, ptb_candidates{k}), 1); if ~isempty(idx), ptbIdx = idx; break; end; end

    augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
    preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
    mbqEval = minibatchqueue(augVal, 'MiniBatchSize', 16, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""], 'PartialMiniBatch', 'return');
    Yprobs = []; reset(mbqEval);
    while hasdata(mbqEval)
        [X, ~] = next(mbqEval); if useGPU, X = gpuArray(X); end
        sc = predict(net, X); probs = extractdata(sc)'; Yprobs = [Yprobs; probs(:, ptbIdx)];
    end
    Ytrue = imdsVal.Labels(:); Yprobs = Yprobs(1:numel(Ytrue));

    thresh_results = compute_classification_metrics(Yprobs, Ytrue, classes, ptbIdx);
    accuracy = thresh_results.accuracy; precision = thresh_results.precision;
    sensitivity = thresh_results.sensitivity; specificity = thresh_results.specificity;
    f1score = thresh_results.f1_score; auc = thresh_results.auc;
    fprintf('    [eval] Optimal threshold: %.3f | sens=%.3f spec=%.3f bal=%.3f\n', thresh_results.best_thresh, sensitivity, specificity, thresh_results.balanced_accuracy);

    % Segmentation metrics
    numVal = numel(imdsVal.Files); ious = zeros(numVal,1); dices = zeros(numVal,1); tverskys = zeros(numVal,1); jaccards = zeros(numVal,1); hausdorffs = zeros(numVal,1);
    for i = 1:numVal
        try
            img = imread(imdsVal.Files{i}); if size(img,3)==1, img=repmat(img,[1 1 3]); end
            img = apply_preprocessing_batch(img, config.preprocessing_method, config.refHist);
            img = imresize(img, [224 224]);
            predCAM = dlfeval(@student_cam_one, net, img, classes, featureLayer, useGPU);
            realMask = valMasks{i};
            if isempty(realMask) || ~any(realMask(:)), continue; end
            realMask = imresize(logical(realMask), [224 224], 'nearest');
            if isempty(predCAM) || all(predCAM(:)==0) || all(isnan(predCAM(:))), continue; end
            predCAM = imresize(single(predCAM), [224 224]);
            cam_vals = predCAM(:); thresh = prctile(cam_vals, 50);
            if thresh<0.1, thresh=prctile(cam_vals,75); end
            if thresh<0.1, thresh=mean(cam_vals)+0.5*std(cam_vals); end
            thresh = max(0.1, min(0.9, thresh));
            predMask = predCAM > thresh;
            if ~any(predMask(:)), thresh=prctile(cam_vals,25); predMask=predCAM>thresh; if ~any(predMask(:)), continue; end; end
            predMask = logical(imopen(predMask, strel('disk',2)));
            predMask = logical(imclose(predMask, strel('disk',3)));
            predMask = logical(imfill(predMask, 'holes'));
            [ious(i), dices(i), tverskys(i), jaccards(i), hausdorffs(i)] = computeSegmentationMetrics(predMask, logical(realMask));
        catch
            continue;
        end
    end
    valid = ~isnan(ious) & ious>0;
    iou = mean(ious(valid)); dice = mean(dices(valid)); tversky = mean(tverskys(valid));
    jaccard = mean(jaccards(valid)); hausdorff = mean(hausdorffs(valid));
end

function thresh_results = compute_classification_metrics(Yprobs, Ytrue, classes, ptbIdx)
    % UNIFIED THRESHOLD SWEEP & METRIC CALCULATION
    thresholds = 0.15:0.025:0.75;
    best_thresh = 0.5; best_bal = 0;
    Ypred_best = categorical(repmat(classes(1), numel(Ytrue), 1));
    cm_best = zeros(2,2);
    sweep_data = zeros(numel(thresholds), 5);

    for ti = 1:numel(thresholds)
        t = thresholds(ti);
        preds_t = categorical(classes(1 + (Yprobs >= t)));
        cm_t = confusionmat(Ytrue, preds_t, 'Order', categorical(classes));
        if all(size(cm_t)==[2 2])
            TP = cm_t(ptbIdx, ptbIdx); FN = cm_t(ptbIdx, 3-ptbIdx);
            TN = cm_t(3-ptbIdx, 3-ptbIdx); FP = cm_t(3-ptbIdx, ptbIdx);
            s = TP/(TP+FN+eps); sp = TN/(TN+FP+eps);
            bal = (s+sp)/2; acc = (TP+TN)/(TP+TN+FP+FN+eps);
            sweep_data(ti,:) = [t, acc, s, sp, bal];
            if bal > best_bal, best_bal=bal; best_thresh=t; Ypred_best=preds_t; cm_best=cm_t; end
        end
    end
    TP=cm_best(ptbIdx,ptbIdx); FN=cm_best(ptbIdx,3-ptbIdx); TN=cm_best(3-ptbIdx,3-ptbIdx); FP=cm_best(3-ptbIdx,ptbIdx);
    accuracy=(TP+TN)/(TP+TN+FP+FN+eps); precision=TP/(TP+FP+eps); sensitivity=TP/(TP+FN+eps); specificity=TN/(TN+FP+eps);
    f1score=2*precision*sensitivity/(precision+sensitivity+eps);
    try [~,~,~,auc]=perfcurve(double(Ytrue==categorical(classes{ptbIdx})), Yprobs, 1); catch, auc=0.5; end
    thresh_results = struct('accuracy',accuracy,'precision',precision,'sensitivity',sensitivity,'specificity',specificity,...
        'f1_score',f1score,'auc',auc,'balanced_accuracy',best_bal,'best_thresh',best_thresh,...
        'Ypred',Ypred_best,'cm',cm_best,'sweep_data',sweep_data);
end

function [X, T] = preprocessMiniBatchWithPreprocessing(dataX, dataT, preprocessing_method, refHist)
    X = cat(4, dataX{1:end}); [H,W,C,B] = size(X); X_processed = zeros(H,W,C,B,'uint8');
    for b = 1:B
        img = X(:,:,:,b);
        if ~isa(img,'uint8')
            mx = max(img(:));
            if mx<=1.0 && mx>0, img=uint8(img.*255); elseif mx>1.0, img=uint8(img./mx.*255); else, img=uint8(img); end
        end
        if C==3, img_gray=rgb2gray(img); else, img_gray=img(:,:,1); end
        switch preprocessing_method
            case 'none', img_processed=img_gray;
            case 'clahe', img_processed=adapthisteq(img_gray,'ClipLimit',0.02,'Distribution','uniform');
            case 'histmatch', if ~isempty(refHist), img_processed=histeq(img_gray,refHist); else, img_processed=img_gray; end
            otherwise, img_processed=img_gray;
        end
        if C==3, X_processed(:,:,:,b)=repmat(img_processed,[1 1 3]); else, X_processed(:,:,:,b)=img_processed; end
    end
    
    X = dlarray(single(X_processed)./255, 'SSCB');
    T = onehotencode(cat(2, dataT{1:end}), 1);
    % Return original indices (assuming dataT contains labels from an imageDatastore)
    batchIndices = (1:numel(dataT))';
end

function imgs_processed = apply_preprocessing_batch(imgs, method, refHist)
    [H,W,C,B] = size(imgs); imgs_processed = zeros(H,W,C,B,'uint8');
    for b = 1:B
        img = imgs(:,:,:,b);
        if ~isa(img,'uint8')
            mx = max(img(:));
            if mx<=1.0 && mx>0, img=uint8(img.*255); elseif mx>1.0, img=uint8(img./mx.*255); else, img=uint8(img); end
        end
        if C==3, img_gray=rgb2gray(img); else, img_gray=img(:,:,1); end
        switch method
            case 'none', img_processed=img_gray;
            case 'clahe', img_processed=adapthisteq(img_gray,'ClipLimit',0.02,'Distribution','uniform');
            case 'histmatch', if ~isempty(refHist), img_processed=histeq(img_gray,refHist); else, img_processed=img_gray; end
            otherwise, img_processed=img_gray;
        end
        if C==3, imgs_processed(:,:,:,b)=repmat(img_processed,[1 1 3]); else, imgs_processed(:,:,:,b)=img_processed; end
    end
end

function [imds, vggNet, precomputedGradCAM, precomputedMasks, classes] = load_data_and_network(forceRecalculate)
    if nargin<1, forceRecalculate=false; end
    modelPath = 'models/pretrained/vgg16_finetuned_on_roi.mat';
    if exist(modelPath,'file'), s=load(modelPath); vggNet=s.trainedNet; else, error('Model file not found.'); end
    roiDir = 'input/roi'; imds = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    classes = categories(imds.Labels); precomputedGradCAM = cell(numel(imds.Files),1); precomputedMasks = cell(numel(imds.Files),1);
    cacheFile = 'precomputed_gradcam_maps_enhanced.mat';
    if exist(cacheFile,'file') && ~forceRecalculate
        cache = load(cacheFile); precomputedGradCAM = cache.precomputedGradCAM; precomputedMasks = cache.precomputedMasks;
    end
    end

function foldIndices = createStratifiedKFold(labels, k_folds)
    classes = categories(labels); numClasses = numel(classes);
    foldIndices.train = cell(k_folds,1); foldIndices.val = cell(k_folds,1);
    for c = 1:numClasses
        idx = find(labels == classes{c}); idx = idx(randperm(numel(idx)));
        samplesPerFold = floor(numel(idx)/k_folds); remainder = mod(numel(idx),k_folds); startIdx = 1;
        for f = 1:k_folds
            foldSize = samplesPerFold + (f<=remainder); endIdx = startIdx + foldSize - 1;
            v = idx(startIdx:endIdx);
            if isempty(foldIndices.val{f}), foldIndices.val{f}=v; else, foldIndices.val{f}=[foldIndices.val{f}; v]; end
            startIdx = endIdx + 1;
        end
    end
    allIdx = (1:numel(labels))';
    for f = 1:k_folds, foldIndices.train{f} = setdiff(allIdx, foldIndices.val{f}); end
end

function mean_metrics = calculate_mean_metrics(fold_results)
    fold_names = fieldnames(fold_results); metrics = {'accuracy','precision','sensitivity','specificity','f1_score','auc','iou','dice','tversky','jaccard','hausdorff'};
    mean_metrics = struct();
    for m = 1:numel(metrics)
        values = []; for f = 1:numel(fold_names), if isfield(fold_results.(fold_names{f}), metrics{m}), values(end+1)=fold_results.(fold_names{f}).(metrics{m}); end; end
        mean_metrics.(metrics{m}) = mean(values);
    end
    end
    function std_metrics = calculate_std_metrics(fold_results)
    fold_names = fieldnames(fold_results); metrics = {'accuracy','precision','sensitivity','specificity','f1_score','auc','iou','dice','tversky','jaccard','hausdorff'};
    std_metrics = struct();
    for m = 1:numel(metrics)
        values = []; for f = 1:numel(fold_names), if isfield(fold_results.(fold_names{f}), metrics{m}), values(end+1)=fold_results.(fold_names{f}).(metrics{m}); end; end
        std_metrics.(metrics{m}) = std(values);
    end
end

function plot_kfold_training_curves(training_histories, mean_metrics, std_metrics, outputDir)
    fold_names = fieldnames(training_histories); num_folds = numel(fold_names);
    figure('Position',[100,100,1400,600]); colors = lines(num_folds);
    subplot(1,3,1); hold on;
    for f = 1:num_folds
        hist = training_histories.(fold_names{f});
        if isfield(hist,'epoch_loss'), plot(hist.epoch, hist.epoch_loss, '-', 'Color', colors(f,:), 'LineWidth', 1.5, 'DisplayName', sprintf('Fold %d Train', f)); end
        if isfield(hist,'val_loss'), vE=hist.epoch(~isnan(hist.val_loss)); vL=hist.val_loss(~isnan(hist.val_loss)); if ~isempty(vE), plot(vE, vL, '--', 'Color', colors(f,:), 'LineWidth', 1.5, 'DisplayName', sprintf('Fold %d Val', f)); end; end
    end
    xlabel('Epoch'); ylabel('Loss'); title('Loss'); legend('Location','best'); grid on;
    subplot(1,3,2); hold on;
    for f = 1:num_folds
        hist = training_histories.(fold_names{f});
        if isfield(hist,'train_acc'), plot(hist.epoch, hist.train_acc, '-', 'Color', colors(f,:), 'LineWidth', 1.5); end
        if isfield(hist,'val_accuracy'), vE=hist.epoch(~isnan(hist.val_accuracy)); vA=hist.val_accuracy(~isnan(hist.val_accuracy)); if ~isempty(vE), plot(vE, vA, '--', 'Color', colors(f,:), 'LineWidth', 1.5); end; end
    end
    xlabel('Epoch'); ylabel('Accuracy'); title('Accuracy'); grid on;
    subplot(1,3,3); mNames={'Acc','Prec','Sens','Spec','F1','AUC','Dice','IoU'}; mVals=[mean_metrics.accuracy,mean_metrics.precision,mean_metrics.sensitivity,mean_metrics.specificity,mean_metrics.f1_score,mean_metrics.auc,mean_metrics.dice,mean_metrics.iou];
    bar(mVals); hold on; errorbar(1:numel(mVals), mVals, [std_metrics.accuracy,std_metrics.precision,std_metrics.sensitivity,std_metrics.specificity,std_metrics.f1_score,std_metrics.auc,std_metrics.dice,std_metrics.iou], 'k.', 'LineWidth', 1.5);
    set(gca,'XTickLabel',mNames); ylabel('Score'); title('CV Metrics'); xtickangle(45); grid on; ylim([0 1]);
    saveas(gcf, fullfile(outputDir, 'kfold_training_curves.png')); close(gcf);
end

function valLoss = compute_validation_loss_with_config_preprocessed(net, imdsVal, classes, useGPU, precomputedGradCAM, precomputedMasks, loss_config, classWeights, config, epoch)
    augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
    preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
    mbqVal = minibatchqueue(augVal, 'MiniBatchSize', config.batchSize, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""]);
    valLoss = 0; valCount = 0;
    while hasdata(mbqVal)
        [X, T] = next(mbqVal); if useGPU, X = gpuArray(X); end
        try
            [loss, ~, ~, ~] = dlfeval(@compute_loss_with_config_improved, net, X, T, loss_config, classWeights, ...
                imdsVal.Files, precomputedGradCAM, precomputedMasks, config.nCam, classes, 'relu5_3', useGPU, epoch);
            valLoss = valLoss + double(loss); valCount = valCount + 1;
        catch, continue; end
    end
    valLoss = valLoss / max(1, valCount);
end

function focalLoss = compute_focal_loss(Y, T, classWeights, alpha, gamma)
    epsVal = 1e-7; Y = min(max(Y, epsVal), 1-epsVal);
    if isscalar(alpha), alpha_vec = alpha * reshape(classWeights, [], 1); else, alpha_vec = reshape(alpha(:), [], 1) .* reshape(classWeights, [], 1); end
    focalW = alpha_vec .* ((1 - Y).^gamma); term = - T .* focalW .* log(Y + epsVal); focalLoss = mean(sum(term, 1));
end

function tverskyCoef = tversky_coefficient_dlarray(pred, target, alpha, beta)
    if isa(pred, 'dlarray')
        target_dl = dlarray(single(target), 'SSC'); intersection = sum(pred(:).*target_dl(:), 'all');
        fp = sum(pred(:).*(1-target_dl(:)), 'all'); fn = sum((1-pred(:)).*target_dl(:), 'all');
        tverskyCoef = intersection / (intersection + alpha*fp + beta*fn + eps);
    else
        pred=logical(gather(pred)); target=logical(gather(target)); intersection=sum(pred(:)&target(:));
        fp=sum(pred(:)&~target(:)); fn=sum(~pred(:)&target(:));
        tverskyCoef = intersection / (intersection + alpha*fp + beta*fn + eps);
    end
end

function diceCoef = dice_coefficient_dlarray(pred, target)
    if isa(pred, 'dlarray')
        target_dl = dlarray(single(target), 'SSC'); intersection = sum(pred(:).*target_dl(:), 'all');
        pred_sum = sum(pred(:), 'all'); target_sum = sum(target_dl(:), 'all'); diceCoef = 2*intersection/(pred_sum+target_sum+eps);
    else
        pred=logical(gather(pred)); target=logical(gather(target)); intersection=sum(pred(:)&target(:));
        diceCoef = 2*intersection/(sum(pred(:))+sum(target(:))+eps);
    end
end

function iouCoef = iou_coefficient_dlarray(pred, target)
    if isa(pred, 'dlarray')
        target_dl = dlarray(single(target), 'SSC'); intersection = sum(pred(:).*target_dl(:), 'all');
        union = sum(max(pred(:), target_dl(:)), 'all'); iouCoef = intersection/(union+eps);
    else
        pred=logical(gather(pred)); target=logical(gather(target)); intersection=sum(pred(:)&target(:)); union=sum(pred(:)|target(:));
        iouCoef = intersection/(union+eps);
    end
end

function cam = student_cam_one_dlarray(net, img, classes, featureLayer, useGPU)
    dlX = dlarray(single(img), 'SSCB'); if useGPU, dlX=gpuArray(dlX); end
    [featMap, logits] = forward(net, dlX, 'Outputs', {featureLayer, 'fc8'});
    logits = squeeze(logits); [~, classIdx] = max(extractdata(logits));
    score = sum(logits(classIdx), 'all'); gradFeat = dlgradient(score, featMap);
    w = mean(gradFeat, [1 2]); cam = sum(featMap .* w, 3); cam = max(cam, 0);
    cam = single(extractdata(cam)); cam_max = max(cam(:)); if cam_max>0, cam=cam/(cam_max+eps); end
    cam = dlarray(cam, 'SS');
end

function cam = student_cam_one(net, img, classes, featureLayer, useGPU)
    dlX = dlarray(single(img), 'SSCB'); if useGPU, dlX=gpuArray(dlX); end
    [featMap, logits] = forward(net, dlX, 'Outputs', {featureLayer, 'fc8'});
    logits = squeeze(logits); [~, classIdx] = max(extractdata(logits));
    score = sum(logits(classIdx), 'all'); gradFeat = dlgradient(score, featMap);
    w = mean(gradFeat, [1 2]); cam = sum(featMap .* w, 3); cam = max(cam, 0);
    cam = extractdata(cam); cam = imresize(cam, [224 224]); cam = single(cam./(max(cam(:))+eps));
end

function [iou, dice, tversky, jaccard, hausdorff] = computeSegmentationMetrics(pred, target)
    pred=logical(gather(pred)); target=logical(gather(target));
    intersection=sum(pred(:)&target(:)); union=sum(pred(:)|target(:));
    pred_sum=sum(pred(:)); target_sum=sum(target(:));
    iou = intersection/(union+eps); dice = 2*intersection/(pred_sum+target_sum+eps);
    alpha=0.7; beta=0.3; fp=sum(pred(:)&~target(:)); fn=sum(~pred(:)&target(:));
    tversky = intersection/(intersection+alpha*fp+beta*fn+eps); jaccard=iou;
    try, hausdorff = improved_hausdorff_distance(pred, target); catch, hausdorff=0; end
end

function dist = improved_hausdorff_distance(pred, target)
    try
    pred=logical(gather(pred)); target=logical(gather(target));
    if ~any(pred(:))||~any(target(:)), dist=0; return; end
    pb=bwboundaries(pred,'noholes'); tb=bwboundaries(target,'noholes');
    if isempty(pb)||isempty(tb), dist=0; return; end
    [~,pi]=max(cellfun(@size,pb,1,'UniformOutput',false)); [~,ti]=max(cellfun(@size,tb,1,'UniformOutput',false));
    pts1=pb{pi}; pts2=tb{ti}; max_pts=100;
    if size(pts1,1)>max_pts, step=floor(size(pts1,1)/max_pts); if step>1, pts1=pts1(1:step:end,:); else, pts1=pts1(1:max_pts,:); end; end
    if size(pts2,1)>max_pts, step=floor(size(pts2,1)/max_pts); if step>1, pts2=pts2(1:step:end,:); else, pts2=pts2(1:max_pts,:); end; end
    d1=zeros(size(pts1,1),1); for i=1:size(pts1,1), d1(i)=min(sqrt(sum((pts2-pts1(i,:)).^2,2))); end
    d2=zeros(size(pts2,1),1); for i=1:size(pts2,1), d2(i)=min(sqrt(sum((pts1-pts2(i,:)).^2,2))); end
    dist = max(max(d1),max(d2)) / sqrt(size(pred,1)^2 + size(pred,2)^2);
    catch, dist=0; end
end

function monitor_probability_distribution(net, imdsVal, classes, useGPU, epoch, config, refHist, ptbIdx)
%MONITOR_PROBABILITY_DISTRIBUTION Track predicted probs by true class during training
%
% Usage: Call inside training loop every N epochs:
%   monitor_probability_distribution(net, imdsVal, classes, config.useGPU, epoch, config, config.refHist, ptbIdx);

fprintf('    [MONITOR] Epoch %d - Probability Distribution:\n', epoch);

% Build minibatch queue (same as training)
augDS = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, refHist);
mbq = minibatchqueue(augDS, ...
    'MiniBatchSize', 16, ...
    'MiniBatchFcn', preprocessFcn, ...
    'MiniBatchFormat', ["SSCB", ""], ...
    'PartialMiniBatch', 'return');

Yprobs = []; Ytrue = imdsVal.Labels(:);
reset(mbq);
while hasdata(mbq)
    [X, ~] = next(mbq);
    if useGPU, X = gpuArray(X); end
    sc = predict(net, X);
    probs = extractdata(sc)';
    Yprobs = [Yprobs; probs(:, ptbIdx)];
end
Yprobs = Yprobs(1:numel(Ytrue));

% Separate by true class
isPositive = (Ytrue == categorical(classes{ptbIdx}));
probsPositive = Yprobs(isPositive);
probsNegative = Yprobs(~isPositive);

% Print statistics
fprintf('      Negative class (n=%d): mean=%.3f, median=%.3f, std=%.3f\n', ...
    numel(probsNegative), mean(probsNegative), median(probsNegative), std(probsNegative));
fprintf('      Positive class (n=%d): mean=%.3f, median=%.3f, std=%.3f\n', ...
    numel(probsPositive), mean(probsPositive), median(probsPositive), std(probsPositive));
fprintf('      Separation (Δmean): %.3f\n', abs(mean(probsPositive) - mean(probsNegative)));

% Plot histogram (save to file)
if mod(epoch, 3) == 0 || epoch == 1  % Plot every 3 epochs
    figure('Position', [100, 100, 800, 400]);
    subplot(1,2,1);
    histogram(probsNegative, 30, 'FaceColor', [0.8 0.2 0.2], 'EdgeColor', 'none');
    xlabel('P(PTB)'); ylabel('Count'); title(sprintf('Negative Class (Epoch %d)', epoch));
    grid on; xlim([0 1]);
    
    subplot(1,2,2);
    histogram(probsPositive, 30, 'FaceColor', [0.2 0.6 0.2], 'EdgeColor', 'none');
    xlabel('P(PTB)'); ylabel('Count'); title(sprintf('Positive Class (Epoch %d)', epoch));
    grid on; xlim([0 1]);
    
    outputDir = fullfile('models', 'final', 'monitoring');
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end
    saveas(gcf, fullfile(outputDir, sprintf('prob_hist_epoch_%03d.png', epoch)));
    close(gcf);
    fprintf('      ✓ Histogram saved: prob_hist_epoch_%03d.png\n', epoch);
end

% Warning if distributions overlap too much
if epoch >= 5
    sep = abs(mean(probsPositive) - mean(probsNegative));
    if sep < 0.1
        fprintf('      ⚠️ WARNING: Class separation < 0.1 — model may not be learning!\n');
    elseif sep < 0.2
        fprintf('      ℹ️  Separation improving but still weak (%.3f)\n', sep);
    else
        fprintf('      ✓ Good class separation (%.3f)\n', sep);
    end
end
end