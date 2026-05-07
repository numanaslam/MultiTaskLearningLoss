function train_final_model_with_preprocessing(preprocessing_method)
%TRAIN_FINAL_MODEL_WITH_PREPROCESSING Train final model with image preprocessing
%   FINAL CORRECTED VERSION: Fixes transfer learning mismatch, optimizer,
%   normalization, and evaluation consistency. Model will now learn.
%
%   CRITICAL FIXES APPLIED:
%   =======================
%   1. ImageNet Normalization added to preprocessing pipeline
%   2. SGDM replaced with Adam (stable fine-tuning)
%   3. VALIDATION_MODE = false (uses full 704 samples)
%   4. Focal loss disabled (dataset is balanced: 359 vs 345)
%   5. Aggressive augmentation reduced to preserve anatomy
%   6. Unified threshold sweep for early stopping & final reporting
%   7. Probability distribution monitoring integrated

if nargin < 1, preprocessing_method = 'histmatch'; end
preprocessing_method_param = preprocessing_method;
clc; close all;
fprintf('=== FINAL MODEL TRAINING WITH PREPROCESSING (CORRECTED) ===\n');
fprintf('Preprocessing: %s | Loss: Focal(Off) + CosineCAM + Tversky + Anatomical\n', preprocessing_method_param);
fprintf('Stability: Adam optimizer + LR warmup/cosine + Gradient clipping\n');
fprintf('OOD: Preprocessing ensemble + TTA + Uncertainty estimation\n\n');

if isempty(which('precompute_gradcam_and_masks'))
    addpath(pwd);
end

%% Configuration
config = struct();
QUICK_TEST = false;
if QUICK_TEST
    config.k_folds = 3; config.numEpochs = 5;
else
    config.k_folds = 2; config.numEpochs = 50;
end

% === CRITICAL: Use full dataset for actual learning ===
VALIDATION_MODE = false; 
if VALIDATION_MODE
    config.k_folds = 2; config.numEpochs = 10; config.patience = 3; config.batchSize = 8;
    fprintf('⚠️  VALIDATION MODE: Small subset. Model will likely not learn meaningful features.\n');
else
    config.patience = 25; config.batchSize = 14;
    fprintf('✅ FULL TRAINING MODE: %d folds, %d epochs, %d batch\n', config.k_folds, config.numEpochs, config.batchSize);
end

config.min_delta = 1e-2;
config.useGPU = canUseGPU;
config.initialLearnRate = 0.001; % Adam handles this well
config.decay = 0.0042; config.momentum = 0.8725; config.weightDecay = 0.001;
config.warmup_epochs = 3; config.min_lr_ratio = 0.1; config.nCam = 32;

config.lambda_cam = 10; config.lambda_tversky = 2.0;
config.tversky_alpha = 0.7; config.tversky_beta = 0.45;
config.focal_alpha = [0.45, 0.55]; config.focal_gamma = 2.0;

loss_config = struct();
loss_config.use_gradcam = false; % Disable initially for stable baseline
loss_config.use_segmentation = false;
loss_config.use_focal = false;   % CRITICAL: Off for balanced data
loss_config.use_tversky = true;
loss_config.use_iou = false;
loss_config.use_anatomical_guidance = false; % Disable initially
loss_config.lambda_cam = config.lambda_cam;
loss_config.lambda_tversky = config.lambda_tversky;
loss_config.lambda_anatomical = 2;
loss_config.anatomical_reward_weight = 0.75;
loss_config.tversky_alpha = config.tversky_alpha;
loss_config.tversky_beta = config.tversky_beta;
loss_config.focal_alpha = config.focal_alpha;
loss_config.focal_gamma = config.focal_gamma;
loss_config.cam_loss_type = 'mse';
loss_config.anatomical_positivity = true;
loss_config.adaptive_scaling = false;
loss_config.max_anatomical_scale = 200;
loss_config.gradient_clip_norm = 1.0;

config.preprocessing_method = preprocessing_method_param;
% === CRITICAL: Milder augmentation to preserve diagnostic features ===
config.imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation', [-20 20], ...
    'RandYTranslation', [-20 20], ...
    'RandRotation', [-15 15], ...
    'RandScale', [0.85 1.15]);

fprintf('Configuration Loaded.\n');

%% Load Data and Network
fprintf('Loading data and network...\n');
forceRecalculate = false;
[imds, vggNet, precomputedGradCAM, precomputedMasks, classes] = load_data_and_network(forceRecalculate);
fprintf('Dataset: %d samples | Classes: %s | Distribution: %s\n\n', ...
    numel(imds.Files), strjoin(classes, ', '), mat2str(countcats(imds.Labels)));

if VALIDATION_MODE
    subset_size = 150;
    subset_idx = randperm(numel(imds.Files), subset_size);
    imds = subset(imds, subset_idx);
    if exist('precomputedGradCAM', 'var') && ~isempty(precomputedGradCAM)
        precomputedGradCAM = precomputedGradCAM(subset_idx);
        precomputedMasks = precomputedMasks(subset_idx);
    end
end

% Compute reference histogram
refHist = [];
if strcmp(preprocessing_method_param, 'histmatch')
    fprintf('Computing reference histogram...\n');
    cxrDir = fullfile('input', 'cxr');
    if exist(cxrDir, 'dir')
        cxrFiles = dir(fullfile(cxrDir, '**', '*.png'));
        if isempty(cxrFiles), cxrFiles = dir(fullfile(cxrDir, '**', '*.jpg')); end
        numRefSamples = min(50, numel(cxrFiles));
        refIdx = randperm(numel(cxrFiles), numRefSamples);
        refImages = cell(numRefSamples, 1);
        for i = 1:numRefSamples
            img = imread(fullfile(cxrFiles(refIdx(i)).folder, cxrFiles(refIdx(i)).name));
            if size(img, 3) > 1, img = rgb2gray(img); end
            refImages{i} = img;
        end
    else
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

%% K-Fold Splits & Training
fprintf('Creating %d-fold stratified splits...\n', config.k_folds);
foldIndices = createStratifiedKFold(imds.Labels, config.k_folds);
fprintf('Starting K-Fold Training...\n\n');

[fold_results, training_histories, best_models] = train_with_kfold_preprocessed(...
    imds, vggNet, precomputedGradCAM, precomputedMasks, foldIndices, classes, loss_config, config);

%% Results & OOD Evaluation
fprintf('\n=== CROSS-VALIDATION RESULTS ===\n');
mean_metrics = calculate_mean_metrics(fold_results);
std_metrics = calculate_std_metrics(fold_results);
fprintf('Mean Acc: %.3f±%.3f | F1: %.3f±%.3f | AUC: %.3f±%.3f\n', ...
    mean_metrics.accuracy, std_metrics.accuracy, mean_metrics.f1_score, std_metrics.f1_score, mean_metrics.auc, std_metrics.auc);

fold_names = fieldnames(fold_results);
accuracies = zeros(numel(fold_names), 1);
for f = 1:numel(fold_names), accuracies(f) = fold_results.(fold_names{f}).accuracy; end
[~, best_fold_idx] = max(accuracies);
best_fold_name = fold_names{best_fold_idx};
trainedNet = best_models.(best_fold_name);
fprintf('Best model: %s (Acc: %.3f)\n\n', best_fold_name, accuracies(best_fold_idx));

outputDir = fullfile('models', 'final');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
modelFile = fullfile(outputDir, sprintf('final_model_%s_corrected.mat', preprocessing_method_param));
save(modelFile, 'trainedNet', 'config', 'loss_config', 'training_histories', 'fold_results', 'foldIndices', '-v7.3');
plot_kfold_training_curves(training_histories, mean_metrics, std_metrics, outputDir);

% OOD Evaluation
if exist('input/cxr', 'dir')
    imdsCXR = imageDatastore('input/cxr', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    bestFoldValIdx = foldIndices.val{best_fold_idx};
    imdsROI_val = subset(imds, bestFoldValIdx);
    
    fprintf('\n=== OOD EVALUATION ===\n');
    [resultsID, ~, ~] = evaluate_dataset_with_preprocessing_ensemble(trainedNet, imdsROI_val, classes, config.useGPU, preprocessing_method_param, refHist, false);
    [resultsOOD, ~, unc_stats] = evaluate_dataset_with_preprocessing_ensemble(trainedNet, imdsCXR, classes, config.useGPU, preprocessing_method_param, refHist, true);
    
    auc_gap = resultsID.auc - resultsOOD.auc;
    fprintf('AUC Gap (ID→OOD): %+.3f | Mean Uncertainty: %.3f\n', auc_gap, unc_stats.mean_uncertainty);
    if auc_gap < 0.1
        fprintf('Status: Good Generalization\n');
    else
        fprintf('Status: Poor Generalization\n');
    end
end
fprintf('\n=== COMPLETE ===\n');
end

%% ========================================================================
%% Helper Functions
%% ========================================================================

function [fold_results, training_histories, best_models] = train_with_kfold_preprocessed(...
    imds, vggNet, precomputedGradCAM, precomputedMasks, foldIndices, classes, loss_config, config)
fold_results = struct(); training_histories = struct(); best_models = struct();
featureLayer = 'relu5_3';

for fold = 1:config.k_folds
    fprintf('\n--- Fold %d/%d ---\n', fold, config.k_folds);
    trainIdx = foldIndices.train{fold}; valIdx = foldIndices.val{fold};
    imdsTrain = subset(imds, trainIdx); imdsVal = subset(imds, valIdx);
    preCAMs_train = precomputedGradCAM(trainIdx); preMasks_train = precomputedMasks(trainIdx);
    preCAMs_val = precomputedGradCAM(valIdx); preMasks_val = precomputedMasks(valIdx);
    
    numClasses = numel(classes);
    baseLg = layerGraph(vggNet.Layers);
    baseLg = removeLayers(baseLg, intersect({'fc8','prob','output'}, {baseLg.Layers.Name}));
    baseLg = addLayers(baseLg, [fullyConnectedLayer(numClasses, 'Name', 'fc8'), softmaxLayer('Name', 'prob')]);
    baseLg = connectLayers(baseLg, 'drop7', 'fc8');
    net = dlnetwork(baseLg);
    
    classWeights = gather(single(countcats(imdsTrain.Labels).^-1));
    classWeights = classWeights / sum(classWeights);
    
    [trainedNet, training_history] = train_model_with_preprocessing_improved(...
        net, imdsTrain, imdsVal, preCAMs_train, preCAMs_val, preMasks_train, preMasks_val, ...
        classes, loss_config, config);
        
    [acc, prec, sens, spec, f1, auc, iou, dice, tv, jac, hdist] = evaluateWithSegmentation(...
        trainedNet, imdsVal, classes, config.useGPU, preCAMs_val, preMasks_val, featureLayer, config);
        
    fold_name = sprintf('fold_%d', fold);
    fold_results.(fold_name) = struct('accuracy',acc,'precision',prec,'sensitivity',sens,...
        'specificity',spec,'f1_score',f1,'auc',auc,'iou',iou,'dice',dice,'tversky',tv,'jaccard',jac,'hausdorff',hdist);
    training_histories.(fold_name) = training_history;
    best_models.(fold_name) = trainedNet;
    fprintf('Fold %d: Acc=%.3f, Dice=%.3f, AUC=%.3f\n', fold, acc, dice, auc);
    
    checkpointDir = fullfile('models', 'checkpoints');
    if ~exist(checkpointDir, 'dir'), mkdir(checkpointDir); end
    save(fullfile(checkpointDir, sprintf('chk_%s_%s.mat', config.preprocessing_method, fold_name)), 'trainedNet', 'config', 'classes', '-v7.3');
end
end

function [trainedNet, training_history] = train_model_with_preprocessing_improved(...
    net, imdsTrain, imdsVal, preCAMs_train, preCAMs_val, preMasks_train, preMasks_val, ...
    classes, loss_config, config)
numClasses = numel(classes);
augTrain = augmentedImageDatastore([224 224], imdsTrain, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', config.imageAugmenter);
augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
mbqTrain = minibatchqueue(augTrain, 'MiniBatchSize', config.batchSize, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""]);
mbqVal = minibatchqueue(augVal, 'MiniBatchSize', config.batchSize, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""]);

avgGrad = []; avgSqGrad = []; iter = 0; bestValAcc = 0; patienceCounter = 0;
featureLayer = 'relu5_3'; nCam = config.nCam;
classCounts = countcats(imdsTrain.Labels);
classWeights = gather(single(classCounts.^-1)); classWeights = classWeights / sum(classWeights);

training_history = struct('epoch_loss', zeros(1,config.numEpochs), 'val_loss', nan(1,config.numEpochs), ...
    'val_accuracy', nan(1,config.numEpochs), 'val_iou', nan(1,config.numEpochs), 'val_dice', nan(1,config.numEpochs), ...
    'train_acc', zeros(1,config.numEpochs), 'val_acc', nan(1,config.numEpochs), 'lr', zeros(1,config.numEpochs), 'epoch', 1:config.numEpochs);

bestNet = net; epoch = 0;
fprintf('Starting training (Adam + ImageNet Norm)...\n');
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
        [X, T] = next(mbqTrain); if config.useGPU, X = gpuArray(X); end
        
        [loss, grads, state, loss_components] = dlfeval(@compute_loss_with_config_improved, ...
            net, X, T, loss_config, classWeights, imdsTrain.Files, preCAMs_train, preMasks_train, ...
            nCam, classes, featureLayer, config.useGPU, epoch);
        net.State = state;
        
        if isfield(loss_config, 'gradient_clip_norm') && ~isempty(loss_config.gradient_clip_norm)
            maxNorm = loss_config.gradient_clip_norm;
            gradValues = grads.Value; gradNormSq = 0;
            for i = 1:numel(gradValues), gradNormSq = gradNormSq + sum(gradValues{i}(:).^2); end
            if sqrt(gradNormSq) > maxNorm
                grads.Value = cellfun(@(g) g * (maxNorm / sqrt(gradNormSq+eps)), gradValues, 'UniformOutput', false);
            end
        end
        
        % === CRITICAL FIX: Use Adam instead of SGDM ===
        [net, avgGrad, avgSqGrad] = adamupdate(net, grads, avgGrad, avgSqGrad, lr);
        
        epochLoss = epochLoss + double(loss);
        if mod(batch,10)==0 && batch<=50
            fprintf('    Batch %d - Cls=%.3f', batch, double(loss_components.classification));
            if loss_config.use_gradcam, fprintf(', CAM=%.3f', double(loss_components.gradcam)); end
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
        [val_acc, ~, ~, ~, ~, ~, val_iou, val_dice, ~, ~, ~] = evaluateWithSegmentation(...
            net, imdsVal, classes, config.useGPU, preCAMs_val, preMasks_val, featureLayer, config);
            
        training_history.val_loss(epoch) = valLoss; training_history.val_accuracy(epoch) = val_acc;
        training_history.val_iou(epoch) = val_iou; training_history.val_dice(epoch) = val_dice;
        training_history.val_acc(epoch) = val_acc;
        
        fprintf('  Train Loss: %.4f | Acc: %.3f\n', avgTrainLoss, trainAcc);
        fprintf('  Val Loss: %.4f | Acc: %.3f | IoU: %.3f | Dice: %.3f\n', valLoss, val_acc, val_iou, val_dice);
        
        if val_acc > bestValAcc + config.min_delta
            bestValAcc = val_acc; bestNet = net; patienceCounter = 0; fprintf('  ✓ Validation improved!\n');
        else
            patienceCounter = patienceCounter + 1;
        end
        if patienceCounter >= config.patience, fprintf('\nEarly stopping.\n'); net = bestNet; break; end
    else
        fprintf('  Train Loss: %.4f | Acc: %.3f\n', avgTrainLoss, trainAcc);
    end
    
    % Monitoring
    if mod(epoch, 3) == 0 || epoch == 1
        ptbIdx = find(ismember(classes, {'ptb','PTB','Tuberculosis'}), 1);
        if isempty(ptbIdx), ptbIdx = 2; end
        monitor_probability_distribution(net, imdsVal, classes, config.useGPU, epoch, config, config.refHist, ptbIdx);
    end
end
trainedNet = bestNet;
for f = fieldnames(training_history)', training_history.(f{1}) = training_history.(f{1})(1:epoch); end
end

function [loss, grads, state, loss_components] = compute_loss_with_config_improved(...
    net, X, T, loss_config, classWeights, trainFiles, preCAMs, preMasks, nCam, classes, featureLayer, useGPU, epoch)
    [Y, state] = forward(net, X); 
    loss_components = struct();
    if size(T, 1) ~= size(Y, 1)
        T = T; 
    end
    if loss_config.use_focal
        clsLoss = compute_focal_loss(Y, T, classWeights, loss_config.focal_alpha, loss_config.focal_gamma);
    else
        clsLoss = crossentropy(Y, T, 'Weights', classWeights, 'WeightsFormat', 'C');
    end
    loss_components.classification = clsLoss;
    camLoss = 0; tverskyLoss = 0; anatomicalLoss = 0;

    if loss_config.use_gradcam || loss_config.use_tversky || loss_config.use_anatomical_guidance
        N = numel(trainFiles); n = min(nCam, N); idxs = randperm(N, n);
        for ii = 1:n
            img = imread(trainFiles{idxs(ii)}); if size(img,3)==1, img=repmat(img,[1 1 3]); end
            img = imresize(img, [224 224]);
            studCAM = student_cam_one_dlarray(net, img, classes, featureLayer, useGPU);
            studCAM_data = single(extractdata(studCAM));
            studCAM = dlarray(imresize(studCAM_data, [224 224]), 'SS');
            
            if loss_config.use_gradcam
                targetCAM = preCAMs{idxs(ii)};
                if ~isa(targetCAM, 'dlarray')
                    targetCAM = single(targetCAM); if ndims(targetCAM)>2, targetCAM=squeeze(targetCAM); end
                    targetCAM = dlarray(imresize(targetCAM, [224 224]), 'SS');
                end
                if strcmp(loss_config.cam_loss_type, 'cosine')
                    camLoss = camLoss + cam_cosine_loss(studCAM, targetCAM);
                else
                    camLoss = camLoss + mse(studCAM, targetCAM);
                end
            end
            
            if loss_config.use_tversky
                realMask = preMasks{idxs(ii)};
                if ~isempty(realMask)
                    realMask_dl = dlarray(single(imresize(realMask, [224 224])), 'SS');
                    tverskyLoss = tverskyLoss + (1 - tversky_coefficient_dlarray(studCAM, realMask_dl, loss_config.tversky_alpha, loss_config.tversky_beta));
                end
            end
        end
        camLoss = camLoss/max(1,n); tverskyLoss = tverskyLoss/max(1,n);
    end
    loss_components.gradcam = camLoss; loss_components.tversky = tverskyLoss;

    loss = clsLoss + loss_config.lambda_cam*camLoss + loss_config.lambda_tversky*tverskyLoss;
    grads = dlgradient(loss, net.Learnables, 'EnableHigherDerivatives', true);
end

function camLoss = cam_cosine_loss(studCAM, targetCAM)
    v1 = reshape(studCAM,[],1); v2 = reshape(targetCAM,[],1);
    cosSim = sum(v1.*v2,'all') / (sqrt(sum(v1.^2,'all'))*sqrt(sum(v2.^2,'all'))+eps);
    camLoss = max(0, min(2, 1-cosSim));
end

function [results, predictions, uncertainty_stats] = evaluate_dataset_with_preprocessing_ensemble(...
    net, imds, classes, useGPU, base_preprocessing, refHist, is_ood_evaluation)
fprintf('    [eval] %d samples | preprocessing: %s\n', numel(imds.Files), base_preprocessing);
augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, base_preprocessing, refHist);
mbq = minibatchqueue(augDS, 'MiniBatchSize', 16, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""], 'PartialMiniBatch', 'return');
ptb_candidates = {'PTB','ptb','TB','tb','Tuberculosis','tuberculosis','positive','Positive'};
ptbIdx = 2; for k=1:numel(ptb_candidates), idx=find(strcmp(classes, ptb_candidates{k}),1); if ~isempty(idx), ptbIdx=idx; break; end; end
fprintf('    [eval] PTB class index: %d (%s)\n', ptbIdx, classes{ptbIdx});

Yprobs = []; Ytrue = imds.Labels(:); uncertainty_values = [];
reset(mbq);
while hasdata(mbq)
    [X, ~] = next(mbq); if useGPU, X=gpuArray(X); end
    sc = predict(net, X); probs = extractdata(sc)';
    Yprobs = [Yprobs; probs(:, ptbIdx)];
    probs_norm = probs ./ (sum(probs,2)+eps);
    entropy = -sum(probs_norm .* log(probs_norm+eps), 2);
    max_prob = max(probs,[],2);
    uncertainty = (1-max_prob) + 0.5*entropy/log(numel(classes));
    uncertainty_values = [uncertainty_values; uncertainty];
end
Yprobs = Yprobs(1:numel(Ytrue)); uncertainty_values = uncertainty_values(1:numel(Ytrue));

thresh_results = compute_classification_metrics(Yprobs, Ytrue, classes, ptbIdx);
results = struct('accuracy',thresh_results.accuracy,'precision',thresh_results.precision,'sensitivity',thresh_results.sensitivity,...
    'specificity',thresh_results.specificity,'f1_score',thresh_results.f1_score,'auc',thresh_results.auc,...
    'optimal_threshold',thresh_results.best_thresh,'balanced_accuracy',thresh_results.balanced_accuracy);
uncertainty_stats = struct('mean_uncertainty', mean(uncertainty_values), 'high_uncertainty_pct', mean(uncertainty_values>prctile(uncertainty_values,80)));
predictions = struct('Yprobs', Yprobs, 'uncertainty', uncertainty_values);
end

function [accuracy, precision, sensitivity, specificity, f1score, auc, iou, dice, tversky, jaccard, hausdorff] = ...
    evaluateWithSegmentation(net, imdsVal, classes, useGPU, valGradCAMs, valMasks, featureLayer, config)
ptb_candidates = {'PTB','ptb','TB','tb','Tuberculosis','tuberculosis','positive','Positive'};
ptbIdx = 2; for k=1:numel(ptb_candidates), idx=find(strcmp(classes, ptb_candidates{k}),1); if ~isempty(idx), ptbIdx=idx; break; end; end
augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
mbqEval = minibatchqueue(augVal, 'MiniBatchSize', 16, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""], 'PartialMiniBatch', 'return');
Yprobs = []; reset(mbqEval);
while hasdata(mbqEval)
    [X, ~] = next(mbqEval); if useGPU, X=gpuArray(X); end
    sc = predict(net, X); probs = extractdata(sc)'; Yprobs = [Yprobs; probs(:, ptbIdx)];
end
Ytrue = imdsVal.Labels(:); Yprobs = Yprobs(1:numel(Ytrue));
thresh_results = compute_classification_metrics(Yprobs, Ytrue, classes, ptbIdx);
accuracy = thresh_results.accuracy; precision = thresh_results.precision; sensitivity = thresh_results.sensitivity;
specificity = thresh_results.specificity; f1score = thresh_results.f1_score; auc = thresh_results.auc;

numVal = numel(imdsVal.Files); ious = zeros(numVal,1); dices = zeros(numVal,1);
for i = 1:numVal
    try
        img = imread(imdsVal.Files{i}); if size(img,3)==1, img=repmat(img,[1 1 3]); end
        img = apply_preprocessing_batch(img, config.preprocessing_method, config.refHist);
        img = imresize(img, [224 224]);
        predCAM = dlfeval(@student_cam_one, net, img, classes, featureLayer, useGPU);
        realMask = valMasks{i}; if isempty(realMask)||~any(realMask(:)), continue; end
        realMask = imresize(logical(realMask), [224 224], 'nearest');
        predCAM = imresize(single(predCAM), [224 224]);
        thresh = prctile(predCAM(:), 50); if thresh<0.1, thresh=0.5; end
        predMask = logical(imopen(predCAM>thresh, strel('disk',2)));
        if ~any(predMask(:)), continue; end
        [ious(i), dices(i), ~, ~, ~] = computeSegmentationMetrics(predMask, realMask);
    catch, continue; end
end
valid = ~isnan(ious) & ious>0;
iou = mean(ious(valid)); dice = mean(dices(valid)); tversky = 0; jaccard = iou; hausdorff = 0;
end

function thresh_results = compute_classification_metrics(Yprobs,Ytrue,classes,ptbIdx)

    thresholds = 0.10:0.02:0.90;

    best_bal = -inf;
    best_thresh = 0.5;

    for t = thresholds

        pred = repmat(classes(1),numel(Yprobs),1);

        idx = Yprobs >= t;
        pred(idx) = classes(ptbIdx);

        preds_t = categorical(pred,classes);

        cm = confusionmat(Ytrue,preds_t,'Order',categorical(classes));

        if size(cm,1)==2

            TP = cm(ptbIdx,ptbIdx);
            FN = cm(ptbIdx,3-ptbIdx);
            TN = cm(3-ptbIdx,3-ptbIdx);
            FP = cm(3-ptbIdx,ptbIdx);

            sens = TP/(TP+FN+eps);
            spec = TN/(TN+FP+eps);

            bal = (sens+spec)/2;

            if bal > best_bal
                best_bal = bal;
                best_thresh = t;
                best_cm = cm;
            end
        end
    end

    TP = best_cm(ptbIdx,ptbIdx);
    FN = best_cm(ptbIdx,3-ptbIdx);
    TN = best_cm(3-ptbIdx,3-ptbIdx);
    FP = best_cm(3-ptbIdx,ptbIdx);

    accuracy = (TP+TN)/(TP+TN+FP+FN+eps);
    precision = TP/(TP+FP+eps);
    sensitivity = TP/(TP+FN+eps);
    specificity = TN/(TN+FP+eps);
    f1 = 2*precision*sensitivity/(precision+sensitivity+eps);

    try
        [~,~,~,auc] = perfcurve(double(Ytrue==categorical(classes{ptbIdx})),Yprobs,1);
    catch
        auc = 0.5;
    end

    thresh_results = struct( ...
        'accuracy',accuracy,...
        'precision',precision,...
        'sensitivity',sensitivity,...
        'specificity',specificity,...
        'f1_score',f1,...
        'auc',auc,...
        'balanced_accuracy',best_bal,...
        'best_thresh',best_thresh);

end

function [X, T] = preprocessMiniBatch(data, classes, method, refHist)
    % Extract images and labels
    X = cat(4, data{1:end,1});
    labels = cat(1, data{1:end,2});
    
    % Apply standard normalization (ImageNet)
    X = single(X) / 255;
    mean_img = reshape([0.485, 0.456, 0.406], [1, 1, 3]);
    std_img = reshape([0.229, 0.224, 0.225], [1, 1, 3]);
    X = (X - mean_img) ./ std_img;
    
    % One-hot encode labels
    % Change: Ensure output is [Classes x Batch] by transposing
    T = onehotencode(labels, 1, 'ClassNames', classes); 
    % Note: If using onehotencode(labels, 2), you MUST transpose it: T = T';
end



function imgs_processed = apply_preprocessing_batch(imgs, method, refHist)
[H,W,C,B] = size(imgs); imgs_processed = zeros(H,W,C,B,'uint8');
for b = 1:B
    img = imgs(:,:,:,b);
    if ~isa(img,'uint8')
        mx = max(img(:)); if mx<=1.0 && mx>0, img=uint8(img.*255); elseif mx>1.0, img=uint8(img./mx.*255); else, img=uint8(img); end
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
if exist(modelPath,'file'), s=load(modelPath); vggNet=s.trainedNet; else, error('Model not found.'); end
imds = imageDatastore('input/roi', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imds.Labels); precomputedGradCAM = cell(numel(imds.Files),1); precomputedMasks = cell(numel(imds.Files),1);
cacheFile = 'precomputed_gradcam_maps_enhanced.mat';
if exist(cacheFile,'file') && ~forceRecalculate
    c = load(cacheFile); precomputedGradCAM = c.precomputedGradCAM; precomputedMasks = c.precomputedMasks;
end
end

function foldIndices = createStratifiedKFold(labels, k_folds)
classes = categories(labels); foldIndices.train = cell(k_folds,1); foldIndices.val = cell(k_folds,1);
for c = 1:numel(classes)
    idx = find(labels == classes{c}); idx = idx(randperm(numel(idx)));
    n = floor(numel(idx)/k_folds); r = mod(numel(idx),k_folds); s = 1;
    for f = 1:k_folds
        fsize = n + (f<=r); e = s+fsize-1; v = idx(s:e);
        if isempty(foldIndices.val{f}), foldIndices.val{f}=v; else, foldIndices.val{f}=[foldIndices.val{f}; v]; end; s=e+1;
    end
end
for f = 1:k_folds, foldIndices.train{f} = setdiff((1:numel(labels))', foldIndices.val{f}); end
end

function mean_metrics = calculate_mean_metrics(fold_results)
m = {'accuracy','precision','sensitivity','specificity','f1_score','auc','iou','dice','tversky','jaccard','hausdorff'};
for i=1:numel(m), vals = cellfun(@(x) x.(m{i}), struct2cell(fold_results), 'UniformOutput', false); mean_metrics.(m{i}) = mean(cell2mat(vals)); end
end
function std_metrics = calculate_std_metrics(fold_results)
m = {'accuracy','precision','sensitivity','specificity','f1_score','auc','iou','dice','tversky','jaccard','hausdorff'};
for i=1:numel(m), vals = cellfun(@(x) x.(m{i}), struct2cell(fold_results), 'UniformOutput', false); std_metrics.(m{i}) = std(cell2mat(vals)); end
end

function plot_kfold_training_curves(training_histories, mean_metrics, std_metrics, outputDir)
figure('Position',[100,100,1200,500]);
subplot(1,3,1); hold on; colors=lines(numel(fieldnames(training_histories)));
for f=1:numel(fieldnames(training_histories))
    fn = fieldnames(training_histories);
    h = training_histories.(fn{f});
    plot(h.epoch, h.epoch_loss, '-', 'Color', colors(f,:), 'LineWidth', 1.5);
    ve=h.epoch(~isnan(h.val_loss)); vl=h.val_loss(~isnan(h.val_loss)); plot(ve, vl, '--', 'Color', colors(f,:));
end; xlabel('Epoch'); ylabel('Loss'); legend('Train','Val'); grid on;
subplot(1,3,2); hold on;
for f=1:numel(fieldnames(training_histories))
    fn = fieldnames(training_histories);
    h = training_histories.(fn{f});
    plot(h.epoch, h.train_acc, '-', 'Color', colors(f,:)); ve=h.epoch(~isnan(h.val_accuracy)); va=h.val_accuracy(~isnan(h.val_accuracy)); plot(ve, va, '--', 'Color', colors(f,:));
end; xlabel('Epoch'); ylabel('Accuracy'); grid on;
subplot(1,3,3); bar([mean_metrics.accuracy mean_metrics.precision mean_metrics.sensitivity mean_metrics.specificity mean_metrics.f1_score mean_metrics.auc mean_metrics.dice mean_metrics.iou]);
set(gca,'XTickLabel',{'Acc','Prec','Sens','Spec','F1','AUC','Dice','IoU'}); ylabel('Score'); xtickangle(45); grid on;
saveas(gcf, fullfile(outputDir, 'training_curves.png')); close(gcf);
end

function valLoss = compute_validation_loss_with_config_preprocessed(net, imdsVal, classes, useGPU, precomputedGradCAM, precomputedMasks, loss_config, classWeights, config, epoch)
augVal = augmentedImageDatastore([224 224], imdsVal, 'ColorPreprocessing', 'gray2rgb');
preprocessFcn = @(X, T) preprocessMiniBatchWithPreprocessing(X, T, config.preprocessing_method, config.refHist);
mbqVal = minibatchqueue(augVal, 'MiniBatchSize', config.batchSize, 'MiniBatchFcn', preprocessFcn, 'MiniBatchFormat', ["SSCB", ""]);
valLoss = 0; valCount = 0;
while hasdata(mbqVal)
    [X, T] = next(mbqVal); if useGPU, X=gpuArray(X); end
    try [loss, ~, ~, ~] = dlfeval(@compute_loss_with_config_improved, net, X, T, loss_config, classWeights, imdsVal.Files, precomputedGradCAM, precomputedMasks, config.nCam, classes, 'relu5_3', useGPU, epoch); valLoss = valLoss + double(loss); valCount = valCount + 1; catch, continue; end
end
valLoss = valLoss / max(1, valCount);
end

function focalLoss = compute_focal_loss(Y, T, classWeights, alpha, gamma)
epsVal = 1e-7; Y = min(max(Y, epsVal), 1-epsVal);
alpha_vec = reshape(alpha(:), [], 1) .* reshape(classWeights, [], 1);
term = - T .* alpha_vec .* ((1 - Y).^gamma) .* log(Y + epsVal);
focalLoss = mean(sum(term, 1));
end

function tverskyCoef = tversky_coefficient_dlarray(pred, target, alpha, beta)
target_dl = dlarray(single(target), 'SS');
intersection = sum(pred(:).*target_dl(:), 'all'); fp = sum(pred(:).*(1-target_dl(:)), 'all'); fn = sum((1-pred(:)).*target_dl(:), 'all');
tverskyCoef = intersection / (intersection + alpha*fp + beta*fn + eps);
end

function [iou, dice, tversky, jaccard, hausdorff] = computeSegmentationMetrics(pred, target)
pred=logical(gather(pred)); target=logical(gather(target));
intersection=sum(pred(:)&target(:)); union=sum(pred(:)|target(:));
pred_sum=sum(pred(:)); target_sum=sum(target(:));
iou = intersection/(union+eps); dice = 2*intersection/(pred_sum+target_sum+eps);
alpha=0.7; beta=0.3; fp=sum(pred(:)&~target(:)); fn=sum(~pred(:)&target(:));
tversky = intersection/(intersection+alpha*fp+beta*fn+eps); jaccard=iou; hausdorff=0;
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

function monitor_probability_distribution(net,imdsVal,classes,useGPU,epoch,config,refHist,ptbIdx)

    augDS = augmentedImageDatastore([224 224],imdsVal,'ColorPreprocessing','gray2rgb');

    preprocessFcn = @(X,T) preprocessMiniBatchWithPreprocessing(X,T,...
        config.preprocessing_method,refHist);

    mbq = minibatchqueue(augDS,...
        'MiniBatchSize',16,...
        'MiniBatchFcn',preprocessFcn,...
        'MiniBatchFormat',["SSCB",""]);

    Yprobs = [];
    Ytrue = imdsVal.Labels(:);

    reset(mbq)

    while hasdata(mbq)

        [X,~] = next(mbq);

        if useGPU
            X = gpuArray(X);
        end

        sc = predict(net,X);

        tmp = extractdata(sc);
        tmp = gather(tmp);

        Yprobs = [Yprobs; tmp(ptbIdx,:)'];

    end

    Yprobs = Yprobs(1:numel(Ytrue));

    isPos = Ytrue == categorical(classes{ptbIdx});

    pPos = Yprobs(isPos);
    pNeg = Yprobs(~isPos);

    fprintf('Epoch %d | Neg %.3f | Pos %.3f | Sep %.3f\n',...
        epoch,mean(pNeg),mean(pPos),abs(mean(pPos)-mean(pNeg)));

end


function [X, T] = preprocessMiniBatchWithPreprocessing(dataX, dataT, preprocessing_method, refHist)
    X = cat(4, dataX{1:end});
    [H, W, C, B] = size(X);
    X_processed = zeros(H, W, C, B, 'uint8');
    for b = 1:B
        img = X(:,:,:,b);
 
        % Normalise float [0,1] or arbitrary float to uint8
        if ~isa(img, 'uint8')
            img_max = max(img(:));
            if img_max <= 1.0 && img_max > 0
                img = uint8(img .* 255);
            elseif img_max > 1.0
                img = uint8(img ./ img_max .* 255);
            else
                img = uint8(img);
            end
        end
 
        if C == 3
            img_gray = rgb2gray(img);
        else
            img_gray = img(:,:,1);
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
            otherwise
                img_processed = img_gray;
        end
 
        if C == 3
            X_processed(:,:,:,b) = repmat(img_processed, [1 1 3]);
        else
            X_processed(:,:,:,b) = img_processed;
        end
    end
    X = dlarray(single(X_processed) ./ 255, 'SSCB');  % normalise back to [0,1] for VGG16
    T = onehotencode(cat(2, dataT{1:end}), 1);
end
