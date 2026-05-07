function evaluate_ood_with_preprocessing(useTTA, numAugmentations)
%EVALUATE_OOD_WITH_PREPROCESSING Evaluate OOD performance with intensity preprocessing and TTA
%   This script tests different preprocessing methods to improve OOD performance:
%   - No preprocessing (baseline)
%   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
%   - Histogram matching
%   - Z-score normalization
%   - Min-max normalization
%
%   Optional TTA (Test-Time Augmentation) can be enabled to further improve robustness.
%
%   Inputs:
%   useTTA - (optional) Boolean, enable TTA (default: true)
%   numAugmentations - (optional) Number of augmentations per image for TTA (default: 12)
%
%   Usage Examples:
%   % Default: TTA enabled with 12 augmentations
%   evaluate_ood_with_preprocessing()
%
%   % Custom TTA settings
%   evaluate_ood_with_preprocessing(true, 16)  % 16 augmentations
%
%   % Disable TTA
%   evaluate_ood_with_preprocessing(false)
%
%   Compares results to identify best preprocessing for OOD generalization.

if nargin < 1
    useTTA = true;  % Default: enable TTA
end
if nargin < 2
    numAugmentations = 12;  % Default: 12 augmentations per image
end

% Save input parameters before clearing
useTTA_param = useTTA;
numAugmentations_param = numAugmentations;

clc; close all;  % Don't use 'clear' as it clears function input variables
fprintf('=== OOD EVALUATION WITH PREPROCESSING ===\n');
if useTTA_param
    fprintf('TTA Enabled: %d augmentations per image\n', numAugmentations_param);
end
fprintf('Testing different preprocessing methods to improve OOD performance...\n\n');

%% Configuration
modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');
if ~exist(modelFile, 'file')
    % Fallback to alternative models
    modelFile = fullfile('models', 'final', 'final_model_focal_gradcam_tversky_kfold.mat');
    if ~exist(modelFile, 'file')
        modelFile = 'vgg16_multitask_trained_optimal.mat';
        if ~exist(modelFile, 'file')
            error('Model file not found. Please ensure model is trained first.');
        end
    end
end

% Data paths
roiDir = fullfile('input', 'roi');
cxrDir = fullfile('input', 'cxr');

% Output directory
outputDir = fullfile('results', 'ood_preprocessing');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Load Model
fprintf('Loading model: %s\n', modelFile);
s = load(modelFile);
trainedNet = s.trainedNet;
if isfield(s, 'config')
    config = s.config;
    useGPU = config.useGPU;
else
    useGPU = canUseGPU;
end
fprintf('Model loaded. GPU: %s\n\n', string(useGPU));

%% Load Datasets
fprintf('Loading datasets...\n');
imdsROI = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imdsROI.Labels);

fprintf('  ROI (ID): %d samples\n', numel(imdsROI.Files));
fprintf('  CXR (OOD): %d samples\n', numel(imdsCXR.Files));
fprintf('  Classes: %s\n\n', strjoin(classes, ', '));

%% Compute Reference Histogram (Target Domain)
% For 'histmatch', we must align all images (ID and OOD) to the Target Domain (CXR)
% This matches the training strategy.
fprintf('Computing reference histogram from Target Domain (CXR)...\n');
numRefSamples = min(50, numel(imdsCXR.Files));
refIdx = randperm(numel(imdsCXR.Files), numRefSamples);
allPixels = [];
for i = 1:numRefSamples
    img = imread(imdsCXR.Files{refIdx(i)});
    if size(img, 3) > 1, img = rgb2gray(img); end
    allPixels = [allPixels; img(:)];
end
targetRefHist = imhist(uint8(allPixels));
fprintf('  Reference histogram computed from %d CXR samples.\n\n', numRefSamples);

%% Define Preprocessing Methods
preprocessing_methods = {
    'none', 'No preprocessing (baseline)';
    'clahe', 'CLAHE (Contrast Limited Adaptive Histogram Equalization)';
    'histmatch', 'Histogram matching to CXR reference';
    'zscore', 'Z-score normalization';
    'minmax', 'Min-max normalization [0, 255]';
    'clahe_zscore', 'CLAHE + Z-score normalization';
};

numMethods = size(preprocessing_methods, 1);
results = struct();

%% Evaluate Each Preprocessing Method
fprintf('=== EVALUATING PREPROCESSING METHODS ===\n\n');

for m = 1:numMethods
    method_name = preprocessing_methods{m, 1};
    method_desc = preprocessing_methods{m, 2};
    
    fprintf('--- Method %d/%d: %s ---\n', m, numMethods, method_desc);
    
    % Evaluate ID (ROI)
    fprintf('  Evaluating ID (ROI)...\n');
    if useTTA_param
        [resultsID, ~] = evaluate_with_preprocessing_tta(trainedNet, imdsROI, classes, useGPU, method_name, numAugmentations_param, targetRefHist);
    else
        [resultsID, ~] = evaluate_with_preprocessing(trainedNet, imdsROI, classes, useGPU, method_name, targetRefHist);
    end
    
    % Evaluate OOD (Full CXR)
    fprintf('  Evaluating OOD (Full CXR)...\n');
    if useTTA_param
        [resultsOOD, ~] = evaluate_with_preprocessing_tta(trainedNet, imdsCXR, classes, useGPU, method_name, numAugmentations_param, targetRefHist);
    else
        [resultsOOD, ~] = evaluate_with_preprocessing(trainedNet, imdsCXR, classes, useGPU, method_name, targetRefHist);
    end
    
    % Calculate degradation
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
    
    % Store results
    results.(method_name) = struct();
    results.(method_name).description = method_desc;
    results.(method_name).id = resultsID;
    results.(method_name).ood = resultsOOD;
    results.(method_name).degradation = degradation;
    results.(method_name).mean_degradation = mean_degradation;
    
    fprintf('  ID Accuracy: %.3f, OOD Accuracy: %.3f\n', resultsID.accuracy, resultsOOD.accuracy);
    fprintf('  Mean Degradation: %.2f%%\n\n', mean_degradation);
end

%% Compare Results
fprintf('\n=== PREPROCESSING COMPARISON RESULTS ===\n\n');
fprintf('%-20s | ID Acc | OOD Acc | Degradation\n', 'Method');
fprintf('%s\n', repmat('-', 1, 60));

method_names = preprocessing_methods(:, 1);
for m = 1:numMethods
    method_name = method_names{m};
    r = results.(method_name);
    fprintf('%-20s | %.3f  | %.3f   | %.2f%%\n', ...
        r.description, r.id.accuracy, r.ood.accuracy, r.mean_degradation);
end

%% Find Best Method
mean_degradations = zeros(numMethods, 1);
for m = 1:numMethods
    mean_degradations(m) = results.(method_names{m}).mean_degradation;
end
[~, best_idx] = min(mean_degradations);
best_method = method_names{best_idx};

fprintf('\n=== BEST PREPROCESSING METHOD ===\n');
fprintf('Method: %s\n', results.(best_method).description);
fprintf('ID Accuracy: %.3f\n', results.(best_method).id.accuracy);
fprintf('OOD Accuracy: %.3f\n', results.(best_method).ood.accuracy);
fprintf('Mean Degradation: %.2f%%\n', results.(best_method).mean_degradation);
fprintf('Improvement over baseline: %.2f%%\n', ...
    results.none.mean_degradation - results.(best_method).mean_degradation);

%% Create Comparison Visualizations
fprintf('\n=== GENERATING COMPARISON VISUALIZATIONS ===\n');
create_preprocessing_comparison_plots(results, preprocessing_methods, outputDir, useTTA_param, numAugmentations_param);

%% Save Results
save(fullfile(outputDir, 'preprocessing_comparison_results.mat'), 'results', 'preprocessing_methods', 'useTTA_param', 'numAugmentations_param', '-v7.3');
fprintf('\nResults saved to: %s\n', fullfile(outputDir, 'preprocessing_comparison_results.mat'));

fprintf('\n=== EVALUATION COMPLETE ===\n');
if useTTA_param
    fprintf('TTA was enabled with %d augmentations per image\n', numAugmentations_param);
end
end

%% Helper Functions

function [results, predictions] = evaluate_with_preprocessing(net, imds, classes, useGPU, preprocessing_method, refHist)
% Evaluate dataset with specified preprocessing method

if nargin < 6
    refHist = [];
end

augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
reset(augDS);

Ypred = categorical.empty(0,1);
Yprobs = [];
Ytrue = imds.Labels(:);

batchCount = 0;
while hasdata(augDS)
    batchCount = batchCount + 1;
    if mod(batchCount, 20) == 0
        fprintf('    Batch %d...\n', batchCount);
    end
    
    tbl = read(augDS);
    imgs = cat(4, tbl.input{:});
    
    % Apply preprocessing
    imgs = apply_preprocessing(imgs, preprocessing_method, refHist);
    
    % Convert to single precision (matches training)
    if ~isa(imgs, 'single')
        imgs = single(imgs);
    end
    
    if useGPU, imgs = gpuArray(imgs); end
    dlX = dlarray(imgs, 'SSCB');
    
    % Forward pass
    sc = predict(net, dlX);
    
    % Extract predictions
    probs_batch = extractdata(sc);
    if ndims(probs_batch) == 4
        probs_batch = squeeze(probs_batch);
    end
    
    if size(probs_batch, 1) == numel(classes)
        [~, maxIdx] = max(probs_batch, [], 1);
        lab = categorical(classes(maxIdx));
        Ypred = [Ypred; lab(:)];
        
        if numel(classes) == 2
            Yprobs = [Yprobs; probs_batch(2,:)'];
        else
            Yprobs = [Yprobs; max(probs_batch, [], 1)'];
        end
    end
end

% Calculate metrics
match = (categorical(Ytrue) == categorical(Ypred));
accuracy = sum(match) / numel(match);

cm = confusionmat(Ytrue, Ypred, 'Order', categorical(classes));
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
    if numel(classes) == 2
        [~,~,~,auc] = perfcurve(double(Ytrue == classes{2}), Yprobs, 1);
    else
        auc = NaN;
    end
catch
    auc = 0.5;
end

results = struct();
results.accuracy = accuracy;
results.precision = precision;
results.sensitivity = sensitivity;
results.specificity = specificity;
results.f1_score = f1score;
results.auc = auc;
results.confusion_matrix = cm;

predictions = struct();
predictions.Ypred = Ypred;
predictions.Ytrue = Ytrue;
predictions.Yprobs = Yprobs;
end

function imgs_processed = apply_preprocessing(imgs, method, refHist)
% Apply preprocessing to batch of images
%   imgs: [H x W x C x B] batch of images (uint8, [0, 255])
%   method: preprocessing method name
%   refHist: reference histogram for histogram matching (optional)

[H, W, C, B] = size(imgs);
imgs_processed = zeros(size(imgs), 'uint8');

for b = 1:B
    img = imgs(:, :, :, b);
    
    % Convert to grayscale for intensity preprocessing (then replicate to RGB)
    if C == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    switch method
        case 'none'
            % No preprocessing - keep as is
            img_processed = img_gray;
            
        case 'clahe'
            % CLAHE - Contrast Limited Adaptive Histogram Equalization
            % Very common for medical X-ray images
            img_processed = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
            
        case 'histmatch'
            % Histogram matching to reference
            if ~isempty(refHist)
                img_processed = histeq(img_gray, refHist);
            else
                img_processed = img_gray;  % Fallback if no reference
            end
            
        case 'zscore'
            % Z-score normalization: (x - mean) / std
            img_double = double(img_gray);
            img_mean = mean(img_double(:));
            img_std = std(img_double(:));
            if img_std > 0
                img_normalized = (img_double - img_mean) / img_std;
                % Scale back to [0, 255] range
                img_normalized = (img_normalized - min(img_normalized(:))) / ...
                    (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
            else
                img_normalized = img_double;
            end
            img_processed = uint8(img_normalized);
            
        case 'minmax'
            % Min-max normalization to [0, 255]
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
            % Combined: CLAHE + Z-score
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
    
    % Replicate to RGB if needed (to match training format)
    if C == 3
        imgs_processed(:, :, :, b) = repmat(img_processed, [1, 1, 3]);
    else
        imgs_processed(:, :, :, b) = img_processed;
    end
end
end

function [results, predictions] = evaluate_with_preprocessing_tta(net, imds, classes, useGPU, preprocessing_method, numAugmentations, refHist)
% Evaluate dataset with specified preprocessing method and TTA
%   net - Trained network
%   imds - ImageDatastore
%   classes - Class names
%   useGPU - Whether to use GPU
%   preprocessing_method - Preprocessing method name
%   numAugmentations - Number of augmentations per image for TTA
%   refHist - Reference histogram for histmatch

if nargin < 6
    numAugmentations = 12;
end
if nargin < 7
    refHist = [];
end

fprintf('    Using TTA with %d augmentations per image (This may take a while...)\n', numAugmentations);

% Create TTA augmenter (medical-appropriate augmentations)
ttaAugmenter = imageDataAugmenter( ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandRotation', [-10 10], ...
    'RandScale', [0.9 1.1]);

Ypred = categorical.empty(0,1);
Yprobs = [];
Ytrue = imds.Labels(:);

numClasses = numel(classes);

% Process each image with TTA
for imgIdx = 1:numel(imds.Files)
    if mod(imgIdx, 50) == 0
        fprintf('    Processing image %d/%d...\n', imgIdx, numel(imds.Files));
    end
    
    % Load original image
    img = imread(imds.Files{imgIdx});
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    img = imresize(img, [224 224]);
    
    % Collect predictions from multiple augmentations
    all_probs = zeros(numClasses, numAugmentations);
    
    % Process original image (no augmentation) with preprocessing
    img_processed = apply_preprocessing_single(img, preprocessing_method, refHist);
    img_processed = single(img_processed);
    
    if useGPU
        img_dl = dlarray(gpuArray(img_processed), 'SSCB');
    else
        img_dl = dlarray(img_processed, 'SSCB');
    end
    
    try
        sc = predict(net, img_dl);
        probs_data = extractdata(sc);
        if ndims(probs_data) == 4
            probs_data = squeeze(probs_data);
        end
        if size(probs_data, 1) == numClasses
            all_probs(:, 1) = probs_data(:);
        else
            all_probs(:, 1) = ones(numClasses, 1) / numClasses;
        end
    catch
        all_probs(:, 1) = ones(numClasses, 1) / numClasses;
    end
    
    % Augmented versions
    for augIdx = 2:numAugmentations
        try
            % Apply augmentation
            img_aug = apply_augmentation(img, ttaAugmenter);
            img_aug = imresize(img_aug, [224 224]);
            
            % Apply preprocessing to augmented image
            img_aug_processed = apply_preprocessing_single(img_aug, preprocessing_method, refHist);
            img_aug_processed = single(img_aug_processed);
            
            if useGPU
                img_dl = dlarray(gpuArray(img_aug_processed), 'SSCB');
            else
                img_dl = dlarray(img_aug_processed, 'SSCB');
            end
            
            sc = predict(net, img_dl);
            probs_data = extractdata(sc);
            if ndims(probs_data) == 4
                probs_data = squeeze(probs_data);
            end
            
            if size(probs_data, 1) == numClasses
                all_probs(:, augIdx) = probs_data(:);
            else
                all_probs(:, augIdx) = all_probs(:, 1);  % Fallback to original
            end
        catch
            % If augmentation fails, use original prediction
            all_probs(:, augIdx) = all_probs(:, 1);
        end
    end
    
    % Average probabilities across all augmentations
    avg_probs = mean(all_probs, 2);
    
    % Get prediction from averaged probabilities
    [~, maxIdx] = max(avg_probs);
    lab = categorical(classes(maxIdx));
    Ypred = [Ypred; lab];
    
    % Store probability for positive class (binary classification)
    if numel(classes) == 2
        Yprobs = [Yprobs; avg_probs(2)];
    else
        Yprobs = [Yprobs; max(avg_probs)];
    end
end

% Calculate metrics (same as evaluate_with_preprocessing)
match = (categorical(Ytrue) == categorical(Ypred));
accuracy = sum(match) / numel(match);

cm = confusionmat(Ytrue, Ypred, 'Order', categorical(classes));
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
    if numel(classes) == 2
        [~,~,~,auc] = perfcurve(double(Ytrue == classes{2}), Yprobs, 1);
    else
        auc = NaN;
    end
catch
    auc = 0.5;
end

results = struct();
results.accuracy = accuracy;
results.precision = precision;
results.sensitivity = sensitivity;
results.specificity = specificity;
results.f1_score = f1score;
results.auc = auc;
results.confusion_matrix = cm;

predictions = struct();
predictions.Ypred = Ypred;
predictions.Ytrue = Ytrue;
predictions.Yprobs = Yprobs;
end

function img_processed = apply_preprocessing_single(img, method, refHist)
% Apply preprocessing to a single image
%   img: [H x W x C] single image (uint8, [0, 255])
%   method: preprocessing method name
%   refHist: reference histogram (for histogram matching)

% Convert to grayscale for intensity preprocessing
if size(img, 3) == 3
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

% Replicate to RGB if needed
if size(img, 3) == 3
    img_processed = repmat(img_processed, [1, 1, 3]);
end
end

function img_aug = apply_augmentation(img, augmenter)
%APPLY_AUGMENTATION Apply augmentation to a single image
%   Manual implementation to apply augmentation to a single image
%   This is used for Test-Time Augmentation

img_aug = img;

try
    % Get augmentation parameters from augmenter properties
    % Note: MATLAB's imageDataAugmenter properties are accessible
    
    % Translation
    if isprop(augmenter, 'RandXTranslation') && ~isempty(augmenter.RandXTranslation)
        tx_range = augmenter.RandXTranslation;
        tx = rand() * (tx_range(2) - tx_range(1)) + tx_range(1);
    else
        tx = 0;
    end
    
    if isprop(augmenter, 'RandYTranslation') && ~isempty(augmenter.RandYTranslation)
        ty_range = augmenter.RandYTranslation;
        ty = rand() * (ty_range(2) - ty_range(1)) + ty_range(1);
    else
        ty = 0;
    end
    
    if tx ~= 0 || ty ~= 0
        tform = affine2d([1 0 0; 0 1 0; tx ty 1]);
        img_aug = imwarp(img_aug, tform, 'OutputView', imref2d(size(img_aug)), 'FillValues', 0);
    end
    
    % Rotation
    if isprop(augmenter, 'RandRotation') && ~isempty(augmenter.RandRotation)
        angle_range = augmenter.RandRotation;
        angle = rand() * (angle_range(2) - angle_range(1)) + angle_range(1);
        img_aug = imrotate(img_aug, angle, 'bilinear', 'crop');
    end
    
    % Scale (apply before other transformations to maintain size)
    if isprop(augmenter, 'RandScale') && ~isempty(augmenter.RandScale)
        scale_range = augmenter.RandScale;
        scale = rand() * (scale_range(2) - scale_range(1)) + scale_range(1);
        [h, w, c] = size(img_aug);
        newH = round(h * scale);
        newW = round(w * scale);
        
        img_aug = imresize(img_aug, [newH, newW]);
        
        % Crop or pad to maintain original size
        if newH >= h && newW >= w
            % Crop from center
            startH = floor((newH - h) / 2) + 1;
            startW = floor((newW - w) / 2) + 1;
            img_aug = img_aug(startH:startH+h-1, startW:startW+w-1, :);
        elseif newH < h || newW < w
            % Pad with zeros (black padding)
            img_padded = zeros(h, w, c, class(img_aug));
            startH = floor((h - newH) / 2) + 1;
            startW = floor((w - newW) / 2) + 1;
            img_padded(startH:startH+newH-1, startW:startW+newW-1, :) = img_aug;
            img_aug = img_padded;
        end
    end
    
catch ME
    % If augmentation fails, return original image
    warning('Augmentation failed: %s. Using original image.', ME.message);
    img_aug = img;
end
end

function create_preprocessing_comparison_plots(results, preprocessing_methods, outputDir, useTTA, numAugmentations)
% Create comparison visualizations for all preprocessing methods

if nargin < 4
    useTTA = false;
end
if nargin < 5
    numAugmentations = 0;
end

figure('Position', [100, 100, 1600, 1000]);

method_names = preprocessing_methods(:, 1);
numMethods = numel(method_names);

% Extract data
id_accs = zeros(numMethods, 1);
ood_accs = zeros(numMethods, 1);
degradations = zeros(numMethods, 1);
method_labels = cell(numMethods, 1);

for m = 1:numMethods
    method_name = method_names{m};
    r = results.(method_name);
    id_accs(m) = r.id.accuracy;
    ood_accs(m) = r.ood.accuracy;
    degradations(m) = r.mean_degradation;
    method_labels{m} = r.description;
end

% 1. Accuracy comparison
subplot(2, 3, 1);
x = 1:numMethods;
width = 0.35;
bar(x - width/2, id_accs * 100, width, 'FaceColor', [0.2 0.6 0.8], 'DisplayName', 'ID (ROI)');
hold on;
bar(x + width/2, ood_accs * 100, width, 'FaceColor', [0.8 0.4 0.2], 'DisplayName', 'OOD (Full CXR)');
set(gca, 'XTickLabel', method_labels);
xtickangle(45);
ylabel('Accuracy (%)');
title('Accuracy: ID vs OOD');
legend('Location', 'best');
grid on;
ylim([0, 100]);

% 2. Degradation comparison
subplot(2, 3, 2);
[~, sortIdx] = sort(degradations);
bar(degradations(sortIdx), 'FaceColor', [0.8 0.2 0.2]);
set(gca, 'XTickLabel', method_labels(sortIdx));
xtickangle(45);
ylabel('Mean Degradation (%)');
title('Performance Degradation Comparison');
grid on;

% 3. Improvement over baseline
subplot(2, 3, 3);
baseline_degradation = degradations(1);  % 'none' method
improvements = baseline_degradation - degradations;
bar(improvements, 'FaceColor', [0.2 0.8 0.2]);
set(gca, 'XTickLabel', method_labels);
xtickangle(45);
ylabel('Improvement (%)');
title('Improvement Over Baseline (No Preprocessing)');
grid on;
hold on;
plot([0.5, numMethods+0.5], [0, 0], 'k--', 'LineWidth', 1);

% 4. OOD Accuracy by method
subplot(2, 3, 4);
[~, sortIdx] = sort(ood_accs, 'descend');
bar(ood_accs(sortIdx) * 100, 'FaceColor', [0.4 0.6 0.8]);
set(gca, 'XTickLabel', method_labels(sortIdx));
xtickangle(45);
ylabel('OOD Accuracy (%)');
title('OOD Accuracy Ranking');
grid on;

% 5. Detailed metrics comparison (best vs baseline)
subplot(2, 3, 5);
[~, bestIdx] = min(degradations);
best_method = method_names{bestIdx};
baseline_method = method_names{1};

metrics = {'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC'};
baseline_ood = [results.(baseline_method).ood.accuracy, ...
                results.(baseline_method).ood.precision, ...
                results.(baseline_method).ood.sensitivity, ...
                results.(baseline_method).ood.specificity, ...
                results.(baseline_method).ood.f1_score, ...
                results.(baseline_method).ood.auc];
best_ood = [results.(best_method).ood.accuracy, ...
            results.(best_method).ood.precision, ...
            results.(best_method).ood.sensitivity, ...
            results.(best_method).ood.specificity, ...
            results.(best_method).ood.f1_score, ...
            results.(best_method).ood.auc];

x = 1:numel(metrics);
width = 0.35;
bar(x - width/2, baseline_ood * 100, width, 'FaceColor', [0.8 0.4 0.2], 'DisplayName', 'Baseline');
hold on;
bar(x + width/2, best_ood * 100, width, 'FaceColor', [0.2 0.8 0.2], 'DisplayName', 'Best Method');
set(gca, 'XTickLabel', metrics);
xtickangle(45);
ylabel('Score (%)');
title(sprintf('OOD Metrics: Baseline vs %s', results.(best_method).description));
legend('Location', 'best');
grid on;
ylim([0, 100]);

% 6. Summary table
subplot(2, 3, 6);
axis off;
summary_text = {
    'PREPROCESSING COMPARISON SUMMARY';
    '';
    sprintf('Best Method: %s', results.(best_method).description);
    '';
    sprintf('Baseline (No Preprocessing):');
    sprintf('  OOD Accuracy: %.2f%%', baseline_ood(1) * 100);
    sprintf('  Degradation: %.2f%%', degradations(1));
    '';
    sprintf('Best Method (%s):', results.(best_method).description);
    sprintf('  OOD Accuracy: %.2f%%', best_ood(1) * 100);
    sprintf('  Degradation: %.2f%%', degradations(bestIdx));
    '';
    sprintf('Improvement:');
    sprintf('  Accuracy: +%.2f%%', (best_ood(1) - baseline_ood(1)) * 100);
    sprintf('  Degradation: -%.2f%%', degradations(1) - degradations(bestIdx));
};

text(0.1, 0.9, summary_text, 'FontSize', 11, ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

% Create title with TTA info
if useTTA
    title_str = sprintf('Preprocessing Method Comparison for OOD Performance (TTA: %d augs)', numAugmentations);
else
    title_str = 'Preprocessing Method Comparison for OOD Performance (No TTA)';
end
sgtitle(title_str, 'FontSize', 16, 'FontWeight', 'bold');

% Save figure
saveas(gcf, fullfile(outputDir, 'preprocessing_comparison.png'));
fprintf('  Saved: preprocessing_comparison.png\n');
close(gcf);
end

