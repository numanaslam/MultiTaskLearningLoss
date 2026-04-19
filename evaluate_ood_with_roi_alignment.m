function evaluate_ood_with_roi_alignment(preprocessing_method, useTTA, numAugmentations)
%EVALUATE_OOD_WITH_ROI_ALIGNMENT Evaluate OOD performance using ROI alignment (boxing method)
%   This script evaluates the trained model on OOD data by automatically cropping
%   full CXR images to lung bounding box during test time. This matches the
%   training distribution (ROI images) and should significantly reduce degradation.
%
%   The ROI alignment method:
%   1. Detects lung region in full CXR using adaptive thresholding
%   2. Extracts bounding box of lung region
%   3. Crops CXR to bounding box (with margin)
%   4. Resizes to target size and applies preprocessing
%   5. Evaluates with optional TTA
%
%   Inputs:
%       preprocessing_method - (optional) Preprocessing method:
%                              'none' | 'histmatch' | 'zscore' | 'minmax' | 'clahe'
%                              Default: 'histmatch'
%       useTTA - (optional) Boolean, enable TTA (default: true)
%       numAugmentations - (optional) Number of augmentations per image (default: 12)
%
%   Usage Examples:
%       % Default: histmatch preprocessing + TTA
%       evaluate_ood_with_roi_alignment()
%
%       % Custom preprocessing, no TTA
%       evaluate_ood_with_roi_alignment('zscore', false)
%
%       % Custom settings
%       evaluate_ood_with_roi_alignment('histmatch', true, 16)
%
%   Expected Results:
%       - Current (full CXR): ~32% degradation
%       - With ROI alignment: <15% degradation (estimated)
%       - OOD accuracy improvement: 57.5% → 75-80% (estimated)

if nargin < 1
    preprocessing_method = 'histmatch';  % Default: histogram matching
end
if nargin < 2
    useTTA = true;  % Default: enable TTA
end
if nargin < 3
    numAugmentations = 12;  % Default: 12 augmentations per image
end

% Save parameters before clearing
preprocessing_method_param = preprocessing_method;
useTTA_param = useTTA;
numAugmentations_param = numAugmentations;

clc; close all;
fprintf('=== OOD EVALUATION WITH ROI ALIGNMENT (BOXING METHOD) ===\n\n');
fprintf('Method: Automatically crop full CXR to lung bounding box during test time\n');
fprintf('Preprocessing: %s\n', preprocessing_method_param);
fprintf('TTA: %s', string(useTTA_param));
if useTTA_param
    fprintf(' (%d augmentations per image)\n', numAugmentations_param);
else
    fprintf('\n');
end
fprintf('\n');

%% Configuration
% Model file
modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');
if ~exist(modelFile, 'file')
    % Try fallback models
    fallbackModels = {
        fullfile('models', 'final', 'final_model_focal_gradcam_tversky_kfold.mat')
        'vgg16_multitask_trained_optimal.mat'
    };
    for k = 1:numel(fallbackModels)
        if exist(fallbackModels{k}, 'file')
            modelFile = fallbackModels{k};
            break;
        end
    end
    if ~exist(modelFile, 'file')
        error('Model file not found. Please train a model first.');
    end
end

% Data paths - try resized first, then fallback to original
roiDir = fullfile('input', 'roi');
cxrDir = fullfile('input', 'cxr');
maskDir = fullfile('input', 'masks');  % For ROI alignment

% Check if resized directories exist, otherwise use original
if ~exist(roiDir, 'dir')
    roiDir = fullfile('input', 'roi');
    if ~exist(roiDir, 'dir')
        error('ROI directory not found. Tried: %s and %s\nCurrent directory: %s', ...
            fullfile('input', 'roi'), fullfile('input', 'roi'), pwd);
    end
end

if ~exist(cxrDir, 'dir')
    cxrDir = fullfile('input', 'cxr');
    if ~exist(cxrDir, 'dir')
        error('CXR directory not found. Tried: %s and %s\nCurrent directory: %s', ...
            fullfile('input', 'cxr'), fullfile('input', 'cxr'), pwd);
    end
end

if ~exist(maskDir, 'dir')
    maskDir = fullfile('input', 'masks');
    if ~exist(maskDir, 'dir')
        warning('Mask directory not found. ROI alignment will use automatic detection only.');
        maskDir = [];
    end
end

% Output directory
outputDir = fullfile('results', 'ood_roi_alignment');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Load Model
fprintf('Loading model: %s\n', modelFile);
s = load(modelFile);
trainedNet = s.trainedNet;
if isfield(s, 'config') && isfield(s.config, 'useGPU')
    useGPU = s.config.useGPU;
else
    useGPU = canUseGPU;
end
fprintf('Model loaded. GPU: %s\n\n', string(useGPU));

%% Load Datasets
fprintf('Loading datasets...\n');
fprintf('  ROI directory: %s\n', roiDir);
fprintf('  CXR directory: %s\n', cxrDir);
if ~isempty(maskDir)
    fprintf('  Mask directory: %s\n', maskDir);
else
    fprintf('  Mask directory: Not available (will use automatic detection)\n');
end

imdsROI = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imdsROI.Labels);

fprintf('  ROI (ID): %d samples\n', numel(imdsROI.Files));
fprintf('  CXR (OOD): %d samples\n', numel(imdsCXR.Files));
fprintf('  Classes: %s\n\n', strjoin(classes, ', '));

if isempty(imdsROI.Files) || isempty(imdsCXR.Files)
    error('Both ROI and CXR datasets are required. Found %d ROI and %d CXR files.', ...
        numel(imdsROI.Files), numel(imdsCXR.Files));
end

%% Compute Reference Histogram (for histogram matching)
refHist = [];
if strcmp(preprocessing_method_param, 'histmatch')
    fprintf('Computing reference histogram from ROI images...\n');
    numRefSamples = min(200, numel(imdsROI.Files));
    refIdx = randperm(numel(imdsROI.Files), numRefSamples);
    refPixels = [];
    for i = 1:numRefSamples
        img = imread(imdsROI.Files{refIdx(i)});
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        refPixels = [refPixels; img(:)];
    end
    refHist = imhist(uint8(refPixels));
    refHist = refHist / sum(refHist);  % Normalize
    fprintf('  Reference histogram computed from %d samples\n\n', numRefSamples);
end

%% ROI Alignment Configuration
alignmentConfig = struct();
alignmentConfig.targetSize = [224, 224];
alignmentConfig.clipLimit = 0.01;  % For CLAHE in lung detection
alignmentConfig.adaptiveSensitivity = 0.42;  % For adaptive thresholding
alignmentConfig.minAreaRatio = 0.12;  % Minimum lung area ratio
alignmentConfig.openRadius = 5;  % Morphological opening radius
alignmentConfig.closeRadius = 25;  % Morphological closing radius
alignmentConfig.margin = 0.08;  % 8% margin around bounding box
alignmentConfig.fallbackScale = 0.72;  % Fallback center crop scale

%% TTA Configuration
if useTTA_param
    ttaAugmenter = imageDataAugmenter( ...
        'RandXTranslation', [-8 8], ...
        'RandYTranslation', [-8 8], ...
        'RandRotation', [-7 7], ...
        'RandScale', [0.92 1.08]);
else
    ttaAugmenter = [];
end

%% Evaluate on ID (ROI) - Baseline
fprintf('=== EVALUATING ON IN-DISTRIBUTION DATA (ROI) ===\n');
fprintf('No ROI alignment needed (already cropped)...\n');
[resultsID, predictionsID] = evaluate_dataset_standard(...
    trainedNet, imdsROI, classes, useGPU, preprocessing_method_param, refHist);

fprintf('\nID (ROI) Results:\n');
fprintf('  Accuracy: %.3f\n', resultsID.accuracy);
fprintf('  Precision: %.3f\n', resultsID.precision);
fprintf('  Sensitivity: %.3f\n', resultsID.sensitivity);
fprintf('  Specificity: %.3f\n', resultsID.specificity);
fprintf('  F1-Score: %.3f\n', resultsID.f1_score);
fprintf('  AUC: %.3f\n', resultsID.auc);

%% Evaluate on OOD (Full CXR) with ROI Alignment
fprintf('\n=== EVALUATING ON OUT-OF-DISTRIBUTION DATA (Full CXR) ===\n');
fprintf('Applying ROI alignment (boxing method)...\n');
fprintf('  - Detecting lung regions in full CXR images\n');
fprintf('  - Cropping to bounding box\n');
fprintf('  - Resizing and preprocessing\n');
if useTTA_param
    fprintf('  - Using TTA with %d augmentations\n', numAugmentations_param);
end
fprintf('\n');

[resultsOOD, predictionsOOD, metadata] = evaluate_dataset_with_roi_alignment(...
    trainedNet, imdsCXR, classes, useGPU, preprocessing_method_param, refHist, ...
    alignmentConfig, useTTA_param, numAugmentations_param, ttaAugmenter, maskDir);

fprintf('\nOOD (Full CXR with ROI Alignment) Results:\n');
fprintf('  Accuracy: %.3f\n', resultsOOD.accuracy);
fprintf('  Precision: %.3f\n', resultsOOD.precision);
fprintf('  Sensitivity: %.3f\n', resultsOOD.sensitivity);
fprintf('  Specificity: %.3f\n', resultsOOD.specificity);
fprintf('  F1-Score: %.3f\n', resultsOOD.f1_score);
fprintf('  AUC: %.3f\n', resultsOOD.auc);

% Report alignment statistics
if isfield(metadata, 'usedFallback')
    numFallback = sum(metadata.usedFallback);
    fprintf('\nROI Alignment Statistics:\n');
    fprintf('  Successful alignments: %d/%d (%.1f%%)\n', ...
        numel(metadata.usedFallback) - numFallback, numel(metadata.usedFallback), ...
        (1 - numFallback/numel(metadata.usedFallback)) * 100);
    fprintf('  Fallback to center crop: %d (%.1f%%)\n', ...
        numFallback, numFallback/numel(metadata.usedFallback) * 100);
    if isfield(metadata, 'maskAreaRatio')
        avgMaskRatio = mean(metadata.maskAreaRatio(~metadata.usedFallback));
        fprintf('  Average lung area ratio: %.2f%%\n', avgMaskRatio * 100);
    end
end

%% Calculate Performance Degradation
fprintf('\n=== PERFORMANCE DEGRADATION ANALYSIS ===\n');
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

fprintf('Performance Degradation:\n');
fprintf('  Accuracy: %.2f%%\n', degradation.accuracy);
fprintf('  Precision: %.2f%%\n', degradation.precision);
fprintf('  Sensitivity: %.2f%%\n', degradation.sensitivity);
fprintf('  Specificity: %.2f%%\n', degradation.specificity);
fprintf('  F1-Score: %.2f%%\n', degradation.f1_score);
fprintf('  AUC: %.2f%%\n', degradation.auc);
fprintf('  Mean Degradation: %.2f%%\n', mean_degradation);

% Interpret degradation
fprintf('\nInterpretation:\n');
if mean_degradation < 5
    fprintf('  Status: ✓ Excellent generalization (<5%% degradation)\n');
elseif mean_degradation < 15
    fprintf('  Status: ✓ Good generalization (<15%% degradation)\n');
elseif mean_degradation < 30
    fprintf('  Status: ⚠ Moderate degradation (15-30%%)\n');
else
    fprintf('  Status: ✗ High degradation (>30%%)\n');
end

% Compare with full CXR (if available)
fprintf('\n=== COMPARISON WITH FULL CXR (NO ALIGNMENT) ===\n');
fprintf('Expected improvement over full CXR evaluation:\n');
fprintf('  Full CXR degradation: ~32%%\n');
fprintf('  ROI Alignment degradation: %.2f%%\n', mean_degradation);
improvement = 32.0 - mean_degradation;
fprintf('  Improvement: %.2f%% reduction in degradation\n', improvement);
fprintf('  OOD Accuracy improvement: %.1f%% → %.1f%%\n', ...
    57.5, resultsOOD.accuracy * 100);

%% Save Results
fprintf('\n=== SAVING RESULTS ===\n');
results = struct();
results.id_results = resultsID;
results.ood_results = resultsOOD;
results.degradation = degradation;
results.mean_degradation = mean_degradation;
results.preprocessing_method = preprocessing_method_param;
results.useTTA = useTTA_param;
results.numAugmentations = numAugmentations_param;
results.metadata = metadata;
results.alignment_config = alignmentConfig;

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
saveFile = fullfile(outputDir, sprintf('roi_alignment_results_%s_%s.mat', ...
    preprocessing_method_param, timestamp));
save(saveFile, 'results', '-v7.3');
fprintf('Results saved to: %s\n', saveFile);

%% Generate Comparison Plot
fprintf('\n=== GENERATING COMPARISON PLOT ===\n');
create_comparison_plot(resultsID, resultsOOD, degradation, outputDir, ...
    preprocessing_method_param, useTTA_param, numAugmentations_param);

fprintf('\n=== EVALUATION COMPLETE ===\n');
fprintf('Summary:\n');
fprintf('  Preprocessing: %s\n', preprocessing_method_param);
fprintf('  TTA: %s', string(useTTA_param));
if useTTA_param
    fprintf(' (%d augmentations)\n', numAugmentations_param);
else
    fprintf('\n');
end
fprintf('  ID Accuracy: %.3f\n', resultsID.accuracy);
fprintf('  OOD Accuracy (with ROI alignment): %.3f\n', resultsOOD.accuracy);
fprintf('  Mean Degradation: %.2f%%\n', mean_degradation);
if mean_degradation < 15
    status_str = '✓ Good generalization';
elseif mean_degradation < 30
    status_str = '⚠ Moderate degradation';
else
    status_str = '✗ High degradation';
end
fprintf('  Status: %s\n', status_str);

end

%% Helper Functions


function [results, predictions] = evaluate_dataset_standard(net, imds, classes, useGPU, ...
    preprocessing_method, refHist)
%EVALUATE_DATASET_STANDARD Evaluate dataset with preprocessing (no ROI alignment)

augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
reset(augDS);

Ypred = categorical.empty(0,1);
Yprobs = [];
Ytrue = imds.Labels(:);

while hasdata(augDS)
    tbl = read(augDS);
    imgs = cat(4, tbl.input{:});
    
    % Apply preprocessing
    imgs = apply_preprocessing_batch(imgs, preprocessing_method, refHist);
    
    if useGPU
        imgs = gpuArray(imgs);
    end
    dlX = dlarray(single(imgs), 'SSCB');
    
    sc = predict(net, dlX);
    lab = onehotdecode(extractdata(sc), classes, 1);
    Ypred = [Ypred; lab(:)];
    
    probs_batch = extractdata(sc);
    if size(probs_batch, 1) == numel(classes)
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

function [results, predictions, metadata] = evaluate_dataset_with_roi_alignment(...
    net, imds, classes, useGPU, preprocessing_method, refHist, alignmentConfig, ...
    useTTA, numAugmentations, ttaAugmenter, maskDir)
%EVALUATE_DATASET_WITH_ROI_ALIGNMENT Evaluate dataset with ROI alignment (boxing method)

Ypred = categorical.empty(0,1);
Yprobs = [];
Ytrue = imds.Labels(:);
numClasses = numel(classes);

metadata = struct();
metadata.boundingBoxes = cell(numel(imds.Files), 1);
metadata.usedFallback = false(numel(imds.Files), 1);
metadata.maskAreaRatio = zeros(numel(imds.Files), 1);

fprintf('Processing %d images with ROI alignment...\n', numel(imds.Files));

for idx = 1:numel(imds.Files)
    if mod(idx, 50) == 0
        fprintf('  Progress: %d/%d (%.1f%%)\n', idx, numel(imds.Files), ...
            idx/numel(imds.Files)*100);
    end
    
    % Load full CXR image
    img = imread(imds.Files{idx});
    
    % Try to load corresponding mask (if available)
    mask = [];
    if ~isempty(maskDir)
        [~, imgName, ~] = fileparts(imds.Files{idx});
        maskPath = find_mask_file(maskDir, imgName);
        if ~isempty(maskPath) && exist(maskPath, 'file')
            mask = imread(maskPath);
            if size(mask, 3) > 1
                mask = rgb2gray(mask);
            end
            mask = mask > 128;  % Binarize
        end
    end
    
    % Align CXR to ROI (crop to lung bounding box)
    [alignedImg, bbox, maskRatio, usedFallback] = align_cxr_to_roi(...
        img, mask, alignmentConfig);
    
    metadata.boundingBoxes{idx} = bbox;
    metadata.usedFallback(idx) = usedFallback;
    metadata.maskAreaRatio(idx) = maskRatio;
    
    % Apply preprocessing and get prediction
    if useTTA
        probs = predict_with_tta_roi_aligned(net, alignedImg, classes, useGPU, ...
            preprocessing_method, refHist, alignmentConfig.targetSize, ...
            numAugmentations, ttaAugmenter);
    else
        probs = predict_single_roi_aligned(net, alignedImg, classes, useGPU, ...
            preprocessing_method, refHist, alignmentConfig.targetSize);
    end
    
    [~, maxIdx] = max(probs);
    Ypred = [Ypred; categorical(classes(maxIdx))];
    if numClasses == 2
        Yprobs = [Yprobs; probs(2)];
    else
        Yprobs = [Yprobs; max(probs)];
    end
end

fprintf('  Completed: %d/%d images\n\n', numel(imds.Files), numel(imds.Files));

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

function [alignedImg, bbox, maskRatio, usedFallback] = align_cxr_to_roi(img, mask, opts)
%ALIGN_CXR_TO_ROI Crop full CXR to lung bounding box
%   If mask is provided, uses it directly. Otherwise, detects lung region.

if size(img, 3) == 1
    imgGray = img;
else
    imgGray = rgb2gray(img);
end

usedFallback = false;

% Use provided mask if available
if ~isempty(mask) && any(mask(:))
    % Ensure mask and image are same size
    if size(mask, 1) ~= size(imgGray, 1) || size(mask, 2) ~= size(imgGray, 2)
        mask = imresize(double(mask), [size(imgGray, 1), size(imgGray, 2)], 'nearest') > 0.5;
    end
    lungMask = logical(mask);
else
    % Detect lung region using adaptive thresholding
    imgGray = mat2gray(imgGray);
    imgEnh = adapthisteq(imgGray, 'NumTiles', [8 8], 'ClipLimit', opts.clipLimit);
    imgInv = imcomplement(imgEnh);
    
    try
        lungMask = imbinarize(imgInv, 'adaptive', 'ForegroundPolarity', 'bright', ...
            'Sensitivity', opts.adaptiveSensitivity);
    catch
        thresh = graythresh(imgInv);
        lungMask = imgInv > thresh * 0.9;
    end
    
    % Morphological operations
    lungMask = imopen(lungMask, strel('disk', opts.openRadius));
    lungMask = imclose(lungMask, strel('disk', opts.closeRadius));
    lungMask = imfill(lungMask, 'holes');
    lungMask = bwareaopen(lungMask, round(opts.minAreaRatio * numel(lungMask)));
    
    if any(lungMask(:))
        lungMask = logical(bwareafilt(lungMask, 1));  % Keep largest component
    else
        lungMask = false(size(lungMask));
    end
end

maskRatio = nnz(lungMask) / numel(lungMask);

% Extract bounding box and crop
if ~any(lungMask(:))
    % Fallback: center crop
    [alignedImg, bbox] = central_crop(img, opts.fallbackScale);
    usedFallback = true;
else
    stats = regionprops(lungMask, 'BoundingBox');
    bb = stats(1).BoundingBox;
    
    % Add margin
    marginX = opts.margin * bb(3);
    marginY = opts.margin * bb(4);
    x1 = max(1, floor(bb(1) - marginX));
    y1 = max(1, floor(bb(2) - marginY));
    x2 = min(size(img, 2), ceil(bb(1) + bb(3) + marginX));
    y2 = min(size(img, 1), ceil(bb(2) + bb(4) + marginY));
    
    alignedImg = img(y1:y2, x1:x2, :);
    bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1];
    usedFallback = false;
end

if ~isa(alignedImg, 'uint8')
    alignedImg = im2uint8(mat2gray(alignedImg));
end
end

function [cropped, bbox] = central_crop(img, scale)
%CENTRAL_CROP Fallback center crop when lung detection fails
[H, W, ~] = size(img);
scale = min(max(scale, 0.4), 1.0);
newH = max(1, round(H * scale));
newW = max(1, round(W * scale));
startY = floor((H - newH) / 2) + 1;
startX = floor((W - newW) / 2) + 1;
cropped = img(startY:startY+newH-1, startX:startX+newW-1, :);
bbox = [startX, startY, newW, newH];
if ~isa(cropped, 'uint8')
    cropped = im2uint8(mat2gray(cropped));
end
end

function probs = predict_single_roi_aligned(net, img, classes, useGPU, ...
    preprocessing_method, refHist, targetSize)
%PREDICT_SINGLE_ROI_ALIGNED Single prediction for ROI-aligned image

% Prepare image
img_processed = prepare_image_for_prediction(img, preprocessing_method, refHist, targetSize);

% Forward pass
dlX = dlarray(single(img_processed), 'SSCB');
if useGPU
    dlX = gpuArray(dlX);
end

sc = predict(net, dlX);
probs = gather(extractdata(sc));
if ndims(probs) == 4
    probs = squeeze(probs);
end
if size(probs, 1) ~= numel(classes)
    probs = ones(numel(classes), 1, 'single') / numel(classes);
else
    probs = probs(:, 1);
end
end

function probs = predict_with_tta_roi_aligned(net, img, classes, useGPU, ...
    preprocessing_method, refHist, targetSize, numAugmentations, ttaAugmenter)
%PREDICT_WITH_TTA_ROI_ALIGNED TTA prediction for ROI-aligned image

numClasses = numel(classes);
all_probs = zeros(numClasses, numAugmentations);

% Original image (no augmentation)
all_probs(:, 1) = predict_single_roi_aligned(net, img, classes, useGPU, ...
    preprocessing_method, refHist, targetSize);

% Augmented versions
for augIdx = 2:numAugmentations
    try
        img_aug = apply_augmentation(img, ttaAugmenter);
        all_probs(:, augIdx) = predict_single_roi_aligned(net, img_aug, classes, useGPU, ...
            preprocessing_method, refHist, targetSize);
    catch
        all_probs(:, augIdx) = all_probs(:, 1);  % Fallback to original
    end
end

% Average probabilities
probs = mean(all_probs, 2);
end

function img_processed = prepare_image_for_prediction(img, method, refHist, targetSize)
%PREPARE_IMAGE_FOR_PREDICTION Prepare image for network input

% Convert to grayscale if needed
if size(img, 3) == 1
    img_gray = img;
else
    img_gray = rgb2gray(img);
end

% Resize to target size
if ~isempty(targetSize)
    img_gray = imresize(img_gray, targetSize);
end

% Apply preprocessing
img_processed = apply_preprocessing_single(img_gray, method, refHist);

% Convert to RGB
if size(img_processed, 3) == 1
    img_processed = repmat(img_processed, [1, 1, 3]);
end

% Reshape for network
img_processed = reshape(img_processed, size(img_processed,1), size(img_processed,2), size(img_processed,3), 1);
end

function img_processed = apply_preprocessing_single(img, method, refHist)
%APPLY_PREPROCESSING_SINGLE Apply preprocessing to single image

if ~isa(img, 'uint8')
    img = im2uint8(mat2gray(img));
end

switch method
    case 'none'
        img_processed = img;
        
    case 'histmatch'
        if ~isempty(refHist)
            img_processed = histeq(img, refHist);
        else
            img_processed = img;
        end
        
    case 'zscore'
        img_double = double(img);
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
        img_double = double(img);
        img_min = min(img_double(:));
        img_max = max(img_double(:));
        if img_max > img_min
            img_normalized = (img_double - img_min) / (img_max - img_min) * 255;
        else
            img_normalized = img_double;
        end
        img_processed = uint8(img_normalized);
        
    case 'clahe'
        img_processed = adapthisteq(img, 'ClipLimit', 0.02, 'Distribution', 'uniform');
        
    otherwise
        img_processed = img;
end
end

function img_aug = apply_augmentation(img, augmenter)
%APPLY_AUGMENTATION Apply augmentation to image
augDS = augmentedImageDatastore([size(img,1), size(img,2)], {img}, ...
    'DataAugmentation', augmenter);
tbl = read(augDS);
img_aug = tbl.input{1};
end

function imgs_processed = apply_preprocessing_batch(imgs, method, refHist)
%APPLY_PREPROCESSING_BATCH Apply preprocessing to batch of images
[H, W, C, B] = size(imgs);
imgs_processed = zeros(size(imgs), 'uint8');

for b = 1:B
    img = imgs(:, :, :, b);
    if C == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    img_processed = apply_preprocessing_single(img_gray, method, refHist);
    
    if C == 3
        imgs_processed(:, :, :, b) = repmat(img_processed, [1, 1, 3]);
    else
        imgs_processed(:, :, :, b) = img_processed;
    end
end
end

function maskPath = find_mask_file(maskDir, imageName)
%FIND_MASK_FILE Find mask file for given image name
nameBase = imageName;
ext = '.png';

maskPatterns = {
    fullfile(maskDir, 'normal', [nameBase '_mask' ext])
    fullfile(maskDir, 'ptb', [nameBase '_mask' ext])
    fullfile(maskDir, 'normal', [nameBase ext])
    fullfile(maskDir, 'ptb', [nameBase ext])
};

for p = 1:numel(maskPatterns)
    if exist(maskPatterns{p}, 'file')
        maskPath = maskPatterns{p};
        return;
    end
end

maskPath = [];
end

function create_comparison_plot(resultsID, resultsOOD, degradation, outputDir, ...
    preprocessing_method, useTTA, numAugmentations)
%CREATE_COMPARISON_PLOT Create visualization comparing ID vs OOD results

figure('Position', [100, 100, 1200, 600]);

% Subplot 1: Metrics comparison
subplot(1, 2, 1);
metrics = {'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC'};
idVals = [resultsID.accuracy, resultsID.precision, resultsID.sensitivity, ...
    resultsID.specificity, resultsID.f1_score, resultsID.auc];
oodVals = [resultsOOD.accuracy, resultsOOD.precision, resultsOOD.sensitivity, ...
    resultsOOD.specificity, resultsOOD.f1_score, resultsOOD.auc];

x = 1:numel(metrics);
width = 0.35;
bar(x - width/2, idVals * 100, width, 'FaceColor', [0.2 0.6 0.8], 'DisplayName', 'ID (ROI)');
hold on;
bar(x + width/2, oodVals * 100, width, 'FaceColor', [0.8 0.4 0.2], 'DisplayName', 'OOD (CXR + ROI Alignment)');
set(gca, 'XTickLabel', metrics);
ylabel('Score (%)');
title('ID vs OOD Performance (with ROI Alignment)');
legend('Location', 'best');
grid on;
ylim([0, 100]);

% Subplot 2: Degradation
subplot(1, 2, 2);
degradVals = [degradation.accuracy, degradation.precision, degradation.sensitivity, ...
    degradation.specificity, degradation.f1_score, degradation.auc];
bar(degradVals, 'FaceColor', [0.8 0.2 0.2]);
set(gca, 'XTickLabel', metrics);
ylabel('Degradation (%)');
title('Performance Degradation');
grid on;
hold on;
yline(15, '--r', 'Good (<15%)', 'LineWidth', 2);
yline(30, '--y', 'Moderate (<30%)', 'LineWidth', 2);

% Add title with settings
if useTTA
    titleStr = sprintf('OOD Evaluation with ROI Alignment\nPreprocessing: %s, TTA: %d augs', ...
        preprocessing_method, numAugmentations);
else
    titleStr = sprintf('OOD Evaluation with ROI Alignment\nPreprocessing: %s, No TTA', ...
        preprocessing_method);
end
sgtitle(titleStr, 'FontSize', 12, 'FontWeight', 'bold');

% Save figure
figFile = fullfile(outputDir, sprintf('roi_alignment_comparison_%s.png', ...
    datestr(now, 'yyyymmdd_HHMMSS')));
saveas(gcf, figFile);
fprintf('  Comparison plot saved to: %s\n', figFile);
close(gcf);
end

