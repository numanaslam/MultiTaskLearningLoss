function evaluate_ood_roi_alignment(preprocessing_method, useTTA, numAugmentations)
%EVALUATE_OOD_ROI_ALIGNMENT Evaluate OOD performance after ROI alignment + preprocessing
%   This script addresses the distribution gap between ROI-trained images and
%   full-resolution CXR test images by explicitly extracting a lung ROI from each
%   CXR before inference. The ROI extraction uses adaptive thresholding and
%   morphology to crop/pad the lung field so that inference sees a field-of-view
%   similar to training data.
%
%   Inputs:
%       preprocessing_method - 'none' | 'clahe' | 'histmatch' | 'zscore' |
%                              'minmax' | 'clahe_zscore'. Default: 'histmatch'
%       useTTA - Enable ROI-preserving test-time augmentation (default: true)
%       numAugmentations - Number of augmentations per case if TTA enabled
%                          (default: 12)
%
%   Usage:
%       evaluate_ood_roi_alignment();                     % histmatch + TTA
%       evaluate_ood_roi_alignment('clahe', false);       % CLAHE, no TTA
%       evaluate_ood_roi_alignment('clahe_zscore', true, 8);

if nargin < 1, preprocessing_method = 'histmatch'; end
if nargin < 2, useTTA = true; end
if nargin < 3, numAugmentations = 12; end

preprocessing_method_param = preprocessing_method;
useTTA_param = useTTA;
numAugmentations_param = numAugmentations;

clc; close all;
fprintf('=== OOD EVALUATION WITH ROI ALIGNMENT ===\n');
fprintf('Preprocessing : %s\n', preprocessing_method_param);
fprintf('ROI Alignment : enabled (lungs automatically cropped)\n');
if useTTA_param
    fprintf('TTA          : %d augmentations/image\n', numAugmentations_param);
else
    fprintf('TTA          : disabled\n');
end
fprintf('\n');

%% Model + Data Paths
modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');
if ~exist(modelFile, 'file')
    fallbackCandidates = {
        fullfile('models', 'final', 'final_model_focal_gradcam_tversky_kfold.mat')
        'vgg16_multitask_trained_optimal.mat'
    };
    for k = 1:numel(fallbackCandidates)
        if exist(fallbackCandidates{k}, 'file')
            modelFile = fallbackCandidates{k};
            break;
        end
    end
    if ~exist(modelFile, 'file')
        error('Model file not found. Please train the final model first.');
    end
end

roiDir = fullfile('input', 'roi');
cxrDir = fullfile('input', 'cxr');
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
fprintf('Model loaded. GPU available: %s\n\n', string(useGPU));

%% Load Datasets
fprintf('Loading datasets...\n');
imdsROI = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imdsROI.Labels);

fprintf('  ROI (ID) : %d samples\n', numel(imdsROI.Files));
fprintf('  CXR (OOD): %d samples\n', numel(imdsCXR.Files));
fprintf('  Classes  : %s\n\n', strjoin(classes, ', '));

if isempty(imdsROI.Files) || isempty(imdsCXR.Files)
    error('Both ROI and CXR datasets are required for this evaluation.');
end

%% Reference Histogram from ROI training distribution
fprintf('Computing ROI reference histogram for %s normalization...\n', preprocessing_method_param);
[refHist, refSampleCount] = compute_roi_reference_histogram(imdsROI, 200);
fprintf('  Histogram built from %d ROI samples\n\n', refSampleCount);

%% ROI Alignment + TTA configuration
alignmentOptions = struct( ...
    'targetSize', [224 224], ...
    'clipLimit', 0.01, ...
    'adaptiveSensitivity', 0.42, ...
    'minAreaRatio', 0.12, ...
    'openRadius', 5, ...
    'closeRadius', 25, ...
    'margin', 0.08, ...
    'fallbackScale', 0.72);

ttaAugmenter = imageDataAugmenter( ...
    'RandXTranslation', [-8 8], ...
    'RandYTranslation', [-8 8], ...
    'RandRotation', [-7 7], ...
    'RandScale', [0.92 1.08]);

evalOptions = struct();
evalOptions.preprocessing_method = preprocessing_method_param;
evalOptions.refHist = refHist;
evalOptions.useGPU = useGPU;
evalOptions.alignment = alignmentOptions;
evalOptions.tta = struct('enabled', useTTA_param, ...
                         'numAugmentations', numAugmentations_param, ...
                         'augmenter', ttaAugmenter);

%% Evaluate ID (ROI) - reference baseline
fprintf('Evaluating ROI dataset (no additional alignment needed)...\n');
[resultsID, predictionsID] = evaluate_dataset_with_alignment( ...
    trainedNet, imdsROI, classes, evalOptions, false);

%% Evaluate OOD (Full CXR) with ROI alignment
fprintf('\nEvaluating Full CXR dataset with ROI alignment...\n');
[resultsOOD, predictionsOOD, oodMeta] = evaluate_dataset_with_alignment( ...
    trainedNet, imdsCXR, classes, evalOptions, true);

%% Report Degradation
degradation = struct();
fields = {'accuracy','precision','sensitivity','specificity','f1_score','auc'};
for f = 1:numel(fields)
    key = fields{f};
    degradation.(key) = (resultsID.(key) - resultsOOD.(key)) / max(resultsID.(key), eps) * 100;
end
mean_degradation = mean(struct2array(degradation));

fprintf('\n=== ROI Alignment OOD Results ===\n');
fprintf('Preprocessing      : %s\n', preprocessing_method_param);
fprintf('ID  Accuracy       : %.3f\n', resultsID.accuracy);
fprintf('OOD Accuracy       : %.3f\n', resultsOOD.accuracy);
fprintf('Mean Degradation   : %.2f%%\n', mean_degradation);
fprintf('Precision drop     : %.2f%%\n', degradation.precision);
fprintf('Sensitivity drop   : %.2f%%\n', degradation.sensitivity);
fprintf('Specificity drop   : %.2f%%\n', degradation.specificity);
fprintf('F1 drop            : %.2f%%\n', degradation.f1_score);
fprintf('AUC drop           : %.2f%%\n', degradation.auc);
if mean_degradation < 15
    fprintf('Status             : ✓ acceptable generalization with ROI alignment\n');
elseif mean_degradation < 30
    fprintf('Status             : ⚠ moderate degradation (investigate further)\n');
else
    fprintf('Status             : ✗ severe degradation (needs additional adaptation)\n');
end

%% Persist results
resultsStruct = struct();
resultsStruct.preprocessing = preprocessing_method_param;
resultsStruct.useTTA = useTTA_param;
resultsStruct.numAugmentations = numAugmentations_param;
resultsStruct.id = resultsID;
resultsStruct.ood = resultsOOD;
resultsStruct.degradation = degradation;
resultsStruct.mean_degradation = mean_degradation;
resultsStruct.predictions_id = predictionsID;
resultsStruct.predictions_ood = predictionsOOD;
resultsStruct.ood_alignment = oodMeta;

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
saveFile = fullfile(outputDir, sprintf('roi_alignment_results_%s_%s.mat', preprocessing_method_param, timestamp));
save(saveFile, 'resultsStruct', '-v7.3');
fprintf('\nResults + metadata saved to: %s\n', saveFile);
fprintf('=== Evaluation complete ===\n');
end

%% Helper Functions

function [refHist, numSamples] = compute_roi_reference_histogram(imds, numSamples)
%COMPUTE_ROI_REFERENCE_HISTOGRAM Average histogram from ROI dataset
numFiles = numel(imds.Files);
numSamples = min(max(1, numSamples), numFiles);
sampleIdx = round(linspace(1, numFiles, numSamples));

histAccum = zeros(256, numSamples, 'double');
for i = 1:numSamples
    img = imread(imds.Files{sampleIdx(i)});
    if size(img, 3) > 1, img = rgb2gray(img); end
    histAccum(:, i) = imhist(img);
end
refHist = mean(histAccum, 2);
refHist = refHist / max(sum(refHist), eps);
end

function [results, predictions, metadata] = evaluate_dataset_with_alignment( ...
    net, imds, classes, opts, alignToROI)
%EVALUATE_DATASET_WITH_ALIGNMENT Evaluate dataset with optional ROI alignment
Ytrue = imds.Labels(:);
Ypred = categorical.empty(0, 1);
Yprobs = zeros(0, 1);

numClasses = numel(classes);
metadata = struct();
metadata.boundingBoxes = cell(numel(imds.Files), 1);
metadata.usedFallback = false(numel(imds.Files), 1);
metadata.maskAreaRatio = zeros(numel(imds.Files), 1);

for idx = 1:numel(imds.Files)
    if mod(idx, 50) == 0
        fprintf('  Processing image %d/%d...\n', idx, numel(imds.Files));
    end
    
    img = imread(imds.Files{idx});
    if alignToROI
        [alignedImg, bbox, maskRatio, usedFallback] = align_cxr_to_roi(img, opts.alignment);
        metadata.boundingBoxes{idx} = bbox;
        metadata.usedFallback(idx) = usedFallback;
        metadata.maskAreaRatio(idx) = maskRatio;
    else
        alignedImg = img;
        metadata.boundingBoxes{idx} = [];
        metadata.usedFallback(idx) = false;
        metadata.maskAreaRatio(idx) = 0;
    end
    
    if opts.tta.enabled
        probs = predict_with_tta(net, alignedImg, classes, opts, numClasses);
    else
        inputImg = prepare_image_for_network(alignedImg, opts.preprocessing_method, opts.refHist, opts.alignment.targetSize);
        probs = forward_single_pass(net, inputImg, classes, opts.useGPU);
    end
    
    [~, maxIdx] = max(probs);
    Ypred = [Ypred; categorical(classes(maxIdx))];
    if numClasses == 2
        Yprobs = [Yprobs; probs(2)];
    else
        Yprobs = [Yprobs; max(probs)];
    end
end

cm = confusionmat(Ytrue, Ypred, 'Order', categorical(classes));
if numClasses == 2 && all(size(cm) == [2 2])
    TP = cm(2,2); FP = cm(1,2); FN = cm(2,1); TN = cm(1,1);
else
    [~, posIdx] = min(countcats(Ytrue));
    pos = classes{posIdx};
    TP = sum(Ypred == pos & Ytrue == pos);
    FP = sum(Ypred == pos & Ytrue ~= pos);
    FN = sum(Ypred ~= pos & Ytrue == pos);
    TN = sum(Ypred ~= pos & Ytrue ~= pos);
end

results = struct();
results.accuracy = sum(Ypred == Ytrue) / numel(Ytrue);
results.precision = TP / (TP + FP + eps);
results.sensitivity = TP / (TP + FN + eps);
results.specificity = TN / (TN + FP + eps);
results.f1_score = 2 * results.precision * results.sensitivity / (results.precision + results.sensitivity + eps);
try
    if numClasses == 2
        [~,~,~,auc] = perfcurve(double(Ytrue == classes{2}), Yprobs, 1);
    else
        auc = NaN;
    end
catch
    auc = 0.5;
end
results.auc = auc;
results.confusion_matrix = cm;

predictions = struct();
predictions.Ypred = Ypred;
predictions.Ytrue = Ytrue;
predictions.Yprobs = Yprobs;
end

function probs = predict_with_tta(net, alignedImg, classes, opts, numClasses)
%PREDICT_WITH_TTA Predict using ROI-preserving TTA
numAug = max(1, opts.tta.numAugmentations);
all_probs = zeros(numClasses, numAug);

baseImg = prepare_image_for_network(alignedImg, opts.preprocessing_method, opts.refHist, opts.alignment.targetSize);
all_probs(:,1) = forward_single_pass(net, baseImg, classes, opts.useGPU);

for k = 2:numAug
    augImg = apply_augmentation(alignedImg, opts.tta.augmenter);
    augImg = prepare_image_for_network(augImg, opts.preprocessing_method, opts.refHist, opts.alignment.targetSize);
    all_probs(:,k) = forward_single_pass(net, augImg, classes, opts.useGPU);
end

probs = mean(all_probs, 2);
end

function data = prepare_image_for_network(img, method, refHist, targetSize)
%PREPARE_IMAGE_FOR_NETWORK Aligns dtype + size + preprocessing
if isempty(img)
    img = uint8(zeros(targetSize(1), targetSize(2), 3));
end
if ~isa(img, 'uint8')
    img = im2uint8(mat2gray(img));
end
if size(img, 3) == 1
    img = repmat(img, [1 1 3]);
end
if ~isempty(targetSize)
    img = imresize(img, targetSize);
end
img = apply_preprocessing_single(img, method, refHist);
data = single(img);
data = reshape(data, size(data,1), size(data,2), size(data,3), 1);
end

function probs = forward_single_pass(net, imgBatch, classes, useGPU)
%FORWARD_SINGLE_PASS Run network forward pass for a single sample
dlX = dlarray(imgBatch, 'SSCB');
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
    probs = probs(:,1);
end
end

function [alignedImg, bbox, maskRatio, usedFallback] = align_cxr_to_roi(img, opts)
%ALIGN_CXR_TO_ROI Rough lung ROI extractor for full CXR images
if size(img, 3) == 1
    imgGray = img;
else
    imgGray = rgb2gray(img);
end
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

lungMask = imopen(lungMask, strel('disk', opts.openRadius));
lungMask = imclose(lungMask, strel('disk', opts.closeRadius));
lungMask = imfill(lungMask, 'holes');
lungMask = bwareaopen(lungMask, round(opts.minAreaRatio * numel(lungMask)));

if any(lungMask(:))
    lungMask = logical(bwareafilt(lungMask, 1));
else
    lungMask = false(size(lungMask));
end
maskRatio = nnz(lungMask) / numel(lungMask);

if ~any(lungMask(:))
    [alignedImg, bbox] = central_crop(img, opts.fallbackScale);
    usedFallback = true;
else
    stats = regionprops(lungMask, 'BoundingBox');
    bb = stats(1).BoundingBox;
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
%CENTRAL_CROP Fallback central crop when alignment fails
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

function img_processed = apply_preprocessing_single(img, method, refHist)
%APPLY_PREPROCESSING_SINGLE Intensity preprocessing helper
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
        img_std = std(img_double(:));
        if img_std > 0
            img_normalized = (img_double - mean(img_double(:))) / img_std;
            img_normalized = (img_normalized - min(img_normalized(:))) / ...
                (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
        else
            img_normalized = img_double;
        end
        img_processed = uint8(img_normalized);
    case 'minmax'
        img_double = double(img_gray);
        if max(img_double(:)) > min(img_double(:))
            img_processed = uint8( (img_double - min(img_double(:))) / ...
                (max(img_double(:)) - min(img_double(:))) * 255 );
        else
            img_processed = uint8(img_double);
        end
    case 'clahe_zscore'
        img_clahe = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
        img_double = double(img_clahe);
        img_std = std(img_double(:));
        if img_std > 0
            img_normalized = (img_double - mean(img_double(:))) / img_std;
            img_normalized = (img_normalized - min(img_normalized(:))) / ...
                (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
        else
            img_normalized = img_double;
        end
        img_processed = uint8(img_normalized);
    otherwise
        img_processed = img_gray;
end

if size(img, 3) == 3
    img_processed = repmat(img_processed, [1 1 3]);
end
end

function img_aug = apply_augmentation(img, augmenter)
%APPLY_AUGMENTATION Apply augmentation to a single image
img_aug = img;

try
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
    
    if isprop(augmenter, 'RandRotation') && ~isempty(augmenter.RandRotation)
        angle_range = augmenter.RandRotation;
        angle = rand() * (angle_range(2) - angle_range(1)) + angle_range(1);
        img_aug = imrotate(img_aug, angle, 'bilinear', 'crop');
    end
    
    if isprop(augmenter, 'RandScale') && ~isempty(augmenter.RandScale)
        scale_range = augmenter.RandScale;
        scale = rand() * (scale_range(2) - scale_range(1)) + scale_range(1);
        [h, w, c] = size(img_aug);
        newH = round(h * scale);
        newW = round(w * scale);
        newH = max(1, newH);
        newW = max(1, newW);
        
        img_aug = imresize(img_aug, [newH, newW]);
        
        if newH >= h && newW >= w
            startH = floor((newH - h) / 2) + 1;
            startW = floor((newW - w) / 2) + 1;
            img_aug = img_aug(startH:startH+h-1, startW:startW+w-1, :);
        else
            img_padded = zeros(h, w, c, class(img_aug));
            startH = floor((h - newH) / 2) + 1;
            startW = floor((w - newW) / 2) + 1;
            img_padded(startH:startH+newH-1, startW:startW+newW-1, :) = img_aug;
            img_aug = img_padded;
        end
    end
catch ME
    warning('Augmentation failed: %s. Using original image.', ME.message);
    img_aug = img;
end

if ~isa(img_aug, 'uint8')
    img_aug = im2uint8(mat2gray(img_aug));
end
end
