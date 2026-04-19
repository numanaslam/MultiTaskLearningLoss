function debug_model_attention(modelFile, numSamples)
%DEBUG_MODEL_ATTENTION Debug whether model is learning on right areas
%   This script visualizes GradCAM attention maps and compares them with
%   ground truth masks to verify the model is focusing on lung regions.
%
%   Inputs:
%       modelFile - Path to trained model (optional)
%                   Default: 'models/final/final_model_histmatch_kfold.mat'
%       numSamples - Number of samples to visualize (optional)
%                    Default: 10
%
%   Outputs:
%       - Visualizations showing attention vs ground truth
%       - Statistics on attention quality (IoU, Dice)
%       - Analysis of attention distribution (lung vs non-lung)
%       - Saved figures in 'results/attention_debug/'
%
%   Usage:
%       debug_model_attention()
%       debug_model_attention('models/final/final_model_histmatch_kfold.mat', 20)

if nargin < 1
    modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');
end
if nargin < 2
    numSamples = 10;
end

clc; close all;
fprintf('=== DEBUGGING MODEL ATTENTION ===\n\n');

%% Load Model
fprintf('Loading model: %s\n', modelFile);
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
roiDir = fullfile('input', 'roi');
cxrDir = fullfile('input', 'cxr');
maskDir = fullfile('input', 'masks');

% Check if directories exist, try resized if not
if ~exist(roiDir, 'dir')
    roiDir = fullfile('input', 'resized', 'roi');
end
if ~exist(cxrDir, 'dir')
    cxrDir = fullfile('input', 'resized', 'cxr');
end
if ~exist(maskDir, 'dir')
    maskDir = fullfile('input', 'resized', 'masks');
end

imdsROI = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imdsROI.Labels);

fprintf('  ROI (ID): %d samples\n', numel(imdsROI.Files));
fprintf('  CXR (OOD): %d samples\n', numel(imdsCXR.Files));
fprintf('  Classes: %s\n\n', strjoin(classes, ', '));

if isempty(imdsROI.Files) || isempty(imdsCXR.Files)
    error('Both ROI and CXR datasets are required.');
end

%% Load Masks
fprintf('Loading ground truth masks...\n');
maskFiles = cell(numel(imdsROI.Files), 1);
for i = 1:numel(imdsROI.Files)
    [~, imgName, ~] = fileparts(imdsROI.Files{i});
    % Extract label from path
    pathParts = strsplit(imdsROI.Files{i}, filesep);
    label = '';
    for p = 1:numel(pathParts)
        if strcmp(pathParts{p}, 'normal') || strcmp(pathParts{p}, 'ptb')
            label = pathParts{p};
            break;
        end
    end
    maskPath = find_mask_file(maskDir, imgName, label);
    maskFiles{i} = maskPath;
end
numMasksFound = sum(~cellfun(@isempty, maskFiles));
fprintf('  Masks loaded: %d/%d (%.1f%%)\n\n', numMasksFound, numel(maskFiles), ...
    numMasksFound/numel(maskFiles)*100);

%% Analyze Attention on ID (ROI) Data
fprintf('=== ANALYZING ATTENTION ON ID (ROI) DATA ===\n');
[attentionStatsID, sampleResultsID] = analyze_attention_on_dataset(...
    trainedNet, imdsROI, maskFiles, classes, useGPU, numSamples, 'ID (ROI)');

%% Analyze Attention on OOD (Full CXR) Data
fprintf('\n=== ANALYZING ATTENTION ON OOD (Full CXR) DATA ===\n');
% Get corresponding masks for CXR images
maskFilesCXR = cell(numel(imdsCXR.Files), 1);
for i = 1:numel(imdsCXR.Files)
    [~, imgName, ~] = fileparts(imdsCXR.Files{i});
    % Extract label from path
    pathParts = strsplit(imdsCXR.Files{i}, filesep);
    label = '';
    for p = 1:numel(pathParts)
        if strcmp(pathParts{p}, 'normal') || strcmp(pathParts{p}, 'ptb')
            label = pathParts{p};
            break;
        end
    end
    maskPath = find_mask_file(maskDir, imgName, label);
    maskFilesCXR{i} = maskPath;
end

[attentionStatsOOD, sampleResultsOOD] = analyze_attention_on_dataset(...
    trainedNet, imdsCXR, maskFilesCXR, classes, useGPU, numSamples, 'OOD (Full CXR)');

%% Compare ID vs OOD Attention
fprintf('\n=== COMPARING ID vs OOD ATTENTION ===\n');
fprintf('ID (ROI) Attention Quality:\n');
fprintf('  Mean IoU: %.3f ± %.3f\n', attentionStatsID.mean_iou, attentionStatsID.std_iou);
fprintf('  Mean Dice: %.3f ± %.3f\n', attentionStatsID.mean_dice, attentionStatsID.std_dice);
fprintf('  Attention in lung region: %.1f%%\n', attentionStatsID.attention_in_lungs * 100);
fprintf('  Attention outside lungs: %.1f%%\n', attentionStatsID.attention_outside_lungs * 100);

fprintf('\nOOD (Full CXR) Attention Quality:\n');
fprintf('  Mean IoU: %.3f ± %.3f\n', attentionStatsOOD.mean_iou, attentionStatsOOD.std_iou);
fprintf('  Mean Dice: %.3f ± %.3f\n', attentionStatsOOD.mean_dice, attentionStatsOOD.std_dice);
fprintf('  Attention in lung region: %.1f%%\n', attentionStatsOOD.attention_in_lungs * 100);
fprintf('  Attention outside lungs: %.1f%%\n', attentionStatsOOD.attention_outside_lungs * 100);

%% Create Visualizations
fprintf('\n=== CREATING VISUALIZATIONS ===\n');
outputDir = fullfile('results', 'attention_debug');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

create_attention_visualizations(sampleResultsID, sampleResultsOOD, outputDir, ...
    attentionStatsID, attentionStatsOOD);

%% Save Results
fprintf('\n=== SAVING RESULTS ===\n');
results = struct();
results.id_stats = attentionStatsID;
results.ood_stats = attentionStatsOOD;
results.sample_results_id = sampleResultsID;
results.sample_results_ood = sampleResultsOOD;
results.model_file = modelFile;

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
saveFile = fullfile(outputDir, sprintf('attention_debug_results_%s.mat', timestamp));
save(saveFile, 'results', '-v7.3');
fprintf('Results saved to: %s\n', saveFile);

%% Summary
fprintf('\n=== DEBUGGING COMPLETE ===\n');
fprintf('Results saved to: %s\n', outputDir);
fprintf('\nKey Findings:\n');
if attentionStatsID.attention_in_lungs > 0.7
    fprintf('  ✓ Model focuses on lung regions (ID: %.1f%%)\n', attentionStatsID.attention_in_lungs * 100);
else
    fprintf('  ⚠ Model may not be focusing enough on lungs (ID: %.1f%%)\n', attentionStatsID.attention_in_lungs * 100);
end
if attentionStatsOOD.attention_in_lungs > 0.6
    fprintf('  ✓ Model focuses on lung regions (OOD: %.1f%%)\n', attentionStatsOOD.attention_in_lungs * 100);
else
    fprintf('  ⚠ Model may not be focusing enough on lungs (OOD: %.1f%%)\n', attentionStatsOOD.attention_in_lungs * 100);
end
if attentionStatsID.mean_iou > 0.5
    fprintf('  ✓ Good alignment with ground truth masks (ID IoU: %.3f)\n', attentionStatsID.mean_iou);
else
    fprintf('  ⚠ Attention may not align well with masks (ID IoU: %.3f)\n', attentionStatsID.mean_iou);
end
if attentionStatsOOD.mean_iou > 0.4
    fprintf('  ✓ Reasonable alignment with ground truth masks (OOD IoU: %.3f)\n', attentionStatsOOD.mean_iou);
else
    fprintf('  ⚠ Attention may not align well with masks (OOD IoU: %.3f)\n', attentionStatsOOD.mean_iou);
end

end

%% Helper Functions

function [stats, sampleResults] = analyze_attention_on_dataset(...
    net, imds, maskFiles, classes, useGPU, numSamples, datasetName)

featureLayer = 'relu5_3';
numFiles = numel(imds.Files);
numSamples = min(numSamples, numFiles);
sampleIdx = round(linspace(1, numFiles, numSamples));

ious = [];
dices = [];
attention_in_lungs = [];
attention_outside_lungs = [];
sampleResults = struct();

fprintf('  Analyzing %d samples from %s...\n', numSamples, datasetName);

for i = 1:numSamples
    idx = sampleIdx(i);
    
    % Load image
    img = imread(imds.Files{idx});
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    img_resized = imresize(img, [224 224]);
    
    % Compute GradCAM (must use dlfeval for gradient computation)
    try
        cam = dlfeval(@compute_gradcam_for_image, net, img_resized, imds.Labels(idx), classes, featureLayer, useGPU);
    catch ME
        warning('Failed to compute GradCAM for sample %d: %s', idx, ME.message);
        cam = ones(224, 224) * 0.5;
    end
    
    % Load ground truth mask
    realMask = [];
    iou = 0;
    dice = 0;
    attention_in_lung = 0;
    attention_outside_lung = 0;
    
    if ~isempty(maskFiles{idx}) && exist(maskFiles{idx}, 'file')
        try
            realMask = imread(maskFiles{idx});
            if size(realMask, 3) > 1
                realMask = rgb2gray(realMask);
            end
            realMask = imresize(realMask, [224 224], 'nearest');
            realMask = realMask > 128;
            
            % Convert CAM to binary mask
            cam_values = cam(:);
            if std(cam_values) > 0.01
                threshold = prctile(cam_values, 75);
            else
                threshold = 0.5;
            end
            predMask = cam > threshold;
            predMask = imopen(predMask, strel('disk', 2));
            predMask = imclose(predMask, strel('disk', 3));
            predMask = imfill(predMask, 'holes');
            predMask = logical(predMask);
            
            % Compute metrics
            [iou, dice] = compute_iou_dice(predMask, realMask);
            ious(end+1) = iou;
            dices(end+1) = dice;
            
            % Compute attention distribution
            cam_normalized = cam / (max(cam(:)) + eps);
            if any(realMask(:))
                attention_in_lung = sum(cam_normalized(realMask)) / (sum(realMask(:)) + eps);
            end
            if any(~realMask(:))
                attention_outside_lung = sum(cam_normalized(~realMask)) / (sum(~realMask(:)) + eps);
            end
            attention_in_lungs(end+1) = attention_in_lung;
            attention_outside_lungs(end+1) = attention_outside_lung;
        catch ME
            warning('Failed to process mask for sample %d: %s', idx, ME.message);
        end
    end
    
    % Store sample results
    sampleResults(i).image = img_resized;
    sampleResults(i).cam = cam;
    sampleResults(i).predMask = predMask;
    sampleResults(i).realMask = realMask;
    sampleResults(i).label = imds.Labels(idx);
    sampleResults(i).iou = iou;
    sampleResults(i).dice = dice;
    sampleResults(i).attention_in_lungs = attention_in_lung;
    sampleResults(i).attention_outside_lungs = attention_outside_lung;
    sampleResults(i).file_path = imds.Files{idx};
end

stats = struct();
if ~isempty(ious)
    stats.mean_iou = mean(ious);
    stats.std_iou = std(ious);
    stats.mean_dice = mean(dices);
    stats.std_dice = std(dices);
    stats.attention_in_lungs = mean(attention_in_lungs);
    stats.attention_outside_lungs = mean(attention_outside_lungs);
else
    stats.mean_iou = 0;
    stats.std_iou = 0;
    stats.mean_dice = 0;
    stats.std_dice = 0;
    stats.attention_in_lungs = 0;
    stats.attention_outside_lungs = 0;
end

if ~isempty(ious)
    fprintf('  Mean IoU: %.3f ± %.3f\n', stats.mean_iou, stats.std_iou);
    fprintf('  Mean Dice: %.3f ± %.3f\n', stats.mean_dice, stats.std_dice);
    fprintf('  Attention in lungs: %.1f%%\n', stats.attention_in_lungs * 100);
    fprintf('  Attention outside lungs: %.1f%%\n', stats.attention_outside_lungs * 100);
else
    fprintf('  ⚠ No masks found for analysis\n');
end
end

function cam = compute_gradcam_for_image(net, img, label, classes, featureLayer, useGPU)
% Compute GradCAM for a single image
% This function is designed to be called with dlfeval to enable gradient computation

if size(img, 3) == 1
    img = repmat(img, [1 1 3]);
end

dlX = dlarray(single(img), 'SSCB');
if useGPU
    dlX = gpuArray(dlX);
end

% Forward pass to get feature maps and logits
[featMap, logits] = forward(net, dlX, 'Outputs', {featureLayer, 'fc8'});
logits = squeeze(logits);

% Get the predicted class
[~, classIdx] = max(extractdata(logits));

% Compute score for the predicted class
score = sum(logits(classIdx), 'all');

% Compute gradients with respect to feature maps
gradFeat = dlgradient(score, featMap);

% Compute weights as mean of gradients
w = mean(gradFeat, [1 2]);

% Compute CAM as weighted sum of feature maps
cam = sum(featMap .* w, 3);
cam = max(cam, 0);

% Extract and normalize
cam = extractdata(cam);
cam = imresize(cam, [224 224]);
cam_max = max(cam(:));
if cam_max > 0
    cam = single(cam ./ (cam_max + eps));
else
    cam = single(cam);
end
end

function [iou, dice] = compute_iou_dice(pred, target)
pred = logical(pred);
target = logical(target);
intersection = sum(pred(:) & target(:));
union = sum(pred(:) | target(:));
pred_sum = sum(pred(:));
target_sum = sum(target(:));
iou = intersection / (union + eps);
dice = 2 * intersection / (pred_sum + target_sum + eps);
end

function maskPath = find_mask_file(maskDir, imageName, label)
% Find mask file for given image name
nameBase = imageName;
ext = '.png';

% Build search patterns - prioritize label-specific paths
maskPatterns = {};
if nargin >= 3 && ~isempty(label) && (strcmp(label, 'normal') || strcmp(label, 'ptb'))
    maskPatterns{end+1} = fullfile(maskDir, label, [nameBase '_mask' ext]);
    maskPatterns{end+1} = fullfile(maskDir, label, [nameBase ext]);
end

% Then try both labels
maskPatterns{end+1} = fullfile(maskDir, 'normal', [nameBase '_mask' ext]);
maskPatterns{end+1} = fullfile(maskDir, 'ptb', [nameBase '_mask' ext]);
maskPatterns{end+1} = fullfile(maskDir, 'normal', [nameBase ext]);
maskPatterns{end+1} = fullfile(maskDir, 'ptb', [nameBase ext]);

% Remove duplicates
[~, uniqueIdx] = unique(maskPatterns, 'stable');
maskPatterns = maskPatterns(uniqueIdx);

% Search for mask file
for p = 1:numel(maskPatterns)
    if exist(maskPatterns{p}, 'file')
        maskPath = maskPatterns{p};
        return;
    end
end

maskPath = [];
end

function create_attention_visualizations(sampleResultsID, sampleResultsOOD, outputDir, ...
    statsID, statsOOD)
% Create comprehensive visualizations

numSamples = min(numel(sampleResultsID), numel(sampleResultsOOD));
numSamples = min(numSamples, 6);  % Show up to 6 samples

% Main comparison figure
figure('Position', [100, 100, 1800, 1200]);

for i = 1:numSamples
    % ID (ROI) visualization
    subplot(numSamples, 6, (i-1)*6 + 1);
    img_id = sampleResultsID(i).image;
    if max(img_id(:)) > 1
        img_id = double(img_id) / 255;
    end
    imshow(img_id);
    title(sprintf('ID: %s', string(sampleResultsID(i).label)), 'FontSize', 10);
    
    subplot(numSamples, 6, (i-1)*6 + 2);
    cam_id = sampleResultsID(i).cam;
    imshow(cam_id);
    colormap('jet');
    title(sprintf('ID CAM\nIoU: %.2f', sampleResultsID(i).iou), 'FontSize', 10);
    
    subplot(numSamples, 6, (i-1)*6 + 3);
    if ~isempty(sampleResultsID(i).realMask) && any(sampleResultsID(i).realMask(:))
        overlay = img_id;
        if size(overlay, 3) == 1
            overlay = repmat(overlay, [1 1 3]);
        end
        overlay(:,:,1) = overlay(:,:,1) + 0.3 * double(sampleResultsID(i).realMask);
        overlay(:,:,2) = overlay(:,:,2) + 0.3 * double(sampleResultsID(i).predMask);
        overlay = min(overlay, 1);
        overlay = max(overlay, 0);
        imshow(overlay);
        title('ID: Red=GT, Green=Pred', 'FontSize', 10);
    else
        imshow(img_id);
        title('ID: No mask', 'FontSize', 10);
    end
    
    % OOD (Full CXR) visualization
    subplot(numSamples, 6, (i-1)*6 + 4);
    img_ood = sampleResultsOOD(i).image;
    if max(img_ood(:)) > 1
        img_ood = double(img_ood) / 255;
    end
    imshow(img_ood);
    title(sprintf('OOD: %s', string(sampleResultsOOD(i).label)), 'FontSize', 10);
    
    subplot(numSamples, 6, (i-1)*6 + 5);
    cam_ood = sampleResultsOOD(i).cam;
    imshow(cam_ood);
    colormap('jet');
    title(sprintf('OOD CAM\nIoU: %.2f', sampleResultsOOD(i).iou), 'FontSize', 10);
    
    subplot(numSamples, 6, (i-1)*6 + 6);
    if ~isempty(sampleResultsOOD(i).realMask) && any(sampleResultsOOD(i).realMask(:))
        overlay = img_ood;
        if size(overlay, 3) == 1
            overlay = repmat(overlay, [1 1 3]);
        end
        overlay(:,:,1) = overlay(:,:,1) + 0.3 * double(sampleResultsOOD(i).realMask);
        overlay(:,:,2) = overlay(:,:,2) + 0.3 * double(sampleResultsOOD(i).predMask);
        overlay = min(overlay, 1);
        overlay = max(overlay, 0);
        imshow(overlay);
        title('OOD: Red=GT, Green=Pred', 'FontSize', 10);
    else
        imshow(img_ood);
        title('OOD: No mask', 'FontSize', 10);
    end
end

sgtitle('Model Attention Debugging: ID vs OOD', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'attention_debug_comparison.png'));
fprintf('  Saved: attention_debug_comparison.png\n');
close(gcf);

% Statistics plot
figure('Position', [100, 100, 1400, 600]);

subplot(1, 3, 1);
ious_id = [sampleResultsID.iou];
ious_ood = [sampleResultsOOD.iou];
ious_id = ious_id(ious_id > 0);
ious_ood = ious_ood(ious_ood > 0);
if ~isempty(ious_id) && ~isempty(ious_ood) && numel(ious_id) > 0 && numel(ious_ood) > 0
    % Ensure we have valid numeric data
    ious_id = double(ious_id(:));
    ious_ood = double(ious_ood(:));
    % Remove any NaN or Inf values
    ious_id = ious_id(isfinite(ious_id));
    ious_ood = ious_ood(isfinite(ious_ood));
    
    if ~isempty(ious_id) && ~isempty(ious_ood) && numel(ious_id) > 0 && numel(ious_ood) > 0
        % Use bar plot with error bars instead of boxplot for robustness
        mean_id = mean(ious_id);
        mean_ood = mean(ious_ood);
        std_id = std(ious_id);
        std_ood = std(ious_ood);
        
        bar([mean_id, mean_ood]);
        hold on;
        errorbar([1, 2], [mean_id, mean_ood], [std_id, std_ood], 'k.', 'LineWidth', 1.5);
        set(gca, 'XTickLabel', {'ID (ROI)', 'OOD (CXR)'});
        ylabel('IoU');
        title('Attention-Mask Alignment (IoU)');
        grid on;
        maxVal = max([ious_id; ious_ood]);
        ylim([0, max(0.3, maxVal * 1.2)]);
        hold off;
    else
        text(0.5, 0.5, 'No valid IoU data', 'HorizontalAlignment', 'center');
        title('Attention-Mask Alignment (IoU)');
    end
else
    text(0.5, 0.5, 'No valid IoU data', 'HorizontalAlignment', 'center');
    title('Attention-Mask Alignment (IoU)');
end

subplot(1, 3, 2);
attention_id = [sampleResultsID.attention_in_lungs];
attention_ood = [sampleResultsOOD.attention_in_lungs];
attention_id = attention_id(attention_id > 0);
attention_ood = attention_ood(attention_ood > 0);
if ~isempty(attention_id) && ~isempty(attention_ood) && numel(attention_id) > 0 && numel(attention_ood) > 0
    % Ensure we have valid numeric data
    attention_id = double(attention_id(:));
    attention_ood = double(attention_ood(:));
    % Remove any NaN or Inf values
    attention_id = attention_id(isfinite(attention_id));
    attention_ood = attention_ood(isfinite(attention_ood));
    
    if ~isempty(attention_id) && ~isempty(attention_ood) && numel(attention_id) > 0 && numel(attention_ood) > 0
        % Use bar plot with error bars instead of boxplot for robustness
        mean_id = mean(attention_id);
        mean_ood = mean(attention_ood);
        std_id = std(attention_id);
        std_ood = std(attention_ood);
        
        bar([mean_id, mean_ood]);
        hold on;
        errorbar([1, 2], [mean_id, mean_ood], [std_id, std_ood], 'k.', 'LineWidth', 1.5);
        set(gca, 'XTickLabel', {'ID (ROI)', 'OOD (CXR)'});
        ylabel('Attention Ratio');
        title('Attention in Lung Region');
        grid on;
        maxVal = max([attention_id; attention_ood]);
        ylim([0, max(0.2, maxVal * 1.2)]);
        hold off;
    else
        text(0.5, 0.5, 'No valid attention data', 'HorizontalAlignment', 'center');
        title('Attention in Lung Region');
    end
else
    text(0.5, 0.5, 'No valid attention data', 'HorizontalAlignment', 'center');
    title('Attention in Lung Region');
end

subplot(1, 3, 3);
axis off;
summary_text = {
    'ATTENTION QUALITY SUMMARY';
    '';
    sprintf('ID (ROI) Statistics:');
    sprintf('  Mean IoU: %.3f ± %.3f', statsID.mean_iou, statsID.std_iou);
    sprintf('  Mean Dice: %.3f ± %.3f', statsID.mean_dice, statsID.std_dice);
    sprintf('  Attention in lungs: %.1f%%', statsID.attention_in_lungs * 100);
    sprintf('  Attention outside: %.1f%%', statsID.attention_outside_lungs * 100);
    '';
    sprintf('OOD (Full CXR) Statistics:');
    sprintf('  Mean IoU: %.3f ± %.3f', statsOOD.mean_iou, statsOOD.std_iou);
    sprintf('  Mean Dice: %.3f ± %.3f', statsOOD.mean_dice, statsOOD.std_dice);
    sprintf('  Attention in lungs: %.1f%%', statsOOD.attention_in_lungs * 100);
    sprintf('  Attention outside: %.1f%%', statsOOD.attention_outside_lungs * 100);
    '';
    'Interpretation:';
    '  • IoU > 0.5: Good alignment';
    '  • Attention in lungs > 70%%: Good focus';
    '  • Lower OOD values expected';
};

text(0.1, 0.9, summary_text, 'FontSize', 11, ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'FontName', 'FixedWidth');

sgtitle('Attention Quality Statistics', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'attention_statistics.png'));
fprintf('  Saved: attention_statistics.png\n');
close(gcf);

% Detailed attention overlay visualization
figure('Position', [100, 100, 1600, 1000]);
numSamples = min(numSamples, 4);

for i = 1:numSamples
    % ID overlay
    subplot(numSamples, 4, (i-1)*4 + 1);
    img_id = sampleResultsID(i).image;
    if max(img_id(:)) > 1
        img_id = double(img_id) / 255;
    end
    if size(img_id, 3) == 1
        img_id = repmat(img_id, [1 1 3]);
    end
    cam_id = sampleResultsID(i).cam;
    overlay_id = img_id;
    overlay_id(:,:,1) = overlay_id(:,:,1) + 0.4 * cam_id;
    overlay_id = min(overlay_id, 1);
    imshow(overlay_id);
    title(sprintf('ID: %s (Overlay)', string(sampleResultsID(i).label)), 'FontSize', 10);
    
    subplot(numSamples, 4, (i-1)*4 + 2);
    if ~isempty(sampleResultsID(i).realMask) && any(sampleResultsID(i).realMask(:))
        imshow(sampleResultsID(i).realMask);
        title('ID: Ground Truth Mask', 'FontSize', 10);
    else
        imshow(img_id);
        title('ID: No mask available', 'FontSize', 10);
    end
    
    % OOD overlay
    subplot(numSamples, 4, (i-1)*4 + 3);
    img_ood = sampleResultsOOD(i).image;
    if max(img_ood(:)) > 1
        img_ood = double(img_ood) / 255;
    end
    if size(img_ood, 3) == 1
        img_ood = repmat(img_ood, [1 1 3]);
    end
    cam_ood = sampleResultsOOD(i).cam;
    overlay_ood = img_ood;
    overlay_ood(:,:,1) = overlay_ood(:,:,1) + 0.4 * cam_ood;
    overlay_ood = min(overlay_ood, 1);
    imshow(overlay_ood);
    title(sprintf('OOD: %s (Overlay)', string(sampleResultsOOD(i).label)), 'FontSize', 10);
    
    subplot(numSamples, 4, (i-1)*4 + 4);
    if ~isempty(sampleResultsOOD(i).realMask) && any(sampleResultsOOD(i).realMask(:))
        imshow(sampleResultsOOD(i).realMask);
        title('OOD: Ground Truth Mask', 'FontSize', 10);
    else
        imshow(img_ood);
        title('OOD: No mask available', 'FontSize', 10);
    end
end

sgtitle('Attention Overlay Visualization', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'attention_overlay.png'));
fprintf('  Saved: attention_overlay.png\n');
close(gcf);

fprintf('  All visualizations saved to: %s\n', outputDir);
end

