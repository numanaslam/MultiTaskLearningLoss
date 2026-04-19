%ANALYZE_ID_OOD_DATASETS Analyze ROI and CXR datasets for ID/OOD evaluation
%   This script verifies that the resized ROI and CXR datasets are suitable
%   for In-Distribution (ID) and Out-of-Distribution (OOD) evaluation.

clc; close all;
fprintf('=== ID/OOD DATASET ANALYSIS ===\n\n');

%% Paths
roiDir = fullfile('input', 'resized', 'roi');
cxrDir = fullfile('input', 'resized', 'cxr');

%% Check directories exist
if ~exist(roiDir, 'dir')
    error('ROI directory not found: %s', roiDir);
end
if ~exist(cxrDir, 'dir')
    error('CXR directory not found: %s', cxrDir);
end

%% Load datasets
fprintf('Loading datasets...\n');
imdsROI = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

fprintf('  ROI (ID) dataset: %d samples\n', numel(imdsROI.Files));
fprintf('  CXR (OOD) dataset: %d samples\n', numel(imdsCXR.Files));

%% Check classes
classesROI = categories(imdsROI.Labels);
classesCXR = categories(imdsCXR.Labels);
fprintf('  ROI classes: %s\n', strjoin(classesROI, ', '));
fprintf('  CXR classes: %s\n', strjoin(classesCXR, ', '));

if ~isequal(sort(classesROI), sort(classesCXR))
    warning('Class mismatch between ROI and CXR datasets!');
end

%% Analyze image properties
fprintf('\n=== IMAGE PROPERTIES ANALYSIS ===\n');

% Sample images from each class
sampleROI = imread(imdsROI.Files{1});
sampleCXR = imread(imdsCXR.Files{1});

if size(sampleROI, 3) > 1
    sampleROI = rgb2gray(sampleROI);
end
if size(sampleCXR, 3) > 1
    sampleCXR = rgb2gray(sampleCXR);
end

fprintf('ROI (ID) Sample Image:\n');
fprintf('  Size: %d x %d\n', size(sampleROI, 2), size(sampleROI, 1));
fprintf('  Min/Max intensity: %d / %d\n', min(sampleROI(:)), max(sampleROI(:)));
fprintf('  Mean intensity: %.2f\n', mean(double(sampleROI(:))));
fprintf('  Std intensity: %.2f\n', std(double(sampleROI(:))));
fprintf('  Non-zero pixels: %d (%.1f%%)\n', ...
    nnz(sampleROI), nnz(sampleROI) / numel(sampleROI) * 100);

fprintf('\nCXR (OOD) Sample Image:\n');
fprintf('  Size: %d x %d\n', size(sampleCXR, 2), size(sampleCXR, 1));
fprintf('  Min/Max intensity: %d / %d\n', min(sampleCXR(:)), max(sampleCXR(:)));
fprintf('  Mean intensity: %.2f\n', mean(double(sampleCXR(:))));
fprintf('  Std intensity: %.2f\n', std(double(sampleCXR(:))));
fprintf('  Non-zero pixels: %d (%.1f%%)\n', ...
    nnz(sampleCXR), nnz(sampleCXR) / numel(sampleCXR) * 100);

%% Check size consistency
fprintf('\n=== SIZE CONSISTENCY CHECK ===\n');
roiSizes = zeros(numel(imdsROI.Files), 2);
cxrSizes = zeros(numel(imdsCXR.Files), 2);

fprintf('Checking ROI image sizes...\n');
for i = 1:min(100, numel(imdsROI.Files))  % Check first 100
    img = imread(imdsROI.Files{i});
    roiSizes(i, :) = [size(img, 2), size(img, 1)];
end
uniqueROISizes = unique(roiSizes(1:min(100, numel(imdsROI.Files)), :), 'rows');
fprintf('  Unique ROI sizes (sample): %d\n', size(uniqueROISizes, 1));
if size(uniqueROISizes, 1) == 1
    fprintf('  ✓ All ROI images are %d x %d\n', uniqueROISizes(1, 1), uniqueROISizes(1, 2));
else
    fprintf('  ⚠ Multiple ROI sizes detected\n');
end

fprintf('Checking CXR image sizes...\n');
for i = 1:min(100, numel(imdsCXR.Files))  % Check first 100
    img = imread(imdsCXR.Files{i});
    cxrSizes(i, :) = [size(img, 2), size(img, 1)];
end
uniqueCXRSizes = unique(cxrSizes(1:min(100, numel(imdsCXR.Files)), :), 'rows');
fprintf('  Unique CXR sizes (sample): %d\n', size(uniqueCXRSizes, 1));
if size(uniqueCXRSizes, 1) == 1
    fprintf('  ✓ All CXR images are %d x %d\n', uniqueCXRSizes(1, 1), uniqueCXRSizes(1, 2));
else
    fprintf('  ⚠ Multiple CXR sizes detected\n');
end

%% Compare same filename images
fprintf('\n=== CONTENT COMPARISON ===\n');
% Find common filenames
[~, roiNames, roiExts] = cellfun(@fileparts, imdsROI.Files, 'UniformOutput', false);
[~, cxrNames, cxrExts] = cellfun(@fileparts, imdsCXR.Files, 'UniformOutput', false);

roiFullNames = cellfun(@(n, e) [n, e], roiNames, roiExts, 'UniformOutput', false);
cxrFullNames = cellfun(@(n, e) [n, e], cxrNames, cxrExts, 'UniformOutput', false);

commonNames = intersect(roiFullNames, cxrFullNames);
fprintf('Common filenames: %d\n', numel(commonNames));

if numel(commonNames) > 0
    % Compare first common file
    [~, roiIdx] = ismember(commonNames{1}, roiFullNames);
    [~, cxrIdx] = ismember(commonNames{1}, cxrFullNames);
    
    roiImg = imread(imdsROI.Files{roiIdx});
    cxrImg = imread(imdsCXR.Files{cxrIdx});
    
    if size(roiImg, 3) > 1, roiImg = rgb2gray(roiImg); end
    if size(cxrImg, 3) > 1, cxrImg = rgb2gray(cxrImg); end
    
    % Ensure same size
    if size(roiImg) ~= size(cxrImg)
        cxrImg = imresize(cxrImg, size(roiImg));
    end
    
    % Calculate differences
    diff = double(roiImg) - double(cxrImg);
    mse = mean(diff(:).^2);
    rmse = sqrt(mse);
    maxDiff = max(abs(diff(:)));
    meanDiff = mean(abs(diff(:)));
    
    % Correlation
    correlation = corrcoef(double(roiImg(:)), double(cxrImg(:)));
    correlation = correlation(1, 2);
    
    fprintf('Comparing: %s\n', commonNames{1});
    fprintf('  MSE: %.4f\n', mse);
    fprintf('  RMSE: %.4f\n', rmse);
    fprintf('  Max Absolute Difference: %.2f\n', maxDiff);
    fprintf('  Mean Absolute Difference: %.4f\n', meanDiff);
    fprintf('  Correlation: %.4f\n', correlation);
    
    if mse > 1000 && abs(correlation) < 0.5
        fprintf('\n  ✓ Images are DIFFERENT\n');
        fprintf('    ROI appears to be a cropped lung region\n');
        fprintf('    CXR appears to be the full chest X-ray\n');
        fprintf('    This is CORRECT for ID vs OOD evaluation\n');
    elseif mse < 100 && correlation > 0.9
        fprintf('\n  ⚠ Images are VERY SIMILAR\n');
        fprintf('    They may be the same image\n');
        fprintf('    This would NOT be suitable for ID vs OOD evaluation\n');
    else
        fprintf('\n  ? Images have moderate differences\n');
        fprintf('    Visual inspection recommended\n');
    end
end

%% Class distribution
fprintf('\n=== CLASS DISTRIBUTION ===\n');
for c = 1:numel(classesROI)
    roiCount = sum(imdsROI.Labels == classesROI{c});
    cxrCount = sum(imdsCXR.Labels == classesCXR{c});
    fprintf('  %s: ROI=%d, CXR=%d\n', classesROI{c}, roiCount, cxrCount);
end

%% Summary and Recommendations
fprintf('\n=== SUMMARY AND RECOMMENDATIONS ===\n');
fprintf('\nDataset Configuration:\n');
fprintf('  ID (In-Distribution): ROI images (cropped lung regions)\n');
fprintf('  OOD (Out-of-Distribution): CXR images (full chest X-rays)\n');
fprintf('\n✓ Datasets are suitable for ID/OOD evaluation if:\n');
fprintf('  1. ROI images are cropped lung regions (lower intensity, fewer non-zero pixels)\n');
fprintf('  2. CXR images are full chest X-rays (higher intensity, more content)\n');
fprintf('  3. Both have consistent image sizes (227x227)\n');
fprintf('  4. Both have matching class distributions\n');
fprintf('\n⚠ Note: Model should be trained on ROI images (ID) and evaluated on CXR images (OOD)\n');
fprintf('   to measure domain shift and generalization performance.\n');

fprintf('\n=== ANALYSIS COMPLETE ===\n');

