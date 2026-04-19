% Analyze histograms of ROI vs CXR images
clc; close all;

% Paths
roiDir = fullfile('/Users/numanaslam/Desktop/Research/Research/Matlab/codes/custom network/input/roi/normal');
cxrDir = fullfile('/Users/numanaslam/Desktop/Research/Research/Matlab/codes/custom network/input/cxr/normal');

% Load ROI images
roiFiles = dir(fullfile(roiDir, '*.png'));
if isempty(roiFiles)
    roiFiles = dir(fullfile(roiDir, '*.jpg'));
end
numRefSamples = min(50, numel(roiFiles));
fprintf('Computing reference histogram from %d ROI samples...\n', numRefSamples);

refImages = cell(numRefSamples, 1);
allPixels = [];
for i = 1:numRefSamples
    img = imread(fullfile(roiFiles(i).folder, roiFiles(i).name));
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    refImages{i} = img;
    allPixels = [allPixels; img(:)];
end
refHist = imhist(uint8(allPixels));

% Load a CXR image
cxrFiles = dir(fullfile(cxrDir, '*.png'));
if isempty(cxrFiles)
    cxrFiles = dir(fullfile(cxrDir, '*.jpg'));
end
if isempty(cxrFiles)
    error('No CXR images found');
end

cxrImg = imread(fullfile(cxrFiles(1).folder, cxrFiles(1).name));
if size(cxrImg, 3) > 1
    cxrImg = rgb2gray(cxrImg);
end

% Apply histmatch
cxrMatched = histeq(cxrImg, refHist);

% Plot
f = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');

subplot(2, 3, 1);
imshow(refImages{1});
title('Sample ROI Image');

subplot(2, 3, 2);
imshow(cxrImg);
title('Original CXR Image');

subplot(2, 3, 3);
imshow(cxrMatched);
title('HistMatched CXR Image');

subplot(2, 3, 4);
imhist(refImages{1});
title('ROI Histogram');

subplot(2, 3, 5);
imhist(cxrImg);
title('Original CXR Histogram');

subplot(2, 3, 6);
imhist(cxrMatched);
title('HistMatched CXR Histogram');

saveas(f, 'histogram_analysis.png');
fprintf('Histogram analysis saved to histogram_analysis.png\n');

% Calculate stats
fprintf('ROI Mean Intensity: %.2f\n', mean(refImages{1}(:)));
fprintf('CXR Original Mean: %.2f\n', mean(cxrImg(:)));
fprintf('CXR Matched Mean: %.2f\n', mean(cxrMatched(:)));

% Check for saturation
fprintf('CXR Matched Saturation (255): %.2f%%\n', sum(cxrMatched(:)==255)/numel(cxrMatched)*100);
fprintf('CXR Matched Black (0): %.2f%%\n', sum(cxrMatched(:)==0)/numel(cxrMatched)*100);
