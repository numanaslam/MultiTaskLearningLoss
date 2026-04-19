function visualize_precomputed_gradcam()
%VISUALIZE_PRECOMPUTED_GRADCAM Visualize GradCAM maps to verify correctness
%   This script loads the pre-trained model, computes GradCAM maps for a few
%   random samples, and displays them alongside the original images and masks.

clc; close all;

% Configuration
modelPath = fullfile('models', 'pretrained', 'vgg16_finetuned_on_roi.mat');
roiDir = fullfile('input', 'roi');
maskDir = fullfile('input', 'masks');
outputDir = 'results_visualizations';
featureLayer = 'relu5_3';
numSamples = 5;

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('=== VISUALIZING GRADCAM MAPS ===\n');

% 1. Load Model
if ~exist(modelPath, 'file')
    error('Model file not found: %s', modelPath);
end
fprintf('Loading model from: %s\n', modelPath);
data = load(modelPath);
if isfield(data, 'trainedNet')
    net = data.trainedNet;
else
    error('Variable ''trainedNet'' not found in model file.');
end

% 2. Load Dataset
if ~exist(roiDir, 'dir')
    error('ROI directory not found: %s', roiDir);
end
fprintf('Loading dataset from: %s\n', roiDir);
imds = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
numImages = numel(imds.Files);
fprintf('  Found %d images.\n', numImages);

% 3. Select Random Samples
rng('shuffle');
indices = randperm(numImages, numSamples);

% 4. Visualize
figure('Position', [100, 100, 1200, 300 * numSamples], 'Visible', 'off');
tiledlayout(numSamples, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:numSamples
    idx = indices(i);
    imgPath = imds.Files{idx};
    [~, name, ext] = fileparts(imgPath);
    [imgDir, ~] = fileparts(imgPath);
    [~, subdir] = fileparts(imgDir);
    
    % Load Image
    img = imread(imgPath);
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    img = imresize(img, [224 224]);
    
    % Compute GradCAM
    label = imds.Labels(idx);
    try
        % Convert to dlnetwork if needed (gradCAM supports both but dlnetwork is preferred)
        if isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
            % Use activation-based method if gradCAM function not available for DAGNetwork
            % But here we assume gradCAM works as in precompute script
            map = gradCAM(net, img, char(label), 'FeatureLayer', featureLayer);
        else
            map = gradCAM(net, dlarray(single(img),'SSCB'), label, 'FeatureLayer', featureLayer);
            map = extractdata(map);
        end
    catch ME
        warning('GradCAM failed: %s. Using dummy map.', ME.message);
        map = zeros(224, 224);
    end
    
    if size(map, 1) ~= 224
        map = imresize(map, [224 224]);
    end
    
    % Normalize Map
    map = double(map);
    map = (map - min(map(:))) / (max(map(:)) - min(map(:)) + eps);
    
    % Load Mask
    maskPath = fullfile(maskDir, subdir, [name '_mask' ext]);
    if exist(maskPath, 'file')
        mask = imread(maskPath);
        if size(mask, 3) > 1, mask = rgb2gray(mask); end
        mask = imbinarize(mask);
        mask = imresize(mask, [224 224], 'nearest');
    else
        mask = false(224, 224);
        warning('Mask not found: %s', maskPath);
    end
    
    % Create Overlay
    heatmap = jet(256);
    % Convert map to index
    mapIdx = gray2ind(map, 256);
    rgbMap = ind2rgb(mapIdx, heatmap);
    overlay = 0.5 * double(img)/255 + 0.5 * rgbMap;
    
    % Plot
    % 1. Original
    nexttile;
    imshow(img);
    title(sprintf('%s (%s)', name, char(label)), 'Interpreter', 'none');
    if i==1, ylabel('Original'); end
    
    % 2. GradCAM
    nexttile;
    imshow(map, []);
    colormap(gca, 'jet');
    title('GradCAM (Teacher)');
    
    % 3. Mask
    nexttile;
    imshow(mask);
    title('Lung Mask (Ground Truth)');
    
    % 4. Overlay
    nexttile;
    imshow(overlay);
    title('GradCAM Overlay');
end

% Save Figure
outputFile = fullfile(outputDir, 'gradcam_verification.png');
exportgraphics(gcf, outputFile, 'Resolution', 150);
fprintf('Visualization saved to: %s\n', outputFile);

end
