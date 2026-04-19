function run_precompute_gradcam(forceRecalculate)
%RUN_PRECOMPUTE_GRADCAM Wrapper script to precompute GradCAM maps and masks
%   run_precompute_gradcam() - Recalculate GradCAM maps
%   run_precompute_gradcam(true) - Force recalculation (default)
%   run_precompute_gradcam(false) - Use cache if available
%
%   This script loads the dataset and network, then calls
%   precompute_gradcam_and_masks to compute GradCAM maps for all images.

if nargin < 1
    forceRecalculate = true;  % Default to recalculating
end

fprintf('=== PRECOMPUTING GRADCAM MAPS ===\n\n');

% Add paths if needed - ensure src/utils is first in path
addpath('src/utils');

% Clear any cached/compiled versions that might have wrong signature
if exist('precompute_gradcam_and_masks', 'builtin') == 0
    clear precompute_gradcam_and_masks;
end

% Verify we're using the correct function
funcPath = which('precompute_gradcam_and_masks');
if isempty(funcPath)
    error('precompute_gradcam_and_masks function not found. Make sure src/utils is in path.');
end
fprintf('Using function from: %s\n', funcPath);

% Verify function signature
try
    % This will fail if signature is wrong, but we catch it
    nargin('precompute_gradcam_and_masks');
catch
    warning('Function signature check failed. Clearing and retrying...');
    clear precompute_gradcam_and_masks;
    rehash;
end

%% Load Network
fprintf('Loading network...\n');
modelPath = fullfile('models', 'pretrained', 'vgg16_finetuned_on_roi.mat');
if ~exist(modelPath, 'file')
    % Fallback to root directory
    modelPath = 'vgg16_finetuned_on_roi.mat';
end
if ~exist(modelPath, 'file')
    error('Network file not found: %s', modelPath);
end
s = load(modelPath);
vggNet = s.trainedNet;
fprintf('  Network loaded: %s\n', modelPath);

%% Load Dataset
fprintf('Loading dataset...\n');
roiDir = fullfile('input', 'roi');
if ~exist(roiDir, 'dir')
    error('ROI directory not found: %s', roiDir);
end
imds = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imds.Labels);
fprintf('  Dataset loaded: %d samples, %d classes (%s)\n', ...
    numel(imds.Files), numel(classes), strjoin(classes, ', '));

%% Setup Parameters
workingGradCAMLayer = 'relu5_3';
maskDir = fullfile('input', 'masks');
gradCAMCacheFile = 'precomputed_gradcam_maps_enhanced.mat';

%% Check Cache
if ~forceRecalculate && exist(gradCAMCacheFile, 'file')
    fprintf('\nCache file exists: %s\n', gradCAMCacheFile);
    response = input('Recalculate anyway? (y/n): ', 's');
    if strcmpi(response, 'y')
        forceRecalculate = true;
    end
end

%% Precompute GradCAM
if forceRecalculate
    fprintf('\nComputing GradCAM maps (this may take a while)...\n');
    fprintf('Using feature layer: %s\n', workingGradCAMLayer);
    fprintf('Mask directory: %s\n\n', maskDir);
    
    % Call the function with explicit path to avoid conflicts
    [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(imds, vggNet, workingGradCAMLayer, maskDir);
    
    % Save to cache
    cachedFileList = imds.Files;
    save(gradCAMCacheFile, 'precomputedGradCAM', 'precomputedMasks', 'cachedFileList', '-v7.3');
    fprintf('\n✓ GradCAM maps saved to: %s\n', gradCAMCacheFile);
else
    fprintf('\nUsing cached GradCAM maps from: %s\n', gradCAMCacheFile);
    if exist(gradCAMCacheFile, 'file')
        cache = load(gradCAMCacheFile);
        fprintf('  Cached maps: %d images\n', numel(cache.precomputedGradCAM));
    else
        fprintf('  Warning: Cache file not found. Run with forceRecalculate=true\n');
    end
end

fprintf('\n=== COMPLETE ===\n');

end

