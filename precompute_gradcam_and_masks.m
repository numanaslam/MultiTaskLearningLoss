function [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir)
%PRECOMPUTE_GRADCAM_AND_MASKS Compute GradCAM maps and load masks from disk
%   [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(imds, vggNet, workingGradCAMLayer, maskDir, gradcamMasksDir)
%
%   This function:
%   1. COMPUTES GradCAM maps for all images (actual computation)
%   2. LOADS masks from disk (input/masks/) - masks are NOT calculated, just loaded
%   3. Optionally saves computed GradCAM maps to gradcamMasksDir
%   
%   Masks are loaded from maskDir with '_mask' suffix (e.g., CHNCXR_0001_0_mask.png)
%   GradCAM maps are saved to gradcamMasksDir (optional, can be empty)
%
%   Inputs:
%       imds - ImageDatastore containing the images
%       vggNet - Pre-trained VGG16 network (dlnetwork)
%       workingGradCAMLayer - Feature layer name for GradCAM (e.g., 'relu5_3')
%       maskDir - Directory containing mask images (e.g., 'input/masks')
%       gradcamMasksDir - (Optional) Directory to save computed GradCAM maps
%
%   Outputs:
%       precomputedGradCAM - Cell array of GradCAM maps (one per image)
%       precomputedMasks - Cell array of binary masks (one per image)
%
%   Example:
%       [precomputedGradCAM, precomputedMasks] = precompute_gradcam_and_masks(...
%           imds, vggNet, 'relu5_3', fullfile('input', 'masks'), fullfile('input', 'gradcam_masks'));

if nargin < 5
    gradcamMasksDir = '';  % Optional: don't save GradCAM maps if not provided
end

fprintf('  Computing GradCAM maps and loading masks for %d images...\n', numel(imds.Files));
fprintf('  (GradCAM: computed | Masks: loaded from %s)\n', maskDir);
if ~isempty(gradcamMasksDir)
    fprintf('  (GradCAM maps will be saved to: %s)\n', gradcamMasksDir);
end
fprintf('  [DEBUG] Current working directory: %s\n', pwd);
fprintf('  [DEBUG] maskDir: %s\n', maskDir);
fprintf('  [DEBUG] maskDir exists: %d (7=dir exists, 0=not found)\n', exist(maskDir, 'dir'));
precomputedGradCAM = cell(numel(imds.Files), 1);
precomputedMasks = cell(numel(imds.Files), 1);

useGPU = canUseGPU;

for i = 1:numel(imds.Files)
    try
        % Load and preprocess image
        img = imread(imds.Files{i});
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);  % Convert grayscale to RGB
        end
        img = imresize(img, [224 224]);
        
        % Ensure image is uint8 for gradCAM
        if ~isa(img, 'uint8')
            img = uint8(img);
        end
        
        % Generate GradCAM with explicit true-label targeting.
        trueLabel = char(string(imds.Labels(i)));
        gradCAMMap = gradCAM(vggNet, img, trueLabel, 'FeatureLayer', workingGradCAMLayer);
        
        % Ensure map is on CPU and properly sized
        if useGPU && isa(gradCAMMap, 'gpuArray')
            gradCAMMap = gather(gradCAMMap);
        end
        gradCAMMap = double(gradCAMMap);

        % Robust CAM post-processing:
        % 1) smooth high-frequency noise
        % 2) suppress diffuse low-activation background
        % 3) renormalize to [0, 1]
        gradCAMMap = postprocess_gradcam_map(gradCAMMap);
        
        precomputedGradCAM{i} = gradCAMMap;
        
        % Optionally save GradCAM map to disk
        if ~isempty(gradcamMasksDir)
            [~, name, ext] = fileparts(imds.Files{i});
            [imgDir, ~] = fileparts(imds.Files{i});
            [~, subdir] = fileparts(imgDir);
            
            % Create subdirectory if it doesn't exist
            gradcamSubdir = fullfile(gradcamMasksDir, subdir);
            if ~exist(gradcamSubdir, 'dir')
                mkdir(gradcamSubdir);
            end
            
            % Save GradCAM map as image
            gradcamPath = fullfile(gradcamSubdir, [name '_gradcam' ext]);
            % Normalize to 0-255 for saving
            gradcamImg = uint8(255 * gradCAMMap);
            imwrite(gradcamImg, gradcamPath);
        end
        
        % Load corresponding mask from disk (NOT calculating - just loading existing file)
        % Mask files have '_mask' suffix before extension (e.g., CHNCXR_0001_0_mask.png)
        [~, name, ext] = fileparts(imds.Files{i});
        [imgDir, ~] = fileparts(imds.Files{i});  % Get directory of image
        [~, subdir] = fileparts(imgDir);  % Get subdirectory name (normal or ptb)
        
        % Construct mask path - handle both relative and absolute paths
        if isempty(fileparts(maskDir)) || strcmp(fileparts(maskDir), '.')
            % maskDir is relative, construct relative path
            maskPath = fullfile(maskDir, subdir, [name '_mask' ext]);
        else
            % maskDir is absolute, use as-is
            maskPath = fullfile(maskDir, subdir, [name '_mask' ext]);
        end
        
        % DEBUG: Print detailed mask path info for first few images
        if i <= 5
            fprintf('    [DEBUG MASK PATH] Image %d:\n', i);
            fprintf('      Image file: %s\n', imds.Files{i});
            fprintf('      Extracted name: %s, ext: %s, subdir: %s\n', name, ext, subdir);
            fprintf('      maskDir: %s\n', maskDir);
            fprintf('      Constructed maskPath: %s\n', maskPath);
            fprintf('      File exists check: %d (2=file, 7=folder, 0=not found)\n', exist(maskPath, 'file'));
            
            % Also try alternative paths to help debug
            altPath1 = fullfile(maskDir, subdir, [name ext]);  % Without _mask
            fprintf('      Alt path (no _mask): %s exists=%d\n', altPath1, exist(altPath1, 'file'));
            
            % Check if maskDir/subdir exists
            subdirPath = fullfile(maskDir, subdir);
            fprintf('      Subdir exists: %s exists=%d\n', subdirPath, exist(subdirPath, 'dir'));
            
            % List first few files in subdir if it exists
            if exist(subdirPath, 'dir')
                files = dir(fullfile(subdirPath, '*.png'));
                if numel(files) > 0
                    fprintf('      First 3 files in subdir: %s', files(1).name);
                    if numel(files) > 1
                        fprintf(', %s', files(2).name);
                    end
                    if numel(files) > 2
                        fprintf(', %s', files(3).name);
                    end
                    fprintf('\n');
                else
                    fprintf('      No PNG files found in subdir!\n');
                end
            else
                fprintf('      ERROR: Subdir does not exist!\n');
            end
        end
        
        if exist(maskPath, 'file')
            mask = imread(maskPath);
            if size(mask, 3) > 1
                mask = rgb2gray(mask);
            end
            mask = imbinarize(mask);
            mask = imresize(mask, [224 224], 'nearest');
            precomputedMasks{i} = logical(mask);
            
            % DEBUG: Check if mask has foreground pixels
            if i <= 3
                numForeground = sum(mask(:));
                fprintf('    [DEBUG] Mask loaded: %d foreground pixels (%.2f%%)\n', ...
                    numForeground, 100*numForeground/numel(mask));
            end
        else
            % Create empty mask if not found
            precomputedMasks{i} = false(224, 224);
            if i <= 3
                fprintf('    [DEBUG] WARNING: Mask file NOT FOUND! Creating empty mask.\n');
            end
        end
        
        if mod(i, 50) == 0
            fprintf('    Processed %d/%d images\n', i, numel(imds.Files));
        end
        
    catch ME
        fprintf('    Warning: Error processing image %d (%s): %s\n', i, imds.Files{i}, ME.message);
        % Create fallback values
        precomputedGradCAM{i} = 0.5 * ones(224, 224);
        precomputedMasks{i} = false(224, 224);
    end
end

fprintf('  GradCAM computation complete!\n');
end

function cam = postprocess_gradcam_map(cam)
    if isempty(cam) || all(~isfinite(cam(:)))
        cam = 0.5 * ones(224, 224);
        return;
    end

    cam(~isfinite(cam)) = 0;
    cam = max(cam, 0);

    % Mild smoothing improves consistency for noisy PTB heatmaps.
    cam = imgaussfilt(cam, 1.0);

    % Normalize first.
    cmin = min(cam(:));
    cmax = max(cam(:));
    if cmax > cmin
        cam = (cam - cmin) / (cmax - cmin + eps);
    else
        cam = zeros(size(cam));
    end

    % Suppress low-importance diffuse background.
    bg = prctile(cam(:), 60);
    cam = max(cam - bg, 0);

    % Final renormalization.
    cmax = max(cam(:));
    if cmax > 0
        cam = cam / (cmax + eps);
    end
end
