%RECREATE_DATASET_FROM_ORIGINAL Recreate ROI, masks, and CXR datasets from original sources
%   This script:
%   1. Reads image names from input/resized directories (to know which images to process)
%   2. Loads original CXR images and masks from the original dataset
%   3. Creates ROI images by cropping using masks
%   4. Resizes all images to 227x227
%   5. Saves to input/resized/roi, input/resized/masks, input/resized/cxr
%
%   Original dataset locations:
%   - CXR: /Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/cxr_png
%   - Masks: /Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/masks

clc; close all;
fprintf('=== RECREATING DATASET FROM ORIGINAL SOURCES ===\n\n');

%% Configuration
% Original dataset paths
originalCXRDir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/cxr_png';
originalMasksDir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/masks';

% Output directories - using new folder to compare with old version
outputBaseDir = fullfile('input', 'resized_new');
outputROIDir = fullfile(outputBaseDir, 'roi');
outputMasksDir = fullfile(outputBaseDir, 'masks');
outputCXRDir = fullfile(outputBaseDir, 'cxr');

% Target size
targetSize = [227, 227];

% Create output directories
classes = {'normal', 'ptb'};
for c = 1:numel(classes)
    className = classes{c};
    if ~exist(fullfile(outputROIDir, className), 'dir')
        mkdir(fullfile(outputROIDir, className));
    end
    if ~exist(fullfile(outputMasksDir, className), 'dir')
        mkdir(fullfile(outputMasksDir, className));
    end
    if ~exist(fullfile(outputCXRDir, className), 'dir')
        mkdir(fullfile(outputCXRDir, className));
    end
end

%% Get reference image names from existing resized directory
fprintf('Reading reference image names from input/resized...\n');
referenceROIDir = fullfile('input', 'resized', 'roi');
referenceCXRDir = fullfile('input', 'resized', 'cxr');

if ~exist(referenceROIDir, 'dir') || ~exist(referenceCXRDir, 'dir')
    error('Reference directories not found. Please ensure input/resized/roi and input/resized/cxr exist.');
end

% Get all image names from reference directories
allImageNames = struct('name', {}, 'class', {});
classes = {'normal', 'ptb'};

for c = 1:numel(classes)
    className = classes{c};
    roiRefDir = fullfile(referenceROIDir, className);
    cxrRefDir = fullfile(referenceCXRDir, className);
    
    if exist(roiRefDir, 'dir')
        roiFiles = dir(fullfile(roiRefDir, '*.png'));
        for i = 1:numel(roiFiles)
            if ~roiFiles(i).isdir
                allImageNames(end+1) = struct('name', roiFiles(i).name, 'class', className);
            end
        end
    end
    
    if exist(cxrRefDir, 'dir')
        cxrFiles = dir(fullfile(cxrRefDir, '*.png'));
        for i = 1:numel(cxrFiles)
            if ~cxrFiles(i).isdir
                % Check if already added
                existingNames = {allImageNames.name};
                if ~any(strcmp(existingNames, cxrFiles(i).name))
                    allImageNames(end+1) = struct('name', cxrFiles(i).name, 'class', className);
                end
            end
        end
    end
end

fprintf('Found %d unique image names to process\n\n', numel(allImageNames));

%% Process each image
stats = struct();
stats.total = numel(allImageNames);
stats.processed = 0;
stats.roi_created = 0;
stats.mask_created = 0;
stats.cxr_created = 0;
stats.errors = 0;
stats.missing_original = 0;
stats.missing_mask = 0;

fprintf('Processing images...\n');

for imgIdx = 1:numel(allImageNames)
    imgInfo = allImageNames(imgIdx);
    imgName = imgInfo.name;
    className = imgInfo.class;
    
    if mod(imgIdx, 50) == 0
        fprintf('  Processing %d/%d: %s\n', imgIdx, numel(allImageNames), imgName);
    end
    
    try
        % Construct original file paths
        % Original CXR image
        originalCXRPath = fullfile(originalCXRDir, imgName);
        
        % Original mask (may have _mask suffix or not)
        [~, nameBase, ext] = fileparts(imgName);
        maskName1 = [nameBase, '_mask', ext];  % Try with _mask suffix
        maskName2 = imgName;  % Try without suffix
        
        originalMaskPath1 = fullfile(originalMasksDir, maskName1);
        originalMaskPath2 = fullfile(originalMasksDir, maskName2);
        
        % Check which mask file exists
        if exist(originalMaskPath1, 'file')
            originalMaskPath = originalMaskPath1;
        elseif exist(originalMaskPath2, 'file')
            originalMaskPath = originalMaskPath2;
        else
            % Try in subdirectories
            maskDirs = dir(originalMasksDir);
            maskPath = '';
            for d = 1:numel(maskDirs)
                if maskDirs(d).isdir && ~strcmp(maskDirs(d).name, '.') && ~strcmp(maskDirs(d).name, '..')
                    testPath1 = fullfile(originalMasksDir, maskDirs(d).name, maskName1);
                    testPath2 = fullfile(originalMasksDir, maskDirs(d).name, maskName2);
                    if exist(testPath1, 'file')
                        maskPath = testPath1;
                        break;
                    elseif exist(testPath2, 'file')
                        maskPath = testPath2;
                        break;
                    end
                end
            end
            originalMaskPath = maskPath;
        end
        
        % Check if original files exist
        if ~exist(originalCXRPath, 'file')
            warning('Original CXR not found: %s', originalCXRPath);
            stats.missing_original = stats.missing_original + 1;
            continue;
        end
        
        if isempty(originalMaskPath) || ~exist(originalMaskPath, 'file')
            warning('Original mask not found for: %s (tried %s and %s)', imgName, maskName1, maskName2);
            stats.missing_mask = stats.missing_mask + 1;
            % Continue without mask - will create CXR only
        end
        
        % Load original CXR image
        cxrOriginal = imread(originalCXRPath);
        if size(cxrOriginal, 3) > 1
            cxrOriginal = rgb2gray(cxrOriginal);
        end
        % Normalize to full 0-255 range before converting to uint8. Many of the
        % Kaggle masks/CXRs are stored as 12/16-bit PNGs; a direct uint8 cast
        % was discarding the high bits and making lungs appear very dark.
        if ~isa(cxrOriginal, 'uint8')
            cxrOriginal = double(cxrOriginal);
            cxrOriginal = cxrOriginal - min(cxrOriginal(:));
            maxVal = max(cxrOriginal(:));
            if maxVal > 0
                cxrOriginal = cxrOriginal / maxVal * 255;
            end
            cxrOriginal = uint8(round(cxrOriginal));
        end
        
        % Load mask if available
        maskOriginal = [];
        if ~isempty(originalMaskPath) && exist(originalMaskPath, 'file')
            maskOriginal = imread(originalMaskPath);
            if size(maskOriginal, 3) > 1
                maskOriginal = rgb2gray(maskOriginal);
            end
            % Binarize mask (threshold at 128)
            maskOriginal = maskOriginal > 128;
            
            % Ensure mask and CXR have same size
            if size(maskOriginal, 1) ~= size(cxrOriginal, 1) || ...
               size(maskOriginal, 2) ~= size(cxrOriginal, 2)
                maskOriginal = imresize(double(maskOriginal), [size(cxrOriginal, 1), size(cxrOriginal, 2)], 'nearest') > 0.5;
            end
        end
        
        % Create ROI by applying mask to CXR (lungs only region)
        if ~isempty(maskOriginal) && any(maskOriginal(:))
            % Apply mask: keep lung regions, set non-lung to background (0)
            roiImage = cxrOriginal;
            roiImage(~maskOriginal) = 0;  % Mask out non-lung regions
            
            % Get bounding box of mask for cropping
            [rows, cols] = find(maskOriginal);
            if ~isempty(rows) && ~isempty(cols)
                minRow = max(1, min(rows) - 5);  % Add small padding
                maxRow = min(size(cxrOriginal, 1), max(rows) + 5);
                minCol = max(1, min(cols) - 5);
                maxCol = min(size(cxrOriginal, 2), max(cols) + 5);
                
                % Crop ROI (now contains only lung regions with background)
                roiImage = roiImage(minRow:maxRow, minCol:maxCol);
                maskCropped = maskOriginal(minRow:maxRow, minCol:maxCol);
            else
                % Empty mask - use center crop
                [h, w] = size(cxrOriginal);
                centerH = round(h/2);
                centerW = round(w/2);
                cropSize = min(h, w) * 0.6;  % 60% of smaller dimension
                minRow = max(1, round(centerH - cropSize/2));
                maxRow = min(h, round(centerH + cropSize/2));
                minCol = max(1, round(centerW - cropSize/2));
                maxCol = min(w, round(centerW + cropSize/2));
                roiImage = cxrOriginal(minRow:maxRow, minCol:maxCol);
                maskCropped = maskOriginal(minRow:maxRow, minCol:maxCol);
            end
        else
            % No mask - use center crop (fallback)
            [h, w] = size(cxrOriginal);
            centerH = round(h/2);
            centerW = round(w/2);
            cropSize = min(h, w) * 0.6;  % 60% of smaller dimension
            minRow = max(1, round(centerH - cropSize/2));
            maxRow = min(h, round(centerH + cropSize/2));
            minCol = max(1, round(centerW - cropSize/2));
            maxCol = min(w, round(centerW + cropSize/2));
            roiImage = cxrOriginal(minRow:maxRow, minCol:maxCol);
            if ~isempty(maskOriginal)
                maskCropped = maskOriginal(minRow:maxRow, minCol:maxCol);
            else
                maskCropped = [];
            end
        end
        
        % Resize images in double precision but keep the 0-255 range, then cast back.
        roiResized = imresize(double(roiImage), targetSize, 'bicubic');
        roiResized = uint8(min(max(round(roiResized), 0), 255));
        
        cxrResized = imresize(double(cxrOriginal), targetSize, 'bicubic');
        cxrResized = uint8(min(max(round(cxrResized), 0), 255));
        
        % For mask: use 'nearest' to preserve binary nature
        if ~isempty(maskCropped)
            maskResized = imresize(double(maskCropped), targetSize, 'nearest') > 0.5;
        else
            % Create empty mask if none available
            maskResized = false(targetSize);
        end
        
        % Ensure correct data types
        % imresize preserves input type, but ensure uint8 for images and logical for masks
        if ~isa(roiResized, 'uint8')
            roiResized = uint8(roiResized);
        end
        if ~isa(cxrResized, 'uint8')
            cxrResized = uint8(cxrResized);
        end
        if ~isa(maskResized, 'logical')
            maskResized = logical(maskResized);
        end
        
        % Save ROI image
        roiOutputPath = fullfile(outputROIDir, className, imgName);
        imwrite(roiResized, roiOutputPath);
        stats.roi_created = stats.roi_created + 1;
        
        % Save mask image
        maskOutputName = [nameBase, '_mask', ext];
        maskOutputPath = fullfile(outputMasksDir, className, maskOutputName);
        imwrite(uint8(maskResized) * 255, maskOutputPath);  % Save as uint8 (0 or 255)
        stats.mask_created = stats.mask_created + 1;
        
        % Save CXR image
        cxrOutputPath = fullfile(outputCXRDir, className, imgName);
        imwrite(cxrResized, cxrOutputPath);
        stats.cxr_created = stats.cxr_created + 1;
        
        stats.processed = stats.processed + 1;
        
    catch ME
        warning('Error processing %s: %s', imgName, ME.message);
        stats.errors = stats.errors + 1;
    end
end

%% Print summary
fprintf('\n=== PROCESSING SUMMARY ===\n');
fprintf('Total images to process: %d\n', stats.total);
fprintf('Successfully processed: %d\n', stats.processed);
fprintf('ROI images created: %d\n', stats.roi_created);
fprintf('Mask images created: %d\n', stats.mask_created);
fprintf('CXR images created: %d\n', stats.cxr_created);
fprintf('Errors: %d\n', stats.errors);
fprintf('Missing original CXR: %d\n', stats.missing_original);
fprintf('Missing masks: %d\n', stats.missing_mask);

fprintf('\n=== VALIDATION ===\n');
validate_recreated_dataset(outputROIDir, outputMasksDir, outputCXRDir);

fprintf('\n=== DATASET RECREATION COMPLETE ===\n');
fprintf('All images have been recreated and saved to: %s\n', outputBaseDir);
fprintf('All images have been standardized to:\n');
fprintf('  - Size: %d x %d\n', targetSize(1), targetSize(2));
fprintf('  - Format: Grayscale uint8\n');
fprintf('  - ROI: Cropped using masks (or center crop if mask unavailable)\n');
fprintf('  - CXR: Full images resized with bicubic interpolation\n');
fprintf('  - Masks: Binary masks resized with nearest neighbor\n');
fprintf('\nNote: New images saved to "%s" to compare with old "input/resized"\n', outputBaseDir);

%% Helper function to validate recreated dataset
function validate_recreated_dataset(roiDir, masksDir, cxrDir)
    classes = {'normal', 'ptb'};
    
    for c = 1:numel(classes)
        className = classes{c};
        roiClassDir = fullfile(roiDir, className);
        masksClassDir = fullfile(masksDir, className);
        cxrClassDir = fullfile(cxrDir, className);
        
        fprintf('\n%s class:\n', className);
        
        % Count files
        if exist(roiClassDir, 'dir')
            roiFiles = dir(fullfile(roiClassDir, '*.png'));
            fprintf('  ROI files: %d\n', numel(roiFiles));
        else
            fprintf('  ROI directory not found\n');
        end
        
        if exist(masksClassDir, 'dir')
            maskFiles = dir(fullfile(masksClassDir, '*_mask.png'));
            fprintf('  Mask files: %d\n', numel(maskFiles));
        else
            fprintf('  Masks directory not found\n');
        end
        
        if exist(cxrClassDir, 'dir')
            cxrFiles = dir(fullfile(cxrClassDir, '*.png'));
            fprintf('  CXR files: %d\n', numel(cxrFiles));
        else
            fprintf('  CXR directory not found\n');
        end
        
        % Check sample image sizes
        if exist(roiClassDir, 'dir') && numel(roiFiles) > 0
            sampleROI = imread(fullfile(roiClassDir, roiFiles(1).name));
            fprintf('  ROI sample size: %d x %d\n', size(sampleROI, 2), size(sampleROI, 1));
        end
        
        if exist(cxrClassDir, 'dir') && numel(cxrFiles) > 0
            sampleCXR = imread(fullfile(cxrClassDir, cxrFiles(1).name));
            fprintf('  CXR sample size: %d x %d\n', size(sampleCXR, 2), size(sampleCXR, 1));
        end
        
        if exist(masksClassDir, 'dir') && numel(maskFiles) > 0
            sampleMask = imread(fullfile(masksClassDir, maskFiles(1).name));
            fprintf('  Mask sample size: %d x %d\n', size(sampleMask, 2), size(sampleMask, 1));
            fprintf('  Mask foreground pixels: %d (%.1f%%)\n', ...
                nnz(sampleMask > 128), nnz(sampleMask > 128) / numel(sampleMask) * 100);
        end
    end
end
