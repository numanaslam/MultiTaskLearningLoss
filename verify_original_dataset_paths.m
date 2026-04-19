%VERIFY_ORIGINAL_DATASET_PATHS Verify that original dataset paths are correct
%   This script checks if the original dataset directories exist and
%   can be accessed before running the full recreation script.

clc; close all;
fprintf('=== VERIFYING ORIGINAL DATASET PATHS ===\n\n');

%% Configuration
originalCXRDir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/cxr_png';
originalMasksDir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/masks';

%% Check CXR directory
fprintf('Checking CXR directory...\n');
if exist(originalCXRDir, 'dir')
    fprintf('  ✓ CXR directory exists: %s\n', originalCXRDir);
    
    % Count files
    cxrFiles = dir(fullfile(originalCXRDir, '*.png'));
    fprintf('  ✓ Found %d PNG files\n', numel(cxrFiles));
    
    if numel(cxrFiles) > 0
        % Show sample file
        fprintf('  Sample file: %s\n', cxrFiles(1).name);
        
        % Try to read sample
        try
            sampleImg = imread(fullfile(originalCXRDir, cxrFiles(1).name));
            fprintf('  ✓ Sample image readable: %d x %d\n', size(sampleImg, 2), size(sampleImg, 1));
        catch ME
            fprintf('  ✗ Error reading sample: %s\n', ME.message);
        end
    end
else
    fprintf('  ✗ CXR directory NOT FOUND: %s\n', originalCXRDir);
    fprintf('  Please check the path and update the script.\n');
end

%% Check Masks directory
fprintf('\nChecking Masks directory...\n');
if exist(originalMasksDir, 'dir')
    fprintf('  ✓ Masks directory exists: %s\n', originalMasksDir);
    
    % Check for subdirectories
    maskDirs = dir(originalMasksDir);
    subdirs = maskDirs([maskDirs.isdir] & ~strcmp({maskDirs.name}, '.') & ~strcmp({maskDirs.name}, '..'));
    
    if numel(subdirs) > 0
        fprintf('  Found %d subdirectories:\n', numel(subdirs));
        for i = 1:min(5, numel(subdirs))
            fprintf('    - %s\n', subdirs(i).name);
        end
    end
    
    % Count files (including subdirectories)
    maskFiles = dir(fullfile(originalMasksDir, '**', '*.png'));
    fprintf('  ✓ Found %d PNG files (including subdirectories)\n', numel(maskFiles));
    
    if numel(maskFiles) > 0
        % Show sample file
        fprintf('  Sample file: %s\n', maskFiles(1).name);
        
        % Try to read sample
        try
            sampleMask = imread(fullfile(maskFiles(1).folder, maskFiles(1).name));
            fprintf('  ✓ Sample mask readable: %d x %d\n', size(sampleMask, 2), size(sampleMask, 1));
            fprintf('  Mask foreground: %d pixels (%.1f%%)\n', ...
                nnz(sampleMask > 128), nnz(sampleMask > 128) / numel(sampleMask) * 100);
        catch ME
            fprintf('  ✗ Error reading sample: %s\n', ME.message);
        end
    end
else
    fprintf('  ✗ Masks directory NOT FOUND: %s\n', originalMasksDir);
    fprintf('  Please check the path and update the script.\n');
end

%% Check reference directories
fprintf('\nChecking reference directories (input/resized)...\n');
referenceROIDir = fullfile('input', 'resized', 'roi');
referenceCXRDir = fullfile('input', 'resized', 'cxr');

if exist(referenceROIDir, 'dir')
    fprintf('  ✓ Reference ROI directory exists\n');
    roiFiles = dir(fullfile(referenceROIDir, '**', '*.png'));
    fprintf('  ✓ Found %d reference ROI files\n', numel(roiFiles));
else
    fprintf('  ⚠ Reference ROI directory not found (will use CXR reference)\n');
end

if exist(referenceCXRDir, 'dir')
    fprintf('  ✓ Reference CXR directory exists\n');
    cxrFiles = dir(fullfile(referenceCXRDir, '**', '*.png'));
    fprintf('  ✓ Found %d reference CXR files\n', numel(cxrFiles));
else
    fprintf('  ✗ Reference CXR directory not found\n');
    fprintf('  Cannot determine which images to process.\n');
end

%% Summary
fprintf('\n=== VERIFICATION SUMMARY ===\n');
if exist(originalCXRDir, 'dir') && exist(originalMasksDir, 'dir')
    if exist(referenceROIDir, 'dir') || exist(referenceCXRDir, 'dir')
        fprintf('✓ All paths verified. Ready to run recreate_dataset_from_original.m\n');
    else
        fprintf('⚠ Original datasets found, but reference directories missing.\n');
        fprintf('  You may need to manually specify which images to process.\n');
    end
else
    fprintf('✗ Some paths are incorrect. Please update the script.\n');
end

fprintf('\n');

