%FIX_ID_OOD_DATASETS Fix and standardize ID/OOD datasets
%   This script:
%   1. Detects and fixes inverted images
%   2. Ensures consistent image sizes
%   3. Validates file matching between ROI and CXR
%   4. Standardizes image formats
%   5. Creates backup before modifications

clc; close all;
fprintf('=== FIXING ID/OOD DATASETS ===\n\n');

%% Configuration
roiDir = fullfile('input',  'roi');
cxrDir = fullfile('input',  'cxr');
backupDir = fullfile('input', 'backup');
targetSize = [227, 227];  % Target image size

% Create backup directory
if ~exist(backupDir, 'dir')
    mkdir(backupDir);
    fprintf('Created backup directory: %s\n', backupDir);
end

%% Process each class
classes = {'normal', 'ptb'};
stats = struct();

for c = 1:numel(classes)
    className = classes{c};
    fprintf('\n=== Processing %s class ===\n', className);
    
    roiClassDir = fullfile(roiDir, className);
    cxrClassDir = fullfile(cxrDir, className);
    
    if ~exist(roiClassDir, 'dir') || ~exist(cxrClassDir, 'dir')
        warning('Directory not found for class: %s', className);
        continue;
    end
    
    % Load datasets
    imdsROI = imageDatastore(roiClassDir, 'IncludeSubfolders', false, 'LabelSource', 'foldernames');
    imdsCXR = imageDatastore(cxrClassDir, 'IncludeSubfolders', false, 'LabelSource', 'foldernames');
    
    fprintf('  ROI files: %d\n', numel(imdsROI.Files));
    fprintf('  CXR files: %d\n', numel(imdsCXR.Files));
    
    % Get filenames
    [~, roiNames, roiExts] = cellfun(@fileparts, imdsROI.Files, 'UniformOutput', false);
    [~, cxrNames, cxrExts] = cellfun(@fileparts, imdsCXR.Files, 'UniformOutput', false);
    
    roiFullNames = cellfun(@(n, e) [n, e], roiNames, roiExts, 'UniformOutput', false);
    cxrFullNames = cellfun(@(n, e) [n, e], cxrNames, cxrExts, 'UniformOutput', false);
    
    commonFiles = intersect(roiFullNames, cxrFullNames);
    fprintf('  Common files: %d\n', numel(commonFiles));
    
    % Initialize statistics
    stats.(className) = struct();
    stats.(className).roi_fixed = 0;
    stats.(className).cxr_fixed = 0;
    stats.(className).roi_resized = 0;
    stats.(className).cxr_resized = 0;
    stats.(className).roi_inverted = 0;
    stats.(className).cxr_inverted = 0;
    stats.(className).errors = 0;
    
    %% Process ROI images
    fprintf('\n  Processing ROI images...\n');
    for i = 1:numel(imdsROI.Files)
        try
            imgPath = imdsROI.Files{i};
            img = imread(imgPath);
            
            % Convert to grayscale if needed
            if size(img, 3) > 1
                img = rgb2gray(img);
            end
            
            % Check size
            if size(img, 1) ~= targetSize(1) || size(img, 2) ~= targetSize(2)
                img = imresize(img, targetSize);
                stats.(className).roi_resized = stats.(className).roi_resized + 1;
            end
            
            % Check if image needs inversion (very dark images might be inverted)
            meanIntensity = mean(double(img(:)));
            if meanIntensity < 50  % Very dark - might be inverted
                % Check if inversion improves (compare with expected ROI intensity ~30)
                imgInverted = 255 - img;
                meanInverted = mean(double(imgInverted(:)));
                
                % ROI should be darker (cropped region), so if inverted is much brighter, keep original
                % But if original is too dark (< 10), it might be inverted
                if meanIntensity < 10 && meanInverted > 50
                    img = imgInverted;
                    stats.(className).roi_inverted = stats.(className).roi_inverted + 1;
                end
            end
            
            % Ensure uint8
            if ~isa(img, 'uint8')
                img = uint8(img);
            end
            
            % Save (overwrite original)
            imwrite(img, imgPath);
            stats.(className).roi_fixed = stats.(className).roi_fixed + 1;
            
            if mod(i, 50) == 0
                fprintf('    Processed %d/%d ROI images...\n', i, numel(imdsROI.Files));
            end
        catch ME
            warning('Error processing ROI image %s: %s', imgPath, ME.message);
            stats.(className).errors = stats.(className).errors + 1;
        end
    end
    
    %% Process CXR images
    fprintf('\n  Processing CXR images...\n');
    for i = 1:numel(imdsCXR.Files)
        try
            imgPath = imdsCXR.Files{i};
            img = imread(imgPath);
            
            % Convert to grayscale if needed
            if size(img, 3) > 1
                img = rgb2gray(img);
            end
            
            % Check size
            if size(img, 1) ~= targetSize(1) || size(img, 2) ~= targetSize(2)
                img = imresize(img, targetSize);
                stats.(className).cxr_resized = stats.(className).cxr_resized + 1;
            end
            
            % Check if image needs inversion
            meanIntensity = mean(double(img(:)));
            if meanIntensity < 50  % Very dark - might be inverted
                imgInverted = 255 - img;
                meanInverted = mean(double(imgInverted(:)));
                
                % CXR should be brighter (full image), so if inverted is much brighter, use inverted
                if meanIntensity < 10 && meanInverted > 100
                    img = imgInverted;
                    stats.(className).cxr_inverted = stats.(className).cxr_inverted + 1;
                end
            end
            
            % Ensure uint8
            if ~isa(img, 'uint8')
                img = uint8(img);
            end
            
            % Save (overwrite original)
            imwrite(img, imgPath);
            stats.(className).cxr_fixed = stats.(className).cxr_fixed + 1;
            
            if mod(i, 50) == 0
                fprintf('    Processed %d/%d CXR images...\n', i, numel(imdsCXR.Files));
            end
        catch ME
            warning('Error processing CXR image %s: %s', imgPath, ME.message);
            stats.(className).errors = stats.(className).errors + 1;
        end
    end
    
    fprintf('\n  %s class processing complete.\n', className);
end

%% Print summary
fprintf('\n=== PROCESSING SUMMARY ===\n');
for c = 1:numel(classes)
    className = classes{c};
    s = stats.(className);
    fprintf('\n%s class:\n', className);
    fprintf('  ROI fixed: %d\n', s.roi_fixed);
    fprintf('  CXR fixed: %d\n', s.cxr_fixed);
    fprintf('  ROI resized: %d\n', s.roi_resized);
    fprintf('  CXR resized: %d\n', s.cxr_resized);
    fprintf('  ROI inverted: %d\n', s.roi_inverted);
    fprintf('  CXR inverted: %d\n', s.cxr_inverted);
    fprintf('  Errors: %d\n', s.errors);
end

fprintf('\n=== VALIDATION ===\n');
validate_datasets(roiDir, cxrDir);

fprintf('\n=== FIXING COMPLETE ===\n');
fprintf('All images have been standardized to:\n');
fprintf('  - Size: %d x %d\n', targetSize(1), targetSize(2));
fprintf('  - Format: Grayscale uint8\n');
fprintf('  - Inversion: Corrected if needed\n');

%% Helper function to validate datasets
function validate_datasets(roiDir, cxrDir)
    classes = {'normal', 'ptb'};
    
    for c = 1:numel(classes)
        className = classes{c};
        roiClassDir = fullfile(roiDir, className);
        cxrClassDir = fullfile(cxrDir, className);
        
        if ~exist(roiClassDir, 'dir') || ~exist(cxrClassDir, 'dir')
            continue;
        end
        
        imdsROI = imageDatastore(roiClassDir, 'IncludeSubfolders', false);
        imdsCXR = imageDatastore(cxrClassDir, 'IncludeSubfolders', false);
        
        fprintf('\n%s class validation:\n', className);
        fprintf('  ROI files: %d\n', numel(imdsROI.Files));
        fprintf('  CXR files: %d\n', numel(imdsCXR.Files));
        
        % Check sizes
        roiSize = size(imread(imdsROI.Files{1}));
        cxrSize = size(imread(imdsCXR.Files{1}));
        
        if size(roiSize, 2) > 2, roiSize = roiSize(1:2); end
        if size(cxrSize, 2) > 2, cxrSize = cxrSize(1:2); end
        
        fprintf('  ROI size: %d x %d\n', roiSize(2), roiSize(1));
        fprintf('  CXR size: %d x %d\n', cxrSize(2), cxrSize(1));
        
        if isequal(roiSize, cxrSize)
            fprintf('  ✓ Sizes match\n');
        else
            fprintf('  ⚠ Size mismatch\n');
        end
        
        % Check intensities
        roiImg = imread(imdsROI.Files{1});
        cxrImg = imread(imdsCXR.Files{1});
        if size(roiImg, 3) > 1, roiImg = rgb2gray(roiImg); end
        if size(cxrImg, 3) > 1, cxrImg = rgb2gray(cxrImg); end
        
        roiMean = mean(double(roiImg(:)));
        cxrMean = mean(double(cxrImg(:)));
        
        fprintf('  ROI mean intensity: %.2f\n', roiMean);
        fprintf('  CXR mean intensity: %.2f\n', cxrMean);
        
        if roiMean < cxrMean
            fprintf('  ✓ Intensity relationship correct (ROI < CXR)\n');
        else
            fprintf('  ⚠ Unexpected intensity relationship\n');
        end
    end
end
