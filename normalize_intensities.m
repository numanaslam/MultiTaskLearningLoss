%NORMALIZE_INTENSITIES Normalize image intensities to reduce variation
%   This script normalizes intensities across images to ensure consistency.
%   Strategy: Model trained on ROI, so ROI images are NOT modified.
%             Only CXR (OOD) images are normalized.
%   
%   Options:
%   1. Histogram matching (match to reference histogram from ROI)
%   2. Z-score normalization (per image)
%   3. Min-max normalization (per image)
%   4. Percentile-based normalization

clc; close all;
fprintf('=== NORMALIZING CXR (OOD) IMAGE INTENSITIES ===\n');
fprintf('Strategy: Model trained on ROI, so ROI images will NOT be modified.\n');
fprintf('          Only CXR (OOD) images will be normalized.\n\n');

%% Configuration
roiDir = fullfile('input',  'roi');
cxrDir = fullfile('input', 'cxr');
classes = {'normal', 'ptb'};
method = 'histogram_matching';  % Options: 'histogram_matching', 'zscore', 'minmax', 'percentile'
normalizeROI = false;  % Set to true to normalize ROI as well (not recommended)

% Create backup
backupDir = fullfile('input', 'backup_before_normalization');
if ~exist(backupDir, 'dir')
    mkdir(backupDir);
    fprintf('Created backup directory: %s\n', backupDir);
end

%% Compute reference histogram from ROI images (for histogram matching)
if strcmp(method, 'histogram_matching')
    fprintf('Computing reference histogram from ROI images...\n');
    refHist = zeros(256, 1);
    refCount = 0;
    
    for c = 1:numel(classes)
        className = classes{c};
        roiClassDir = fullfile(roiDir, className);
        
        if ~exist(roiClassDir, 'dir')
            continue;
        end
        
        files = dir(fullfile(roiClassDir, '*.png'));
        % Use first 50 images to compute reference
        for i = 1:min(50, numel(files))
            if files(i).isdir, continue; end
            try
                img = imread(fullfile(roiClassDir, files(i).name));
                if size(img, 3) > 1
                    img = rgb2gray(img);
                end
                hist = imhist(img);
                refHist = refHist + hist;
                refCount = refCount + 1;
            catch
                continue;
            end
        end
    end
    
    refHist = refHist / refCount;
    fprintf('  Reference histogram computed from %d ROI images\n', refCount);
end

%% Normalize ROI images (optional - not recommended)
stats = struct();
stats.roi_processed = 0;
stats.cxr_processed = 0;
stats.errors = 0;

if normalizeROI
    fprintf('\nNormalizing ROI images (WARNING: Model trained on original ROI intensities)...\n');
    
    for c = 1:numel(classes)
        className = classes{c};
        roiClassDir = fullfile(roiDir, className);
        
        if ~exist(roiClassDir, 'dir')
            continue;
        end
        
        files = dir(fullfile(roiClassDir, '*.png'));
        fprintf('  Processing %s class: %d ROI images\n', className, numel(files));
        
        for i = 1:numel(files)
            if files(i).isdir, continue; end
            
            try
                imgPath = fullfile(roiClassDir, files(i).name);
                
                % Backup
                backupPath = fullfile(backupDir, 'roi', className, files(i).name);
                if ~exist(fullfile(backupDir, 'roi', className), 'dir')
                    mkdir(fullfile(backupDir, 'roi', className));
                end
                if ~exist(backupPath, 'file')
                    copyfile(imgPath, backupPath);
                end
                
                % Load image
                img = imread(imgPath);
                if size(img, 3) > 1
                    img = rgb2gray(img);
                end
                
                % Apply normalization
                switch method
                    case 'histogram_matching'
                        imgNorm = histeq(img, refHist);
                    case 'zscore'
                        imgDouble = double(img);
                        imgMean = mean(imgDouble(:));
                        imgStd = std(imgDouble(:));
                        if imgStd > 0
                            imgNorm = (imgDouble - imgMean) / imgStd;
                            imgNorm = (imgNorm - min(imgNorm(:))) / (max(imgNorm(:)) - min(imgNorm(:)) + eps) * 255;
                        else
                            imgNorm = imgDouble;
                        end
                        imgNorm = uint8(imgNorm);
                    case 'minmax'
                        imgDouble = double(img);
                        imgMin = min(imgDouble(:));
                        imgMax = max(imgDouble(:));
                        if imgMax > imgMin
                            imgNorm = (imgDouble - imgMin) / (imgMax - imgMin) * 255;
                        else
                            imgNorm = imgDouble;
                        end
                        imgNorm = uint8(imgNorm);
                    case 'percentile'
                        imgDouble = double(img);
                        p1 = prctile(imgDouble(:), 1);
                        p99 = prctile(imgDouble(:), 99);
                        if p99 > p1
                            imgNorm = (imgDouble - p1) / (p99 - p1) * 255;
                            imgNorm = max(0, min(255, imgNorm));
                        else
                            imgNorm = imgDouble;
                        end
                        imgNorm = uint8(imgNorm);
                    otherwise
                        imgNorm = img;
                end
                
                % Save normalized image
                imwrite(imgNorm, imgPath);
                stats.roi_processed = stats.roi_processed + 1;
                
            catch ME
                warning('Error processing ROI image %s: %s', files(i).name, ME.message);
                stats.errors = stats.errors + 1;
            end
        end
    end
else
    fprintf('\nSkipping ROI normalization (model training data - keeping as-is)\n');
end

%% Normalize CXR images
fprintf('\nNormalizing CXR images...\n');
for c = 1:numel(classes)
    className = classes{c};
    cxrClassDir = fullfile(cxrDir, className);
    
    if exist(cxrClassDir, 'dir')
        cxrFiles = dir(fullfile(cxrClassDir, '*.png'));
        fprintf('  Processing %s class: %d CXR images\n', className, numel(cxrFiles));
        
        for i = 1:numel(cxrFiles)
            if cxrFiles(i).isdir, continue; end
            
            try
                imgPath = fullfile(cxrClassDir, cxrFiles(i).name);
                
                % Backup
                backupPath = fullfile(backupDir, 'cxr', className, cxrFiles(i).name);
                if ~exist(fullfile(backupDir, 'cxr', className), 'dir')
                    mkdir(fullfile(backupDir, 'cxr', className));
                end
                if ~exist(backupPath, 'file')
                    copyfile(imgPath, backupPath);
                end
                
                % Load image
                img = imread(imgPath);
                if size(img, 3) > 1
                    img = rgb2gray(img);
                end
                
                % Apply normalization (same method as ROI)
                switch method
                    case 'histogram_matching'
                        imgNorm = histeq(img, refHist);
                    case 'zscore'
                        imgDouble = double(img);
                        imgMean = mean(imgDouble(:));
                        imgStd = std(imgDouble(:));
                        if imgStd > 0
                            imgNorm = (imgDouble - imgMean) / imgStd;
                            imgNorm = (imgNorm - min(imgNorm(:))) / (max(imgNorm(:)) - min(imgNorm(:)) + eps) * 255;
                        else
                            imgNorm = imgDouble;
                        end
                        imgNorm = uint8(imgNorm);
                    case 'minmax'
                        imgDouble = double(img);
                        imgMin = min(imgDouble(:));
                        imgMax = max(imgDouble(:));
                        if imgMax > imgMin
                            imgNorm = (imgDouble - imgMin) / (imgMax - imgMin) * 255;
                        else
                            imgNorm = imgDouble;
                        end
                        imgNorm = uint8(imgNorm);
                    case 'percentile'
                        imgDouble = double(img);
                        p1 = prctile(imgDouble(:), 1);
                        p99 = prctile(imgDouble(:), 99);
                        if p99 > p1
                            imgNorm = (imgDouble - p1) / (p99 - p1) * 255;
                            imgNorm = max(0, min(255, imgNorm));
                        else
                            imgNorm = imgDouble;
                        end
                        imgNorm = uint8(imgNorm);
                    otherwise
                        imgNorm = img;
                end
                
                % Save normalized image
                imwrite(imgNorm, imgPath);
                stats.cxr_processed = stats.cxr_processed + 1;
                
            catch ME
                warning('Error processing CXR image %s: %s', cxrFiles(i).name, ME.message);
                stats.errors = stats.errors + 1;
            end
        end
    end
end

%% Summary
fprintf('\n=== NORMALIZATION SUMMARY ===\n');
fprintf('Method: %s\n', method);
if normalizeROI
    fprintf('ROI images processed: %d\n', stats.roi_processed);
else
    fprintf('ROI images: Skipped (model training data - kept as-is)\n');
end
fprintf('CXR images processed: %d\n', stats.cxr_processed);
fprintf('Errors: %d\n', stats.errors);
fprintf('\nBackup saved to: %s\n', backupDir);
fprintf('\nStrategy: Only CXR (OOD) images were normalized to reduce intensity variation.\n');
fprintf('         ROI images were kept unchanged to maintain model compatibility.\n');
fprintf('\n=== NORMALIZATION COMPLETE ===\n');

