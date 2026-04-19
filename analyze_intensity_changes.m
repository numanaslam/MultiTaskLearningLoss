%ANALYZE_INTENSITY_CHANGES Analyze why some images got darker after resizing
%   This script compares original vs resized images to understand intensity changes

clc; close all;
fprintf('=== ANALYZING INTENSITY CHANGES ===\n\n');

%% Configuration
originalCXRDir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/cxr_png';
resizedNewDir = fullfile('input', 'resized_new', 'cxr');
resizedOldDir = fullfile('input', 'resized', 'cxr');

% Images to analyze
testImages = {
    'CHNCXR_0020_0.png', 'normal';
    'CHNCXR_0030_0.png', 'normal';
    'CHNCXR_0088_0.png', 'normal';
    'CHNCXR_0001_0.png', 'normal';  % Compare with a darker one
    'CHNCXR_0005_0.png', 'normal';  % Compare with another darker one
};

fprintf('Analyzing %d images...\n\n', size(testImages, 1));

for i = 1:size(testImages, 1)
    imgName = testImages{i, 1};
    className = testImages{i, 2};
    
    fprintf('--- %s ---\n', imgName);
    
    % Load original
    originalPath = fullfile(originalCXRDir, imgName);
    if ~exist(originalPath, 'file')
        fprintf('  Original not found: %s\n', originalPath);
        continue;
    end
    
    % Load resized new
    resizedNewPath = fullfile(resizedNewDir, className, imgName);
    if ~exist(resizedNewPath, 'file')
        fprintf('  Resized new not found: %s\n', resizedNewPath);
        continue;
    end
    
    % Load resized old (if exists)
    resizedOldPath = fullfile(resizedOldDir, className, imgName);
    hasOld = exist(resizedOldPath, 'file');
    
    % Read images
    imgOriginal = imread(originalPath);
    if size(imgOriginal, 3) > 1
        imgOriginal = rgb2gray(imgOriginal);
    end
    if ~isa(imgOriginal, 'uint8')
        imgOriginal = uint8(imgOriginal);
    end
    
    imgResizedNew = imread(resizedNewPath);
    if size(imgResizedNew, 3) > 1
        imgResizedNew = rgb2gray(imgResizedNew);
    end
    
    if hasOld
        imgResizedOld = imread(resizedOldPath);
        if size(imgResizedOld, 3) > 1
            imgResizedOld = rgb2gray(imgResizedOld);
        end
    end
    
    % Resize original to match resized for comparison
    imgOriginalResized = imresize(imgOriginal, size(imgResizedNew), 'bicubic');
    
    % Calculate statistics
    origMean = mean(double(imgOriginal(:)));
    origStd = std(double(imgOriginal(:)));
    origMin = double(min(imgOriginal(:)));
    origMax = double(max(imgOriginal(:)));
    
    origResizedMean = mean(double(imgOriginalResized(:)));
    origResizedStd = std(double(imgOriginalResized(:)));
    
    newMean = mean(double(imgResizedNew(:)));
    newStd = std(double(imgResizedNew(:)));
    newMin = double(min(imgResizedNew(:)));
    newMax = double(max(imgResizedNew(:)));
    
    if hasOld
        oldMean = mean(double(imgResizedOld(:)));
        oldStd = std(double(imgResizedOld(:)));
    end
    
    % Print statistics
    fprintf('  Original (full size):\n');
    fprintf('    Mean: %.2f, Std: %.2f, Range: [%.0f, %.0f]\n', origMean, origStd, origMin, origMax);
    
    fprintf('  Original (resized to %dx%d):\n', size(imgResizedNew, 2), size(imgResizedNew, 1));
    fprintf('    Mean: %.2f, Std: %.2f\n', origResizedMean, origResizedStd);
    
    fprintf('  Resized New:\n');
    fprintf('    Mean: %.2f, Std: %.2f, Range: [%.0f, %.0f]\n', newMean, newStd, newMin, newMax);
    fprintf('    Mean change: %.2f (%.1f%%)\n', newMean - origResizedMean, ...
        (newMean - origResizedMean) / origResizedMean * 100);
    
    if hasOld
        fprintf('  Resized Old:\n');
        fprintf('    Mean: %.2f, Std: %.2f\n', oldMean, oldStd);
        fprintf('    Difference (new - old): %.2f\n', newMean - oldMean);
    end
    
    % Check if image might be inverted
    invertedMean = mean(double(255 - imgOriginal(:)));
    if abs(invertedMean - origMean) > 50
        fprintf('  ⚠ Possible inversion: Inverted mean = %.2f vs Original mean = %.2f\n', ...
            invertedMean, origMean);
    end
    
    fprintf('\n');
end

fprintf('=== ANALYSIS COMPLETE ===\n');

