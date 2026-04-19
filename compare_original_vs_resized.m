%COMPARE_ORIGINAL_VS_RESIZED Compare intensities between original and resized images
%   This script compares a sample image from original dataset with resized version
%   to check if intensities are preserved during resizing.

clc; close all;
fprintf('=== COMPARING ORIGINAL VS RESIZED INTENSITIES ===\n\n');

%% Configuration
originalCXRDir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/cxr_png';
resizedCXRDir = fullfile('input', 'resized', 'cxr');

%% Get a sample image
sampleImage = 'CHNCXR_0001_0.png';
originalPath = fullfile(originalCXRDir, sampleImage);
resizedPath = fullfile(resizedCXRDir, 'normal', sampleImage);

if ~exist(originalPath, 'file')
    % Try to find any image
    files = dir(fullfile(originalCXRDir, '*.png'));
    if ~isempty(files)
        sampleImage = files(1).name;
        originalPath = fullfile(originalCXRDir, sampleImage);
        [~, nameBase, ~] = fileparts(sampleImage);
        % Try to find in resized
        resizedPath = fullfile(resizedCXRDir, 'normal', sampleImage);
        if ~exist(resizedPath, 'file')
            resizedPath = fullfile(resizedCXRDir, 'ptb', sampleImage);
        end
    end
end

if ~exist(originalPath, 'file')
    error('Original image not found: %s', originalPath);
end
if ~exist(resizedPath, 'file')
    error('Resized image not found: %s', resizedPath);
end

fprintf('Comparing: %s\n', sampleImage);
fprintf('  Original: %s\n', originalPath);
fprintf('  Resized: %s\n\n', resizedPath);

%% Load images
originalImg = imread(originalPath);
resizedImg = imread(resizedPath);

if size(originalImg, 3) > 1
    originalImg = rgb2gray(originalImg);
end
if size(resizedImg, 3) > 1
    resizedImg = rgb2gray(resizedImg);
end

% Resize original to match resized for comparison
originalResized = imresize(originalImg, size(resizedImg), 'bicubic');

%% Compare intensities
fprintf('=== INTENSITY COMPARISON ===\n');
fprintf('Original image:\n');
fprintf('  Size: %d x %d\n', size(originalImg, 2), size(originalImg, 1));
fprintf('  Mean intensity: %.2f\n', mean(double(originalImg(:))));
fprintf('  Std intensity: %.2f\n', std(double(originalImg(:))));
fprintf('  Min/Max: %d / %d\n', min(originalImg(:)), max(originalImg(:)));

fprintf('\nOriginal (resized to %d x %d for comparison):\n', size(resizedImg, 2), size(resizedImg, 1));
fprintf('  Mean intensity: %.2f\n', mean(double(originalResized(:))));
fprintf('  Std intensity: %.2f\n', std(double(originalResized(:))));
fprintf('  Min/Max: %d / %d\n', min(originalResized(:)), max(originalResized(:)));

fprintf('\nResized image:\n');
fprintf('  Size: %d x %d\n', size(resizedImg, 2), size(resizedImg, 1));
fprintf('  Mean intensity: %.2f\n', mean(double(resizedImg(:))));
fprintf('  Std intensity: %.2f\n', std(double(resizedImg(:))));
fprintf('  Min/Max: %d / %d\n', min(resizedImg(:)), max(resizedImg(:)));

%% Calculate differences
meanDiff = abs(mean(double(originalResized(:))) - mean(double(resizedImg(:))));
stdDiff = abs(std(double(originalResized(:))) - std(double(resizedImg(:))));

fprintf('\n=== DIFFERENCES ===\n');
fprintf('Mean intensity difference: %.2f\n', meanDiff);
fprintf('Std intensity difference: %.2f\n', stdDiff);

if meanDiff > 5 || stdDiff > 5
    fprintf('  ⚠ Significant intensity change detected!\n');
    fprintf('  This indicates resizing is affecting intensities.\n');
else
    fprintf('  ✓ Intensities are well preserved.\n');
end

%% Visualize
figure('Position', [100, 100, 1400, 600]);

subplot(2, 3, 1);
imshow(originalImg);
title(sprintf('Original (%d x %d)', size(originalImg, 2), size(originalImg, 1)));

subplot(2, 3, 2);
imshow(originalResized);
title(sprintf('Original Resized (%d x %d)', size(originalResized, 2), size(originalResized, 1)));

subplot(2, 3, 3);
imshow(resizedImg);
title(sprintf('Resized Version (%d x %d)', size(resizedImg, 2), size(resizedImg, 1)));

subplot(2, 3, 4);
diffImg = abs(double(originalResized) - double(resizedImg));
imshow(diffImg, []);
title(sprintf('Absolute Difference (Max: %.1f)', max(diffImg(:))));
colorbar;

subplot(2, 3, 5);
histogram(double(originalResized(:)), 50, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Original Resized');
hold on;
histogram(double(resizedImg(:)), 50, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Resized');
xlabel('Intensity');
ylabel('Frequency');
title('Intensity Histograms');
legend('Location', 'best');
grid on;

subplot(2, 3, 6);
scatter(double(originalResized(:)), double(resizedImg(:)), 1, '.');
hold on;
plot([0, 255], [0, 255], 'r--', 'LineWidth', 2);
xlabel('Original Resized Intensity');
ylabel('Resized Intensity');
title('Intensity Correlation');
axis equal;
xlim([0, 255]);
ylim([0, 255]);
grid on;
correlation = corrcoef(double(originalResized(:)), double(resizedImg(:)));
fprintf('\nCorrelation: %.4f\n', correlation(1, 2));

sgtitle(sprintf('Intensity Comparison: %s', sampleImage), 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
outputDir = 'results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
saveas(gcf, fullfile(outputDir, 'original_vs_resized_comparison.png'));
fprintf('\nComparison figure saved to: %s\n', fullfile(outputDir, 'original_vs_resized_comparison.png'));

fprintf('\n=== ANALYSIS COMPLETE ===\n');

