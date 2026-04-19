% Compare two images to see if they are the same when resized
clc; close all;

% Image paths
img1_path = fullfile('input', 'resized', 'cxr', 'normal', 'CHNCXR_0001_0.png');
img2_path = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/data/Lung Segmentation/CXR_png/CHNCXR_0001_0.png';

fprintf('=== IMAGE COMPARISON ===\n\n');

% Load images
fprintf('Loading images...\n');
if exist(img1_path, 'file')
    img1 = imread(img1_path);
    fprintf('  Image 1: %s\n', img1_path);
    fprintf('    Size: %d x %d, Type: %s\n', size(img1, 1), size(img1, 2), class(img1));
else
    error('Image 1 not found: %s', img1_path);
end

if exist(img2_path, 'file')
    img2 = imread(img2_path);
    fprintf('  Image 2: %s\n', img2_path);
    fprintf('    Size: %d x %d, Type: %s\n', size(img2, 1), size(img2, 2), class(img2));
else
    error('Image 2 not found: %s', img2_path);
end

% Convert to grayscale if needed
if size(img1, 3) > 1
    img1_gray = rgb2gray(img1);
else
    img1_gray = img1;
end

if size(img2, 3) > 1
    img2_gray = rgb2gray(img2);
else
    img2_gray = img2;
end

% Resize image 2 to match image 1
fprintf('\nResizing Image 2 to match Image 1 size (%d x %d)...\n', size(img1_gray, 1), size(img1_gray, 2));
img2_resized = imresize(img2_gray, [size(img1_gray, 1), size(img1_gray, 2)]);

% Compare images
fprintf('\n=== COMPARISON RESULTS ===\n');

% 1. Size comparison
fprintf('Size Comparison:\n');
fprintf('  Image 1 (resized): %d x %d\n', size(img1_gray, 1), size(img1_gray, 2));
fprintf('  Image 2 (original): %d x %d\n', size(img2_gray, 1), size(img2_gray, 2));
fprintf('  Image 2 (resized): %d x %d\n', size(img2_resized, 1), size(img2_resized, 2));

% 2. Pixel-wise difference
diff = double(img1_gray) - double(img2_resized);
mse = mean(diff(:).^2);
rmse = sqrt(mse);
max_diff = max(abs(diff(:)));
mean_diff = mean(abs(diff(:)));

fprintf('\nPixel-wise Comparison:\n');
fprintf('  Mean Squared Error (MSE): %.4f\n', mse);
fprintf('  Root Mean Squared Error (RMSE): %.4f\n', rmse);
fprintf('  Maximum Absolute Difference: %.2f\n', max_diff);
fprintf('  Mean Absolute Difference: %.4f\n', mean_diff);

% 3. Structural Similarity Index (SSIM)
if exist('ssim', 'file')
    ssim_val = ssim(img1_gray, img2_resized);
    fprintf('  SSIM: %.4f (1.0 = identical)\n', ssim_val);
else
    fprintf('  SSIM: Not available (Image Processing Toolbox required)\n');
end

% 4. Histogram comparison
hist1 = imhist(img1_gray);
hist2 = imhist(img2_resized);
hist_diff = sum(abs(hist1 - hist2)) / sum(hist1);
fprintf('  Histogram Difference: %.4f%%\n', hist_diff * 100);

% 5. Visual comparison
fprintf('\n=== VISUAL COMPARISON ===\n');
figure('Position', [100, 100, 1400, 600]);

subplot(2, 3, 1);
imshow(img1_gray);
title(sprintf('Image 1 (Resized)\n%d x %d', size(img1_gray, 1), size(img1_gray, 2)));

subplot(2, 3, 2);
imshow(img2_gray);
title(sprintf('Image 2 (Original)\n%d x %d', size(img2_gray, 1), size(img2_gray, 2)));

subplot(2, 3, 3);
imshow(img2_resized);
title(sprintf('Image 2 (Resized to match Image 1)\n%d x %d', size(img2_resized, 1), size(img2_resized, 2)));

subplot(2, 3, 4);
imshow(abs(diff), []);
title(sprintf('Absolute Difference\nMax: %.2f, Mean: %.4f', max_diff, mean_diff));
colorbar;

subplot(2, 3, 5);
plot(hist1, 'b-', 'LineWidth', 1.5); hold on;
plot(hist2, 'r--', 'LineWidth', 1.5);
xlabel('Intensity');
ylabel('Frequency');
title('Histogram Comparison');
legend('Image 1', 'Image 2', 'Location', 'best');
grid on;

subplot(2, 3, 6);
% Show side-by-side comparison
comparison = [img1_gray, img2_resized];
imshow(comparison);
title('Side-by-Side: Image 1 | Image 2 (Resized)');

sgtitle(sprintf('Image Comparison: MSE=%.4f, RMSE=%.4f, MaxDiff=%.2f', mse, rmse, max_diff), ...
    'FontSize', 14, 'FontWeight', 'bold');

% Save comparison figure
outputDir = 'results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
saveas(gcf, fullfile(outputDir, 'image_comparison.png'));
fprintf('\nComparison figure saved to: %s\n', fullfile(outputDir, 'image_comparison.png'));

% Conclusion
fprintf('\n=== CONCLUSION ===\n');
if mse < 1.0 && max_diff < 5
    fprintf('✓ Images are VERY SIMILAR (likely the same image)\n');
    fprintf('  MSE < 1.0 and Max Difference < 5 pixels\n');
elseif mse < 10.0 && max_diff < 20
    fprintf('⚠ Images are SIMILAR but may have minor differences\n');
    fprintf('  MSE < 10.0 and Max Difference < 20 pixels\n');
else
    fprintf('✗ Images are DIFFERENT\n');
    fprintf('  Significant pixel differences detected\n');
end

fprintf('\nNote: Image 1 appears to be already resized (24KB file).\n');
fprintf('      Image 2 is the original high-resolution image (5.9MB file).\n');

