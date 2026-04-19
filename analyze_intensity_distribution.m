%ANALYZE_INTENSITY_DISTRIBUTION Analyze intensity distribution across all images
%   This script analyzes intensity distributions to identify:
%   1. Images with unusually bright or dark intensities
%   2. Potential inverted images
%   3. Intensity inconsistencies that may affect model performance

clc; close all;
fprintf('=== ANALYZING INTENSITY DISTRIBUTION ===\n\n');

%% Configuration
roiDir = fullfile('input', 'resized', 'roi');
cxrDir = fullfile('input', 'resized', 'cxr');
classes = {'normal', 'ptb'};

%% Analyze ROI images
fprintf('Analyzing ROI images...\n');
allROIIntensities = [];
allROIFiles = {};
allROIClasses = {};

for c = 1:numel(classes)
    className = classes{c};
    roiClassDir = fullfile(roiDir, className);
    
    if ~exist(roiClassDir, 'dir')
        continue;
    end
    
    files = dir(fullfile(roiClassDir, '*.png'));
    for i = 1:numel(files)
        if files(i).isdir, continue; end
        
        imgPath = fullfile(roiClassDir, files(i).name);
        try
            img = imread(imgPath);
            if size(img, 3) > 1
                img = rgb2gray(img);
            end
            
            meanInt = mean(double(img(:)));
            allROIIntensities = [allROIIntensities; meanInt];
            allROIFiles{end+1} = files(i).name;
            allROIClasses{end+1} = className;
        catch
            continue;
        end
    end
end

fprintf('  Analyzed %d ROI images\n', numel(allROIIntensities));

%% Analyze CXR images
fprintf('\nAnalyzing CXR images...\n');
allCXRIntensities = [];
allCXRFiles = {};
allCXRClasses = {};

for c = 1:numel(classes)
    className = classes{c};
    cxrClassDir = fullfile(cxrDir, className);
    
    if ~exist(cxrClassDir, 'dir')
        continue;
    end
    
    files = dir(fullfile(cxrClassDir, '*.png'));
    for i = 1:numel(files)
        if files(i).isdir, continue; end
        
        imgPath = fullfile(cxrClassDir, files(i).name);
        try
            img = imread(imgPath);
            if size(img, 3) > 1
                img = rgb2gray(img);
            end
            
            meanInt = mean(double(img(:)));
            allCXRIntensities = [allCXRIntensities; meanInt];
            allCXRFiles{end+1} = files(i).name;
            allCXRClasses{end+1} = className;
        catch
            continue;
        end
    end
end

fprintf('  Analyzed %d CXR images\n', numel(allCXRIntensities));

%% Statistics
fprintf('\n=== INTENSITY STATISTICS ===\n');
fprintf('\nROI Images:\n');
fprintf('  Mean: %.2f\n', mean(allROIIntensities));
fprintf('  Std: %.2f\n', std(allROIIntensities));
fprintf('  Min: %.2f (%s)\n', min(allROIIntensities), allROIFiles{allROIIntensities == min(allROIIntensities)});
fprintf('  Max: %.2f (%s)\n', max(allROIIntensities), allROIFiles{allROIIntensities == max(allROIIntensities)});
fprintf('  Median: %.2f\n', median(allROIIntensities));

fprintf('\nCXR Images:\n');
fprintf('  Mean: %.2f\n', mean(allCXRIntensities));
fprintf('  Std: %.2f\n', std(allCXRIntensities));
fprintf('  Min: %.2f (%s)\n', min(allCXRIntensities), allCXRFiles{allCXRIntensities == min(allCXRIntensities)});
fprintf('  Max: %.2f (%s)\n', max(allCXRIntensities), allCXRFiles{allCXRIntensities == max(allCXRIntensities)});
fprintf('  Median: %.2f\n', median(allCXRIntensities));

%% Identify outliers
roiMean = mean(allROIIntensities);
roiStd = std(allROIIntensities);
roiOutliers = abs(allROIIntensities - roiMean) > 2 * roiStd;

cxrMean = mean(allCXRIntensities);
cxrStd = std(allCXRIntensities);
cxrOutliers = abs(allCXRIntensities - cxrMean) > 2 * cxrStd;

fprintf('\n=== OUTLIERS (2 std deviations) ===\n');
fprintf('\nROI Outliers:\n');
outlierIndices = find(roiOutliers);
for i = 1:min(10, numel(outlierIndices))
    idx = outlierIndices(i);
    fprintf('  %s (%s): %.2f (expected: %.2f ± %.2f)\n', ...
        allROIFiles{idx}, allROIClasses{idx}, allROIIntensities(idx), roiMean, roiStd);
end

fprintf('\nCXR Outliers:\n');
outlierIndices = find(cxrOutliers);
for i = 1:min(10, numel(outlierIndices))
    idx = outlierIndices(i);
    fprintf('  %s (%s): %.2f (expected: %.2f ± %.2f)\n', ...
        allCXRFiles{idx}, allCXRClasses{idx}, allCXRIntensities(idx), cxrMean, cxrStd);
end

%% Check specific image
checkImage = 'CHNCXR_0088_0.png';
fprintf('\n=== CHECKING SPECIFIC IMAGE: %s ===\n', checkImage);

roiIdx = find(contains(allROIFiles, checkImage, 'IgnoreCase', true));
if ~isempty(roiIdx)
    fprintf('ROI Image:\n');
    fprintf('  Intensity: %.2f\n', allROIIntensities(roiIdx(1)));
    fprintf('  Z-score: %.2f (%.2f std from mean)\n', ...
        (allROIIntensities(roiIdx(1)) - roiMean) / roiStd, ...
        (allROIIntensities(roiIdx(1)) - roiMean) / roiStd);
    
    if allROIIntensities(roiIdx(1)) > roiMean + 2*roiStd
        fprintf('  ⚠ This image is BRIGHTER than expected for ROI\n');
        fprintf('  Possible causes:\n');
        fprintf('    1. Image may be inverted\n');
        fprintf('    2. Original image had different intensity distribution\n');
        fprintf('    3. Cropping may have captured bright background\n');
    end
end

cxrIdx = find(contains(allCXRFiles, checkImage, 'IgnoreCase', true));
if ~isempty(cxrIdx)
    fprintf('\nCXR Image:\n');
    fprintf('  Intensity: %.2f\n', allCXRIntensities(cxrIdx(1)));
    fprintf('  Z-score: %.2f (%.2f std from mean)\n', ...
        (allCXRIntensities(cxrIdx(1)) - cxrMean) / cxrStd, ...
        (allCXRIntensities(cxrIdx(1)) - cxrMean) / cxrStd);
end

%% Create visualization
fprintf('\n=== CREATING VISUALIZATION ===\n');
figure('Position', [100, 100, 1200, 800]);

% Histogram of ROI intensities
subplot(2, 2, 1);
histogram(allROIIntensities, 50, 'FaceColor', [0.2 0.6 0.8]);
hold on;
xline(roiMean, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.1f', roiMean));
xline(roiMean + 2*roiStd, 'g--', 'LineWidth', 1, 'DisplayName', '+2σ');
xline(roiMean - 2*roiStd, 'g--', 'LineWidth', 1, 'DisplayName', '-2σ');
xlabel('Mean Intensity');
ylabel('Frequency');
title('ROI Intensity Distribution');
legend('Location', 'best');
grid on;

% Histogram of CXR intensities
subplot(2, 2, 2);
histogram(allCXRIntensities, 50, 'FaceColor', [0.8 0.4 0.2]);
hold on;
xline(cxrMean, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.1f', cxrMean));
xline(cxrMean + 2*cxrStd, 'g--', 'LineWidth', 1, 'DisplayName', '+2σ');
xline(cxrMean - 2*cxrStd, 'g--', 'LineWidth', 1, 'DisplayName', '-2σ');
xlabel('Mean Intensity');
ylabel('Frequency');
title('CXR Intensity Distribution');
legend('Location', 'best');
grid on;

% Box plot comparison
subplot(2, 2, 3);
boxplot([allROIIntensities; allCXRIntensities], ...
    [ones(size(allROIIntensities)); 2*ones(size(allCXRIntensities))], ...
    'Labels', {'ROI', 'CXR'});
ylabel('Mean Intensity');
title('Intensity Comparison: ROI vs CXR');
grid on;

% Outliers plot
subplot(2, 2, 4);
% Use plot instead of scatter for better compatibility
plot(1:numel(allROIIntensities), allROIIntensities, 'b.', 'MarkerSize', 8);
hold on;
if any(roiOutliers)
    plot(find(roiOutliers), allROIIntensities(roiOutliers), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
yline(roiMean, 'r--', 'LineWidth', 2);
yline(roiMean + 2*roiStd, 'g--', 'LineWidth', 1);
yline(roiMean - 2*roiStd, 'g--', 'LineWidth', 1);
xlabel('Image Index');
ylabel('Mean Intensity');
title('ROI Intensity Distribution');
if any(roiOutliers)
    legend('All Images', 'Outliers (>2σ)', 'Mean', '±2σ', 'Location', 'best');
else
    legend('All Images', 'Mean', '±2σ', 'Location', 'best');
end
grid on;

sgtitle('Intensity Distribution Analysis', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
outputDir = 'results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
saveas(gcf, fullfile(outputDir, 'intensity_distribution_analysis.png'));
fprintf('  Saved: %s\n', fullfile(outputDir, 'intensity_distribution_analysis.png'));

fprintf('\n=== ANALYSIS COMPLETE ===\n');

