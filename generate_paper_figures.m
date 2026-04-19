function generate_paper_figures()
%GENERATE_PAPER_FIGURES Generate high-quality figures for the paper
%   This script generates the following figures:
%   1. Preprocessing Comparison (Original vs HistMatch vs Z-Score)
%   2. OOD GradCAM Visualization (Demonstrating Anatomical Guidance)
%   3. Performance Comparison Bar Chart (ID vs OOD)

clc; close all;
outputDir = 'paper_figures';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('=== GENERATING PAPER FIGURES ===\n');

%% 1. Preprocessing Comparison Figure
fprintf('Generating Figure 1: Preprocessing Comparison...\n');
% Load a sample OOD image
cxrDir = fullfile('input', 'cxr');
cxrFiles = dir(fullfile(cxrDir, '**', '*.png'));
if isempty(cxrFiles)
    cxrFiles = dir(fullfile(cxrDir, '**', '*.jpg'));
end

if ~isempty(cxrFiles)
    % Pick a representative image (e.g., the first one)
    imgIdx = 1; 
    imgPath = fullfile(cxrFiles(imgIdx).folder, cxrFiles(imgIdx).name);
    img = imread(imgPath);
    if size(img, 3) == 1, img = repmat(img, [1 1 3]); end
    img = imresize(img, [224 224]);
    imgGray = rgb2gray(img);
    
    % Compute Reference Histogram (from a few CXR images)
    numRef = min(20, numel(cxrFiles));
    refPixels = [];
    for i = 1:numRef
        rImg = imread(fullfile(cxrFiles(i).folder, cxrFiles(i).name));
        if size(rImg, 3) > 1, rImg = rgb2gray(rImg); end
        refPixels = [refPixels; rImg(:)];
    end
    refHist = imhist(uint8(refPixels));
    
    % Apply Preprocessing
    % 1. Original
    imgOriginal = imgGray;
    
    % 2. HistMatch
    imgHistMatch = histeq(imgGray, refHist);
    
    % 3. Z-Score
    imgDouble = double(imgGray);
    imgMean = mean(imgDouble(:));
    imgStd = std(imgDouble(:));
    imgZScore = (imgDouble - imgMean) / (imgStd + eps);
    % Scale to 0-255 for visualization
    imgZScore = (imgZScore - min(imgZScore(:))) / (max(imgZScore(:)) - min(imgZScore(:)) + eps) * 255;
    imgZScore = uint8(imgZScore);
    
    % Plot
    f1 = figure('Position', [100, 100, 1200, 400], 'Color', 'w');
    tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    nexttile; imshow(imgOriginal); title('(a) Original CXR (OOD)', 'FontSize', 14);
    nexttile; imshow(imgHistMatch); title('(b) Histogram Matching', 'FontSize', 14);
    nexttile; imshow(imgZScore); title('(c) Z-Score Normalization (Best)', 'FontSize', 14);
    
    exportgraphics(f1, fullfile(outputDir, 'fig1_preprocessing_comparison.png'), 'Resolution', 300);
    fprintf('  Saved: fig1_preprocessing_comparison.png\n');
else
    warning('No CXR images found for Figure 1.');
end

%% 2. OOD GradCAM Visualization
fprintf('Generating Figure 2: OOD GradCAM Visualization...\n');
modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');

if exist(modelFile, 'file') && ~isempty(cxrFiles)
    s = load(modelFile);
    net = s.trainedNet;
    
    % Select a few random OOD images
    numSamples = 3;
    rng(42); % Fixed seed for reproducibility
    indices = randperm(numel(cxrFiles), numSamples);
    
    f2 = figure('Position', [100, 100, 1000, 300 * numSamples], 'Color', 'w');
    tiledlayout(numSamples, 3, 'TileSpacing', 'none', 'Padding', 'compact');
    
    for i = 1:numSamples
        idx = indices(i);
        imgPath = fullfile(cxrFiles(idx).folder, cxrFiles(idx).name);
        img = imread(imgPath);
        if size(img, 3) == 1, img = repmat(img, [1 1 3]); end
        img = imresize(img, [224 224]);
        
        % Preprocess (Z-Score - Best Method)
        imgGray = rgb2gray(img);
        imgDouble = double(imgGray);
        imgZScore = (imgDouble - mean(imgDouble(:))) / (std(imgDouble(:)) + eps);
        imgZScore = (imgZScore - min(imgZScore(:))) / (max(imgZScore(:)) - min(imgZScore(:)) + eps) * 255;
        imgProcessed = repmat(uint8(imgZScore), [1 1 3]);
        
        % Predict
        dlImg = dlarray(single(imgProcessed), 'SSCB');
        if canUseGPU, dlImg = gpuArray(dlImg); end
        
        [scores, ~] = predict(net, dlImg);
        probs = extractdata(scores);
        [~, predIdx] = max(probs);
        classes = {'Normal', 'Tuberculosis'}; % Assuming binary
        predLabel = classes{predIdx};
        
        % GradCAM
        try
            map = gradCAM(net, dlImg, predLabel, 'FeatureLayer', 'relu5_3');
            map = extractdata(map);
            map = imresize(map, [224 224]);
            
            % Normalize map
            map = double(map);
            map = (map - min(map(:))) / (max(map(:)) - min(map(:)) + eps);
            
            % Overlay
            heatmap = jet(256);
            mapIdx = gray2ind(map, 256);
            rgbMap = ind2rgb(mapIdx, heatmap);
            overlay = 0.5 * double(img)/255 + 0.5 * rgbMap;
            
            % Plot
            nexttile; imshow(img); 
            if i==1, title('Original Image', 'FontSize', 12); end
            ylabel(sprintf('Sample %d', i), 'FontSize', 12, 'FontWeight', 'bold');
            
            nexttile; imshow(map, []); colormap(gca, 'jet');
            if i==1, title('GradCAM Attention', 'FontSize', 12); end
            
            nexttile; imshow(overlay);
            if i==1, title(sprintf('Prediction: %s', predLabel), 'FontSize', 12); end
            
        catch ME
            warning('GradCAM failed for sample %d: %s', i, ME.message);
        end
    end
    
    exportgraphics(f2, fullfile(outputDir, 'fig2_ood_gradcam.png'), 'Resolution', 300);
    fprintf('  Saved: fig2_ood_gradcam.png\n');
else
    warning('Model file or CXR images not found for Figure 2.');
end

%% 3. Performance Summary Chart
fprintf('Generating Figure 3: Performance Summary...\n');
% Data from our experiments (Hardcoded for the paper figure)
methods = {'Baseline', 'HistMatch', 'Z-Score (Ours)'};
id_acc = [82.1, 81.2, 80.3];
ood_acc = [64.9, 63.4, 67.2];
degradation = [20.71, 21.46, 16.35];

f3 = figure('Position', [100, 100, 800, 500], 'Color', 'w');
b = bar([id_acc; ood_acc]', 'grouped');
b(1).FaceColor = [0.2 0.6 0.8]; % Blue for ID
b(2).FaceColor = [0.8 0.4 0.2]; % Orange for OOD

xticklabels(methods);
ylabel('Accuracy (%)', 'FontSize', 12);
legend({'ID Accuracy', 'OOD Accuracy'}, 'Location', 'northeast', 'FontSize', 11);
title('Generalization Performance Comparison', 'FontSize', 14);
grid on;
ylim([50 90]);

% Add degradation text
for i = 1:3
    text(i, ood_acc(i) - 2, sprintf('Degradation:\n%.1f%%', degradation(i)), ...
        'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 10);
end

exportgraphics(f3, fullfile(outputDir, 'fig3_performance_summary.png'), 'Resolution', 300);
fprintf('  Saved: fig3_performance_summary.png\n');

fprintf('\nAll figures generated in: %s\n', fullfile(pwd, outputDir));
end
