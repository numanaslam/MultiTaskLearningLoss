function analyze_normal_class_failure()
%ANALYZE_NORMAL_CLASS_FAILURE Deep dive into why normal class fails on OOD
%   This script analyzes:
%   1. Attention maps for misclassified normal images
%   2. Feature activations for normal vs PTB
%   3. Intensity statistics for misclassified normal images
%   4. Comparison of attention patterns between ID and OOD normal images

clc; close all;
fprintf('=== ANALYZING NORMAL CLASS FAILURE ON OOD ===\n\n');

%% Configuration
modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');
if ~exist(modelFile, 'file')
    error('Model file not found: %s', modelFile);
end

roiDir = fullfile('input', 'roi');
cxrDir = fullfile('input', 'cxr');

outputDir = fullfile('results', 'normal_class_analysis');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Load Model
fprintf('Loading model...\n');
s = load(modelFile);
trainedNet = s.trainedNet;
if isfield(s, 'config')
    config = s.config;
    useGPU = config.useGPU;
else
    useGPU = canUseGPU;
end

classes = {'normal', 'ptb'};
featureLayer = 'relu5_3';

%% Load Datasets
fprintf('Loading datasets...\n');
imdsROI = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Filter normal class only
normalROI = subset(imdsROI, imdsROI.Labels == 'normal');
normalCXR = subset(imdsCXR, imdsCXR.Labels == 'normal');

fprintf('  ROI normal: %d samples\n', numel(normalROI.Files));
fprintf('  CXR normal: %d samples\n', numel(normalCXR.Files));

%% Evaluate and Get Predictions
fprintf('\nEvaluating predictions...\n');
[~, predROI, probsROI] = evaluate_with_gradcam(trainedNet, normalROI, classes, useGPU, featureLayer);
[~, predCXR, probsCXR] = evaluate_with_gradcam(trainedNet, normalCXR, classes, useGPU, featureLayer);

% Find misclassified samples
misclassROI = find(predROI ~= 'normal');
misclassCXR = find(predCXR ~= 'normal');

fprintf('  ROI misclassified: %d/%d (%.1f%%)\n', numel(misclassROI), numel(predROI), ...
    numel(misclassROI)/numel(predROI)*100);
fprintf('  CXR misclassified: %d/%d (%.1f%%)\n', numel(misclassCXR), numel(predCXR), ...
    numel(misclassCXR)/numel(predCXR)*100);

%% Analyze Attention Maps
fprintf('\nAnalyzing attention maps...\n');
numSamples = min(10, numel(misclassCXR));
if numSamples > 0
    sampleIdx = randperm(numel(misclassCXR), numSamples);
    
    figure('Position', [100, 100, 1600, 800]);
    for i = 1:numSamples
        idx = misclassCXR(sampleIdx(i));
        imgPath = normalCXR.Files{idx};
        img = imread(imgPath);
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        img = imresize(img, [224, 224]);
        
        % Get GradCAM
        cam = compute_gradcam(trainedNet, img, classes, featureLayer, useGPU);
        
        % Get prediction confidence
        conf = probsCXR(idx, 2);  % PTB probability
        
        subplot(2, 5, i);
        imshow(img, []);
        hold on;
        camOverlay = imresize(cam, [224, 224]);
        h = imshow(camOverlay, []);
        set(h, 'AlphaData', 0.5);
        colormap(gca, 'jet');
        title(sprintf('Conf: %.2f (Pred: %s)', conf, char(predCXR(idx))));
    end
    sgtitle('Attention Maps: Misclassified Normal CXR Images', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(gcf, fullfile(outputDir, 'misclassified_normal_attention.png'));
    fprintf('  Saved: misclassified_normal_attention.png\n');
end

%% Compare ID vs OOD Attention Patterns
fprintf('\nComparing ID vs OOD attention patterns...\n');
numCompare = min(5, min(numel(normalROI.Files), numel(normalCXR.Files)));
compareIdx = randperm(min(numel(normalROI.Files), numel(normalCXR.Files)), numCompare);

figure('Position', [100, 100, 1800, 600]);
for i = 1:numCompare
    idx = compareIdx(i);
    
    % ROI (ID)
    imgROI = imread(normalROI.Files{idx});
    if size(imgROI, 3) > 1
        imgROI = rgb2gray(imgROI);
    end
    imgROI = imresize(imgROI, [224, 224]);
    camROI = compute_gradcam(trainedNet, imgROI, classes, featureLayer, useGPU);
    
    % CXR (OOD) - find matching image
    [~, nameROI, ~] = fileparts(normalROI.Files{idx});
    cxrMatch = find(contains(normalCXR.Files, nameROI), 1);
    if ~isempty(cxrMatch)
        imgCXR = imread(normalCXR.Files{cxrMatch});
        if size(imgCXR, 3) > 1
            imgCXR = rgb2gray(imgCXR);
        end
        imgCXR = imresize(imgCXR, [224, 224]);
        camCXR = compute_gradcam(trainedNet, imgCXR, classes, featureLayer, useGPU);
        
        % Plot
        subplot(2, numCompare, i);
        imshow(imgROI, []);
        hold on;
        h = imshow(imresize(camROI, [224, 224]), []);
        set(h, 'AlphaData', 0.5);
        colormap(gca, 'jet');
        title(sprintf('ID (ROI): %s', char(predROI(idx))));
        
        subplot(2, numCompare, numCompare + i);
        imshow(imgCXR, []);
        hold on;
        h = imshow(imresize(camCXR, [224, 224]), []);
        set(h, 'AlphaData', 0.5);
        colormap(gca, 'jet');
        title(sprintf('OOD (CXR): %s', char(predCXR(cxrMatch))));
    end
end
sgtitle('Attention Pattern Comparison: ID (ROI) vs OOD (CXR) for Normal Class', ...
    'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'attention_comparison_id_vs_ood.png'));
fprintf('  Saved: attention_comparison_id_vs_ood.png\n');

%% Analyze Intensity Statistics
fprintf('\nAnalyzing intensity statistics...\n');
intensityROI = analyze_intensity_distribution(normalROI);
intensityCXR = analyze_intensity_distribution(normalCXR);
intensityMisclass = analyze_intensity_distribution(subset(normalCXR, misclassCXR));

figure('Position', [100, 100, 1200, 400]);
subplot(1, 3, 1);
histogram(intensityROI, 50, 'Normalization', 'probability');
xlabel('Mean Intensity');
ylabel('Probability');
title('ID (ROI) Normal Images');
grid on;

subplot(1, 3, 2);
histogram(intensityCXR, 50, 'Normalization', 'probability');
xlabel('Mean Intensity');
ylabel('Probability');
title('OOD (CXR) Normal Images');
grid on;

subplot(1, 3, 3);
histogram(intensityMisclass, 50, 'Normalization', 'probability');
xlabel('Mean Intensity');
ylabel('Probability');
title('Misclassified OOD Normal Images');
grid on;

sgtitle('Intensity Distribution Analysis', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'intensity_distribution.png'));
fprintf('  Saved: intensity_distribution.png\n');

fprintf('\nIntensity Statistics:\n');
fprintf('  ID (ROI) mean: %.2f ± %.2f\n', mean(intensityROI), std(intensityROI));
fprintf('  OOD (CXR) mean: %.2f ± %.2f\n', mean(intensityCXR), std(intensityCXR));
fprintf('  Misclassified mean: %.2f ± %.2f\n', mean(intensityMisclass), std(intensityMisclass));

%% Analyze Black Region Attention
fprintf('\nAnalyzing attention on black regions...\n');
numAnalyze = min(20, numel(misclassCXR));
if numAnalyze > 0
    blackAttention = zeros(numAnalyze, 1);
    lungAttention = zeros(numAnalyze, 1);
    
    sampleIdx = randperm(numel(misclassCXR), numAnalyze);
    for i = 1:numAnalyze
        idx = misclassCXR(sampleIdx(i));
        imgPath = normalCXR.Files{idx};
        img = imread(imgPath);
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        img = imresize(img, [224, 224]);
        
        cam = compute_gradcam(trainedNet, img, classes, featureLayer, useGPU);
        cam = imresize(cam, [224, 224]);
        
        % Define black regions (intensity < 10)
        blackMask = img < 10;
        lungMask = img > 50;  % Approximate lung region
        
        blackAttention(i) = mean(cam(blackMask));
        lungAttention(i) = mean(cam(lungMask));
    end
    
    figure('Position', [100, 100, 800, 400]);
    subplot(1, 2, 1);
    scatter(blackAttention, lungAttention, 'filled');
    xlabel('Attention on Black Regions');
    ylabel('Attention on Lung Regions');
    title('Attention Distribution');
    grid on;
    
    subplot(1, 2, 2);
    bar([mean(blackAttention), mean(lungAttention)]);
    set(gca, 'XTickLabel', {'Black Regions', 'Lung Regions'});
    ylabel('Mean Attention');
    title('Average Attention by Region');
    grid on;
    
    saveas(gcf, fullfile(outputDir, 'black_region_attention.png'));
    fprintf('  Saved: black_region_attention.png\n');
    
    fprintf('\nAttention Analysis:\n');
    fprintf('  Mean attention on black regions: %.4f\n', mean(blackAttention));
    fprintf('  Mean attention on lung regions: %.4f\n', mean(lungAttention));
    fprintf('  Ratio (black/lung): %.2f\n', mean(blackAttention) / (mean(lungAttention) + eps));
end

%% Generate Summary Report
fprintf('\n=== GENERATING SUMMARY REPORT ===\n');
reportFile = fullfile(outputDir, 'normal_class_failure_report.txt');
fid = fopen(reportFile, 'w');

fprintf(fid, '=== NORMAL CLASS FAILURE ANALYSIS REPORT ===\n\n');
fprintf(fid, 'Model: %s\n', modelFile);
fprintf(fid, 'Analysis Date: %s\n\n', datestr(now));

fprintf(fid, '=== KEY METRICS ===\n');
fprintf(fid, 'ID (ROI) Error Rate: %.1f%% (%d/%d)\n', ...
    numel(misclassROI)/numel(predROI)*100, numel(misclassROI), numel(predROI));
fprintf(fid, 'OOD (CXR) Error Rate: %.1f%% (%d/%d)\n', ...
    numel(misclassCXR)/numel(predCXR)*100, numel(misclassCXR), numel(predCXR));
fprintf(fid, 'Error Increase: %.1f%%\n\n', ...
    (numel(misclassCXR)/numel(predCXR) - numel(misclassROI)/numel(predROI))*100);

fprintf(fid, '=== INTENSITY STATISTICS ===\n');
fprintf(fid, 'ID (ROI) mean: %.2f ± %.2f\n', mean(intensityROI), std(intensityROI));
fprintf(fid, 'OOD (CXR) mean: %.2f ± %.2f\n', mean(intensityCXR), std(intensityCXR));
fprintf(fid, 'Misclassified mean: %.2f ± %.2f\n\n', mean(intensityMisclass), std(intensityMisclass));

if exist('blackAttention', 'var')
    fprintf(fid, '=== ATTENTION ANALYSIS ===\n');
    fprintf(fid, 'Mean attention on black regions: %.4f\n', mean(blackAttention));
    fprintf(fid, 'Mean attention on lung regions: %.4f\n', mean(lungAttention));
    fprintf(fid, 'Ratio (black/lung): %.2f\n\n', mean(blackAttention) / (mean(lungAttention) + eps));
end

fprintf(fid, '=== RECOMMENDATIONS ===\n');
fprintf(fid, '1. Increase anatomical guidance weight (λ_anatomical = 2.0-5.0)\n');
fprintf(fid, '2. Mask out black regions during training/evaluation\n');
fprintf(fid, '3. Train on mixed dataset (ROI + Full CXR)\n');
fprintf(fid, '4. Apply stronger preprocessing (CLAHE + Z-score)\n');
fprintf(fid, '5. Use adversarial training for domain robustness\n');
fprintf(fid, '6. Fine-tune on OOD data if available\n');

fclose(fid);
fprintf('  Saved: normal_class_failure_report.txt\n');

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Results saved to: %s\n', outputDir);

end

%% Helper Functions

function [accuracy, predictions, probabilities] = evaluate_with_gradcam(net, imds, classes, useGPU, featureLayer)
    augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
    reset(augDS);
    
    predictions = categorical.empty(0,1);
    probabilities = [];
    Ytrue = imds.Labels(:);
    
    while hasdata(augDS)
        tbl = read(augDS);
        imgs = cat(4, tbl.input{:});
        if useGPU, imgs = gpuArray(imgs); end
        dlX = dlarray(single(imgs), 'SSCB');
        
        sc = predict(net, dlX);
        lab = onehotdecode(extractdata(sc), classes, 1);
        predictions = [predictions; lab(:)];
        
        probs_batch = extractdata(sc);
        if size(probs_batch, 1) == numel(classes)
            if numel(classes) == 2
                probabilities = [probabilities; probs_batch(2,:)'];
            else
                probabilities = [probabilities; max(probs_batch, [], 1)'];
            end
        else
            probabilities = [probabilities; probs_batch(:)];
        end
    end
    
    match = (categorical(Ytrue) == categorical(predictions));
    accuracy = sum(match) / numel(match);
end

function cam = compute_gradcam(net, img, classes, featureLayer, useGPU)
    % Wrap GradCAM computation in dlfeval to properly trace gradients
    cam_dl = dlfeval(@gradcam_helper, net, img, classes, featureLayer, useGPU);
    cam = extractdata(cam_dl);
    cam = imresize(cam, [224, 224]);
    cam = single(cam ./ (max(cam(:)) + eps));
end

function cam = gradcam_helper(net, img, classes, featureLayer, useGPU)
    % Prepare image
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end
    img = imresize(img, [224, 224]);
    
    % Convert to dlarray
    dlX = dlarray(single(img), 'SSCB');
    if useGPU, dlX = gpuArray(dlX); end
    
    % Forward pass
    [featMap, logits] = forward(net, dlX, 'Outputs', {featureLayer, 'fc8'});
    logits = squeeze(logits);
    
    % Get class index and score
    [~, classIdx] = max(extractdata(logits));
    score = sum(logits(classIdx), 'all');
    
    % Compute gradients
    gradFeat = dlgradient(score, featMap);
    w = mean(gradFeat, [1 2]);
    
    % Compute CAM
    cam = sum(featMap .* w, 3);
    cam = max(cam, 0);
    
    % Normalize
    cam_max = max(cam, [], 'all');
    if cam_max > 0
        cam = cam ./ (cam_max + eps);
    end
end

function intensities = analyze_intensity_distribution(imds)
    intensities = zeros(numel(imds.Files), 1);
    for i = 1:numel(imds.Files)
        img = imread(imds.Files{i});
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        intensities(i) = mean(img(:));
    end
end

