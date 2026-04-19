function analyze_ood_performance_degradation()
%ANALYZE_OOD_PERFORMANCE_DEGRADATION Analyze why OOD accuracy is still relatively low
%   This script performs comprehensive analysis to identify root causes of
%   OOD performance degradation:
%   1. Feature distribution analysis
%   2. Prediction confidence analysis
%   3. Misclassification pattern analysis
%   4. Intensity/statistical differences
%   5. Attention map analysis (GradCAM)
%   6. Class-wise performance breakdown

clc; close all;
fprintf('=== ANALYZING OOD PERFORMANCE DEGRADATION ===\n\n');

%% Configuration
modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');
if ~exist(modelFile, 'file')
    modelFile = fullfile('models', 'final', 'final_model_focal_gradcam_tversky_kfold.mat');
    if ~exist(modelFile, 'file')
        error('Model file not found.');
    end
end

% Data paths - use new resized dataset
roiDir = fullfile('input', 'roi');
cxrDir = fullfile('input',  'cxr');

% Output directory
outputDir = fullfile('results', 'ood_analysis');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Load Model
fprintf('Loading model: %s\n', modelFile);
s = load(modelFile);
trainedNet = s.trainedNet;
if isfield(s, 'config')
    config = s.config;
    useGPU = config.useGPU;
else
    useGPU = canUseGPU;
end
fprintf('Model loaded. GPU: %s\n', string(useGPU));

% Extract preprocessing information
preprocessing_method = 'none';
refHist = [];
if isfield(s, 'results') && isfield(s.results, 'preprocessing_method')
    preprocessing_method = s.results.preprocessing_method;
end
if isfield(s, 'config') && isfield(s.config, 'refHist')
    refHist = s.config.refHist;
end

% Display model configuration
if isfield(s, 'loss_config')
    loss_config = s.loss_config;
    fprintf('Model Configuration:\n');
    if isfield(loss_config, 'use_anatomical_guidance')
        fprintf('  Anatomical Guidance: %s\n', mat2str(loss_config.use_anatomical_guidance));
        if loss_config.use_anatomical_guidance && isfield(loss_config, 'lambda_anatomical')
            fprintf('  λ_anatomical: %.4f\n', loss_config.lambda_anatomical);
        end
    end
    if isfield(loss_config, 'use_focal')
        fprintf('  Focal Loss: %s\n', mat2str(loss_config.use_focal));
    end
    if isfield(loss_config, 'use_tversky')
        fprintf('  Tversky Loss: %s\n', mat2str(loss_config.use_tversky));
    end
    if isfield(loss_config, 'use_gradcam')
        fprintf('  GradCAM Loss: %s\n', mat2str(loss_config.use_gradcam));
    end
end
fprintf('  Preprocessing Method: %s\n', preprocessing_method);
fprintf('\n');

%% Load Datasets
fprintf('Loading datasets...\n');
imdsROI = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imdsROI.Labels);

fprintf('  ROI (ID): %d samples\n', numel(imdsROI.Files));
fprintf('  CXR (OOD): %d samples\n', numel(imdsCXR.Files));
fprintf('  Classes: %s\n', strjoin(classes, ', '));

% Compute reference histogram if needed and not saved
if strcmp(preprocessing_method, 'histmatch') && isempty(refHist)
    fprintf('\nComputing reference histogram from ROI images...\n');
    numRefSamples = min(50, numel(imdsROI.Files));
    refIdx = randperm(numel(imdsROI.Files), numRefSamples);
    refImages = cell(numRefSamples, 1);
    for i = 1:numRefSamples
        img = imread(imdsROI.Files{refIdx(i)});
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        refImages{i} = img;
    end
    % Compute average histogram
    allPixels = [];
    for i = 1:numRefSamples
        allPixels = [allPixels; refImages{i}(:)];
    end
    refHist = imhist(uint8(allPixels));
    fprintf('  Reference histogram computed from %d samples\n', numRefSamples);
end
fprintf('\n');

% Compute reference histogram if needed and not saved
if strcmp(preprocessing_method, 'histmatch') && isempty(refHist)
    fprintf('Computing reference histogram from ROI images...\n');
    numRefSamples = min(50, numel(imdsROI.Files));
    refIdx = randperm(numel(imdsROI.Files), numRefSamples);
    refImages = cell(numRefSamples, 1);
    for i = 1:numRefSamples
        img = imread(imdsROI.Files{refIdx(i)});
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        refImages{i} = img;
    end
    % Compute average histogram
    allPixels = [];
    for i = 1:numRefSamples
        allPixels = [allPixels; refImages{i}(:)];
    end
    refHist = imhist(uint8(allPixels));
    fprintf('  Reference histogram computed from %d samples\n\n', numRefSamples);
end

%% 1. INTENSITY STATISTICS ANALYSIS
fprintf('=== 1. INTENSITY STATISTICS ANALYSIS ===\n');
analyze_intensity_statistics(imdsROI, imdsCXR, outputDir);

%% 2. PREDICTION CONFIDENCE ANALYSIS
fprintf('\n=== 2. PREDICTION CONFIDENCE ANALYSIS ===\n');
[confID, confOOD, predID, predOOD, trueID, trueOOD] = analyze_prediction_confidence(...
    trainedNet, imdsROI, imdsCXR, classes, useGPU, outputDir, preprocessing_method, refHist);

%% 3. MISCLASSIFICATION PATTERN ANALYSIS
fprintf('\n=== 3. MISCLASSIFICATION PATTERN ANALYSIS ===\n');
analyze_misclassification_patterns(predID, predOOD, trueID, trueOOD, classes, outputDir);

%% 4. FEATURE DISTRIBUTION ANALYSIS
fprintf('\n=== 4. FEATURE DISTRIBUTION ANALYSIS ===\n');
analyze_feature_distributions(trainedNet, imdsROI, imdsCXR, classes, useGPU, outputDir);

%% 5. ATTENTION MAP ANALYSIS (GradCAM)
fprintf('\n=== 5. ATTENTION MAP ANALYSIS ===\n');
analyze_attention_maps(trainedNet, imdsROI, imdsCXR, classes, useGPU, outputDir);

%% 6. CLASS-WISE PERFORMANCE BREAKDOWN
fprintf('\n=== 6. CLASS-WISE PERFORMANCE BREAKDOWN ===\n');
analyze_classwise_performance(predID, predOOD, trueID, trueOOD, classes, outputDir);

%% 7. GENERATE SUMMARY REPORT
fprintf('\n=== GENERATING SUMMARY REPORT ===\n');
generate_summary_report(outputDir, confID, confOOD, predID, predOOD, trueID, trueOOD, classes);

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Results saved to: %s\n', outputDir);
end

%% Helper Functions

function analyze_intensity_statistics(imdsROI, imdsCXR, outputDir)
    fprintf('  Computing intensity statistics...\n');
    
    % Sample images for analysis
    numSamples = min(100, numel(imdsROI.Files));
    roiIntensities = zeros(numSamples, 1);
    cxrIntensities = zeros(numSamples, 1);
    
    for i = 1:numSamples
        roiImg = imread(imdsROI.Files{i});
        if size(roiImg, 3) > 1, roiImg = rgb2gray(roiImg); end
        roiIntensities(i) = mean(double(roiImg(:)));
        
        cxrImg = imread(imdsCXR.Files{i});
        if size(cxrImg, 3) > 1, cxrImg = rgb2gray(cxrImg); end
        cxrIntensities(i) = mean(double(cxrImg(:)));
    end
    
    fprintf('    ROI mean intensity: %.2f ± %.2f\n', mean(roiIntensities), std(roiIntensities));
    fprintf('    CXR mean intensity: %.2f ± %.2f\n', mean(cxrIntensities), std(cxrIntensities));
    fprintf('    Difference: %.2f\n', mean(cxrIntensities) - mean(roiIntensities));
    
    % Plot
    figure('Position', [100, 100, 1200, 400]);
    subplot(1, 3, 1);
    histogram(roiIntensities, 30, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'ROI (ID)');
    hold on;
    histogram(cxrIntensities, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'CXR (OOD)');
    xlabel('Mean Intensity');
    ylabel('Frequency');
    title('Intensity Distribution Comparison');
    legend('Location', 'best');
    grid on;
    
    subplot(1, 3, 2);
    % Create grouping variable for boxplot
    allIntensities = [roiIntensities; cxrIntensities];
    % Remove any NaN or Inf values
    validIdx = isfinite(allIntensities);
    allIntensities = allIntensities(validIdx);
    group = [ones(numel(roiIntensities), 1); 2*ones(numel(cxrIntensities), 1)];
    group = group(validIdx);
    
    % Try boxplot, fallback to bar plot if it fails
    try
        if numel(unique(allIntensities)) > 1 && numel(allIntensities) > 0
            boxplot(allIntensities, group, 'Labels', {'ROI (ID)', 'CXR (OOD)'});
            ylabel('Mean Intensity');
            title('Intensity Box Plot');
            grid on;
        else
            error('Insufficient data variation');
        end
    catch
        % Fallback: use bar plot with error bars
        bar([mean(roiIntensities), mean(cxrIntensities)]);
        hold on;
        errorbar([1, 2], [mean(roiIntensities), mean(cxrIntensities)], ...
                 [std(roiIntensities), std(cxrIntensities)], 'k.', 'LineWidth', 1.5);
        set(gca, 'XTickLabel', {'ROI (ID)', 'CXR (OOD)'});
        ylabel('Mean Intensity');
        title('Intensity Comparison');
        grid on;
    end
    
    subplot(1, 3, 3);
    scatter(roiIntensities, cxrIntensities, 20, 'filled');
    hold on;
    plot([0, 255], [0, 255], 'r--', 'LineWidth', 2);
    xlabel('ROI Intensity');
    ylabel('CXR Intensity');
    title('Intensity Correlation');
    axis equal;
    grid on;
    
    saveas(gcf, fullfile(outputDir, 'intensity_statistics.png'));
    close(gcf);
end

function [confID, confOOD, predID, predOOD, trueID, trueOOD] = analyze_prediction_confidence(...
    net, imdsROI, imdsCXR, classes, useGPU, outputDir, preprocessing_method, refHist)
    fprintf('  Analyzing prediction confidence...\n');
    fprintf('  Using preprocessing: %s\n', preprocessing_method);
    
    % Use preprocessing-aware evaluation if preprocessing is specified
    if ~strcmp(preprocessing_method, 'none')
        % Use preprocessing-aware evaluation
        fprintf('  Applying preprocessing during evaluation...\n');
        try
            [resultsID, predStructID] = evaluate_dataset_with_preprocessing(...
                net, imdsROI, classes, useGPU, preprocessing_method, refHist);
            [resultsOOD, predStructOOD] = evaluate_dataset_with_preprocessing(...
                net, imdsCXR, classes, useGPU, preprocessing_method, refHist);
            
            predID = predStructID.Ypred;
            confID = predStructID.Yprobs;
            trueID = predStructID.Ytrue;
            
            predOOD = predStructOOD.Ypred;
            confOOD = predStructOOD.Yprobs;
            trueOOD = predStructOOD.Ytrue;
        catch ME
            fprintf('  Warning: Preprocessing evaluation failed: %s\n', ME.message);
            fprintf('  Falling back to method without preprocessing...\n');
            [predID, confID, trueID] = evaluate_with_confidence(net, imdsROI, classes, useGPU);
            [predOOD, confOOD, trueOOD] = evaluate_with_confidence(net, imdsCXR, classes, useGPU);
        end
    else
        % No preprocessing specified
        fprintf('  No preprocessing applied (preprocessing_method = ''none'')\n');
        [predID, confID, trueID] = evaluate_with_confidence(net, imdsROI, classes, useGPU);
        [predOOD, confOOD, trueOOD] = evaluate_with_confidence(net, imdsCXR, classes, useGPU);
    end
    
    fprintf('    ID mean confidence: %.3f ± %.3f\n', mean(confID), std(confID));
    fprintf('    OOD mean confidence: %.3f ± %.3f\n', mean(confOOD), std(confOOD));
    fprintf('    Confidence drop: %.3f\n', mean(confID) - mean(confOOD));
    
    % Plot
    figure('Position', [100, 100, 1200, 400]);
    subplot(1, 3, 1);
    histogram(confID, 30, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'ID');
    hold on;
    histogram(confOOD, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'OOD');
    xlabel('Prediction Confidence');
    ylabel('Frequency');
    title('Confidence Distribution');
    legend('Location', 'best');
    grid on;
    
    subplot(1, 3, 2);
    % Create grouping variable for boxplot
    allConf = [confID; confOOD];
    % Remove any NaN or Inf values
    validIdx = isfinite(allConf);
    allConf = allConf(validIdx);
    group = [ones(numel(confID), 1); 2*ones(numel(confOOD), 1)];
    group = group(validIdx);
    
    % Try boxplot, fallback to bar plot if it fails
    try
        if numel(unique(allConf)) > 1 && numel(allConf) > 0
            boxplot(allConf, group, 'Labels', {'ID', 'OOD'});
            ylabel('Confidence');
            title('Confidence Box Plot');
            grid on;
        else
            error('Insufficient data variation');
        end
    catch
        % Fallback: use bar plot with error bars
        bar([mean(confID), mean(confOOD)]);
        hold on;
        errorbar([1, 2], [mean(confID), mean(confOOD)], ...
                 [std(confID), std(confOOD)], 'k.', 'LineWidth', 1.5);
        set(gca, 'XTickLabel', {'ID', 'OOD'});
        ylabel('Mean Confidence');
        title('Confidence Comparison');
        grid on;
    end
    
    subplot(1, 3, 3);
    correctID = (predID == trueID);
    correctOOD = (predOOD == trueOOD);
    % Ensure we have matching indices
    numSamples = min(numel(confID), numel(confOOD));
    confID_plot = confID(1:numSamples);
    confOOD_plot = confOOD(1:numSamples);
    correctID_plot = correctID(1:numSamples);
    correctOOD_plot = correctOOD(1:numSamples);
    
    % Plot correct predictions
    correctBoth = correctID_plot & correctOOD_plot;
    if any(correctBoth)
        scatter(confID_plot(correctBoth), confOOD_plot(correctBoth), 20, 'g', 'filled', 'DisplayName', 'Both Correct');
        hold on;
    end
    
    % Plot incorrect predictions
    incorrectBoth = ~correctID_plot | ~correctOOD_plot;
    if any(incorrectBoth)
        scatter(confID_plot(incorrectBoth), confOOD_plot(incorrectBoth), 20, 'r', 'filled', 'DisplayName', 'Any Incorrect');
    end
    
    xlabel('ID Confidence');
    ylabel('OOD Confidence');
    title('Confidence: ID vs OOD');
    legend('Location', 'best');
    grid on;
    
    saveas(gcf, fullfile(outputDir, 'confidence_analysis.png'));
    close(gcf);
end

function [predictions, confidences, trueLabels] = evaluate_with_confidence(net, imds, classes, useGPU)
    augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
    reset(augDS);
    
    predictions = categorical.empty(0,1);
    confidences = [];
    trueLabels = imds.Labels(:);
    
    while hasdata(augDS)
        tbl = read(augDS);
        imgs = cat(4, tbl.input{:});
        imgs = single(imgs);
        if useGPU, imgs = gpuArray(imgs); end
        dlX = dlarray(imgs, 'SSCB');
        
        sc = predict(net, dlX);
        probs = extractdata(sc);
        if ndims(probs) == 4
            probs = squeeze(probs);
        end
        
        if size(probs, 1) == numel(classes)
            [maxProbs, maxIdx] = max(probs, [], 1);
            lab = categorical(classes(maxIdx));
            predictions = [predictions; lab(:)];
            confidences = [confidences; maxProbs(:)];
        end
    end
    
    confidences = double(confidences);
end

function analyze_misclassification_patterns(predID, predOOD, trueID, trueOOD, classes, outputDir)
    fprintf('  Analyzing misclassification patterns...\n');
    
    % Confusion matrices
    cmID = confusionmat(trueID, predID, 'Order', categorical(classes));
    cmOOD = confusionmat(trueOOD, predOOD, 'Order', categorical(classes));
    
    fprintf('    ID Accuracy: %.3f\n', sum(predID == trueID) / numel(trueID));
    fprintf('    OOD Accuracy: %.3f\n', sum(predOOD == trueOOD) / numel(trueOOD));
    
    % Class-wise errors
    for c = 1:numel(classes)
        classIdx = (trueID == classes{c});
        idErr = sum(predID(classIdx) ~= trueID(classIdx)) / sum(classIdx) * 100;
        
        classIdxOOD = (trueOOD == classes{c});
        oodErr = sum(predOOD(classIdxOOD) ~= trueOOD(classIdxOOD)) / sum(classIdxOOD) * 100;
        
        fprintf('    %s: ID error=%.1f%%, OOD error=%.1f%%\n', classes{c}, idErr, oodErr);
    end
    
    % Plot confusion matrices
    figure('Position', [100, 100, 1000, 400]);
    subplot(1, 2, 1);
    imagesc(cmID);
    colorbar;
    colormap(gca, 'hot');
    set(gca, 'XTickLabel', classes, 'YTickLabel', classes);
    xlabel('Predicted');
    ylabel('True');
    title('ID Confusion Matrix');
    for i = 1:numel(classes)
        for j = 1:numel(classes)
            text(j, i, num2str(cmID(i,j)), 'HorizontalAlignment', 'center', 'Color', 'white');
        end
    end
    
    subplot(1, 2, 2);
    imagesc(cmOOD);
    colorbar;
    colormap(gca, 'hot');
    set(gca, 'XTickLabel', classes, 'YTickLabel', classes);
    xlabel('Predicted');
    ylabel('True');
    title('OOD Confusion Matrix');
    for i = 1:numel(classes)
        for j = 1:numel(classes)
            text(j, i, num2str(cmOOD(i,j)), 'HorizontalAlignment', 'center', 'Color', 'white');
        end
    end
    
    saveas(gcf, fullfile(outputDir, 'misclassification_patterns.png'));
    close(gcf);
end

function analyze_feature_distributions(net, imdsROI, imdsCXR, classes, useGPU, outputDir)
    fprintf('  Extracting features from intermediate layer...\n');
    
    % Extract features from a middle layer (e.g., relu5_3 for VGG16)
    try
        % Try to get features from a convolutional layer
        layerName = 'relu5_3';
        if isa(net, 'dlnetwork')
            % For dlnetwork, we need to use forward pass
            featuresID = extract_features_dlnetwork(net, imdsROI, layerName, useGPU, 50);
            featuresOOD = extract_features_dlnetwork(net, imdsCXR, layerName, useGPU, 50);
        else
            fprintf('    Warning: Feature extraction not implemented for this network type\n');
            return;
        end
        
        % Reduce dimensionality using PCA for visualization
        allFeatures = [featuresID; featuresOOD];
        [coeff, score, ~] = pca(allFeatures);
        scoreID = score(1:size(featuresID,1), 1:2);
        scoreOOD = score(size(featuresID,1)+1:end, 1:2);
        
        % Plot
        figure('Position', [100, 100, 800, 600]);
        scatter(scoreID(:,1), scoreID(:,2), 30, 'b', 'filled', 'DisplayName', 'ID (ROI)');
        hold on;
        scatter(scoreOOD(:,1), scoreOOD(:,2), 30, 'r', 'filled', 'DisplayName', 'OOD (CXR)');
        xlabel('PC1');
        ylabel('PC2');
        title('Feature Space Visualization (PCA)');
        legend('Location', 'best');
        grid on;
        
        saveas(gcf, fullfile(outputDir, 'feature_distributions.png'));
        close(gcf);
        
        fprintf('    Feature space separation analyzed\n');
    catch ME
        fprintf('    Warning: Feature extraction failed: %s\n', ME.message);
    end
end

function features = extract_features_dlnetwork(net, imds, layerName, useGPU, maxSamples)
    augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
    reset(augDS);
    
    features = [];
    count = 0;
    
    while hasdata(augDS) && count < maxSamples
        tbl = read(augDS);
        imgs = cat(4, tbl.input{:});
        imgs = single(imgs);
        if useGPU, imgs = gpuArray(imgs); end
        dlX = dlarray(imgs, 'SSCB');
        
        try
            % Forward pass to get features
            dlY = forward(net, dlX, 'Outputs', layerName);
            feat = extractdata(dlY);
            feat = reshape(feat, [], size(feat, 4));
            features = [features; feat'];
            count = count + size(imgs, 4);
        catch
            break;
        end
    end
end

function analyze_attention_maps(net, imdsROI, imdsCXR, classes, useGPU, outputDir)
    fprintf('  Analyzing attention maps (GradCAM)...\n');
    
    % Sample a few images
    numSamples = 5;
    sampleIdx = round(linspace(1, min(numel(imdsROI.Files), numel(imdsCXR.Files)), numSamples));
    
    figure('Position', [100, 100, 1600, 800]);
    
    for i = 1:numSamples
        % ID image
        roiImg = imread(imdsROI.Files{sampleIdx(i)});
        if size(roiImg, 3) > 1, roiImg = rgb2gray(roiImg); end
        roiImg = imresize(roiImg, [224 224]);
        roiLabel = imdsROI.Labels(sampleIdx(i));
        
        % OOD image
        cxrImg = imread(imdsCXR.Files{sampleIdx(i)});
        if size(cxrImg, 3) > 1, cxrImg = rgb2gray(cxrImg); end
        cxrImg = imresize(cxrImg, [224 224]);
        cxrLabel = imdsCXR.Labels(sampleIdx(i));
        
        % Compute GradCAM
        try
            roiCAM = compute_gradcam_simple(net, roiImg, roiLabel, classes, useGPU);
            cxrCAM = compute_gradcam_simple(net, cxrImg, cxrLabel, classes, useGPU);
        catch
            roiCAM = ones(224, 224) * 0.5;
            cxrCAM = ones(224, 224) * 0.5;
        end
        
        % Plot
        subplot(numSamples, 4, (i-1)*4 + 1);
        imshow(roiImg);
        title(sprintf('ID: %s', string(roiLabel)));
        
        subplot(numSamples, 4, (i-1)*4 + 2);
        imshow(roiCAM);
        title('ID GradCAM');
        
        subplot(numSamples, 4, (i-1)*4 + 3);
        imshow(cxrImg);
        title(sprintf('OOD: %s', string(cxrLabel)));
        
        subplot(numSamples, 4, (i-1)*4 + 4);
        imshow(cxrCAM);
        title('OOD GradCAM');
    end
    
    sgtitle('Attention Map Comparison: ID vs OOD', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(gcf, fullfile(outputDir, 'attention_maps.png'));
    close(gcf);
end

function cam = compute_gradcam_simple(net, img, label, classes, useGPU)
    try
        if size(img, 3) == 1
            img = repmat(img, [1 1 3]);
        end
        img = single(img);
        if useGPU, img = gpuArray(img); end
        dlX = dlarray(img, 'SSCB');
        
        % Use MATLAB's gradCAM if available
        if isa(net, 'dlnetwork')
            cam = gradCAM(net, img, label, 'FeatureLayer', 'relu5_3');
        else
            cam = ones(224, 224) * 0.5;
        end
    catch
        cam = ones(224, 224) * 0.5;
    end
end

function analyze_classwise_performance(predID, predOOD, trueID, trueOOD, classes, outputDir)
    fprintf('  Analyzing class-wise performance...\n');
    
    metrics = struct();
    for c = 1:numel(classes)
        className = classes{c};
        
        % ID metrics
        idClassIdx = (trueID == classes{c});
        idPredClass = (predID == classes{c});
        idTP = sum(idClassIdx & idPredClass);
        idFP = sum(~idClassIdx & idPredClass);
        idFN = sum(idClassIdx & ~idPredClass);
        idTN = sum(~idClassIdx & ~idPredClass);
        
        idPrecision = idTP / (idTP + idFP + eps);
        idRecall = idTP / (idTP + idFN + eps);
        idF1 = 2 * idPrecision * idRecall / (idPrecision + idRecall + eps);
        
        % OOD metrics
        oodClassIdx = (trueOOD == classes{c});
        oodPredClass = (predOOD == classes{c});
        oodTP = sum(oodClassIdx & oodPredClass);
        oodFP = sum(~oodClassIdx & oodPredClass);
        oodFN = sum(oodClassIdx & ~oodPredClass);
        oodTN = sum(~oodClassIdx & ~oodPredClass);
        
        oodPrecision = oodTP / (oodTP + oodFP + eps);
        oodRecall = oodTP / (oodTP + oodFN + eps);
        oodF1 = 2 * oodPrecision * oodRecall / (oodPrecision + oodRecall + eps);
        
        metrics.(className) = struct();
        metrics.(className).id = struct('precision', idPrecision, 'recall', idRecall, 'f1', idF1);
        metrics.(className).ood = struct('precision', oodPrecision, 'recall', oodRecall, 'f1', oodF1);
        
        fprintf('    %s:\n', className);
        fprintf('      ID:  Precision=%.3f, Recall=%.3f, F1=%.3f\n', idPrecision, idRecall, idF1);
        fprintf('      OOD: Precision=%.3f, Recall=%.3f, F1=%.3f\n', oodPrecision, oodRecall, oodF1);
    end
    
    % Plot
    figure('Position', [100, 100, 1200, 400]);
    metricNames = {'Precision', 'Recall', 'F1-Score'};
    for m = 1:3
        subplot(1, 3, m);
        idVals = [];
        oodVals = [];
        for c = 1:numel(classes)
            className = classes{c};
            switch m
                case 1
                    idVals(c) = metrics.(className).id.precision;
                    oodVals(c) = metrics.(className).ood.precision;
                case 2
                    idVals(c) = metrics.(className).id.recall;
                    oodVals(c) = metrics.(className).ood.recall;
                case 3
                    idVals(c) = metrics.(className).id.f1;
                    oodVals(c) = metrics.(className).ood.f1;
            end
        end
        
        x = 1:numel(classes);
        width = 0.35;
        bar(x - width/2, idVals * 100, width, 'FaceColor', 'b', 'DisplayName', 'ID');
        hold on;
        bar(x + width/2, oodVals * 100, width, 'FaceColor', 'r', 'DisplayName', 'OOD');
        set(gca, 'XTickLabel', classes);
        ylabel('Score (%)');
        title(metricNames{m});
        legend('Location', 'best');
        grid on;
        ylim([0, 100]);
    end
    
    saveas(gcf, fullfile(outputDir, 'classwise_performance.png'));
    close(gcf);
end

function generate_summary_report(outputDir, confID, confOOD, predID, predOOD, trueID, trueOOD, classes)
    fid = fopen(fullfile(outputDir, 'analysis_summary.txt'), 'w');
    
    fprintf(fid, '=== OOD PERFORMANCE DEGRADATION ANALYSIS SUMMARY ===\n\n');
    
    % Overall metrics
    idAcc = sum(predID == trueID) / numel(trueID);
    oodAcc = sum(predOOD == trueOOD) / numel(trueOOD);
    degradation = (idAcc - oodAcc) / idAcc * 100;
    
    fprintf(fid, 'Overall Performance:\n');
    fprintf(fid, '  ID Accuracy:   %.3f (%.1f%%)\n', idAcc, idAcc*100);
    fprintf(fid, '  OOD Accuracy:  %.3f (%.1f%%)\n', oodAcc, oodAcc*100);
    fprintf(fid, '  Degradation:   %.2f%%\n\n', degradation);
    
    % Confidence analysis
    fprintf(fid, 'Confidence Analysis:\n');
    fprintf(fid, '  ID mean confidence:  %.3f ± %.3f\n', mean(confID), std(confID));
    fprintf(fid, '  OOD mean confidence: %.3f ± %.3f\n', mean(confOOD), std(confOOD));
    fprintf(fid, '  Confidence drop:     %.3f\n\n', mean(confID) - mean(confOOD));
    
    % Class-wise breakdown
    fprintf(fid, 'Class-wise Performance:\n');
    for c = 1:numel(classes)
        className = classes{c};
        idClassIdx = (trueID == classes{c});
        oodClassIdx = (trueOOD == classes{c});
        
        idErr = sum(predID(idClassIdx) ~= trueID(idClassIdx)) / sum(idClassIdx) * 100;
        oodErr = sum(predOOD(oodClassIdx) ~= trueOOD(oodClassIdx)) / sum(oodClassIdx) * 100;
        
        fprintf(fid, '  %s:\n', className);
        fprintf(fid, '    ID error rate:  %.1f%%\n', idErr);
        fprintf(fid, '    OOD error rate: %.1f%%\n', oodErr);
        fprintf(fid, '    Error increase: %.1f%%\n\n', oodErr - idErr);
    end
    
    % Key findings
    fprintf(fid, 'Key Findings:\n');
    fprintf(fid, '1. Domain shift: Model trained on ROI (cropped) but tested on full CXR\n');
    fprintf(fid, '2. Confidence drop: OOD predictions are less confident\n');
    fprintf(fid, '3. Class imbalance: Check if one class degrades more than the other\n');
    fprintf(fid, '4. Feature mismatch: ROI and CXR may have different feature distributions\n\n');
    
    % Recommendations
    fprintf(fid, 'Recommendations:\n');
    fprintf(fid, '1. Train on mixed dataset (ROI + Full CXR)\n');
    fprintf(fid, '2. Use domain adaptation techniques\n');
    fprintf(fid, '3. Apply better preprocessing (CLAHE + Z-score showed improvement)\n');
    fprintf(fid, '4. Consider adversarial training for domain robustness\n');
    fprintf(fid, '5. Fine-tune on OOD data (if available)\n');
    
    fclose(fid);
    fprintf('  Summary report saved\n');
end

%% Preprocessing Functions (copied from train_final_model_with_preprocessing_anatomical.m)

function [results, predictions] = evaluate_dataset_with_preprocessing(net, imds, classes, useGPU, ...
    preprocessing_method, refHist)
% Evaluate dataset with preprocessing (same as evaluate_dataset_simple but with preprocessing)

augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
reset(augDS);

Ypred = categorical.empty(0,1);
Yprobs = [];
Ytrue = imds.Labels(:);

while hasdata(augDS)
    tbl = read(augDS);
    imgs = cat(4, tbl.input{:});
    
    % Apply preprocessing
    imgs = apply_preprocessing_batch(imgs, preprocessing_method, refHist);
    
    if useGPU, imgs = gpuArray(imgs); end
    dlX = dlarray(single(imgs), 'SSCB');
    
    sc = predict(net, dlX);
    lab = onehotdecode(extractdata(sc), classes, 1);
    Ypred = [Ypred; lab(:)];
    
    probs_batch = extractdata(sc);
    if size(probs_batch, 1) == numel(classes)
        if numel(classes) == 2
            Yprobs = [Yprobs; probs_batch(2,:)'];
        else
            Yprobs = [Yprobs; max(probs_batch, [], 1)'];
        end
    else
        Yprobs = [Yprobs; probs_batch(:)];
    end
end

% Calculate metrics (same as evaluate_dataset_simple)
match = (categorical(Ytrue) == categorical(Ypred));
accuracy = sum(match) / numel(match);

cm = confusionmat(Ytrue, Ypred, 'Order', categorical(classes));
if numel(classes)==2 && all(size(cm)==[2 2])
    TP = cm(2,2); FP = cm(1,2); FN = cm(2,1); TN = cm(1,1);
else
    [~,posIdx] = min(countcats(Ytrue));
    pos = classes{posIdx};
    TP = sum(Ypred==pos & Ytrue==pos);
    FP = sum(Ypred==pos & Ytrue~=pos);
    FN = sum(Ypred~=pos & Ytrue==pos);
    TN = sum(Ypred~=pos & Ytrue~=pos);
end

precision = TP / (TP + FP + eps);
sensitivity = TP / (TP + FN + eps);
specificity = TN / (TN + FP + eps);
f1score = 2*precision*sensitivity / (precision + sensitivity + eps);

try
    if numel(classes) == 2
        [~,~,~,auc] = perfcurve(double(Ytrue == classes{2}), Yprobs, 1);
    else
        auc = NaN;
    end
catch
    auc = 0.5;
end

results = struct();
results.accuracy = accuracy;
results.precision = precision;
results.sensitivity = sensitivity;
results.specificity = specificity;
results.f1_score = f1score;
results.auc = auc;
results.confusion_matrix = cm;

predictions = struct();
predictions.Ypred = Ypred;
predictions.Ytrue = Ytrue;
predictions.Yprobs = Yprobs;
end

function imgs_processed = apply_preprocessing_batch(imgs, method, refHist)
% Apply preprocessing to batch of images (same as evaluate_ood_with_preprocessing.m)
[H, W, C, B] = size(imgs);
imgs_processed = zeros(size(imgs), 'uint8');

for b = 1:B
    img = imgs(:, :, :, b);
    
    if C == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    switch method
        case 'none'
            img_processed = img_gray;
        case 'clahe'
            img_processed = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
        case 'histmatch'
            if ~isempty(refHist)
                img_processed = histeq(img_gray, refHist);
            else
                img_processed = img_gray;
            end
        case 'zscore'
            img_double = double(img_gray);
            img_mean = mean(img_double(:));
            img_std = std(img_double(:));
            if img_std > 0
                img_normalized = (img_double - img_mean) / img_std;
                img_normalized = (img_normalized - min(img_normalized(:))) / ...
                    (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
            else
                img_normalized = img_double;
            end
            img_processed = uint8(img_normalized);
        case 'minmax'
            img_double = double(img_gray);
            img_min = min(img_double(:));
            img_max = max(img_double(:));
            if img_max > img_min
                img_normalized = (img_double - img_min) / (img_max - img_min) * 255;
            else
                img_normalized = img_double;
            end
            img_processed = uint8(img_normalized);
        case 'clahe_zscore'
            img_clahe = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
            img_double = double(img_clahe);
            img_mean = mean(img_double(:));
            img_std = std(img_double(:));
            if img_std > 0
                img_normalized = (img_double - img_mean) / img_std;
                img_normalized = (img_normalized - min(img_normalized(:))) / ...
                    (max(img_normalized(:)) - min(img_normalized(:)) + eps) * 255;
            else
                img_normalized = img_double;
            end
            img_processed = uint8(img_normalized);
        otherwise
            img_processed = img_gray;
    end
    
    if C == 3
        imgs_processed(:, :, :, b) = repmat(img_processed, [1, 1, 3]);
    else
        imgs_processed(:, :, :, b) = img_processed;
    end
end
end

