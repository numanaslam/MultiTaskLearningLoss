function evaluate_ood_performance()
%EVALUATE_OOD_PERFORMANCE Evaluate trained model on OOD data with visualizations
%   This script loads the trained model and evaluates it on:
%   - In-Distribution (ID): ROI images
%   - Out-of-Distribution (OOD): Full CXR images
%   
%   Generates comprehensive visualizations and comparison reports.
%
%   Requirements:
%   - Trained model: models/final/final_model_focal_gradcam_tversky_kfold.mat
%   - ROI images: input/roi/
%   - Full CXR images: input/cxr/

clc; clear; close all;
fprintf('=== OUT-OF-DISTRIBUTION EVALUATION ===\n');
fprintf('Loading trained model and evaluating on ID/OOD datasets...\n\n');

%% Load Trained Model
modelFile = fullfile('models', 'final', 'final_model_focal_gradcam_tversky_kfold.mat');
if ~exist(modelFile, 'file')
    error('Trained model not found: %s\nPlease run train_final_model() first.', modelFile);
end

fprintf('Loading model from: %s\n', modelFile);
s = load(modelFile);
trainedNet = s.trainedNet;
config = s.config;
classes = {'normal', 'ptb'};  % Default classes, adjust if different

% Get classes from model if available
if isfield(s, 'results') && isfield(s.results, 'fold_results')
    % Try to infer classes from results
    fold_names = fieldnames(s.results.fold_results);
    if ~isempty(fold_names)
        % Classes should be in the confusion matrix or elsewhere
        % For now, use default
    end
end

fprintf('Model loaded successfully!\n\n');

%% Load Datasets
fprintf('Loading datasets...\n');

% Get base directory
roiDir = fullfile('input', 'roi');
roiImds = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
if numel(roiImds.Files) > 0
    firstImgPath = roiImds.Files{1};
    imgDir = fileparts(firstImgPath);
    roiDirPath = fileparts(imgDir);
    baseDir = fileparts(roiDirPath);
    cxrDir = fullfile(baseDir, 'cxr');
else
    baseDir = 'input';
    cxrDir = fullfile('input', 'cxr');
end

fprintf('  ROI directory: %s\n', roiDir);
fprintf('  CXR directory: %s\n', cxrDir);

% Load CXR dataset
if ~exist(cxrDir, 'dir')
    error('CXR directory not found: %s\nPlease ensure Full CXR images are in this directory.', cxrDir);
end

cxrImds = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(roiImds.Labels);

fprintf('  ROI dataset: %d samples\n', numel(roiImds.Files));
fprintf('  CXR dataset: %d samples\n', numel(cxrImds.Files));
fprintf('  Classes: %s\n\n', strjoin(classes, ', '));

%% Evaluate on ID (ROI) Dataset
fprintf('=== EVALUATING ON IN-DISTRIBUTION (ROI) DATA ===\n');
[resultsID, predictionsID] = evaluate_dataset_with_visualization(...
    trainedNet, roiImds, classes, config.useGPU, 'ID (ROI)');

fprintf('\nID (ROI) Results:\n');
fprintf('  Accuracy: %.3f\n', resultsID.accuracy);
fprintf('  Precision: %.3f\n', resultsID.precision);
fprintf('  Sensitivity: %.3f\n', resultsID.sensitivity);
fprintf('  Specificity: %.3f\n', resultsID.specificity);
fprintf('  F1-Score: %.3f\n', resultsID.f1_score);
fprintf('  AUC: %.3f\n', resultsID.auc);

%% Evaluate on OOD (Full CXR) Dataset
fprintf('\n=== EVALUATING ON OUT-OF-DISTRIBUTION (Full CXR) DATA ===\n');
[resultsOOD, predictionsOOD] = evaluate_dataset_with_visualization(...
    trainedNet, cxrImds, classes, config.useGPU, 'OOD (Full CXR)');

fprintf('\nOOD (Full CXR) Results:\n');
fprintf('  Accuracy: %.3f\n', resultsOOD.accuracy);
fprintf('  Precision: %.3f\n', resultsOOD.precision);
fprintf('  Sensitivity: %.3f\n', resultsOOD.sensitivity);
fprintf('  Specificity: %.3f\n', resultsOOD.specificity);
fprintf('  F1-Score: %.3f\n', resultsOOD.f1_score);
fprintf('  AUC: %.3f\n', resultsOOD.auc);

%% Calculate Performance Degradation
fprintf('\n=== PERFORMANCE DEGRADATION ANALYSIS ===\n');
degradation = struct();
degradation.accuracy = (resultsID.accuracy - resultsOOD.accuracy) / resultsID.accuracy * 100;
degradation.precision = (resultsID.precision - resultsOOD.precision) / resultsID.precision * 100;
degradation.sensitivity = (resultsID.sensitivity - resultsOOD.sensitivity) / resultsID.sensitivity * 100;
degradation.specificity = (resultsID.specificity - resultsOOD.specificity) / resultsID.specificity * 100;
degradation.f1_score = (resultsID.f1_score - resultsOOD.f1_score) / resultsID.f1_score * 100;
degradation.auc = (resultsID.auc - resultsOOD.auc) / resultsID.auc * 100;

mean_degradation = mean([degradation.accuracy, degradation.precision, ...
                        degradation.sensitivity, degradation.specificity, ...
                        degradation.f1_score, degradation.auc]);

fprintf('Performance Degradation:\n');
fprintf('  Accuracy: %.2f%%\n', degradation.accuracy);
fprintf('  Precision: %.2f%%\n', degradation.precision);
fprintf('  Sensitivity: %.2f%%\n', degradation.sensitivity);
fprintf('  Specificity: %.2f%%\n', degradation.specificity);
fprintf('  F1-Score: %.2f%%\n', degradation.f1_score);
fprintf('  AUC: %.2f%%\n', degradation.auc);
fprintf('  Mean Degradation: %.2f%%\n', mean_degradation);

% Interpret degradation
if mean_degradation < 5
    fprintf('  Status: ✓ Excellent generalization (<5%% degradation)\n');
elseif mean_degradation < 15
    fprintf('  Status: ✓ Good generalization (<15%% degradation)\n');
elseif mean_degradation < 30
    fprintf('  Status: ⚠ Moderate degradation (15-30%%)\n');
else
    fprintf('  Status: ✗ High degradation (>30%%)\n');
end

%% Create Visualizations
fprintf('\n=== GENERATING VISUALIZATIONS ===\n');
outputDir = fullfile('results', 'ood');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 1. Performance Comparison Bar Chart
create_performance_comparison_chart(resultsID, resultsOOD, degradation, outputDir);

% 2. Confusion Matrices
create_confusion_matrices(resultsID, resultsOOD, classes, outputDir);

% 3. ROC Curves
create_roc_curves(predictionsID, predictionsOOD, classes, outputDir);

% 4. Metric Degradation Chart
create_degradation_chart(degradation, mean_degradation, outputDir);

% 5. Prediction Distribution Comparison
create_prediction_distribution_comparison(predictionsID, predictionsOOD, outputDir);

% 6. Comprehensive Summary Figure
create_comprehensive_summary(resultsID, resultsOOD, degradation, mean_degradation, ...
    predictionsID, predictionsOOD, outputDir);

fprintf('\n=== EVALUATION COMPLETE ===\n');
fprintf('All visualizations saved to: %s\n', outputDir);
fprintf('\nSummary:\n');
fprintf('  ID Accuracy: %.3f\n', resultsID.accuracy);
fprintf('  OOD Accuracy: %.3f\n', resultsOOD.accuracy);
fprintf('  Mean Degradation: %.2f%%\n', mean_degradation);

end

%% Helper Functions

function [results, predictions] = evaluate_dataset_with_visualization(net, imds, classes, useGPU, datasetName)
% Evaluate dataset and return results with predictions
fprintf('  Evaluating %s dataset...\n', datasetName);

augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
reset(augDS);

Ypred = categorical.empty(0,1);
Yprobs = [];
Ytrue = imds.Labels(:);

batchCount = 0;
while hasdata(augDS)
    batchCount = batchCount + 1;
    if mod(batchCount, 10) == 0
        fprintf('    Processed %d batches...\n', batchCount);
    end
    
    tbl = read(augDS);
    imgs = cat(4, tbl.input{:});
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

% Calculate metrics
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

function create_performance_comparison_chart(resultsID, resultsOOD, degradation, outputDir)
% Create bar chart comparing ID vs OOD performance
figure('Position', [100, 100, 1200, 600]);

metrics = {'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC'};
id_values = [resultsID.accuracy, resultsID.precision, resultsID.sensitivity, ...
             resultsID.specificity, resultsID.f1_score, resultsID.auc];
ood_values = [resultsOOD.accuracy, resultsOOD.precision, resultsOOD.sensitivity, ...
              resultsOOD.specificity, resultsOOD.f1_score, resultsOOD.auc];

x = 1:numel(metrics);
width = 0.35;

subplot(1, 2, 1);
b1 = bar(x - width/2, id_values, width, 'FaceColor', [0.2 0.6 0.8], 'DisplayName', 'ID (ROI)');
hold on;
b2 = bar(x + width/2, ood_values, width, 'FaceColor', [0.8 0.4 0.2], 'DisplayName', 'OOD (Full CXR)');
set(gca, 'XTickLabel', metrics);
ylabel('Score');
title('Performance Comparison: ID vs OOD');
legend('Location', 'best');
grid on;
ylim([0, 1]);
xtickangle(45);

% Add value labels on bars
for i = 1:numel(metrics)
    text(i - width/2, id_values(i) + 0.02, sprintf('%.3f', id_values(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
    text(i + width/2, ood_values(i) + 0.02, sprintf('%.3f', ood_values(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Degradation chart
subplot(1, 2, 2);
degradation_values = [degradation.accuracy, degradation.precision, degradation.sensitivity, ...
                      degradation.specificity, degradation.f1_score, degradation.auc];
colors = repmat([0.8 0.2 0.2], numel(metrics), 1);
colors(degradation_values < 15, :) = repmat([0.2 0.8 0.2], sum(degradation_values < 15), 1);
colors(degradation_values >= 15 & degradation_values < 30, :) = ...
    repmat([1.0 0.6 0.0], sum(degradation_values >= 15 & degradation_values < 30), 1);

b = bar(x, degradation_values, 'FaceColor', 'flat', 'CData', colors);
set(gca, 'XTickLabel', metrics);
ylabel('Degradation (%)');
title('Performance Degradation');
grid on;
ylim([0, max(50, max(degradation_values) * 1.2)]);
xtickangle(45);

% Add value labels
for i = 1:numel(metrics)
    text(i, degradation_values(i) + max(50, max(degradation_values) * 1.2) * 0.02, ...
        sprintf('%.1f%%', degradation_values(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end

% Add reference lines
hold on;
plot([0.5, numel(metrics)+0.5], [5, 5], 'g--', 'LineWidth', 1, 'DisplayName', 'Excellent (<5%)');
plot([0.5, numel(metrics)+0.5], [15, 15], 'y--', 'LineWidth', 1, 'DisplayName', 'Good (<15%)');
plot([0.5, numel(metrics)+0.5], [30, 30], 'r--', 'LineWidth', 1, 'DisplayName', 'Moderate (<30%)');
legend('Location', 'best');

saveas(gcf, fullfile(outputDir, 'performance_comparison.png'));
fprintf('  Saved: performance_comparison.png\n');
close(gcf);
end

function create_confusion_matrices(resultsID, resultsOOD, classes, outputDir)
% Create confusion matrices for ID and OOD
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
cmID = resultsID.confusion_matrix;
imagesc(cmID);
colormap(gca, 'Blues');
colorbar;
textStrings = num2str(cmID(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(cmID, 2), 1:size(cmID, 1));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', ...
    'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', classes, 'YTickLabel', classes, 'XTick', 1:numel(classes), 'YTick', 1:numel(classes));
xlabel('Predicted');
ylabel('True');
title('Confusion Matrix: ID (ROI)');
axis square;

subplot(1, 2, 2);
cmOOD = resultsOOD.confusion_matrix;
imagesc(cmOOD);
colormap(gca, 'Reds');
colorbar;
textStrings = num2str(cmOOD(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(cmOOD, 2), 1:size(cmOOD, 1));
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', ...
    'Color', 'k', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTickLabel', classes, 'YTickLabel', classes, 'XTick', 1:numel(classes), 'YTick', 1:numel(classes));
xlabel('Predicted');
ylabel('True');
title('Confusion Matrix: OOD (Full CXR)');
axis square;

saveas(gcf, fullfile(outputDir, 'confusion_matrices.png'));
fprintf('  Saved: confusion_matrices.png\n');
close(gcf);
end

function create_roc_curves(predictionsID, predictionsOOD, classes, outputDir)
% Create ROC curves for ID and OOD
figure('Position', [100, 100, 1000, 800]);

if numel(classes) == 2
    % ID ROC
    [XID, YID, ~, AUCID] = perfcurve(double(predictionsID.Ytrue == classes{2}), ...
        predictionsID.Yprobs, 1);
    plot(XID, YID, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('ID (ROI), AUC=%.3f', AUCID));
    hold on;
    
    % OOD ROC
    [XOOD, YOOD, ~, AUCOOD] = perfcurve(double(predictionsOOD.Ytrue == classes{2}), ...
        predictionsOOD.Yprobs, 1);
    plot(XOOD, YOOD, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('OOD (Full CXR), AUC=%.3f', AUCOOD));
    
    % Diagonal reference
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1, 'DisplayName', 'Random (AUC=0.5)');
    
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curves: ID vs OOD');
    legend('Location', 'southeast');
    grid on;
    axis square;
end

saveas(gcf, fullfile(outputDir, 'roc_curves.png'));
fprintf('  Saved: roc_curves.png\n');
close(gcf);
end

function create_degradation_chart(degradation, mean_degradation, outputDir)
% Create detailed degradation chart
figure('Position', [100, 100, 1000, 600]);

metrics = {'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC'};
degradation_values = [degradation.accuracy, degradation.precision, degradation.sensitivity, ...
                      degradation.specificity, degradation.f1_score, degradation.auc];

% Color coding
colors = repmat([0.8 0.2 0.2], numel(metrics), 1);
colors(degradation_values < 5, :) = repmat([0.2 0.8 0.2], sum(degradation_values < 5), 1);
colors(degradation_values >= 5 & degradation_values < 15, :) = ...
    repmat([0.4 0.7 0.2], sum(degradation_values >= 5 & degradation_values < 15), 1);
colors(degradation_values >= 15 & degradation_values < 30, :) = ...
    repmat([1.0 0.6 0.0], sum(degradation_values >= 15 & degradation_values < 30), 1);

b = bar(1:numel(metrics), degradation_values, 'FaceColor', 'flat', 'CData', colors);
set(gca, 'XTickLabel', metrics);
ylabel('Degradation (%)');
title(sprintf('Performance Degradation Analysis (Mean: %.2f%%)', mean_degradation));
grid on;
ylim([0, max(50, max(degradation_values) * 1.2)]);

% Add value labels
for i = 1:numel(metrics)
    text(i, degradation_values(i) + max(50, max(degradation_values) * 1.2) * 0.02, ...
        sprintf('%.1f%%', degradation_values(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

% Add reference lines
hold on;
plot([0.5, numel(metrics)+0.5], [5, 5], 'g--', 'LineWidth', 2, 'DisplayName', 'Excellent (<5%)');
plot([0.5, numel(metrics)+0.5], [15, 15], 'y--', 'LineWidth', 2, 'DisplayName', 'Good (<15%)');
plot([0.5, numel(metrics)+0.5], [30, 30], 'r--', 'LineWidth', 2, 'DisplayName', 'Moderate (<30%)');
legend('Location', 'best');

xtickangle(45);

saveas(gcf, fullfile(outputDir, 'degradation_analysis.png'));
fprintf('  Saved: degradation_analysis.png\n');
close(gcf);
end

function create_prediction_distribution_comparison(predictionsID, predictionsOOD, outputDir)
% Compare prediction probability distributions
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
histogram(predictionsID.Yprobs, 30, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
xlabel('Predicted Probability (Positive Class)');
ylabel('Frequency');
title('ID (ROI): Prediction Probability Distribution');
grid on;
xlim([0, 1]);

subplot(1, 2, 2);
histogram(predictionsOOD.Yprobs, 30, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k');
xlabel('Predicted Probability (Positive Class)');
ylabel('Frequency');
title('OOD (Full CXR): Prediction Probability Distribution');
grid on;
xlim([0, 1]);

saveas(gcf, fullfile(outputDir, 'prediction_distributions.png'));
fprintf('  Saved: prediction_distributions.png\n');
close(gcf);
end

function create_comprehensive_summary(resultsID, resultsOOD, degradation, mean_degradation, ...
    predictionsID, predictionsOOD, outputDir)
% Create comprehensive summary figure with all key metrics
figure('Position', [50, 50, 1600, 1000]);

% 1. Performance metrics comparison
subplot(2, 3, 1);
metrics = {'Acc', 'Prec', 'Sens', 'Spec', 'F1', 'AUC'};
id_values = [resultsID.accuracy, resultsID.precision, resultsID.sensitivity, ...
             resultsID.specificity, resultsID.f1_score, resultsID.auc];
ood_values = [resultsOOD.accuracy, resultsOOD.precision, resultsOOD.sensitivity, ...
              resultsOOD.specificity, resultsOOD.f1_score, resultsOOD.auc];
x = 1:numel(metrics);
width = 0.35;
bar(x - width/2, id_values, width, 'FaceColor', [0.2 0.6 0.8], 'DisplayName', 'ID');
hold on;
bar(x + width/2, ood_values, width, 'FaceColor', [0.8 0.4 0.2], 'DisplayName', 'OOD');
set(gca, 'XTickLabel', metrics);
ylabel('Score');
title('Performance Metrics');
legend('Location', 'best');
grid on;
ylim([0, 1]);

% 2. Degradation
subplot(2, 3, 2);
degradation_values = [degradation.accuracy, degradation.precision, degradation.sensitivity, ...
                      degradation.specificity, degradation.f1_score, degradation.auc];
colors = repmat([0.8 0.2 0.2], numel(metrics), 1);
colors(degradation_values < 15, :) = repmat([0.2 0.8 0.2], sum(degradation_values < 15), 1);
bar(1:numel(metrics), degradation_values, 'FaceColor', 'flat', 'CData', colors);
set(gca, 'XTickLabel', metrics);
ylabel('Degradation (%)');
title(sprintf('Degradation (Mean: %.1f%%)', mean_degradation));
grid on;
hold on;
plot([0.5, numel(metrics)+0.5], [15, 15], 'y--', 'LineWidth', 1);
plot([0.5, numel(metrics)+0.5], [30, 30], 'r--', 'LineWidth', 1);

% 3. Confusion Matrix ID
subplot(2, 3, 3);
cmID = resultsID.confusion_matrix;
imagesc(cmID);
colormap(gca, 'Blues');
colorbar;
textStrings = num2str(cmID(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(cmID, 2), 1:size(cmID, 1));
text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', 'Color', 'k', 'FontWeight', 'bold');
set(gca, 'XTickLabel', {'Normal', 'PTB'}, 'YTickLabel', {'Normal', 'PTB'});
xlabel('Predicted');
ylabel('True');
title('Confusion Matrix: ID');
axis square;

% 4. Confusion Matrix OOD
subplot(2, 3, 4);
cmOOD = resultsOOD.confusion_matrix;
imagesc(cmOOD);
colormap(gca, 'Reds');
colorbar;
textStrings = num2str(cmOOD(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:size(cmOOD, 2), 1:size(cmOOD, 1));
text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center', 'Color', 'k', 'FontWeight', 'bold');
set(gca, 'XTickLabel', {'Normal', 'PTB'}, 'YTickLabel', {'Normal', 'PTB'});
xlabel('Predicted');
ylabel('True');
title('Confusion Matrix: OOD');
axis square;

% 5. ROC Curves
subplot(2, 3, 5);
classes = {'normal', 'ptb'};
if numel(classes) == 2 && isfield(predictionsID, 'Yprobs') && isfield(predictionsOOD, 'Yprobs')
    try
        [XID, YID, ~, AUCID] = perfcurve(double(predictionsID.Ytrue == classes{2}), ...
            predictionsID.Yprobs, 1);
        plot(XID, YID, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('ID (AUC=%.3f)', AUCID));
        hold on;
        [XOOD, YOOD, ~, AUCOOD] = perfcurve(double(predictionsOOD.Ytrue == classes{2}), ...
            predictionsOOD.Yprobs, 1);
        plot(XOOD, YOOD, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('OOD (AUC=%.3f)', AUCOOD));
        plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title('ROC Curves');
        legend('Location', 'southeast');
        grid on;
        axis square;
    catch
        axis off;
        text(0.5, 0.5, 'ROC curves unavailable', 'HorizontalAlignment', 'center');
    end
else
    axis off;
    text(0.5, 0.5, 'ROC curves unavailable', 'HorizontalAlignment', 'center');
end

% 6. Summary text
subplot(2, 3, 6);
axis off;
summary_text = {
    sprintf('EVALUATION SUMMARY'), '';
    sprintf('ID (ROI) Performance:'), ...
    sprintf('  Accuracy: %.3f', resultsID.accuracy), ...
    sprintf('  AUC: %.3f', resultsID.auc), ...
    sprintf('  F1-Score: %.3f', resultsID.f1_score), '';
    sprintf('OOD (Full CXR) Performance:'), ...
    sprintf('  Accuracy: %.3f', resultsOOD.accuracy), ...
    sprintf('  AUC: %.3f', resultsOOD.auc), ...
    sprintf('  F1-Score: %.3f', resultsOOD.f1_score), '';
    sprintf('Performance Degradation:'), ...
    sprintf('  Mean: %.2f%%', mean_degradation), ...
    sprintf('  Accuracy: %.2f%%', degradation.accuracy), ...
    sprintf('  AUC: %.2f%%', degradation.auc), '';
    sprintf('Status: %s', ...
        iif(mean_degradation < 5, 'Excellent', ...
            mean_degradation < 15, 'Good', ...
            mean_degradation < 30, 'Moderate', 'High Degradation'))
};
text(0.1, 0.5, summary_text, 'FontSize', 11, 'FontFamily', 'monospace', ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left');

sgtitle('Out-of-Distribution Evaluation: Comprehensive Summary', 'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, fullfile(outputDir, 'comprehensive_summary.png'));
fprintf('  Saved: comprehensive_summary.png\n');
close(gcf);
end

function result = iif(condition1, value1, condition2, value2, condition3, value3, default)
% Simple if-elseif-else helper function
if condition1
    result = value1;
elseif condition2
    result = value2;
elseif condition3
    result = value3;
else
    result = default;
end
end

