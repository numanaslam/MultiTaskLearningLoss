function analyze_loss_components(modelFile)
%ANALYZE_LOSS_COMPONENTS Analyze loss component values from training
%   This script loads a trained model and analyzes the loss components
%   to understand why anatomical guidance might not be working.
%
%   Input:
%       modelFile - Path to trained model (optional)

if nargin < 1
    modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');
end

clc; close all;
fprintf('=== ANALYZING LOSS COMPONENTS ===\n\n');

%% Load Model
fprintf('Loading model: %s\n', modelFile);
if ~exist(modelFile, 'file')
    error('Model file not found: %s', modelFile);
end

s = load(modelFile);
if isfield(s, 'training_histories')
    training_histories = s.training_histories;
else
    error('Training histories not found in model file.');
end

if isfield(s, 'loss_config')
    loss_config = s.loss_config;
    fprintf('Loss Configuration:\n');
    fprintf('  λ_cam: %.4f\n', loss_config.lambda_cam);
    fprintf('  λ_tversky: %.4f\n', loss_config.lambda_tversky);
    fprintf('  λ_anatomical: %.4f\n', loss_config.lambda_anatomical);
    fprintf('  Use anatomical guidance: %s\n', mat2str(loss_config.use_anatomical_guidance));
else
    fprintf('  ⚠ Loss config not found in model file\n');
end

fprintf('\n');

%% Analyze Training Histories
fold_names = fieldnames(training_histories);
num_folds = numel(fold_names);

fprintf('Analyzing %d folds...\n\n', num_folds);

% Extract loss values (use cell arrays to handle variable lengths)
all_train_losses = {};
all_val_losses = {};
all_train_accs = {};
all_val_accs = {};
all_val_ious = {};
all_val_dices = {};

for f = 1:num_folds
    hist = training_histories.(fold_names{f});
    
    if isfield(hist, 'epoch_loss')
        train_losses = hist.epoch_loss;
        all_train_losses{end+1} = train_losses(:);
    end
    
    if isfield(hist, 'val_loss')
        val_losses = hist.val_loss(~isnan(hist.val_loss));
        all_val_losses{end+1} = val_losses(:);
    end
    
    if isfield(hist, 'train_acc')
        train_accs = hist.train_acc;
        all_train_accs{end+1} = train_accs(:);
    end
    
    if isfield(hist, 'val_accuracy')
        val_accs = hist.val_accuracy(~isnan(hist.val_accuracy));
        all_val_accs{end+1} = val_accs(:);
    end
    
    if isfield(hist, 'val_iou')
        val_ious = hist.val_iou(~isnan(hist.val_iou));
        all_val_ious{end+1} = val_ious(:);
    end
    
    if isfield(hist, 'val_dice')
        val_dices = hist.val_dice(~isnan(hist.val_dice));
        all_val_dices{end+1} = val_dices(:);
    end
end

% Convert cell arrays to single arrays for statistics (concatenate all values)
all_train_losses_vec = [];
for i = 1:numel(all_train_losses)
    all_train_losses_vec = [all_train_losses_vec; all_train_losses{i}(:)];
end

all_val_losses_vec = [];
for i = 1:numel(all_val_losses)
    all_val_losses_vec = [all_val_losses_vec; all_val_losses{i}(:)];
end

all_train_accs_vec = [];
for i = 1:numel(all_train_accs)
    all_train_accs_vec = [all_train_accs_vec; all_train_accs{i}(:)];
end

all_val_accs_vec = [];
for i = 1:numel(all_val_accs)
    all_val_accs_vec = [all_val_accs_vec; all_val_accs{i}(:)];
end

all_val_ious_vec = [];
for i = 1:numel(all_val_ious)
    all_val_ious_vec = [all_val_ious_vec; all_val_ious{i}(:)];
end

all_val_dices_vec = [];
for i = 1:numel(all_val_dices)
    all_val_dices_vec = [all_val_dices_vec; all_val_dices{i}(:)];
end

%% Compute Statistics
fprintf('=== LOSS STATISTICS ===\n');
if ~isempty(all_train_losses_vec)
    fprintf('Training Loss:\n');
    fprintf('  Mean: %.2f\n', mean(all_train_losses_vec(:)));
    fprintf('  Std: %.2f\n', std(all_train_losses_vec(:)));
    fprintf('  Min: %.2f\n', min(all_train_losses_vec(:)));
    fprintf('  Max: %.2f\n', max(all_train_losses_vec(:)));
    fprintf('\n');
end

if ~isempty(all_val_losses_vec)
    fprintf('Validation Loss:\n');
    fprintf('  Mean: %.2f\n', mean(all_val_losses_vec(:)));
    fprintf('  Std: %.2f\n', std(all_val_losses_vec(:)));
    fprintf('  Min: %.2f\n', min(all_val_losses_vec(:)));
    fprintf('  Max: %.2f\n', max(all_val_losses_vec(:)));
    fprintf('\n');
end

%% Estimate Loss Component Scales
fprintf('=== ESTIMATED LOSS COMPONENT SCALES ===\n');
fprintf('(Based on typical values and lambda weights)\n\n');

if isfield(s, 'loss_config')
    % Estimate classification loss (typically 0.5-2.0 for crossentropy/focal)
    est_cls_loss = 1.0;
    fprintf('Classification Loss (estimated): ~%.2f\n', est_cls_loss);
    
    % Estimate GradCAM loss (typically 0.1-1.0)
    est_cam_loss = 0.5;
    weighted_cam = est_cam_loss * loss_config.lambda_cam;
    fprintf('GradCAM Loss: ~%.2f × λ_cam (%.2f) = ~%.2f\n', ...
        est_cam_loss, loss_config.lambda_cam, weighted_cam);
    
    % Estimate Tversky loss (typically 0.2-0.8)
    est_tversky_loss = 0.5;
    weighted_tversky = est_tversky_loss * loss_config.lambda_tversky;
    fprintf('Tversky Loss: ~%.2f × λ_tversky (%.2f) = ~%.2f\n', ...
        est_tversky_loss, loss_config.lambda_tversky, weighted_tversky);
    
    % Estimate anatomical loss (typically 0.1-0.5 for normalized CAM)
    est_anatomical_loss = 0.3;
    weighted_anatomical = est_anatomical_loss * loss_config.lambda_anatomical;
    fprintf('Anatomical Loss: ~%.2f × λ_anatomical (%.2f) = ~%.2f\n', ...
        est_anatomical_loss, loss_config.lambda_anatomical, weighted_anatomical);
    
    total_estimated = est_cls_loss + weighted_cam + weighted_tversky + weighted_anatomical;
    fprintf('\nTotal Estimated Loss: ~%.2f\n', total_estimated);
    fprintf('Anatomical Loss Contribution: %.1f%%\n', ...
        (weighted_anatomical / total_estimated) * 100);
    
    fprintf('\n⚠ ISSUE: Anatomical loss might be too small relative to other losses!\n');
    fprintf('  Recommendation: Increase λ_anatomical to 10.0-20.0\n');
end

%% Plot Training Curves
fprintf('\n=== CREATING VISUALIZATIONS ===\n');
outputDir = fullfile('results', 'loss_analysis');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

figure('Position', [100, 100, 1400, 800]);

% Plot 1: Training and Validation Loss
subplot(2, 3, 1);
hold on;
colors = lines(num_folds);
for f = 1:num_folds
    hist = training_histories.(fold_names{f});
    if isfield(hist, 'epoch_loss')
        epochs = hist.epoch;
        plot(epochs, hist.epoch_loss, '-', 'Color', colors(f,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Fold %d Train', f));
    end
    if isfield(hist, 'val_loss')
        val_epochs = hist.epoch(~isnan(hist.val_loss));
        val_loss = hist.val_loss(~isnan(hist.val_loss));
        if ~isempty(val_epochs)
            plot(val_epochs, val_loss, '--', 'Color', colors(f,:), 'LineWidth', 1.5, ...
                'DisplayName', sprintf('Fold %d Val', f));
        end
    end
end
xlabel('Epoch');
ylabel('Loss');
title('Training and Validation Loss');
legend('Location', 'best', 'NumColumns', 2);
grid on;

% Plot 2: Accuracy
subplot(2, 3, 2);
hold on;
for f = 1:num_folds
    hist = training_histories.(fold_names{f});
    if isfield(hist, 'train_acc')
        epochs = hist.epoch;
        plot(epochs, hist.train_acc, '-', 'Color', colors(f,:), 'LineWidth', 1.5);
    end
    if isfield(hist, 'val_accuracy')
        val_epochs = hist.epoch(~isnan(hist.val_accuracy));
        val_acc = hist.val_accuracy(~isnan(hist.val_accuracy));
        if ~isempty(val_epochs)
            plot(val_epochs, val_acc, '--', 'Color', colors(f,:), 'LineWidth', 1.5);
        end
    end
end
xlabel('Epoch');
ylabel('Accuracy');
title('Training and Validation Accuracy');
grid on;

% Plot 3: IoU
subplot(2, 3, 3);
hold on;
for f = 1:num_folds
    hist = training_histories.(fold_names{f});
    if isfield(hist, 'val_iou')
        val_epochs = hist.epoch(~isnan(hist.val_iou));
        val_iou = hist.val_iou(~isnan(hist.val_iou));
        if ~isempty(val_epochs)
            plot(val_epochs, val_iou, '-', 'Color', colors(f,:), 'LineWidth', 1.5, ...
                'DisplayName', sprintf('Fold %d', f));
        end
    end
end
xlabel('Epoch');
ylabel('IoU');
title('Validation IoU (Attention-Mask Alignment)');
legend('Location', 'best');
grid on;
ylim([0, 0.3]);

% Plot 4: Dice
subplot(2, 3, 4);
hold on;
for f = 1:num_folds
    hist = training_histories.(fold_names{f});
    if isfield(hist, 'val_dice')
        val_epochs = hist.epoch(~isnan(hist.val_dice));
        val_dice = hist.val_dice(~isnan(hist.val_dice));
        if ~isempty(val_epochs)
            plot(val_epochs, val_dice, '-', 'Color', colors(f,:), 'LineWidth', 1.5);
        end
    end
end
xlabel('Epoch');
ylabel('Dice');
title('Validation Dice (Attention-Mask Alignment)');
grid on;
ylim([0, 0.5]);

% Plot 5: Loss Component Comparison (estimated)
subplot(2, 3, 5);
if isfield(s, 'loss_config')
    component_names = {'Classification', 'GradCAM', 'Tversky', 'Anatomical'};
    est_values = [est_cls_loss, weighted_cam, weighted_tversky, weighted_anatomical];
    bar(est_values);
    set(gca, 'XTickLabel', component_names);
    ylabel('Estimated Weighted Loss');
    title('Estimated Loss Component Contributions');
    xtickangle(45);
    grid on;
end

% Plot 6: Recommendations
subplot(2, 3, 6);
axis off;
rec_text = {
    'RECOMMENDATIONS:';
    '';
    '1. Increase λ_anatomical:';
    '   Current: 5.0';
    '   Try: 10.0 - 20.0';
    '';
    '2. Check if anatomical loss is computed:';
    '   - Only on 16 random samples per batch';
    '   - May need more samples';
    '';
    '3. Consider loss rebalancing:';
    '   - Reduce λ_cam or λ_tversky';
    '   - Increase anatomical loss weight';
    '';
    '4. Verify masks are loaded correctly:';
    '   - Check precomputedMasks';
    '   - Ensure masks match images';
    '';
    '5. Try different anatomical loss formulation:';
    '   - Reward attention IN lungs';
    '   - Not just penalize outside';
};

text(0.1, 0.95, rec_text, 'FontSize', 10, ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'FontName', 'FixedWidth');

sgtitle('Loss Component Analysis', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'loss_component_analysis.png'));
fprintf('  Saved: loss_component_analysis.png\n');
close(gcf);

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Results saved to: %s\n', outputDir);

end

