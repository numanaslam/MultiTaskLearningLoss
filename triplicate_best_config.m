%% ========================================================================
% TRIPLICATE BEST CONFIG — three random seeds at lambda_gradcam = 2.0
%
% Runs `rework.m` three times with the best configuration identified in
% the ablation (cosine alignment, lambda_gradcam = 2.0), changing ONLY the
% random seed (42, 43, 44). This produces the mean and standard deviation
% across seeds that a reviewer will expect to see in the methods section.
%
% Requirements:
%   - rework.m must already have the cosine alignment patch applied
%     (line ~819 should use the cosine formulation, not MSE).
%   - rework.m must respect ABLATION_SEED env var (line ~24 patch).
%
% Output:
%   triplicate_results/seed_42/      full rework.m artifacts + figures
%   triplicate_results/seed_43/
%   triplicate_results/seed_44/
%   triplicate_results/summary.csv   per-seed metrics, one row each
%   triplicate_results/aggregate.csv mean ± std across the three seeds
%
% Cost: ~3 × 5 minutes on RTX 3070, ~15-20 minutes total.
%
% Usage:
%   >> triplicate_best_config
%% ========================================================================

clear; clc; close all;

% --- Configuration -------------------------------------------------------
SEEDS = [42, 43, 44];
LAMBDA_GRADCAM = 2.0;          % best config from ablation
TRIPLICATE_ROOT = 'triplicate_results';

if ~exist(TRIPLICATE_ROOT, 'dir'), mkdir(TRIPLICATE_ROOT); end

summaryPath = fullfile(TRIPLICATE_ROOT, 'summary.csv');
fid = fopen(summaryPath, 'w');
fprintf(fid, 'seed,id_auc,test_auc,id_acc,test_acc,id_ece,test_ece,energy_ratio,pointing_game,ts_cosine\n');
fclose(fid);

% --- Required dependencies ---------------------------------------------
required = {'rework.m', 'publication_kit/publication_figures.m', ...
            'publication_kit/compute_faithfulness_metrics.m'};
for k = 1:numel(required)
    if ~exist(required{k}, 'file')
        error('Required file not found: %s. Run from the project root.', required{k});
    end
end

% --- Run each seed -------------------------------------------------------
for iSeed = 1:numel(SEEDS)
    seedVal = SEEDS(iSeed);
    runDir = fullfile(TRIPLICATE_ROOT, sprintf('seed_%d', seedVal));
    if ~exist(runDir, 'dir'), mkdir(runDir); end

    fprintf('\n');
    fprintf('========================================================================\n');
    fprintf('  TRIPLICATE %d/%d: seed = %d (lambda_gradcam = %.2f)\n', ...
        iSeed, numel(SEEDS), seedVal, LAMBDA_GRADCAM);
    fprintf('  Output: %s\n', runDir);
    fprintf('========================================================================\n');

    setenv('ABLATION_LAMBDA_GRADCAM', sprintf('%.6f', LAMBDA_GRADCAM));
    setenv('ABLATION_SEED',           sprintf('%d', seedVal));
    setenv('ABLATION_MODEL_PATH',     fullfile(runDir, 'model.mat'));
    setenv('ABLATION_OUTPUT_DIR',     runDir);
    setenv('ABLATION_EXPERIMENT_TAG', sprintf('seed%d', seedVal));

    try
        rework;
    catch ME
        warning('Run failed for seed=%d: %s', seedVal, ME.message);
        clearvars -except SEEDS LAMBDA_GRADCAM TRIPLICATE_ROOT iSeed summaryPath;
        continue;
    end

    fprintf('\n--- Generating figures for seed=%d ---\n', seedVal);
    try
        publication_figures(fullfile(runDir, 'model.mat'), fullfile(runDir, 'figures'));
    catch ME
        warning('Figure generation failed for seed=%d: %s', seedVal, ME.message);
    end

    metricsPath = fullfile(runDir, 'figures', 'table_metrics.csv');
    if exist(metricsPath, 'file')
        T = readtable(metricsPath, 'TextType', 'string');
        idRow = T(contains(T.dataset, 'ID'), :);
        testRow = T(contains(T.dataset, 'TEST'), :);
        if ~isempty(idRow) && ~isempty(testRow)
            fid = fopen(summaryPath, 'a');
            fprintf(fid, '%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', ...
                seedVal, ...
                getOr(idRow, 'auc', NaN), getOr(testRow, 'auc', NaN), ...
                getOr(idRow, 'accuracy', NaN), getOr(testRow, 'accuracy', NaN), ...
                getOr(idRow, 'ece', NaN), getOr(testRow, 'ece', NaN), ...
                getOr(testRow, 'energy_ratio_mean', NaN), ...
                getOr(testRow, 'pointing_game', NaN), ...
                getOr(testRow, 'ts_cosine', NaN));
            fclose(fid);
        end
    end

    clearvars -except SEEDS LAMBDA_GRADCAM TRIPLICATE_ROOT iSeed summaryPath;
end

% --- Clean up environment ------------------------------------------------
setenv('ABLATION_LAMBDA_GRADCAM', '');
setenv('ABLATION_SEED', '');
setenv('ABLATION_MODEL_PATH', '');
setenv('ABLATION_OUTPUT_DIR', '');
setenv('ABLATION_EXPERIMENT_TAG', '');

% --- Aggregate statistics ------------------------------------------------
fprintf('\n');
fprintf('========================================================================\n');
fprintf('  TRIPLICATE COMPLETE\n');
fprintf('========================================================================\n');

if exist(summaryPath, 'file')
    T = readtable(summaryPath);

    metricCols = {'id_auc', 'test_auc', 'id_acc', 'test_acc', ...
                  'id_ece', 'test_ece', 'energy_ratio', 'pointing_game', 'ts_cosine'};
    aggregatePath = fullfile(TRIPLICATE_ROOT, 'aggregate.csv');
    fa = fopen(aggregatePath, 'w');
    fprintf(fa, 'metric,mean,std,min,max,n\n');
    fprintf('\n--- Aggregate over %d seeds ---\n', height(T));
    fprintf('%-15s  %-8s  %-8s  %-8s  %-8s\n', 'metric', 'mean', 'std', 'min', 'max');
    for k = 1:numel(metricCols)
        col = metricCols{k};
        if ~ismember(col, T.Properties.VariableNames), continue; end
        vals = T.(col);
        vals = vals(isfinite(vals));
        if isempty(vals), continue; end
        m = mean(vals); s = std(vals);
        fprintf('%-15s  %.4f    %.4f    %.4f    %.4f\n', col, m, s, min(vals), max(vals));
        fprintf(fa, '%s,%.6f,%.6f,%.6f,%.6f,%d\n', col, m, s, min(vals), max(vals), numel(vals));
    end
    fclose(fa);

    fprintf('\nPer-seed summary: %s\n', summaryPath);
    fprintf('Aggregate stats:  %s\n', aggregatePath);
    fprintf('\nReport in your paper as:\n');
    if any(strcmp(T.Properties.VariableNames, 'test_auc'))
        v = T.test_auc(isfinite(T.test_auc));
        if ~isempty(v)
            fprintf('  Test AUC: %.3f ± %.3f (n=%d)\n', mean(v), std(v), numel(v));
        end
    end
    if any(strcmp(T.Properties.VariableNames, 'test_acc'))
        v = T.test_acc(isfinite(T.test_acc));
        if ~isempty(v)
            fprintf('  Test accuracy: %.3f ± %.3f (n=%d)\n', mean(v), std(v), numel(v));
        end
    end
    if any(strcmp(T.Properties.VariableNames, 'ts_cosine'))
        v = T.ts_cosine(isfinite(T.ts_cosine));
        if ~isempty(v)
            fprintf('  Teacher-student CAM cosine: %.3f ± %.3f (n=%d)\n', mean(v), std(v), numel(v));
        end
    end
end


%% =========================== HELPERS ====================================
function v = getOr(tableRow, columnName, defaultVal)
    if any(strcmp(tableRow.Properties.VariableNames, columnName))
        v = tableRow.(columnName)(1);
        if isnan(v), v = defaultVal; end
    else
        v = defaultVal;
    end
end
