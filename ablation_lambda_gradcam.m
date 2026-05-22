%% ========================================================================
% LAMBDA_GRADCAM ABLATION — does the CAM alignment loss actually work?
%
% Runs `rework.m` three times with lambda_gradcam set to 0.1, 0.5, and 2.0
% (current default is 0.01). Each run uses the SAME random seed, the SAME
% data split, and the SAME training schedule — only lambda_gradcam differs.
%
% Watches for TS cosine (teacher-student CAM cosine) to see whether the
% alignment loss is doing useful work as the weight increases.
%
% Outputs:
%   ablation_results/lambda_gradcam_0p10/    full rework.m artifacts
%   ablation_results/lambda_gradcam_0p50/    full rework.m artifacts
%   ablation_results/lambda_gradcam_2p00/    full rework.m artifacts
%   ablation_results/summary.csv             one row per run with key metrics
%
% Cost: ~3 × 5 minutes on RTX 3070 (faster than typical since we cut to
%       fewer epochs — see CONFIG.EPOCHS override below).
%
% Run from the same directory you normally run rework.m from.
%
% Usage:
%   >> ablation_lambda_gradcam
%% ========================================================================

clear; clc; close all;

% --- Ablation grid -------------------------------------------------------
LAMBDA_VALUES = [0.1, 0.5, 2.0];
ABLATION_ROOT = 'ablation_results';
% No EPOCHS override — use rework.m's default (30) so this matches the
% baseline conditions exactly. ONLY lambda_gradcam differs across runs.

if ~exist(ABLATION_ROOT, 'dir'), mkdir(ABLATION_ROOT); end

summaryPath = fullfile(ABLATION_ROOT, 'summary.csv');
fid = fopen(summaryPath, 'w');
fprintf(fid, 'lambda_gradcam,id_auc,test_auc,id_acc,test_acc,id_ece,test_ece,energy_ratio,pointing_game,ts_cosine,final_cam_align_loss,final_tversky_loss\n');
fclose(fid);

% --- Required dependencies ---------------------------------------------
required = {'rework.m', 'publication_kit/publication_figures.m', ...
            'publication_kit/compute_faithfulness_metrics.m'};
for k = 1:numel(required)
    if ~exist(required{k}, 'file')
        error('Required file not found: %s. Run from the project root.', required{k});
    end
end

% --- Run each ablation ----------------------------------------------------
for iAblation = 1:numel(LAMBDA_VALUES)
    lambdaVal = LAMBDA_VALUES(iAblation);
    lambdaTag = strrep(sprintf('%.2f', lambdaVal), '.', 'p');
    runDir = fullfile(ABLATION_ROOT, sprintf('lambda_gradcam_%s', lambdaTag));
    if ~exist(runDir, 'dir'), mkdir(runDir); end

    fprintf('\n');
    fprintf('========================================================================\n');
    fprintf('  ABLATION %d/%d: lambda_gradcam = %.2f\n', iAblation, numel(LAMBDA_VALUES), lambdaVal);
    fprintf('  Output: %s\n', runDir);
    fprintf('========================================================================\n');

    % Set environment variables that rework.m can pick up (see patch note)
    setenv('ABLATION_LAMBDA_GRADCAM', sprintf('%.6f', lambdaVal));
    setenv('ABLATION_MODEL_PATH',     fullfile(runDir, 'model.mat'));
    setenv('ABLATION_OUTPUT_DIR',     runDir);
    setenv('ABLATION_EXPERIMENT_TAG', sprintf('lambda%s', lambdaTag));

    % Run rework.m as a script. It will detect the env vars and adapt.
    try
        rework;
    catch ME
        warning('Run failed for lambda=%.2f: %s', lambdaVal, ME.message);
        clearvars -except LAMBDA_VALUES ABLATION_ROOT iAblation summaryPath;
        continue;
    end

    % Generate the publication figures for this run
    fprintf('\n--- Generating figures for lambda=%.2f ---\n', lambdaVal);
    try
        publication_figures(fullfile(runDir, 'model.mat'), fullfile(runDir, 'figures'));
    catch ME
        warning('Figure generation failed for lambda=%.2f: %s', lambdaVal, ME.message);
    end

    % Read the figure-kit's master metrics table and append summary row
    metricsPath = fullfile(runDir, 'figures', 'table_metrics.csv');
    if exist(metricsPath, 'file')
        T = readtable(metricsPath, 'TextType', 'string');
        idRow = T(contains(T.dataset, 'ID'), :);
        testRow = T(contains(T.dataset, 'TEST'), :);
        if ~isempty(idRow) && ~isempty(testRow)
            fid = fopen(summaryPath, 'a');
            fprintf(fid, '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,,\n', ...
                lambdaVal, ...
                getOr(idRow, 'auc', NaN), getOr(testRow, 'auc', NaN), ...
                getOr(idRow, 'accuracy', NaN), getOr(testRow, 'accuracy', NaN), ...
                getOr(idRow, 'ece', NaN), getOr(testRow, 'ece', NaN), ...
                getOr(testRow, 'energy_ratio_mean', NaN), ...
                getOr(testRow, 'pointing_game', NaN), ...
                getOr(testRow, 'ts_cosine', NaN));
            fclose(fid);
        end
    end

    % Clear workspace between runs to prevent state contamination
    clearvars -except LAMBDA_VALUES ABLATION_ROOT iAblation summaryPath;
end

% --- Clean up environment variables --------------------------------------
setenv('ABLATION_LAMBDA_GRADCAM', '');
setenv('ABLATION_MODEL_PATH', '');
setenv('ABLATION_OUTPUT_DIR', '');
setenv('ABLATION_EXPERIMENT_TAG', '');

% --- Print final summary -------------------------------------------------
fprintf('\n');
fprintf('========================================================================\n');
fprintf('  ABLATION COMPLETE\n');
fprintf('========================================================================\n');
fprintf('Summary: %s\n', summaryPath);
if exist(summaryPath, 'file')
    type(summaryPath);
end

fprintf('\nKey numbers to inspect:\n');
fprintf('  ts_cosine      Should rise with lambda if CAM loss works.\n');
fprintf('                 Currently 0.41 at lambda=0.01 (baseline).\n');
fprintf('                 If 0.7+ at lambda=0.5 or 2.0 without accuracy crash → CAM loss works.\n');
fprintf('                 If still ~0.4 at lambda=2.0 → MSE formulation is broken; switch to cosine.\n');
fprintf('  test_acc       Should stay near 0.85 baseline. If drops below 0.75, accuracy is collapsing.\n');
fprintf('  energy_ratio   Should rise as CAM gets more anatomical.\n');
fprintf('  pointing_game  Should rise alongside energy_ratio.\n');


%% =========================== HELPERS ====================================
function v = getOr(tableRow, columnName, defaultVal)
    if any(strcmp(tableRow.Properties.VariableNames, columnName))
        v = tableRow.(columnName)(1);
        if isnan(v), v = defaultVal; end
    else
        v = defaultVal;
    end
end
