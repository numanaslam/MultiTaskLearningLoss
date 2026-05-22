function publication_figures(modelPath, outputDir, extraOodPaths)
%PUBLICATION_FIGURES  Generate publication-ready figures from a trained model.
%
%   publication_figures(modelPath)
%       Loads modelPath and writes figures next to it under ./figures/
%   publication_figures(modelPath, outputDir)
%       Writes figures to outputDir (created if needed).
%   publication_figures(modelPath, outputDir, extraOodPaths)
%       extraOodPaths is a cell array of directory paths for cross-site OOD
%       evaluation. Each generates its own column in the comparison figures.
%
%   Example:
%       publication_figures('models/trained_mixed_vgg16.mat', 'paper_figures', ...
%           {'C:/data/montgomery', 'C:/data/tbx11k'});
%
%   Produces (all at 300 DPI, vector PDF + raster PNG):
%       fig01_roc_curves.{pdf,png}            ROC, all evaluation sets
%       fig02_pr_curves.{pdf,png}             Precision-recall, all eval sets
%       fig03_reliability.{pdf,png}           Calibration diagrams + ECE
%       fig04_loss_curves.{pdf,png}           Training curves per fold
%       fig05_alignment_progress.{pdf,png}    Teacher-student cosine over epochs
%       fig06_faithfulness_metrics.{pdf,png}  Energy ratio + pointing-game
%       fig07_gradcam_comparison.{pdf,png}    Teacher vs student CAM grid
%       fig08_failure_modes.{pdf,png}         High-confidence wrong predictions
%       fig09_threshold_analysis.{pdf,png}    Sens/spec/F1 vs threshold
%       fig10_confusion_matrices.{pdf,png}    Grid of confusion matrices
%       table_metrics.csv                     Master metrics table for paper

    if nargin < 2 || isempty(outputDir)
        outputDir = fullfile(fileparts(modelPath), 'figures');
    end
    if nargin < 3
        extraOodPaths = {};
    end

    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    fprintf('=== publication_figures ===\n');
    fprintf('  Model: %s\n', modelPath);
    fprintf('  Output: %s\n', outputDir);

    % -- Load model and artifacts ------------------------------------------
    S = load(modelPath);
    dlnet         = S.dlnet;
    teacherNet    = S.teacherNet;
    uniqueClasses = S.uniqueClasses;
    CONFIG        = S.CONFIG;
    idPredictions = S.idPredictions;
    oodPredictions = S.oodPredictions;   % held-out test
    posIdx = inferPositiveIdx(uniqueClasses);

    % Optional: epoch history if you applied the rework.m logging patch
    if isfield(S, 'epochHistory')
        epochHistory = S.epochHistory;
        haveHistory = true;
    else
        epochHistory = [];
        haveHistory = false;
        warning('publication_figures:NoHistory', ...
            ['No epochHistory found. Loss-curve and alignment-progress ' ...
             'figures will be skipped. Apply the rework.m logging patch ' ...
             'to enable them.']);
    end

    % -- Style setup --------------------------------------------------------
    setPaperStyle();

    % ----------------------------------------------------------------------
    % FIG 1: ROC curves
    % ----------------------------------------------------------------------
    fprintf('  [1/10] ROC curves...\n');
    fig = figure('Visible', 'off', 'Position', [100 100 600 540]);
    hold on;
    rocSets = collectEvalSets(idPredictions, oodPredictions, posIdx);

    for k = 1:numel(rocSets)
        [fpr, tpr, ~, auc] = perfcurve( ...
            rocSets(k).trueIdx == posIdx, rocSets(k).posProb, true);
        plot(fpr, tpr, '-', 'LineWidth', 1.8, ...
            'DisplayName', sprintf('%s (AUC = %.3f)', rocSets(k).name, auc));
        rocSets(k).auc = auc;
    end
    plot([0 1], [0 1], 'k:', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title('ROC Curves');
    legend('Location', 'southeast', 'FontSize', 9);
    axis square; grid on; xlim([0 1]); ylim([0 1]);
    hold off;
    exportPub(fig, fullfile(outputDir, 'fig01_roc_curves'));
    close(fig);

    % ----------------------------------------------------------------------
    % FIG 2: Precision-recall
    % ----------------------------------------------------------------------
    fprintf('  [2/10] PR curves...\n');
    fig = figure('Visible', 'off', 'Position', [100 100 600 540]);
    hold on;
    prSets = rocSets;   % same data, different metric
    for k = 1:numel(prSets)
        [recall, precision, ~, ap] = perfcurve( ...
            prSets(k).trueIdx == posIdx, prSets(k).posProb, true, ...
            'xCrit', 'reca', 'yCrit', 'prec');
        % perfcurve returns NaN at recall=0; drop those rows
        keep = isfinite(recall) & isfinite(precision);
        plot(recall(keep), precision(keep), '-', 'LineWidth', 1.8, ...
            'DisplayName', sprintf('%s (AP = %.3f)', prSets(k).name, ap));
        prSets(k).ap = ap;
    end
    posPrev = mean(rocSets(1).trueIdx == posIdx);
    yline(posPrev, 'k:', sprintf('chance (prev. = %.2f)', posPrev), ...
        'LabelHorizontalAlignment', 'left', 'FontSize', 8);
    xlabel('Recall'); ylabel('Precision');
    title('Precision-Recall Curves');
    legend('Location', 'southwest', 'FontSize', 9);
    axis square; grid on; xlim([0 1]); ylim([0 1]);
    hold off;
    exportPub(fig, fullfile(outputDir, 'fig02_pr_curves'));
    close(fig);

    % ----------------------------------------------------------------------
    % FIG 3: Reliability diagram (calibration)
    % ----------------------------------------------------------------------
    fprintf('  [3/10] Reliability diagram...\n');
    fig = figure('Visible', 'off', 'Position', [100 100 1100 460]);
    nBins = 10;
    for k = 1:numel(rocSets)
        subplot(1, numel(rocSets), k);
        [binCenters, binAcc, binConf, binCounts, ece] = ...
            reliabilityCurve(rocSets(k).posProb, ...
                             rocSets(k).trueIdx == posIdx, nBins);
        bar(binCenters, binAcc, 1.0, 'FaceColor', [0.3 0.6 0.9], ...
            'EdgeColor', 'none', 'FaceAlpha', 0.7);
        hold on;
        plot(binCenters, binConf, 'r-o', 'LineWidth', 1.4, 'MarkerSize', 5);
        plot([0 1], [0 1], 'k--', 'LineWidth', 0.8);
        % bin-count annotation
        for b = 1:numel(binCenters)
            if binCounts(b) > 0
                text(binCenters(b), 0.02, sprintf('%d', binCounts(b)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 7);
            end
        end
        xlabel('Predicted probability (positive class)');
        ylabel('Fraction positive');
        title(sprintf('%s | ECE = %.3f', rocSets(k).name, ece));
        axis square; grid on; xlim([0 1]); ylim([0 1]);
        if k == 1
            legend({'Empirical', 'Mean confidence', 'Perfect'}, ...
                'Location', 'northwest', 'FontSize', 8);
        end
        rocSets(k).ece = ece;
        hold off;
    end
    sgtitle('Calibration Reliability Diagrams', 'FontWeight', 'bold');
    exportPub(fig, fullfile(outputDir, 'fig03_reliability'));
    close(fig);

    % ----------------------------------------------------------------------
    % FIG 4 + FIG 5: Loss curves and alignment progress (if available)
    % ----------------------------------------------------------------------
    if haveHistory
        fprintf('  [4/10] Loss curves...\n');
        fig = figure('Visible', 'off', 'Position', [100 100 1100 460]);
        nFolds = numel(epochHistory);

        for f = 1:nFolds
            subplot(1, nFolds, f);
            h = epochHistory(f);
            epochs = 1:numel(h.ce);
            yyaxis left;
            plot(epochs, h.ce, '-o', 'LineWidth', 1.4, 'MarkerSize', 3.5, ...
                'DisplayName', 'CE'); hold on;
            plot(epochs, h.tversky, '-s', 'LineWidth', 1.4, 'MarkerSize', 3.5, ...
                'DisplayName', 'Tversky');
            plot(epochs, h.gradcam, '-^', 'LineWidth', 1.4, 'MarkerSize', 3.5, ...
                'DisplayName', 'CAM Align');
            ylabel('Loss component');
            yyaxis right;
            plot(epochs, h.valAcc * 100, 'k:', 'LineWidth', 1.5, ...
                'DisplayName', 'Val Acc'); hold off;
            ylabel('Validation Accuracy (%)'); ylim([50 100]);
            xlabel('Epoch'); title(sprintf('Fold %d', f));
            grid on;
            if f == 1, legend('Location', 'northeast', 'FontSize', 8); end
        end
        sgtitle('Training Dynamics by Fold', 'FontWeight', 'bold');
        exportPub(fig, fullfile(outputDir, 'fig04_loss_curves'));
        close(fig);

        fprintf('  [5/10] Alignment progress...\n');
        if isfield(epochHistory(1), 'tsCosine') && ~isempty(epochHistory(1).tsCosine)
            fig = figure('Visible', 'off', 'Position', [100 100 700 480]);
            hold on;
            for f = 1:nFolds
                h = epochHistory(f);
                plot(1:numel(h.tsCosine), h.tsCosine, '-o', ...
                    'LineWidth', 1.6, 'MarkerSize', 4, ...
                    'DisplayName', sprintf('Fold %d', f));
            end
            xlabel('Epoch'); ylabel('Teacher-Student CAM Cosine');
            title('Attention Alignment Over Training');
            legend('Location', 'southeast', 'FontSize', 9);
            grid on; ylim([0 1]);
            % Annotate when aux losses turn on
            xline(CONFIG.AUX_START_EPOCH, 'k--', 'Aux ramp', ...
                'LabelOrientation', 'horizontal', 'FontSize', 8);
            hold off;
            exportPub(fig, fullfile(outputDir, 'fig05_alignment_progress'));
            close(fig);
        else
            fprintf('    (skipped — tsCosine not in epochHistory)\n');
        end
    else
        fprintf('  [4-5/10] Skipped (no epochHistory in mat file)\n');
    end

    % ----------------------------------------------------------------------
    % FIG 6: Faithfulness metrics (energy ratio + pointing game)
    % ----------------------------------------------------------------------
    fprintf('  [6/10] Faithfulness metrics...\n');
    targetLayer = 'relu5_3';
    studentFCName = findLastFC(dlnet);

    fSets = struct([]);
    nextSlot = 1;

    % Fold-Val set: only available if the training save kept these vars.
    if isfield(S, 'globalBestValFiles') && ~isempty(S.globalBestValFiles)
        fSets(nextSlot).name   = 'Fold-Val';
        fSets(nextSlot).files  = S.globalBestValFiles;
        fSets(nextSlot).labels = S.globalBestValLabels;
        if isfield(S, 'globalBestValMasks')
            fSets(nextSlot).masks = S.globalBestValMasks;
        else
            fSets(nextSlot).masks = resolveMasksFromConfig( ...
                S.globalBestValFiles, S.globalBestValLabels, ...
                uniqueClasses, CONFIG);
        end
        nextSlot = nextSlot + 1;
    else
        fprintf('    (Fold-Val skipped — globalBestValFiles not in .mat)\n');
    end

    % Held-out TEST set
    if isfield(S, 'heldoutFiles') && ~isempty(S.heldoutFiles)
        fSets(nextSlot).name   = 'Held-out TEST';
        fSets(nextSlot).files  = S.heldoutFiles;
        fSets(nextSlot).labels = S.heldoutLabels;
        fSets(nextSlot).masks  = resolveMasksFromConfig( ...
            S.heldoutFiles, S.heldoutLabels, uniqueClasses, CONFIG);
        nextSlot = nextSlot + 1;
    end

    if isempty(fSets)
        warning('publication_figures:NoFaithfulnessSets', ...
            'No evaluation sets available for faithfulness metrics. Skipping fig 6 and fig 7.');
    end

    for k = 1:numel(fSets)
        fprintf('    %s ...\n', fSets(k).name);
        fSets(k).metrics = compute_faithfulness_metrics( ...
            dlnet, teacherNet, fSets(k).files, fSets(k).labels, ...
            fSets(k).masks, targetLayer, studentFCName, uniqueClasses);
    end

if ~isempty(fSets)
    fig = figure('Visible', 'off', 'Position', [100 100 1100 460]);
    subplot(1, 2, 1);
    erData = cell(numel(fSets), 1);
    erLabels = cell(numel(fSets), 1);
    for k = 1:numel(fSets)
        erData{k} = fSets(k).metrics.energy_ratio;
        erLabels{k} = sprintf('%s\n(mean = %.2f)', fSets(k).name, ...
            mean(fSets(k).metrics.energy_ratio));
    end
    boxchart(repelem((1:numel(fSets))', cellfun(@numel, erData)), ...
             vertcat(erData{:}));
    set(gca, 'XTick', 1:numel(fSets), 'XTickLabel', erLabels, 'FontSize', 9);
    ylabel('Energy ratio (CAM mass inside lung)');
    title('Anatomical Faithfulness');
    grid on; ylim([0 1]);

    subplot(1, 2, 2);
    pgVals = arrayfun(@(s) mean(s.metrics.pointing_game), fSets);
    tsVals = arrayfun(@(s) mean(s.metrics.ts_cosine), fSets);
    bar(categorical({fSets.name}), [pgVals(:), tsVals(:)], 'grouped');
    ylabel('Score'); ylim([0 1]);
    legend({'Pointing-game accuracy', 'Teacher-student cosine'}, ...
        'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9);
    title('CAM Quality Metrics');
    grid on;
    sgtitle('Explanation Faithfulness', 'FontWeight', 'bold');
    exportPub(fig, fullfile(outputDir, 'fig06_faithfulness_metrics'));
    close(fig);
end

    % ----------------------------------------------------------------------
    % FIG 7: Grad-CAM comparison grid (teacher | student | mask | overlay)
    % ----------------------------------------------------------------------
    fprintf('  [7/10] Grad-CAM comparison grid...\n');
if ~isempty(fSets)
    [tpIdx, tnIdx, fpIdx, fnIdx] = pickConfusionExamples( ...
        fSets(end).files, fSets(end).labels, ...
        oodPredictions.predIdx, posIdx, 2);
    pickedIdx = [tpIdx(:); tnIdx(:); fpIdx(:); fnIdx(:)];
    pickedLabel = [repmat({'TP'}, numel(tpIdx), 1); ...
                   repmat({'TN'}, numel(tnIdx), 1); ...
                   repmat({'FP'}, numel(fpIdx), 1); ...
                   repmat({'FN'}, numel(fnIdx), 1)];
    nRows = numel(pickedIdx);
    if nRows > 0
        fig = figure('Visible', 'off', 'Position', [100 100 1100 220 * nRows]);
        for r = 1:nRows
            idx = pickedIdx(r);
            img = imread(fSets(end).files{idx});
            if size(img, 3) == 1, img = cat(3, img, img, img); end
            img = imresize(img, [224 224]);

            mask = readMaskSafe(fSets(end).masks{idx});

            studentCAM = singleImageCAM(dlnet, img, posIdx, targetLayer, studentFCName);
            teacherCAM = singleImageCAM(teacherNet, img, posIdx, targetLayer, ...
                findLastFC(teacherNet));

            subplot(nRows, 4, (r-1)*4 + 1);
            imshow(img); title(sprintf('%s | Input', pickedLabel{r}), 'FontSize', 9);

            subplot(nRows, 4, (r-1)*4 + 2);
            imshow(overlayCAM(img, teacherCAM));
            title('Teacher CAM', 'FontSize', 9);

            subplot(nRows, 4, (r-1)*4 + 3);
            imshow(overlayCAM(img, studentCAM));
            title('Student CAM', 'FontSize', 9);

            subplot(nRows, 4, (r-1)*4 + 4);
            if ~isempty(mask)
                imshow(overlayMask(img, mask));
                title('Lung Mask', 'FontSize', 9);
            else
                imshow(img); title('No mask', 'FontSize', 9);
            end
        end
        sgtitle('Grad-CAM Comparison: Teacher vs Student (with lung mask)', ...
            'FontWeight', 'bold');
        exportPub(fig, fullfile(outputDir, 'fig07_gradcam_comparison'));
        close(fig);
    end
end

    % ----------------------------------------------------------------------
    % FIG 8: Failure-mode gallery (high-confidence wrong predictions)
    % ----------------------------------------------------------------------
    fprintf('  [8/10] Failure-mode gallery...\n');
    % Pick the file source: prefer the last fSets entry (held-out test),
    % fall back to oodPredictions.files which evaluateFileSet stored.
    if ~isempty(fSets)
        fig8Files = fSets(end).files;
    elseif isfield(oodPredictions, 'files') && ~isempty(oodPredictions.files)
        fig8Files = oodPredictions.files;
    else
        fig8Files = {};
    end

    wrongMask = oodPredictions.predIdx ~= oodPredictions.trueIdx & ...
                isfinite(oodPredictions.trueIdx);
    confs = oodPredictions.confidence;
    confs(~wrongMask) = -inf;
    [~, sortByConf] = sort(confs, 'descend');
    nFail = min(8, sum(wrongMask));
    if nFail > 0 && ~isempty(fig8Files)
        failIdx = sortByConf(1:nFail);
        fig = figure('Visible', 'off', 'Position', [100 100 1200 320 * ceil(nFail/4)]);
        for r = 1:nFail
            idx = failIdx(r);
            img = imread(fig8Files{idx});
            if size(img, 3) == 1, img = cat(3, img, img, img); end
            img = imresize(img, [224 224]);
            studentCAM = singleImageCAM(dlnet, img, ...
                oodPredictions.predIdx(idx), targetLayer, studentFCName);

            subplot(ceil(nFail/4), 4, r);
            imshow(overlayCAM(img, studentCAM));
            trueLbl = uniqueClasses{oodPredictions.trueIdx(idx)};
            predLbl = uniqueClasses{oodPredictions.predIdx(idx)};
            title(sprintf('GT:%s | Pred:%s (conf %.2f)', ...
                trueLbl, predLbl, oodPredictions.confidence(idx)), ...
                'FontSize', 9, 'Interpreter', 'none');
        end
        sgtitle('High-Confidence Failures (worst cases first)', ...
            'FontWeight', 'bold');
        exportPub(fig, fullfile(outputDir, 'fig08_failure_modes'));
        close(fig);
    end

    % ----------------------------------------------------------------------
    % FIG 9: Threshold sensitivity analysis
    % ----------------------------------------------------------------------
    fprintf('  [9/10] Threshold analysis...\n');
    fig = figure('Visible', 'off', 'Position', [100 100 700 480]);
    thresholds = 0.05:0.01:0.95;
    sensV = zeros(size(thresholds));
    specV = zeros(size(thresholds));
    f1V = zeros(size(thresholds));
    posIdx_local = posIdx;
    negIdx_local = 3 - posIdx;
    truth = oodPredictions.trueIdx == posIdx_local;
    prob  = oodPredictions.probs(:, posIdx_local);
    for ti = 1:numel(thresholds)
        pred = prob >= thresholds(ti);
        tp = sum(truth & pred);
        fn = sum(truth & ~pred);
        tn = sum(~truth & ~pred);
        fp = sum(~truth & pred);
        sensV(ti) = tp / max(tp + fn, 1);
        specV(ti) = tn / max(tn + fp, 1);
        f1V(ti)   = 2 * tp / max(2 * tp + fp + fn, 1);
    end
    hold on;
    plot(thresholds, sensV, '-', 'LineWidth', 1.6, 'DisplayName', 'Sensitivity');
    plot(thresholds, specV, '-', 'LineWidth', 1.6, 'DisplayName', 'Specificity');
    plot(thresholds, f1V,   '-', 'LineWidth', 1.6, 'DisplayName', 'F1');
    if isfield(S, 'idTunedThreshold') && isfinite(S.idTunedThreshold)
        xline(S.idTunedThreshold, 'k--', sprintf('ID-tuned (%.2f)', S.idTunedThreshold), ...
            'LabelOrientation', 'horizontal', 'FontSize', 8);
    end
    xline(0.5, ':k', 'argmax (0.50)', ...
        'LabelOrientation', 'horizontal', 'FontSize', 8);
    xlabel('Decision threshold');
    ylabel('Metric value');
    title('Held-out TEST: Metric vs Threshold');
    legend('Location', 'east', 'FontSize', 9);
    grid on; ylim([0 1]); hold off;
    exportPub(fig, fullfile(outputDir, 'fig09_threshold_analysis'));
    close(fig);

    % ----------------------------------------------------------------------
    % FIG 10: Confusion matrix grid
    % ----------------------------------------------------------------------
    fprintf(' [10/10] Confusion matrices...\n');
    fig = figure('Visible', 'off', 'Position', [100 100 1100 460]);
    cmSets = collectEvalSets(idPredictions, oodPredictions, posIdx);
    for k = 1:numel(cmSets)
        subplot(1, numel(cmSets), k);
        cm = confMatrix(cmSets(k).trueIdx, cmSets(k).predIdx, numel(uniqueClasses));
        imagesc(cm); axis image; colormap(gca, parula);
        set(gca, 'XTick', 1:numel(uniqueClasses), 'XTickLabel', uniqueClasses, ...
            'YTick', 1:numel(uniqueClasses), 'YTickLabel', uniqueClasses);
        xlabel('Predicted'); ylabel('True');
        for r = 1:size(cm, 1)
            for c = 1:size(cm, 2)
                txtColor = 'w'; if cm(r, c) < max(cm(:))/2, txtColor = 'k'; end
                text(c, r, sprintf('%d', cm(r, c)), 'HorizontalAlignment', 'center', ...
                    'Color', txtColor, 'FontWeight', 'bold');
            end
        end
        title(sprintf('%s (Acc = %.3f)', cmSets(k).name, ...
            sum(diag(cm))/sum(cm(:))));
    end
    sgtitle('Confusion Matrices', 'FontWeight', 'bold');
    exportPub(fig, fullfile(outputDir, 'fig10_confusion_matrices'));
    close(fig);

    % ----------------------------------------------------------------------
    % MASTER METRICS TABLE
    % ----------------------------------------------------------------------
    fprintf('  Writing master metrics table...\n');
    writeMasterMetricsTable(rocSets, fSets, ...
        fullfile(outputDir, 'table_metrics.csv'));

    fprintf('Done. %d figures + 1 table written to %s\n', 10, outputDir);
end


%% =========================== HELPERS ====================================

function setPaperStyle()
    set(groot, 'defaultAxesFontName', 'Helvetica');
    set(groot, 'defaultAxesFontSize', 10);
    set(groot, 'defaultAxesLineWidth', 0.8);
    set(groot, 'defaultLineLineWidth', 1.5);
    set(groot, 'defaultFigureColor', 'w');
end

function exportPub(fig, basePath)
    % Export both PDF (vector) and PNG (raster at 300 DPI) for paper use.
    set(fig, 'PaperPositionMode', 'auto');
    try
        exportgraphics(fig, [basePath '.pdf'], 'ContentType', 'vector');
        exportgraphics(fig, [basePath '.png'], 'Resolution', 300);
    catch
        saveas(fig, [basePath '.png']);
        try, saveas(fig, [basePath '.pdf']); catch, end
    end
end

function sets = collectEvalSets(idPred, oodPred, posIdx)
    sets = struct([]);
    sets(1).name = 'Fold-Val (ID)';
    sets(1).trueIdx = idPred.trueIdx;
    sets(1).predIdx = idPred.predIdx;
    sets(1).posProb = idPred.probs(:, posIdx);
    sets(2).name = 'Held-out TEST';
    sets(2).trueIdx = oodPred.trueIdx;
    sets(2).predIdx = oodPred.predIdx;
    sets(2).posProb = oodPred.probs(:, posIdx);
end

function posIdx = inferPositiveIdx(classNames)
    posIdx = find(strcmpi(classNames, 'ptb'), 1, 'first');
    if isempty(posIdx), posIdx = min(2, numel(classNames)); end
end

function [centers, accs, confs, counts, ece] = reliabilityCurve(probs, truth, nBins)
    edges = linspace(0, 1, nBins + 1);
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    accs = zeros(nBins, 1);
    confs = zeros(nBins, 1);
    counts = zeros(nBins, 1);
    for b = 1:nBins
        if b < nBins
            inBin = probs >= edges(b) & probs < edges(b+1);
        else
            inBin = probs >= edges(b) & probs <= edges(b+1);
        end
        counts(b) = sum(inBin);
        if counts(b) > 0
            accs(b)  = mean(truth(inBin));
            confs(b) = mean(probs(inBin));
        end
    end
    weights = counts / max(sum(counts), 1);
    ece = sum(weights .* abs(accs - confs));
end

function cm = confMatrix(trueIdx, predIdx, k)
    valid = isfinite(trueIdx) & isfinite(predIdx);
    cm = accumarray([trueIdx(valid), predIdx(valid)], 1, [k k]);
end

function fcName = findLastFC(net)
    layers = net.Layers;
    fcIdx = find(arrayfun(@(L) isa(L, 'nnet.cnn.layer.FullyConnectedLayer'), layers), 1, 'last');
    fcName = layers(fcIdx).Name;
end

function cam = singleImageCAM(net, img, classIdx, targetLayer, fcName)
    if ~isa(net, 'dlnetwork'), net = dlnetwork(layerGraph(net)); end
    XData = reshape(single(img), [size(img,1), size(img,2), 3, 1]);
    if hasGPU(net), XData = gpuArray(XData); end
    X = dlarray(XData, 'SSCB');
    cam = dlfeval(@camForward, net, X, classIdx, targetLayer, fcName);
    cam = squeeze(gather(extractdata(cam)));
end

function cam = camForward(net, X, classIdx, targetLayer, fcName)
    [rawLogits, act] = forward(net, X, 'Outputs', {fcName, targetLayer});
    logitsVec = reshape(stripdims(rawLogits), [], 1);
    logits = dlarray(logitsVec, 'CB');
    tgt = zeros(size(logitsVec), 'like', logitsVec); tgt(classIdx) = 1;
    obj = sum(logits .* dlarray(tgt, 'CB'), 1);
    g = dlgradient(obj, act);
    w = mean(g, [1 2]);
    cam = relu(sum(act .* w, 3));
    cam = dlresize(cam, 'OutputSize', [size(X, 1) size(X, 2)]);
    cam = cam ./ (max(cam, [], [1 2]) + 1e-6);
end

function tf = hasGPU(net)
    tf = false;
    if isempty(net.Learnables) || isempty(net.Learnables.Value), return; end
    v = net.Learnables.Value{1};
    tf = isa(v, 'gpuArray') || (isa(v, 'dlarray') && isa(extractdata(v), 'gpuArray'));
end

function overlay = overlayCAM(img, cam)
    img = im2single(img);
    cam = min(max(cam, 0), 1);
    hm = ind2rgb(uint8(255 * cam), jet(256));
    overlay = 0.55 * img + 0.45 * hm;
    overlay = min(max(overlay, 0), 1);
end

function overlay = overlayMask(img, mask)
    img = im2single(img);
    contour = mask & ~imerode(mask, strel('disk', 2));
    overlay = img;
    overlay(:,:,1) = overlay(:,:,1) + 0.6 * single(contour);
    overlay = min(max(overlay, 0), 1);
end

function mask = readMaskSafe(maskPath)
    mask = [];
    if isempty(maskPath) || ~exist(maskPath, 'file'), return; end
    m = imread(maskPath);
    if size(m, 3) > 1, m = rgb2gray(m); end
    m = imresize(m, [224 224], 'nearest');
    mask = single(m) / single(max(m(:)) + eps) > 0.5;
end

function [tp, tn, fp, fn] = pickConfusionExamples(files, labels, predIdx, posIdx, nEach)
    negIdx = 3 - posIdx;
    tp = find(labels == posIdx & predIdx == posIdx);
    tn = find(labels == negIdx & predIdx == negIdx);
    fp = find(labels == negIdx & predIdx == posIdx);
    fn = find(labels == posIdx & predIdx == negIdx);
    tp = tp(1:min(nEach, numel(tp)));
    tn = tn(1:min(nEach, numel(tn)));
    fp = fp(1:min(nEach, numel(fp)));
    fn = fn(1:min(nEach, numel(fn)));
end

function masks = resolveMasksFromConfig(files, labels, classNames, CONFIG)
    masks = cell(numel(files), 1);
    for i = 1:numel(files)
        [~, name, ~] = fileparts(files{i});
        cls = classNames{labels(i)};
        candidate = fullfile(CONFIG.MASK_ROOT, cls, [name '_mask.png']);
        if exist(candidate, 'file')
            masks{i} = candidate;
        else
            hits = dir(fullfile(CONFIG.MASK_ROOT, '**', [name '_mask.png']));
            if ~isempty(hits)
                masks{i} = fullfile(hits(1).folder, hits(1).name);
            else
                masks{i} = '';
            end
        end
    end
end

function writeMasterMetricsTable(rocSets, fSets, outPath)
    fid = fopen(outPath, 'w');
    if fid < 0, return; end
    fprintf(fid, 'dataset,accuracy,auc,ap,ece,energy_ratio_mean,pointing_game,ts_cosine\n');
    for k = 1:numel(rocSets)
        acc = mean(rocSets(k).trueIdx == rocSets(k).predIdx);
        auc = getfieldOrNaN(rocSets(k), 'auc');
        ap  = getfieldOrNaN(rocSets(k), 'ap');
        ece = getfieldOrNaN(rocSets(k), 'ece');
        er = NaN; pg = NaN; tsc = NaN;
        % Match faithfulness set by name where possible
        for j = 1:numel(fSets)
            if contains(rocSets(k).name, fSets(j).name)
                er  = mean(fSets(j).metrics.energy_ratio);
                pg  = mean(fSets(j).metrics.pointing_game);
                tsc = mean(fSets(j).metrics.ts_cosine);
                break;
            end
        end
        fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', ...
            rocSets(k).name, acc, auc, ap, ece, er, pg, tsc);
    end
    fclose(fid);
end

function v = getfieldOrNaN(s, f)
    if isfield(s, f), v = s.(f); else, v = NaN; end
end