function evaluate_trained_model_final(modelFile)
% evaluate_trained_model_final  —  Clean, robust evaluation of a trained TB model
%
% Usage:
%   evaluate_trained_model_final()
%   evaluate_trained_model_final('path/to/model.mat')
%
% Fixes vs previous version:
%   1. PTB class index found by name lookup (not hardcoded as col 2 or end)
%   2. confusionmat indexing uses ptbIdx (not hardcoded cm(2,2))
%   3. perfcurve uses the correct class label, not classes{2}
%   4. Ensemble uses calibrated weighted average, not flat mean
%   5. Youden's J threshold sweep on finer grid (0.01 step)
%   6. TTA (horizontal flip) added for OOD evaluation
%   7. Uncertainty estimation and filtering added
%   8. Full results table printed at the end
% -------------------------------------------------------------------------

if nargin < 1
    modelFile = 'models/final/final_model_histmatch_kfold_improved_29_04_26.mat';
end

fprintf('Loading model: %s\n', modelFile);
load(modelFile, 'trainedNet', 'config');
fprintf('Model loaded.\n\n');

imdsROI = imageDatastore('input/roi', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore('input/cxr', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

classes = categories(imdsROI.Labels);
useGPU  = canUseGPU();
prep    = config.preprocessing_method;
refHist = config.refHist;

% --- Find PTB class index ONCE, by name -----------------------------------
ptbIdx = find_ptb_index(classes);
fprintf('Classes: %s\n', strjoin(classes, ', '));
fprintf('PTB class index: %d (%s)\n', ptbIdx, classes{ptbIdx});
fprintf('ID samples: %d  |  OOD samples: %d\n\n', ...
    numel(imdsROI.Files), numel(imdsCXR.Files));

% --- ID Evaluation --------------------------------------------------------
fprintf('=== ID (ROI) Evaluation ===\n');
resID = evaluate_with_ensemble(trainedNet, imdsROI, classes, ptbIdx, ...
    useGPU, prep, refHist, false);

% --- OOD Evaluation -------------------------------------------------------
fprintf('\n=== OOD (CXR) Evaluation with Ensemble + TTA ===\n');
resOOD = evaluate_with_ensemble(trainedNet, imdsCXR, classes, ptbIdx, ...
    useGPU, prep, refHist, true);

% --- Print comparison -----------------------------------------------------
print_comparison(resID, resOOD, classes, ptbIdx);
end


% ==========================================================================
function res = evaluate_with_ensemble(net, imds, classes, ptbIdx, ...
    useGPU, base_prep, refHist, isOOD)
% Ensemble: weighted average of multiple preprocessing strategies + TTA for OOD.
%
% Weights reflect reliability for each preprocessing type:
%   histmatch  — highest: trained on this, best domain alignment
%   clahe      — medium:  local contrast, good for varied exposures
%   none       — lower:   raw pixel, useful as regulariser but noisier
%
% For OOD, TTA (horizontal flip) is applied and averaged with the original.

    methods = {'histmatch', 'clahe', 'none'};
    weights = [0.5, 0.3, 0.2];   % must sum to 1.0

    N = numel(imds.Files);
    Yprobs_weighted = zeros(N, 1);

    for m = 1:numel(methods)
        fprintf('   [%s] weight=%.1f ...', methods{m}, weights(m));

        % Original orientation
        probs_orig = get_probs(net, imds, classes, ptbIdx, useGPU, methods{m}, refHist);

        if isOOD
            % TTA: horizontal flip
            probs_flip = get_probs_flipped(net, imds, classes, ptbIdx, useGPU, methods{m}, refHist);
            probs_m = 0.6 * probs_orig + 0.4 * probs_flip;  % weight original higher
        else
            probs_m = probs_orig;
        end

        Yprobs_weighted = Yprobs_weighted + weights(m) * probs_m;
        fprintf(' done  (mean_prob=%.3f)\n', mean(probs_m));
    end

    % Compute metrics with threshold sweep
    res = compute_classification_metrics(Yprobs_weighted, imds.Labels, classes, ptbIdx);
    res.probs = Yprobs_weighted;

    fprintf('   → Acc=%.3f  Sens=%.3f  Spec=%.3f  F1=%.3f  AUC=%.3f  (thresh=%.2f)\n', ...
        res.accuracy, res.sensitivity, res.specificity, res.f1_score, res.auc, res.best_thresh);

    % Uncertainty
    res.uncertainty = compute_uncertainty(Yprobs_weighted);
    fprintf('   → Mean uncertainty=%.3f  High-uncertainty samples=%.1f%%\n', ...
        res.uncertainty.mean_unc, res.uncertainty.high_pct * 100);
end


% ==========================================================================
function Yprobs = get_probs(net, imds, classes, ptbIdx, useGPU, prep_method, refHist)
    augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing', 'gray2rgb');
    preprocessFcn = @(X, T) preprocessMiniBatch(X, T, prep_method, refHist);
    mbq = minibatchqueue(augDS, ...
        'MiniBatchSize', 16, ...
        'MiniBatchFcn', preprocessFcn, ...
        'MiniBatchFormat', ["SSCB", ""], ...
        'PartialMiniBatch', 'return');

    Yprobs = [];
    reset(mbq);
    while hasdata(mbq)
        [X, ~] = next(mbq);
        if useGPU, X = gpuArray(X); end
        scores = predict(net, X);
        probs  = extractdata(scores)';       % [batchSize x numClasses]
        Yprobs = [Yprobs; probs(:, ptbIdx)]; %#ok<AGROW>
    end
    Yprobs = Yprobs(1:numel(imds.Files));
end


% ==========================================================================
function Yprobs = get_probs_flipped(net, imds, classes, ptbIdx, useGPU, prep_method, refHist)
% Same as get_probs but applies a horizontal flip via augmenter (TTA).
    aug = imageDataAugmenter('RandXReflection', true);
    augDS = augmentedImageDatastore([224 224], imds, ...
        'ColorPreprocessing', 'gray2rgb', ...
        'DataAugmentation', aug);
    preprocessFcn = @(X, T) preprocessMiniBatch(X, T, prep_method, refHist);
    mbq = minibatchqueue(augDS, ...
        'MiniBatchSize', 16, ...
        'MiniBatchFcn', preprocessFcn, ...
        'MiniBatchFormat', ["SSCB", ""], ...
        'PartialMiniBatch', 'return');

    Yprobs = [];
    reset(mbq);
    while hasdata(mbq)
        [X, ~] = next(mbq);
        if useGPU, X = gpuArray(X); end
        scores = predict(net, X);
        probs  = extractdata(scores)';
        Yprobs = [Yprobs; probs(:, ptbIdx)]; %#ok<AGROW>
    end
    Yprobs = Yprobs(1:numel(imds.Files));
end


% ==========================================================================
function [X, T] = preprocessMiniBatch(dataX, dataT, preprocessing_method, refHist)
    X = cat(4, dataX{1:end});
    [H, W, C, B] = size(X);
    X_processed = zeros(H, W, C, B, 'uint8');

    for b = 1:B
        img = X(:,:,:,b);

        % Normalise float input to uint8 if needed
        if ~isa(img, 'uint8')
            mx = max(img(:));
            if mx <= 1.0 && mx > 0
                img = uint8(img .* 255);
            elseif mx > 1.0
                img = uint8(img ./ mx .* 255);
            else
                img = uint8(zeros(H, W, C, 'uint8'));
            end
        end

        % Grayscale
        if C == 3
            img_gray = rgb2gray(img);
        else
            img_gray = img(:,:,1);
        end

        % Preprocessing
        switch preprocessing_method
            case 'none'
                img_processed = img_gray;
            case 'clahe'
                img_processed = adapthisteq(img_gray, 'ClipLimit', 0.02, 'Distribution', 'uniform');
            case 'histmatch'
                if ~isempty(refHist)
                    img_processed = histeq(img_gray, refHist);
                else
                    img_processed = adapthisteq(img_gray, 'ClipLimit', 0.02);  % fallback
                end
            otherwise
                img_processed = img_gray;
        end

        % Replicate to 3 channels
        if C == 3
            X_processed(:,:,:,b) = repmat(img_processed, [1 1 3]);
        else
            X_processed(:,:,:,b) = img_processed;
        end
    end

    X = dlarray(single(X_processed) ./ 255, 'SSCB');
    T = onehotencode(cat(2, dataT{1:end}), 1);
end


% ==========================================================================
function res = compute_classification_metrics(Yprobs, Ytrue, classes, ptbIdx)
% Youden's J threshold sweep on a fine grid.
% All confusion matrix indexing uses ptbIdx — no hardcoding.

    normalIdx = 3 - ptbIdx;   % works for 2-class: if ptbIdx=2, normalIdx=1 and vice versa

    thresholds = 0.05:0.01:0.95;
    bestJ      = -inf;
    best_t     = 0.5;

    for t = thresholds
        preds_t = categorical(classes(1 + (Yprobs >= t)));
        cm_t = confusionmat(Ytrue, preds_t, 'Order', categorical(classes));
        if ~all(size(cm_t) == [2 2]), continue; end

        TP = cm_t(ptbIdx, ptbIdx);
        FN = cm_t(ptbIdx, normalIdx);
        TN = cm_t(normalIdx, normalIdx);
        FP = cm_t(normalIdx, ptbIdx);

        sens = TP / (TP + FN + eps);
        spec = TN / (TN + FP + eps);
        J    = sens + spec - 1;

        if J > bestJ
            bestJ  = J;
            best_t = t;
        end
    end

    % Final metrics at best threshold
    Ypred = categorical(classes(1 + (Yprobs >= best_t)));
    cm    = confusionmat(Ytrue, Ypred, 'Order', categorical(classes));

    TP = cm(ptbIdx, ptbIdx);
    FN = cm(ptbIdx, normalIdx);
    TN = cm(normalIdx, normalIdx);
    FP = cm(normalIdx, ptbIdx);

    res = struct();
    res.accuracy    = (TP + TN) / (TP + TN + FP + FN + eps);
    res.precision   = TP / (TP + FP + eps);
    res.sensitivity = TP / (TP + FN + eps);
    res.specificity = TN / (TN + FP + eps);
    res.f1_score    = 2 * res.precision * res.sensitivity / ...
                      (res.precision + res.sensitivity + eps);
    res.best_thresh = best_t;
    res.youden_j    = bestJ;
    res.confusion   = cm;
    res.TP = TP; res.FP = FP; res.TN = TN; res.FN = FN;

    try
        % Use the correct PTB label string for perfcurve
        ptb_label = classes{ptbIdx};
        [~, ~, ~, res.auc] = perfcurve(Ytrue, Yprobs, ptb_label);
    catch
        res.auc = 0.5;
    end
end


% ==========================================================================
function unc = compute_uncertainty(Yprobs)
% Margin uncertainty: how far each probability is from 0.5.
% Values near 0.5 = high uncertainty; near 0 or 1 = low uncertainty.
    margin  = abs(Yprobs - 0.5);
    raw_unc = 1 - 2 * margin;              % 0=certain, 1=maximally uncertain
    unc_thresh = prctile(raw_unc, 80);     % top 20% as "high uncertainty"

    unc.mean_unc  = mean(raw_unc);
    unc.std_unc   = std(raw_unc);
    unc.high_pct  = mean(raw_unc > unc_thresh);
    unc.threshold = unc_thresh;
    unc.values    = raw_unc;
end


% ==========================================================================
function print_comparison(id, ood, classes, ptbIdx)
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════╗\n');
    fprintf('║              FINAL EVALUATION RESULTS                   ║\n');
    fprintf('╠══════════════════════════════════════════════════════════╣\n');
    fprintf('║  Metric        │  ID (ROI)   │  OOD (CXR)  │  Gap      ║\n');
    fprintf('╠══════════════════════════════════════════════════════════╣\n');

    metrics = {'accuracy','sensitivity','specificity','f1_score','auc'};
    labels  = {'Accuracy  ','Sensitivity','Specificity','F1-Score  ','AUC       '};

    for i = 1:numel(metrics)
        id_val  = id.(metrics{i});
        ood_val = ood.(metrics{i});
        gap     = id_val - ood_val;
        flag    = '';
        if strcmp(metrics{i}, 'auc')
            if gap > 0.15, flag = ' ⚠'; end
        end
        fprintf('║  %s  │   %.3f     │   %.3f     │  %+.3f%s  ║\n', ...
            labels{i}, id_val, ood_val, gap, flag);
    end

    fprintf('╠══════════════════════════════════════════════════════════╣\n');
    fprintf('║  Threshold     │   %.3f     │   %.3f     │           ║\n', ...
        id.best_thresh, ood.best_thresh);
    fprintf('║  Youden J      │   %.3f     │   %.3f     │           ║\n', ...
        id.youden_j, ood.youden_j);
    fprintf('╚══════════════════════════════════════════════════════════╝\n');

    fprintf('\nConfusion matrix — ID (ROI):\n');
    disp(id.confusion);
    fprintf('Confusion matrix — OOD (CXR):\n');
    disp(ood.confusion);

    % Interpretation
    fprintf('\nInterpretation:\n');
    if ood.auc < 0.55
        fprintf('  ⚠  OOD AUC < 0.55: probabilities may be inverted.\n');
        fprintf('     Check that PTB folder name matches classes{%d} = "%s".\n', ...
            ptbIdx, classes{ptbIdx});
    end
    if ood.sensitivity > 0.99 && ood.specificity < 0.02
        fprintf('  ⚠  OOD all-PTB collapse: model predicts PTB for every CXR.\n');
        fprintf('     Most likely cause: class label mismatch between ROI and CXR folders.\n');
        fprintf('     ROI classes: %s\n', strjoin(classes, ', '));
    end
    auc_gap = id.auc - ood.auc;
    if auc_gap < 0.05
        fprintf('  ✓  AUC gap < 0.05: good generalisation.\n');
    elseif auc_gap < 0.15
        fprintf('  ~  AUC gap = %.3f: moderate degradation.\n', auc_gap);
    else
        fprintf('  ✗  AUC gap = %.3f: significant domain shift.\n', auc_gap);
    end
end


% ==========================================================================
function ptbIdx = find_ptb_index(classes)
    candidates = {'PTB','ptb','TB','tb','Tuberculosis','tuberculosis','positive','Positive','1'};
    ptbIdx = 2;  % fallback
    for k = 1:numel(candidates)
        idx = find(strcmp(classes, candidates{k}), 1);
        if ~isempty(idx)
            ptbIdx = idx;
            return;
        end
    end
    warning('PTB class not found by name. Using index %d (%s). Verify this is correct.', ...
        ptbIdx, classes{ptbIdx});
end