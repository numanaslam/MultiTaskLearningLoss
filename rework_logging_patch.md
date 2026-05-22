# rework.m logging patch

Two small additions to `rework.m` that enable figures 4 (loss curves) and 5 (alignment progress) in `publication_figures.m`. Without these, `publication_figures` will skip those figures and print a warning. Everything else still works.

## Edit 1 — Initialize epoch history

**Location:** Right after the line `for fold = 1:CONFIG.K_FOLDS` (around line 230 in the patched `rework.m` you have now).

**Add immediately after the `fprintf('\n--- Fold %d/%d ---\n', fold, CONFIG.K_FOLDS);` line:**

```matlab
% --- Initialize epoch-level history for this fold ------------------------
foldHistory = struct();
foldHistory.ce       = [];
foldHistory.kd       = [];
foldHistory.tversky  = [];
foldHistory.gradcam  = [];
foldHistory.valAcc   = [];
foldHistory.tsCosine = [];   % teacher-student CAM cosine on val (one sample)
```

## Edit 2 — Append to history each epoch

**Location:** Inside the epoch loop, immediately after the `fprintf('  -> Val Pred Counts: ...')` line, BEFORE the early-stopping block.

**Add this block:**

```matlab
% --- Append metrics to fold history --------------------------------------
foldHistory.ce(end+1)      = avgCE;
foldHistory.kd(end+1)      = avgKD;
foldHistory.tversky(end+1) = avgTversky;
foldHistory.gradcam(end+1) = avgGradcam;
foldHistory.valAcc(end+1)  = avgValAcc;

% Compute teacher-student cosine on one held-out val image as a cheap
% epoch-level signal of alignment progress. We sample the first val image
% with a valid mask.
foldHistory.tsCosine(end+1) = computeEpochTSCosine( ...
    dlnet, teacherNet, valFiles, valMasks, targetLayerName, ...
    studentFCName, teacherFCName);
```

## Edit 3 — Store history at end of fold

**Location:** Right before the `end` of the fold loop (after `globalBestValMasks = valMasks;` block).

**Add:**

```matlab
% --- Save this fold's history --------------------------------------------
if fold == 1
    epochHistory = foldHistory;
else
    epochHistory(fold) = foldHistory;
end
```

## Edit 4 — Include epochHistory in final save

**Location:** The `save(CONFIG.MODEL_SAVE_PATH, ...)` call near the end of the file.

**Change:**

```matlab
save(CONFIG.MODEL_SAVE_PATH, 'dlnet', 'teacherNet', 'CONFIG', 'uniqueClasses', ...
    'globalBestFold', 'globalBestValAcc', 'foldBestValAccs', ...
    'idMetrics', 'idPredictions', 'oodMetrics', 'oodPredictions', ...
    'idThresholdMetrics', 'oodThresholdMetrics', 'idTunedThreshold', ...
    'heldoutFiles', 'heldoutLabels', ...
    'trueOodMetrics', 'trueOodPredictions', 'trueOodThresholdMetrics', ...
    'trueOodAvailable', 'epochHistory', '-v7.3');   % <-- added epochHistory
```

## Edit 5 — Add the helper function

**Location:** Append to the end of `rework.m`, alongside the other helper functions.

**Add:**

```matlab
function cosVal = computeEpochTSCosine(dlnet, teacherNet, valFiles, valMasks, targetLayer, studentFCName, teacherFCName)
% Cheap per-epoch alignment probe: cosine between student CAM and teacher
% CAM on the first val image that has a non-empty mask. Returns NaN if no
% suitable sample exists.

    cosVal = NaN;
    for i = 1:numel(valFiles)
        if isempty(valMasks{i}) || ~exist(valMasks{i}, 'file'), continue; end
        img = imread(valFiles{i});
        if size(img, 3) == 1, img = cat(3, img, img, img); end
        img = imresize(img, [224 224]);

        try
            sCAM = singleCAM(dlnet, img, 2, targetLayer, studentFCName);   % class 2 = positive
            tCAM = singleCAM(teacherNet, img, 2, targetLayer, teacherFCName);
            sv = sCAM(:); tv = tCAM(:);
            cosVal = (sv' * tv) / (sqrt(sum(sv.^2)) * sqrt(sum(tv.^2)) + eps);
        catch
            cosVal = NaN;
        end
        return;
    end
end

function cam = singleCAM(net, img, classIdx, targetLayer, fcName)
    if ~isa(net, 'dlnetwork'), net = dlnetwork(layerGraph(net)); end
    XData = reshape(single(img), [size(img,1), size(img,2), 3, 1]);
    useGPU = false;
    if ~isempty(net.Learnables) && ~isempty(net.Learnables.Value)
        v = net.Learnables.Value{1};
        useGPU = isa(v, 'gpuArray') || (isa(v, 'dlarray') && isa(extractdata(v), 'gpuArray'));
    end
    if useGPU, XData = gpuArray(XData); end
    X = dlarray(XData, 'SSCB');
    camDl = dlfeval(@(n,x) localCamForward(n, x, classIdx, targetLayer, fcName), net, X);
    cam = squeeze(gather(extractdata(camDl)));
end

function camN = localCamForward(net, X, classIdx, targetLayer, fcName)
    [rawLogits, act] = forward(net, X, 'Outputs', {fcName, targetLayer});
    logitsVec = reshape(stripdims(rawLogits), [], 1);
    logits = dlarray(logitsVec, 'CB');
    tgt = zeros(size(logitsVec), 'like', logitsVec); tgt(classIdx) = 1;
    obj = sum(logits .* dlarray(tgt, 'CB'), 1);
    g = dlgradient(obj, act);
    w = mean(g, [1 2]);
    cam = relu(sum(act .* w, 3));
    cam = dlresize(cam, 'OutputSize', [size(X, 1) size(X, 2)]);
    camN = cam ./ (max(cam, [], [1 2]) + 1e-6);
end
```

## After applying these edits

Re-train (or re-run an existing model with the epoch loop alone). The saved `.mat` now contains an `epochHistory` struct array indexed by fold. Then `publication_figures.m` will produce figures 4 and 5 automatically.

## Cost

Each `computeEpochTSCosine` call adds about 0.2 seconds per epoch (one extra image through both teacher and student). Across 30 epochs × 2 folds = ~12 seconds total overhead per training run. Negligible.
