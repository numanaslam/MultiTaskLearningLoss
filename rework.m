%% ========================================================================
% Multi-Task Learning for PTB Classification (Teacher-Student Distillation)
% Optimized for Stability, Multi-Loss Tracking, and Uncertainty-Aware OOD
% Compatible with older MATLAB versions (No 'Cast' or 'ExecutionEnvironment')
%
% PATCH (2026-05-21):
%   - Added held-out test-set reservation BEFORE k-fold cv so the test set
%     never appears in any training fold.
%   - Replaced the OOD evaluation block with the held-out test set as the
%     primary "OOD" report (downstream code unchanged).
%   - Added a separate true-OOD evaluation that only runs when
%     CONFIG.OOD_DATA_ROOT points to a different directory than
%     CONFIG.DATA_ROOT (prevents the data-leakage bug).
%% ========================================================================
% Skip workspace clear if invoked by an ablation script that needs to keep
% its loop variables. Detected via the ABLATION_LAMBDA_GRADCAM env var.
if isempty(getenv('ABLATION_LAMBDA_GRADCAM'))
    clear;
    clc; close all;
end

%% ======================= 1. CONFIGURATION ===============================
CONFIG = struct();
% Random seed — overridable via ABLATION_SEED env var for triplicate runs
ablSeed = getenv('ABLATION_SEED');
if ~isempty(ablSeed)
    rng(str2double(ablSeed), 'twister');
    fprintf('  [ABLATION] random seed overridden to %s\n', ablSeed);
else
    rng(42, 'twister');
end
% Data Paths (Update these paths to match your local setup)
CONFIG.DATA_ROOT       = 'C:\numan\input\cxr';      % Images: e.g. cxr/normal, cxr/ptb
CONFIG.MASK_ROOT       = 'C:\numan\input\masks';    % Masks: masks/normal, masks/ptb
CONFIG.MODEL_SAVE_PATH = 'models\trained_mixed_vgg16.mat';
CONFIG.TEACHER_MODEL   = 'models\pretrained\vgg16_finetuned_on_roi.mat';

% Training Hyperparameters
CONFIG.EPOCHS          = 30;
CONFIG.MINI_BATCH_SIZE = 16;
CONFIG.INITIAL_LEARNING_RATE = 3e-5;
CONFIG.USE_GPU         = true;

% Default experiment:
% Use Grad-CAM during training, but keep it weak and pair it with mask-driven
% Tversky supervision. This is the best Grad-CAM-inclusive setting tested so far.
CONFIG.LOSS_WEIGHTS = struct();
CONFIG.LOSS_WEIGHTS.classification = 1.0;
CONFIG.LOSS_WEIGHTS.gradcam        = 0.01;
CONFIG.LOSS_WEIGHTS.tversky        = 0.05;
CONFIG.LOSS_WEIGHTS.kd             = 0.0;

% Curriculum / stabilization
CONFIG.KD_START_EPOCH      = 10;
CONFIG.KD_RAMP_EPOCHS      = 5;
CONFIG.AUX_START_EPOCH     = 8;
CONFIG.AUX_RAMP_EPOCHS     = 4;
CONFIG.EARLY_STOP_PATIENCE = 6;
CONFIG.MIN_EPOCHS_BEFORE_EARLY_STOP = 16;
CONFIG.GRADIENT_CLIP_NORM  = 1.0;

% Evaluation / reporting
CONFIG.K_FOLDS = 2;
CONFIG.QUICK_TEST = false;
CONFIG.PREPROCESSING_METHOD = 'none';
CONFIG.UNCERTAINTY_THRESHOLD = 0.55;   % Shannon Entropy cutoff for OOD detection
CONFIG.CALIBRATION_TEMPERATURE = 1.35;

% True OOD evaluation (set to a DIFFERENT dataset for cross-site evaluation).
% If this equals DATA_ROOT, the script will skip true-OOD evaluation to
% prevent training-set leakage. To enable, point this at e.g. Montgomery
% (if you trained on Shenzhen), TBX11K, or NIH ChestX-ray14 TB subset.
CONFIG.OOD_DATA_ROOT = 'C:\numan\input\cxr';
CONFIG.OOD_MAX_SAMPLES = inf;

% Held-out test-set reservation (new): a stratified fraction of the input
% dataset is held out BEFORE k-fold cv and never appears in any training
% fold. Used as the primary "OOD" report (it's the same dataset but unseen).
CONFIG.HELDOUT_FRACTION = 0.20;        % 20% held out as test set

CONFIG.ID_VIS_SAMPLES = 6;
CONFIG.OOD_VIS_SAMPLES = 6;
CONFIG.RESULTS_ROOT = fullfile('results', 'rework');
CONFIG.EXPERIMENT_NAME = 'ce_tversky_weak_gradcam';
CONFIG.RUN_OUTPUT_DIR = fullfile(CONFIG.RESULTS_ROOT, ...
    [CONFIG.EXPERIMENT_NAME '_' datestr(now, 'yyyy-mm-dd_HHMMSS')]);

%% ======================= 1b. ABLATION OVERRIDES =========================
% If the ablation_lambda_gradcam.m script sets these environment variables,
% override the matching CONFIG fields. Has no effect during normal use.
%
% Recognized env vars:
%   ABLATION_LAMBDA_GRADCAM   numeric, e.g. '0.500000'
%   ABLATION_EPOCHS           integer, e.g. '20'
%   ABLATION_MODEL_PATH       full path for the saved .mat
%   ABLATION_OUTPUT_DIR       directory for the run's artifacts
%   ABLATION_EXPERIMENT_TAG   tag appended to EXPERIMENT_NAME
ablLambda = getenv('ABLATION_LAMBDA_GRADCAM');
if ~isempty(ablLambda)
    CONFIG.LOSS_WEIGHTS.gradcam = str2double(ablLambda);
    fprintf('  [ABLATION] lambda_gradcam overridden to %.4f\n', ...
        CONFIG.LOSS_WEIGHTS.gradcam);
end
ablEpochs = getenv('ABLATION_EPOCHS');
if ~isempty(ablEpochs)
    CONFIG.EPOCHS = str2double(ablEpochs);
    fprintf('  [ABLATION] EPOCHS overridden to %d\n', CONFIG.EPOCHS);
end
ablModelPath = getenv('ABLATION_MODEL_PATH');
if ~isempty(ablModelPath)
    CONFIG.MODEL_SAVE_PATH = ablModelPath;
    fprintf('  [ABLATION] MODEL_SAVE_PATH overridden to %s\n', CONFIG.MODEL_SAVE_PATH);
end
ablOutputDir = getenv('ABLATION_OUTPUT_DIR');
if ~isempty(ablOutputDir)
    CONFIG.RUN_OUTPUT_DIR = ablOutputDir;
    fprintf('  [ABLATION] RUN_OUTPUT_DIR overridden to %s\n', CONFIG.RUN_OUTPUT_DIR);
end
ablTag = getenv('ABLATION_EXPERIMENT_TAG');
if ~isempty(ablTag)
    CONFIG.EXPERIMENT_NAME = sprintf('%s_%s', CONFIG.EXPERIMENT_NAME, ablTag);
end

%% ======================= 2. DATA LOADING ================================
fprintf('=== Loading Data from Structured Folders ===\n');
if ~exist(CONFIG.DATA_ROOT, 'dir')
    mkdir(fullfile(CONFIG.DATA_ROOT, 'normal'));
    mkdir(fullfile(CONFIG.DATA_ROOT, 'ptb'));
    mkdir(fullfile(CONFIG.MASK_ROOT, 'normal')); mkdir(fullfile(CONFIG.MASK_ROOT, 'ptb'));
    imwrite(uint8(randi([0 255], [224 224 3])), fullfile(CONFIG.DATA_ROOT, 'normal', 'sample1.png'));
    imwrite(uint8(randi([0 255], [224 224 3])), fullfile(CONFIG.DATA_ROOT, 'ptb', 'sample2.png'));
    imwrite(uint8(randi([0 255], [224 224])), fullfile(CONFIG.MASK_ROOT, 'normal', 'sample1_mask.png'));
    imwrite(uint8(randi([0 255], [224 224])), fullfile(CONFIG.MASK_ROOT, 'ptb', 'sample2_mask.png'));
    fprintf('--> Created dummy datasets since paths were missing. Replace with real assets.\n');
end

[imgPaths, imgLabels, uniqueClasses] = scanDataFolder(CONFIG.DATA_ROOT);
numClasses = numel(uniqueClasses);
numImages = numel(imgPaths);

fprintf('Found %d images. Classes: %s\n', numImages, strjoin(uniqueClasses, ', '));
if numImages == 0
    error('No images found. Check path: %s', CONFIG.DATA_ROOT);
end

% Encode string labels into numeric arrays
labelIDs = zeros(numImages, 1);
for i = 1:numImages
    labelIDs(i) = find(strcmp(uniqueClasses, imgLabels{i}));
end

% Map image elements to dense spatial segmentations
maskMap = containers.Map('KeyType', 'char', 'ValueType', 'char');
fprintf('Mapping masks...\n');
for i = 1:numImages
    [~, name, ~] = fileparts(imgPaths{i});
    className = imgLabels{i};
    maskFileName = [name '_mask.png'];
    maskPath = fullfile(CONFIG.MASK_ROOT, className, maskFileName);

    if exist(maskPath, 'file')
        maskMap(imgPaths{i}) = maskPath;
    else
        maskFound = dir(fullfile(CONFIG.MASK_ROOT, '**', maskFileName));
        if ~isempty(maskFound)
            maskMap(imgPaths{i}) = fullfile(maskFound(1).folder, maskFound(1).name);
        else
            maskMap(imgPaths{i}) = '';
        end
    end
end

if CONFIG.QUICK_TEST
    imgPaths = imgPaths(1:min(10, numImages));
    labelIDs = labelIDs(1:min(10, numImages));
    imgLabels = imgLabels(1:min(10, numImages));
    numImages = numel(imgPaths);
    fprintf('*** QUICK TEST MODE: Using subset samples ***\n');
end

%% ======================= 3. HELD-OUT TEST SET + CV ======================
% PATCH (2026-05-21): Reserve a stratified held-out test set BEFORE k-fold cv
% so it never appears in any training fold. This is the primary fix for the
% leakage where OOD_DATA_ROOT == DATA_ROOT caused inflated OOD numbers.
% Determinism is provided by rng(42) above.

fprintf('Reserving %.0f%% of images as held-out test set...\n', ...
    100 * CONFIG.HELDOUT_FRACTION);
heldoutPartition = cvpartition(labelIDs, 'HoldOut', CONFIG.HELDOUT_FRACTION, ...
    'Stratify', true);
heldoutIdx   = test(heldoutPartition);    % logical, true for held-out
trainPoolIdx = ~heldoutIdx;

heldoutFiles  = imgPaths(heldoutIdx);
heldoutLabels = labelIDs(heldoutIdx);
heldoutMasks  = cell(numel(heldoutFiles), 1);
for k = 1:numel(heldoutFiles)
    heldoutMasks{k} = maskMap(heldoutFiles{k});
end

% Restrict the pool the cv-partition will see
imgPaths  = imgPaths(trainPoolIdx);
labelIDs  = labelIDs(trainPoolIdx);
imgLabels = imgLabels(trainPoolIdx);
numImages = numel(imgPaths);

fprintf('  Held-out test : %d images (class counts: %s)\n', ...
    numel(heldoutFiles), mat2str(accumarray(heldoutLabels, 1)'));
fprintf('  Training pool : %d images (class counts: %s)\n', ...
    numImages, mat2str(accumarray(labelIDs, 1)'));

fprintf('Creating %d-fold stratified splits on training pool...\n', CONFIG.K_FOLDS);
cv = cvpartition(labelIDs, 'KFold', CONFIG.K_FOLDS, 'Stratify', true);

%% ======================= 4. NETWORK INITIALIZATION ======================
teacherModelDir = fileparts(CONFIG.TEACHER_MODEL);
if ~isempty(teacherModelDir) && ~exist(teacherModelDir, 'dir')
    mkdir(teacherModelDir);
end

if ~exist(CONFIG.TEACHER_MODEL, 'file')
    tLayers = getVGG16Layers(true);
    tLayers(end-2:end) = [];
    % ADDED FLATTENLAYER TO PREVENT SS LAYOUT ISSUES
    tLayers(end+1) = flattenLayer('Name', 'plat_flatten');
    tLayers(end+1) = fullyConnectedLayer(numClasses, 'Name', 'fc_new');
    teacherNet = dlnetwork(tLayers);
    save(CONFIG.TEACHER_MODEL, 'teacherNet');
    fprintf('--> Created a randomized placeholder Teacher Network for distillation.\n');
else
    S = load(CONFIG.TEACHER_MODEL);
    if isfield(S, 'teacherNet'),     teacherNet = S.teacherNet;
    elseif isfield(S, 'fold_model'), teacherNet = S.fold_model;
    elseif isfield(S, 'net'),        teacherNet = S.net;
    elseif isfield(S, 'trainedNet'), teacherNet = S.trainedNet;
    else
        tLayers = getVGG16Layers(true);
        tLayers = tLayers(1:end-3);
        tLayers(end+1) = flattenLayer('Name', 'plat_flatten');
        tLayers(end+1) = fullyConnectedLayer(numClasses, 'Name', 'fc_new');
        teacherNet = dlnetwork(tLayers);
    end
    % Convert to dlnetwork if needed
    if isa(teacherNet, 'DAGNetwork') || isa(teacherNet, 'SeriesNetwork')
        fprintf('  Converting teacher %s to dlnetwork...', class(teacherNet));
        try
            teacherNet = dag2dlnetwork(teacherNet);
        catch
            lg = layerGraph(teacherNet);
            try
                lg = removeLayers(lg, 'output');
            catch %#ok<CTCH>
            end
            try
                lg = removeLayers(lg, 'classoutput');
            catch %#ok<CTCH>
            end
            teacherNet = dlnetwork(lg);
        end
        fprintf('  Teacher conversion complete.');
    end
    % Verify teacher is not outputting random predictions
    % Pretrained VGG16 expects image intensities on the original 0-255 scale.
    % Its input layer performs zero-center normalization internally.
    testX = dlarray(single(randi([0 255], [224, 224, 3, 1])), 'SSCB');
    testFCName = findFCOutputLayer(teacherNet);
    testOut = extractdata(predict(teacherNet, testX, 'Outputs', testFCName));
    testProbs = exp(testOut) ./ sum(exp(testOut));
    fprintf('  Teacher sanity check — max prob: %.3f (>0.6 = trained, ~0.5 = random)', max(testProbs(:)));
end

teacherFCName = findFCOutputLayer(teacherNet);

fprintf('Initializing Student Network (VGG16 Logit Mode)...\n');

executionEnv = 'cpu';
if CONFIG.USE_GPU && gpuDeviceCount() > 0
    gpuDevice(1);
    executionEnv = 'gpu';
    fprintf('Using GPU: %s\n', gpuDevice().Name);
end

layers = getVGG16Layers(true);
layers(end-2:end) = [];
% RECONSTRUCTED OUTPUT BLOCKS WITH AN EXPLICIT FLATTEN STAGE
layers(end+1) = flattenLayer('Name', 'stud_flatten');
layers(end+1) = fullyConnectedLayer(numClasses, ...
    'WeightLearnRateFactor', 1, ...
    'BiasLearnRateFactor', 1, ...
    'Name', 'fc_new');
targetLayerName = resolveSharedFeatureLayer(layers, teacherNet, {'relu5_3', 'conv5_3'});
fprintf('Using feature layer: %s\n', targetLayerName);

%% ======================= 5. TRAINING LOOP ===============================
fprintf('\n=== STARTING K-FOLD MULTI-TASK CV LOOP ===\n');

globalBestNet = [];
globalBestValAcc = -inf;
globalBestFold = NaN;
globalBestValFiles = {};
globalBestValLabels = [];
globalBestValMasks = {};
foldBestValAccs = zeros(CONFIG.K_FOLDS, 1);

for fold = 1:CONFIG.K_FOLDS
    fprintf('\n--- Fold %d/%d ---\n', fold, CONFIG.K_FOLDS);
    trainIdx = training(cv, fold);
    valIdx   = test(cv, fold);

    trainFiles = imgPaths(trainIdx); trainLabels = labelIDs(trainIdx);
    trainMasks = cell(size(trainFiles));
    for k=1:numel(trainFiles), trainMasks{k} = maskMap(trainFiles{k}); end

    valFiles = imgPaths(valIdx); valLabels = labelIDs(valIdx);
    valMasks = cell(size(valFiles)); for k=1:numel(valFiles), valMasks{k} = maskMap(valFiles{k}); end

    valQueue = createMiniBatchQueue(valFiles, valLabels, valMasks, CONFIG.MINI_BATCH_SIZE, executionEnv);

    dlnet = dlnetwork(layers);
    studentFCName = findFCOutputLayer(dlnet);
    trailingAvgSqGrad = [];
    trailingAvgStep = [];
    bestValAcc = -inf;
    bestNet = dlnet;
    patienceCounter = 0;

    iteration = 0;
    for epoch = 1:CONFIG.EPOCHS
        epochLoss = 0; epochAcc = 0; numBatches = 0;
        epochCE = 0; epochKD = 0; epochTversky = 0; epochGradcam = 0;
        [epochTrainFiles, epochTrainLabels, epochTrainMasks] = shuffleTrainingTriples(trainFiles, trainLabels, trainMasks);
        trainQueue = createMiniBatchQueue(epochTrainFiles, epochTrainLabels, epochTrainMasks, CONFIG.MINI_BATCH_SIZE, executionEnv);
        reset(trainQueue);
        tic;
        while hasdata(trainQueue)
            iteration = iteration + 1;
            [X, T, M] = next(trainQueue);

            if epoch < floor(CONFIG.EPOCHS*0.6), learnRate = CONFIG.INITIAL_LEARNING_RATE;
            elseif epoch < floor(CONFIG.EPOCHS*0.9), learnRate = CONFIG.INITIAL_LEARNING_RATE * 0.1;
            else, learnRate = CONFIG.INITIAL_LEARNING_RATE * 0.01;
            end

            % Ramp auxiliary loss weights over first 10 epochs.
            % Epochs 1-5:  CE only (ramp = 0.0) — establish classification baseline
            % Epochs 6-10: half weight (ramp = 0.5) — introduce auxiliary signal
            % Epoch 11+:   full weight (ramp = 1.0) — full multi-task training
            kdRamp = min(1.0, max(0.0, (epoch - CONFIG.KD_START_EPOCH) / max(CONFIG.KD_RAMP_EPOCHS, 1)));
            auxRamp = min(1.0, max(0.0, (epoch - CONFIG.AUX_START_EPOCH) / max(CONFIG.AUX_RAMP_EPOCHS, 1)));
            adaptiveLossWeights = CONFIG.LOSS_WEIGHTS;
            adaptiveLossWeights.kd       = CONFIG.LOSS_WEIGHTS.kd * kdRamp;
            adaptiveLossWeights.tversky  = CONFIG.LOSS_WEIGHTS.tversky * auxRamp;
            adaptiveLossWeights.gradcam  = CONFIG.LOSS_WEIGHTS.gradcam  * auxRamp;

            [loss, grads, YPred, metrics] = dlfeval(@modelLoss, ...
                dlnet, teacherNet, X, T, M, adaptiveLossWeights, ...
                targetLayerName, studentFCName, teacherFCName);
            if CONFIG.GRADIENT_CLIP_NORM > 0
                grads = clipGradients(grads, CONFIG.GRADIENT_CLIP_NORM);
            end
            [dlnet, trailingAvgSqGrad, trailingAvgStep] = adamupdate(dlnet, grads, ...
                trailingAvgSqGrad, trailingAvgStep, iteration, learnRate);

            epochLoss = epochLoss + extractdata(loss);
            epochCE = epochCE + metrics.ce;
            epochKD = epochKD + metrics.kd;
            epochTversky = epochTversky + metrics.tversky;
            epochGradcam = epochGradcam + metrics.gradcam;
            [~, predClass] = max(extractdata(YPred), [], 1);
            [~, trueClass] = max(extractdata(T), [], 1);
            epochAcc = epochAcc + mean(predClass == trueClass);
            numBatches = numBatches + 1;
        end
        avgTrainLoss = epochLoss / numBatches;
        avgTrainAcc  = epochAcc / numBatches;
        avgCE = epochCE / numBatches;
        avgKD = epochKD / numBatches;
        avgTversky = epochTversky / numBatches;
        avgGradcam = epochGradcam / numBatches;
        tEpoch = toc;

        valAcc = 0; valBatches = 0;
        valPredCounts = zeros(numClasses, 1);
        valTrueCounts = zeros(numClasses, 1);
        reset(valQueue);
        while hasdata(valQueue)
            [vX, vT, ~] = next(valQueue);
            vRawLogits = predict(dlnet, vX);

            % Enforce matching clean format for validation predictions
            [vC, vB] = size(vT);
            vLogits = dlarray(reshape(stripdims(vRawLogits), [vC, vB]), 'CB');
            vYPred = softmax(vLogits);

            [~, vPredClass] = max(extractdata(vYPred), [], 1);
            [~, vTrueClass] = max(extractdata(vT), [], 1);
            valPredCounts = valPredCounts + accumarray(vPredClass(:), 1, [numClasses 1]);
            valTrueCounts = valTrueCounts + accumarray(vTrueClass(:), 1, [numClasses 1]);
            valAcc = valAcc + mean(vPredClass == vTrueClass);
            valBatches = valBatches + 1;
        end
        avgValAcc = valAcc / valBatches;
        valPredCountsCPU = toRowVectorCPU(valPredCounts);
        valTrueCountsCPU = toRowVectorCPU(valTrueCounts);
        fprintf('Epoch %d/%d - Loss: %.4f | Train Acc: %.2f%% | Val Acc: %.2f%% | (%.2fs)\n', ...
            epoch, CONFIG.EPOCHS, avgTrainLoss, avgTrainAcc*100, avgValAcc*100, tEpoch);
        fprintf('  -> Breakdowns: CE=%.3f, KD=%.3f, Tversky=%.3f, CAM_Align=%.3f\n', ...
            avgCE, avgKD, avgTversky, avgGradcam);
        fprintf('  -> Val Pred Counts: [%s] | Val True Counts: [%s]\n', ...
            strtrim(sprintf('%.0f ', valPredCountsCPU)), ...
            strtrim(sprintf('%.0f ', valTrueCountsCPU)));

        if avgValAcc > bestValAcc + 1e-4
            bestValAcc = avgValAcc;
            bestNet = dlnet;
            patienceCounter = 0;
        else
            patienceCounter = patienceCounter + 1;
        end

        if epoch >= CONFIG.MIN_EPOCHS_BEFORE_EARLY_STOP && patienceCounter >= CONFIG.EARLY_STOP_PATIENCE
            fprintf('Early stopping triggered on fold %d at epoch %d (best val acc %.2f%%)\n', ...
                fold, epoch, bestValAcc * 100);
            dlnet = bestNet;
            break;
        end
    end

    dlnet = bestNet;
    foldBestValAccs(fold) = bestValAcc;
    if bestValAcc > globalBestValAcc
        globalBestValAcc = bestValAcc;
        globalBestNet = dlnet;
        globalBestFold = fold;
        globalBestValFiles = valFiles;
        globalBestValLabels = valLabels;
        globalBestValMasks = valMasks;
    end
end

%% ======================= 6. POST-TRAINING EVALUATION ====================
fprintf('\n=== POST-TRAINING EVALUATION ===\n');
if isempty(globalBestNet)
    error('No best model was recorded during training.');
end

dlnet = globalBestNet;
studentFCName = findFCOutputLayer(dlnet);

if ~exist(CONFIG.RESULTS_ROOT, 'dir'), mkdir(CONFIG.RESULTS_ROOT); end
if ~exist(CONFIG.RUN_OUTPUT_DIR, 'dir'), mkdir(CONFIG.RUN_OUTPUT_DIR); end
modelSaveDir = fileparts(CONFIG.MODEL_SAVE_PATH);
if ~isempty(modelSaveDir) && ~exist(modelSaveDir, 'dir'), mkdir(modelSaveDir); end

fprintf('Best fold selected: %d (val acc %.2f%%)\n', globalBestFold, globalBestValAcc * 100);
fprintf('Saving artifacts to: %s\n', CONFIG.RUN_OUTPUT_DIR);

% --- Fold-validation set (ID) --------------------------------------------
[idMetrics, idPredictions] = evaluateFileSet( ...
    dlnet, globalBestValFiles, globalBestValLabels, globalBestValMasks, ...
    CONFIG.MINI_BATCH_SIZE, executionEnv, uniqueClasses, CONFIG);
fprintf('ID Fold-Val   | Acc=%.3f | Sens=%.3f | Spec=%.3f | F1=%.3f | Mean entropy=%.3f\n', ...
    idMetrics.accuracy, idMetrics.sensitivity, idMetrics.specificity, ...
    idMetrics.f1, idMetrics.mean_entropy);

% --- "OOD" evaluation = held-out TEST set --------------------------------
% PATCH (2026-05-21): Previously this block used CONFIG.OOD_DATA_ROOT, which
% often pointed at the same directory as CONFIG.DATA_ROOT and leaked training
% data into the OOD report (you saw 95% "OOD" accuracy that was actually
% train accuracy). The held-out test set is the same distribution as
% training but was UNSEEN during training — the proper generalization test.
% Variable names (oodMetrics/oodPredictions/oodFiles/oodLabels) are kept so
% the existing downstream CSV/figure/threshold code works unchanged.
fprintf('\n--- Evaluating on held-out TEST set (%d images, unseen during training) ---\n', ...
    numel(heldoutFiles));
oodFiles  = heldoutFiles;
oodLabels = heldoutLabels;
[oodMetrics, oodPredictions] = evaluateFileSet( ...
    dlnet, oodFiles, oodLabels, heldoutMasks, ...
    CONFIG.MINI_BATCH_SIZE, executionEnv, uniqueClasses, CONFIG);
oodAvailable = true;
fprintf('Held-out TEST | Acc=%.3f | Sens=%.3f | Spec=%.3f | F1=%.3f | Mean entropy=%.3f | OOD rate=%.1f%%\n', ...
    oodMetrics.accuracy, oodMetrics.sensitivity, oodMetrics.specificity, ...
    oodMetrics.f1, oodMetrics.mean_entropy, 100 * oodMetrics.ood_flag_rate);

% --- Optional: true cross-site OOD (different dataset) -------------------
% Activates ONLY if OOD_DATA_ROOT points to a directory DIFFERENT from
% DATA_ROOT. The leakage check resolves both paths to canonical form before
% comparing, so a trailing slash or case difference does not bypass it.
trueOodMetrics = createEmptyMetrics(uniqueClasses);
trueOodPredictions = struct();
trueOodFiles = {};
trueOodLabels = [];
trueOodAvailable = false;

if exist(CONFIG.OOD_DATA_ROOT, 'dir')
    dataRootResolved = char(java.io.File(CONFIG.DATA_ROOT).getCanonicalPath());
    oodRootResolved  = char(java.io.File(CONFIG.OOD_DATA_ROOT).getCanonicalPath());

    if ~strcmpi(dataRootResolved, oodRootResolved)
        fprintf('\n--- Evaluating on true OOD set: %s ---\n', CONFIG.OOD_DATA_ROOT);
        [trueOodFiles, trueOodLabelNames, ~] = scanImageFolderFlexible(CONFIG.OOD_DATA_ROOT);
        if isfinite(CONFIG.OOD_MAX_SAMPLES) && numel(trueOodFiles) > CONFIG.OOD_MAX_SAMPLES
            subsetIdx = randperm(numel(trueOodFiles), CONFIG.OOD_MAX_SAMPLES);
            trueOodFiles = trueOodFiles(subsetIdx);
            if ~isempty(trueOodLabelNames)
                trueOodLabelNames = trueOodLabelNames(subsetIdx);
            end
        end

        if ~isempty(trueOodFiles) && ~isempty(trueOodLabelNames)
            trueOodLabels = encodeLabelNames(trueOodLabelNames, uniqueClasses);
            keep = trueOodLabels > 0;
            trueOodFiles = trueOodFiles(keep);
            trueOodLabels = trueOodLabels(keep);

            if ~isempty(trueOodFiles)
                trueOodMasks = repmat({''}, numel(trueOodFiles), 1);
                [trueOodMetrics, trueOodPredictions] = evaluateFileSet( ...
                    dlnet, trueOodFiles, trueOodLabels, trueOodMasks, ...
                    CONFIG.MINI_BATCH_SIZE, executionEnv, uniqueClasses, CONFIG);
                trueOodAvailable = true;
                fprintf('True OOD      | Acc=%.3f | Sens=%.3f | Spec=%.3f | F1=%.3f | Mean entropy=%.3f | OOD rate=%.1f%%\n', ...
                    trueOodMetrics.accuracy, trueOodMetrics.sensitivity, trueOodMetrics.specificity, ...
                    trueOodMetrics.f1, trueOodMetrics.mean_entropy, 100 * trueOodMetrics.ood_flag_rate);
            else
                fprintf('No labeled OOD images remained after class-name filtering.\n');
            end
        else
            fprintf('No OOD images found under: %s\n', CONFIG.OOD_DATA_ROOT);
        end
    else
        fprintf('\nSkipping true-OOD evaluation: OOD_DATA_ROOT equals DATA_ROOT (would leak training data).\n');
        fprintf('To enable: set CONFIG.OOD_DATA_ROOT to a DIFFERENT dataset directory.\n');
    end
else
    fprintf('\nOOD directory not found: %s\n', CONFIG.OOD_DATA_ROOT);
end

% --- Save CSVs, figures, and Grad-CAM galleries --------------------------
saveEvaluationMetricsTable(idMetrics, oodMetrics, ...
    fullfile(CONFIG.RUN_OUTPUT_DIR, 'evaluation_metrics.csv'));
savePredictionsCsv(fullfile(CONFIG.RUN_OUTPUT_DIR, 'id_predictions.csv'), ...
    idPredictions, uniqueClasses);
saveEvaluationSummaryFigure(idMetrics, idPredictions, uniqueClasses, ...
    'Fold Validation (ID)', fullfile(CONFIG.RUN_OUTPUT_DIR, 'id_summary.png'));
saveGradcamGallery(dlnet, globalBestValFiles, globalBestValLabels, idPredictions, ...
    uniqueClasses, targetLayerName, studentFCName, CONFIG, ...
    fullfile(CONFIG.RUN_OUTPUT_DIR, 'id_gradcam_gallery.png'), ...
    'Fold Validation (ID)', CONFIG.ID_VIS_SAMPLES);

% --- Threshold tuning on ID, applied to held-out test --------------------
idTunedThreshold = NaN;
idThresholdMetrics = createEmptyMetrics(uniqueClasses);
oodThresholdMetrics = createEmptyMetrics(uniqueClasses);
if numel(uniqueClasses) == 2 && all(isfinite(idPredictions.trueIdx))
    posIdx = inferPositiveClassIdx(uniqueClasses);
    idTunedThreshold = tuneBinaryThreshold(idPredictions.trueIdx, ...
        idPredictions.probs(:, posIdx), uniqueClasses);
    idThresholdMetrics = computeThresholdMetrics( ...
        idPredictions.trueIdx, idPredictions.probs(:, posIdx), idTunedThreshold, ...
        idPredictions.confidence, idPredictions.entropy, idPredictions.isOOD, uniqueClasses);
    fprintf('\nID-tuned PTB threshold: %.3f\n', idTunedThreshold);
    fprintf('ID Fold-Val (thresholded)   | Acc=%.3f | Sens=%.3f | Spec=%.3f | F1=%.3f\n', ...
        idThresholdMetrics.accuracy, idThresholdMetrics.sensitivity, ...
        idThresholdMetrics.specificity, idThresholdMetrics.f1);

    if oodAvailable && all(isfinite(oodPredictions.trueIdx))
        oodThresholdMetrics = computeThresholdMetrics( ...
            oodPredictions.trueIdx, oodPredictions.probs(:, posIdx), idTunedThreshold, ...
            oodPredictions.confidence, oodPredictions.entropy, oodPredictions.isOOD, uniqueClasses);
        fprintf('Held-out TEST (ID threshold)| Acc=%.3f | Sens=%.3f | Spec=%.3f | F1=%.3f\n', ...
            oodThresholdMetrics.accuracy, oodThresholdMetrics.sensitivity, ...
            oodThresholdMetrics.specificity, oodThresholdMetrics.f1);
    end
end

% --- Save held-out test artifacts ----------------------------------------
if oodAvailable
    savePredictionsCsv(fullfile(CONFIG.RUN_OUTPUT_DIR, 'heldout_test_predictions.csv'), ...
        oodPredictions, uniqueClasses);
    saveEvaluationSummaryFigure(oodMetrics, oodPredictions, uniqueClasses, ...
        'Held-out TEST (same distribution, unseen)', ...
        fullfile(CONFIG.RUN_OUTPUT_DIR, 'heldout_test_summary.png'));
    saveGradcamGallery(dlnet, oodFiles, oodLabels, oodPredictions, ...
        uniqueClasses, targetLayerName, studentFCName, CONFIG, ...
        fullfile(CONFIG.RUN_OUTPUT_DIR, 'heldout_test_gradcam_gallery.png'), ...
        'Held-out TEST', CONFIG.OOD_VIS_SAMPLES);
end

saveThresholdMetricsTable(idMetrics, oodMetrics, idThresholdMetrics, oodThresholdMetrics, ...
    idTunedThreshold, fullfile(CONFIG.RUN_OUTPUT_DIR, 'evaluation_metrics_with_thresholds.csv'));
if numel(uniqueClasses) == 2
    saveProbabilityDiagnosticsFigure(idPredictions, oodPredictions, uniqueClasses, ...
        fullfile(CONFIG.RUN_OUTPUT_DIR, 'probability_diagnostics.png'));
end

% --- Save true-OOD artifacts (if a separate dataset was provided) --------
trueOodThresholdMetrics = createEmptyMetrics(uniqueClasses);
if trueOodAvailable
    savePredictionsCsv(fullfile(CONFIG.RUN_OUTPUT_DIR, 'true_ood_predictions.csv'), ...
        trueOodPredictions, uniqueClasses);
    saveEvaluationSummaryFigure(trueOodMetrics, trueOodPredictions, uniqueClasses, ...
        'True OOD (different dataset)', ...
        fullfile(CONFIG.RUN_OUTPUT_DIR, 'true_ood_summary.png'));
    saveGradcamGallery(dlnet, trueOodFiles, trueOodLabels, trueOodPredictions, ...
        uniqueClasses, targetLayerName, studentFCName, CONFIG, ...
        fullfile(CONFIG.RUN_OUTPUT_DIR, 'true_ood_gradcam_gallery.png'), ...
        'True OOD', CONFIG.OOD_VIS_SAMPLES);

    if numel(uniqueClasses) == 2 && isfinite(idTunedThreshold) && ...
            all(isfinite(trueOodPredictions.trueIdx))
        posIdx = inferPositiveClassIdx(uniqueClasses);
        trueOodThresholdMetrics = computeThresholdMetrics( ...
            trueOodPredictions.trueIdx, trueOodPredictions.probs(:, posIdx), idTunedThreshold, ...
            trueOodPredictions.confidence, trueOodPredictions.entropy, ...
            trueOodPredictions.isOOD, uniqueClasses);
        fprintf('True OOD (ID threshold)     | Acc=%.3f | Sens=%.3f | Spec=%.3f | F1=%.3f\n', ...
            trueOodThresholdMetrics.accuracy, trueOodThresholdMetrics.sensitivity, ...
            trueOodThresholdMetrics.specificity, trueOodThresholdMetrics.f1);
    end

    % Append true-OOD row to the metrics CSV
    fid = fopen(fullfile(CONFIG.RUN_OUTPUT_DIR, 'evaluation_metrics.csv'), 'a');
    if fid >= 0
        fprintf(fid, 'true_ood,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
            trueOodMetrics.accuracy, trueOodMetrics.precision, trueOodMetrics.sensitivity, ...
            trueOodMetrics.specificity, trueOodMetrics.f1, trueOodMetrics.mean_confidence, ...
            trueOodMetrics.mean_entropy, trueOodMetrics.mean_positive_prob, ...
            trueOodMetrics.ood_flag_rate);
        fclose(fid);
    end
end

save(CONFIG.MODEL_SAVE_PATH, 'dlnet', 'teacherNet', 'CONFIG', 'uniqueClasses', ...
    'globalBestFold', 'globalBestValAcc', 'foldBestValAccs', ...
    'globalBestValFiles', 'globalBestValLabels', 'globalBestValMasks', ...
    'idMetrics', 'idPredictions', 'oodMetrics', 'oodPredictions', ...
    'idThresholdMetrics', 'oodThresholdMetrics', 'idTunedThreshold', ...
    'heldoutFiles', 'heldoutLabels', ...
    'trueOodMetrics', 'trueOodPredictions', 'trueOodThresholdMetrics', ...
    'trueOodAvailable', '-v7.3');
fprintf('\nModel and evaluation artifacts saved to: %s\n', CONFIG.MODEL_SAVE_PATH);
fprintf('Output directory: %s\n', CONFIG.RUN_OUTPUT_DIR);

%% ======================= 7. BACKBONE PIPELINE FUNCTIONS =================
function [imgPaths, imgLabels, uniqueClasses] = scanDataFolder(dataRoot)
    extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp'};
    imgPaths = {}; imgLabels = {};
    dirContent = dir(dataRoot);
    subDirs = dirContent([dirContent.isdir] & ~strcmp({dirContent.name}, '.') & ~strcmp({dirContent.name}, '..'));
    uniqueClasses = {subDirs.name};
    for i = 1:numel(subDirs)
        currentDir = fullfile(dataRoot, subDirs(i).name);
        for j = 1:numel(extensions)
            files = dir(fullfile(currentDir, extensions{j}));
            for k = 1:numel(files)
                imgPaths{end+1, 1} = fullfile(files(k).folder, files(k).name); %#ok<AGROW>
                imgLabels{end+1, 1} = subDirs(i).name; %#ok<AGROW>
            end
        end
    end
end

function [imgPaths, imgLabels, uniqueClasses] = scanImageFolderFlexible(dataRoot)
    [imgPaths, imgLabels, uniqueClasses] = scanDataFolder(dataRoot);
    if ~isempty(imgPaths)
        return;
    end

    extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp'};
    imgPaths = {};
    for j = 1:numel(extensions)
        files = dir(fullfile(dataRoot, '**', extensions{j}));
        for k = 1:numel(files)
            imgPaths{end+1, 1} = fullfile(files(k).folder, files(k).name); %#ok<AGROW>
        end
    end
    imgLabels = {};
    uniqueClasses = {};
end

function labelIDs = encodeLabelNames(labelNames, classNames)
    if isempty(labelNames)
        labelIDs = [];
        return;
    end

    labelIDs = zeros(numel(labelNames), 1);
    for i = 1:numel(labelNames)
        matchIdx = find(strcmp(classNames, labelNames{i}), 1, 'first');
        if isempty(matchIdx)
            labelIDs(i) = 0;
        else
            labelIDs(i) = matchIdx;
        end
    end
end

function queue = createMiniBatchQueue(files, labels, masks, batchSize, env)
    imDs = imageDatastore(files, 'ReadFcn', @readAndPrepareImage);
    lblDs = arrayDatastore(dummyvar(labels)', 'IterationDimension', 2);

    validMasks = masks;
    for i = 1:numel(validMasks)
        if isempty(validMasks{i}) || ~exist(validMasks{i}, 'file')
            dummyMask = fullfile(pwd, 'dummy_temp_mask.png');
            if ~exist(dummyMask, 'file'), imwrite(zeros(224, 224, 'uint8'), dummyMask); end
            validMasks{i} = dummyMask;
        end
    end
    maskDs = imageDatastore(validMasks, 'ReadFcn', @readAndPrepareMask);

    combinedDs = combine(imDs, lblDs, maskDs);
    outEnv = repmat({env}, 1, 3);
    queue = minibatchqueue(combinedDs, 'MiniBatchSize', batchSize, ...
        'OutputAsDlarray', [true, true, true], ...
        'OutputCast', {'single', 'single', 'single'}, ...
        'MiniBatchFormat', {'SSCB', 'CB', 'SSCB'}, ...
        'OutputEnvironment', outEnv);
end

function img = readAndPrepareImage(filename)
    img = imread(filename);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    elseif size(img, 3) > 3
        img = img(:, :, 1:3);
    end
    img = imresize(img, [224, 224]);
    % Keep the 0-255 intensity range for pretrained VGG16.
    % The imageInputLayer already applies zero-center normalization.
    img = single(img);
end

function mask = readAndPrepareMask(filename)
    mask = imread(filename);
    if size(mask, 3) > 1
        mask = rgb2gray(mask);
    end
    mask = imresize(mask, [224, 224], 'nearest');
    mask = single(mask) / 255;
    mask = single(mask > 0.5);
    mask = reshape(mask, [224, 224, 1]);
end

function [loss, gradients, YPred, metrics] = modelLoss(dlnet, teacherNet, X, T, M, lossWeights, targetLayer, studentFCName, teacherFCName)
    % 1. Forward passes extracting features for Grad-CAM
    [rawStudentLogits, sActivations] = forward(dlnet, X, 'Outputs', {studentFCName, targetLayer});

    [numClasses, batchSize] = size(T);

    % Now safely reshapes because flattenLayer removes spatial metadata natively
    studentLogits = dlarray(reshape(stripdims(rawStudentLogits), [numClasses, batchSize]), 'CB');
    YPred = softmax(studentLogits);
    % Use the true-class logit objective for Grad-CAM instead of the combined loss.
    studentClassObjective = mean(sum(studentLogits .* T, 1));

    metrics = struct('ce', 0, 'kd', 0, 'tversky', 0, 'gradcam', 0);

    % Cross Entropy Base Loss
    ceLoss = crossentropy(YPred, T);
    metrics.ce = extractdata(ceLoss);

    % Knowledge Distillation Loss
    kdLoss = 0;
    teacherClassObjective = [];
    if ~isempty(teacherNet) && ~isempty(teacherFCName)
        [rawTeacherLogits, tActivations] = forward(teacherNet, X, 'Outputs', {teacherFCName, targetLayer});

        teacherLogits = dlarray(reshape(stripdims(rawTeacherLogits), [numClasses, batchSize]), 'CB');
        teacherClassObjective = mean(sum(teacherLogits .* T, 1));

        teacherSoft = softmax(teacherLogits / 2.0);
        studentSoft = softmax(studentLogits / 2.0);
        kdLoss = -sum(teacherSoft .* log(studentSoft + 1e-6 ), 'all') / size(X, 4);
        metrics.kd = extractdata(kdLoss);
    else
        tActivations = [];
    end

    % Calculate balanced classification/distillation objective path first
    kdWeight = 0;
    if isfield(lossWeights, 'kd')
        kdWeight = lossWeights.kd;
    end
    lossClassification = (lossWeights.classification * ceLoss) + (kdWeight * kdLoss);

    % 2. Dynamic single-pass backpropagation tracking for feature layers
    sGrads = dlgradient(studentClassObjective, sActivations, 'RetainData', true);

    % Generate Student Grad-CAM Profile map
    sWeights = mean(sGrads, [1 2]);
    gradcamMap = relu(sum(sActivations .* sWeights, 3));
    gradcamResized = dlresize(gradcamMap, 'OutputSize', [size(X, 1), size(X, 2)]);
    gradcamNorm = gradcamResized ./ (max(gradcamResized, [], [1 2]) + 1e-6);

    % 3. Tversky Semantic Localization Loss Integration
    tverskyLoss = 0;
    if lossWeights.tversky > 0 && ~isempty(M)
        alpha = 0.3; beta = 0.7;
        M_slice = M(:,:,1,:);
        num = sum(gradcamNorm .* M_slice, [1 2]);
        den = num + alpha * sum(gradcamNorm .* (1 - M_slice), [1 2]) + beta * sum((1 - gradcamNorm) .* M_slice, [1 2]);
        tverskyLoss = mean(1 - (num ./ (den + 1e-6)));
        metrics.tversky = extractdata(tverskyLoss);
    end

    % 4. Teacher-Student Structural Mapping Alignment Loss (COSINE)
    %    Replaced MSE with cosine (2026-05-21) — empirically transfers
    %    attention without the gradient pathology of normalised-MSE.
    %    Best configuration: lambda_gradcam = 2.0 with cosine.
    gradcamAlignLoss = 0;
    if lossWeights.gradcam > 0 && ~isempty(teacherNet) && ~isempty(tActivations)
        tGrads = dlgradient(teacherClassObjective, tActivations, 'RetainData', true);

        tWeights = mean(tGrads, [1 2]);
        tGradcam = relu(sum(tActivations .* tWeights, 3));
        tGradcamNorm = dlresize(tGradcam, 'OutputSize', [size(X, 1), size(X, 2)]);
        tGradcamNorm = tGradcamNorm ./ (max(tGradcamNorm, [], [1 2]) + 1e-6);

        % Cosine similarity per sample, averaged across the batch.
        sV = reshape(gradcamNorm,   [], size(gradcamNorm,   4));   % HW x B
        tV = reshape(tGradcamNorm,  [], size(tGradcamNorm,  4));   % HW x B
        dotP = sum(sV .* tV, 1);
        nS = sqrt(sum(sV .^ 2, 1) + 1e-6);
        nT = sqrt(sum(tV .^ 2, 1) + 1e-6);
        gradcamAlignLoss = mean(1 - dotP ./ (nS .* nT));
        metrics.gradcam = extractdata(gradcamAlignLoss);
    end

    % 5. Assemble final multi-task overall loss function
    loss = lossClassification + (lossWeights.tversky * tverskyLoss) + (lossWeights.gradcam * gradcamAlignLoss);
    gradients = dlgradient(loss, dlnet.Learnables);
end

function [finalVerdict, uncertaintyScore, isOOD] = evaluateOODUncertainty(dlnet, cxrImage, thresholdEntropy, calibrationTemp) %#ok<DEFNU>
    if size(cxrImage, 3) == 1, cxrImage = cat(3, cxrImage, cxrImage, cxrImage); end
    X_dl = dlarray(single(imresize(cxrImage, [224, 224])), 'SSCB');

    rawLogits = predict(dlnet, X_dl);

    % Safe execution environment handling for single evaluation
    calibratedLogits = dlarray(reshape(stripdims(rawLogits), [], 1), 'CB') / calibrationTemp;
    calibratedProbs = extractdata(softmax(calibratedLogits));

    uncertaintyScore = -sum(calibratedProbs .* log(calibratedProbs + 1e-6));
    [~, finalVerdict] = max(calibratedProbs);
    if uncertaintyScore > thresholdEntropy
        isOOD = true;
        finalVerdict = -1;
    else
        isOOD = false;
    end
end

function [metrics, predictions] = evaluateFileSet(dlnet, files, labelIDs, masks, batchSize, env, classNames, config)
    if nargin < 4 || isempty(masks)
        masks = repmat({''}, numel(files), 1);
    end
    if nargin < 5 || isempty(batchSize)
        batchSize = 16;
    end
    if nargin < 6 || isempty(env)
        env = 'cpu';
    end

    numSamples = numel(files);
    numClasses = numel(classNames);
    hasLabels = ~isempty(labelIDs) && numel(labelIDs) == numSamples;

    if numSamples == 0
        metrics = createEmptyMetrics(classNames);
        predictions = struct('files', {{}}, 'trueIdx', [], 'predIdx', [], ...
            'probs', zeros(0, numClasses), 'confidence', [], 'entropy', [], 'isOOD', []);
        return;
    end

    if hasLabels
        queueLabels = labelIDs;
    else
        queueLabels = ones(numSamples, 1);
    end

    evalQueue = createMiniBatchQueue(files, queueLabels, masks, batchSize, env);
    probsAll = zeros(numSamples, numClasses);
    predIdxAll = zeros(numSamples, 1);
    trueIdxAll = nan(numSamples, 1);
    confidenceAll = zeros(numSamples, 1);
    entropyAll = zeros(numSamples, 1);

    cursor = 1;
    reset(evalQueue);
    while hasdata(evalQueue)
        [X, T, ~] = next(evalQueue);
        rawLogits = predict(dlnet, X);
        batchSizeActual = size(X, 4);
        logits = dlarray(reshape(stripdims(rawLogits), [numClasses, batchSizeActual]), 'CB');
        probs = extractdata(softmax(logits / config.CALIBRATION_TEMPERATURE));
        if isa(probs, 'gpuArray')
            probs = gather(probs);
        end
        probs = probs';

        [confBatch, predBatch] = max(probs, [], 2);
        entropyBatch = -sum(probs .* log(probs + 1e-6), 2);

        batchRange = cursor:(cursor + batchSizeActual - 1);
        probsAll(batchRange, :) = probs;
        predIdxAll(batchRange) = predBatch;
        confidenceAll(batchRange) = confBatch;
        entropyAll(batchRange) = entropyBatch;

        if hasLabels
            [~, trueBatch] = max(extractdata(T), [], 1);
            if isa(trueBatch, 'gpuArray')
                trueBatch = gather(trueBatch);
            end
            trueIdxAll(batchRange) = trueBatch(:);
        end

        cursor = cursor + batchSizeActual;
    end

    isOODAll = entropyAll > config.UNCERTAINTY_THRESHOLD;
    metrics = computePredictionMetrics(trueIdxAll, predIdxAll, probsAll, confidenceAll, entropyAll, isOODAll, classNames);
    predictions = struct();
    predictions.files = files;
    predictions.trueIdx = trueIdxAll;
    predictions.predIdx = predIdxAll;
    predictions.probs = probsAll;
    predictions.confidence = confidenceAll;
    predictions.entropy = entropyAll;
    predictions.isOOD = isOODAll;
end

function metrics = createEmptyMetrics(classNames)
    numClasses = numel(classNames);
    metrics = struct();
    metrics.accuracy = NaN;
    metrics.precision = NaN;
    metrics.sensitivity = NaN;
    metrics.specificity = NaN;
    metrics.f1 = NaN;
    metrics.mean_confidence = NaN;
    metrics.mean_entropy = NaN;
    metrics.mean_positive_prob = NaN;
    metrics.ood_flag_rate = NaN;
    metrics.confusion_matrix = nan(numClasses, numClasses);
end

function metrics = computePredictionMetrics(trueIdx, predIdx, probs, confidence, entropyVals, isOODVals, classNames)
    numClasses = numel(classNames);
    metrics = createEmptyMetrics(classNames);
    metrics.mean_confidence = mean(confidence);
    metrics.mean_entropy = mean(entropyVals);
    metrics.ood_flag_rate = mean(isOODVals);

    posIdx = inferPositiveClassIdx(classNames);
    if size(probs, 2) >= posIdx
        metrics.mean_positive_prob = mean(probs(:, posIdx));
    end

    validTrue = isfinite(trueIdx);
    if ~any(validTrue)
        return;
    end

    trueIdx = trueIdx(validTrue);
    predIdx = predIdx(validTrue);
    cm = accumarray([trueIdx(:), predIdx(:)], 1, [numClasses, numClasses]);
    metrics.confusion_matrix = cm;
    metrics.accuracy = sum(diag(cm)) / max(sum(cm(:)), 1);

    if numClasses == 2
        negIdx = 3 - posIdx;
        tp = cm(posIdx, posIdx);
        fn = cm(posIdx, negIdx);
        tn = cm(negIdx, negIdx);
        fp = cm(negIdx, posIdx);
        metrics.precision = tp / max(tp + fp, 1);
        metrics.sensitivity = tp / max(tp + fn, 1);
        metrics.specificity = tn / max(tn + fp, 1);
        metrics.f1 = 2 * tp / max(2 * tp + fp + fn, 1);
    else
        precisionPerClass = diag(cm) ./ max(sum(cm, 1)', 1);
        recallPerClass = diag(cm) ./ max(sum(cm, 2), 1);
        metrics.precision = mean(precisionPerClass);
        metrics.sensitivity = mean(recallPerClass);
        metrics.specificity = NaN;
        metrics.f1 = NaN;
    end
end

function posIdx = inferPositiveClassIdx(classNames)
    posIdx = find(strcmpi(classNames, 'ptb'), 1, 'first');
    if isempty(posIdx)
        posIdx = min(2, numel(classNames));
    end
end

function saveEvaluationMetricsTable(idMetrics, oodMetrics, outPath)
    fid = fopen(outPath, 'w');
    if fid < 0
        warning('Unable to write metrics table: %s', outPath);
        return;
    end

    fprintf(fid, 'dataset,accuracy,precision,sensitivity,specificity,f1,mean_confidence,mean_entropy,mean_positive_prob,ood_flag_rate\n');
    fprintf(fid, 'id_fold_val,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        idMetrics.accuracy, idMetrics.precision, idMetrics.sensitivity, idMetrics.specificity, ...
        idMetrics.f1, idMetrics.mean_confidence, idMetrics.mean_entropy, ...
        idMetrics.mean_positive_prob, idMetrics.ood_flag_rate);
    fprintf(fid, 'heldout_test,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        oodMetrics.accuracy, oodMetrics.precision, oodMetrics.sensitivity, oodMetrics.specificity, ...
        oodMetrics.f1, oodMetrics.mean_confidence, oodMetrics.mean_entropy, ...
        oodMetrics.mean_positive_prob, oodMetrics.ood_flag_rate);
    fclose(fid);
end

function saveThresholdMetricsTable(idMetrics, oodMetrics, idThresholdMetrics, oodThresholdMetrics, threshold, outPath)
    fid = fopen(outPath, 'w');
    if fid < 0
        warning('Unable to write threshold metrics table: %s', outPath);
        return;
    end

    fprintf(fid, 'mode,dataset,threshold,accuracy,precision,sensitivity,specificity,f1,mean_confidence,mean_entropy,mean_positive_prob,ood_flag_rate\n');
    writeMetricsRow(fid, 'argmax', 'id_fold_val', NaN, idMetrics);
    writeMetricsRow(fid, 'argmax', 'heldout_test', NaN, oodMetrics);
    writeMetricsRow(fid, 'id_tuned_threshold', 'id_fold_val', threshold, idThresholdMetrics);
    writeMetricsRow(fid, 'id_tuned_threshold', 'heldout_test', threshold, oodThresholdMetrics);
    fclose(fid);
end

function writeMetricsRow(fid, modeName, datasetName, threshold, metrics)
    fprintf(fid, '%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        modeName, datasetName, threshold, ...
        metrics.accuracy, metrics.precision, metrics.sensitivity, metrics.specificity, ...
        metrics.f1, metrics.mean_confidence, metrics.mean_entropy, ...
        metrics.mean_positive_prob, metrics.ood_flag_rate);
end

function savePredictionsCsv(outPath, predictions, classNames)
    fid = fopen(outPath, 'w');
    if fid < 0
        warning('Unable to write predictions CSV: %s', outPath);
        return;
    end

    posIdx = inferPositiveClassIdx(classNames);
    fprintf(fid, 'file,true_label,pred_label,positive_prob,max_confidence,entropy,is_ood\n');
    for i = 1:numel(predictions.files)
        trueLabel = indexToLabel(predictions.trueIdx, i, classNames);
        predLabel = indexToLabel(predictions.predIdx, i, classNames);
        fprintf(fid, '"%s",%s,%s,%.6f,%.6f,%.6f,%d\n', ...
            predictions.files{i}, trueLabel, predLabel, ...
            predictions.probs(i, posIdx), predictions.confidence(i), ...
            predictions.entropy(i), predictions.isOOD(i));
    end
    fclose(fid);
end

function threshold = tuneBinaryThreshold(trueIdx, posProb, classNames)
    posIdx = inferPositiveClassIdx(classNames);
    negIdx = 3 - posIdx;
    thresholds = 0.10:0.01:0.95;
    bestBalancedAcc = -inf;
    threshold = 0.5;

    for t = thresholds
        predIdx = negIdx * ones(size(posProb));
        predIdx(posProb >= t) = posIdx;
        tp = sum(trueIdx == posIdx & predIdx == posIdx);
        fn = sum(trueIdx == posIdx & predIdx == negIdx);
        tn = sum(trueIdx == negIdx & predIdx == negIdx);
        fp = sum(trueIdx == negIdx & predIdx == posIdx);

        sens = tp / max(tp + fn, 1);
        spec = tn / max(tn + fp, 1);
        balAcc = 0.5 * (sens + spec);
        if balAcc > bestBalancedAcc
            bestBalancedAcc = balAcc;
            threshold = t;
        end
    end
end

function metrics = computeThresholdMetrics(trueIdx, posProb, threshold, confidence, entropyVals, isOODVals, classNames)
    posIdx = inferPositiveClassIdx(classNames);
    negIdx = 3 - posIdx;
    predIdx = negIdx * ones(size(posProb));
    predIdx(posProb >= threshold) = posIdx;

    probs = zeros(numel(posProb), numel(classNames));
    probs(:, posIdx) = posProb;
    probs(:, negIdx) = 1 - posProb;
    metrics = computePredictionMetrics(trueIdx, predIdx, probs, confidence, entropyVals, isOODVals, classNames);
end

function saveProbabilityDiagnosticsFigure(idPredictions, oodPredictions, classNames, outPath)
    posIdx = inferPositiveClassIdx(classNames);
    fig = figure('Visible', 'off', 'Position', [100 100 1200 760]);

    subplot(2, 2, 1);
    plotProbabilityByTruth(idPredictions, posIdx, classNames, 'ID Fold-Val');

    subplot(2, 2, 2);
    plotProbabilityByTruth(oodPredictions, posIdx, classNames, 'Held-out TEST');

    subplot(2, 2, 3);
    if isfield(idPredictions, 'probs') && ~isempty(idPredictions.probs)
        histogram(idPredictions.probs(:, posIdx), 20);
    end
    xlabel(sprintf('%s probability', classNames{posIdx}));
    ylabel('Count');
    title('ID Positive-Score Histogram');

    subplot(2, 2, 4);
    if ~isempty(oodPredictions) && isfield(oodPredictions, 'probs') && ~isempty(oodPredictions.probs)
        histogram(oodPredictions.probs(:, posIdx), 20);
    end
    xlabel(sprintf('%s probability', classNames{posIdx}));
    ylabel('Count');
    title('Held-out TEST Positive-Score Histogram');

    saveas(fig, outPath);
    close(fig);
end

function plotProbabilityByTruth(predictions, posIdx, classNames, titleText)
    if ~isfield(predictions, 'probs') || isempty(predictions.probs)
        axis off;
        text(0.1, 0.5, 'No predictions available.', 'FontSize', 12);
        title(titleText);
        return;
    end

    hold on;
    validTrue = isfinite(predictions.trueIdx);
    if any(validTrue)
        negMask = validTrue & predictions.trueIdx ~= posIdx;
        posMask = validTrue & predictions.trueIdx == posIdx;
        if any(negMask)
            scatter(find(negMask), predictions.probs(negMask, posIdx), 18, 'b', 'filled');
        end
        if any(posMask)
            scatter(find(posMask), predictions.probs(posMask, posIdx), 18, 'r', 'filled');
        end
        legend({classNames{3 - posIdx}, classNames{posIdx}}, 'Location', 'best');
    else
        scatter(1:numel(predictions.probs(:, posIdx)), predictions.probs(:, posIdx), 18, 'k', 'filled');
    end
    ylim([0 1]);
    yline(0.5, 'k--');
    xlabel('Sample index');
    ylabel(sprintf('%s probability', classNames{posIdx}));
    title(titleText);
    hold off;
end

function saveEvaluationSummaryFigure(metrics, predictions, classNames, figureTitle, outPath)
    fig = figure('Visible', 'off', 'Position', [100 100 1200 760]);
    numClasses = numel(classNames);
    posIdx = inferPositiveClassIdx(classNames);

    subplot(2, 2, 1);
    cm = metrics.confusion_matrix;
    if all(isfinite(cm(:)))
        imagesc(cm);
        axis image;
        colormap(gca, parula);
        colorbar;
        title('Confusion Matrix');
        set(gca, 'XTick', 1:numClasses, 'XTickLabel', classNames, ...
            'YTick', 1:numClasses, 'YTickLabel', classNames);
        xlabel('Predicted');
        ylabel('True');
        for r = 1:size(cm, 1)
            for c = 1:size(cm, 2)
                text(c, r, sprintf('%d', cm(r, c)), 'HorizontalAlignment', 'center', ...
                    'Color', 'w', 'FontWeight', 'bold');
            end
        end
    else
        axis off;
        text(0.1, 0.5, 'Labels unavailable for confusion matrix.', 'FontSize', 12);
    end

    subplot(2, 2, 2);
    histogram(predictions.probs(:, posIdx), 15);
    xlabel(sprintf('%s probability', classNames{posIdx}));
    ylabel('Count');
    title('Positive-Class Probability');

    subplot(2, 2, 3);
    histogram(predictions.entropy, 15);
    hold on;
    plot([metrics.mean_entropy metrics.mean_entropy], ylim, 'r--', 'LineWidth', 1.2);
    hold off;
    xlabel('Entropy');
    ylabel('Count');
    title('Prediction Uncertainty');

    subplot(2, 2, 4);
    bar([metrics.accuracy, metrics.precision, metrics.sensitivity, metrics.specificity, metrics.f1], 0.6);
    ylim([0 1]);
    set(gca, 'XTickLabel', {'Acc', 'Prec', 'Sens', 'Spec', 'F1'});
    title('Scalar Metrics');
    ylabel('Score');

    annotation(fig, 'textbox', [0.08 0.93 0.84 0.05], 'String', figureTitle, ...
        'LineStyle', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
    saveas(fig, outPath);
    close(fig);
end

function saveGradcamGallery(dlnet, files, labelIDs, predictions, classNames, targetLayer, fcName, config, outPath, galleryTitle, maxSamples)
    if isempty(files)
        return;
    end

    sampleIdx = selectVisualizationIndices(predictions, maxSamples);
    n = numel(sampleIdx);
    posIdx = inferPositiveClassIdx(classNames);
    fig = figure('Visible', 'off', 'Position', [100 100 max(320 * n, 900) 650]);

    for i = 1:n
        idx = sampleIdx(i);
        rawImg = imread(files{idx});
        [displayImg, overlayImg] = buildGradcamVisualization( ...
            dlnet, rawImg, predictions.predIdx(idx), targetLayer, fcName, config);

        subplot(2, n, i);
        imshow(displayImg);
        gtLabel = indexToLabel(labelIDs, idx, classNames);
        predLabel = classNames{predictions.predIdx(idx)};
        title(sprintf('GT:%s | Pred:%s', gtLabel, predLabel), 'Interpreter', 'none', 'FontSize', 8);

        subplot(2, n, i + n);
        imshow(overlayImg);
        title(sprintf('p(%s)=%.2f | H=%.2f | OOD=%d', ...
            classNames{posIdx}, predictions.probs(idx, posIdx), predictions.entropy(idx), predictions.isOOD(idx)), ...
            'Interpreter', 'none', 'FontSize', 8);
    end

    annotation(fig, 'textbox', [0.08 0.93 0.84 0.05], 'String', galleryTitle, ...
        'LineStyle', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
    saveas(fig, outPath);
    close(fig);
end

function sampleIdx = selectVisualizationIndices(predictions, maxSamples)
    n = numel(predictions.files);
    if n <= maxSamples
        sampleIdx = 1:n;
        return;
    end

    lowCount = floor(maxSamples / 2);
    highCount = maxSamples - lowCount;
    [~, lowOrder] = sort(predictions.entropy, 'ascend');
    [~, highOrder] = sort(predictions.entropy, 'descend');
    sampleIdx = unique([lowOrder(1:lowCount); highOrder(1:highCount)], 'stable');

    if numel(sampleIdx) < maxSamples
        remaining = setdiff((1:n)', sampleIdx, 'stable');
        sampleIdx = [sampleIdx; remaining(1:(maxSamples - numel(sampleIdx)))];
    elseif numel(sampleIdx) > maxSamples
        sampleIdx = sampleIdx(1:maxSamples);
    end
end

function labelName = indexToLabel(labelIDs, idx, classNames)
    labelName = 'unknown';
    if isempty(labelIDs) || numel(labelIDs) < idx
        return;
    end
    if isfinite(labelIDs(idx)) && labelIDs(idx) >= 1 && labelIDs(idx) <= numel(classNames)
        labelName = classNames{labelIDs(idx)};
    end
end

function [displayImg, overlayImg] = buildGradcamVisualization(dlnet, rawImg, classIdx, targetLayer, fcName, config) %#ok<INUSD>
    img = rawImg;
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    elseif size(img, 3) > 3
        img = img(:, :, 1:3);
    end
    img = imresize(img, [224, 224]);
    displayImg = normalizeForDisplay(img);

    cam = computeSingleImageGradcam(dlnet, img, classIdx, targetLayer, fcName);
    cam = min(max(cam, 0), 1);
    heatmap = ind2rgb(uint8(255 * cam), jet(256));
    overlayImg = 0.55 * displayImg + 0.45 * heatmap;
    overlayImg = min(max(overlayImg, 0), 1);
end

function cam = computeSingleImageGradcam(dlnet, img, classIdx, targetLayer, fcName)
    cam = dlfeval(@singleImageGradcamForward, dlnet, img, classIdx, targetLayer, fcName);
    cam = extractdata(cam);
    if isa(cam, 'gpuArray')
        cam = gather(cam);
    end
    cam = squeeze(cam);
end

function camNorm = singleImageGradcamForward(dlnet, img, classIdx, targetLayer, fcName)
    XData = reshape(single(img), [size(img, 1), size(img, 2), size(img, 3), 1]);
    if networkUsesGPU(dlnet)
        XData = gpuArray(XData);
    end
    X = dlarray(XData, 'SSCB');

    [rawLogits, activations] = forward(dlnet, X, 'Outputs', {fcName, targetLayer});
    logitsVec = reshape(stripdims(rawLogits), [], 1);
    logits = dlarray(logitsVec, 'CB');

    logitsData = extractdata(logitsVec);
    targetVec = zeros(size(logitsData), 'like', logitsData);
    targetVec(classIdx) = 1;
    targetWeights = dlarray(targetVec, 'CB');
    classObjective = sum(logits .* targetWeights, 1);

    grads = dlgradient(classObjective, activations);
    weights = mean(grads, [1 2]);
    cam = relu(sum(activations .* weights, 3));
    cam = dlresize(cam, 'OutputSize', [224, 224]);
    camNorm = cam ./ (max(cam, [], [1 2]) + 1e-6);
end

function tf = networkUsesGPU(dlnet)
    tf = false;
    if isempty(dlnet.Learnables) || isempty(dlnet.Learnables.Value)
        return;
    end
    firstVal = dlnet.Learnables.Value{1};
    if isa(firstVal, 'gpuArray')
        tf = true;
    elseif isa(firstVal, 'dlarray')
        tf = isa(extractdata(firstVal), 'gpuArray');
    end
end

function imgOut = normalizeForDisplay(imgIn)
    imgOut = single(imgIn);
    if max(imgOut(:)) > 1
        imgOut = imgOut / 255;
    end
    imgOut = min(max(imgOut, 0), 1);
end

function grads = clipGradients(grads, maxNorm)
    gradValues = grads.Value;
    gradNormSq = 0;
    for i = 1:numel(gradValues)
        if isempty(gradValues{i})
            continue;
        end
        gradNormSq = gradNormSq + sum(gradValues{i}(:).^2);
    end
    gradNorm = sqrt(gradNormSq + eps);
    if gradNorm > maxNorm
        scale = maxNorm / gradNorm;
        grads.Value = cellfun(@(g) g * scale, gradValues, 'UniformOutput', false);
    end
end

function rowVec = toRowVectorCPU(x)
    if isa(x, 'gpuArray')
        rowVec = gather(x(:)');
    else
        rowVec = x(:)';
    end
end

function [filesOut, labelsOut, masksOut] = shuffleTrainingTriples(filesIn, labelsIn, masksIn)
    numItems = numel(filesIn);
    perm = randperm(numItems);
    filesOut = filesIn(perm);
    labelsOut = labelsIn(perm);
    masksOut = masksIn(perm);
end

function layers = getVGG16Layers(preferPretrained)
    if nargin < 1
        preferPretrained = true;
    end

    if preferPretrained
        try
            netOrLayers = vgg16('Weights', 'imagenet');
        catch
            try
                netOrLayers = vgg16;
            catch
                warning('Pretrained VGG16 weights unavailable. Falling back to untrained VGG16 layers.');
                netOrLayers = vgg16('Weights', 'none');
            end
        end
    else
        netOrLayers = vgg16('Weights', 'none');
    end

    if isa(netOrLayers, 'SeriesNetwork') || isa(netOrLayers, 'DAGNetwork')
        layers = netOrLayers.Layers;
    else
        layers = netOrLayers;
    end
end

function layers = getNetworkLayers(net)
    if isa(net, 'dlnetwork')
        layers = net.Layers;
    elseif isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
        layers = net.Layers;
    else
        layers = net;
    end
end

function fcName = findFCOutputLayer(net)
    layers = getNetworkLayers(net);
    fcIdx = find(arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.FullyConnectedLayer'), layers), 1, 'last');
    if isempty(fcIdx)
        error('Could not locate a fully connected output layer in the provided network.');
    end
    fcName = layers(fcIdx).Name;
end

function layerName = resolveSharedFeatureLayer(studentNetOrLayers, teacherNet, preferredNames)
    studentLayers = getNetworkLayers(studentNetOrLayers);
    teacherLayers = getNetworkLayers(teacherNet);

    studentNames = {studentLayers.Name};
    teacherNames = {teacherLayers.Name};

    for i = 1:numel(preferredNames)
        candidate = preferredNames{i};
        if any(strcmp(studentNames, candidate)) && any(strcmp(teacherNames, candidate))
            layerName = candidate;
            return;
        end
    end

    error('Could not find a shared feature layer. Checked: %s', strjoin(preferredNames, ', '));
end