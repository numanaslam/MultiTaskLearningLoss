function evaluate_saved_model(modelPath)
%EVALUATE_SAVED_MODEL Evaluate a saved model on ID and OOD datasets
%   Usage: evaluate_saved_model('models/checkpoints/checkpoint_histmatch_fold_1.mat')

if nargin < 1
    [file, path] = uigetfile('*.mat', 'Select Model File');
    if isequal(file, 0)
        fprintf('No file selected. Exiting.\n');
        return;
    end
    modelPath = fullfile(path, file);
end

fprintf('=== EVALUATING SAVED MODEL ===\n');
fprintf('Loading model from: %s\n', modelPath);

% Load model
data = load(modelPath);

% Handle different variable names (checkpoint vs final model)
if isfield(data, 'trainedNet')
    net = data.trainedNet;
elseif isfield(data, 'fold_model')
    net = data.fold_model;
else
    error('Could not find network variable (trainedNet or fold_model) in file.');
end

if isfield(data, 'config')
    config = data.config;
    preprocessing_method = config.preprocessing_method;
    refHist = config.refHist;
    useGPU = config.useGPU;
else
    warning('Config not found in file. Using defaults.');
    preprocessing_method = 'histmatch'; % Guessing default
    refHist = [];
    useGPU = canUseGPU();
end

if isfield(data, 'classes')
    classes = data.classes;
else
    % Try to infer classes from network output
    try
        classes = net.Layers(end).Classes;
    catch
        warning('Classes not found. Assuming ["Normal", "PTB"].');
        classes = categorical({'Normal', 'PTB'});
    end
end

fprintf('Model loaded successfully.\n');
fprintf('Preprocessing: %s\n', preprocessing_method);

%% Load Datasets
fprintf('\n--- Loading Datasets ---\n');

% 1. Load In-Distribution (ROI) Data
% We need to find the input directory. Assuming standard structure.
baseDir = pwd;
inputDir = fullfile(baseDir, 'input');
roiDir = fullfile(inputDir, 'roi');

if ~exist(roiDir, 'dir')
    % Try to find it relative to script
    scriptPath = fileparts(mfilename('fullpath'));
    roiDir = fullfile(scriptPath, 'input', 'roi');
end

if exist(roiDir, 'dir')
    fprintf('Loading ROI dataset (ID) from: %s\n', roiDir);
    imdsROI = imageDatastore(roiDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    fprintf('  Found %d images.\n', numel(imdsROI.Files));
else
    error('ROI directory not found at %s', roiDir);
end

% 2. Load Out-of-Distribution (CXR) Data
cxrDir = fullfile(inputDir, 'cxr');
if ~exist(cxrDir, 'dir')
    cxrDir = fullfile(scriptPath, 'input', 'cxr');
end

if exist(cxrDir, 'dir')
    fprintf('Loading CXR dataset (OOD) from: %s\n', cxrDir);
    imdsCXR = imageDatastore(cxrDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    fprintf('  Found %d images.\n', numel(imdsCXR.Files));
else
    warning('CXR directory not found at %s. Skipping OOD evaluation.', cxrDir);
    imdsCXR = [];
end

%% Evaluate
fprintf('\n--- Starting Evaluation ---\n');

% Evaluate ID
fprintf('Evaluating In-Distribution (ROI)...\n');
[resultsID, ~] = evaluate_dataset_with_preprocessing(net, imdsROI, classes, ...
    useGPU, preprocessing_method, refHist);
print_metrics('ID (ROI)', resultsID);

% Evaluate OOD
if ~isempty(imdsCXR)
    fprintf('\nEvaluating Out-of-Distribution (CXR)...\n');
    [resultsOOD, ~] = evaluate_dataset_with_preprocessing(net, imdsCXR, classes, ...
        useGPU, preprocessing_method, refHist);
    print_metrics('OOD (CXR)', resultsOOD);
    
    % Calculate Degradation
    degradation = calculate_degradation(resultsID, resultsOOD);
    print_degradation(degradation);
end

end

function print_metrics(name, results)
    fprintf('  %s Performance:\n', name);
    fprintf('    Accuracy: %.3f\n', results.accuracy);
    fprintf('    Precision: %.3f\n', results.precision);
    fprintf('    Sensitivity: %.3f\n', results.sensitivity);
    fprintf('    Specificity: %.3f\n', results.specificity);
    fprintf('    F1-Score: %.3f\n', results.f1_score);
    fprintf('    AUC: %.3f\n', results.auc);
end

function degradation = calculate_degradation(id, ood)
    metrics = {'accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score', 'auc'};
    degradation = struct();
    degradation.mean = 0;
    
    for i = 1:numel(metrics)
        m = metrics{i};
        deg = (id.(m) - ood.(m)) / id.(m) * 100;
        degradation.(m) = deg;
        degradation.mean = degradation.mean + deg;
    end
    degradation.mean = degradation.mean / numel(metrics);
end

function print_degradation(deg)
    fprintf('\n  Performance Degradation:\n');
    fprintf('    Accuracy: %.2f%%\n', deg.accuracy);
    fprintf('    Mean Degradation: %.2f%%\n', deg.mean);
    
    if deg.mean < 5
        fprintf('    Status: ✓ Excellent generalization (<5%% degradation)\n');
    elseif deg.mean < 15
        fprintf('    Status: ✓ Good generalization (<15%% degradation)\n');
    elseif deg.mean < 30
        fprintf('    Status: ⚠ Moderate degradation (15-30%%)\n');
    else
        fprintf('    Status: ✗ High degradation (>30%%)\n');
    end
end

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
