function train_vgg16_domain_baselines(config)
%TRAIN_VGG16_DOMAIN_BASELINES Train VGG16 teacher baselines on ROI, CXR, and mixed domains.
%   This creates three teacher models using the same paired train/validation split:
%     1. ROI-only
%     2. Full-CXR-only
%     3. Mixed ROI + Full-CXR
%
%   Usage:
%       train_vgg16_domain_baselines
%       train_vgg16_domain_baselines(struct('max_epochs', 40, 'train_fraction', 0.9))
%
%   Saved models:
%       models/pretrained/vgg16_finetuned_on_roi.mat
%       models/pretrained/vgg16_finetuned_on_cxr.mat
%       models/pretrained/vgg16_finetuned_on_mixed.mat

if nargin < 1
    config = struct();
end

config = populate_default_config(config);

clc;
close all;

fprintf('=== VGG16 DOMAIN BASELINE TRAINING ===\n');
fprintf('ROI dir: %s\n', config.roi_dir);
fprintf('CXR dir: %s\n', config.cxr_dir);
fprintf('Output dir: %s\n', config.output_dir);
fprintf('Train fraction: %.2f\n', config.train_fraction);
fprintf('Random seed: %d\n', config.random_seed);
fprintf('Domains: %s\n\n', strjoin(config.train_domains, ', '));

rng(config.random_seed);

manifest = build_paired_manifest(config.roi_dir, config.cxr_dir);
[trainManifest, valManifest] = split_manifest_stratified(manifest, config.train_fraction, config.random_seed);

classNames = categories(categorical(manifest.label));
fprintf('Paired samples found: %d\n', height(manifest));
for c = 1:numel(classNames)
    classMask = strcmp(manifest.label, classNames{c});
    fprintf('  %s: %d pairs\n', classNames{c}, sum(classMask));
end
fprintf('Train pairs: %d | Val pairs: %d\n\n', height(trainManifest), height(valManifest));

baseNet = vgg16;
summary = struct([]);

for d = 1:numel(config.train_domains)
    domainName = lower(string(config.train_domains{d}));
    fprintf('\n=== Training domain: %s ===\n', upper(char(domainName)));

    domainSpec = create_domain_spec(domainName, trainManifest, valManifest, classNames);
    fprintf('  Train images: %d\n', numel(domainSpec.train_files));
    fprintf('  Val images: %d\n', numel(domainSpec.val_files));

    [trainedNet, trainInfo, metrics, confusionMatrix] = train_single_domain_model(baseNet, domainSpec, classNames, config);

    modelMeta = struct();
    modelMeta.domain_name = char(domainName);
    modelMeta.train_pair_count = height(trainManifest);
    modelMeta.val_pair_count = height(valManifest);
    modelMeta.train_image_count = numel(domainSpec.train_files);
    modelMeta.val_image_count = numel(domainSpec.val_files);
    modelMeta.class_names = classNames;
    modelMeta.train_fraction = config.train_fraction;
    modelMeta.random_seed = config.random_seed;
    modelMeta.train_sources = domainSpec.train_sources;
    modelMeta.val_sources = domainSpec.val_sources;
    modelMeta.metrics = metrics;
    modelMeta.confusion_matrix = confusionMatrix;
    modelMeta.split_summary = struct( ...
        'train_keys', {trainManifest.key}, ...
        'val_keys', {valManifest.key});

    modelFile = fullfile(config.output_dir, sprintf('vgg16_finetuned_on_%s.mat', char(domainName)));
    save(modelFile, 'trainedNet', 'trainInfo', 'modelMeta', 'config', '-v7.3');
    fprintf('  Saved model: %s\n', modelFile);

    summary(end + 1).domain = char(domainName); %#ok<AGROW>
    summary(end).model_file = modelFile;
    summary(end).accuracy = metrics.accuracy;
    summary(end).precision = metrics.precision;
    summary(end).sensitivity = metrics.sensitivity;
    summary(end).specificity = metrics.specificity;
    summary(end).f1_score = metrics.f1_score;
    summary(end).auc = metrics.auc;
end

summaryFile = fullfile(config.output_dir, 'vgg16_domain_baseline_summary.mat');
save(summaryFile, 'summary', 'manifest', 'trainManifest', 'valManifest', 'config', '-v7.3');

fprintf('\n=== Training complete ===\n');
for i = 1:numel(summary)
    fprintf('  %s: Acc=%.3f Sens=%.3f Spec=%.3f AUC=%.3f\n', ...
        upper(summary(i).domain), summary(i).accuracy, summary(i).sensitivity, ...
        summary(i).specificity, summary(i).auc);
end
fprintf('Summary saved to: %s\n', summaryFile);
end

function config = populate_default_config(config)
    defaults = struct();
    defaults.roi_dir = fullfile(pwd, 'input', 'roi');
    defaults.cxr_dir = fullfile(pwd, 'input', 'cxr');
    defaults.output_dir = fullfile(pwd, 'models', 'pretrained');
    defaults.train_domains = {'roi', 'cxr', 'mixed'};
    defaults.train_fraction = 0.90;
    defaults.random_seed = 42;
    defaults.input_size = [224 224 3];
    defaults.batch_size = 32;
    defaults.max_epochs = 40;
    defaults.initial_learn_rate = 1e-4;
    defaults.momentum = 0.9;
    defaults.weight_decay = 1e-4;
    defaults.learn_rate_drop_factor = 0.2;
    defaults.learn_rate_drop_period = 12;
    defaults.validation_frequency = [];
    defaults.show_training_plot = true;
    defaults.augmenter = imageDataAugmenter( ...
        'RandXTranslation', [-8 8], ...
        'RandYTranslation', [-8 8], ...
        'RandRotation', [-7 7], ...
        'RandScale', [0.95 1.05]);

    defaultFields = fieldnames(defaults);
    for i = 1:numel(defaultFields)
        fieldName = defaultFields{i};
        if ~isfield(config, fieldName) || isempty(config.(fieldName))
            config.(fieldName) = defaults.(fieldName);
        end
    end

    if ~exist(config.roi_dir, 'dir')
        error('ROI directory not found: %s', config.roi_dir);
    end
    if ~exist(config.cxr_dir, 'dir')
        error('CXR directory not found: %s', config.cxr_dir);
    end
    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end
end

function manifest = build_paired_manifest(roiDir, cxrDir)
    roiEntries = build_domain_index(roiDir);
    cxrEntries = build_domain_index(cxrDir);

    if isempty(roiEntries)
        error('No ROI images found in %s', roiDir);
    end
    if isempty(cxrEntries)
        error('No CXR images found in %s', cxrDir);
    end

    roiKeys = {roiEntries.key};
    cxrKeys = {cxrEntries.key};
    sharedKeys = intersect(roiKeys, cxrKeys, 'stable');

    if isempty(sharedKeys)
        error('No paired ROI/CXR filenames were found across the two domains.');
    end

    if numel(sharedKeys) < numel(roiKeys)
        fprintf('Warning: %d ROI images have no CXR match and will be skipped.\n', numel(roiKeys) - numel(sharedKeys));
    end
    if numel(sharedKeys) < numel(cxrKeys)
        fprintf('Warning: %d CXR images have no ROI match and will be skipped.\n', numel(cxrKeys) - numel(sharedKeys));
    end

    roiMap = containers.Map(roiKeys, num2cell(1:numel(roiEntries)));
    cxrMap = containers.Map(cxrKeys, num2cell(1:numel(cxrEntries)));

    labels = cell(numel(sharedKeys), 1);
    roiFiles = cell(numel(sharedKeys), 1);
    cxrFiles = cell(numel(sharedKeys), 1);

    for i = 1:numel(sharedKeys)
        key = sharedKeys{i};
        roiEntry = roiEntries(roiMap(key));
        cxrEntry = cxrEntries(cxrMap(key));
        labels{i} = roiEntry.label;
        roiFiles{i} = roiEntry.file;
        cxrFiles{i} = cxrEntry.file;
    end

    manifest = table(sharedKeys(:), labels, roiFiles, cxrFiles, ...
        'VariableNames', {'key', 'label', 'roi_file', 'cxr_file'});
end

function entries = build_domain_index(domainDir)
    labelDirs = dir(domainDir);
    labelDirs = labelDirs([labelDirs.isdir]);
    labelDirs = labelDirs(~ismember({labelDirs.name}, {'.', '..'}));

    entries = struct('key', {}, 'label', {}, 'file', {});
    for i = 1:numel(labelDirs)
        labelName = labelDirs(i).name;
        files = list_image_files(fullfile(domainDir, labelName));
        for j = 1:numel(files)
            entries(end + 1).key = fullfile(labelName, files(j).name); %#ok<AGROW>
            entries(end).label = labelName;
            entries(end).file = fullfile(files(j).folder, files(j).name);
        end
    end
end

function files = list_image_files(folderPath)
    exts = {'*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'};
    files = [];
    for i = 1:numel(exts)
        files = [files; dir(fullfile(folderPath, exts{i}))]; %#ok<AGROW>
    end
    if ~isempty(files)
        [~, order] = sort({files.name});
        files = files(order);
    end
end

function [trainManifest, valManifest] = split_manifest_stratified(manifest, trainFraction, randomSeed)
    rng(randomSeed);

    trainRows = [];
    valRows = [];
    classNames = unique(manifest.label, 'stable');

    for i = 1:numel(classNames)
        classMask = strcmp(manifest.label, classNames{i});
        classIdx = find(classMask);
        classIdx = classIdx(randperm(numel(classIdx)));

        numTrain = max(1, floor(trainFraction * numel(classIdx)));
        if numTrain >= numel(classIdx) && numel(classIdx) > 1
            numTrain = numel(classIdx) - 1;
        end

        trainRows = [trainRows; classIdx(1:numTrain)]; %#ok<AGROW>
        if numTrain < numel(classIdx)
            valRows = [valRows; classIdx(numTrain + 1:end)]; %#ok<AGROW>
        end
    end

    trainRows = sort(trainRows);
    valRows = sort(valRows);

    trainManifest = manifest(trainRows, :);
    valManifest = manifest(valRows, :);
end

function domainSpec = create_domain_spec(domainName, trainManifest, valManifest, classNames)
    switch char(domainName)
        case 'roi'
            domainSpec.train_files = trainManifest.roi_file;
            domainSpec.val_files = valManifest.roi_file;
            domainSpec.train_labels = categorical(trainManifest.label, classNames);
            domainSpec.val_labels = categorical(valManifest.label, classNames);
            domainSpec.train_sources = repmat({'roi'}, height(trainManifest), 1);
            domainSpec.val_sources = repmat({'roi'}, height(valManifest), 1);

        case 'cxr'
            domainSpec.train_files = trainManifest.cxr_file;
            domainSpec.val_files = valManifest.cxr_file;
            domainSpec.train_labels = categorical(trainManifest.label, classNames);
            domainSpec.val_labels = categorical(valManifest.label, classNames);
            domainSpec.train_sources = repmat({'cxr'}, height(trainManifest), 1);
            domainSpec.val_sources = repmat({'cxr'}, height(valManifest), 1);

        case 'mixed'
            domainSpec.train_files = [trainManifest.roi_file; trainManifest.cxr_file];
            domainSpec.val_files = [valManifest.roi_file; valManifest.cxr_file];
            domainSpec.train_labels = categorical([trainManifest.label; trainManifest.label], classNames);
            domainSpec.val_labels = categorical([valManifest.label; valManifest.label], classNames);
            domainSpec.train_sources = [repmat({'roi'}, height(trainManifest), 1); repmat({'cxr'}, height(trainManifest), 1)];
            domainSpec.val_sources = [repmat({'roi'}, height(valManifest), 1); repmat({'cxr'}, height(valManifest), 1)];

        otherwise
            error('Unsupported domain name: %s', char(domainName));
    end
end

function [trainedNet, trainInfo, metrics, cm] = train_single_domain_model(baseNet, domainSpec, classNames, config)
    imdsTrain = make_labeled_imds(domainSpec.train_files, domainSpec.train_labels);
    imdsVal = make_labeled_imds(domainSpec.val_files, domainSpec.val_labels);

    augTrain = augmentedImageDatastore(config.input_size(1:2), imdsTrain, ...
        'ColorPreprocessing', 'gray2rgb', ...
        'DataAugmentation', config.augmenter);
    augVal = augmentedImageDatastore(config.input_size(1:2), imdsVal, ...
        'ColorPreprocessing', 'gray2rgb');

    lgraph = create_vgg16_transfer_graph(baseNet, numel(classNames));

    valFrequency = config.validation_frequency;
    if isempty(valFrequency)
        valFrequency = max(1, floor(numel(domainSpec.train_files) / config.batch_size));
    end

    if config.show_training_plot
        plotMode = 'training-progress';
    else
        plotMode = 'none';
    end

    options = trainingOptions('sgdm', ...
        'MiniBatchSize', config.batch_size, ...
        'MaxEpochs', config.max_epochs, ...
        'InitialLearnRate', config.initial_learn_rate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', config.learn_rate_drop_factor, ...
        'LearnRateDropPeriod', config.learn_rate_drop_period, ...
        'Momentum', config.momentum, ...
        'L2Regularization', config.weight_decay, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augVal, ...
        'ValidationFrequency', valFrequency, ...
        'Verbose', true, ...
        'VerboseFrequency', valFrequency, ...
        'ExecutionEnvironment', 'auto', ...
        'Plots', plotMode);

    [trainedNet, trainInfo] = trainNetwork(augTrain, lgraph, options);

    reset(augVal);
    [predLabels, scores] = classify(trainedNet, augVal);
    trueLabels = imdsVal.Labels;

    cm = confusionmat(trueLabels, predLabels, 'Order', categorical(classNames));
    metrics = compute_binary_metrics(trueLabels, predLabels, scores, classNames, cm);
end

function imds = make_labeled_imds(files, labels)
    imds = imageDatastore(files);
    imds.Labels = labels;
end

function lgraph = create_vgg16_transfer_graph(net, numClasses)
    lgraph = layerGraph(net);
    lgraph = removeLayers(lgraph, {'fc8', 'prob', 'output'});

    newLayers = [
        fullyConnectedLayer(numClasses, 'Name', 'fc8', ...
            'WeightLearnRateFactor', 10, ...
            'BiasLearnRateFactor', 10)
        softmaxLayer('Name', 'prob')
        classificationLayer('Name', 'output')];

    lgraph = addLayers(lgraph, newLayers);
    lgraph = connectLayers(lgraph, 'drop7', 'fc8');
end

function metrics = compute_binary_metrics(trueLabels, predLabels, scores, classNames, cm)
    posIdx = infer_positive_class_index(classNames);
    negIdx = 3 - posIdx;

    TP = cm(posIdx, posIdx);
    FN = cm(posIdx, negIdx);
    TN = cm(negIdx, negIdx);
    FP = cm(negIdx, posIdx);

    metrics = struct();
    metrics.accuracy = mean(predLabels == trueLabels);
    metrics.precision = TP / (TP + FP + eps);
    metrics.sensitivity = TP / (TP + FN + eps);
    metrics.specificity = TN / (TN + FP + eps);
    metrics.f1_score = 2 * metrics.precision * metrics.sensitivity / (metrics.precision + metrics.sensitivity + eps);

    try
        posScores = scores(:, posIdx);
        aucLabels = double(trueLabels == classNames{posIdx});
        [~, ~, ~, metrics.auc] = perfcurve(aucLabels, posScores, 1);
    catch
        metrics.auc = NaN;
    end
end

function posIdx = infer_positive_class_index(classNames)
    posIdx = 2;
    ptbCandidates = {'PTB', 'ptb', 'TB', 'tb', 'Tuberculosis', 'tuberculosis', 'positive', 'Positive'};

    for i = 1:numel(ptbCandidates)
        idx = find(strcmp(classNames, ptbCandidates{i}), 1);
        if ~isempty(idx)
            posIdx = idx;
            return;
        end
    end

    if numel(classNames) == 2
        if strcmpi(classNames{1}, 'normal')
            posIdx = 2;
        elseif strcmpi(classNames{2}, 'normal')
            posIdx = 1;
        end
    end
end
