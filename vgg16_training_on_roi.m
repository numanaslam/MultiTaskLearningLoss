clc; clear; close all;

% Load pretrained VGG16
% vgg16Path = 'vgg16.mat';
% vggData = load(vgg16Path);
% if isfield(vggData.ans, 'net')
%     net = vggData.net;
% elseif isfield(vggData.ans, 'vggData')
%     net = vggData.netTransfer;
% else
%     error('Could not find VGG16 network in the .mat file.');
% end
net = vgg16;
% Prepare ROI dataset
dataDir = fullfile(pwd, 'input', 'roi');
imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split into training and validation
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.9, 'randomized');

% Resize images to VGG16 input size
inputSize = [224 224 3];
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsVal   = augmentedImageDatastore(inputSize(1:2), imdsVal,   'ColorPreprocessing', 'gray2rgb');

% Modify VGG16 for transfer learning
lgraph = layerGraph(net);

% Remove old classification layers
lgraph = removeLayers(lgraph, {'fc8','prob','output'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc8')
    softmaxLayer('Name', 'prob')
    classificationLayer('Name', 'output')];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'drop7', 'fc8');

% Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% Save the trained model
save('vgg16_finetuned_on_roi.mat', 'trainedNet');
disp('Training complete. Model saved as vgg16_finetuned_on_roi.mat');