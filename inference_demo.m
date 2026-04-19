function inference_demo(imagePath)
%INFERENCE_DEMO Run prediction on a single image without ground truth
%   This script demonstrates how to use the trained model on a new image
%   where no mask or label is available.
%
%   Usage:
%       inference_demo('path/to/your/image.png')

if nargin < 1
    % Default to a sample image if none provided
    imageFiles = dir(fullfile('input', 'roi', '**', '*.png'));
    if ~isempty(imageFiles)
        imagePath = fullfile(imageFiles(1).folder, imageFiles(1).name);
    else
        error('Please provide an image path.');
    end
end

fprintf('=== INFERENCE DEMO ===\n');
fprintf('Image: %s\n', imagePath);

%% 1. Load Model
modelFile = fullfile('models', 'final', 'final_model_histmatch_kfold.mat');
if ~exist(modelFile, 'file')
    error('Model file not found: %s', modelFile);
end

fprintf('Loading model...\n');
s = load(modelFile);
net = s.trainedNet;
config = s.config;
classes = {'Normal', 'Tuberculosis'}; % Assuming these are the classes

% Check for GPU
if canUseGPU
    useGPU = true;
    fprintf('Using GPU for inference.\n');
else
    useGPU = false;
    fprintf('Using CPU for inference.\n');
end

%% 2. Load and Preprocess Image
fprintf('Preprocessing image...\n');
img = imread(imagePath);

% Resize to model input size
inputSize = [224 224];
imgResized = imresize(img, inputSize);

% Convert to grayscale if needed (for histmatch)
if size(imgResized, 3) == 3
    imgGray = rgb2gray(imgResized);
else
    imgGray = imgResized;
end

% Apply Histogram Matching (Crucial for OOD)
% We need the reference histogram from the training target domain (CXR)
% In a real deployment, this histogram should be saved with the model.
% For this demo, we'll recompute it from the CXR folder if available,
% or use the image itself (fallback, less optimal).

cxrDir = fullfile('input', 'cxr');
if exist(cxrDir, 'dir')
    % Quick reference computation (should be cached in production)
    cxrFiles = dir(fullfile(cxrDir, '**', '*.png'));
    if ~isempty(cxrFiles)
        numRef = min(10, numel(cxrFiles));
        refPixels = [];
        for i = 1:numRef
            rImg = imread(fullfile(cxrFiles(i).folder, cxrFiles(i).name));
            if size(rImg, 3) > 1, rImg = rgb2gray(rImg); end
            refPixels = [refPixels; rImg(:)];
        end
        refHist = imhist(uint8(refPixels));
        imgProcessed = histeq(imgGray, refHist);
        fprintf('  Applied Histogram Matching (using CXR reference).\n');
    else
        imgProcessed = imgGray;
        fprintf('  Skipped Histogram Matching (CXR images not found).\n');
    end
else
    imgProcessed = imgGray;
    fprintf('  Skipped Histogram Matching (CXR directory not found).\n');
end

% Replicate to RGB for VGG16
imgRGB = repmat(imgProcessed, [1 1 3]);

% Convert to dlarray
imgSingle = single(imgRGB);
if useGPU
    dlImg = dlarray(gpuArray(imgSingle), 'SSCB');
else
    dlImg = dlarray(imgSingle, 'SSCB');
end

%% 3. Run Prediction
fprintf('Running prediction...\n');
scores = predict(net, dlImg);
probs = extractdata(scores);

if useGPU
    probs = gather(probs);
end

[maxProb, predIdx] = max(probs);
predLabel = classes{predIdx};

fprintf('\n=== RESULTS ===\n');
fprintf('Prediction: %s\n', predLabel);
fprintf('Confidence: %.2f%%\n', maxProb * 100);
fprintf('Scores:\n');
for i = 1:numel(classes)
    fprintf('  %s: %.4f\n', classes{i}, probs(i));
end

%% 4. Visualize (Optional)
% Generate GradCAM to show where the model is looking
try
    scoreMap = gradCAM(net, dlImg, predLabel);
    
    figure;
    subplot(1, 2, 1);
    imshow(imgResized);
    title('Input Image');
    
    subplot(1, 2, 2);
    imshow(imgResized);
    hold on;
    imagesc(scoreMap, 'AlphaData', 0.5);
    colormap jet;
    hold off;
    title(sprintf('Prediction: %s (%.1f%%)', predLabel, maxProb*100));
    
    fprintf('\nVisualization displayed.\n');
catch
    fprintf('\nCould not generate GradCAM visualization.\n');
end

end
