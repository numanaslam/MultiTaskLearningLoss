function evaluate_trained_model_final(modelFile)
if nargin < 1
    modelFile = 'models/final/final_model_histmatch_kfold_improved_29_04_26.mat';
end

load(modelFile, 'trainedNet', 'config');
fprintf('Model loaded.\n');

imdsROI = imageDatastore('input/roi', 'IncludeSubfolders',true,'LabelSource','foldernames');
imdsCXR = imageDatastore('input/cxr', 'IncludeSubfolders',true,'LabelSource','foldernames');

classes = categories(imdsROI.Labels);
useGPU = canUseGPU;
prep = config.preprocessing_method;
refHist = config.refHist;

fprintf('ID samples: %d | OOD samples: %d\n', numel(imdsROI.Files), numel(imdsCXR.Files));

% ID Evaluation
fprintf('\n=== ID (ROI) Evaluation ===\n');
resID = evaluate_with_ensemble(trainedNet, imdsROI, classes, useGPU, prep, refHist, false);

% OOD Evaluation with Ensemble
fprintf('\n=== OOD (CXR) Evaluation with Preprocessing Ensemble ===\n');
resOOD = evaluate_with_ensemble(trainedNet, imdsCXR, classes, useGPU, prep, refHist, true);

print_comparison(resID, resOOD);
end

function res = evaluate_with_ensemble(net, imds, classes, useGPU, base_prep, refHist, isOOD)
    methods = {'histmatch', 'clahe', 'none'};
    Yprobs = zeros(numel(imds.Files), 1);
    
    for m = 1:length(methods)
        fprintf('   Processing with %s...\n', methods{m});
        probs_m = get_probs(net, imds, classes, useGPU, methods{m}, refHist, isOOD);
        Yprobs = Yprobs + probs_m;
    end
    Yprobs = Yprobs / length(methods);
    
    thresh_res = compute_classification_metrics(Yprobs, imds.Labels, classes);
    
    res = struct('accuracy', thresh_res.accuracy, 'sensitivity', thresh_res.sensitivity, ...
                 'specificity', thresh_res.specificity, 'f1_score', thresh_res.f1_score, ...
                 'auc', thresh_res.auc, 'best_thresh', thresh_res.best_thresh);
    
    fprintf('   → Acc=%.3f  Sens=%.3f  Spec=%.3f  F1=%.3f  AUC=%.3f\n', ...
        res.accuracy, res.sensitivity, res.specificity, res.f1_score, res.auc);
end

function Yprobs = get_probs(net, imds, classes, useGPU, prep_method, refHist, isOOD)
    augDS = augmentedImageDatastore([224 224], imds, 'ColorPreprocessing','gray2rgb');
    preprocessFcn = @(X,T) preprocessMiniBatchWithPreprocessing(X,T,prep_method,refHist);
    mbq = minibatchqueue(augDS, 'MiniBatchSize',16, 'MiniBatchFcn',preprocessFcn, ...
        'MiniBatchFormat',["SSCB",""], 'PartialMiniBatch','return');
    
    Yprobs = [];
    while hasdata(mbq)
        [X,~] = next(mbq);
        if useGPU, X = gpuArray(X); end
        scores = predict(net, X);
        probs = extractdata(scores)';
        Yprobs = [Yprobs; probs(:,end)];
    end
end

function [X, T] = preprocessMiniBatchWithPreprocessing(dataX, dataT, preprocessing_method, refHist)
    X = cat(4, dataX{1:end});
    [H,W,C,B] = size(X);
    X_processed = zeros(H,W,C,B,'uint8');
    
    for b = 1:B
        img = X(:,:,:,b);
        if ~isa(img,'uint8')
            mx = max(img(:));
            if mx<=1.0 && mx>0
                img = uint8(img.*255);
            elseif mx>1.0
                img = uint8(img./mx.*255);
            end
        end
        if C==3, img_gray = rgb2gray(img); else, img_gray = img(:,:,1); end
        
        switch preprocessing_method
            case 'none', img_processed = img_gray;
            case 'clahe', img_processed = adapthisteq(img_gray,'ClipLimit',0.02,'Distribution','uniform');
            case 'histmatch'
                if ~isempty(refHist)
                    img_processed = histeq(img_gray, refHist);
                else
                    img_processed = img_gray;
                end
            otherwise, img_processed = img_gray;
        end
        
        if C==3
            X_processed(:,:,:,b) = repmat(img_processed,[1 1 3]);
        else
            X_processed(:,:,:,b) = img_processed;
        end
    end
    X = dlarray(single(X_processed)./255, 'SSCB');
    T = onehotencode(cat(2, dataT{1:end}), 1);
end


% Keep the same compute_classification_metrics and print_comparison from previous script

% Run it
% evaluate_trained_model_final();

evaluate_trained_model_final('models/final/final_model_histmatch_kfold_improved_29_04_26.mat');