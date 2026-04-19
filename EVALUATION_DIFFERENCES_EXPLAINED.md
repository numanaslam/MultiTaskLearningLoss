# Why Evaluation Results Differ Between Training and Analysis Scripts

## Summary

The two scripts use **different evaluation methods**, causing different results:

| Aspect | Training Script | Analysis Script |
|--------|----------------|----------------|
| **ID Dataset** | Validation set from best fold (~113 samples) | Full ROI dataset (566 samples) |
| **OOD Dataset** | Full CXR dataset (566 samples) | Full CXR dataset (566 samples) |
| **Preprocessing** | ✅ **YES** - Histogram matching (`histmatch`) | ❌ **NO** - No preprocessing |
| **Evaluation Function** | `evaluate_dataset_with_preprocessing` | `evaluate_with_confidence` |
| **ID Accuracy** | 90.4% | 97.7% |
| **OOD Accuracy** | 61.1% | 54.1% |
| **Degradation** | 32.28% | 44.67% |

## Key Differences

### 1. **Preprocessing Mismatch** ⚠️ **CRITICAL**

**Training Script (`train_final_model_with_preprocessing_anatomical.m`):**
```matlab
% Line 232-233: Applies histogram matching during evaluation
[resultsID, ~] = evaluate_dataset_with_preprocessing(trainedNet, imdsROI_val, classes, ...
    config.useGPU, preprocessing_method_param, refHist);  % preprocessing_method_param = 'histmatch'
```

**Analysis Script (`analyze_ood_performance_degradation.m`):**
```matlab
% Line 200: NO preprocessing applied
[predID, confID, trueID] = evaluate_with_confidence(net, imdsROI, classes, useGPU);
% Function just resizes and converts to RGB, no histogram matching
```

**Impact:**
- Model was **trained WITH histogram matching**
- Analysis script evaluates **WITHOUT preprocessing**
- This causes a **distribution mismatch** → worse OOD performance (54.1% vs 61.1%)

### 2. **Different ID Evaluation Sets**

**Training Script:**
- Uses validation set from best fold: `subset(imds, bestFoldValIdx)` (~113 samples)
- This is a **subset** of the full ROI dataset
- More conservative estimate

**Analysis Script:**
- Uses **full ROI dataset** (566 samples)
- Includes training data → higher accuracy (97.7% vs 90.4%)

### 3. **Different Evaluation Functions**

**Training Script:**
- `evaluate_dataset_with_preprocessing()`:
  - Applies histogram matching
  - Uses `augmentedImageDatastore` with preprocessing
  - Returns structured results

**Analysis Script:**
- `evaluate_with_confidence()`:
  - No preprocessing
  - Simple resizing and RGB conversion
  - Returns predictions and confidences

## Why This Matters

1. **Preprocessing Consistency:**
   - Model expects preprocessed images (histogram-matched)
   - Evaluating without preprocessing causes distribution shift
   - This explains why OOD accuracy is lower in analysis (54.1% vs 61.1%)

2. **ID Accuracy Difference:**
   - Training script uses validation set (unseen during training)
   - Analysis script uses full dataset (includes training data)
   - This explains higher ID accuracy in analysis (97.7% vs 90.4%)

## Solution: Fix the Analysis Script

Update `analyze_ood_performance_degradation.m` to use preprocessing:

```matlab
% In analyze_prediction_confidence function, replace:
[predID, confID, trueID] = evaluate_with_confidence(net, imdsROI, classes, useGPU);
[predOOD, confOOD, trueOOD] = evaluate_with_confidence(net, imdsCXR, classes, useGPU);

% With:
% Load preprocessing config from model file
if isfield(s, 'results') && isfield(s.results, 'preprocessing_method')
    preprocessing_method = s.results.preprocessing_method;
    refHist = [];
    if strcmp(preprocessing_method, 'histmatch')
        % Compute reference histogram (same as training)
        % ... (code to compute refHist)
    end
    [resultsID, predStructID] = evaluate_dataset_with_preprocessing(...
        net, imdsROI, classes, useGPU, preprocessing_method, refHist);
    [resultsOOD, predStructOOD] = evaluate_dataset_with_preprocessing(...
        net, imdsCXR, classes, useGPU, preprocessing_method, refHist);
    
    predID = predStructID.Ypred;
    confID = predStructID.Yprobs;
    trueID = predStructID.Ytrue;
    
    predOOD = predStructOOD.Ypred;
    confOOD = predStructOOD.Yprobs;
    trueOOD = predStructOOD.Ytrue;
else
    % Fallback to original method
    [predID, confID, trueID] = evaluate_with_confidence(net, imdsROI, classes, useGPU);
    [predOOD, confOOD, trueOOD] = evaluate_with_confidence(net, imdsCXR, classes, useGPU);
end
```

## Expected Results After Fix

After fixing the analysis script to use preprocessing:
- **ID Accuracy:** Should match training script (~90-95%)
- **OOD Accuracy:** Should match training script (~61%)
- **Degradation:** Should match training script (~32%)

## Recommendation

**Always use the same preprocessing during evaluation as during training** to ensure fair comparison and accurate performance metrics.

