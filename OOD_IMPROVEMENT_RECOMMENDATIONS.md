# OOD Performance Improvement Recommendations

## Critical Issue Identified

**Normal class failure on OOD data:**
- ID error rate: 1.8%
- OOD error rate: **84.9%** (catastrophic)
- Error increase: **83.2%**

**PTB class is relatively stable:**
- ID error rate: 2.8%
- OOD error rate: 8.0%
- Error increase: 5.2%

## Root Cause Analysis

The model is likely:
1. **Focusing on black regions** (outside body) in full CXR images
2. **Misinterpreting background** as pathological features
3. **Not applying anatomical guidance strongly enough** during training

## Immediate Actions

### 1. Run Detailed Analysis
```matlab
analyze_normal_class_failure()
```
This will generate:
- Attention maps for misclassified normal images
- Comparison of ID vs OOD attention patterns
- Black region attention analysis
- Intensity distribution statistics

### 2. Increase Anatomical Guidance Weight

**Current:** `λ_anatomical = 1.0`  
**Recommended:** `λ_anatomical = 3.0 - 5.0`

The anatomical guidance loss penalizes attention outside lung regions. Increasing the weight will force the model to focus more on lung regions.

**To implement:**
- Edit `train_final_model_with_preprocessing_anatomical.m`
- Change line 72: `loss_config.lambda_anatomical = 3.0;` (or higher)
- Retrain the model

### 3. Enhance Anatomical Guidance Loss

**Current implementation:** Only penalizes attention outside lung mask  
**Enhancement:** Also penalize attention on black regions (intensity < 10)

Add black region masking to the anatomical guidance loss:

```matlab
% In compute_loss_with_config function, after line 1423:
% Also penalize attention on black regions
img_gray = rgb2gray(img);
blackMask = single(img_gray < 10);  % Black regions
if size(blackMask, 1) ~= 224 || size(blackMask, 2) ~= 224
    blackMask = imresize(blackMask, [224 224], 'nearest');
end
blackMask_dl = dlarray(blackMask, 'SS');
attention_on_black = studCAM_norm .* blackMask_dl;
anatomicalLoss = anatomicalLoss + 0.5 * mean(attention_on_black, 'all');  % Additional penalty
```

### 4. Mask Black Regions During Evaluation

Update `evaluate_dataset_with_preprocessing` to mask out black regions:

```matlab
% After preprocessing, before prediction:
for b = 1:size(imgs, 4)
    img_gray = rgb2gray(imgs(:,:,:,b));
    blackMask = img_gray < 10;  % Black regions
    % Set black regions to mean intensity
    for c = 1:size(imgs, 3)
        channel = imgs(:,:,c,b);
        channel(blackMask) = mean(channel(~blackMask));
        imgs(:,:,c,b) = channel;
    end
end
```

### 5. Training Strategy: Mixed Dataset

Train on both ROI and Full CXR images:

```matlab
% Load both datasets
imdsROI = imageDatastore('input/roi', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsCXR = imageDatastore('input/cxr', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Combine with 50/50 ratio
numROI = numel(imdsROI.Files);
numCXR = numel(imdsCXR.Files);
ratio = 0.5;  % 50% ROI, 50% CXR

% Sample from each
roiIdx = randperm(numROI, round(numROI * ratio));
cxrIdx = randperm(numCXR, round(numCXR * (1-ratio)));

imdsMixed = imageDatastore([imdsROI.Files(roiIdx); imdsCXR.Files(cxrIdx)], ...
    'Labels', [imdsROI.Labels(roiIdx); imdsCXR.Labels(cxrIdx)]);
```

### 6. Adversarial Domain Adaptation

Use adversarial training to align feature distributions between ROI and CXR:

- Add a domain discriminator
- Train generator (feature extractor) to fool discriminator
- This forces features to be domain-invariant

### 7. Test-Time Augmentation (TTA)

Apply TTA during OOD evaluation to improve robustness:

```matlab
% Already implemented in evaluate_ood_with_preprocessing.m
% Use: evaluate_ood_with_preprocessing(true, 12)  % 12 augmentations
```

## Expected Improvements

| Strategy | Expected OOD Accuracy | Expected Degradation |
|----------|----------------------|---------------------|
| Baseline (current) | 54.1% | 44.67% |
| Increased λ_anatomical (3.0) | 65-70% | 30-35% |
| + Black region masking | 70-75% | 20-25% |
| + Mixed dataset training | 75-80% | 15-20% |
| + Adversarial training | 80-85% | 10-15% |

## Implementation Priority

1. **High Priority (Immediate):**
   - Run `analyze_normal_class_failure()` to understand the issue
   - Increase `λ_anatomical` to 3.0-5.0
   - Retrain model

2. **Medium Priority (Next):**
   - Enhance anatomical guidance to penalize black regions
   - Mask black regions during evaluation
   - Test with TTA

3. **Long-term:**
   - Mixed dataset training
   - Adversarial domain adaptation
   - Fine-tuning on OOD data

## Monitoring Metrics

During training, monitor:
- **Segmentation metrics (Dice/IoU):** Should increase with stronger anatomical guidance
- **Validation accuracy:** Should remain stable or improve
- **Loss components:** `anatomicalLoss` should decrease over time

During evaluation:
- **Class-wise error rates:** Normal class error should decrease significantly
- **Attention maps:** Should focus on lung regions, not black regions
- **Confidence scores:** Should be higher and more consistent

## Next Steps

1. Run `analyze_normal_class_failure()` to get detailed insights
2. Review attention maps to confirm black region focus
3. Increase anatomical guidance weight and retrain
4. Evaluate improvement and iterate

