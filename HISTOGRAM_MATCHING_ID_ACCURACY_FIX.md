# Fixing ID Accuracy Drop with Histogram Matching

## Problem Analysis

**Current Results:**
- **Baseline (no preprocessing)**: ID 82.5%, OOD 55.0%, Degradation 35.17%
- **Histogram matching**: ID 77.3%, OOD 57.2%, Degradation 22.86%

**Issue:**
- Histogram matching **reduces ID accuracy by 5.2%** (82.5% → 77.3%)
- This happens because histogram matching is applied to **both ID and OOD** data
- But the model was **trained on ROI images without histogram matching**
- So applying histogram matching to ROI images changes their distribution from training

## Root Cause

1. **Training distribution**: ROI images (original intensity distribution)
2. **ID evaluation**: ROI images + histogram matching → **Distribution mismatch!**
3. **OOD evaluation**: Full CXR + histogram matching → Better alignment

The model expects ROI images with their original intensity distribution, but histogram matching changes this distribution, causing ID accuracy to drop.

## Solutions

### Solution 1: **Apply Histogram Matching Only to OOD Data** ⭐ RECOMMENDED

**Approach:**
- **ID evaluation**: No preprocessing (matches training)
- **OOD evaluation**: Histogram matching (aligns with ROI distribution)

**Expected Results:**
- ID accuracy: ~82.5% (same as baseline, no drop)
- OOD accuracy: ~57.2% (same improvement)
- Degradation: ~30.7% (better than 35.17%, worse than 22.86%)

**Implementation:**
```matlab
% In evaluation script
if strcmp(dataset_type, 'ID')
    preprocessing_method = 'none';  % No preprocessing for ID
else
    preprocessing_method = 'histmatch';  % Histogram matching for OOD
end
```

**Pros:**
- Preserves ID accuracy (no distribution mismatch)
- Still improves OOD performance
- Simple to implement

**Cons:**
- Degradation slightly higher than applying to both (but still better than baseline)

### Solution 2: **Train Model with Histogram Matching** ⭐ BEST LONG-TERM

**Approach:**
- Apply histogram matching during **training** (to ROI images)
- Apply histogram matching during **evaluation** (to both ID and OOD)

**Expected Results:**
- ID accuracy: ~82-85% (model trained with histogram matching)
- OOD accuracy: ~57-60% (consistent preprocessing)
- Degradation: ~22-25% (similar to current, but both metrics improve)

**Implementation:**
- Modify training script to apply histogram matching to training data
- Model learns to work with histogram-matched images
- Both ID and OOD use same preprocessing

**Pros:**
- Consistent preprocessing across training and evaluation
- Model optimized for histogram-matched images
- Best long-term solution

**Cons:**
- Requires retraining the model
- Takes time and computational resources

### Solution 3: **Conditional Preprocessing**

**Approach:**
- Use different preprocessing strategies for ID vs OOD
- ID: Minimal/no preprocessing (preserve training distribution)
- OOD: Histogram matching (align with ROI distribution)

**Expected Results:**
- ID accuracy: ~82.5% (preserved)
- OOD accuracy: ~57.2% (improved)
- Degradation: ~30.7%

**Pros:**
- Best of both worlds
- No retraining needed

**Cons:**
- More complex evaluation pipeline
- Need to track which preprocessing to use

### Solution 4: **Fine-tune Model on Histogram-Matched ROI**

**Approach:**
- Start with pre-trained model
- Fine-tune on histogram-matched ROI images
- Evaluate with histogram matching on both ID and OOD

**Expected Results:**
- ID accuracy: ~80-83% (fine-tuned for histogram matching)
- OOD accuracy: ~58-60% (consistent preprocessing)
- Degradation: ~22-25%

**Pros:**
- Faster than full retraining
- Model adapts to histogram matching

**Cons:**
- Still requires training step
- May need to balance fine-tuning to avoid overfitting

### Solution 5: **Adaptive Preprocessing**

**Approach:**
- Detect if image is ROI or full CXR
- Apply preprocessing only if needed (full CXR)
- Keep ROI images as-is

**Expected Results:**
- ID accuracy: ~82.5% (preserved)
- OOD accuracy: ~57.2% (improved)
- Degradation: ~30.7%

**Pros:**
- Automatic detection
- Optimal preprocessing per image type

**Cons:**
- More complex implementation
- Need reliable ROI vs CXR detection

## Recommended Approach

### **Immediate Fix (No Retraining):**
**Solution 1**: Apply histogram matching only to OOD data

**Implementation Steps:**
1. Modify evaluation script to use different preprocessing for ID vs OOD
2. ID evaluation: `preprocessing_method = 'none'`
3. OOD evaluation: `preprocessing_method = 'histmatch'`

**Expected Improvement:**
- ID accuracy: 82.5% (restored)
- OOD accuracy: 57.2% (maintained)
- Degradation: ~30.7% (better than 35.17%)

### **Long-term Solution:**
**Solution 2**: Train model with histogram matching

**Implementation Steps:**
1. Modify training script to apply histogram matching to training data
2. Retrain model with histogram-matched ROI images
3. Evaluate with histogram matching on both ID and OOD

**Expected Improvement:**
- ID accuracy: 82-85% (model optimized for histogram matching)
- OOD accuracy: 57-60% (consistent preprocessing)
- Degradation: 22-25% (best overall performance)

## Code Changes Needed

### For Immediate Fix (Solution 1):

Modify `evaluate_ood_with_preprocessing.m`:

```matlab
% Evaluate ID (ROI) - NO preprocessing (matches training)
fprintf('  Evaluating ID (ROI)...\n');
[resultsID, ~] = evaluate_with_preprocessing(trainedNet, imdsROI, classes, useGPU, 'none');

% Evaluate OOD (Full CXR) - WITH histogram matching
fprintf('  Evaluating OOD (Full CXR)...\n');
[resultsOOD, ~] = evaluate_with_preprocessing(trainedNet, imdsCXR, classes, useGPU, 'histmatch');
```

### For Long-term Solution (Solution 2):

Modify training script to apply histogram matching during training:
- Apply histogram matching to ROI images in training loop
- Compute reference histogram from training ROI images
- Model learns to work with histogram-matched images

## Summary

**The ID accuracy drop is caused by applying histogram matching to ROI images, which changes their distribution from training.**

**Quick fix**: Apply histogram matching only to OOD data (preserves ID accuracy)

**Best solution**: Train model with histogram matching (optimizes for both ID and OOD)

