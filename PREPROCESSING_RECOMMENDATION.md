# Preprocessing Method Recommendation for Training

## Current Results Analysis

**Training with `histmatch`:**
- ID Accuracy: 85.1%
- OOD Accuracy: 57.5%
- Degradation: 32.0%

## Preprocessing Methods Comparison

### Available Methods:
1. **'none'** - No preprocessing (baseline)
2. **'histmatch'** - Histogram matching to ROI reference
3. **'zscore'** - Z-score normalization
4. **'minmax'** - Min-max normalization
5. **'clahe'** - CLAHE (Contrast Limited Adaptive Histogram Equalization)
6. **'clahe_zscore'** - CLAHE + Z-score

## Recommendation: **'histmatch'** or **'zscore'**

### Option 1: **'histmatch'** (Current Choice) ✅

**Pros:**
- ✅ Directly matches CXR histogram to ROI histogram
- ✅ Reduces intensity distribution shift (2.2x → ~1.0x)
- ✅ Preserves relative intensity relationships
- ✅ Already implemented and tested
- ✅ Best for aligning ID and OOD intensity distributions

**Cons:**
- ⚠️ Still 32% degradation (content shift remains)
- ⚠️ Requires reference histogram computation

**Best for:**
- When you want to minimize intensity distribution shift
- When training and test data have different intensity characteristics
- Current setup (already working)

### Option 2: **'zscore'** (Alternative) ⭐

**Pros:**
- ✅ Standardizes intensity distributions (zero mean, unit variance)
- ✅ More robust to outliers than minmax
- ✅ Common in medical imaging
- ✅ No reference needed (per-image normalization)
- ✅ May generalize better across different datasets

**Cons:**
- ⚠️ Doesn't directly match ROI distribution
- ⚠️ May lose some intensity relationships

**Best for:**
- When you want standardized features
- When dealing with multiple datasets
- When you want per-image normalization

### Option 3: **'none'** (Baseline)

**Pros:**
- ✅ Preserves original intensities
- ✅ No preprocessing overhead
- ✅ Baseline for comparison

**Cons:**
- ❌ 2.2x intensity difference between ID and OOD
- ❌ Higher degradation expected (~40-45%)

**Best for:**
- Baseline comparison only
- When original intensities are critical

## Final Recommendation

### **Use 'histmatch' for Training** ✅

**Reasoning:**
1. **Already implemented and tested** - You have working code
2. **Best intensity alignment** - Directly matches CXR to ROI histogram
3. **Current results are reasonable** - 32% degradation is expected given content shift
4. **Consistent with evaluation** - Same preprocessing in training and testing

### **Alternative: Try 'zscore' for Comparison**

If you want to experiment:
- Train a model with `'zscore'` preprocessing
- Compare OOD performance
- Choose the better one

## Important Note

**Preprocessing alone won't solve the 32% degradation!**

The main issue is **content/context shift**:
- ROI: 52% non-zero pixels (cropped lung)
- CXR: 98% non-zero pixels (full chest)
- Different spatial context

**To reduce degradation further, you need:**
1. ✅ Preprocessing (histmatch) - **Already done**
2. ⚠️ **Mixed training** (70% ROI + 30% CXR) - **Recommended next step**
3. ⚠️ Fine-tuning on OOD data
4. ⚠️ Domain adaptation techniques

## Implementation

### Current Training Script:
```matlab
% Line 23 in train_final_model_with_preprocessing_anatomical.m
preprocessing_method = 'histmatch';  % ✅ Current choice
```

### To Change:
```matlab
preprocessing_method = 'zscore';  % Alternative option
```

## Expected Results

### With 'histmatch' (Current):
- ID Accuracy: ~85%
- OOD Accuracy: ~57-60%
- Degradation: ~30-32%

### With 'zscore' (Estimated):
- ID Accuracy: ~83-85%
- OOD Accuracy: ~55-58%
- Degradation: ~30-35%

### With 'none' (Baseline):
- ID Accuracy: ~85%
- OOD Accuracy: ~50-55%
- Degradation: ~40-45%

## Conclusion

**Stick with 'histmatch'** for now, as it:
1. Already gives reasonable results
2. Best aligns intensity distributions
3. Is consistent with your evaluation setup

**Focus on mixed training** (ROI + CXR) to reduce the remaining 32% degradation, rather than changing preprocessing.

