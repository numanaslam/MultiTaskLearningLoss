# Test Run Analysis - Updated Configuration

## Test Configuration
- **K-Folds**: 3 (reduced from 5 for faster testing)
- **Epochs**: 5 (reduced from 50 for faster testing)
- **Updated Loss Weights**: Applied all recommendations

## Results Summary

### Cross-Validation Performance (3 folds, 5 epochs)

| Metric | Previous (5 folds, 50 epochs) | Current Test | Change |
|--------|-------------------------------|--------------|--------|
| **Accuracy** | 0.767 ± 0.043 | 0.594 ± 0.116 | ⚠️ -22.6% |
| **Precision** | 0.952 ± 0.033 | 0.558 ± 0.082 | ⚠️ -41.4% |
| **Sensitivity** | 0.554 ± 0.099 | **0.962 ± 0.035** | ✅ **+73.6%** |
| **Specificity** | 0.972 ± 0.022 | **0.240 ± 0.258** | ⚠️ **-75.3%** |
| **F1-Score** | 0.696 ± 0.076 | 0.703 ± 0.055 | ✅ +1.0% |
| **AUC** | 0.874 ± 0.048 | 0.675 ± 0.282 | ⚠️ -22.8% |
| **IoU** | 0.167 ± 0.054 | 0.139 ± 0.035 | ⚠️ -16.8% |
| **Dice** | 0.274 ± 0.075 | 0.226 ± 0.054 | ⚠️ -17.5% |

## Key Findings

### ✅ **SUCCESS: Sensitivity Dramatically Improved**
- **0.554 → 0.962** (+73.6%)
- Model now detects PTB cases very well
- This was the primary goal

### ⚠️ **PROBLEM: Specificity Collapsed**
- **0.972 → 0.240** (-75.3%)
- Model is now predicting PTB for almost everything
- High false positive rate

### ⚠️ **Overall Accuracy Decreased**
- **0.767 → 0.594** (-22.6%)
- Due to specificity collapse

### 📊 **Loss Component Analysis**

From the training logs:
- **Classification Loss**: ~0.1-0.3 (still very small)
- **GradCAM Loss**: ~1400-1900 (increased with λ_cam=1.0)
- **Tversky Loss**: ~0.7-0.9 (high, as expected with λ_tversky=5.0)
- **Anatomical Loss**: Sometimes negative (reward > penalty, which is good!)

**Loss Breakdown Example (Epoch 3, Fold 1):**
- Cls: 0.18
- CAM: 1777.82 (×1.0 = 1777.82)
- Tversky: 0.94 (×5.0 = 4.72)
- Anatomical: 51.26 (×15.0 = 768.85)
- **Total**: ~2550

## Root Cause Analysis

### Why Specificity Collapsed

The combination of two strong PTB biases created an **over-correction**:

1. **Focal Alpha**: `[0.25, 0.75]` = **3x weight** for PTB class
2. **Class Weights**: `1.5x multiplier` for PTB class
3. **Combined Effect**: ~**4.5x total bias** toward PTB

This caused the model to predict PTB for almost all cases, leading to:
- High sensitivity (catches all PTB cases)
- Low specificity (also predicts PTB for normal cases)

## Adjustments Made

### 1. **Reduced Focal Alpha**
```matlab
% Before: [0.25, 0.75] (3x PTB weight)
% After:  [0.35, 0.65] (1.86x PTB weight)
config.focal_alpha = [0.35, 0.65];
```

### 2. **Reduced Class Weight Multiplier**
```matlab
% Before: 1.5x PTB weight
% After:  1.2x PTB weight
classWeights = [baseWeights(1), baseWeights(2) * 1.2];
```

### 3. **Combined Effect**
- **Previous**: 3.0 × 1.5 = **4.5x bias**
- **New**: 1.86 × 1.2 = **2.23x bias**
- **Reduction**: ~50% less bias toward PTB

## Expected Improvements

With the balanced configuration:

### Target Metrics:
- **Sensitivity**: 0.962 → **0.75-0.85** (slight decrease, but still good)
- **Specificity**: 0.240 → **0.70-0.85** (significant improvement)
- **Accuracy**: 0.594 → **0.75-0.80** (improved balance)
- **F1-Score**: 0.703 → **0.75-0.80** (maintained or improved)

### Segmentation Metrics (with more epochs):
- **IoU**: 0.139 → **0.25-0.35** (with 50 epochs)
- **Dice**: 0.226 → **0.40-0.50** (with 50 epochs)

## Recommendations for Next Test Run

### Option 1: Quick Test (Current Settings)
- Keep: 3 folds, 5 epochs
- Test the balanced configuration
- Verify sensitivity/specificity balance

### Option 2: Full Training (Recommended)
- Use: 5 folds, 50 epochs
- Apply balanced configuration
- Monitor:
  - Sensitivity should be 0.70-0.80
  - Specificity should be 0.70-0.85
  - IoU/Dice should improve with more epochs

### Option 3: Fine-Tuning
If specificity is still too low:
- Further reduce focal_alpha to `[0.40, 0.60]`
- Reduce class weight to `1.15x`

If sensitivity drops too much:
- Increase focal_alpha to `[0.30, 0.70]`
- Increase class weight to `1.25x`

## Loss Component Observations

### Positive Signs:
1. **Anatomical loss sometimes negative**: This means the model is learning to focus on lungs (reward > penalty)
2. **Tversky loss high but stable**: Expected with increased weight, should improve segmentation over time
3. **GradCAM loss in expected range**: ~1500-1800 with λ_cam=1.0

### Areas to Monitor:
1. **Classification loss still very small**: May need explicit weight if sensitivity/specificity don't balance
2. **IoU/Dice need more epochs**: Segmentation improvements require longer training

## Conclusion

The test run successfully **demonstrated that sensitivity can be improved** (0.554 → 0.962), but revealed that **the initial configuration was too aggressive**, causing specificity collapse.

**The balanced configuration should achieve:**
- Sensitivity: 0.75-0.85 (good, but not perfect)
- Specificity: 0.70-0.85 (much better)
- Overall better clinical utility

**Next Steps:**
1. Run with balanced configuration (3 folds, 5 epochs) to verify balance
2. If balanced, run full training (5 folds, 50 epochs)
3. Monitor all metrics and adjust if needed

