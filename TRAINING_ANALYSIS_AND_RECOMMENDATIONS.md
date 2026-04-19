# Training Analysis and Next Steps

## Executive Summary

The model training completed successfully with **5-fold cross-validation**. While the model shows good generalization (14.93% OOD degradation), there are **critical issues** that need to be addressed:

1. **Low Sensitivity (0.554)**: Model is too conservative, missing ~45% of PTB cases
2. **Poor Segmentation Quality**: IoU (0.167) and Dice (0.274) are well below target (>0.5)
3. **OOD Specificity Collapse**: 48.61% degradation on normal cases in OOD setting
4. **Class Imbalance**: High precision (0.952) but low sensitivity indicates bias toward negative class

---

## Detailed Performance Analysis

### Cross-Validation Results (5-Fold)

| Metric | Mean ± Std | Interpretation |
|--------|------------|----------------|
| **Accuracy** | 0.767 ± 0.043 | ✓ Acceptable |
| **Precision** | 0.952 ± 0.033 | ✓ Excellent (but indicates bias) |
| **Sensitivity** | 0.554 ± 0.099 | ⚠️ **CRITICAL: Too low** |
| **Specificity** | 0.972 ± 0.022 | ✓ Excellent |
| **F1-Score** | 0.696 ± 0.076 | ⚠️ Moderate (limited by sensitivity) |
| **AUC** | 0.874 ± 0.048 | ✓ Good |
| **IoU** | 0.167 ± 0.054 | ⚠️ **CRITICAL: Very low** |
| **Dice** | 0.274 ± 0.075 | ⚠️ **CRITICAL: Very low** |

**Best Fold**: fold_4 (Accuracy: 0.823)

### OOD Evaluation Results

| Metric | ID (ROI) | OOD (Full CXR) | Degradation |
|--------|----------|----------------|-------------|
| **Accuracy** | 0.780 | 0.646 | 17.16% |
| **Precision** | 0.852 | 0.599 | 29.71% |
| **Sensitivity** | 0.667 | 0.843 | -26.52% (improved) |
| **Specificity** | 0.889 | 0.457 | **48.61%** ⚠️⚠️ |
| **F1-Score** | 0.748 | 0.700 | 6.36% |
| **AUC** | 0.873 | 0.748 | 14.26% |

**Mean Degradation: 14.93%** (considered "Good generalization")

---

## Critical Issues Identified

### 1. **Low Sensitivity (0.554) - HIGH PRIORITY**

**Problem**: Model is missing ~45% of PTB cases. This is clinically unacceptable.

**Root Causes**:
- Class imbalance: Model is biased toward predicting "normal" (high specificity, low sensitivity)
- Focal loss parameters may need adjustment
- Threshold may be too high (if using probability threshold)

**Impact**: 
- High false negative rate
- Clinically dangerous (missing disease cases)

### 2. **Poor Segmentation Quality (IoU: 0.167, Dice: 0.274) - HIGH PRIORITY**

**Problem**: Segmentation metrics are far below target (>0.5).

**Root Causes**:
- Tversky loss may not be strong enough (λ_tversky = 2.5)
- GradCAM loss (λ_cam = 0.5) may be too weak after reduction
- Anatomical loss may be conflicting with segmentation objectives
- Model may not be learning proper attention maps

**Impact**:
- Poor attention localization
- Reduced interpretability
- May contribute to OOD degradation

### 3. **OOD Specificity Collapse (48.61% degradation) - MEDIUM PRIORITY**

**Problem**: Model fails dramatically on normal cases in OOD setting.

**Root Causes**:
- Model learned ROI-specific features that don't generalize
- Full CXR images have different background/context
- Preprocessing (histmatch) may not fully address distribution shift
- Model may be overfitting to ROI characteristics

**Impact**:
- High false positive rate on OOD data
- Reduced clinical utility on full CXR images

### 4. **Loss Component Balance**

**Current Configuration**:
- λ_cam: 0.5 (reduced from 4.53)
- λ_tversky: 2.5
- λ_anatomical: 15.0
- Anatomical reward weight: 0.5

**Observations from Training**:
- GradCAM loss: ~600-900 (after ×0.5)
- Tversky loss: ~1.8-2.2 (after ×2.5)
- Anatomical loss: ~200-800 (after ×15.0)
- Classification loss: ~0.01-0.08 (very small)

**Issue**: Classification loss is negligible, which may explain low sensitivity.

---

## Recommended Next Steps

### **Priority 1: Fix Low Sensitivity**

#### Option A: Adjust Focal Loss Parameters
```matlab
% Current (likely):
alpha = [0.25, 0.75];  % or similar
gamma = 2.0;

% Recommended:
alpha = [0.5, 0.5];    % Equal weight to both classes
gamma = 1.5;           % Slightly reduce focusing parameter
```

#### Option B: Add Class Weighting to Classification Loss
```matlab
% Increase weight for positive class (PTB)
classWeights = [1.0, 2.0];  % [normal, PTB]
% Apply in focal loss computation
```

#### Option C: Adjust Decision Threshold
```matlab
% If using probability threshold, lower it
threshold = 0.4;  % Instead of 0.5
```

#### Option D: Increase Classification Loss Weight
```matlab
% Add explicit weight to classification loss
lambda_cls = 1.0;  % Currently may be 1.0 but loss is too small
% Or increase focal loss contribution
```

**Action**: Modify focal loss computation to give more weight to positive class.

---

### **Priority 2: Improve Segmentation Quality**

#### Option A: Increase Tversky Loss Weight
```matlab
lambda_tversky = 5.0;  % Increase from 2.5
```

#### Option B: Increase GradCAM Loss Weight (Partial Restoration)
```matlab
lambda_cam = 1.0;  % Increase from 0.5 (but not back to 4.53)
```

#### Option C: Adjust Anatomical Loss Balance
```matlab
% Increase reward weight to encourage lung focus
anatomical_reward_weight = 0.75;  % Increase from 0.5
```

#### Option D: Add Explicit Segmentation Loss
```matlab
% Add Dice loss directly (not just Tversky)
lambda_dice = 2.0;
```

**Action**: Experiment with increasing λ_tversky to 5.0 and λ_cam to 1.0.

---

### **Priority 3: Improve OOD Generalization**

#### Option A: Data Augmentation
```matlab
% Add more aggressive augmentation
- Random cropping (simulate ROI extraction)
- Intensity variations
- Spatial transformations
- Mix-up or CutMix augmentation
```

#### Option B: Test-Time ROI Alignment
```matlab
% Extract ROI from full CXR at test time
% Use lung segmentation to align ROI
% This matches training distribution better
```

#### Option C: Domain Adaptation Techniques
```matlab
% Add domain adversarial loss
% Or use domain-specific batch normalization
```

#### Option D: Multi-Scale Training
```matlab
% Train on both ROI and full CXR images
% Use domain indicator to guide learning
```

**Action**: Implement test-time ROI alignment for OOD evaluation.

---

### **Priority 4: Loss Function Rebalancing**

#### Recommended Configuration:
```matlab
% Classification
lambda_cls = 1.0;  % Ensure classification loss has meaningful weight
classWeights = [1.0, 2.0];  % Favor PTB class

% GradCAM
lambda_cam = 1.0;  % Increase from 0.5 (partial restoration)

% Segmentation
lambda_tversky = 5.0;  % Increase from 2.5

% Anatomical
lambda_anatomical = 15.0;  % Keep current
anatomical_reward_weight = 0.75;  % Increase from 0.5
```

**Action**: Create a new configuration with these parameters and retrain.

---

## Immediate Action Plan

### Step 1: Quick Fix for Sensitivity (30 minutes)
1. Modify focal loss to increase positive class weight
2. Retrain for 10 epochs to see improvement
3. Check if sensitivity improves to >0.70

### Step 2: Segmentation Improvement (2-3 hours)
1. Increase λ_tversky to 5.0
2. Increase λ_cam to 1.0
3. Retrain full model (50 epochs)
4. Monitor IoU/Dice metrics

### Step 3: OOD Evaluation Fix (1-2 hours)
1. Implement test-time ROI extraction
2. Re-evaluate OOD performance
3. Compare with current results

### Step 4: Full Retraining (4-6 hours)
1. Apply all recommended changes
2. Train with new configuration
3. Compare comprehensive results

---

## Expected Improvements

### After Sensitivity Fix:
- Sensitivity: 0.554 → **0.70-0.75** (target)
- F1-Score: 0.696 → **0.75-0.80**
- Precision: May decrease slightly (0.952 → 0.85-0.90)

### After Segmentation Fix:
- IoU: 0.167 → **0.30-0.40** (intermediate target)
- Dice: 0.274 → **0.45-0.55** (target)
- Better attention localization

### After OOD Fix:
- OOD Specificity: 0.457 → **0.65-0.75**
- Overall OOD Accuracy: 0.646 → **0.70-0.75**

---

## Monitoring Metrics

During retraining, monitor:
1. **Sensitivity** (primary): Target >0.70
2. **IoU/Dice** (primary): Target >0.40/0.50
3. **OOD Specificity** (secondary): Target >0.65
4. **Loss components**: Ensure balanced contribution
5. **Validation loss**: Should decrease consistently

---

## Code Modifications Needed

### 1. Focal Loss Modification
```matlab
% In compute_focal_loss function:
% Increase alpha for positive class
alpha = [0.25, 0.75];  % Change to [0.25, 1.0] or [0.5, 1.0]
```

### 2. Loss Configuration
```matlab
% In training script:
loss_config.lambda_cam = 1.0;  % Increase from 0.5
loss_config.lambda_tversky = 5.0;  % Increase from 2.5
loss_config.anatomical_reward_weight = 0.75;  % Increase from 0.5
```

### 3. Class Weights
```matlab
% Add to configuration:
classWeights = [1.0, 2.0];  % [normal, PTB]
```

---

## Questions to Consider

1. **Clinical Priority**: Is sensitivity or specificity more important?
   - If sensitivity is critical (disease detection), prioritize sensitivity fixes
   - If specificity is critical (avoid false alarms), current model may be acceptable

2. **Deployment Scenario**: Will model see ROI or full CXR images?
   - If ROI: Current model is acceptable (ID accuracy: 0.780)
   - If full CXR: OOD fixes are critical

3. **Segmentation Requirement**: Is attention quality important?
   - If yes: Segmentation fixes are high priority
   - If no: Can focus on classification metrics only

---

## Conclusion

The model shows **promising results** but has **critical limitations**:
- ✅ Good overall accuracy and OOD generalization
- ⚠️ Low sensitivity (missing disease cases)
- ⚠️ Poor segmentation quality
- ⚠️ OOD specificity collapse

**Recommended approach**: Address sensitivity first (quick win), then segmentation, then OOD generalization. This should be done iteratively to understand the impact of each change.

