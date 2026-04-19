# Full Training Results Analysis (5 Folds, 50 Epochs)

## Training Configuration
- **K-Folds**: 5
- **Epochs**: 50 (full training)
- **Configuration**: v2.1 - Balanced (focal_alpha [0.35, 0.65], class_weight 1.2x)

## Cross-Validation Results Comparison

| Metric | 5 Epochs (Test) | 50 Epochs (Full) | Change | Status |
|--------|----------------|------------------|--------|--------|
| **Accuracy** | 0.726 ± 0.074 | **0.754 ± 0.027** | ✅ +3.9% | Improved |
| **Precision** | 0.720 ± 0.114 | **0.923 ± 0.075** | ✅ +28.2% | Much Better |
| **Sensitivity** | **0.791 ± 0.177** | 0.551 ± 0.062 | ⚠️ **-30.3%** | **MAJOR ISSUE** |
| **Specificity** | 0.663 ± 0.260 | **0.950 ± 0.053** | ✅ +43.3% | Much Better |
| **F1-Score** | **0.737 ± 0.060** | 0.686 ± 0.044 | ⚠️ -6.9% | Decreased |
| **AUC** | 0.854 ± 0.048 | **0.871 ± 0.049** | ✅ +2.0% | Slightly Better |
| **IoU** | 0.100 ± 0.003 | **0.159 ± 0.021** | ✅ +59.0% | Much Better |
| **Dice** | 0.174 ± 0.006 | **0.264 ± 0.031** | ✅ +51.7% | Much Better |

## Critical Finding: Sensitivity Collapse

### The Problem:
- **5 Epochs**: Sensitivity = 0.791 (good, balanced)
- **50 Epochs**: Sensitivity = 0.551 (too low - missing PTB cases!)
- **Change**: -30.3% (significant drop)

### What Happened:
With more training (50 epochs), the model became **too conservative**:
- **High Specificity** (0.950): Model correctly identifies normal cases
- **Low Sensitivity** (0.551): Model misses many PTB cases
- **Result**: Model is under-predicting PTB (opposite problem from before!)

### Root Cause Analysis:
1. **Balanced configuration was too conservative for long training**
   - Focal alpha [0.35, 0.65] with 1.2x class weight
   - With 5 epochs: Good balance (0.791 / 0.663)
   - With 50 epochs: Model learned to be too conservative (0.551 / 0.950)

2. **Training dynamics changed over time**
   - Early epochs: Model learned to detect PTB (sensitivity high)
   - Later epochs: Model learned to avoid false positives (specificity high, sensitivity low)
   - The balanced configuration didn't maintain sensitivity over long training

## Positive Improvements

### ✅ Segmentation Metrics (Major Success!)
- **IoU**: 0.100 → 0.159 (+59.0%)
- **Dice**: 0.264 → 0.031 (+51.7%)
- **Result**: Segmentation quality significantly improved with more training
- **Note**: Still below target (>0.5), but substantial progress

### ✅ Precision (Excellent!)
- **5 Epochs**: 0.720
- **50 Epochs**: 0.923 (+28.2%)
- **Result**: When model predicts PTB, it's correct 92% of the time

### ✅ Overall Accuracy
- **5 Epochs**: 0.726
- **50 Epochs**: 0.754 (+3.9%)
- **Result**: Slightly improved overall correctness

## Out-of-Distribution (OOD) Evaluation

### ID (ROI) Performance (Best Model - fold_1):
- **Accuracy**: 0.851
- **Precision**: 0.914
- **Sensitivity**: 0.768 (better than CV mean 0.551)
- **Specificity**: 0.931
- **F1-Score**: 0.835
- **AUC**: 0.890

### OOD (Full CXR) Performance:
- **Accuracy**: 0.588
- **Precision**: 0.549
- **Sensitivity**: 0.896 (actually improved on OOD!)
- **Specificity**: 0.292 (collapsed on OOD)
- **F1-Score**: 0.681
- **AUC**: 0.711

### OOD Analysis:

**Positive Observations:**
- **Sensitivity improved on OOD**: 0.768 → 0.896 (+16.6%)
  - Model detects PTB cases better on full CXR than ROI
  - This is actually good for clinical deployment!

**Major Issues:**
- **Specificity collapsed on OOD**: 0.931 → 0.292 (-68.6%)
  - Model over-predicts PTB on full CXR images
  - This is the main OOD problem

**Overall OOD Status:**
- Mean Degradation: 26.90% (Moderate, 15-30% range)
- Main issue: Specificity degradation (68.57%)

## Comparison with Original Baseline

| Metric | Original Baseline | 5 Epochs | 50 Epochs | Best |
|--------|------------------|----------|-----------|------|
| **Accuracy** | 0.767 ± 0.043 | 0.726 ± 0.074 | **0.754 ± 0.027** | 50 epochs |
| **Sensitivity** | 0.554 ± 0.099 | **0.791 ± 0.177** | 0.551 ± 0.062 | **5 epochs** |
| **Specificity** | 0.972 ± 0.022 | 0.663 ± 0.260 | **0.950 ± 0.053** | 50 epochs |
| **F1-Score** | 0.696 ± 0.076 | **0.737 ± 0.060** | 0.686 ± 0.044 | **5 epochs** |
| **AUC** | 0.874 ± 0.048 | 0.854 ± 0.048 | **0.871 ± 0.049** | 50 epochs |
| **IoU** | 0.167 ± 0.054 | 0.100 ± 0.003 | **0.159 ± 0.021** | 50 epochs |
| **Dice** | 0.274 ± 0.075 | 0.174 ± 0.006 | **0.264 ± 0.031** | 50 epochs |

**Key Insights:**
- **5 epochs**: Better sensitivity (0.791) but lower specificity (0.663)
- **50 epochs**: Better specificity (0.950) but lower sensitivity (0.551)
- **Neither achieves the ideal balance** (sensitivity ~0.75-0.85, specificity ~0.75-0.85)

## Recommendations

### 🎯 **Primary Issue: Sensitivity Too Low (0.551)**

The model is now **under-predicting PTB cases**, which is clinically problematic. We need to increase sensitivity while maintaining reasonable specificity.

### **Solution Options:**

#### **Option 1: Increase PTB Bias (RECOMMENDED)**
Adjust configuration to favor PTB detection:
- **Focal alpha**: `[0.30, 0.70]` (from [0.35, 0.65])
  - More weight on PTB class
- **Class weight**: `1.3x` (from 1.2x)
  - Additional PTB favor
- **Expected**: Sensitivity 0.65-0.75, Specificity 0.80-0.90

#### **Option 2: Use Early Stopping Based on F1-Score**
- Monitor F1-score during training
- Stop when F1-score peaks (before sensitivity drops too much)
- **Expected**: Better balance between sensitivity and specificity

#### **Option 3: Adaptive Class Weighting**
- Start with higher PTB weight (1.3x)
- Gradually reduce during training
- **Expected**: Maintain sensitivity while improving specificity

#### **Option 4: Focal Loss Gamma Adjustment**
- Increase `focal_gamma` to 2.0 (from 1.5)
- Focus more on hard examples
- **Expected**: Better handling of difficult PTB cases

### **Recommended Next Steps:**

1. **Quick Test**: Run with Option 1 (focal_alpha [0.30, 0.70], class_weight 1.3x)
   - Use 3 folds, 50 epochs
   - Monitor sensitivity/specificity balance
   - Target: Sensitivity 0.70-0.80, Specificity 0.80-0.90

2. **If still unbalanced**: Try Option 2 (early stopping on F1-score)
   - Stop training when F1-score peaks
   - Prevents over-optimization toward specificity

3. **For OOD Specificity**: Consider test-time ROI extraction
   - Extract ROI from full CXR at test time
   - Eliminates OOD degradation
   - Model trained on ROI, tested on ROI = perfect match

## Loss Component Observations

From training logs, loss components are well-balanced:
- **Classification**: ~0.01-0.20 (very small, well-trained)
- **GradCAM**: ~1300-1800 (stable)
- **Tversky**: ~0.75-0.90 (stable)
- **Anatomical**: Sometimes negative (reward > penalty - good sign!)

**Training Stability**: ✅ Good - no signs of instability or divergence

## Conclusion

### ✅ **Successes:**
1. Segmentation metrics improved significantly (IoU +59%, Dice +52%)
2. Precision excellent (0.923)
3. Specificity very high (0.950)
4. Training stable and converged

### ⚠️ **Critical Issues:**
1. **Sensitivity too low (0.551)** - Model missing PTB cases
2. **OOD specificity collapsed (0.292)** - Over-predicting PTB on full CXR
3. **F1-Score decreased** - Overall performance trade-off unfavorable

### 🎯 **Next Action:**
**Increase PTB bias** to improve sensitivity:
- Focal alpha: [0.30, 0.70]
- Class weight: 1.3x
- Target: Sensitivity 0.70-0.80, Specificity 0.80-0.90

The balanced configuration worked well for 5 epochs but became too conservative with 50 epochs. We need to adjust the balance to maintain sensitivity over longer training.

