# Training Results Log - PTB Detection Model

## Document Purpose
This document tracks all training runs with complete hyperparameters, identifies issues, documents fixes, and maintains a history of model performance improvements.

**How to Add New Results**: Copy the template section below and fill in all hyperparameters and results.

---

## Template for New Training Run

```markdown
## Training Run #X: [Description]
**Date**: [YYYY-MM-DD]  
**Purpose**: [Brief description]  
**K-Folds**: [number]  
**Epochs**: [number]  

### Complete Hyperparameters
- **Preprocessing Method**: [histmatch/clahe/zscore/minmax/clahe_zscore/none]
- **Learning Rate**: [value]
- **Batch Size**: [value]
- **Weight Decay**: [value]
- **Momentum**: [value]
- **Decay**: [value]
- **Early Stopping Patience**: [value]
- **Min Delta**: [value]

### Loss Function Hyperparameters
- **lambda_cam**: [value]
- **lambda_tversky**: [value]
- **lambda_anatomical**: [value]
- **tversky_alpha**: [value]
- **tversky_beta**: [value]
- **focal_alpha**: [value or array]
- **focal_gamma**: [value]
- **anatomical_reward_weight**: [value]
- **class_weight_multiplier**: [value]x for PTB
- **Combined PTB Bias**: [calculated value]x

### Network Configuration
- **Base Network**: [VGG16/VGG19/etc]
- **Feature Layer**: [relu5_3/etc]
- **nCam (samples for anatomical loss)**: [value]
- **Use GPU**: [Yes/No]

### Data Augmentation
- **RandXTranslation**: [range]
- **RandYTranslation**: [range]
- **RandRotation**: [range]
- **RandScale**: [range]

### Cross-Validation Results
| Metric | Mean ± Std |
|--------|------------|
| Accuracy | [value] ± [value] |
| Precision | [value] ± [value] |
| Sensitivity | [value] ± [value] |
| Specificity | [value] ± [value] |
| F1-Score | [value] ± [value] |
| AUC | [value] ± [value] |
| IoU | [value] ± [value] |
| Dice | [value] ± [value] |
| Tversky | [value] ± [value] |
| Jaccard | [value] ± [value] |
| Hausdorff | [value] ± [value] |

### Best Model Performance (ID - ROI)
- **Fold**: [fold number]
- **Accuracy**: [value]
- **Precision**: [value]
- **Sensitivity**: [value]
- **Specificity**: [value]
- **F1-Score**: [value]
- **AUC**: [value]
- **IoU**: [value]
- **Dice**: [value]

### OOD Evaluation (Full CXR)
**ID (ROI) Performance:**
- Accuracy: [value]
- Precision: [value]
- Sensitivity: [value]
- Specificity: [value]
- F1-Score: [value]
- AUC: [value]

**OOD (Full CXR) Performance:**
- Accuracy: [value]
- Precision: [value]
- Sensitivity: [value]
- Specificity: [value]
- F1-Score: [value]
- AUC: [value]

**Performance Degradation:**
- Accuracy: [value]%
- Precision: [value]%
- Sensitivity: [value]%
- Specificity: [value]%
- F1-Score: [value]%
- AUC: [value]%
- Mean Degradation: [value]%

### Loss Component Analysis (Typical Values)
- **Classification Loss**: [range]
- **GradCAM Loss**: [range] (×lambda_cam = [range])
- **Tversky Loss**: [range] (×lambda_tversky = [range])
- **Anatomical Loss**: [range] (×lambda_anatomical = [range])
- **Total Loss**: [range]

### Issues Identified
1. [Issue description]
2. [Issue description]

### Root Cause Analysis
[Detailed analysis of issues]

### Fixes Applied / Recommendations
1. [Fix description]
2. [Fix description]

### Notes
[Any additional observations or comments]
```

---

## Training Run #1: Original Baseline
**Date**: Initial baseline  
**Purpose**: Baseline performance measurement  
**K-Folds**: 5  
**Epochs**: 50  

### Complete Hyperparameters
- **Preprocessing Method**: [Not specified in original]
- **Learning Rate**: [Not specified]
- **Batch Size**: [Not specified]
- **Weight Decay**: [Not specified]
- **Momentum**: [Not specified]
- **Decay**: [Not specified]
- **Early Stopping Patience**: [Not specified]
- **Min Delta**: [Not specified]

### Loss Function Hyperparameters
- **lambda_cam**: 0.5
- **lambda_tversky**: 2.5
- **lambda_anatomical**: [Not specified]
- **tversky_alpha**: [Not specified]
- **tversky_beta**: [Not specified]
- **focal_alpha**: 0.25 (scalar)
- **focal_gamma**: 2.0
- **anatomical_reward_weight**: [Not specified]
- **class_weight_multiplier**: 1.0x (no multiplier)
- **Combined PTB Bias**: ~1.0x (baseline)

### Network Configuration
- **Base Network**: VGG16
- **Feature Layer**: relu5_3
- **nCam**: [Not specified]
- **Use GPU**: [Not specified]

### Cross-Validation Results
| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.767 ± 0.043 |
| Precision | 0.952 ± 0.033 |
| Sensitivity | 0.554 ± 0.099 ⚠️ |
| Specificity | 0.972 ± 0.022 |
| F1-Score | 0.696 ± 0.076 |
| AUC | 0.874 ± 0.048 |
| IoU | 0.167 ± 0.054 |
| Dice | 0.274 ± 0.075 |

### Issues Identified
1. **CRITICAL**: Sensitivity too low (0.554) - Model missing PTB cases
2. Segmentation metrics low (IoU 0.167, Dice 0.274)
3. Poor clinical utility due to low sensitivity

### Fixes Applied
- Increased `lambda_cam` from 0.5 to 1.0
- Increased `lambda_tversky` from 2.5 to 5.0
- Adjusted `focal_alpha` to favor PTB class
- Applied class weight multiplier to favor PTB

---

## Training Run #2: Aggressive PTB Favoring (Test)
**Date**: After baseline analysis  
**Purpose**: Test aggressive PTB bias to improve sensitivity  
**K-Folds**: 3  
**Epochs**: 5 (quick test)  

### Complete Hyperparameters
- **Preprocessing Method**: histmatch
- **Learning Rate**: 0.0002
- **Batch Size**: 14
- **Weight Decay**: 0.001
- **Momentum**: 0.8725
- **Decay**: 0.0042
- **Early Stopping Patience**: 10
- **Min Delta**: 1e-2

### Loss Function Hyperparameters
- **lambda_cam**: 1.0
- **lambda_tversky**: 5.0
- **lambda_anatomical**: 15.0
- **tversky_alpha**: 0.7
- **tversky_beta**: 0.3
- **focal_alpha**: [0.25, 0.75] (3x PTB weight)
- **focal_gamma**: 1.5
- **anatomical_reward_weight**: 0.5
- **class_weight_multiplier**: 1.5x for PTB
- **Combined PTB Bias**: 3.0 × 1.5 = 4.5x

### Network Configuration
- **Base Network**: VGG16
- **Feature Layer**: relu5_3
- **nCam**: 32
- **Use GPU**: Yes

### Data Augmentation
- **RandXTranslation**: [-10, 10]
- **RandYTranslation**: [-10, 10]
- **RandRotation**: [-10, 10]
- **RandScale**: [0.9, 1.1]

### Cross-Validation Results
| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.594 ± 0.116 |
| Precision | 0.558 ± 0.082 |
| Sensitivity | 0.962 ± 0.035 |
| Specificity | 0.240 ± 0.258 ⚠️ |
| F1-Score | 0.703 ± 0.055 |
| AUC | 0.675 ± 0.282 |
| IoU | 0.139 ± 0.035 |
| Dice | 0.226 ± 0.054 |

### Issues Identified
1. **CRITICAL**: Specificity collapsed (0.240) - Model predicting PTB for almost everything
2. **CRITICAL**: Accuracy decreased (0.594) - Overall performance worse
3. Over-correction: Too much PTB bias (4.5x combined)

### Root Cause Analysis
- Focal alpha [0.25, 0.75] = 3x PTB weight
- Class weight 1.5x = 1.5x PTB weight
- Combined: 3.0 × 1.5 = **4.5x total bias** toward PTB
- Result: Model predicts PTB for almost all cases

### Fixes Applied
- **Reduced focal_alpha**: [0.25, 0.75] → [0.35, 0.65] (1.86x PTB weight)
- **Reduced class_weight**: 1.5x → 1.2x PTB weight
- **New combined bias**: 1.86 × 1.2 = **2.23x** (50% reduction)

---

## Training Run #3: Balanced Configuration (Test)
**Date**: After aggressive test  
**Purpose**: Test balanced PTB bias  
**K-Folds**: 3  
**Epochs**: 5 (quick test)  

### Complete Hyperparameters
- **Preprocessing Method**: histmatch
- **Learning Rate**: 0.0002
- **Batch Size**: 14
- **Weight Decay**: 0.001
- **Momentum**: 0.8725
- **Decay**: 0.0042
- **Early Stopping Patience**: 10
- **Min Delta**: 1e-2

### Loss Function Hyperparameters
- **lambda_cam**: 1.0
- **lambda_tversky**: 5.0
- **lambda_anatomical**: 15.0
- **tversky_alpha**: 0.7
- **tversky_beta**: 0.3
- **focal_alpha**: [0.35, 0.65] (1.86x PTB weight)
- **focal_gamma**: 1.5
- **anatomical_reward_weight**: 0.75
- **class_weight_multiplier**: 1.2x for PTB
- **Combined PTB Bias**: 1.86 × 1.2 = 2.23x

### Network Configuration
- **Base Network**: VGG16
- **Feature Layer**: relu5_3
- **nCam**: 32
- **Use GPU**: Yes

### Data Augmentation
- **RandXTranslation**: [-10, 10]
- **RandYTranslation**: [-10, 10]
- **RandRotation**: [-10, 10]
- **RandScale**: [0.9, 1.1]

### Cross-Validation Results
| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.726 ± 0.074 |
| Precision | 0.720 ± 0.114 |
| Sensitivity | 0.791 ± 0.177 |
| Specificity | 0.663 ± 0.260 |
| F1-Score | 0.737 ± 0.060 |
| AUC | 0.854 ± 0.048 |
| IoU | 0.100 ± 0.003 |
| Dice | 0.174 ± 0.006 |

### Best Model Performance (ID - ROI)
- **Fold**: fold_3
- **Accuracy**: 0.799
- **Precision**: [Not specified]
- **Sensitivity**: [Not specified]
- **Specificity**: [Not specified]
- **F1-Score**: [Not specified]
- **AUC**: [Not specified]
- **IoU**: 0.103
- **Dice**: 0.181

### OOD Evaluation (Full CXR)
**ID (ROI) Performance:**
- Accuracy: 0.791
- Precision: 0.784
- Sensitivity: 0.791
- Specificity: 0.790
- F1-Score: 0.788
- AUC: 0.894

**OOD (Full CXR) Performance:**
- Accuracy: 0.638
- Precision: 0.593
- Sensitivity: 0.832
- Specificity: 0.451
- F1-Score: 0.692
- AUC: 0.739

**Performance Degradation:**
- Accuracy: 19.33%
- Precision: 24.41%
- Sensitivity: -5.13%
- Specificity: 42.87%
- F1-Score: 12.12%
- AUC: 17.29%
- Mean Degradation: 18.48%

### Loss Component Analysis (Typical Values)
- **Classification Loss**: ~0.05-0.20
- **GradCAM Loss**: ~1400-1800 (×1.0 = 1400-1800)
- **Tversky Loss**: ~0.85-0.90 (×5.0 = 4.25-4.50)
- **Anatomical Loss**: ~17-47 (×15.0 = 255-705)
- **Total Loss**: ~1800-2200

### Issues Identified
1. Segmentation metrics still low (IoU 0.100, Dice 0.174) - Expected with only 5 epochs
2. OOD specificity: 0.451 (42.87% degradation) - Still needs improvement
3. Overall: Good balance achieved, but needs full training

### Fixes Applied
- Configuration validated as balanced
- Proceeded to full training (5 folds, 50 epochs)

---

## Training Run #4: Balanced Configuration (Full Training)
**Date**: After balanced test validation  
**Purpose**: Full training with balanced configuration  
**K-Folds**: 5  
**Epochs**: 50  

### Complete Hyperparameters
- **Preprocessing Method**: histmatch
- **Learning Rate**: 0.0002
- **Batch Size**: 14
- **Weight Decay**: 0.001
- **Momentum**: 0.8725
- **Decay**: 0.0042
- **Early Stopping Patience**: 10
- **Min Delta**: 1e-2

### Loss Function Hyperparameters
- **lambda_cam**: 1.0
- **lambda_tversky**: 5.0
- **lambda_anatomical**: 15.0
- **tversky_alpha**: 0.7
- **tversky_beta**: 0.3
- **focal_alpha**: [0.35, 0.65] (1.86x PTB weight)
- **focal_gamma**: 1.5
- **anatomical_reward_weight**: 0.75
- **class_weight_multiplier**: 1.2x for PTB
- **Combined PTB Bias**: 1.86 × 1.2 = 2.23x

### Network Configuration
- **Base Network**: VGG16
- **Feature Layer**: relu5_3
- **nCam**: 32
- **Use GPU**: Yes

### Data Augmentation
- **RandXTranslation**: [-10, 10]
- **RandYTranslation**: [-10, 10]
- **RandRotation**: [-10, 10]
- **RandScale**: [0.9, 1.1]

### Cross-Validation Results
| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.754 ± 0.027 |
| Precision | 0.923 ± 0.075 |
| Sensitivity | 0.551 ± 0.062 ⚠️ |
| Specificity | 0.950 ± 0.053 |
| F1-Score | 0.686 ± 0.044 |
| AUC | 0.871 ± 0.049 |
| IoU | 0.159 ± 0.021 |
| Dice | 0.264 ± 0.031 |
| Tversky | 0.316 ± 0.036 |

### Best Model Performance (ID - ROI)
- **Fold**: fold_1
- **Accuracy**: 0.780
- **Precision**: [Not specified]
- **Sensitivity**: [Not specified]
- **Specificity**: [Not specified]
- **F1-Score**: [Not specified]
- **AUC**: [Not specified]
- **IoU**: 0.158
- **Dice**: 0.260

### OOD Evaluation (Full CXR)
**ID (ROI) Performance:**
- Accuracy: 0.851
- Precision: 0.914
- Sensitivity: 0.768
- Specificity: 0.931
- F1-Score: 0.835
- AUC: 0.890

**OOD (Full CXR) Performance:**
- Accuracy: 0.588
- Precision: 0.549
- Sensitivity: 0.896
- Specificity: 0.292 ⚠️
- F1-Score: 0.681
- AUC: 0.711

**Performance Degradation:**
- Accuracy: 30.90%
- Precision: 39.94%
- Sensitivity: -16.60%
- Specificity: 68.57% ⚠️
- F1-Score: 18.45%
- AUC: 20.16%
- Mean Degradation: 26.90%

### Loss Component Analysis (Typical Values)
- **Classification Loss**: ~0.01-0.20
- **GradCAM Loss**: ~1300-1800 (×1.0 = 1300-1800)
- **Tversky Loss**: ~0.75-0.90 (×5.0 = 3.75-4.50)
- **Anatomical Loss**: Sometimes negative (reward > penalty - good sign!)
- **Total Loss**: ~1500-2000

### Issues Identified
1. **CRITICAL**: Sensitivity dropped significantly (0.791 → 0.551, -30.3%)
   - Model became too conservative with longer training
   - Missing PTB cases (opposite problem from Run #2)
2. **CRITICAL**: OOD specificity collapsed (0.931 → 0.292, -68.6%)
   - Model over-predicts PTB on full CXR images
3. F1-Score decreased (0.737 → 0.686, -6.9%)
4. **Positive**: Segmentation metrics improved significantly (IoU +59%, Dice +52%)

### Root Cause Analysis
- **Training Dynamics**: Configuration worked well for 5 epochs but became too conservative with 50 epochs
- **Early Epochs**: Model learned to detect PTB (sensitivity high)
- **Later Epochs**: Model learned to avoid false positives (specificity high, sensitivity low)
- **Result**: Balanced configuration didn't maintain sensitivity over long training

### Fixes Applied / Recommendations
1. **Increase PTB Bias**:
   - `focal_alpha`: [0.35, 0.65] → [0.30, 0.70] (2.33x PTB weight)
   - `class_weight_multiplier`: 1.2x → 1.3x for PTB
   - New combined bias: 2.33 × 1.3 = **3.03x** (moderate increase)
2. **Expected Results**: Sensitivity 0.70-0.80, Specificity 0.80-0.90
3. **Alternative**: Early stopping based on F1-score

---

## Summary of All Runs

| Run | Epochs | Sensitivity | Specificity | F1-Score | Status |
|-----|--------|------------|-------------|----------|--------|
| #1: Baseline | 50 | 0.554 ⚠️ | 0.972 | 0.696 | Too conservative |
| #2: Aggressive | 5 | 0.962 | 0.240 ⚠️ | 0.703 | Too aggressive |
| #3: Balanced (Test) | 5 | 0.791 ✅ | 0.663 | 0.737 ✅ | Good balance |
| #4: Balanced (Full) | 50 | 0.551 ⚠️ | 0.950 | 0.686 | Too conservative |

### Key Insights
1. **Short training (5 epochs)**: Balanced configuration works well
2. **Long training (50 epochs)**: Balanced configuration becomes too conservative
3. **Need**: Configuration that maintains sensitivity over longer training
4. **Solution**: Increase PTB bias for longer training runs

---

## Configuration Evolution

### Run #1: Baseline
```matlab
focal_alpha = 0.25;  % Scalar
class_weight = baseWeights;  % No multiplier
lambda_cam = 0.5;
lambda_tversky = 2.5;
Combined bias: ~1.0x
```

### Run #2: Aggressive (Too Much)
```matlab
focal_alpha = [0.25, 0.75];  % 3x PTB weight
class_weight_multiplier = 1.5x;  % 1.5x PTB weight
Combined bias: 4.5x  % TOO MUCH
Result: Specificity collapsed (0.240)
```

### Run #3 & #4: Balanced (Too Conservative for Long Training)
```matlab
focal_alpha = [0.35, 0.65];  % 1.86x PTB weight
class_weight_multiplier = 1.2x;  % 1.2x PTB weight
Combined bias: 2.23x
Result: Good for 5 epochs, too conservative for 50 epochs
```

### Recommended: Run #5 (Next)
```matlab
focal_alpha = [0.30, 0.70];  % 2.33x PTB weight
class_weight_multiplier = 1.3x;  % 1.3x PTB weight
Combined bias: 3.03x  % Moderate increase
Expected: Sensitivity 0.70-0.80, Specificity 0.80-0.90
```

---

## Performance Targets

### Clinical Utility Targets
- **Sensitivity**: 0.75-0.85 (detect PTB cases)
- **Specificity**: 0.75-0.85 (avoid false positives)
- **F1-Score**: 0.75-0.80 (balanced performance)
- **AUC**: >0.85 (good discrimination)

### Segmentation Targets
- **IoU**: >0.5 (good overlap)
- **Dice**: >0.5 (good similarity)

### OOD Targets
- **Mean Degradation**: <15% (good generalization)
- **OOD Specificity**: >0.70 (acceptable false positive rate)

---

**Last Updated**: After Run #5 (Quick Test - 3 folds, 5 epochs)  
**Next Run**: Run #6 (Reduced PTB bias - find optimal balance)

---

## Training Run #5: Increased PTB Bias (Quick Test)
**Date**: 2024-12-XX  
**Purpose**: Test increased PTB bias (3.03x) to maintain sensitivity over long training  
**K-Folds**: 3  
**Epochs**: 5 (Quick test mode)  

### Complete Hyperparameters
- **Preprocessing Method**: histmatch
- **Learning Rate**: 0.0002
- **Batch Size**: 14
- **Weight Decay**: 0.001
- **Momentum**: 0.8725
- **Decay**: 0.0042
- **Early Stopping Patience**: 10
- **Min Delta**: 0.01

### Loss Function Hyperparameters
- **lambda_cam**: 1.0 (increased from 0.5)
- **lambda_tversky**: 5.0 (increased from 2.5)
- **lambda_anatomical**: 15.0
- **tversky_alpha**: 0.7
- **tversky_beta**: 0.3
- **focal_alpha**: [0.30, 0.70] (increased from [0.35, 0.65])
- **focal_gamma**: 1.5
- **anatomical_reward_weight**: 0.75 (increased from 0.5)
- **class_weight_multiplier**: 1.3x for PTB (increased from 1.2x)
- **Combined PTB Bias**: 3.03x (2.33 × 1.3, increased from 2.23x)

### Network Configuration
- **Base Network**: VGG16
- **Feature Layer**: relu5_3
- **nCam (samples for anatomical loss)**: 32
- **Use GPU**: Yes

### Data Augmentation
- **RandXTranslation**: [-10, 10]
- **RandYTranslation**: [-10, 10]
- **RandRotation**: [-15, 15]
- **RandScale**: [0.9, 1.1]

### Cross-Validation Results
| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.570 ± 0.114 ⚠️ |
| Precision | 0.562 ± 0.113 ⚠️ |
| Sensitivity | 0.899 ± 0.168 ✅ |
| Specificity | 0.255 ± 0.385 ⚠️⚠️ |
| F1-Score | 0.675 ± 0.021 |
| AUC | 0.786 ± 0.126 |
| IoU | 0.101 ± 0.030 |
| Dice | 0.175 ± 0.047 |
| Tversky | 0.214 ± 0.054 |

### Best Model Performance (ID - ROI)
- **Fold**: fold_3
- **Accuracy**: 0.701
- **Precision**: [Not specified in output]
- **Sensitivity**: [Not specified in output]
- **Specificity**: [Not specified in output]
- **F1-Score**: [Not specified in output]
- **AUC**: [Not specified in output]
- **IoU**: 0.074
- **Dice**: 0.134

### OOD Evaluation (Full CXR)
**ID (ROI) Performance:**
- Accuracy: 0.675
- Precision: 0.605
- Sensitivity: 0.974 ✅
- Specificity: 0.387 ⚠️⚠️
- F1-Score: 0.747
- AUC: 0.882

**OOD (Full CXR) Performance:**
- Accuracy: 0.503
- Precision: 0.496
- Sensitivity: 0.986 ✅
- Specificity: 0.039 ⚠️⚠️⚠️ (CRITICAL)
- F1-Score: 0.660
- AUC: 0.696

**Performance Degradation:**
- Accuracy: 25.53%
- Precision: 18.01%
- Sensitivity: -1.19% (improved)
- Specificity: 89.91% ⚠️⚠️⚠️ (CRITICAL COLLAPSE)
- F1-Score: 11.58%
- AUC: 21.10%
- Mean Degradation: 27.49%

### Loss Component Analysis (Typical Values)
- **Classification Loss**: ~0.06-0.25
- **GradCAM Loss**: ~1300-1900 (×1.0 = 1300-1900)
- **Tversky Loss**: ~0.75-0.95 (×5.0 = 3.75-4.75)
- **Anatomical Loss**: ~15-60 (×15.0 = 225-900)
- **Total Loss**: ~1500-2600

### Issues Identified
1. **CRITICAL**: Specificity collapsed severely
   - CV: 0.255 ± 0.385 (very low, high variance)
   - ID: 0.387 (low)
   - OOD: 0.039 (near-zero, 89.91% degradation) ⚠️⚠️⚠️
   - Model predicting PTB for almost all cases
2. **CRITICAL**: High variance in results
   - Accuracy std: 0.114 (high)
   - Specificity std: 0.385 (very high)
   - Indicates model instability
3. **Positive**: Sensitivity maintained (0.899-0.986)
   - PTB bias is working for sensitivity
   - But at the cost of specificity
4. **Moderate**: Accuracy and F1-Score acceptable but not optimal
   - Accuracy: 0.570 (below target 0.75-0.80)
   - F1-Score: 0.675 (below target 0.75-0.80)

### Root Cause Analysis
- **Over-correction**: 3.03x PTB bias is too high
  - Model learned to predict PTB for almost everything
  - Similar to Run #2 (4.5x bias) but less severe
- **Training dynamics**: Even with 5 epochs, the high bias caused specificity collapse
- **OOD generalization**: Model performs worse on full CXR (specificity 0.039)
  - Suggests model is overfitting to PTB class
- **Variance**: High standard deviations indicate the model is unstable across folds

### Fixes Applied / Recommendations
1. **Reduce PTB Bias** (CRITICAL):
   - `focal_alpha`: [0.30, 0.70] → [0.32, 0.68] (2.13x PTB weight, reduced from 2.33x)
   - `class_weight_multiplier`: 1.3x → 1.25x for PTB (reduced from 1.3x)
   - New combined bias: 2.13 × 1.25 = **2.66x** (reduced from 3.03x)
   - Target: Balance between Run #3/#4 (2.23x) and Run #5 (3.03x)
2. **Alternative Strategy**: Use adaptive bias
   - Start with 2.5x bias
   - Monitor sensitivity/specificity during training
   - Adjust if needed
3. **Early Stopping**: Consider stopping based on F1-score or balanced accuracy
4. **Expected Results**: 
   - Sensitivity: 0.75-0.85 (maintained)
   - Specificity: 0.70-0.85 (recovered)
   - F1-Score: 0.75-0.80 (improved)

### Notes
- This was a quick test (3 folds, 5 epochs) to validate the approach
- Results show that 3.03x bias is too aggressive
- Need to find the "sweet spot" between 2.23x (too conservative for long training) and 3.03x (too aggressive)
- OOD specificity collapse (0.039) is a critical issue that must be addressed

---

## Summary of All Runs

| Run | Epochs | Sensitivity | Specificity | F1-Score | Status |
|-----|--------|------------|-------------|----------|--------|
| #1: Baseline | 50 | 0.554 ⚠️ | 0.972 | 0.696 | Too conservative |
| #2: Aggressive | 5 | 0.962 | 0.240 ⚠️ | 0.703 | Too aggressive (4.5x bias) |
| #3: Balanced (Test) | 5 | 0.791 ✅ | 0.663 | 0.737 ✅ | Good balance (2.23x bias) |
| #4: Balanced (Full) | 50 | 0.551 ⚠️ | 0.950 | 0.686 | Too conservative (2.23x bias) |
| #5: Increased Bias (Test) | 5 | 0.899 ✅ | 0.255 ⚠️⚠️ | 0.675 | Too aggressive (3.03x bias) |

### Key Insights
1. **Short training (5 epochs)**: 
   - 2.23x bias works well (Run #3)
   - 3.03x bias is too much (Run #5)
2. **Long training (50 epochs)**: 
   - 2.23x bias becomes too conservative (Run #4)
   - Need intermediate value (2.5-2.7x)
3. **Optimal Range**: Between 2.23x and 3.03x
   - Target: 2.5-2.7x combined bias
   - Expected: Sensitivity 0.75-0.85, Specificity 0.70-0.85

---

## Configuration Evolution

### Run #1: Baseline
```matlab
focal_alpha = 0.25;  % Scalar
class_weight = baseWeights;  % No multiplier
lambda_cam = 0.5;
lambda_tversky = 2.5;
Combined bias: ~1.0x
```

### Run #2: Aggressive (Too Much)
```matlab
focal_alpha = [0.25, 0.75];  % 3x PTB weight
class_weight_multiplier = 1.5x;  % 1.5x PTB weight
Combined bias: 4.5x  % TOO MUCH
Result: Specificity collapsed (0.240)
```

### Run #3 & #4: Balanced (Too Conservative for Long Training)
```matlab
focal_alpha = [0.35, 0.65];  % 1.86x PTB weight
class_weight_multiplier = 1.2x;  % 1.2x PTB weight
Combined bias: 2.23x
Result: Good for 5 epochs, too conservative for 50 epochs
```

### Run #5: Increased Bias (Too Aggressive)
```matlab
focal_alpha = [0.30, 0.70];  % 2.33x PTB weight
class_weight_multiplier = 1.3x;  % 1.3x PTB weight
Combined bias: 3.03x  % TOO MUCH
Result: Specificity collapsed (0.255 CV, 0.039 OOD)
```

### Recommended: Run #6 (Next)
```matlab
focal_alpha = [0.32, 0.68];  % 2.13x PTB weight
class_weight_multiplier = 1.25x;  % 1.25x PTB weight
Combined bias: 2.66x  % Intermediate value
Expected: Sensitivity 0.75-0.85, Specificity 0.70-0.85
```
