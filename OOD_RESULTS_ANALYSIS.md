# OOD Evaluation Results Analysis

## Training Configuration
- **Preprocessing**: Histogram matching (histmatch)
- **Loss Function**: Focal + GradCAM + Tversky + Anatomical Guidance
- **K-Fold CV**: 5 folds
- **Best Fold**: fold_4

## In-Distribution (ID) Performance
**Excellent performance on ROI images:**
- Accuracy: **0.851** (85.1%)
- Precision: **0.833** (83.3%)
- Sensitivity: **0.870** (87.0%)
- Specificity: **0.833** (83.3%)
- F1-Score: **0.851** (85.1%)
- AUC: **0.914** (91.4%)

## Out-of-Distribution (OOD) Performance
**Significant degradation on full CXR images:**
- Accuracy: **0.575** (57.5%) ⚠️
- Precision: **0.555** (55.5%) ⚠️
- Sensitivity: **0.678** (67.8%)
- Specificity: **0.476** (47.6%) ⚠️⚠️
- F1-Score: **0.610** (61.0%)
- AUC: **0.611** (61.1%) ⚠️⚠️

## Performance Degradation Analysis

| Metric | ID | OOD | Degradation |
|--------|----|----|-------------| 
| Accuracy | 85.1% | 57.5% | **-32.4%** |
| Precision | 83.3% | 55.5% | **-33.5%** |
| Sensitivity | 87.0% | 67.8% | **-22.0%** |
| Specificity | 83.3% | 47.6% | **-42.8%** ⚠️ |
| F1-Score | 85.1% | 61.0% | **-28.3%** |
| AUC | 91.4% | 61.1% | **-33.2%** |

**Mean Degradation: 32.04%** - **High degradation (>30%)**

## Key Observations

### 1. **Intensity Normalization Helped, But Not Enough**
- Histogram matching reduced intensity shift from 2.2x to ~1.0x
- However, **32% degradation** still indicates significant distribution shift
- **Conclusion**: Intensity difference is not the only factor

### 2. **Specificity is Most Affected (-42.8%)**
- Model struggles to correctly identify **normal cases** on full CXR
- High false positive rate on OOD data
- Suggests model learned ROI-specific features that don't generalize

### 3. **Sensitivity is Least Affected (-22.0%)**
- Model still detects **PTB cases** relatively well
- Disease features may be more robust across distributions
- But still significant drop from 87% to 68%

### 4. **AUC Dropped Dramatically (91.4% → 61.1%)**
- AUC of 61.1% is barely better than random (50%)
- Indicates poor discrimination ability on OOD data
- Model confidence/calibration is severely affected

## Root Causes of High Degradation

### 1. **Content Distribution Shift** (Primary Factor)
- **ROI (ID)**: 52-53% non-zero pixels (cropped lung region)
- **CXR (OOD)**: 95-99% non-zero pixels (full chest image)
- **Impact**: Model trained on focused lung region, tested on full anatomy
- **Solution**: Train on full CXR images, or use attention mechanisms

### 2. **Contextual Information Difference**
- **ROI**: Only lung region visible (no background, no other anatomy)
- **CXR**: Full chest context (heart, ribs, diaphragm, background)
- **Impact**: Model may rely on absence of context as a feature
- **Solution**: Include full CXR images in training

### 3. **Spatial Distribution Shift**
- **ROI**: Lung region centered, cropped tightly
- **CXR**: Lung region in different spatial context
- **Impact**: Model learned spatial patterns specific to cropped images
- **Solution**: Data augmentation with full images

### 4. **Feature Distribution Shift**
- Model learned features specific to cropped ROI characteristics
- These features don't transfer well to full CXR images
- **Solution**: Domain adaptation or fine-tuning on mixed data

## Recommendations

### Immediate Actions

#### 1. **Train on Mixed Data** (Highest Priority)
```matlab
% Mix ROI and CXR images during training
% 70% ROI (ID) + 30% CXR (OOD) in training set
% This exposes model to both distributions
```

#### 2. **Apply Intensity Normalization to Training Data**
- Current: Only test-time preprocessing
- Better: Apply histogram matching during training too
- **Status**: ✅ Already implemented in your script

#### 3. **Fine-tune on OOD Data**
```matlab
% Fine-tune pre-trained model on full CXR images
% Use smaller learning rate (1e-5 to 1e-6)
% Freeze early layers, fine-tune later layers
```

### Advanced Strategies

#### 4. **Domain Adaptation Techniques**
- **Adversarial Domain Adaptation**: Train discriminator to distinguish ID/OOD
- **Domain Mixup**: Mix ROI and CXR features during training
- **Domain-Specific Batch Normalization**: Separate BN for ID/OOD

#### 5. **Attention Mechanisms**
- **Spatial Attention**: Focus on lung regions in full CXR
- **Channel Attention**: Adapt features for different contexts
- **Self-Attention**: Model long-range dependencies in full images

#### 6. **Test-Time Augmentation (TTA)**
- Apply multiple augmentations at test time
- Average predictions for robustness
- Can improve OOD performance by 2-5%

### Alternative Approaches

#### 7. **Train Entirely on Full CXR Images**
- If possible, train model on full CXR from start
- No distribution shift = no degradation
- Requires re-labeling or using existing CXR labels

#### 8. **Ensemble Methods**
- Train separate models on ROI and CXR
- Combine predictions at test time
- Can improve robustness

## Expected Improvements

### With Mixed Training (70% ROI + 30% CXR):
- **Expected Degradation**: 15-20% (down from 32%)
- **Expected OOD Accuracy**: 65-70% (up from 57.5%)

### With Fine-tuning on OOD:
- **Expected Degradation**: 10-15%
- **Expected OOD Accuracy**: 70-75%

### With Domain Adaptation:
- **Expected Degradation**: 8-12%
- **Expected OOD Accuracy**: 75-80%

## Next Steps

1. **Implement Mixed Training** (Priority 1)
   - Modify training script to include 30% CXR images
   - Re-train with same configuration
   - Expected: 15-20% degradation

2. **Fine-tune on OOD Data** (Priority 2)
   - Load trained model
   - Fine-tune on full CXR images
   - Expected: 10-15% degradation

3. **Compare Results**
   - Evaluate all approaches
   - Choose best strategy
   - Document findings

## Conclusion

**Current Status**: 
- ✅ Excellent ID performance (85.1% accuracy)
- ❌ Poor OOD performance (57.5% accuracy)
- ⚠️ High degradation (32%) despite intensity normalization

**Key Insight**: 
Intensity normalization alone is insufficient. The primary issue is **content/context distribution shift**, not just intensity. The model needs to see full CXR images during training to generalize well.

**Recommended Path Forward**:
1. Implement mixed training (ROI + CXR)
2. Fine-tune on OOD data
3. Apply domain adaptation if needed
4. Target: <15% degradation, >70% OOD accuracy

