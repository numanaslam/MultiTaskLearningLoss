# How to Improve Out-of-Distribution (OOD) Accuracy

## Current Problem
Your model shows significant performance degradation on OOD (full CXR) data:
- **ID (ROI) Accuracy**: ~100%
- **OOD (Full CXR) Accuracy**: ~55-60%
- **Degradation**: ~40-45%

This is a common problem in medical imaging when models trained on cropped regions are evaluated on full images.

## Techniques to Improve OOD Accuracy

### 1. **Enhanced Data Augmentation** ⭐ Easy, Quick
**What it does**: Uses stronger augmentation during training to improve generalization.

**Implementation**:
- Increased rotation range: [-15, 15] degrees
- Larger translation: [-20, 20] pixels
- Scale variation: [0.9, 1.1]
- Shear transformation
- Horizontal/vertical flips

**Expected Improvement**: +5-10% OOD accuracy

**When to use**: Quick improvement without changing training data.

---

### 2. **Mixed Training (ROI + Full CXR)** ⭐⭐ **RECOMMENDED**
**What it does**: Trains on both ROI and full CXR images simultaneously.

**Implementation**:
- Combine ROI and CXR datasets (50% each)
- Model sees both distributions during training
- Learns domain-invariant features

**Expected Improvement**: +10-20% OOD accuracy

**When to use**: You have labeled CXR images. **Best overall approach.**

**Advantages**:
- Model learns to handle both image types
- Better generalization
- Maintains ID performance

---

### 3. **Domain Adaptation**
**What it does**: Explicitly aligns feature distributions between ROI and CXR domains.

**Implementation**:
- Adversarial domain alignment
- Correlation alignment (CORAL)
- Batch normalization statistics alignment

**Expected Improvement**: +15-25% OOD accuracy

**When to use**: You want explicit domain alignment (more complex).

---

### 4. **Test-Time Augmentation (TTA)** ⭐ **NO RETRAINING**
**What it does**: Averages predictions from multiple augmented test images.

**Implementation**:
- Create 5-10 augmented versions of each test image
- Average predictions across augmentations
- Improves robustness without retraining

**Expected Improvement**: +3-8% OOD accuracy

**When to use**: Quick improvement without retraining. Easy to implement.

**Advantages**:
- No model retraining required
- Can be applied immediately
- Works with existing models

---

### 5. **Fine-tuning on OOD Data**
**What it does**: Fine-tunes the trained model on CXR images.

**Implementation**:
- Load trained model
- Fine-tune on CXR dataset with lower learning rate
- Adapts model to OOD distribution

**Expected Improvement**: +15-25% OOD accuracy

**When to use**: You have labeled CXR data for fine-tuning.

**Note**: May slightly reduce ID performance.

---

### 6. **Ensemble Methods**
**What it does**: Trains multiple models and averages predictions.

**Implementation**:
- Train 3-5 models with different:
  - Initializations
  - Augmentations
  - Data splits
- Average predictions at inference

**Expected Improvement**: +5-15% OOD accuracy

**When to use**: When computational resources allow multiple training runs.

---

## Recommended Approach

### Phase 1: Quick Wins (No Retraining)
1. **Test-Time Augmentation (TTA)** - Immediate improvement
2. Evaluate and measure improvement

### Phase 2: Better Training Strategy
1. **Mixed Training (ROI + CXR)** - Best overall approach
2. Train model on combined dataset
3. Evaluate on both ID and OOD

### Phase 3: Advanced Techniques
1. **Domain Adaptation** - If mixed training insufficient
2. **Ensemble Methods** - For best possible results

---

## Implementation Guide

### Quick Start: Test-Time Augmentation

```matlab
% This doesn't require retraining - just different inference
resultsTTA = technique4_test_time_augmentation(trainedNet, cxrDir, useGPU);
```

### Best Practice: Mixed Training

```matlab
% Train on both ROI and CXR images
improvedNet = technique2_mixed_training(trainedNet, optimal_params, roiDir, cxrDir, useGPU);
```

### Full Pipeline

```matlab
% Run the improvement script
improve_ood_accuracy();

% Or call individual techniques:
improvedNet = technique2_mixed_training(trainedNet, optimal_params, roiDir, cxrDir, useGPU);
```

---

## Expected Results

| Technique | OOD Accuracy | Improvement | Retraining? |
|-----------|--------------|-------------|-------------|
| Baseline | ~55-60% | - | No |
| Enhanced Augmentation | ~60-70% | +5-10% | Yes |
| Mixed Training | ~70-80% | +15-25% | Yes |
| TTA | ~58-68% | +3-8% | **No** |
| Fine-tuning OOD | ~70-85% | +15-30% | Yes |
| Ensemble | ~75-85% | +20-30% | Yes |

---

## Key Insights

1. **Mixed Training is Best**: Training on both ROI and CXR gives best OOD performance.

2. **TTA is Easiest**: No retraining required, immediate improvement.

3. **Domain Shift is Significant**: ROI → Full CXR is a major distribution shift.

4. **Sensitivity Drops More**: Your sensitivity (TPR) drops significantly (~70% degradation).
   - Consider class weights during training
   - Use balanced sampling
   - Apply focal loss for imbalanced classes

---

## Additional Recommendations

### Address Sensitivity Issue
Your model's sensitivity (ability to detect positive cases) drops significantly:
- **ID Sensitivity**: 100%
- **OOD Sensitivity**: ~30%
- **Degradation**: ~70%

**Solutions**:
1. Use class-weighted loss
2. Apply focal loss for hard examples
3. Oversample positive class during training
4. Adjust decision threshold (lower for OOD)

### Data Preprocessing
Ensure consistent preprocessing:
- Both ID and OOD should use same preprocessing
- Current: single precision, [0, 255] range
- Check normalization consistency

### Evaluation Strategy
1. Evaluate on multiple OOD datasets
2. Monitor both ID and OOD performance
3. Use calibration to adjust confidence thresholds
4. Consider domain-specific metrics

---

## Code Usage

### Run Improvement Script
```matlab
improve_ood_accuracy();
```

### Use Specific Technique
```matlab
% Mixed training (recommended)
improvedNet = technique2_mixed_training(trainedNet, optimal_params, roiDir, cxrDir, useGPU);

% Test-time augmentation (quick)
results = technique4_test_time_augmentation(trainedNet, cxrDir, useGPU);
```

---

## Next Steps

1. **Start with TTA** (technique 4) - immediate improvement, no retraining
2. **Implement Mixed Training** (technique 2) - best long-term solution
3. **Monitor sensitivity** - address class imbalance issue
4. **Evaluate comprehensively** - test on multiple OOD datasets
5. **Iterate** - combine techniques for best results

---

## Questions?

- Which technique should I start with? → **Mixed Training (Technique 2)** or **TTA (Technique 4)**
- Do I need to retrain? → TTA doesn't require retraining
- How long does it take? → TTA: minutes, Mixed Training: hours
- Will ID performance drop? → Slight drop possible, but OOD improves significantly

