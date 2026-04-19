# Out-of-Distribution (OOD) Test Script

## Overview

The `test_ood_full_cxr.m` script evaluates your model (trained on ROI images) on full chest X-ray images to assess generalization and distribution shift impact.

## What It Does

1. **Loads the trained model** (`vgg16_finetuned_on_roi.mat`)
2. **Tests on two datasets**:
   - **In-Distribution (ID)**: ROI images (same as training data)
   - **Out-of-Distribution (OOD)**: Full CXR images (different from training data)
3. **Compares performance** between ID and OOD
4. **Generates comprehensive visualizations** and analysis

## Usage

Simply run in MATLAB:

```matlab
test_ood_full_cxr()
```

## Requirements

- Trained model file: `vgg16_finetuned_on_roi.mat`
- ROI images in: `input/roi/`
- Full CXR images in: `input/cxr/`
- Masks in: `input/masks/` (optional, for segmentation metrics)

## Output

The script creates an `ood_results/` directory with:

### Figures (`ood_results/figures/`):
1. **`performance_comparison.png`** - Side-by-side comparison of all metrics
2. **`confusion_matrices.png`** - Confusion matrices for ID and OOD
3. **`roc_comparison.png`** - ROC curves comparing ID vs OOD
4. **`distribution_shift.png`** - Analysis of prediction distribution shifts
5. **`sample_visualizations.png`** - Sample images with predictions

### Data Files:
1. **`ood_evaluation_results.mat`** - Complete results in MATLAB format
2. **`detailed_comparison.txt`** - Human-readable summary table

## Metrics Evaluated

- **Accuracy**: Overall classification correctness
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

## Performance Degradation Analysis

The script calculates:
- **Percentage degradation** for each metric
- **Average degradation** across all metrics
- **Distribution shift** indicators

## Interpretation

- **Low degradation (<5%)**: Model generalizes well to full CXR
- **Moderate degradation (5-10%)**: Some domain shift, but acceptable
- **High degradation (>10%)**: Significant distribution shift, may need adaptation

## Recommendations Based on Results

- **If degradation is high**: Consider fine-tuning on full CXR images
- **If degradation is moderate**: Domain adaptation techniques may help
- **If degradation is low**: Model is robust and ready for deployment

## Example Output

```
=== OUT-OF-DISTRIBUTION EVALUATION ===
Model: Trained on ROI images
Test: Full CXR images (OOD) + ROI images (ID) for comparison

Loading trained model...
Model loaded successfully.

Loading datasets...
ROI dataset (ID): 566 samples
Full CXR dataset (OOD): 566 samples
Classes: normal, ptb

=== EVALUATING ON IN-DISTRIBUTION DATA (ROI) ===
In-Distribution (ROI) Results:
  Accuracy: 0.973
  Sensitivity: 0.975
  Specificity: 0.970
  Precision: 0.970
  F1-Score: 0.973
  AUC: 0.991

=== EVALUATING ON OUT-OF-DISTRIBUTION DATA (Full CXR) ===
Out-of-Distribution (Full CXR) Results:
  Accuracy: 0.945
  Sensitivity: 0.948
  Specificity: 0.942
  Precision: 0.942
  F1-Score: 0.945
  AUC: 0.978

=== COMPARATIVE ANALYSIS ===
ACCURACY: ID=0.973, OOD=0.945, Degradation=2.88%
SENSITIVITY: ID=0.975, OOD=0.948, Degradation=2.77%
SPECIFICITY: ID=0.970, OOD=0.942, Degradation=2.89%
PRECISION: ID=0.970, OOD=0.942, Degradation=2.89%
F1-SCORE: ID=0.973, OOD=0.945, Degradation=2.88%
AUC: ID=0.991, OOD=0.978, Degradation=1.31%

Average Performance Degradation: 2.61%
```

## Notes

- The script automatically handles GPU/CPU selection
- All visualizations are saved in high resolution
- Results are timestamped for reproducibility
- The script includes error handling for missing files or toolboxes

