# Preprocessing Evaluation Results Analysis

## Summary

**Best Method**: Histogram matching to ROI reference
- **ID Accuracy**: 77.3%
- **OOD Accuracy**: 57.2%
- **Mean Degradation**: 22.86%
- **Improvement over baseline**: 12.31% reduction in degradation

## Detailed Results

| Method | ID Accuracy | OOD Accuracy | Degradation | Improvement |
|--------|-------------|--------------|-------------|-------------|
| **No preprocessing (baseline)** | 82.5% | 55.0% | 35.17% | - |
| **Histogram matching** | 77.3% | 57.2% | **22.86%** | **+12.31%** |
| Min-max normalization | 82.4% | 55.0% | 33.97% | +1.20% |
| Z-score normalization | 81.7% | 54.5% | 34.34% | +0.83% |
| CLAHE | 78.8% | 51.7% | 35.63% | -0.46% |
| CLAHE + Z-score | 78.1% | 51.7% | 35.18% | -0.01% |

## Key Observations

### 1. Histogram Matching is the Clear Winner
- **Reduced degradation by 12.31%** (from 35.17% to 22.86%)
- **Improved OOD accuracy by 2.2%** (from 55.0% to 57.2%)
- This confirms that intensity distribution alignment helps, but **does not fully solve the OOD problem**

### 2. Trade-off: ID vs OOD Performance
- Histogram matching slightly **reduced ID accuracy** (82.5% → 77.3%)
- This is a **worthwhile trade-off** for better OOD generalization
- The 5.2% ID drop is acceptable given the 12.3% degradation reduction

### 3. Other Methods Showed Limited Benefit
- Min-max and Z-score normalization: Minimal improvement (~1%)
- CLAHE: Actually **worsened** performance slightly
- Combined methods: No additional benefit

### 4. Remaining Challenge: 22.86% Degradation
- Even with the best preprocessing, degradation remains **high (>20%)**
- This suggests that **intensity differences alone are not the main issue**
- The **content/context shift** (cropped lung vs. full chest) is likely the primary cause

## Comparison with ROI Alignment (Boxing Method)

### Current Results (Preprocessing Only)
- **OOD Accuracy**: 57.2%
- **Degradation**: 22.86%
- **Status**: Moderate improvement, but still significant degradation

### Expected Results (ROI Alignment)
- **OOD Accuracy**: 75-80% (estimated)
- **Degradation**: <15% (estimated)
- **Status**: Should dramatically improve OOD performance

### Why ROI Alignment Should Work Better

1. **Addresses Content Shift**:
   - Preprocessing: Only fixes intensity distribution
   - ROI Alignment: Fixes both intensity AND spatial content (cropped lung region)

2. **Matches Training Distribution**:
   - Training: ROI images (52% non-zero, cropped lung)
   - Current OOD: Full CXR (98% non-zero, full chest)
   - With ROI Alignment: Cropped CXR (52% non-zero, cropped lung) ✅

3. **Spatial Context Alignment**:
   - The model was trained on cropped lung regions
   - Full CXR images contain additional context (ribs, heart, etc.) that the model hasn't seen
   - ROI alignment removes this extraneous context

## Recommendations

### 1. **Immediate Next Step: Run ROI Alignment Evaluation**
   - Use `evaluate_ood_with_roi_alignment.m` to test the boxing method
   - Expected: <15% degradation, 75-80% OOD accuracy
   - This should provide the **biggest improvement**

### 2. **Combine Approaches (If Needed)**
   - If ROI alignment alone doesn't reach <15% degradation:
     - Use ROI alignment + histogram matching preprocessing
     - This addresses both content shift AND intensity differences

### 3. **Training Strategy (Long-term)**
   - **Mixed training**: Train on both ROI and full CXR images
   - **Domain adaptation**: Fine-tune on full CXR after ROI training
   - **Data augmentation**: Include full CXR crops during training

### 4. **Current Best Practice**
   - **For evaluation**: Use histogram matching preprocessing (current best)
   - **For deployment**: Consider ROI alignment at test time
   - **For training**: Continue using ROI images with histogram matching

## Conclusion

**Preprocessing alone is insufficient** to fully address OOD degradation. While histogram matching provides a **12.31% improvement**, the remaining **22.86% degradation** indicates that the **content/context shift** (cropped lung vs. full chest) is the primary issue.

**ROI alignment (boxing method) should be the next evaluation** as it directly addresses the spatial content mismatch, potentially reducing degradation to <15% and improving OOD accuracy to 75-80%.

