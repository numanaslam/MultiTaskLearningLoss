# Expected OOD Distribution Analysis

## Summary

After dataset regeneration, here's what to expect for OOD (Out-of-Distribution) evaluation:

## Distribution Shift Characteristics

### 1. **Intensity Distribution Shift**

#### Normal Class:
- **ROI (ID)**: Mean = 67.4, Median = 36.0, Std = 72.9
- **CXR (OOD)**: Mean = 147.1, Median = 168.0, Std = 74.6
- **Shift**: +79.7 intensity units (2.18x brighter)
- **Distribution Overlap**: 45.8%

#### PTB Class:
- **ROI (ID)**: Mean = 66.3, Median = 55.0, Std = 69.7
- **CXR (OOD)**: Mean = 148.0, Median = 165.0, Std = 68.9
- **Shift**: +81.8 intensity units (2.23x brighter)
- **Distribution Overlap**: 44.8%

### 2. **Content Distribution Shift**

- **ROI (ID)**: ~52-53% non-zero pixels (cropped to lung region)
- **CXR (OOD)**: ~95-99% non-zero pixels (full chest image)
- **Shift**: +42-45% more content (background + anatomy)

### 3. **Statistical Properties**

| Property | ROI (ID) | CXR (OOD) | Shift |
|----------|----------|-----------|-------|
| **Mean Intensity** | 66-67 | 147-148 | +80 (+2.2x) |
| **Median Intensity** | 36-55 | 165-168 | +110-130 |
| **Standard Deviation** | 69-73 | 69-75 | Similar (±1x) |
| **Intensity Range** | [0, 252-255] | [0, 255] | Similar |
| **IQR (25th-75th)** | [0, 128-141] | [96-101, 206-209] | Higher baseline |

## Expected OOD Performance Impact

### 1. **Performance Degradation Factors**

#### ✅ **Moderate Impact** (Expected):
- **Intensity shift (2.2x)**: Model trained on darker images, tested on brighter
- **Content shift (45% more pixels)**: Model sees full chest vs. cropped lung region
- **Distribution overlap (~45%)**: Some overlap, but significant shift

#### ⚠️ **High Impact** (Potential):
- **Median shift**: ROI median (36-55) vs CXR median (165-168) - very different
- **IQR shift**: ROI has many zeros, CXR has few zeros
- **Context change**: Full anatomical context vs. focused lung region

### 2. **Expected Metrics**

Based on typical OOD evaluation with this level of distribution shift:

- **Accuracy degradation**: 10-25% (typical for 2x intensity shift)
- **Confidence drop**: 15-30% (model less confident on OOD)
- **Class-specific impact**: 
  - Normal class may be more affected (larger shift)
  - PTB class may be more robust (disease features more prominent)

### 3. **Distribution Characteristics**

```
ROI (ID) Distribution:
├── Bimodal: Many zeros (background) + lung region intensities
├── Skewed left: Median (36-55) < Mean (66-67)
└── Sparse: 52-53% non-zero pixels

CXR (OOD) Distribution:
├── More uniform: Few zeros, more consistent intensities
├── Skewed right: Median (165-168) > Mean (147-148)
└── Dense: 95-99% non-zero pixels
```

## Visualization

A histogram comparison plot has been saved to `ood_distribution_analysis.png` showing:
- Overlapping intensity distributions
- Mean intensity markers
- Clear separation between ID and OOD

## Recommendations

### 1. **For OOD Evaluation**:
- **Expected**: Moderate to high performance degradation (10-25%)
- **Acceptable**: <15% degradation indicates good generalization
- **Concerning**: >25% degradation suggests overfitting to ROI characteristics

### 2. **To Improve OOD Performance**:
- **Option 1**: Apply intensity normalization (`histmatch` or `zscore`)
- **Option 2**: Fine-tune on mixed ID/OOD data
- **Option 3**: Use domain adaptation techniques
- **Option 4**: Train on full CXR images instead of ROI

### 3. **For Research Reporting**:
- Document the distribution shift clearly
- Report both ID and OOD metrics
- Explain that degradation is expected due to:
  - 2.2x intensity difference
  - 45% content difference
  - Different anatomical context

## Key Takeaways

1. **Significant Distribution Shift**: 2.2x intensity difference + 45% content difference
2. **Moderate Overlap**: ~45% distribution overlap (some similarity)
3. **Realistic OOD Scenario**: Tests true generalization to full chest images
4. **Expected Degradation**: 10-25% performance drop is normal and acceptable
5. **Normalization Available**: Can reduce shift using built-in normalization methods

## Next Steps

1. Run OOD evaluation to measure actual performance
2. Compare with expected degradation (10-25%)
3. If degradation >25%, consider applying intensity normalization
4. Document findings in research paper

