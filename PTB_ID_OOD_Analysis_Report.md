# PTB Class ID/OOD Dataset Analysis Report

## Summary
**Status: ✓ SUITABLE FOR ID/OOD EVALUATION**

The PTB (Pulmonary Tuberculosis) class images in `input/resized/roi/ptb/` and `input/resized/cxr/ptb/` are clearly different and suitable for In-Distribution (ID) vs Out-of-Distribution (OOD) evaluation.

---

## Dataset Statistics

| Property | ROI (ID) | CXR (OOD) | Status |
|----------|----------|-----------|--------|
| **Number of Images** | 287 | 287 | ✓ Perfect match |
| **File Overlap** | 287/287 | 287/287 | ✓ All files match |
| **Image Size** | 227×227 | 227×227 | ✓ Consistent |
| **Size Consistency** | 100% | 100% | ✓ All images same size |

---

## Image Properties Comparison

### Intensity Statistics (Sample of 100 images)

| Metric | ROI (ID) | CXR (OOD) | Ratio (ROI/CXR) |
|--------|----------|-----------|-----------------|
| **Mean Intensity** | 30.58 ± 8.30 | 127.34 ± 29.39 | **0.24** (4.2× lower) |
| **Non-zero Pixels** | ~22-42% | 100% | **~0.3** (3× fewer) |
| **Intensity Range** | 0-242 | 0-249 | Similar range |

### Content Differences

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average MSE** | 18,107.47 ± 7,902.00 | Very high - images are clearly different |
| **Average Correlation** | -0.0421 ± 0.2019 | Near zero - no linear relationship |
| **Histogram Difference** | 164.77% | Very different intensity distributions |
| **Max Absolute Difference** | 241.00 pixels | Maximum possible difference |

---

## Sample Image Comparisons

### Sample 1: CHNCXR_0327_1.png
- **ROI**: Mean=51.53, Non-zero=41.5%, MSE=18,209
- **CXR**: Mean=147.32, Non-zero=100.0%
- **Intensity Ratio**: 0.35 (ROI is 2.9× darker)

### Sample 2: CHNCXR_0328_1.png
- **ROI**: Mean=21.08, Non-zero=22.7%, MSE=12,103
- **CXR**: Mean=99.90, Non-zero=100.0%
- **Intensity Ratio**: 0.21 (ROI is 4.7× darker)

### Sample 3: CHNCXR_0329_1.png
- **ROI**: Mean=38.93, Non-zero=33.3%, MSE=19,712
- **CXR**: Mean=145.49, Non-zero=100.0%
- **Intensity Ratio**: 0.27 (ROI is 3.7× darker)

---

## Key Findings

### ✓ Suitable for ID/OOD Evaluation

1. **Clear Content Differences**
   - ROI images are cropped lung regions (lower intensity, fewer non-zero pixels)
   - CXR images are full chest X-rays (higher intensity, all pixels active)
   - High MSE (18,107) indicates substantial pixel-level differences
   - Near-zero correlation (-0.04) confirms different image content

2. **Consistent Preprocessing**
   - All images resized to 227×227
   - Consistent file naming and structure
   - Perfect file matching between directories

3. **Proper Distribution Shift**
   - ROI (ID): Cropped regions with mean intensity ~30.58
   - CXR (OOD): Full images with mean intensity ~127.34
   - Intensity ratio of 0.24 indicates clear domain shift

---

## Recommendations

### ✓ Use for ID/OOD Evaluation

**ID (In-Distribution) Dataset:**
```
input/resized/roi/ptb/
```
- 287 PTB images
- Cropped lung regions
- Mean intensity: ~30.58
- Non-zero pixels: ~22-42%

**OOD (Out-of-Distribution) Dataset:**
```
input/resized/cxr/ptb/
```
- 287 PTB images
- Full chest X-rays
- Mean intensity: ~127.34
- Non-zero pixels: 100%

### Evaluation Strategy

1. **Train model on ROI images** (ID distribution)
2. **Evaluate on CXR images** (OOD distribution)
3. **Measure performance degradation** to assess generalization
4. **Expected degradation**: ~30-50% based on previous analysis

---

## Comparison with Normal Class

| Property | Normal Class | PTB Class | Status |
|----------|--------------|-----------|--------|
| **ROI Mean Intensity** | 22.70 | 30.58 | PTB slightly brighter |
| **CXR Mean Intensity** | 84.28 | 127.34 | PTB significantly brighter |
| **Intensity Ratio** | 0.27 | 0.24 | Similar ratio |
| **MSE** | 8,459 | 18,107 | PTB shows larger differences |
| **File Count** | 279 | 287 | Similar counts |

**Note**: PTB images show larger intensity differences between ROI and CXR, which may indicate:
- More pronounced disease features in full CXR images
- Different preprocessing or acquisition parameters
- Stronger domain shift for PTB class

---

## Conclusion

**✓ The PTB class datasets are ready for ID/OOD evaluation.**

The clear differences between ROI (cropped lung regions) and CXR (full chest X-rays) images, combined with consistent preprocessing and perfect file matching, make these datasets ideal for:
- Measuring model generalization
- Assessing domain shift
- Evaluating OOD performance
- Comparing ID vs OOD metrics

**Next Steps:**
1. Use `input/resized/roi/ptb/` for ID evaluation
2. Use `input/resized/cxr/ptb/` for OOD evaluation
3. Run evaluation scripts with these directories
4. Compare results with Normal class for comprehensive analysis

