# Intensity Normalization Guide for OOD Evaluation

## Problem

CXR (OOD) images have **2.14x higher intensity** than ROI (ID) images:
- **ROI mean intensity**: ~73.57
- **CXR mean intensity**: ~157.64
- **Intensity ratio**: 2.14x

This intensity difference contributes to performance degradation in OOD evaluation.

## Solution

The `extract_roi_with_mask.py` script now supports intensity normalization to reduce this difference.

## Normalization Methods

### 1. **'none'** (Default)
- **No normalization** - preserves original intensities
- Use when you want to test raw distribution shift
- **Current setting**: `intensity_normalization = 'none'`

### 2. **'histmatch'** (Recommended for OOD)
- **Histogram matching** - matches CXR histogram to ROI histogram
- Best for reducing intensity distribution shift
- Preserves relative intensity relationships
- **Usage**: `intensity_normalization = 'histmatch'`

### 3. **'zscore'**
- **Z-score normalization**: (x - mean) / std
- Normalizes to zero mean and unit variance, then scales to [0, 255]
- Good for standardizing intensity distributions
- **Usage**: `intensity_normalization = 'zscore'`

### 4. **'minmax'**
- **Min-max normalization**: (x - min) / (max - min) * 255
- Scales to [0, 255] range
- Simple and effective
- **Usage**: `intensity_normalization = 'minmax'`

### 5. **'percentile'**
- **Percentile-based normalization** (robust to outliers)
- Uses 1st and 99th percentiles instead of min/max
- More robust to outliers than minmax
- **Usage**: `intensity_normalization = 'percentile'`

### 6. **'lung_region'**
- **Lung region-based normalization**
- Computes statistics from lung region only, then normalizes entire image
- Focuses on relevant anatomical region
- **Usage**: `intensity_normalization = 'lung_region'`

## How to Use

### Step 1: Edit the script
Open `extract_roi_with_mask.py` and find line ~453:

```python
intensity_normalization = 'none'  # Change this
```

### Step 2: Choose a method
For best OOD performance, try:
```python
intensity_normalization = 'histmatch'  # Match CXR to ROI histogram
```

Or for simpler normalization:
```python
intensity_normalization = 'zscore'  # Z-score normalization
```

### Step 3: Re-run the script
```bash
python extract_roi_with_mask.py
```

This will regenerate all images with normalized intensities.

## Expected Results

After normalization, you should see:
- **Reduced intensity difference** between CXR and ROI
- **Improved OOD performance** (lower degradation)
- **More consistent intensity distributions**

## Verification

After running with normalization, check the intensity statistics:

```python
from PIL import Image
import numpy as np

# Load sample images
roi = np.array(Image.open('input/resized/roi/normal/MCUCXR_0023_0.png'))
cxr = np.array(Image.open('input/resized/cxr/normal/MCUCXR_0023_0.png'))

print(f"ROI mean: {roi.mean():.2f}")
print(f"CXR mean: {cxr.mean():.2f}")
print(f"Intensity ratio: {cxr.mean()/roi.mean():.2f}x")
```

**Target**: Intensity ratio should be close to **1.0x** (instead of 2.14x)

## Recommendations

1. **For OOD evaluation**: Use `'histmatch'` or `'zscore'`
2. **For preserving original data**: Use `'none'`
3. **For robust normalization**: Use `'percentile'`
4. **For anatomical focus**: Use `'lung_region'`

## Notes

- Normalization is applied **only to CXR images** (OOD)
- ROI images (ID) remain unchanged to preserve training data
- All normalization methods output uint8 images in [0, 255] range
- The script automatically handles mask-based normalization when masks are available

