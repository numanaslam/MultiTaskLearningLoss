# ROI Alignment (Boxing Method) for Test-Time OOD Evaluation

## Concept

**The Idea**: Crop full CXR images to lung bounding box during test time, matching the training distribution (ROI images).

## How It Works

### Current Problem:
- **Training**: ROI images (52% non-zero pixels, cropped lung region)
- **Testing**: Full CXR images (98% non-zero pixels, full chest)
- **Result**: 32% degradation due to content/context shift

### Solution: ROI Alignment
1. **Detect lung region** in full CXR using adaptive thresholding + morphology
2. **Extract bounding box** of lung region
3. **Crop CXR** to this bounding box (with small margin)
4. **Resize** to target size (224x224)
5. **Apply preprocessing** (histmatch, etc.)
6. **Evaluate** with optional TTA

### Result:
- **Content shift eliminated**: 98% → ~52% non-zero pixels (matches ROI)
- **Spatial context matched**: Cropped lung region (matches training)
- **Expected**: Much lower degradation (<15%)

## Implementation

### Script: `evaluate_ood_roi_alignment.m`

**Usage:**
```matlab
% Default: histmatch preprocessing + TTA
evaluate_ood_roi_alignment()

% Custom preprocessing
evaluate_ood_roi_alignment('zscore', true, 12)

% No TTA
evaluate_ood_roi_alignment('histmatch', false)
```

### How It Works:

1. **Lung Detection** (lines 333-384):
   - Uses adaptive thresholding on enhanced CXR
   - Morphological operations (opening, closing, filling)
   - Extracts largest connected component (lung region)
   - Computes bounding box with margin

2. **Cropping**:
   - Crops CXR to bounding box
   - Adds 8% margin around lung region
   - Fallback to center crop if detection fails

3. **Preprocessing**:
   - Applies same preprocessing as training (histmatch, etc.)
   - Resizes to 224x224
   - Converts to RGB format

4. **Evaluation**:
   - Optional TTA support
   - Same evaluation metrics as standard OOD evaluation

## Expected Results

### Without ROI Alignment (Current):
- ID Accuracy: 85.1%
- OOD Accuracy: 57.5%
- **Degradation: 32.0%**

### With ROI Alignment (Expected):
- ID Accuracy: ~85%
- OOD Accuracy: **~75-80%** (estimated)
- **Degradation: <15%** (estimated)

### Why It Should Work:
1. ✅ **Content match**: 52% non-zero pixels (same as ROI)
2. ✅ **Spatial match**: Cropped lung region (same as ROI)
3. ✅ **Context match**: No background anatomy (same as ROI)
4. ⚠️ **Intensity**: Still needs preprocessing (histmatch)

## Advantages

1. **Eliminates content shift** - Main cause of degradation
2. **No retraining needed** - Works with existing model
3. **Automatic** - No manual annotation required
4. **Robust** - Fallback to center crop if detection fails
5. **Compatible** - Works with preprocessing + TTA

## Limitations

1. **Requires lung detection** - May fail on poor quality images
2. **Fallback needed** - Uses center crop if detection fails
3. **Margin estimation** - 8% margin may not be optimal for all cases
4. **Computational overhead** - Additional processing per image

## Comparison

| Method | Content Shift | Spatial Match | Degradation |
|--------|--------------|---------------|-------------|
| **Full CXR** | 98% non-zero | Full chest | 32% |
| **ROI Alignment** | ~52% non-zero | Cropped lung | **<15%** (expected) |

## Recommendation

**Use ROI Alignment for OOD Evaluation!**

This should dramatically reduce degradation by:
- Eliminating content shift (main issue)
- Matching spatial context
- Aligning with training distribution

**Next Steps:**
1. Run `evaluate_ood_roi_alignment()` with your trained model
2. Compare results with current 32% degradation
3. Expected improvement: Degradation <15%, OOD accuracy >75%

