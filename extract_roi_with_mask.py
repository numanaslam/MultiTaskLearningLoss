#!/usr/bin/env python3
"""
EXTRACT_ROI_WITH_MASK Extract ROI images from CXR using masks while preserving original intensities

This script:
1. Loads CXR images and corresponding masks
2. Applies mask to CXR to create ROI (sets non-lung regions to 0, preserves lung region intensities)
3. Preserves original intensity values from input CXR images
4. Optionally crops to bounding box or keeps full size
5. Saves ROI images
"""

import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

# Fix deprecation warnings
try:
    from PIL.Image import Resampling
    BICUBIC = Resampling.BICUBIC
    NEAREST = Resampling.NEAREST
except ImportError:
    # Fallback for older PIL versions
    BICUBIC = Image.BICUBIC
    NEAREST = Image.NEAREST

def load_image_preserve_intensity(image_path):
    """
    Load image and preserve original intensity values (no normalization)
    Returns image as numpy array with original dtype and values
    """
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Return as numpy array preserving original values
    return np.array(img)

def load_mask_binary(mask_path):
    """
    Load mask and convert to binary (True/False)
    Handles various mask formats (uint8, bool, etc.)
    """
    mask = Image.open(mask_path)
    
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    mask_array = np.array(mask)
    
    # Binarize: threshold at 128
    mask_binary = (mask_array > 128).astype(bool)
    
    return mask_binary

def resize_image_preserve_aspect_ratio(image, target_size, interpolation=BICUBIC):
    """
    Resize image while preserving aspect ratio.
    Pads with zeros if needed to match target size.
    
    Args:
        image: PIL Image or numpy array
        target_size: (width, height) tuple
        interpolation: PIL interpolation method
    
    Returns:
        Resized image as numpy array
    """
    if isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image.astype(np.uint8))
    else:
        img_pil = image
    
    original_width, original_height = img_pil.size
    target_width, target_height = target_size
    
    # Calculate scaling factor to preserve aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    
    # Calculate new size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    img_resized = img_pil.resize((new_width, new_height), interpolation)
    
    # Create new image with target size and paste resized image in center
    img_final = Image.new('L', (target_width, target_height), 0)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    img_final.paste(img_resized, (paste_x, paste_y))
    
    return np.array(img_final, dtype=image.dtype if isinstance(image, np.ndarray) else np.uint8)

def resize_image_fill(image, target_size, interpolation=BICUBIC):
    """
    Resize image to fill target size without preserving aspect ratio.
    Stretches image to fill entire target size (no black borders).
    
    Args:
        image: PIL Image or numpy array
        target_size: (width, height) tuple
        interpolation: PIL interpolation method
    
    Returns:
        Resized image as numpy array
    """
    if isinstance(image, np.ndarray):
        img_pil = Image.fromarray(image.astype(np.uint8))
    else:
        img_pil = image
    
    target_width, target_height = target_size
    
    # Resize directly to target size (stretches to fill, no padding)
    img_resized = img_pil.resize((target_width, target_height), interpolation)
    
    return np.array(img_resized, dtype=image.dtype if isinstance(image, np.ndarray) else np.uint8)

def extract_roi_with_mask(cxr_image, mask_image, crop_to_bbox=True, padding=0):
    """
    Overlay mask on original CXR image and extract overlapping ROI.
    
    Args:
        cxr_image: CXR image as numpy array (preserves original intensities)
        mask_image: Binary mask (True = lung region, False = background)
        crop_to_bbox: If True, crop to bounding box of mask (zoomed-in ROI). 
                      If False, keep full size (same as mask size).
        padding: Padding around bounding box (in pixels). Only used if crop_to_bbox=True.
    
    Returns:
        roi_image: ROI image containing CXR values where mask is True
        mask_cropped: Cropped mask (if crop_to_bbox=True) or original mask
        cxr_cropped: Cropped CXR (if crop_to_bbox=True) or resized CXR
    """
    # Ensure mask is binary
    if mask_image.dtype != bool:
        mask_binary = (mask_image > 128).astype(bool)
    else:
        mask_binary = mask_image
    
    # Get mask size
    mask_height, mask_width = mask_binary.shape
    
    # Resize CXR image to match mask size using bicubic interpolation to preserve intensities
    if cxr_image.shape != mask_binary.shape:
        cxr_pil = Image.fromarray(cxr_image.astype(np.uint8))
        cxr_resized = cxr_pil.resize((mask_width, mask_height), Image.BICUBIC)
        cxr_resized = np.array(cxr_resized, dtype=cxr_image.dtype)
    else:
        cxr_resized = cxr_image.copy()
    
    # Overlay mask: ROI = CXR where mask is True, 0 elsewhere
    roi_image = cxr_resized.copy().astype(cxr_image.dtype)
    roi_image[~mask_binary] = 0
    
    if crop_to_bbox:
        # Get bounding box of mask
        if np.any(mask_binary):
            rows, cols = np.where(mask_binary)
            min_row = max(0, rows.min() - padding)
            max_row = min(mask_height, rows.max() + padding + 1)
            min_col = max(0, cols.min() - padding)
            max_col = min(mask_width, cols.max() + padding + 1)
            
            # Crop ROI, mask, and CXR to bounding box
            roi_cropped = roi_image[min_row:max_row, min_col:max_col]
            mask_cropped = mask_binary[min_row:max_row, min_col:max_col]
            cxr_cropped = cxr_resized[min_row:max_row, min_col:max_col]
            
            return roi_cropped, mask_cropped, cxr_cropped
        else:
            # Empty mask - return original
            return roi_image, mask_binary, cxr_resized
    else:
        # Keep full size
        return roi_image, mask_binary, cxr_resized

def normalize_intensity_histogram_match(source_image, reference_image, mask=None):
    """
    Match histogram of source image to reference image.
    If mask is provided, only match within masked region.
    
    Args:
        source_image: Image to normalize (numpy array)
        reference_image: Reference image for histogram matching (numpy array)
        mask: Optional binary mask (True = region to use for matching)
    
    Returns:
        Normalized image as numpy array
    """
    from scipy import ndimage
    
    source = source_image.astype(np.float64)
    reference = reference_image.astype(np.float64)
    
    if mask is not None:
        # Use only masked regions for histogram matching
        source_values = source[mask].flatten()
        reference_values = reference[mask].flatten()
    else:
        source_values = source.flatten()
        reference_values = reference.flatten()
    
    # Compute histograms
    hist_source, bin_edges_source = np.histogram(source_values, bins=256, range=(0, 256))
    hist_reference, bin_edges_reference = np.histogram(reference_values, bins=256, range=(0, 256))
    
    # Compute cumulative distribution functions
    cdf_source = np.cumsum(hist_source).astype(np.float64)
    cdf_reference = np.cumsum(hist_reference).astype(np.float64)
    
    # Normalize CDFs
    cdf_source = cdf_source / cdf_source[-1] if cdf_source[-1] > 0 else cdf_source
    cdf_reference = cdf_reference / cdf_reference[-1] if cdf_reference[-1] > 0 else cdf_reference
    
    # Create mapping function
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find closest reference CDF value
        idx = np.argmin(np.abs(cdf_reference - cdf_source[i]))
        mapping[i] = idx
    
    # Apply mapping
    normalized = mapping[source.astype(np.uint8)]
    
    return normalized.astype(source_image.dtype)

def normalize_intensity_zscore(image, mask=None):
    """
    Z-score normalization: (x - mean) / std
    Normalizes to zero mean and unit variance, then scales to [0, 255]
    
    Args:
        image: Image to normalize (numpy array)
        mask: Optional binary mask (True = region to use for statistics)
    
    Returns:
        Normalized image as numpy array (uint8, [0, 255])
    """
    img_float = image.astype(np.float64)
    
    if mask is not None:
        mean = np.mean(img_float[mask])
        std = np.std(img_float[mask])
    else:
        mean = np.mean(img_float)
        std = np.std(img_float)
    
    if std > 0:
        normalized = (img_float - mean) / std
        # Scale to [0, 255] range
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8) * 255
    else:
        normalized = img_float
    
    return np.clip(normalized, 0, 255).astype(np.uint8)

def normalize_intensity_minmax(image, mask=None):
    """
    Min-max normalization: (x - min) / (max - min) * 255
    Scales to [0, 255] range
    
    Args:
        image: Image to normalize (numpy array)
        mask: Optional binary mask (True = region to use for statistics)
    
    Returns:
        Normalized image as numpy array (uint8, [0, 255])
    """
    img_float = image.astype(np.float64)
    
    if mask is not None:
        min_val = np.min(img_float[mask])
        max_val = np.max(img_float[mask])
    else:
        min_val = np.min(img_float)
        max_val = np.max(img_float)
    
    if max_val > min_val:
        normalized = (img_float - min_val) / (max_val - min_val) * 255
    else:
        normalized = img_float
    
    return np.clip(normalized, 0, 255).astype(np.uint8)

def normalize_intensity_percentile(image, mask=None, lower_percentile=1, upper_percentile=99):
    """
    Percentile-based normalization (robust to outliers).
    Uses percentiles instead of min/max to handle outliers.
    
    Args:
        image: Image to normalize (numpy array)
        mask: Optional binary mask (True = region to use for statistics)
        lower_percentile: Lower percentile (default: 1)
        upper_percentile: Upper percentile (default: 99)
    
    Returns:
        Normalized image as numpy array (uint8, [0, 255])
    """
    img_float = image.astype(np.float64)
    
    if mask is not None:
        values = img_float[mask].flatten()
    else:
        values = img_float.flatten()
    
    p_low = np.percentile(values, lower_percentile)
    p_high = np.percentile(values, upper_percentile)
    
    if p_high > p_low:
        normalized = np.clip((img_float - p_low) / (p_high - p_low) * 255, 0, 255)
    else:
        normalized = img_float
    
    return normalized.astype(np.uint8)

def normalize_intensity_lung_region(image, mask):
    """
    Normalize based on lung region statistics only.
    Computes mean and std from lung region, then normalizes entire image.
    
    Args:
        image: Image to normalize (numpy array)
        mask: Binary mask (True = lung region)
    
    Returns:
        Normalized image as numpy array (uint8, [0, 255])
    """
    if not np.any(mask):
        # No lung region, return original
        return image.astype(np.uint8)
    
    img_float = image.astype(np.float64)
    lung_values = img_float[mask].flatten()
    
    mean = np.mean(lung_values)
    std = np.std(lung_values)
    
    if std > 0:
        # Normalize entire image using lung region statistics
        normalized = (img_float - mean) / std
        # Scale to [0, 255]
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8) * 255
    else:
        normalized = img_float
    
    return np.clip(normalized, 0, 255).astype(np.uint8)

def normalize_intensity(image, method='none', reference_image=None, mask=None):
    """
    Apply intensity normalization using specified method.
    
    Args:
        image: Image to normalize (numpy array)
        method: Normalization method:
            - 'none': No normalization (preserve original intensities)
            - 'histmatch': Histogram matching to reference image
            - 'zscore': Z-score normalization
            - 'minmax': Min-max normalization
            - 'percentile': Percentile-based normalization
            - 'lung_region': Normalize based on lung region statistics
        reference_image: Reference image for histogram matching (required if method='histmatch')
        mask: Binary mask for lung region (optional, used by some methods)
    
    Returns:
        Normalized image as numpy array
    """
    if method == 'none':
        return image.astype(np.uint8)
    elif method == 'histmatch':
        if reference_image is None:
            raise ValueError("reference_image required for histogram matching")
        return normalize_intensity_histogram_match(image, reference_image, mask)
    elif method == 'zscore':
        return normalize_intensity_zscore(image, mask)
    elif method == 'minmax':
        return normalize_intensity_minmax(image, mask)
    elif method == 'percentile':
        return normalize_intensity_percentile(image, mask)
    elif method == 'lung_region':
        if mask is None:
            raise ValueError("mask required for lung_region normalization")
        return normalize_intensity_lung_region(image, mask)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def find_mask_file(masks_dir, image_name):
    """Find mask file with various naming conventions"""
    name_base = os.path.splitext(image_name)[0]
    ext = os.path.splitext(image_name)[1]
    
    # Try different mask naming patterns
    mask_patterns = [
        f"{name_base}_mask{ext}",
        f"{name_base}_mask.png",
        image_name,
        f"{image_name}.png"
    ]
    
    # Check in root masks directory
    for pattern in mask_patterns:
        mask_path = os.path.join(masks_dir, pattern)
        if os.path.exists(mask_path):
            return mask_path
    
    # Check in subdirectories
    if os.path.isdir(masks_dir):
        for subdir in os.listdir(masks_dir):
            subdir_path = os.path.join(masks_dir, subdir)
            if os.path.isdir(subdir_path):
                for pattern in mask_patterns:
                    mask_path = os.path.join(subdir_path, pattern)
                    if os.path.exists(mask_path):
                        return mask_path
    
    return None

def main():
    print("=== EXTRACTING ROI IMAGES USING MASKS ===\n")
    print("Reading from original data sources\n")
    print("Preserving original intensity values and aspect ratios\n")
    
    # Configuration - Original data sources
    original_cxr_dir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/CXR_png'
    original_masks_dir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/masks'
    
    # Output directory
    output_base_dir = os.path.join('input', 'resized')
    
    # Output subdirectories
    output_cxr_dir = os.path.join(output_base_dir, 'cxr')
    output_roi_dir = os.path.join(output_base_dir, 'roi')
    output_masks_dir = os.path.join(output_base_dir, 'masks')
    
    # Options
    crop_to_bbox = True  # Default: zoomed-in ROI (bounding box crop)
    padding = 0  # No padding by default (exact bounding box)
    target_size = (227, 227)  # Target size for resizing (maintains aspect ratio)
    
    # Intensity normalization options
    # Options: 'none', 'histmatch', 'zscore', 'minmax', 'percentile', 'lung_region'
    # 'histmatch': Match CXR histogram to ROI histogram (best for OOD)
    # 'zscore': Z-score normalization (zero mean, unit variance)
    # 'minmax': Min-max normalization to [0, 255]
    # 'percentile': Percentile-based normalization (robust to outliers)
    # 'lung_region': Normalize based on lung region statistics only
    intensity_normalization = 'none'  # Change to 'histmatch' or 'zscore' to reduce intensity difference
    
    print(f"Original CXR directory: {original_cxr_dir}")
    print(f"Original masks directory: {original_masks_dir}")
    print(f"Target size: {target_size[0]} x {target_size[1]} (stretched to fill, no black borders)")
    print(f"CXR images: Full-size (no cropping)")
    print(f"ROI images: {'Zoomed-in (bounding box crop)' if crop_to_bbox else 'Full-size (no cropping)'}")
    if crop_to_bbox:
        print(f"Padding: {padding} pixels")
    print(f"Intensity normalization: {intensity_normalization}")
    if intensity_normalization != 'none':
        print(f"  → This will normalize CXR intensities to reduce OOD degradation")
    print()
    
    # Create output directories
    classes = ['normal', 'ptb']
    for class_name in classes:
        for output_dir in [output_cxr_dir, output_roi_dir, output_masks_dir]:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total': 0,
        'processed': 0,
        'roi_created': 0,
        'cxr_saved': 0,
        'masks_saved': 0,
        'errors': 0,
        'missing_cxr': 0,
        'missing_mask': 0,
        'empty_mask': 0
    }
    
    # Get all CXR images from original directory (no class subdirectories)
    if not os.path.exists(original_cxr_dir):
        print(f"ERROR: CXR directory not found: {original_cxr_dir}")
        return
    
    # Get all PNG files from original CXR directory
    all_cxr_files = glob.glob(os.path.join(original_cxr_dir, '*.png'))
    stats['total'] = len(all_cxr_files)
    
    print(f"Found {len(all_cxr_files)} CXR images in original directory\n")
    
    # Process each CXR image
    for cxr_path in tqdm(all_cxr_files, desc="Processing images"):
        try:
            img_name = os.path.basename(cxr_path)
            
            # Determine class from filename (assuming naming convention)
            # You may need to adjust this based on your actual naming
            if '_0.png' in img_name or 'normal' in img_name.lower():
                class_name = 'normal'
            elif '_1.png' in img_name or 'ptb' in img_name.lower():
                class_name = 'ptb'
            else:
                # Try to infer from directory structure or default to normal
                class_name = 'normal'
            
            # Output directories for this class
            output_cxr_class_dir = os.path.join(output_cxr_dir, class_name)
            output_roi_class_dir = os.path.join(output_roi_dir, class_name)
            output_masks_class_dir = os.path.join(output_masks_dir, class_name)
            
            # Find corresponding mask
            mask_path = find_mask_file(original_masks_dir, img_name)
            
            if mask_path is None or not os.path.exists(mask_path):
                print(f"    Warning: Mask not found for {img_name}")
                stats['missing_mask'] += 1
                continue
            
            # Load CXR image (preserve original intensities)
            cxr_image_original = load_image_preserve_intensity(cxr_path)
            
            # Load mask
            mask_image_original = load_mask_binary(mask_path)
            
            # Check if mask is empty
            if not np.any(mask_image_original):
                print(f"    Warning: Empty mask for {img_name}")
                stats['empty_mask'] += 1
                # Continue anyway - will create ROI with all zeros
            
            # Extract ROI from original images first (preserve original intensities)
            # Note: CXR should remain full-size, only ROI gets cropped
            roi_image_original, mask_cropped_original, _ = extract_roi_with_mask(
                cxr_image_original, mask_image_original, crop_to_bbox=crop_to_bbox, padding=padding
            )
            
            # Resize CXR: full-size original image (no cropping)
            cxr_final = resize_image_fill(cxr_image_original, target_size, BICUBIC)
            
            # Resize ROI: cropped to bounding box (if crop_to_bbox=True)
            roi_final = resize_image_fill(roi_image_original, target_size, BICUBIC)
            
            # Resize mask: cropped to match ROI bounding box
            mask_final = resize_image_fill(
                mask_cropped_original.astype(np.uint8) * 255, target_size, NEAREST
            )
            mask_final = (mask_final > 128).astype(bool)
            
            # Apply intensity normalization to CXR (to reduce OOD degradation)
            if intensity_normalization != 'none':
                if intensity_normalization == 'histmatch':
                    # Histogram matching: match CXR to ROI histogram
                    cxr_final = normalize_intensity(
                        cxr_final, 
                        method='histmatch', 
                        reference_image=roi_final,
                        mask=mask_final if np.any(mask_final) else None
                    )
                elif intensity_normalization == 'lung_region':
                    # Normalize based on lung region statistics
                    # Resize original mask to match final size for normalization
                    mask_resized = resize_image_fill(
                        mask_image_original.astype(np.uint8) * 255, target_size, NEAREST
                    )
                    mask_resized = (mask_resized > 128).astype(bool)
                    cxr_final = normalize_intensity(
                        cxr_final,
                        method='lung_region',
                        mask=mask_resized
                    )
                else:
                    # Other normalization methods (zscore, minmax, percentile)
                    # Use lung region mask if available
                    mask_for_norm = None
                    if np.any(mask_final):
                        # Resize original mask to match final size
                        mask_resized = resize_image_fill(
                            mask_image_original.astype(np.uint8) * 255, target_size, NEAREST
                        )
                        mask_for_norm = (mask_resized > 128).astype(bool)
                    
                    cxr_final = normalize_intensity(
                        cxr_final,
                        method=intensity_normalization,
                        mask=mask_for_norm
                    )
            
            # Ensure all images are uint8
            if roi_final.dtype != np.uint8:
                roi_final = roi_final.astype(np.uint8)
            if cxr_final.dtype != np.uint8:
                cxr_final = cxr_final.astype(np.uint8)
            
            # Convert mask to uint8 (0 or 255)
            mask_uint8 = (mask_final.astype(np.uint8)) * 255
            
            # Save ROI image
            roi_output_path = os.path.join(output_roi_class_dir, img_name)
            Image.fromarray(roi_final).save(roi_output_path)
            
            # Save CXR image
            cxr_output_path = os.path.join(output_cxr_class_dir, img_name)
            Image.fromarray(cxr_final).save(cxr_output_path)
            
            # Save mask image
            name_base, ext = os.path.splitext(img_name)
            mask_output_name = f"{name_base}_mask{ext}"
            mask_output_path = os.path.join(output_masks_class_dir, mask_output_name)
            Image.fromarray(mask_uint8).save(mask_output_path)
            
            stats['roi_created'] += 1
            stats['cxr_saved'] += 1
            stats['masks_saved'] += 1
            stats['processed'] += 1
            
        except Exception as e:
            print(f"    Error processing {img_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            stats['errors'] += 1
    
    # Print summary
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total CXR images: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"ROI images created: {stats['roi_created']}")
    print(f"CXR images saved: {stats['cxr_saved']}")
    print(f"Mask images saved: {stats['masks_saved']}")
    print(f"Errors: {stats['errors']}")
    print(f"Missing masks: {stats['missing_mask']}")
    print(f"Empty masks: {stats['empty_mask']}")
    
    # Validation - check intensity preservation and sizes
    print("\n=== VALIDATION ===")
    for class_name in classes:
        print(f"\n{class_name} class:")
        
        output_cxr_class_dir = os.path.join(output_cxr_dir, class_name)
        output_roi_class_dir = os.path.join(output_roi_dir, class_name)
        output_masks_class_dir = os.path.join(output_masks_dir, class_name)
        
        if (os.path.exists(output_cxr_class_dir) and 
            os.path.exists(output_roi_class_dir) and 
            os.path.exists(output_masks_class_dir)):
            
            roi_files = glob.glob(os.path.join(output_roi_class_dir, '*.png'))
            
            if roi_files:
                # Check a sample image
                sample_roi_path = roi_files[0]
                sample_img_name = os.path.basename(sample_roi_path)
                
                # Get corresponding CXR and mask paths
                sample_cxr_path = os.path.join(output_cxr_class_dir, sample_img_name)
                name_base, ext = os.path.splitext(sample_img_name)
                sample_mask_path = os.path.join(output_masks_class_dir, f"{name_base}_mask{ext}")
                
                if (os.path.exists(sample_cxr_path) and 
                    os.path.exists(sample_roi_path) and 
                    os.path.exists(sample_mask_path)):
                    
                    cxr_sample = load_image_preserve_intensity(sample_cxr_path)
                    roi_sample = load_image_preserve_intensity(sample_roi_path)
                    mask_sample = load_mask_binary(sample_mask_path)
                    
                    # Check that all have same size
                    size_match = (roi_sample.shape == cxr_sample.shape == mask_sample.shape)
                    
                    # Check intensity preservation in lung regions
                    lung_pixels_roi = roi_sample[mask_sample]
                    lung_pixels_cxr = cxr_sample[mask_sample]
                    
                    if len(lung_pixels_cxr) > 0 and len(lung_pixels_roi) > 0:
                        # Compare intensities (allow tolerance for resizing interpolation)
                        # After resizing, pixel values change slightly due to interpolation
                        # Check if mean values are close (within 5% difference)
                        mean_diff = abs(lung_pixels_cxr.mean() - lung_pixels_roi.mean())
                        mean_relative_diff = mean_diff / (lung_pixels_cxr.mean() + 1e-6)
                        intensity_match = mean_relative_diff < 0.05  # Within 5% difference
                        
                        print(f"  Sample: {sample_img_name}")
                        print(f"  CXR size: {cxr_sample.shape[1]} x {cxr_sample.shape[0]}")
                        print(f"  ROI size: {roi_sample.shape[1]} x {roi_sample.shape[0]}")
                        print(f"  Mask size: {mask_sample.shape[1]} x {mask_sample.shape[0]}")
                        print(f"  All sizes match: {'✓ Yes' if size_match else '✗ No'}")
                        print(f"  CXR intensity range: [{cxr_sample.min()}, {cxr_sample.max()}]")
                        print(f"  ROI intensity range: [{roi_sample.min()}, {roi_sample.max()}]")
                        print(f"  CXR mean (lung region): {lung_pixels_cxr.mean():.2f}")
                        print(f"  ROI mean (lung region): {lung_pixels_roi.mean():.2f}")
                        print(f"  Intensity preserved: {'✓ Yes' if intensity_match else '✗ No'}")
    
    print("\n=== ROI EXTRACTION COMPLETE ===")
    print(f"Output directory: {output_base_dir}")
    print(f"  - CXR images: {output_cxr_dir}")
    print(f"  - ROI images: {output_roi_dir}")
    print(f"  - Mask images: {output_masks_dir}")
    print(f"\nTarget size: {target_size[0]} x {target_size[1]} (stretched to fill, no black borders)")
    print(f"CXR images: Full-size (no cropping, just resized)")
    print(f"ROI images: {'Zoomed-in (bounding box crop)' if crop_to_bbox else 'Full-size (no cropping)'}")
    print("\nNote: All images maintain original intensity values.")
    print("      Images are resized to fill target size (stretched, no aspect ratio preservation).")
    print("      This eliminates black borders.")
    print("      CXR images remain full-size (not cropped).")
    if crop_to_bbox:
        print("      Only ROI and masks are cropped to bounding box of lung regions, then resized.")
    else:
        print("      ROI, CXR, and masks are all resized to target size (stretched to fill).")

if __name__ == '__main__':
    main()

