#!/usr/bin/env python3
"""
RECREATE_DATASET_FROM_ORIGINAL Recreate ROI, masks, and CXR datasets from original sources

This script:
1. Reads image names from input/resized directories (to know which images to process)
2. Loads original CXR images and masks from the original dataset
3. Creates ROI images by cropping using masks
4. Resizes all images to 227x227
5. Saves to input/resized_new/roi, input/resized_new/masks, input/resized_new/cxr

Original dataset locations:
- CXR: /Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/cxr_png
- Masks: /Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/masks
"""

import os
import numpy as np
from PIL import Image
import glob
from pathlib import Path
from tqdm import tqdm

def normalize_image_to_uint8(img_array):
    """
    Normalize image to full 0-255 range before converting to uint8.
    Many Kaggle masks/CXRs are stored as 12/16-bit PNGs; a direct uint8 cast
    was discarding the high bits and making lungs appear very dark.
    """
    img_array = np.array(img_array, dtype=np.float64)
    
    # If already uint8 and in range, return as is
    if img_array.dtype == np.uint8 and img_array.max() <= 255 and img_array.min() >= 0:
        return img_array.astype(np.uint8)
    
    # Normalize to 0-255 range
    img_min = img_array.min()
    img_max = img_array.max()
    
    if img_max > img_min:
        img_normalized = (img_array - img_min) / (img_max - img_min) * 255.0
    else:
        img_normalized = img_array
    
    return np.clip(np.round(img_normalized), 0, 255).astype(np.uint8)

def load_and_normalize_image(image_path):
    """Load image and normalize to uint8 range"""
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    img_array = np.array(img)
    return normalize_image_to_uint8(img_array)

def find_mask_file(masks_dir, image_name):
    """Find mask file with various naming conventions"""
    name_base = os.path.splitext(image_name)[0]
    ext = os.path.splitext(image_name)[1]
    
    # Try different mask naming patterns
    mask_patterns = [
        f"{name_base}_mask{ext}",
        image_name,
        f"{name_base}_mask.png",
        f"{image_name}.png"
    ]
    
    # Check in root masks directory
    for pattern in mask_patterns:
        mask_path = os.path.join(masks_dir, pattern)
        if os.path.exists(mask_path):
            return mask_path
    
    # Check in subdirectories
    for subdir in os.listdir(masks_dir):
        subdir_path = os.path.join(masks_dir, subdir)
        if os.path.isdir(subdir_path):
            for pattern in mask_patterns:
                mask_path = os.path.join(subdir_path, pattern)
                if os.path.exists(mask_path):
                    return mask_path
    
    return None

def create_roi_from_mask(cxr_image, mask_image):
    """
    Create ROI by applying mask to CXR (lungs only region)
    Returns: (roi_image, mask_cropped, bbox)
    """
    # Ensure mask is binary
    if mask_image.max() > 1:
        mask_binary = (mask_image > 128).astype(bool)
    else:
        mask_binary = mask_image.astype(bool)
    
    # Ensure same size
    if mask_binary.shape != cxr_image.shape:
        from scipy.ndimage import zoom
        zoom_factors = (cxr_image.shape[0] / mask_binary.shape[0],
                       cxr_image.shape[1] / mask_binary.shape[1])
        mask_binary = zoom(mask_binary.astype(float), zoom_factors, order=0) > 0.5
    
    # Apply mask: keep lung regions, set non-lung to background (0)
    roi_image = cxr_image.copy()
    roi_image[~mask_binary] = 0
    
    # Get bounding box of mask for cropping
    if np.any(mask_binary):
        rows, cols = np.where(mask_binary)
        min_row = max(0, rows.min() - 5)  # Add small padding
        max_row = min(cxr_image.shape[0], rows.max() + 6)
        min_col = max(0, cols.min() - 5)
        max_col = min(cxr_image.shape[1], cols.max() + 6)
        
        # Crop ROI
        roi_cropped = roi_image[min_row:max_row, min_col:max_col]
        mask_cropped = mask_binary[min_row:max_row, min_col:max_col]
        
        return roi_cropped, mask_cropped
    else:
        # Empty mask - use center crop
        h, w = cxr_image.shape
        center_h = h // 2
        center_w = w // 2
        crop_size = int(min(h, w) * 0.6)  # 60% of smaller dimension
        
        min_row = max(0, center_h - crop_size // 2)
        max_row = min(h, center_h + crop_size // 2)
        min_col = max(0, center_w - crop_size // 2)
        max_col = min(w, center_w + crop_size // 2)
        
        roi_cropped = cxr_image[min_row:max_row, min_col:max_col]
        mask_cropped = mask_binary[min_row:max_row, min_col:max_col]
        
        return roi_cropped, mask_cropped

def resize_image_bicubic(img_array, target_size):
    """Resize image using bicubic interpolation"""
    img_pil = Image.fromarray(img_array)
    img_resized = img_pil.resize(target_size, Image.BICUBIC)
    return np.array(img_resized)

def resize_mask_nearest(mask_array, target_size):
    """Resize mask using nearest neighbor interpolation"""
    mask_pil = Image.fromarray(mask_array.astype(np.uint8) * 255)
    mask_resized = mask_pil.resize(target_size, Image.NEAREST)
    return (np.array(mask_resized) > 128).astype(bool)

def main():
    print("=== RECREATING DATASET FROM ORIGINAL SOURCES ===\n")
    
    # Configuration
    original_cxr_dir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/cxr_png'
    original_masks_dir = '/Users/numanaslam/Desktop/Research/datasets/chest/kaggle_mask_lungs 2/Lung Segmentation/masks'
    
    # Output directories
    output_base_dir = os.path.join('input', 'resized_new')
    output_roi_dir = os.path.join(output_base_dir, 'roi')
    output_masks_dir = os.path.join(output_base_dir, 'masks')
    output_cxr_dir = os.path.join(output_base_dir, 'cxr')
    
    # Target size
    target_size = (227, 227)
    
    # Create output directories
    classes = ['normal', 'ptb']
    for class_name in classes:
        for output_dir in [output_roi_dir, output_masks_dir, output_cxr_dir]:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Get reference image names from existing resized directory
    print("Reading reference image names from input/resized...")
    reference_roi_dir = os.path.join('input', 'resized', 'roi')
    reference_cxr_dir = os.path.join('input', 'resized', 'cxr')
    
    if not os.path.exists(reference_roi_dir) or not os.path.exists(reference_cxr_dir):
        raise FileNotFoundError("Reference directories not found. Please ensure input/resized/roi and input/resized/cxr exist.")
    
    # Get all image names from reference directories
    all_image_names = []
    for class_name in classes:
        roi_ref_dir = os.path.join(reference_roi_dir, class_name)
        cxr_ref_dir = os.path.join(reference_cxr_dir, class_name)
        
        if os.path.exists(roi_ref_dir):
            for img_file in glob.glob(os.path.join(roi_ref_dir, '*.png')):
                img_name = os.path.basename(img_file)
                if (img_name, class_name) not in all_image_names:
                    all_image_names.append((img_name, class_name))
        
        if os.path.exists(cxr_ref_dir):
            for img_file in glob.glob(os.path.join(cxr_ref_dir, '*.png')):
                img_name = os.path.basename(img_file)
                if (img_name, class_name) not in all_image_names:
                    all_image_names.append((img_name, class_name))
    
    print(f"Found {len(all_image_names)} unique image names to process\n")
    
    # Statistics
    stats = {
        'total': len(all_image_names),
        'processed': 0,
        'roi_created': 0,
        'mask_created': 0,
        'cxr_created': 0,
        'errors': 0,
        'missing_original': 0,
        'missing_mask': 0
    }
    
    print("Processing images...")
    
    # Process each image
    for img_name, class_name in tqdm(all_image_names, desc="Processing"):
        try:
            # Construct original file paths
            original_cxr_path = os.path.join(original_cxr_dir, img_name)
            
            # Check if original CXR exists
            if not os.path.exists(original_cxr_path):
                print(f"Warning: Original CXR not found: {original_cxr_path}")
                stats['missing_original'] += 1
                continue
            
            # Find mask file
            original_mask_path = find_mask_file(original_masks_dir, img_name)
            
            if original_mask_path is None or not os.path.exists(original_mask_path):
                print(f"Warning: Original mask not found for: {img_name}")
                stats['missing_mask'] += 1
                # Continue without mask - will create CXR only
            
            # Load original CXR image
            cxr_original = load_and_normalize_image(original_cxr_path)
            
            # Load mask if available
            mask_original = None
            if original_mask_path and os.path.exists(original_mask_path):
                mask_original = load_and_normalize_image(original_mask_path)
                # Binarize mask
                mask_original = (mask_original > 128).astype(bool)
            
            # Create ROI by applying mask to CXR (lungs only region)
            if mask_original is not None and np.any(mask_original):
                roi_image, mask_cropped = create_roi_from_mask(cxr_original, mask_original)
            else:
                # No mask - use center crop (fallback)
                h, w = cxr_original.shape
                center_h = h // 2
                center_w = w // 2
                crop_size = int(min(h, w) * 0.6)  # 60% of smaller dimension
                
                min_row = max(0, center_h - crop_size // 2)
                max_row = min(h, center_h + crop_size // 2)
                min_col = max(0, center_w - crop_size // 2)
                max_col = min(w, center_w + crop_size // 2)
                
                roi_image = cxr_original[min_row:max_row, min_col:max_col]
                if mask_original is not None:
                    mask_cropped = mask_original[min_row:max_row, min_col:max_col]
                else:
                    mask_cropped = None
            
            # Resize images
            # Resize in double precision but keep the 0-255 range, then cast back
            roi_resized = resize_image_bicubic(roi_image, target_size)
            roi_resized = np.clip(np.round(roi_resized), 0, 255).astype(np.uint8)
            
            cxr_resized = resize_image_bicubic(cxr_original, target_size)
            cxr_resized = np.clip(np.round(cxr_resized), 0, 255).astype(np.uint8)
            
            # For mask: use nearest neighbor to preserve binary nature
            if mask_cropped is not None:
                mask_resized = resize_mask_nearest(mask_cropped, target_size)
            else:
                mask_resized = np.zeros(target_size, dtype=bool)
            
            # Save ROI image
            roi_output_path = os.path.join(output_roi_dir, class_name, img_name)
            Image.fromarray(roi_resized).save(roi_output_path)
            stats['roi_created'] += 1
            
            # Save mask image
            name_base, ext = os.path.splitext(img_name)
            mask_output_name = f"{name_base}_mask{ext}"
            mask_output_path = os.path.join(output_masks_dir, class_name, mask_output_name)
            Image.fromarray(mask_resized.astype(np.uint8) * 255).save(mask_output_path)
            stats['mask_created'] += 1
            
            # Save CXR image
            cxr_output_path = os.path.join(output_cxr_dir, class_name, img_name)
            Image.fromarray(cxr_resized).save(cxr_output_path)
            stats['cxr_created'] += 1
            
            stats['processed'] += 1
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            stats['errors'] += 1
    
    # Print summary
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total images to process: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"ROI images created: {stats['roi_created']}")
    print(f"Mask images created: {stats['mask_created']}")
    print(f"CXR images created: {stats['cxr_created']}")
    print(f"Errors: {stats['errors']}")
    print(f"Missing original CXR: {stats['missing_original']}")
    print(f"Missing masks: {stats['missing_mask']}")
    
    # Validation
    print("\n=== VALIDATION ===")
    for class_name in classes:
        print(f"\n{class_name} class:")
        
        roi_class_dir = os.path.join(output_roi_dir, class_name)
        masks_class_dir = os.path.join(output_masks_dir, class_name)
        cxr_class_dir = os.path.join(output_cxr_dir, class_name)
        
        if os.path.exists(roi_class_dir):
            roi_files = glob.glob(os.path.join(roi_class_dir, '*.png'))
            print(f"  ROI files: {len(roi_files)}")
            if roi_files:
                sample_roi = np.array(Image.open(roi_files[0]))
                print(f"  ROI sample size: {sample_roi.shape[1]} x {sample_roi.shape[0]}")
        
        if os.path.exists(masks_class_dir):
            mask_files = glob.glob(os.path.join(masks_class_dir, '*_mask.png'))
            print(f"  Mask files: {len(mask_files)}")
            if mask_files:
                sample_mask = np.array(Image.open(mask_files[0]))
                print(f"  Mask sample size: {sample_mask.shape[1]} x {sample_mask.shape[0]}")
                mask_binary = sample_mask > 128
                print(f"  Mask foreground pixels: {np.sum(mask_binary)} ({np.sum(mask_binary) / mask_binary.size * 100:.1f}%)")
        
        if os.path.exists(cxr_class_dir):
            cxr_files = glob.glob(os.path.join(cxr_class_dir, '*.png'))
            print(f"  CXR files: {len(cxr_files)}")
            if cxr_files:
                sample_cxr = np.array(Image.open(cxr_files[0]))
                print(f"  CXR sample size: {sample_cxr.shape[1]} x {sample_cxr.shape[0]}")
    
    print("\n=== DATASET RECREATION COMPLETE ===")
    print(f"All images have been recreated and saved to: {output_base_dir}")
    print("All images have been standardized to:")
    print(f"  - Size: {target_size[0]} x {target_size[1]}")
    print("  - Format: Grayscale uint8")
    print("  - ROI: Cropped using masks (or center crop if mask unavailable)")
    print("  - CXR: Full images resized with bicubic interpolation")
    print("  - Masks: Binary masks resized with nearest neighbor")
    print(f"\nNote: New images saved to '{output_base_dir}' to compare with old 'input/resized'")

if __name__ == '__main__':
    main()

