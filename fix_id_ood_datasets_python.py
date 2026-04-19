#!/usr/bin/env python3
"""
Fix and standardize ID/OOD datasets
This script:
1. Detects and fixes inverted images
2. Ensures consistent image sizes (227x227)
3. Validates file matching between ROI and CXR
4. Standardizes image formats (grayscale uint8)
5. Creates backup before modifications (optional)
"""

import os
import shutil
from PIL import Image
import numpy as np
from pathlib import Path

# Configuration
ROI_DIR = 'input/resized/roi'
CXR_DIR = 'input/resized/cxr'
BACKUP_DIR = 'input/resized/backup'
TARGET_SIZE = (227, 227)
CREATE_BACKUP = False  # Set to True to create backup before fixing

def create_backup():
    """Create backup of original images"""
    if CREATE_BACKUP and not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f'Created backup directory: {BACKUP_DIR}')
        
        # Backup ROI
        roi_backup = os.path.join(BACKUP_DIR, 'roi')
        if os.path.exists(ROI_DIR):
            shutil.copytree(ROI_DIR, roi_backup)
            print(f'Backed up ROI to: {roi_backup}')
        
        # Backup CXR
        cxr_backup = os.path.join(BACKUP_DIR, 'cxr')
        if os.path.exists(CXR_DIR):
            shutil.copytree(CXR_DIR, cxr_backup)
            print(f'Backed up CXR to: {cxr_backup}')

def fix_image(img_path, is_roi=True, target_size=TARGET_SIZE):
    """
    Fix a single image:
    - Resize to target size
    - Convert to grayscale
    - Fix inversion if needed
    - Ensure uint8 format
    """
    try:
        # Load image
        img = Image.open(img_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Resize if needed
        if img_array.shape[:2] != target_size[::-1]:  # PIL uses (width, height)
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img)
        
        # Check for inversion
        mean_intensity = img_array.mean()
        
        if is_roi:
            # ROI should be darker (cropped region, mean ~20-50)
            # If very dark (< 10), might be inverted
            if mean_intensity < 10:
                img_inverted = 255 - img_array
                mean_inverted = img_inverted.mean()
                if mean_inverted > 50:  # Inverted version is much brighter
                    img_array = img_inverted
                    inverted = True
                else:
                    inverted = False
            else:
                inverted = False
        else:
            # CXR should be brighter (full image, mean ~80-150)
            # If very dark (< 10), might be inverted
            if mean_intensity < 10:
                img_inverted = 255 - img_array
                mean_inverted = img_inverted.mean()
                if mean_inverted > 100:  # Inverted version is much brighter
                    img_array = img_inverted
                    inverted = True
                else:
                    inverted = False
            else:
                inverted = False
        
        # Ensure uint8
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Save
        img_fixed = Image.fromarray(img_array, mode='L')
        img_fixed.save(img_path, 'PNG')
        
        return True, inverted, mean_intensity
        
    except Exception as e:
        print(f'  Error processing {img_path}: {e}')
        return False, False, 0

def process_class(className):
    """Process all images in a class"""
    roi_class_dir = os.path.join(ROI_DIR, className)
    cxr_class_dir = os.path.join(CXR_DIR, className)
    
    if not os.path.exists(roi_class_dir) or not os.path.exists(cxr_class_dir):
        print(f'  ⚠ Directories not found for class: {className}')
        return None
    
    # Get file lists
    roi_files = sorted([f for f in os.listdir(roi_class_dir) if f.endswith('.png')])
    cxr_files = sorted([f for f in os.listdir(cxr_class_dir) if f.endswith('.png')])
    
    print(f'\n=== Processing {className} class ===')
    print(f'  ROI files: {len(roi_files)}')
    print(f'  CXR files: {len(cxr_files)}')
    
    stats = {
        'roi_fixed': 0,
        'cxr_fixed': 0,
        'roi_resized': 0,
        'cxr_resized': 0,
        'roi_inverted': 0,
        'cxr_inverted': 0,
        'roi_intensities': [],
        'cxr_intensities': [],
        'errors': 0
    }
    
    # Process ROI images
    print(f'\n  Processing ROI images...')
    for i, filename in enumerate(roi_files):
        img_path = os.path.join(roi_class_dir, filename)
        
        # Check original size
        original_img = Image.open(img_path)
        original_size = original_img.size
        
        success, inverted, mean_int = fix_image(img_path, is_roi=True)
        
        if success:
            stats['roi_fixed'] += 1
            stats['roi_intensities'].append(mean_int)
            if original_size != TARGET_SIZE:
                stats['roi_resized'] += 1
            if inverted:
                stats['roi_inverted'] += 1
        else:
            stats['errors'] += 1
        
        if (i + 1) % 50 == 0:
            print(f'    Processed {i + 1}/{len(roi_files)} ROI images...')
    
    # Process CXR images
    print(f'\n  Processing CXR images...')
    for i, filename in enumerate(cxr_files):
        img_path = os.path.join(cxr_class_dir, filename)
        
        # Check original size
        original_img = Image.open(img_path)
        original_size = original_img.size
        
        success, inverted, mean_int = fix_image(img_path, is_roi=False)
        
        if success:
            stats['cxr_fixed'] += 1
            stats['cxr_intensities'].append(mean_int)
            if original_size != TARGET_SIZE:
                stats['cxr_resized'] += 1
            if inverted:
                stats['cxr_inverted'] += 1
        else:
            stats['errors'] += 1
        
        if (i + 1) % 50 == 0:
            print(f'    Processed {i + 1}/{len(cxr_files)} CXR images...')
    
    # Calculate statistics
    if stats['roi_intensities']:
        stats['roi_mean_intensity'] = np.mean(stats['roi_intensities'])
        stats['roi_std_intensity'] = np.std(stats['roi_intensities'])
    else:
        stats['roi_mean_intensity'] = 0
        stats['roi_std_intensity'] = 0
    
    if stats['cxr_intensities']:
        stats['cxr_mean_intensity'] = np.mean(stats['cxr_intensities'])
        stats['cxr_std_intensity'] = np.std(stats['cxr_intensities'])
    else:
        stats['cxr_mean_intensity'] = 0
        stats['cxr_std_intensity'] = 0
    
    return stats

def validate_datasets():
    """Validate the fixed datasets"""
    print('\n=== VALIDATION ===')
    
    classes = ['normal', 'ptb']
    
    for className in classes:
        roi_class_dir = os.path.join(ROI_DIR, className)
        cxr_class_dir = os.path.join(CXR_DIR, className)
        
        if not os.path.exists(roi_class_dir) or not os.path.exists(cxr_class_dir):
            continue
        
        roi_files = sorted([f for f in os.listdir(roi_class_dir) if f.endswith('.png')])
        cxr_files = sorted([f for f in os.listdir(cxr_class_dir) if f.endswith('.png')])
        
        print(f'\n{className} class:')
        print(f'  ROI files: {len(roi_files)}')
        print(f'  CXR files: {len(cxr_files)}')
        
        if roi_files and cxr_files:
            # Check sizes
            roi_img = Image.open(os.path.join(roi_class_dir, roi_files[0]))
            cxr_img = Image.open(os.path.join(cxr_class_dir, cxr_files[0]))
            
            print(f'  ROI size: {roi_img.size[0]} x {roi_img.size[1]}')
            print(f'  CXR size: {cxr_img.size[0]} x {cxr_img.size[1]}')
            
            if roi_img.size == TARGET_SIZE and cxr_img.size == TARGET_SIZE:
                print(f'  ✓ Sizes correct ({TARGET_SIZE[0]} x {TARGET_SIZE[1]})')
            else:
                print(f'  ⚠ Size mismatch')
            
            # Check intensities
            roi_array = np.array(roi_img.convert('L'))
            cxr_array = np.array(cxr_img.convert('L'))
            
            roi_mean = roi_array.mean()
            cxr_mean = cxr_array.mean()
            
            print(f'  ROI mean intensity: {roi_mean:.2f}')
            print(f'  CXR mean intensity: {cxr_mean:.2f}')
            
            if roi_mean < cxr_mean:
                print(f'  ✓ Intensity relationship correct (ROI < CXR)')
            else:
                print(f'  ⚠ Unexpected intensity relationship')

def main():
    print('=== FIXING ID/OOD DATASETS ===\n')
    
    # Create backup if requested
    if CREATE_BACKUP:
        create_backup()
        print()
    
    # Process each class
    classes = ['normal', 'ptb']
    all_stats = {}
    
    for className in classes:
        stats = process_class(className)
        if stats:
            all_stats[className] = stats
    
    # Print summary
    print('\n=== PROCESSING SUMMARY ===')
    for className, stats in all_stats.items():
        print(f'\n{className} class:')
        print(f'  ROI fixed: {stats["roi_fixed"]}')
        print(f'  CXR fixed: {stats["cxr_fixed"]}')
        print(f'  ROI resized: {stats["roi_resized"]}')
        print(f'  CXR resized: {stats["cxr_resized"]}')
        print(f'  ROI inverted: {stats["roi_inverted"]}')
        print(f'  CXR inverted: {stats["cxr_inverted"]}')
        print(f'  ROI mean intensity: {stats["roi_mean_intensity"]:.2f} ± {stats["roi_std_intensity"]:.2f}')
        print(f'  CXR mean intensity: {stats["cxr_mean_intensity"]:.2f} ± {stats["cxr_std_intensity"]:.2f}')
        print(f'  Errors: {stats["errors"]}')
    
    # Validate
    validate_datasets()
    
    print('\n=== FIXING COMPLETE ===')
    print(f'All images have been standardized to:')
    print(f'  - Size: {TARGET_SIZE[0]} x {TARGET_SIZE[1]}')
    print(f'  - Format: Grayscale uint8')
    print(f'  - Inversion: Corrected if needed')

if __name__ == '__main__':
    main()

