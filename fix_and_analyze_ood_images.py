#!/usr/bin/env python3
"""
Fix and analyze OOD (CXR) images for ID/OOD evaluation
This script:
1. Analyzes ROI and CXR images in detail
2. Fixes CXR images (OOD) only (model trained on ROI, so ROI kept as-is)
3. Provides detailed statistics and visualizations
4. Detects and fixes inverted images
5. Ensures consistent image sizes and formats
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

# Configuration
ROI_DIR = 'input/resized/roi'
CXR_DIR = 'input/resized/cxr'
OUTPUT_DIR = 'results/image_analysis'
TARGET_SIZE = (227, 227)
CREATE_BACKUP = True
BACKUP_DIR = 'input/resized/backup_cxr'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_image(img_path, is_roi=True):
    """Analyze a single image and return statistics"""
    try:
        img = Image.open(img_path)
        img_array = np.array(img.convert('L'))
        
        stats = {
            'path': img_path,
            'size': img_array.shape,
            'mean_intensity': float(img_array.mean()),
            'std_intensity': float(img_array.std()),
            'min_intensity': int(img_array.min()),
            'max_intensity': int(img_array.max()),
            'non_zero_pixels': int(np.count_nonzero(img_array)),
            'non_zero_ratio': float(np.count_nonzero(img_array) / img_array.size),
            'is_inverted': False,
            'needs_fix': False,
            'fixes_applied': []
        }
        
        # Check if inverted (very dark images)
        if stats['mean_intensity'] < 10:
            img_inverted = 255 - img_array
            mean_inverted = img_inverted.mean()
            
            if is_roi:
                # ROI should be darker (~20-50), but not too dark
                if mean_inverted > 50:
                    stats['is_inverted'] = True
                    stats['needs_fix'] = True
            else:
                # CXR should be brighter (~80-150)
                if mean_inverted > 100:
                    stats['is_inverted'] = True
                    stats['needs_fix'] = True
        
        # Check size
        if img_array.shape[:2] != TARGET_SIZE[::-1]:  # PIL uses (width, height)
            stats['needs_fix'] = True
        
        # Check format
        if img.mode != 'L':
            stats['needs_fix'] = True
        
        return stats, img_array
        
    except Exception as e:
        print(f"Error analyzing {img_path}: {e}")
        return None, None

def fix_cxr_image(img_path, stats):
    """Fix a CXR image based on analysis"""
    try:
        img = Image.open(img_path)
        img_array = np.array(img.convert('L'))
        fixes_applied = []
        
        # Fix inversion
        if stats['is_inverted']:
            img_array = 255 - img_array
            fixes_applied.append('inverted')
        
        # Resize
        if img_array.shape[:2] != TARGET_SIZE[::-1]:
            img = Image.fromarray(img_array, mode='L')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img_array = np.array(img)
            fixes_applied.append('resized')
        
        # Ensure uint8
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            fixes_applied.append('converted_to_uint8')
        
        # Save
        img_fixed = Image.fromarray(img_array, mode='L')
        img_fixed.save(img_path, 'PNG')
        
        return fixes_applied
        
    except Exception as e:
        print(f"Error fixing {img_path}: {e}")
        return []

def process_class(className):
    """Process all images in a class"""
    roi_class_dir = os.path.join(ROI_DIR, className)
    cxr_class_dir = os.path.join(CXR_DIR, className)
    
    if not os.path.exists(roi_class_dir) or not os.path.exists(cxr_class_dir):
        print(f"⚠ Directories not found for class: {className}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing {className.upper()} class")
    print(f"{'='*60}")
    
    # Get file lists
    roi_files = sorted([f for f in os.listdir(roi_class_dir) if f.endswith('.png')])
    cxr_files = sorted([f for f in os.listdir(cxr_class_dir) if f.endswith('.png')])
    
    print(f"ROI files: {len(roi_files)}")
    print(f"CXR files: {len(cxr_files)}")
    
    # Analyze ROI images (read-only)
    print(f"\nAnalyzing ROI images (read-only)...")
    roi_stats_list = []
    roi_intensities = []
    
    for i, filename in enumerate(roi_files):
        img_path = os.path.join(roi_class_dir, filename)
        stats, _ = analyze_image(img_path, is_roi=True)
        if stats:
            roi_stats_list.append(stats)
            roi_intensities.append(stats['mean_intensity'])
        
        if (i + 1) % 50 == 0:
            print(f"  Analyzed {i + 1}/{len(roi_files)} ROI images...")
    
    # Analyze and fix CXR images
    print(f"\nAnalyzing and fixing CXR images...")
    cxr_stats_list = []
    cxr_intensities = []
    cxr_fixes = defaultdict(int)
    
    # Create backup if requested
    if CREATE_BACKUP and not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_class_dir = os.path.join(BACKUP_DIR, className)
        os.makedirs(backup_class_dir, exist_ok=True)
        print(f"Created backup directory: {backup_class_dir}")
    
    for i, filename in enumerate(cxr_files):
        img_path = os.path.join(cxr_class_dir, filename)
        
        # Backup if needed
        if CREATE_BACKUP:
            backup_path = os.path.join(BACKUP_DIR, className, filename)
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(img_path, backup_path)
        
        # Analyze
        stats, _ = analyze_image(img_path, is_roi=False)
        if stats:
            cxr_stats_list.append(stats)
            cxr_intensities.append(stats['mean_intensity'])
            
            # Fix if needed
            if stats['needs_fix']:
                fixes = fix_cxr_image(img_path, stats)
                for fix in fixes:
                    cxr_fixes[fix] += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(cxr_files)} CXR images...")
    
    # Calculate statistics
    roi_mean = np.mean(roi_intensities) if roi_intensities else 0
    roi_std = np.std(roi_intensities) if roi_intensities else 0
    cxr_mean = np.mean(cxr_intensities) if cxr_intensities else 0
    cxr_std = np.std(cxr_intensities) if cxr_intensities else 0
    
    # Print summary
    print(f"\n{className.upper()} Class Summary:")
    print(f"  ROI:")
    print(f"    Mean intensity: {roi_mean:.2f} ± {roi_std:.2f}")
    print(f"    Range: [{min(roi_intensities):.2f}, {max(roi_intensities):.2f}]")
    print(f"  CXR:")
    print(f"    Mean intensity: {cxr_mean:.2f} ± {cxr_std:.2f}")
    print(f"    Range: [{min(cxr_intensities):.2f}, {max(cxr_intensities):.2f}]")
    print(f"  Fixes applied:")
    for fix_type, count in cxr_fixes.items():
        print(f"    {fix_type}: {count}")
    
    # Check intensity relationship
    if roi_mean > 0 and cxr_mean > 0:
        if roi_mean < cxr_mean:
            print(f"  ✓ Intensity relationship correct (ROI < CXR)")
        else:
            print(f"  ⚠ WARNING: ROI intensity >= CXR intensity!")
            print(f"    This may indicate a problem with image preprocessing")
    
    return {
        'class': className,
        'roi_stats': roi_stats_list,
        'cxr_stats': cxr_stats_list,
        'roi_mean': roi_mean,
        'roi_std': roi_std,
        'cxr_mean': cxr_mean,
        'cxr_std': cxr_std,
        'cxr_fixes': dict(cxr_fixes)
    }

def create_visualizations(all_results):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Intensity distribution comparison
    ax = axes[0, 0]
    for result in all_results:
        className = result['class']
        roi_intensities = [s['mean_intensity'] for s in result['roi_stats']]
        cxr_intensities = [s['mean_intensity'] for s in result['cxr_stats']]
        
        ax.hist(roi_intensities, bins=30, alpha=0.5, label=f'{className} ROI', density=True)
        ax.hist(cxr_intensities, bins=30, alpha=0.5, label=f'{className} CXR', density=True)
    
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Intensity Distribution: ROI vs CXR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Mean intensity comparison
    ax = axes[0, 1]
    classes = [r['class'] for r in all_results]
    roi_means = [r['roi_mean'] for r in all_results]
    cxr_means = [r['cxr_mean'] for r in all_results]
    
    x = np.arange(len(classes))
    width = 0.35
    ax.bar(x - width/2, roi_means, width, label='ROI', alpha=0.8)
    ax.bar(x + width/2, cxr_means, width, label='CXR', alpha=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Intensity')
    ax.set_title('Mean Intensity Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Fixes applied
    ax = axes[1, 0]
    fix_types = set()
    for result in all_results:
        fix_types.update(result['cxr_fixes'].keys())
    
    fix_types = sorted(fix_types)
    if fix_types:
        fix_counts = {ft: sum(r['cxr_fixes'].get(ft, 0) for r in all_results) for ft in fix_types}
        ax.bar(fix_types, [fix_counts[ft] for ft in fix_types], alpha=0.8)
        ax.set_xlabel('Fix Type')
        ax.set_ylabel('Count')
        ax.set_title('Fixes Applied to CXR Images')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No fixes needed', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Fixes Applied to CXR Images')
    
    # 4. Intensity ratio (ROI/CXR)
    ax = axes[1, 1]
    ratios = [r['roi_mean'] / (r['cxr_mean'] + 1e-6) for r in all_results]
    ax.bar(classes, ratios, alpha=0.8)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Equal intensity')
    ax.set_xlabel('Class')
    ax.set_ylabel('Intensity Ratio (ROI/CXR)')
    ax.set_title('Intensity Ratio: ROI / CXR')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {os.path.join(OUTPUT_DIR, 'image_analysis.png')}")
    plt.close()

def create_sample_comparison(all_results, num_samples=5):
    """Create side-by-side comparison of sample images"""
    fig, axes = plt.subplots(len(all_results), num_samples * 2, figsize=(20, 4 * len(all_results)))
    
    if len(all_results) == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx, result in enumerate(all_results):
        className = result['class']
        roi_class_dir = os.path.join(ROI_DIR, className)
        cxr_class_dir = os.path.join(CXR_DIR, className)
        
        # Get sample files
        roi_files = sorted([f for f in os.listdir(roi_class_dir) if f.endswith('.png')])[:num_samples]
        cxr_files = sorted([f for f in os.listdir(cxr_class_dir) if f.endswith('.png')])[:num_samples]
        
        for sample_idx in range(num_samples):
            # ROI image
            if sample_idx < len(roi_files):
                roi_path = os.path.join(roi_class_dir, roi_files[sample_idx])
                roi_img = np.array(Image.open(roi_path).convert('L'))
                axes[class_idx, sample_idx * 2].imshow(roi_img, cmap='gray')
                axes[class_idx, sample_idx * 2].set_title(f'ROI: {roi_files[sample_idx][:20]}')
                axes[class_idx, sample_idx * 2].axis('off')
            
            # CXR image
            if sample_idx < len(cxr_files):
                cxr_path = os.path.join(cxr_class_dir, cxr_files[sample_idx])
                cxr_img = np.array(Image.open(cxr_path).convert('L'))
                axes[class_idx, sample_idx * 2 + 1].imshow(cxr_img, cmap='gray')
                axes[class_idx, sample_idx * 2 + 1].set_title(f'CXR: {cxr_files[sample_idx][:20]}')
                axes[class_idx, sample_idx * 2 + 1].axis('off')
    
    plt.suptitle('Sample Image Comparison: ROI vs CXR', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"Sample comparison saved to: {os.path.join(OUTPUT_DIR, 'sample_comparison.png')}")
    plt.close()

def main():
    print("="*60)
    print("FIX AND ANALYZE OOD (CXR) IMAGES")
    print("="*60)
    print(f"Strategy: Model trained on ROI, so ROI images will NOT be modified.")
    print(f"          Only CXR (OOD) images will be fixed.\n")
    
    classes = ['normal', 'ptb']
    all_results = []
    
    # Process each class
    for className in classes:
        result = process_class(className)
        if result:
            all_results.append(result)
    
    # Create visualizations
    if all_results:
        print(f"\n{'='*60}")
        print("Creating visualizations...")
        print(f"{'='*60}")
        create_visualizations(all_results)
        create_sample_comparison(all_results)
        
        # Save detailed statistics
        stats_file = os.path.join(OUTPUT_DIR, 'image_statistics.json')
        with open(stats_file, 'w') as f:
            # Convert numpy types to native Python types for JSON
            json_data = []
            for result in all_results:
                json_result = {
                    'class': result['class'],
                    'roi_mean': float(result['roi_mean']),
                    'roi_std': float(result['roi_std']),
                    'cxr_mean': float(result['cxr_mean']),
                    'cxr_std': float(result['cxr_std']),
                    'cxr_fixes': result['cxr_fixes']
                }
                json_data.append(json_result)
            json.dump(json_data, f, indent=2)
        print(f"Statistics saved to: {stats_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FIXING COMPLETE")
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  - ROI images: Analyzed (NOT modified - model training data)")
    print(f"  - CXR images: Fixed and standardized to:")
    print(f"    * Size: {TARGET_SIZE[0]} x {TARGET_SIZE[1]}")
    print(f"    * Format: Grayscale uint8")
    print(f"    * Inversion: Corrected if needed")
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

