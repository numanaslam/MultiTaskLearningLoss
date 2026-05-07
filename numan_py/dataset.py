import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from skimage.exposure import match_histograms

def extract_lung_roi(img):
    """
    Python equivalent of extract_lung_roi_simple.
    Simple heuristic to crop the chest region.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Thresholding to find lung-like areas
    try:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except:
        # Fallback if Otsu fails
        mask = gray < 128 
        
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: center crop
        h, w = gray.shape
        y1, y2 = int(h*0.1), int(h*0.9)
        x1, x2 = int(w*0.15), int(w*0.85)
        roi = gray[y1:y2, x1:x2]
    else:
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Add margin
        margin = 15
        roi = gray[max(0, y-margin):min(gray.shape[0], y+h+margin),
                   max(0, x-margin):min(gray.shape[1], x+w+margin)]
                   
    return cv2.resize(roi, (224, 224))

class CXRDataset(Dataset):
    def __init__(self, file_paths, labels, ref_hist, 
                 aug_mode=False, is_ood=False, method='histmatch',
                 image_size=224, cxr_dir=None):
        self.paths = file_paths
        self.labels = labels
        self.ref_hist = ref_hist
        self.method = method
        self.is_ood = is_ood
        self.cxr_dir = cxr_dir
        self.image_size = image_size
        
        # Normalization parameters for VGG16 (ImageNet)
        self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = ToTensorV2()
        
        # Augmentation
        if aug_mode:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.15, rotate_limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                self.normalize,
                self.to_tensor
            ])
        else:
            self.transform = A.Compose([
                self.normalize,
                self.to_tensor
            ])

    def _preprocess_image(self, img):
        """Apply histogram matching or CLAHE"""
        if self.method == 'histmatch' and self.ref_hist is not None:
            # ref_hist is a 1D array of counts. We use CDF-based matching.
            # Normalize target histogram and compute CDF
            target_cdf = np.cumsum(self.ref_hist.astype(np.float64))
            target_cdf = target_cdf / (target_cdf[-1] + 1e-8) * 255.0
            
            # Compute source image CDF
            src_hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
            src_cdf = np.cumsum(src_hist.astype(np.float64)) / (img.size + 1e-8) * 255.0
            
            # Create lookup table mapping source intensities to target intensities
            lut = np.interp(src_cdf, target_cdf, range(256)).astype(np.uint8)
            img = lut[img]  # Apply lookup table
            
        elif self.method == 'clahe':
            img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
            
        return img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        # Load image safely
        img = cv2.imread(img_path)
        if img is None:
            # Return black image if file is missing/corrupt
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # ROI extraction for OOD (full CXR)
            if self.is_ood:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                roi = extract_lung_roi(gray)
                img = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            # Preprocessing (HistMatch/CLAHE)
            # Convert to grayscale for processing, then back to 3 channels
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_processed = self._preprocess_image(gray)
            img = cv2.cvtColor(gray_processed, cv2.COLOR_GRAY2RGB)
            
            # Resize
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size))

        # Apply augmentation/normalization
        augmented = self.transform(image=img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return augmented['image'], label

# =============================================================================
# DATA LOADING HELPER FUNCTIONS
# =============================================================================

def load_cxr_dataset(roi_dir='input/roi', extensions=('.png', '.jpg', '.jpeg')):
    """
    Scans 'roi_dir' for subfolders (e.g., 'normal', 'ptb') and returns file paths and labels.
    """
    roi_path = Path(roi_dir)
    files = []
    labels = []
    
    # Map folder names to labels
    # 0 -> Normal, 1 -> PTB (Tuberculosis)
    class_map = {'normal': 0, 'ptb': 1}
    
    for class_name, label in class_map.items():
        class_dir = roi_path / class_name
        if not class_dir.exists():
            print(f"⚠️  Warning: Directory '{class_dir}' not found. Skipping.")
            continue
            
        # Find all image files in the class folder
        count = 0
        for ext in extensions:
            for img_path in class_dir.rglob(f'*{ext}'):
                if img_path.is_file():
                    files.append(str(img_path))
                    labels.append(label)
                    count += 1
        print(f"  Found {count} images in '{class_name}' folder.")

    if len(files) == 0:
        raise FileNotFoundError(f"No images found in {roi_dir}. Please check your folder structure.")
        
    # Shuffle dataset with a fixed seed for reproducibility
    indices = np.random.RandomState(42).permutation(len(files))
    files = [files[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return files, labels

def compute_reference_histogram(cxr_dir='input/cxr', roi_dir='input/roi', n_samples=50):
    """
    Computes an average reference histogram from full CXR images.
    Used for histogram matching.
    """
    cxr_path = Path(cxr_dir)
    images = []
    
    # Try to load from CXR dir first
    if cxr_path.exists():
        img_paths = list(cxr_path.rglob('*.png')) + list(cxr_path.rglob('*.jpg'))
        if img_paths:
            print(f"  Computing reference histogram from {n_samples} CXR images...")
            for p in img_paths[:n_samples]:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img.flatten())
    
    # Fallback to ROI images if CXR dir is empty/missing
    if not images:
        print("  ⚠️  CXR directory empty/missing. Falling back to ROI images for reference.")
        roi_path = Path(roi_dir)
        img_paths = list(roi_path.rglob('*.png')) + list(roi_path.rglob('*.jpg'))
        print(f"  Computing reference histogram from {n_samples} ROI images...")
        for p in img_paths[:n_samples]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.flatten())

    if not images:
        print("  ⚠️  No images found to compute reference histogram. Returning None.")
        return None

    # Combine all pixels and compute histogram
    all_pixels = np.concatenate(images)
    hist, _ = np.histogram(all_pixels, bins=256, range=(0, 256))
    return hist