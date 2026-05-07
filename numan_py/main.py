#!/usr/bin/env python3
"""
main.py - Complete PTB Detection Pipeline with PyTorch
======================================================
Trains VGG16 with custom losses (Focal, GradCAM, Tversky, Anatomical)
for tuberculosis detection on chest X-rays.

Features:
- Folder-based data loading (normal/ptb)
- K-fold cross-validation
- ImageNet normalization + preprocessing ensemble
- Custom loss functions with gradient clipping
- OOD evaluation with uncertainty estimation
- Threshold sweep for balanced accuracy optimization
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
from pathlib import Path
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Local imports
from dataset import CXRDataset, load_cxr_dataset, compute_reference_histogram
from model import VGG16Classifier
from losses import focal_loss, cam_cosine_loss, tversky_loss, anatomical_loss
from engine import Trainer, evaluate_with_ensemble

# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Training and evaluation configuration"""
    def __init__(self):
        # Data
        self.roi_dir = 'input/roi'
        self.cxr_dir = 'input/cxr'
        self.mask_dir = 'input/masks'
        self.image_size = 224
        
        # Model
        self.num_classes = 2
        self.backbone = 'vgg16'
        self.feature_layer = '29'  # relu5_3 in VGG16 features
        
        # Training
        self.lr = 1e-3
        self.warmup_epochs = 3
        self.min_lr = 0.1  # min_lr_ratio for cosine annealing
        self.epochs = 50
        self.batch_size = 14
        self.grad_clip = 1.0
        self.patience = 15  # early stopping
        self.k_folds = 2
        
        # Loss weights
        self.lambda_cam = 10.0
        self.lambda_tversky = 2.0
        self.lambda_anatomical = 2.0
        self.anatomical_reward_weight = 0.75
        self.tversky_alpha = 0.7
        self.tversky_beta = 0.45
        self.focal_alpha = [0.45, 0.55]  # [normal, ptb]
        self.focal_gamma = 2.0
        self.min_delta = 1e-2
        
        # Preprocessing
        self.preprocessing_method = 'histmatch'  # 'none', 'clahe', 'histmatch'
        self.ensemble_methods = ['histmatch', 'clahe', 'none']
        
        # Hardware
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = 4
        
        # Output
        self.output_dir = 'models/final'
        self.checkpoint_dir = 'models/checkpoints'
        
        # Flags
        self.use_gradcam = False  # Disable initially for stable baseline
        self.use_tversky = False
        self.use_anatomical = False
        self.use_focal = False  # Disable for balanced data
        self.quick_test = False
        
    def print_summary(self):
        """Print configuration summary"""
        print(f"\n{'='*60}")
        print(f"PTB DETECTION PIPELINE - CONFIGURATION")
        print(f"{'='*60}")
        print(f"Data: {self.roi_dir} | Classes: {self.num_classes}")
        print(f"Model: {self.backbone} | Feature layer: {self.feature_layer}")
        print(f"Training: LR={self.lr}, Warmup={self.warmup_epochs}, Epochs={self.epochs}")
        print(f"Batch={self.batch_size}, K-Folds={self.k_folds}, Patience={self.patience}")
        print(f"Loss: Focal={self.use_focal}, CAM={self.use_gradcam}, Tversky={self.use_tversky}, Anat={self.use_anatomical}")
        print(f"Preprocessing: {self.preprocessing_method} | Ensemble: {self.ensemble_methods}")
        print(f"Device: {self.device} | Workers: {self.num_workers}")
        print(f"Output: {self.output_dir}")
        if self.quick_test:
            print(f"⚠️  QUICK TEST MODE: Reduced epochs/folds")
        print(f"{'='*60}\n")


# ============================================================================
# Main Training Pipeline
# ============================================================================
def run_pipeline(cfg):
    """Execute complete training and evaluation pipeline"""
    
    # Setup output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # 1. Load dataset with folder-based labels
    print("\n📂 Loading dataset...")
    files, labels = load_cxr_dataset(cfg.roi_dir)
    
    if len(files) == 0:
        print("❌ No images found. Please check your input/roi directory structure.")
        print("Expected: input/roi/normal/*.png and input/roi/ptb/*.png")
        return
    
    n_normal = sum(l == 0 for l in labels)
    n_ptb = sum(l == 1 for l in labels)
    print(f"✅ Loaded {len(files)} images: {n_normal} normal, {n_ptb} PTB")
    
    # Quick test mode
    if cfg.quick_test:
        print("⚡ Quick test mode: subsampling dataset")
        sample_size = min(150, len(files))
        indices = np.random.RandomState(42).permutation(len(files))[:sample_size]
        files = [files[i] for i in indices]
        labels = [labels[i] for i in indices]
        cfg.k_folds = 2
        cfg.epochs = 10
        cfg.patience = 3
        print(f"   Subsampled to {len(files)} images, {cfg.k_folds} folds, {cfg.epochs} epochs")
    
    # 2. Compute reference histogram for histmatch
    ref_hist = None
    if cfg.preprocessing_method == 'histmatch':
        print("\n🎨 Computing reference histogram...")
        ref_hist = compute_reference_histogram(cfg.cxr_dir, cfg.roi_dir, n_samples=50)
        if ref_hist is None:
            print("  ⚠️  Falling back to ROI images for reference histogram")
    
    # 3. K-Fold Cross-Validation
    print(f"\n🔄 Starting {cfg.k_folds}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=42)
    
    fold_results = {}
    training_histories = {}
    best_models = {}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels), 1):
        print(f"\n{'='*60}")
        print(f"📊 FOLD {fold}/{cfg.k_folds}")
        print(f"{'='*60}")
        
        # Create datasets
        train_files = [files[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        train_dataset = CXRDataset(
            train_files, train_labels, ref_hist,
            aug_mode=True, is_ood=False, method=cfg.preprocessing_method,
            image_size=cfg.image_size
        )
        val_dataset = CXRDataset(
            val_files, val_labels, ref_hist,
            aug_mode=False, is_ood=False, method=cfg.preprocessing_method,
            image_size=cfg.image_size
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True,
            num_workers=cfg.num_workers, 
            pin_memory=True  # CRITICAL for GPU speed
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False,
            num_workers=cfg.num_workers, 
            pin_memory=True  # CRITICAL for GPU speed
        )
        
        # Initialize model
        print(f"\n🧠 Initializing {cfg.backbone}...")
        model = VGG16Classifier(num_classes=cfg.num_classes)
        model = model.to(cfg.device)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            device=cfg.device,
            config=cfg,
        )
        
        # Training loop
        print(f"\n🚀 Training for {cfg.epochs} epochs...")
        best_val_acc = 0
        patience_counter = 0
        fold_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 
                       'train_acc': [], 'val_acc': [], 'lr': []}
        
        for epoch in range(1, cfg.epochs + 1):
            # Train one epoch
            train_loss, train_acc, lr = trainer.train_epoch(train_loader, epoch)
            
            # Log progress
            fold_history['epoch'].append(epoch)
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['lr'].append(lr)
            
            # Validation every 3 epochs
            if epoch % 3 == 0 or epoch == 1:
                val_metrics = trainer.evaluate(val_loader, use_ensemble=False)
                val_acc = val_metrics['accuracy']
                val_loss = val_metrics['loss']
                
                fold_history['val_loss'].append(val_loss)
                fold_history['val_acc'].append(val_acc)
                
                print(f"  Epoch {epoch:02d} | LR: {lr:.5f}")
                print(f"    Train: Loss={train_loss:.4f}, Acc={train_acc:.3f}")
                print(f"    Val:   Loss={val_loss:.4f}, Acc={val_acc:.3f}")
                
                # Early stopping
                if val_acc > best_val_acc + cfg.min_delta:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_models[f'fold_{fold}'] = model.state_dict()
                    print(f"    ✓ Validation improved! (Best: {best_val_acc:.3f})")
                else:
                    patience_counter += 1
                    print(f"    ○ No improvement ({patience_counter}/{cfg.patience})")
                
                if patience_counter >= cfg.patience:
                    print(f"\n⏹ Early stopping triggered at epoch {epoch}")
                    break
            else:
                print(f"  Epoch {epoch:02d} | LR: {lr:.5f} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f}")
        
        # Final evaluation on validation set
        print(f"\n📈 Evaluating fold {fold}...")
        val_metrics = trainer.evaluate(val_loader, use_ensemble=False)
        
        # Store results
        fold_name = f'fold_{fold}'
        fold_results[fold_name] = val_metrics
        training_histories[fold_name] = fold_history
        
        # Save checkpoint
        checkpoint_path = os.path.join(cfg.checkpoint_dir, f'{fold_name}_checkpoint.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg.__dict__,
            'metrics': val_metrics,
            'history': fold_history
        }, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # 4. Aggregate cross-validation results
    print(f"\n{'='*60}")
    print(f"📊 CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    metrics_list = ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score', 'auc']
    for metric in metrics_list:
        values = [fold_results[f][metric] for f in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric:12s}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Select best model
    best_fold = max(fold_results, key=lambda f: fold_results[f]['accuracy'])
    print(f"\n🏆 Best model: {best_fold} (Accuracy: {fold_results[best_fold]['accuracy']:.3f})")
    
    # 5. Save final model and results
    print(f"\n💾 Saving results...")
    results = {
        'fold_results': fold_results,
        'training_histories': training_histories,
        'best_fold': best_fold,
        'best_model_state': best_models.get(best_fold),
        'config': cfg.__dict__,
        'ref_hist': ref_hist
    }
    
    results_path = os.path.join(cfg.output_dir, f'results_{cfg.preprocessing_method}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✓ Results saved: {results_path}")
    
    # Save best model separately
    if best_models.get(best_fold):
        model_path = os.path.join(cfg.output_dir, f'best_model_{cfg.preprocessing_method}.pt')
        torch.save(best_models[best_fold], model_path)
        print(f"  ✓ Best model saved: {model_path}")
    
    # 6. Plot training curves
    print(f"\n📈 Generating training curves...")
    plot_training_curves(training_histories, fold_results, cfg.output_dir)
    
    # 7. OOD Evaluation (if CXR directory exists)
    if os.path.exists(cfg.cxr_dir):
        print(f"\n{'='*60}")
        print(f"🔍 OUT-OF-DISTRIBUTION EVALUATION")
        print(f"{'='*60}")
        
        # Load best model
        model = VGG16Classifier(num_classes=cfg.num_classes)
        if best_models.get(best_fold):
            model.load_state_dict(best_models[best_fold])
        model = model.to(cfg.device)
        model.eval()
        
        # Evaluate on ID (validation set from best fold)
        print(f"\n📊 Evaluating In-Distribution (ROI) data...")
        best_val_idx = list(skf.split(files, labels))[int(best_fold.split('_')[1]) - 1][1]
        id_files = [files[i] for i in best_val_idx]
        id_labels = [labels[i] for i in best_val_idx]
        
        id_dataset = CXRDataset(
            id_files, id_labels, ref_hist,
            aug_mode=False, is_ood=False, method=cfg.preprocessing_method,
            image_size=cfg.image_size
        )
        id_loader = DataLoader(id_dataset, batch_size=cfg.batch_size, shuffle=False)
        
        id_metrics = evaluate_with_ensemble(
            model, id_loader, cfg.device, cfg.preprocessing_method, 
            cfg.ensemble_methods, ref_hist, use_ensemble=False
        )
        
        # Evaluate on OOD (full CXR) with preprocessing ensemble
        print(f"\n🌐 Evaluating Out-of-Distribution (Full CXR) with ensemble...")
        ood_dataset = CXRDataset(
            None, None, ref_hist,  # Files loaded internally
            aug_mode=False, is_ood=True, method=cfg.preprocessing_method,
            image_size=cfg.image_size, cxr_dir=cfg.cxr_dir
        )
        ood_loader = DataLoader(ood_dataset, batch_size=cfg.batch_size, shuffle=False)
        
        ood_metrics = evaluate_with_ensemble(
            model, ood_loader, cfg.device, cfg.preprocessing_method,
            cfg.ensemble_methods, ref_hist, use_ensemble=True
        )
        
        # Report degradation
        print(f"\n📉 Performance Degradation (ID → OOD):")
        for metric in ['accuracy', 'auc', 'sensitivity', 'specificity']:
            id_val = id_metrics[metric]
            ood_val = ood_metrics[metric]
            gap = id_val - ood_val
            print(f"  {metric:12s}: {id_val:.3f} → {ood_val:.3f} (gap: {gap:+.3f})")
        
        # Uncertainty analysis
        if 'uncertainty' in ood_metrics:
            unc = ood_metrics['uncertainty']
            print(f"\n🎲 Uncertainty Analysis:")
            print(f"  Mean uncertainty: {np.mean(unc):.3f}")
            print(f"  Std uncertainty:  {np.std(unc):.3f}")
            print(f"  High uncertainty (>{np.percentile(unc, 80):.3f}): {np.mean(unc > np.percentile(unc, 80))*100:.1f}%")
        
        # Save OOD results
        ood_results = {
            'id_metrics': id_metrics,
            'ood_metrics': ood_metrics,
            'degradation': {m: id_metrics[m] - ood_metrics[m] for m in ['accuracy', 'auc', 'sensitivity', 'specificity']}
        }
        ood_path = os.path.join(cfg.output_dir, f'ood_results_{cfg.preprocessing_method}.pkl')
        with open(ood_path, 'wb') as f:
            pickle.dump(ood_results, f)
        print(f"\n  ✓ OOD results saved: {ood_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Final outputs in: {cfg.output_dir}")
    print(f"  • Results: results_{cfg.preprocessing_method}.pkl")
    print(f"  • Model:   best_model_{cfg.preprocessing_method}.pt")
    print(f"  • Plots:   training_curves.png, metrics_bar.png")
    if os.path.exists(cfg.cxr_dir):
        print(f"  • OOD:     ood_results_{cfg.preprocessing_method}.pkl")
    print()


# ============================================================================
# Visualization Helpers
# ============================================================================
def plot_training_curves(histories, results, output_dir):
    """Plot training/validation curves for all folds"""
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for i, (fold_name, history) in enumerate(histories.items()):
        epochs = history['epoch']
        plt.plot(epochs, history['train_loss'], '-', color=colors[i], label=f'{fold_name} Train', alpha=0.7)
        if history['val_loss']:
            plt.plot(epochs[:len(history['val_loss'])], history['val_loss'], '--', color=colors[i], label=f'{fold_name} Val', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(1, 3, 2)
    for i, (fold_name, history) in enumerate(histories.items()):
        epochs = history['epoch']
        plt.plot(epochs, history['train_acc'], '-', color=colors[i], label=f'{fold_name} Train', alpha=0.7)
        if history['val_acc']:
            plt.plot(epochs[:len(history['val_acc'])], history['val_acc'], '--', color=colors[i], label=f'{fold_name} Val', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Final metrics bar chart
    plt.subplot(1, 3, 3)
    metrics = ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score', 'auc']
    x = np.arange(len(metrics))
    width = 0.35
    
    means = [np.mean([results[f][m] for f in results]) for m in metrics]
    stds = [np.std([results[f][m] for f in results]) for m in metrics]
    
    plt.bar(x - width/2, means, width, yerr=stds, capsize=4, label='Mean ± Std')
    plt.xticks(x, [m.replace('_', '\n') for m in metrics], rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Cross-Validation Metrics')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves saved: {plot_path}")


# ============================================================================
# Entry Point
# ============================================================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PTB Detection Pipeline')
    parser.add_argument('--preprocessing', type=str, default='histmatch',
                       choices=['none', 'clahe', 'histmatch'],
                       help='Preprocessing method (default: histmatch)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced data/epochs')
    parser.add_argument('--no-gradcam', action='store_true',
                       help='Disable GradCAM loss')
    parser.add_argument('--no-anatomical', action='store_true',
                       help='Disable anatomical guidance loss')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Initialize configuration
    cfg = Config()
    
    # Apply command line overrides
    cfg.preprocessing_method = args.preprocessing
    cfg.quick_test = args.quick_test
    cfg.use_gradcam = not args.no_gradcam
    cfg.use_anatomical = not args.no_anatomical
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    
    # Print configuration
    cfg.print_summary()
    
    # Run pipeline
    try:
        run_pipeline(cfg)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)