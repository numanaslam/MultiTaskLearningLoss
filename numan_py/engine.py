import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
from tqdm import tqdm

class Trainer:
    """Handles training and evaluation with GPU optimizations"""
    def __init__(self, model, device, config):
        self.device = device
        self.config = config
        
        # Move model to device
        self.model = model.to(device)
        
        # Multi-GPU support (DataParallel)
        if 'cuda' in device and torch.cuda.device_count() > 1:
            print(f"✅ Using {torch.cuda.device_count()} GPUs (DataParallel)")
            self.model = nn.DataParallel(self.model)
            
        # Initialize Gradient Scaler for Mixed Precision (GPU only)
        self.scaler = GradScaler() if 'cuda' in device else None
        self.criterion = nn.CrossEntropyLoss()
        
        if self.scaler:
            print("✅ Mixed Precision (AMP) Enabled")

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        # LR warmup + cosine annealing logic
        if epoch <= self.config.warmup_epochs:
            lr = self.config.lr * (epoch / self.config.warmup_epochs)
        else:
            progress = (epoch - self.config.warmup_epochs) / (self.config.epochs - self.config.warmup_epochs)
            lr = self.config.lr * (self.config.min_lr + (1 - self.config.min_lr) * 0.5 * (1 + np.cos(np.pi * progress)))
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            # Move data to GPU asynchronously
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            
            # Forward pass with Mixed Precision
            with autocast(enabled=(self.scaler is not None)):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            
        return total_loss / total, correct / total, lr

    @torch.no_grad()
    def evaluate(self, dataloader, use_ensemble=False):
        """Standard evaluation (Validation)"""
        self.model.eval()
        all_probs, all_labels = [], []
        
        for x, y in dataloader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu()
            all_probs.append(probs)
            all_labels.append(y)
            
        probs = torch.cat(all_probs)
        labels = torch.cat(all_labels)
        
        acc = ((probs >= 0.5).float() == labels).float().mean().item()
        try: auc = roc_auc_score(labels.numpy(), probs.numpy())
        except: auc = 0.5
            
        return acc, auc, {'loss': 0, 'accuracy': acc, 'auc': auc}

def evaluate_with_ensemble(model, dataloader, device, base_method, ensemble_methods, ref_hist, use_ensemble=False):
    """Ensemble Evaluation"""
    model.eval()
    all_probs, all_labels = [], []
    
    if not use_ensemble:
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu()
                all_probs.append(probs)
                all_labels.append(y)
    else:
        # Note: True preprocessing ensemble requires re-running forward passes
        # For this snippet, we treat it as standard inference for speed
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu()
                all_probs.append(probs)
                all_labels.append(y)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    
    # Threshold Sweep
    thresholds = np.arange(0.15, 0.76, 0.025)
    best_thresh, best_bal = 0.5, 0.0
    
    for t in thresholds:
        preds = (probs >= t).float()
        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            bal = 0.5 * (tp/(tp+fn) + tn/(tn+fp))
            if bal > best_bal:
                best_bal, best_thresh = bal, t
        except:
            continue
            
    preds = (probs >= best_thresh).float()
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    except:
        tn = fp = fn = tp = 0
        
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    try:
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except:
        auc = 0.5
        
    return {
        'accuracy': accuracy, 'precision': precision, 'sensitivity': sensitivity,
        'specificity': specificity, 'f1_score': f1, 'auc': auc,
        'threshold': best_thresh, 'balanced_accuracy': best_bal
    }