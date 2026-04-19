# Domain Adaptation: Aligning ROI and Full CXR Feature Spaces

## Overview

Domain adaptation is a technique to improve a model's performance on a target domain (full CXR images) when trained on a source domain (ROI images). The goal is to learn domain-invariant features that work well across both distributions.

---

## The Problem

### Current Situation:
- **Source Domain (Training)**: ROI images (cropped, focused on lung regions)
- **Target Domain (Testing)**: Full CXR images (complete chest X-rays with more context)
- **Issue**: Model learns ROI-specific features → poor generalization to full CXR

### Distribution Shift:
1. **Spatial differences**: ROI images are pre-cropped, full CXR has variable lung positions
2. **Context differences**: Full CXR includes background, ribs, heart, etc.
3. **Scale differences**: Objects appear at different scales
4. **Feature differences**: Model learns ROI-specific patterns that don't generalize

---

## How Domain Adaptation Works

### Core Idea:
Instead of learning features specific to ROI images, learn **domain-invariant features** that capture the underlying pathology (PTB vs Normal) regardless of whether the input is ROI or full CXR.

### Visual Concept:
```
Traditional Training:
ROI Images → [ROI-Specific Features] → Classification (works only on ROI)

Domain Adaptation:
ROI Images ──┐
            ├→ [Domain-Invariant Features] → Classification (works on both!)
Full CXR ───┘
```

---

## Domain Adaptation Techniques

### 1. **Domain Adversarial Neural Network (DANN)**

**How it works:**
- Add a **domain discriminator** that tries to distinguish ROI vs Full CXR
- Train the feature extractor to **fool** the discriminator (make features indistinguishable)
- Train the classifier to use domain-invariant features for classification

**Architecture:**
```
Input Image
    ↓
Feature Extractor (VGG16 conv layers)
    ↓
    ├─→ Classifier (PTB vs Normal)  [Primary task - minimize classification loss]
    │
    └─→ Domain Discriminator (ROI vs Full CXR)  [Adversarial - maximize domain confusion]
```

**Training Process:**
1. **Forward pass**: Extract features, classify pathology, predict domain
2. **Classification loss**: Minimize pathology classification error
3. **Domain loss**: 
   - **Maximize** domain discriminator accuracy (gradient reversal)
   - **Minimize** domain classification (features become domain-invariant)
4. **Gradient reversal**: Reverse gradients from domain discriminator so feature extractor learns to fool it

**Key Formula:**
```
Total Loss = Classification Loss - λ_domain * Domain Loss
```
(Note: Negative sign because we want to maximize domain confusion)

---

### 2. **Maximum Mean Discrepancy (MMD)**

**How it works:**
- Minimize the distance between feature distributions from ROI and Full CXR
- Uses kernel methods to measure distribution difference
- Forces features to be similar across domains

**Concept:**
```
ROI Features:     [f1, f2, f3, ...]
Full CXR Features: [g1, g2, g3, ...]

Goal: Make mean(ROI features) ≈ mean(Full CXR features)
```

---

### 3. **Correlation Alignment (CORAL)**

**How it works:**
- Aligns second-order statistics (covariance) of features
- Less complex than MMD, faster to compute
- Transforms source features to match target feature distribution

**Algorithm:**
1. Compute covariance matrices for ROI and Full CXR features
2. Transform ROI features: `features_coral = features_roi * transformation_matrix`
3. Transformation matrix derived from covariance matrices

---

### 4. **Mixup Between Domains**

**How it works:**
- Mix ROI and Full CXR images during training
- Create synthetic examples with mixed labels:
  - Image: 0.7 * ROI_image + 0.3 * FullCXR_image
  - Label: 0.7 * ROI_label + 0.3 * FullCXR_label

**Benefits:**
- Encourages model to learn from both domains
- Creates smooth interpolation between distributions

---

## Implementation Strategy for Your Problem

### Recommended: **Domain Adversarial Training (DANN)**

**Why DANN?**
1. **Effective**: Proven to work well for medical imaging domain shifts
2. **Interpretable**: Can visualize learned domain-invariant features
3. **Flexible**: Can use unlabeled full CXR images (if available)
4. **Compatible**: Works with your existing multi-task loss

**Architecture Modification:**
```
Your Current Architecture:
ROI Image → VGG16 → [Classification Head + GradCAM Head + Segmentation Head]

DANN Architecture:
ROI Image ──┐
            ├→ VGG16 Feature Extractor
Full CXR ──┘
            ↓
    ┌───────┴───────┐
    │               │
Classification   Domain Discriminator
(PTB/Normal)    (ROI/FullCXR)
```

---

## Training Process

### Step 1: Prepare Data
- **Source (ROI)**: Labeled training images
- **Target (Full CXR)**: Can be labeled OR unlabeled (DANN advantage)
- Mix batches: 50% ROI, 50% Full CXR

### Step 2: Define Loss Functions
```matlab
% Classification loss (same as before)
classification_loss = crossentropy(predictions, true_labels)

% Domain adversarial loss
domain_loss = -crossentropy(domain_predictions, domain_labels)  % Negative for adversarial

% Existing losses
gradcam_loss = MSE(student_cam, teacher_cam)
segmentation_loss = Dice_loss(pred_mask, true_mask)

% Combined loss
total_loss = classification_loss + 
             lambda_cam * gradcam_loss + 
             lambda_seg * segmentation_loss +
             lambda_domain * domain_loss  % NEW
```

### Step 3: Training Loop
```matlab
for each batch:
    1. Forward pass (ROI images):
       - Extract features
       - Predict pathology class
       - Predict domain (should predict "ROI")
    
    2. Forward pass (Full CXR images):
       - Extract features
       - Predict domain (should predict "Full CXR")
       - If labeled: predict pathology class
    
    3. Compute losses:
       - Classification loss (on labeled images)
       - Domain loss (on all images)
       - GradCAM loss (on ROI only, as before)
       - Segmentation loss (on ROI only, as before)
    
    4. Backward pass:
       - Update classifier: minimize classification loss
       - Update feature extractor: minimize classification + maximize domain confusion
       - Update domain discriminator: maximize domain discrimination
       - Apply gradient reversal to domain loss (so feature extractor tries to fool discriminator)
```

---

## Gradient Reversal Layer (Key Component)

**Purpose:** Make the feature extractor learn to fool the domain discriminator

**How it works:**
```
Forward pass:
  domain_features → domain_discriminator → domain_prediction

Backward pass:
  domain_loss_gradient → gradient_reversal_layer → negative_gradient → feature_extractor
```

**Implementation:**
```matlab
% During forward pass: Pass through unchanged
domain_features_reversed = domain_features;

% During backward pass: Reverse gradients
domain_loss_gradient = -lambda_domain * domain_loss_gradient;  % Reverse sign!
```

---

## Expected Benefits

### 1. **Improved OOD Accuracy**
- Current: 66.3% with TTA
- Expected: 75-85% with domain adaptation
- Mechanism: Features become generalizable across domains

### 2. **Better Feature Quality**
- Features focus on pathology-relevant regions
- Less sensitive to domain-specific artifacts
- More robust to distribution shifts

### 3. **Reduced Performance Gap**
- Current degradation: 33.39%
- Expected degradation: 15-25%
- Better generalization to unseen distributions

---

## Implementation Challenges & Solutions

### Challenge 1: Unlabeled Full CXR Images
**Solution:** DANN can use unlabeled full CXR images for domain adaptation (only labeled ROI for classification)

### Challenge 2: Computational Cost
**Solution:** 
- Use smaller batch sizes (mix 50% ROI + 50% Full CXR)
- Train domain discriminator less frequently than classifier
- Start with lower `lambda_domain` (0.1-0.5), gradually increase

### Challenge 3: Balancing Losses
**Solution:**
- Start with small `lambda_domain` (0.1)
- Monitor both classification and domain discrimination accuracy
- Adjust weights dynamically if one loss dominates

### Challenge 4: Integration with Multi-Task Loss
**Solution:**
- Domain loss is independent
- Apply to all images, but classification/GradCAM/segmentation only on ROI
- Can be added as fourth loss component

---

## Comparison with Other Approaches

| Technique | OOD Improvement | Complexity | Implementation |
|-----------|----------------|------------|----------------|
| **TTA (Current)** | +5-10% | Low | ✅ Implemented |
| **Domain Adversarial** | +10-20% | Medium | Needs implementation |
| **Mixed Training** | +10-15% | Low | Easy, but need labels |
| **More Augmentation** | +3-5% | Very Low | ✅ Already done |
| **CORAL** | +5-10% | Low | Simpler than DANN |

---

## Quick Start Implementation

### Option 1: Simple Domain Adversarial (Recommended Start)
1. Add domain discriminator head to network
2. Mix ROI and Full CXR batches
3. Add domain loss with gradient reversal
4. Train with existing multi-task loss

### Option 2: CORAL (Simpler Alternative)
1. Compute feature statistics from ROI and Full CXR
2. Apply linear transformation to align distributions
3. Retrain classifier on aligned features

### Option 3: Mixed Training (Easiest)
1. Combine ROI and Full CXR in training batches
2. Apply same multi-task loss to both
3. Model learns from both domains simultaneously

---

## Next Steps

1. **Try Mixed Training first** (easiest, if Full CXR labels available)
2. **Implement DANN** (most effective, but more complex)
3. **Combine with TTA** (use both techniques together)

Would you like me to implement one of these approaches?

