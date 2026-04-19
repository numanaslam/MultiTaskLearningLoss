# Folder Structure Organization Plan

## Proposed Structure

```
custom network/
├── src/                          # Source code (organized by function)
│   ├── training/                 # Training scripts
│   ├── evaluation/               # Evaluation scripts
│   ├── utils/                    # Helper functions
│   ├── visualization/            # Visualization scripts
│   └── loss_functions/           # Loss function implementations
│
├── models/                       # Saved model checkpoints
│   ├── checkpoints/              # Training checkpoints
│   ├── final/                    # Final trained models
│   └── pretrained/               # Pretrained models (VGG16, etc.)
│
├── results/                      # Results and outputs
│   ├── training/                 # Training results
│   ├── evaluation/               # Evaluation results
│   ├── ood/                      # OOD evaluation results
│   ├── comparisons/              # Loss function comparisons
│   ├── figures/                  # Generated figures
│   └── reports/                  # Generated reports
│
├── docs/                         # Documentation
│   ├── guides/                   # User guides
│   ├── analysis/                 # Analysis reports
│   └── papers/                   # Paper-related docs
│
├── scripts/                      # Utility scripts
│   ├── data_preprocessing/       # Data organization scripts
│   └── analysis/                 # Analysis scripts
│
├── config/                       # Configuration files
│
├── input/                        # Input data (keep as is)
├── output/                       # Output data (keep as is)
│
└── README.md                     # Main README
```

## File Categorization

### Training Scripts → `src/training/`
- `train_model_fixed_overfitting.m`
- `train_final_model.m` (to be created)
- `compare_loss_functions.m`
- `train_vgg16_gradcam.m`
- `train_vgg16_lime*.m`
- `training_with_precomputed_gradcam.m`
- `kriging_hyperparameter_optimization.m`

### Evaluation Scripts → `src/evaluation/`
- `evaluate_with_tta.m`
- `evaluate_ood_performance.m`
- `evaluate_cam_ood*.m`
- `test_ood_*.m`
- `test_vgg16_on_cxr.m`

### Loss Functions → `src/loss_functions/`
- `custom_loss_gradcam_*.m`
- `Loss_grad_cam.m`
- `VGG_loss_lime_entropy.m`
- `Resnet50_Loss.m`

### Visualization → `src/visualization/`
- `visualize_gradcam.m`
- `generate_research_paper_visualizations.m`
- `create_publication_visualizations_fixed.m`
- `create_comprehensive_visualizations*.m`
- `create_key_visualizations.m`
- `run_visualizations.m`
- `test_visualization_generator.m`

### Utility Scripts → `src/utils/`
- Helper functions (to be extracted from main scripts)
- Data loading functions
- Metric calculation functions

### Data Preprocessing → `scripts/data_preprocessing/`
- `organize_cxr_*.m`
- `resize_and_*.m`
- `precompute_gradcams.m`
- `create_roi_from_cxr_and_masks.m`
- `copy_masks_only.m`
- `remove_images_without_masks.m`
- `analyze_resized_folder.m`

### Analysis Scripts → `scripts/analysis/`
- `segmentation_mask_similarity_metrics.m`
- `vgg16_cam_mask_similarity_metrics.m`
- `ssim_based_loss_*.m`
- `diagnostic_test.m`
- `debug_*.m`
- `function ablation_and_ood_experiments.m`
- `function fast_kfold_segmentation_enhance.m`

### Documentation → `docs/`
- All `.md` files
- All `.tex` files
- `methodology_1.qmd`

### Models → `models/`
- `*.mat` model files
- `vgg16_*.mat`
- `custom_cnn_and_vgg16_*.mat`

### Results → `results/`
- Existing `results/` folder structure
- `results_visualizations/`
- Generated reports and figures

