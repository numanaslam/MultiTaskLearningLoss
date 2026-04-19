# Research Paper Visualization Generator

This MATLAB script generates comprehensive visualizations for a top research paper using the optimal hyperparameters found through Kriging optimization.

## Overview

The script `generate_research_paper_visualizations.m` creates 12 different types of publication-ready figures that demonstrate the effectiveness of the multi-task learning approach for chest X-ray analysis.

## Features

### 1. **Confusion Matrix and Classification Performance**
- Detailed confusion matrix with performance metrics
- Cross-validation consistency analysis
- Statistical significance indicators

### 2. **ROC Curves and AUC Analysis**
- Individual fold ROC curves
- Average ROC curve with confidence intervals
- AUC distribution across folds
- Performance comparison metrics

### 3. **Multi-Task Performance Comparison**
- Classification vs segmentation performance
- Loss function component analysis
- Cross-validation consistency
- Objective function weight distribution

### 4. **Cross-Validation Analysis**
- Box plots for each performance metric
- Statistical summary with confidence intervals
- Variability analysis across folds

### 5. **Loss Function Components Analysis**
- Evolution of loss components during training
- Weighted loss contributions
- Training stability analysis
- Convergence analysis

### 6. **GradCAM Visualization Examples**
- Attention map examples for different cases
- Overlay visualizations
- Quality assessment examples

### 7. **Segmentation Performance Examples**
- Prediction vs ground truth comparisons
- Performance metrics visualization
- Clinical relevance demonstration

### 8. **Hyperparameter Sensitivity Analysis**
- Parameter importance ranking
- Search space coverage analysis
- Optimization efficiency metrics

### 9. **Training Progress and Convergence**
- Training curves for all metrics
- Loss convergence analysis
- Cross-validation consistency
- Early stopping analysis

### 10. **Statistical Analysis and Error Bars**
- Performance with error bars
- 95% confidence intervals
- Coefficient of variation analysis
- Statistical significance testing

### 11. **Baseline Method Comparison**
- Comparison with single-task methods
- Multi-task benefits quantification
- Computational efficiency analysis

### 12. **Attention Map Evolution**
- Attention quality evolution during training
- Attention distribution analysis
- Cross-validation consistency
- Performance correlation analysis

## Usage

### Prerequisites

1. **MATLAB Requirements:**
   - Deep Learning Toolbox
   - Computer Vision Toolbox
   - Statistics and Machine Learning Toolbox
   - Image Processing Toolbox

2. **Data Requirements:**
   - Pre-trained VGG16 network (`vgg16_finetuned_on_roi.mat`)
   - Chest X-ray dataset in `input/roi/` directory
   - Corresponding masks in `input/masks/` directory
   - Precomputed GradCAM maps (optional, will be generated if not available)

3. **File Structure:**
   ```
   project/
   ├── generate_research_paper_visualizations.m
   ├── vgg16_finetuned_on_roi.mat
   ├── input/
   │   ├── roi/
   │   │   ├── normal/
   │   │   └── ptb/
   │   └── masks/
   │       ├── normal/
   │       └── ptb/
   └── paper_figures/ (created automatically)
   ```

### Running the Script

1. **Basic Usage:**
   ```matlab
   generate_research_paper_visualizations
   ```

2. **The script will:**
   - Load the optimal hyperparameters from Kriging optimization
   - Train the model using 5-fold cross-validation
   - Generate all 12 visualization types
   - Save high-resolution figures in `paper_figures/high_res/`
   - Save standard resolution figures in `paper_figures/`
   - Generate LaTeX figure references

### Output Files

The script creates the following output structure:

```
paper_figures/
├── high_res/                    # High-resolution figures (300 DPI)
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── multitask_performance.png
│   ├── cv_analysis.png
│   ├── loss_analysis.png
│   ├── gradcam_examples.png
│   ├── segmentation_examples.png
│   ├── hyperparameter_sensitivity.png
│   ├── training_progress.png
│   ├── statistical_analysis.png
│   ├── baseline_comparison.png
│   └── attention_evolution.png
├── *.png                        # Standard resolution figures
├── *.fig                        # MATLAB figure files
└── latex_figure_references.tex  # LaTeX figure references
```

## Optimal Hyperparameters Used

The script uses the optimal hyperparameters found through Kriging optimization:

- **λ_cam (GradCAM weight):** 4.5290
- **λ_seg (Dice loss weight):** 4.8530
- **Learning Rate:** 0.0002
- **Weight Decay:** 0.0007
- **Batch Size:** 14
- **Momentum:** 0.8725
- **Decay:** 0.0042

**Note:** Tversky loss is used for evaluation and loss function comparisons only, not in the training loop.

## Key Features

### 1. **Publication-Ready Quality**
- High-resolution figures (300 DPI)
- Professional color schemes
- Clear labels and legends
- Consistent formatting

### 2. **Comprehensive Analysis**
- 12 different visualization types
- Multi-task learning with 3 loss components (Classification, GradCAM, Dice)
- Tversky loss for evaluation and comparison
- Statistical analysis with error bars
- Cross-validation consistency
- Baseline comparisons

### 3. **Clinical Relevance**
- GradCAM attention visualizations
- Segmentation performance examples
- Multi-task learning benefits
- Interpretability analysis

### 4. **Technical Rigor**
- Statistical significance testing
- Confidence intervals
- Effect size analysis
- Robust evaluation metrics

## Customization

### Modifying Hyperparameters

To use different hyperparameters, edit the `optimal_params` structure in the script:

```matlab
optimal_params.lambda_cam = 4.5290;      % GradCAM loss weight
optimal_params.lambda_seg = 4.8530;      % Segmentation loss weight
optimal_params.initialLearnRate = 0.0002; % Learning rate
optimal_params.weightDecay = 0.0007;     % Weight decay
optimal_params.batchSize = 14;           % Batch size
```

### Adjusting Training Parameters

Modify training configuration:

```matlab
k_folds = 5;           % Cross-validation folds
numEpochs = 30;        % Training epochs
patience = 10;         % Early stopping patience
min_delta = 1e-5;      % Minimum improvement threshold
```

### Customizing Visualizations

Each visualization function can be modified independently:

- **Color schemes:** Modify `colors` variables in each function
- **Figure sizes:** Adjust `Position` parameters in `figure()` calls
- **Font sizes:** Modify `FontSize` parameters
- **Data ranges:** Adjust axis limits and scaling

## Performance Notes

- **Training Time:** ~2-3 hours on GPU (RTX 3080)
- **Memory Usage:** ~8-12 GB GPU memory
- **Output Size:** ~50-100 MB for all figures
- **Compatibility:** MATLAB R2021a or later

## Troubleshooting

### Common Issues

1. **Out of Memory:**
   - Reduce batch size
   - Use CPU instead of GPU
   - Process fewer samples

2. **Missing Files:**
   - Ensure all required files are in correct locations
   - Check file permissions
   - Verify MATLAB toolbox availability

3. **Figure Quality:**
   - Use high-resolution settings
   - Check DPI settings
   - Verify color profiles

### Error Messages

- **"File not found":** Check file paths and names
- **"Out of memory":** Reduce batch size or use CPU
- **"Toolbox not found":** Install required MATLAB toolboxes

## Citation

If you use this visualization script in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Multi-Task Learning for Chest X-Ray Analysis with Kriging-Optimized Hyperparameters},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Support

For questions or issues:
1. Check the troubleshooting section
2. Verify all prerequisites are met
3. Check MATLAB documentation for specific functions
4. Review the original Kriging optimization results

## License

This script is provided for research purposes. Please ensure compliance with your institution's policies and any applicable licenses for the datasets and pre-trained models used.
