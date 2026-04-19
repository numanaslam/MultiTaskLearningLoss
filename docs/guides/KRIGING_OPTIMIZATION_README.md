# Kriging Hyperparameter Optimization

This directory contains a Kriging-based hyperparameter optimization system for the multi-task learning approach with GradCAM and segmentation losses.

## Files

- `kriging_hyperparameter_optimization.m` - **Complete standalone file** with all optimization and helper functions
- `KRIGING_OPTIMIZATION_README.md` - This documentation file

## How to Use

### Quick Start
```matlab
% Simply run the main function - everything is included!
kriging_hyperparameter_optimization
```

That's it! The file is completely self-contained with all necessary functions included.

## What It Does

The Kriging optimization system:

1. **Defines Hyperparameter Search Space**:
   - `lambda_cam`: GradCAM loss weight [0.0, 0.3]
   - `lambda_seg`: Segmentation loss weight [0.0, 0.5]
   - `initialLearnRate`: Learning rate [1e-5, 1e-2]
   - `decay`: Learning rate decay [0.001, 0.1]
   - `momentum`: SGD momentum [0.8, 0.99]

2. **Uses Gaussian Process (Kriging)**:
   - Fits a surrogate model to predict objective function values
   - Uses Expected Improvement acquisition function
   - More efficient than grid search with fewer evaluations

3. **Optimization Process**:
   - Starts with 10 random initial points
   - Runs up to 50 optimization iterations
   - Each iteration evaluates hyperparameters using 5-fold cross-validation
   - Saves progress every 10 iterations

4. **Objective Function**:
   - Composite metric: 30% accuracy + 30% IoU + 20% Dice + 10% AUC + 10% F1
   - Focuses on segmentation performance while maintaining classification accuracy

## Output Files

- `kriging_optimization_results.mat` - Complete optimization results
- `kriging_optimization_progress.mat` - Intermediate progress (saved every 10 iterations)
- `kriging_optimization_results.png` - Visualization of optimization progress
- `kriging_optimization_report.txt` - Text summary of results

## Key Advantages over Grid Search

1. **Efficiency**: Requires fewer evaluations to find good hyperparameters
2. **Exploration**: Better balance between exploration and exploitation
3. **Uncertainty**: Provides uncertainty estimates for hyperparameter choices
4. **Scalability**: Works well with high-dimensional hyperparameter spaces
5. **Adaptive**: Learns from previous evaluations to guide future searches

## Configuration

You can modify the optimization parameters in `kriging_hyperparameter_optimization.m`:

```matlab
max_iterations = 50;           % Maximum optimization iterations
initial_points = 10;           % Initial random points
k_folds = 5;                   % Cross-validation folds
numEpochs = 20;                % Training epochs per evaluation
```

## Expected Runtime

- Each hyperparameter evaluation: ~5-10 minutes (depending on dataset size)
- Total optimization time: ~4-8 hours for 50 iterations
- Progress is saved every 10 iterations, so you can resume if interrupted

## Results Interpretation

The optimization will find the best hyperparameter combination that maximizes the composite objective function. The results include:

- Best hyperparameters found
- Performance metrics (accuracy, IoU, Dice, AUC, F1)
- Optimization history and convergence plots
- Statistical analysis of results

## Troubleshooting

1. **Memory Issues**: Reduce `miniBatchSize` or `numGradCAMPerBatch`
2. **GPU Issues**: Set `useGPU = false` if GPU memory is insufficient
3. **Long Runtime**: Reduce `max_iterations` or `numEpochs`
4. **Helper Functions Error**: Make sure `kriging_helper_functions.m` is in the same directory

## Comparison with Ablation Studies

This Kriging optimization complements the ablation studies by:
- Finding optimal hyperparameter combinations automatically
- Exploring continuous hyperparameter spaces
- Providing uncertainty estimates
- Being more efficient than exhaustive grid search

The results can be compared with the ablation study results to validate the optimization approach.
