# LaTeX Figure Descriptions for Results Section

## Figure 1: Confusion Matrix and Classification Performance

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{results_visualizations/confusion_matrix.png}
\caption{Confusion matrix and classification performance metrics for the optimized multi-task model. The model achieves 96.8\% accuracy with high sensitivity (96.9\%) and specificity (95.3\%), demonstrating excellent diagnostic capability for chest X-ray classification.}
\label{fig:confusion_matrix}
\end{figure}
```

**Caption Text**: Confusion matrix and classification performance metrics for the optimized multi-task model. The model achieves 96.8% accuracy with high sensitivity (96.9%) and specificity (95.3%), demonstrating excellent diagnostic capability for chest X-ray classification.

---

## Figure 2: ROC Curves and AUC Analysis

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{results_visualizations/roc_curves.png}
\caption{ROC curves for 5-fold cross-validation showing consistent high performance across all folds. The average AUC of 0.994 demonstrates excellent discriminative ability, with individual fold AUCs ranging from 0.990 to 0.996.}
\label{fig:roc_curves}
\end{figure}
```

**Caption Text**: ROC curves for 5-fold cross-validation showing consistent high performance across all folds. The average AUC of 0.994 demonstrates excellent discriminative ability, with individual fold AUCs ranging from 0.990 to 0.996.

---

## Figure 3: Comprehensive Performance Analysis

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{results_visualizations/performance_comparison.png}
\caption{Comprehensive performance analysis showing (a) classification vs segmentation metrics, (b) cross-validation consistency across folds, (c) performance distribution with error bars, and (d) key results summary. The model achieves excellent classification performance (96.8\% accuracy) while maintaining reasonable segmentation quality (27.1\% IoU).}
\label{fig:performance_comparison}
\end{figure}
```

**Caption Text**: Comprehensive performance analysis showing (a) classification vs segmentation metrics, (b) cross-validation consistency across folds, (c) performance distribution with error bars, and (d) key results summary. The model achieves excellent classification performance (96.8% accuracy) while maintaining reasonable segmentation quality (27.1% IoU).

---

## Figure 4: Cross-Validation Analysis

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{results_visualizations/cv_analysis.png}
\caption{Cross-validation analysis showing performance distribution across 5 folds for all metrics. Box plots demonstrate consistent performance with low variance, indicating robust model generalization. Mean performance with standard deviations: Accuracy 96.8±0.8\%, IoU 27.1±2.9\%, Dice 41.5±3.5\%.}
\label{fig:cv_analysis}
\end{figure}
```

**Caption Text**: Cross-validation analysis showing performance distribution across 5 folds for all metrics. Box plots demonstrate consistent performance with low variance, indicating robust model generalization. Mean performance with standard deviations: Accuracy 96.8±0.8%, IoU 27.1±2.9%, Dice 41.5±3.5%.

---

## Figure 5: Kriging Optimization Progress

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{results_visualizations/optimization_progress.png}
\caption{Kriging optimization progress showing (a) objective function evolution, (b) loss weight parameter evolution, (c) improvement over initial configuration, and (d) optimization summary. The Bayesian optimization achieved 166\% improvement and converged to the optimal solution at iteration 2, demonstrating efficient hyperparameter search.}
\label{fig:optimization_progress}
\end{figure}
```

**Caption Text**: Kriging optimization progress showing (a) objective function evolution, (b) loss weight parameter evolution, (c) improvement over initial configuration, and (d) optimization summary. The Bayesian optimization achieved 166% improvement and converged to the optimal solution at iteration 2, demonstrating efficient hyperparameter search.

---

## Table Integration

### Table 1: Hyperparameter Search Space and Optimal Values

```latex
\begin{table}[h]
\centering
\caption{Hyperparameter Search Space and Optimal Values}
\label{tab:hyperparams}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Search Range} & \textbf{Optimal Value} & \textbf{Description} \\
\midrule
$\lambda_{cam}$ & [0.0, 5.0] & 4.5290 & GradCAM loss weight \\
$\lambda_{seg}$ & [0.0, 5.0] & 4.8530 & Segmentation loss weight \\
Learning Rate & [1e-5, 5e-3] & 0.0002 & Initial learning rate \\
Decay & [0.001, 0.1] & 0.0042 & Learning rate decay \\
Momentum & [0.8, 0.99] & 0.8725 & SGD momentum \\
Weight Decay & [1e-6, 1e-3] & 0.0007 & L2 regularization \\
Batch Size & [8, 32] & 14.12 & Training batch size \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 2: Cross-Validation Results

```latex
\begin{table}[h]
\centering
\caption{Cross-Validation Results by Fold}
\label{tab:cv_results}
\begin{tabular}{@{}ccccc@{}}
\toprule
\textbf{Fold} & \textbf{Accuracy (\%)} & \textbf{IoU (\%)} & \textbf{Dice (\%)} & \textbf{AUC (\%)} \\
\midrule
1 & 97.4 & 26.0 & 40.3 & 99.1 \\
2 & 97.4 & 25.4 & 39.4 & 99.4 \\
3 & 95.6 & 24.6 & 38.5 & 99.0 \\
4 & 97.3 & 31.9 & 47.2 & 99.6 \\
5 & 96.4 & 27.5 & 42.3 & 99.5 \\
\midrule
\textbf{Mean ± Std} & \textbf{96.8 ± 0.8} & \textbf{27.1 ± 2.9} & \textbf{41.5 ± 3.5} & \textbf{99.4 ± 0.3} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Usage Instructions

1. **Run the visualization script**: Execute `run_visualizations.m` in MATLAB
2. **Copy the generated PNG files** to your LaTeX project directory
3. **Use the LaTeX code snippets** above in your results section
4. **Adjust figure sizes** as needed for your paper format
5. **Modify captions** to match your specific writing style

## File Organization

```
your_paper/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── performance_comparison.png
│   ├── cv_analysis.png
│   └── optimization_progress.png
└── paper.tex
```

## Additional Notes

- All figures are generated at high resolution (suitable for publication)
- Both PNG and FIG formats are saved for flexibility
- Colors are optimized for both color and grayscale printing
- Font sizes are appropriate for academic publications
- Error bars and confidence intervals are included where relevant
