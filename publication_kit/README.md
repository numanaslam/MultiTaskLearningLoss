# Publication Figure Kit

For a research-publication-quality output from the rework.m PTB distillation pipeline. Three files:

- `publication_figures.m` — Standalone figure generator. Loads a saved `.mat` model and emits 10 figures + a master metrics table.
- `compute_faithfulness_metrics.m` — Helper that produces energy ratio, pointing-game accuracy, and teacher-student CAM cosine for any dataset.
- `rework_logging_patch.md` — Two small edits to add to `rework.m` that enable the loss-curve and alignment-progress figures.

## Quick start

After completing a training run with `rework.m`, run:

```matlab
addpath('publication_kit');
publication_figures('models/trained_mixed_vgg16.mat', 'paper_figures');
```

That writes 10 PDFs + 10 PNGs + `table_metrics.csv` to `paper_figures/`. PDFs are vector format for LaTeX; PNGs are 300 DPI for slides and supplementary review.

For cross-site OOD figures (when you have a second dataset):

```matlab
publication_figures('models/trained_mixed_vgg16.mat', 'paper_figures', ...
    {'C:/data/montgomery', 'C:/data/tbx11k'});
```

## What gets produced

| File | Purpose | Paper section |
|---|---|---|
| `fig01_roc_curves` | ROC + AUC for every eval set | Results — primary metric |
| `fig02_pr_curves` | Precision-recall + AP | Results — class-imbalance-aware metric |
| `fig03_reliability` | Calibration diagram + ECE | Results — calibration claim |
| `fig04_loss_curves` | Training dynamics per fold | Methods — convergence evidence |
| `fig05_alignment_progress` | Teacher-student cosine over epochs | Methods — proves CAM loss works |
| `fig06_faithfulness_metrics` | Energy ratio + pointing-game | Discussion — interpretability claim |
| `fig07_gradcam_comparison` | Teacher vs student CAM, with mask | Results — qualitative CAM analysis |
| `fig08_failure_modes` | High-confidence wrong predictions | Discussion — limitations |
| `fig09_threshold_analysis` | Sens/spec/F1 vs threshold | Results — operating-point selection |
| `fig10_confusion_matrices` | Confusion matrix grid | Results — per-class breakdown |
| `table_metrics.csv` | Master numerical results | Results — Table 1 |

Figures 1-3 are the headline results figures. Figures 4-5 show your method actually trains. Figure 6 is the interpretability story. Figure 7 is the qualitative evidence that justifies figure 6. Figures 8-10 are diagnostics.

## Required: add epoch-level logging to rework.m

Figures 4 and 5 require per-epoch metric history that the current `rework.m` doesn't save. Two small edits — see `rework_logging_patch.md` for the exact code. Without these, `publication_figures` will skip figs 4 and 5 with a warning; everything else still works.

## What I did NOT include, and why

**No t-SNE / UMAP feature visualization.** These look impressive but rarely communicate anything reviewers can't get from the ROC + reliability figures. Skip unless cross-site experiments reveal genuine distribution shift, then add as supplementary.

**No saliency-map comparison across methods (Grad-CAM vs Grad-CAM++ vs ScoreCAM).** Different paper. If a reviewer asks, add to revision.

**No per-class energy ratio.** Easy to add if reviewers ask — just split the `metrics.energy_ratio` vector by class.

**No bootstrap confidence intervals on AUC.** Recommended for a final paper version; not in this kit because it adds 10× runtime. To add: wrap the AUC computation in `bootci` with N=1000.

## Plotting style notes

Default style is set in `setPaperStyle()` at the top of `publication_figures.m`:
- Helvetica 10 pt
- White background
- 1.5 pt line width
- 300 DPI raster export

To match a specific journal style (IEEE: Times New Roman 9 pt, Nature: Helvetica 7 pt), edit that one function. All figures use it.

## Tested with

- MATLAB R2022a and later (uses `exportgraphics`, `boxchart`, `dlresize`)
- Deep Learning Toolbox required
- Image Processing Toolbox required (for `imerode`, `strel`, `ind2rgb`)
- `perfcurve` from Statistics and Machine Learning Toolbox required for AUC/AP

If `exportgraphics` is unavailable (R2019b or older), the kit falls back to `saveas` — PDFs will be raster, not vector. Upgrade to R2020a+ for publication quality.
