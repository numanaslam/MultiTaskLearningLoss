# MultiTaskLearningLoss - Project Learnings So Far

## Scope
This project trains and evaluates a PTB classifier with optional multi-task losses (GradCAM/segmentation/anatomical), plus OOD evaluation on Full CXR with calibration and uncertainty-aware decision rules.

## What We Implemented
- Stability upgrades in training:
  - gradient clipping
  - LR warmup + cosine annealing
  - early stopping on validation accuracy
- Loss-formulation upgrades:
  - cosine CAM alignment option
  - anatomical positivity constraint
  - adaptive anatomical scaling
- OOD evaluation upgrades:
  - threshold tuning on ID validation (balanced/constrained)
  - temperature calibration
  - uncertainty scoring (entropy + confidence)
  - conservative OOD override rule (toggleable)

## Key Empirical Learnings
1. ID metrics can look acceptable while OOD collapses to PTB-heavy predictions.
- Repeatedly observed `%OOD above threshold = 100%` in some runs.
- This yielded `Sensitivity = 1.000` and `Specificity = 0.000` on OOD.

2. Threshold and calibration are necessary but not sufficient.
- ID threshold tuning and temperature scaling improved calibration on ROI validation.
- OOD probability shift remained large in hard runs (e.g., OOD PTB mean ~0.78 vs ID ~0.50).

3. Conservative uncertainty override helps only when predictions are near threshold.
- With strict thresholds and confident OOD-positive probabilities, overrides can be near zero.
- Overrides increase when margin/percentile are relaxed, but may trade off sensitivity/precision.

4. Configuration sensitivity is high in fast-dev (2 folds, 15 epochs).
- We saw large variance across runs/folds and unstable precision/sensitivity balance.
- Some runs improved CV AUC/Dice while OOD specificity still degraded.

5. Best recent balanced ID behavior did not fully transfer to OOD.
- Example recent run:
  - ID: Acc 0.677, Sens 0.665, Spec 0.689, AUC 0.702
  - OOD: Acc 0.475, Sens 1.000, Spec 0.000, AUC 0.592
- Mean degradation remained moderate (~21%).

## What Currently Seems Most Reliable
- Keep training preprocessing aligned (`preprocessing_method = 'none'`) while debugging OOD bias.
- Keep conservative OOD rule toggleable, not always-on for final claims.
- Use constrained threshold mode with explicit sensitivity floor, but monitor OOD specificity impact.
- Track probability-shift diagnostics every run (ID/OOD mean, median, % above threshold).

## Open Problems
- Persistent OOD positive bias (CXR predicted as PTB too often).
- Conservative override not activating enough in highly confident OOD-positive regimes.
- Cross-validation instability in short dev training.

## Recommended Next Experiments
1. Run structured ablation (fixed seed):
- `use_ood_dual_view`: false vs true
- conservative rule: off vs on
- threshold mode: balanced vs constrained

2. Add stronger OOD decision controls (optional branch):
- OOD-only threshold shift upward (`decisionThreshold + delta`)
- uncertainty-gated abstain flag instead of forced class flip

3. Improve stability before final comparison:
- increase folds/epochs for confirmation runs
- repeat each config across multiple seeds

## Current Bottom Line
The pipeline is substantially improved in engineering quality and observability. The main unresolved issue is OOD specificity collapse in certain configurations despite decent ID validation behavior. The next phase should prioritize controlled ablations around OOD decision policy and distribution shift handling.
