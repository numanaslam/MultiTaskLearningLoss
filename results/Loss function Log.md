The overall idea is reasonable, but the current pipeline is not yet a clean test of “train on ROI, generalize to full CXR.” After reading final\_working.m (line 1) and README.md (line 1), and checking it with MATLAB static analysis, the main issue is domain mismatch plus a few implementation inconsistencies.



Main Findings





High: final\_working.m is not a clean executable entrypoint. The file is named final\_working.m, but its primary function is train\_final\_model\_with\_preprocessing at final\_working.m (line 1). MATLAB flags this, and which('train\_final\_model\_with\_preprocessing') resolves to the separate file train\_final\_model\_with\_preprocessing.m, not final\_working.m. If you call train\_final\_model\_with\_preprocessing(...), you may not be running this file at all.



High: the requested preprocessing is ignored. The function accepts preprocessing\_method, prints it, and defaults to 'histmatch', but then hardcodes config.preprocessing\_method = 'none' at final\_working.m (line 111). So histogram-matching or z-score experiments are not actually being trained in this file.



High: the current OOD evaluation is not truly “full CXR” inference. With config.use\_ood\_dual\_view = false at final\_working.m (line 113), the OOD path crops a lung ROI from every full CXR before prediction at final\_working.m (line 1399). That is a different experiment from feeding raw full CXR.



Medium: focal loss still uses inverse-frequency class weights even when use\_class\_weights = false. The flag is set false at final\_working.m (line 89), but compute\_focal\_loss multiplies by classWeights anyway at final\_working.m (line 2429). That makes class-weight ablations semantically incorrect.



Medium: adaptive anatomical scaling is effectively broken. max\_anatomical\_scale = 100 at final\_working.m (line 107), and the scaling formula at final\_working.m (line 1163) always evaluates to 100, so there is no real ramp-up.



Medium: your “full training mode” message says 5 folds, but the code sets k\_folds = 3 at final\_working.m (line 45). That makes results harder to compare with the README and increases variance.



Medium: the model still trains only on ROI data from input/roi at final\_working.m (line 1601). Loss engineering alone will not fully solve a deployment domain that the model never sees during training.





I also verified your dataset shift directly: ROI mean intensity is 66.56, full CXR mean intensity is 148.34, with matched class counts 359 normal / 345 ptb in both domains. That means this is mainly a domain-shift problem, not a class-imbalance problem.



What This Means For The Model



The README’s main conclusion is directionally correct: threshold tuning and calibration help, but they are not the root fix for the OOD gap; the real problem is ROI-trained features not transferring to full-CXR context. I would not spend the next cycle mostly tuning lambda\_cam, lambda\_tversky, or focal alpha. The bigger win is to make training domain-aligned.



Recommended Training Process





Fix the code-path ambiguity first.



Use one real training file, with matching file/function name, and make preprocessing actually configurable.







Establish a clean baseline before multi-task loss.



Train a plain classifier first on:





ROI only



auto-ROI-from-CXR



raw full CXR



mixed ROI + full CXR









Because the classes are nearly balanced, start with plain cross-entropy or BCE before focal loss.



If your deployment input is full CXR, include full CXR during training.



Best option here is paired training, since your ROI and CXR folders appear aligned by filename/count. Use a shared backbone and train both views together:



matlab







L = Lcls(roi) + Lcls(cxr) ...

&#x20; + lambda\_cons \* KL(p\_roi, p\_cxr) ...

&#x20; + lambda\_anat \* Lanatomical ...

&#x20; + lambda\_cam \* Lcam;







The consistency term between ROI and full CXR is more important for generalization than making anatomical loss larger.



Keep auxiliary losses small and normalized.



For this setup, I’d make classification dominant and keep other terms near O(1):



lambda\_cam = 0.05 to 0.1 if using cosine CAM loss



lambda\_tversky = 0.25 to 0.5



lambda\_anatomical = 0.01 to 0.05



Avoid big hidden scale factors like x100 or x1000 unless you log component magnitudes every epoch.







Evaluate three separate test settings.



Report them separately, not as one “OOD” bucket:



ROI validation



full CXR with heuristic ROI extraction



raw full CXR without ROI extraction







Use model selection based on balanced accuracy or AUC, not plain argmax accuracy.



Right now early stopping uses validation accuracy at final\_working.m (line 963). For this problem, balanced accuracy is a better target.



The short version is: the loss is not the main failure point yet; the train/test domain definition is. Train on mixed or paired ROI+CXR, add a cross-domain consistency loss, keep classification dominant, and treat ROI-extracted-CXR and raw full-CXR as separate evaluation regimes.







Plan for this pass:



Create a new VGG16 baseline trainer that uses one paired split and trains three domain baselines from it:

roi, cxr, and mixed.

It will save three separate .mat models plus split/metrics metadata so we can reuse the same teacher family later.



Update final\_working.m so evaluation is case-based rather than one OOD number.

I’m targeting these cases:

roi\_val, cxr\_raw, cxr\_roi, and cxr\_dual\_view.

I’ll also make the teacher backbone selectable by name/path so the student stage can later point to ROI, CXR, or mixed teachers cleanly.



Run MATLAB static checks on the new code and give you a concrete training schedule for:

teacher pretraining, student training with final\_rework.m, and comparison experiments.





\--------------------------

Workable training plan:



Train the three teachers first with the new baseline file and keep the same random\_seed for fair comparison.

Compare the saved validation metrics and pick mixed as the likely primary teacher for student training, while keeping roi as the control baseline.

In final\_working.m, set config.teacher\_model\_spec to 'roi', 'cxr', or 'mixed' before student training/evaluation.

For the first student-loss pass in final\_rework.m, train on ROI only, but run evaluation on all four cases from final\_working.m.

Compare cxr\_raw against cxr\_roi. If cxr\_roi is much better than cxr\_raw, the main problem is still domain/context shift rather than loss design.

Only after that, tune loss weights. Start with teacher choice fixed and vary one loss component at a time.

\---------------------

training on roi



Cross-Validation Results (Mean ± Std across 3 folds):

&#x20; Accuracy: 0.615 ± 0.092

&#x20; Precision: 0.422 ± 0.366

&#x20; Sensitivity: 0.510 ± 0.443

&#x20; Specificity: 0.717 ± 0.246

&#x20; F1-Score: 0.462 ± 0.400

&#x20; AUC: 0.639 ± 0.115

&#x20; IoU: 0.145 ± 0.081

&#x20; Dice: 0.231 ± 0.124

&#x20; Tversky: 0.269 ± 0.143

=== MULTI-CASE EVALUATION ===

Evaluating best model across ROI and Full-CXR usage modes...

&#x20; Found CXR directory: C:\\numan\\input\\cxr

&#x20; CXR dataset loaded: 704 samples



&#x20; Running evaluation suite...

&#x20; Tuned PTB threshold on ROI-val (mode=constrained): 0.520

&#x20; Thresholds: balanced=0.480, constrained=0.520 (sens floor=0.65)

&#x20; Temperature calibration: T=0.80



&#x20; Case: ROI validation (ID)

&#x20;   Accuracy: 0.664

&#x20;   Precision: 0.638

&#x20;   Sensitivity: 0.722

&#x20;   Specificity: 0.608

&#x20;   F1-Score: 0.678

&#x20;   AUC: 0.704

&#x20;   PTB prob mean/median: 0.515 / 0.535

&#x20;   % above threshold: 55.3%

&#x20;   Mean uncertainty: 0.931



&#x20; Case: Full CXR raw

&#x20;   Accuracy: 0.490

&#x20;   Precision: 0.490

&#x20;   Sensitivity: 1.000

&#x20;   Specificity: 0.000

&#x20;   F1-Score: 0.658

&#x20;   AUC: 0.635

&#x20;   PTB prob mean/median: 0.648 / 0.650

&#x20;   % above threshold: 100.0%

&#x20;   Mean uncertainty: 0.819



&#x20; Case: Full CXR with ROI extraction

&#x20;   Accuracy: 0.570

&#x20;   Precision: 0.550

&#x20;   Sensitivity: 0.672

&#x20;   Specificity: 0.471

&#x20;   F1-Score: 0.605

&#x20;   AUC: 0.644

&#x20;   PTB prob mean/median: 0.648 / 0.648

&#x20;   % above threshold: 100.0%

&#x20;   Mean uncertainty: 0.820

&#x20;   Conservative overrides: 282



&#x20; Case: Full CXR dual-view

&#x20;   Accuracy: 0.595

&#x20;   Precision: 0.571

&#x20;   Sensitivity: 0.699

&#x20;   Specificity: 0.496

&#x20;   F1-Score: 0.628

&#x20;   AUC: 0.679

&#x20;   PTB prob mean/median: 0.648 / 0.649

&#x20;   % above threshold: 100.0%

&#x20;   Mean uncertainty: 0.820

&#x20;   Conservative overrides: 282



&#x20; Degradation vs ROI-val for Full CXR raw:

&#x20;   Accuracy: 26.18%

&#x20;   Precision: 23.24%

&#x20;   Sensitivity: -38.55%

&#x20;   Specificity: 100.00%

&#x20;   F1-Score: 2.92%

&#x20;   AUC: 9.73%

&#x20;   Mean Degradation: 20.59%



&#x20; Degradation vs ROI-val for Full CXR with ROI extraction:

&#x20;   Accuracy: 14.19%

&#x20;   Precision: 13.89%

&#x20;   Sensitivity: 6.83%

&#x20;   Specificity: 22.62%

&#x20;   F1-Score: 10.71%

&#x20;   AUC: 8.44%

&#x20;   Mean Degradation: 12.78%



&#x20; Degradation vs ROI-val for Full CXR dual-view:

&#x20;   Accuracy: 10.34%

&#x20;   Precision: 10.55%

&#x20;   Sensitivity: 3.21%

&#x20;   Specificity: 18.50%

&#x20;   F1-Score: 7.25%

&#x20;   AUC: 3.52%

&#x20;   Mean Degradation: 8.90%





**training failed on mixed**





Cross-Validation Results (Mean ± Std across 3 folds):

&#x20; Accuracy: 0.510 ± 0.001

&#x20; Precision: 0.000 ± 0.000

&#x20; Sensitivity: 0.000 ± 0.000

&#x20; Specificity: 1.000 ± 0.000

&#x20; F1-Score: 0.000 ± 0.000

&#x20; AUC: 0.625 ± 0.057

&#x20; IoU: 0.218 ± 0.035

&#x20; Dice: 0.330 ± 0.042

&#x20; Tversky: 0.345 ± 0.043



Best model: fold\_1 (Accuracy: 0.511)



=== SAVING MODEL AND RESULTS ===

Saving to: models\\final\\final\_model\_none\_kfold\_improved.mat

Model and results saved successfully!



=== GENERATING TRAINING CURVES ===

&#x20; Saved: models\\final\\kfold\_training\_curves.png



=== MULTI-CASE EVALUATION ===

Evaluating best model across ROI and Full-CXR usage modes...

&#x20; Found CXR directory: C:\\numan\\input\\cxr

&#x20; CXR dataset loaded: 704 samples



&#x20; Running evaluation suite...

&#x20; Tuned PTB threshold on ROI-val (mode=constrained): 0.200

&#x20; Thresholds: balanced=0.200, constrained=0.200 (sens floor=0.65)

&#x20; Temperature calibration: T=3.00



&#x20; Case: ROI validation (ID)

&#x20;   Accuracy: 0.489

&#x20;   Precision: 0.489

&#x20;   Sensitivity: 1.000

&#x20;   Specificity: 0.000

&#x20;   F1-Score: 0.657

&#x20;   AUC: 0.560

&#x20;   PTB prob mean/median: 0.484 / 0.484

&#x20;   % above threshold: 100.0%

&#x20;   Mean uncertainty: 0.984



&#x20; Case: Full CXR raw

&#x20;   Accuracy: 0.490

&#x20;   Precision: 0.490

&#x20;   Sensitivity: 1.000

&#x20;   Specificity: 0.000

&#x20;   F1-Score: 0.658

&#x20;   AUC: 0.422

&#x20;   PTB prob mean/median: 0.486 / 0.486

&#x20;   % above threshold: 100.0%

&#x20;   Mean uncertainty: 0.986



&#x20; Case: Full CXR with ROI extraction

&#x20;   Accuracy: 0.490

&#x20;   Precision: 0.490

&#x20;   Sensitivity: 1.000

&#x20;   Specificity: 0.000

&#x20;   F1-Score: 0.658

&#x20;   AUC: 0.450

&#x20;   PTB prob mean/median: 0.486 / 0.486

&#x20;   % above threshold: 100.0%

&#x20;   Mean uncertainty: 0.986



&#x20; Case: Full CXR dual-view

&#x20;   Accuracy: 0.490

&#x20;   Precision: 0.490

&#x20;   Sensitivity: 1.000

&#x20;   Specificity: 0.000

&#x20;   F1-Score: 0.658

&#x20;   AUC: 0.420

&#x20;   PTB prob mean/median: 0.486 / 0.486

&#x20;   % above threshold: 100.0%

&#x20;   Mean uncertainty: 0.986



&#x20; Degradation vs ROI-val for Full CXR raw:

&#x20;   Accuracy: -0.14%

&#x20;   Precision: -0.14%

&#x20;   Sensitivity: 0.00%

&#x20;   Specificity: 0.00%

&#x20;   F1-Score: -0.10%

&#x20;   AUC: 24.64%

&#x20;   Mean Degradation: 4.04%



&#x20; Degradation vs ROI-val for Full CXR with ROI extraction:

&#x20;   Accuracy: -0.14%

&#x20;   Precision: -0.14%

&#x20;   Sensitivity: 0.00%

&#x20;   Specificity: 0.00%

&#x20;   F1-Score: -0.10%

&#x20;   AUC: 19.70%

&#x20;   Mean Degradation: 3.22%



&#x20; Degradation vs ROI-val for Full CXR dual-view:

&#x20;   Accuracy: -0.14%

&#x20;   Precision: -0.14%

&#x20;   Sensitivity: 0.00%

&#x20;   Specificity: 0.00%

&#x20;   F1-Score: -0.10%

&#x20;   AUC: 24.99%

&#x20;   Mean Degradation: 4.10%









**updated following settings to** 



config.teacher\_model\_spec = 'mixed';



loss\_config.use\_focal = false;

loss\_config.use\_class\_weights = false;

loss\_config.use\_gradcam = false;

loss\_config.use\_tversky = false;

loss\_config.use\_anatomical\_guidance = false;



config.preprocessing\_method = 'none';

config.patience = 5;











