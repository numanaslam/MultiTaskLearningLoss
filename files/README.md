# PTB Distillation Patch — Gradient Flow Fix

**Target file:** `final_working_mixed.m` (3884 lines)
**Patch version:** v2, 2026-05-21

This patch fixes the **single catastrophic bug** that explains the OOD-evaluation collapse (sensitivity or specificity dropping to 0 on full CXR). Four secondary issues are documented at the end of this README as follow-up work.

---

## TL;DR — what was broken

Inside `compute_loss_with_config_improved` (line 984 of your file), the student Grad-CAM was being detached from the autograd graph three separate times via `extractdata()`. The auxiliary losses (`cam_cosine_loss`, `tversky_coefficient_dlarray`, anatomical penalty) then operated on a fresh leaf `dlarray` with no link back to `net.Learnables`. As a result:

> `dlgradient(loss, net.Learnables, 'EnableHigherDerivatives', true)` back-propagated **only through `clsLoss`**. Every line of code involving `lambda_cam`, `lambda_tversky`, `lambda_anatomical` was a no-op on the weights.

The model trained as **pure cross-entropy on ROI crops**. The header comments listing "Cosine similarity for GradCAM loss", "Positivity constraint on anatomical loss", "Adaptive loss scaling" all describe machinery that wasn't actually training the network. The OOD collapse is the inevitable consequence: a CE-only model on tight ROI crops latches onto crop-boundary statistics and brightness shortcuts, neither of which transfer to full CXR.

---

## Files in this patch

| File | Role |
|---|---|
| `student_cam_differentiable.m` | **NEW.** Differentiable Grad-CAM that keeps the dlarray tracked end to end. Uses inner `dlgradient` with `EnableHigherDerivatives = true`. |
| `label_index_from_path.m` | **NEW.** Maps a training image's parent folder to a class index, so the CAM is computed on the ground-truth class score (not argmax). |
| `compute_loss_with_config_improved.m` | **REPLACES** the local function at lines 984–1241 of `final_working_mixed.m`. Removes every `extractdata` on student CAMs. |
| `gradient_sanity_check.m` | **NEW.** Diagnostic. Confirms gradient norm > 0 from a CAM-only loss to conv1_1 weights. |
| `verify_patch.m` | **NEW.** End-to-end runner that loads a teacher checkpoint and runs the sanity check. |
| `README.md` | This file. |

---

## Integration in 5 steps

1. **Copy the patch directory next to your code:**

   ```matlab
   addpath('path/to/ptb_distillation_patch');
   savepath;   % optional, to persist
   ```

2. **Open `final_working_mixed.m` and replace the body of `compute_loss_with_config_improved`** (the local function starting at line 984, ending at line 1241 with `end` after the `dlgradient` call) with the body of the patch's `compute_loss_with_config_improved.m`. Keep the rest of the file untouched.

   Leave `compute_single_student_cam`, `student_cam_one_dlarray`, `cam_cosine_loss`, `tversky_coefficient_dlarray`, `dice_coefficient_dlarray`, `iou_coefficient_dlarray` exactly where they are — the patched loss doesn't call the old broken helpers, but `precompute_gradcam_and_masks` and the OOD eval path still use `student_cam_one_dlarray` legitimately (they don't need gradients, just CAM visualisation).

3. **Run the sanity check before retraining:**

   ```matlab
   verify_patch('roi');     % or 'mixed' depending on your teacher
   ```

   Expected output (numbers will differ):

   ```
   --- Gradient Sanity Check ---
     Network class : dlnetwork
     Feature layer : relu5_3
     Use GPU       : 1
     Num classes   : 2
     CAM-only gradient norm on conv1_1: 3.142e-04
     PASS - gradient flows from CAM loss to weights.

   === PATCH ACTIVE — safe to start training ===
   ```

   If you see `FAIL - gradient is zero`, the patch is not active — re-check step 2 (you most likely missed removing one of the `extractdata` calls on `studCAM`).

4. **Adjust loss weights for the new regime.** Your current `lambda_cam = 0.01` and `lambda_tversky = 0.05` were empirically chosen back when the losses were no-ops. With real gradients those values are too weak. Recommended starting point (in `train_final_model_with_preprocessing`, around lines 82–83 and 141–143):

   ```matlab
   config.lambda_cam     = 0.30;   % was 0.01
   config.lambda_tversky = 0.30;   % was 0.05
   loss_config.lambda_cam     = 0.30;
   loss_config.lambda_tversky = 0.30;
   ```

5. **Loosen the curriculum so the aux loss is actually applied.** In `train_model_with_preprocessing_improved` (around lines 74–76):

   ```matlab
   config.aux_start_epoch     = 0;            % was 8 — start aux from epoch 1
   config.aux_loss_interval   = 1;            % was 3 — every batch
   config.aux_samples_per_step = config.batchSize;   % was 6 of 14 — every sample
   ```

   With the broken loss, sub-sampling 5 % of updates was a cost-saving move. With real gradients, you want every batch to carry the distillation signal.

---

## What changed inside the loss function

The patched function differs from the original in exactly these ways:

* **Student CAM is produced by `student_cam_differentiable`**, which keeps the dlarray tracked. The old call chain `compute_single_student_cam → student_cam_one_dlarray → extractdata → fresh dlarray('SS')` is gone.
* **No `extractdata` on `studCAM`** anywhere in the loss path. Normalisation is done via `min/max` on the dlarray.
* **CAM score is taken on the GROUND-TRUTH class**, not `argmax(logits)`. Distilling teacher attention onto misclassified samples with argmax injects noise; gt-based scoring is what attention transfer literally means.
* **Aux-loss accumulators are dlarray scalars from the start**, so the running sum stays in the graph regardless of which branch is active.
* **Outer `dlgradient` still uses `EnableHigherDerivatives = true`** — this was already in the original code (line 1240) but was unnecessary then because the inner dlgradient produced no graph. Now it's necessary.

---

## What this patch does NOT fix (follow-up work)

These are the four secondary issues from the original diagnosis. Each is documented for later; none block deploying step 1.

1. **Student trains on ROI only.** `load_data_and_network` at line 2137 always sets `roiDir = 'input/roi'` even when the teacher is mixed/CXR. The student never sees the field of view it will be tested on. Recommended fix: when `teacher_model_spec ∈ {cxr, mixed}`, point the student `imds` at `input/cxr`. After step 1, this is the next-largest lever.
2. **Threshold/temperature tuned on ROI is applied to CXR.** `tune_binary_calibration_on_id_set` (line 1344) sweeps thresholds on ROI-val. Recommended fix: hold out 50–100 labeled CXR images for domain-specific threshold and temperature tuning, plus enforce a `spec_floor` in `tune_binary_thresholds_from_probs` alongside the existing `sensFloor`.
3. **`extract_lung_roi_simple` (line 3493) is unreliable.** Otsu + `regionprops` on full CXR routinely picks the film border or one lobe. Replace with the lung masks you already load from `maskDir`, bounding-box-cropped with small dilation.
4. **Aux-loss curriculum was sized for the broken regime.** Already covered in step 5 above.

---

## Verification ladder

Run these in order after applying the patch. First failure → stop and debug.

1. **Static:** `gradient_sanity_check` prints norm > 0 on conv1_1.
2. **Component balance:** in the first epoch of a real training run, log `clsLoss`, `lambda_cam * camLoss`, `lambda_tversky * tverskyLoss` for the first 50 batches. They should be within about an order of magnitude of each other once `nActive > 0`. If `clsLoss = 2.1` and `lambda_cam * camLoss = 0.0003`, bump `lambda_cam`.
3. **Teacher-student cosine over epochs:** the cosine similarity between student CAM and teacher CAM on the same image, evaluated at the end of each epoch, should rise monotonically (or near-monotonically) over the first 10 epochs. If it doesn't, the aux loss is producing the right scalar but the wrong gradient direction — usually means the feature layer is wrong.
4. **OOD baseline with no calibration:** evaluate the final model on CXR with `T = 1.0` and `decisionThreshold = 0.5`. AUC is the threshold-independent metric. If AUC ≥ 0.75 you have a calibration problem (fix #2 above). If AUC ≈ 0.5, the representation didn't transfer — that's the cue to also fix #1 above (train the student on full CXR).

---

## Expected impact

* **Step 1 alone (this patch):** auxiliary losses become real. ROI-val AUC may drop by 1–2 points (the CE-only model was overfit). CXR AUC moves from ~0.5 toward ~0.65. Sens/spec collapse may still occur because of #2 (calibration).
* **Step 1 + #1 (CXR student):** CXR AUC clears ~0.75–0.80 if the teacher is competent. Collapse lessens.
* **Step 1 + #1 + #2 (CXR calibration set):** sens and spec both > 0 at the operating point and within ~10 % of each other.

The remaining secondary fixes (#3 and step 5) convert "works" into "works reproducibly".
