function [loss, grads, state, loss_components] = compute_loss_with_config_improved(...
    net, X, T, loss_config, classWeights, trainFiles, preCAMs, preMasks, ...
    nCam, classes, featureLayer, useGPU, epoch, preprocessing_method, refHist) %#ok<INUSD>
%COMPUTE_LOSS_WITH_CONFIG_IMPROVED  Multi-task PTB distillation loss (PATCHED).
%
%   PATCH v2 (2026-05-21) — GRADIENT-FLOW FIX
%   -----------------------------------------------------------------------
%   Previous version called extractdata() on student-CAM tensors before
%   feeding them into the auxiliary losses. That detached them from the
%   autograd graph, making lambda_cam / lambda_tversky / lambda_anatomical
%   cosmetic: dlgradient(loss, net.Learnables) only flowed through clsLoss.
%   Symptom: OOD evaluation collapses (sens=0 or spec=0) because the model
%   trained as pure CE on ROI crops.
%
%   This version:
%     * Computes student CAMs via student_cam_differentiable, which keeps
%       the dlarray tracked end to end (inner dlgradient with
%       'EnableHigherDerivatives' = true).
%     * Removes every extractdata on studCAM in the loss path.
%     * Uses the GROUND-TRUTH class score for the CAM (not argmax) so
%       attention transfer is honest on misclassified samples.
%     * Initialises aux-loss accumulators as dlarray scalars so the
%       running sums stay in the graph.
%
%   DROP-IN INSTRUCTIONS:
%     Replace lines 984..1241 of final_working_mixed.m with this whole
%     function body. Also ensure these helpers are on the MATLAB path:
%       student_cam_differentiable.m
%       label_index_from_path.m
%     The existing helpers cam_cosine_loss, tversky_coefficient_dlarray,
%     dice_coefficient_dlarray, iou_coefficient_dlarray, compute_focal_loss,
%     reshape_network_scores and ensure_class_probabilities are reused
%     unchanged.

    % ------------------------------------------------------------------ %
    % 0. Initialise component log (doubles, just for printf-level tracking)
    % ------------------------------------------------------------------ %
    loss_components = struct( ...
        'classification', 0, ...
        'gradcam',        0, ...
        'segmentation',   0, ...
        'tversky',        0, ...
        'iou',            0, ...
        'anatomical',     0);

    % ------------------------------------------------------------------ %
    % 1. Forward pass for classification
    % ------------------------------------------------------------------ %
    [Y, state] = forward(net, X);
    Y = reshape_network_scores(Y, size(T, 1));
    Y = ensure_class_probabilities(Y);
    if ~istable(state), state = []; end

    % ------------------------------------------------------------------ %
    % 2. Classification loss (CE or focal)
    % ------------------------------------------------------------------ %
    if loss_config.use_focal
        focalWeights = ones(size(classWeights), 'like', classWeights);
        if isfield(loss_config, 'use_class_weights') && loss_config.use_class_weights
            focalWeights = classWeights;
        end
        clsLoss = compute_focal_loss(Y, T, focalWeights, ...
            loss_config.focal_alpha, loss_config.focal_gamma);
    else
        if isfield(loss_config, 'use_class_weights') && loss_config.use_class_weights
            clsLoss = crossentropy(Y, T, 'Weights', classWeights, 'WeightsFormat', 'C');
        else
            clsLoss = crossentropy(Y, T);
        end
    end
    loss_components.classification = clsLoss;

    % ------------------------------------------------------------------ %
    % 3. Initialise auxiliary-loss accumulators as TRACKED dlarray scalars.
    %    Critical: if we used plain doubles here, the first '+= dlarray'
    %    upcast would work but later 'camLoss / n' on a non-active path
    %    could break. dlarray(0) up front avoids any ambiguity.
    % ------------------------------------------------------------------ %
    if useGPU
        zeroDl = gpuArray(dlarray(single(0)));
    else
        zeroDl = dlarray(single(0));
    end
    camLoss        = zeroDl;
    segLoss        = zeroDl;
    tverskyLoss    = zeroDl;
    iouLoss        = zeroDl;
    anatomicalLoss = zeroDl;

    hasAuxLoss = any([loss_config.use_gradcam, loss_config.use_segmentation, ...
                     loss_config.use_tversky,  loss_config.use_iou, ...
                     loss_config.use_anatomical_guidance]);

    % ------------------------------------------------------------------ %
    % 4. Auxiliary-loss pass (per-sample, with tracked student CAMs)
    % ------------------------------------------------------------------ %
    nActive = 0;
    if hasAuxLoss
        N = numel(trainFiles);
        n = min(nCam, N);
        if N > 0 && n > 0
            idxs = randperm(N, n);
        else
            idxs = [];
        end

        for ii = 1:numel(idxs)
            idx = idxs(ii);

            % -- 4a. Load image, mirror the training preprocessing path -----
            try
                img = imread(trainFiles{idx});
            catch
                continue;
            end
            if size(img, 3) == 1, img = repmat(img, [1 1 3]); end

            img4d = reshape(img, size(img,1), size(img,2), size(img,3), 1);
            img4d = apply_preprocessing_batch(img4d, preprocessing_method, refHist);
            img_r = imresize(img4d(:,:,:,1), [224 224]);

            if useGPU
                img_dl = gpuArray(dlarray(single(img_r), 'SSCB'));
            else
                img_dl = dlarray(single(img_r), 'SSCB');
            end

            % -- 4b. Teacher CAM (target). Leaf dlarray; no gradient needed.
            targetCAM = preCAMs{idx};
            if isa(targetCAM, 'dlarray'), targetCAM = extractdata(targetCAM); end
            targetCAM = squeeze(single(targetCAM));
            if size(targetCAM,1) ~= 224 || size(targetCAM,2) ~= 224
                targetCAM = imresize(targetCAM, [224 224]);
            end
            t_max = max(targetCAM, [], 'all');
            if t_max > 0, targetCAM = targetCAM / t_max; end
            if useGPU
                targetCAM_dl = gpuArray(dlarray(targetCAM, 'SS'));
            else
                targetCAM_dl = dlarray(targetCAM, 'SS');
            end

            % -- 4c. Lung mask (leaf dlarray, no gradient) -------------------
            realMask = preMasks{idx};
            haveMask = false;
            mask_dl  = [];
            if ~isempty(realMask)
                if isa(realMask, 'dlarray'), realMask = extractdata(realMask); end
                realMask = squeeze(single(realMask > 0.5));
                if size(realMask,1) ~= 224 || size(realMask,2) ~= 224
                    realMask = imresize(realMask, [224 224], 'nearest');
                end
                if any(realMask(:))
                    if useGPU
                        mask_dl = gpuArray(dlarray(realMask, 'SS'));
                    else
                        mask_dl = dlarray(realMask, 'SS');
                    end
                    haveMask = true;
                end
            end

            % -- 4d. Ground-truth class index for the CAM target score ------
            gtIdx = label_index_from_path(trainFiles{idx}, classes);

            % -- 4e. DIFFERENTIABLE STUDENT CAM (this is the fix) ----------
            % studCAM stays as a tracked dlarray, 224x224, in [0,1].
            studCAM = student_cam_differentiable(net, img_dl, featureLayer, gtIdx);
            nActive = nActive + 1;

            % -- 4f. GradCAM loss (cosine or MSE) ---------------------------
            if loss_config.use_gradcam
                if isfield(loss_config,'cam_loss_type') && ...
                        strcmp(loss_config.cam_loss_type, 'cosine')
                    camLoss = camLoss + cam_cosine_loss(studCAM, targetCAM_dl);
                else
                    camLoss = camLoss + mse(studCAM, targetCAM_dl);
                end
            end

            % -- 4g. Segmentation / Tversky / IoU losses on lung mask ------
            if haveMask
                if loss_config.use_segmentation
                    segLoss = segLoss + ...
                        (1 - dice_coefficient_dlarray(studCAM, mask_dl));
                end
                if loss_config.use_tversky
                    tCoef = tversky_coefficient_dlarray(studCAM, mask_dl, ...
                        loss_config.tversky_alpha, loss_config.tversky_beta);
                    tverskyLoss = tverskyLoss + (1 - tCoef);
                end
                if loss_config.use_iou
                    iCoef = iou_coefficient_dlarray(studCAM, mask_dl);
                    iouLoss = iouLoss + (1 - iCoef);
                end
            end

            % -- 4h. Anatomical guidance (penalise CAM outside lung) -------
            if haveMask && loss_config.use_anatomical_guidance
                nonLungMask     = 1 - mask_dl;
                penalty_outside = mean(studCAM .* nonLungMask, 'all');
                reward_inside   = mean(studCAM .* mask_dl,     'all');
                anat_sample = penalty_outside - ...
                    loss_config.anatomical_reward_weight * reward_inside;
                if isfield(loss_config,'anatomical_positivity') && ...
                        loss_config.anatomical_positivity
                    anat_sample = max(anat_sample, 0);
                end
                anatomicalLoss = anatomicalLoss + anat_sample;
            end
        end

        % Average across active samples (keeps dlarray graph alive).
        if nActive > 0
            camLoss        = camLoss        / nActive;
            segLoss        = segLoss        / nActive;
            tverskyLoss    = tverskyLoss    / nActive;
            iouLoss        = iouLoss        / nActive;
            anatomicalLoss = anatomicalLoss / nActive;
        end
    end

    % ------------------------------------------------------------------ %
    % 5. Record components for logging
    % ------------------------------------------------------------------ %
    loss_components.gradcam      = camLoss;
    loss_components.segmentation = segLoss;
    loss_components.tversky      = tverskyLoss;
    loss_components.iou          = iouLoss;
    loss_components.anatomical   = anatomicalLoss;

    % ------------------------------------------------------------------ %
    % 6. Total weighted loss
    % ------------------------------------------------------------------ %
    loss = clsLoss;
    if loss_config.use_gradcam
        loss = loss + loss_config.lambda_cam * camLoss;
    end
    if loss_config.use_segmentation && isfield(loss_config, 'lambda_seg')
        loss = loss + loss_config.lambda_seg * segLoss;
    end
    if loss_config.use_tversky
        loss = loss + loss_config.lambda_tversky * tverskyLoss;
    end
    if loss_config.use_iou && isfield(loss_config, 'lambda_seg')
        loss = loss + loss_config.lambda_seg * iouLoss;
    end
    if loss_config.use_anatomical_guidance
        loss = loss + loss_config.lambda_anatomical * anatomicalLoss;
    end

    % ------------------------------------------------------------------ %
    % 7. Outer gradient w.r.t. net.Learnables.
    %    EnableHigherDerivatives = true is REQUIRED because the inner
    %    dlgradient inside student_cam_differentiable produces a
    %    second-order computational graph.
    % ------------------------------------------------------------------ %
    grads = dlgradient(loss, net.Learnables, 'EnableHigherDerivatives', true);
end
