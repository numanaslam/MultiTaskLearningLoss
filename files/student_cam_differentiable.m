function studCAM = student_cam_differentiable(net, img_dl, featureLayer, posClassIdx)
%STUDENT_CAM_DIFFERENTIABLE  Grad-CAM that keeps its autograd graph alive.
%
%   studCAM = student_cam_differentiable(net, img_dl, featureLayer, posClassIdx)
%
%   Returns a 224x224 dlarray Grad-CAM map with the full computational
%   graph back to net.Learnables. Used by compute_loss_with_config_improved
%   so that lambda_cam, lambda_tversky and lambda_anatomical actually
%   contribute gradient (the previous compute_single_student_cam path
%   called extractdata() and broke the graph).
%
%   REQUIREMENTS:
%     * Must be called from a function that is itself executed via dlfeval.
%     * The outer dlgradient call must use 'EnableHigherDerivatives', true
%       because this routine takes an inner dlgradient.
%     * net must be a dlnetwork with an 'fc8' (or 'fc'-prefixed) head and
%       the named feature layer (default 'relu5_3' for VGG16).
%
%   INPUTS:
%     net           dlnetwork (student); has fc8 + softmax 'prob' head
%     img_dl        dlarray('SSCB'), batch=1, on the same device as net
%     featureLayer  char, e.g. 'relu5_3'
%     posClassIdx   scalar integer, target class for the CAM score
%
%   OUTPUT:
%     studCAM       dlarray, 224x224, normalised to [0,1], TRACKED.
%
%   See also: compute_loss_with_config_improved, gradient_sanity_check.

    % ---------------------------------------------------------------- %
    % 1. Resolve the logit layer (raw fc output, NOT softmax).
    %    We probe at fc8 so the gradient is taken w.r.t. pre-softmax
    %    logits, which is what canonical Grad-CAM uses.
    % ---------------------------------------------------------------- %
    layerNames = {net.Layers.Name};
    if any(strcmp(layerNames, 'fc8'))
        logitLayer = 'fc8';
    else
        fcIdx = find(contains(lower(layerNames), 'fc'), 1, 'last');
        if isempty(fcIdx)
            error('student_cam_differentiable:NoLogitLayer', ...
                'Cannot find an fc layer in the network for the CAM score.');
        end
        logitLayer = layerNames{fcIdx};
    end

    % ---------------------------------------------------------------- %
    % 2. Forward pass: features and logits in ONE call.
    % ---------------------------------------------------------------- %
    [featMap, scores] = forward(net, img_dl, ...
        'Outputs', {featureLayer, logitLayer});
    % featMap : 'SSCB', e.g. 14x14x512x1 for VGG16 / 224 input
    % scores  : 'CB',   e.g. 2x1

    % ---------------------------------------------------------------- %
    % 3. Target-class logit (scalar dlarray, tracked).
    % ---------------------------------------------------------------- %
    score = scores(posClassIdx, 1);

    % ---------------------------------------------------------------- %
    % 4. INNER gradient: dScore / dFeatMap.
    %    EnableHigherDerivatives = true so the OUTER dlgradient over
    %    net.Learnables can take a second derivative through this point.
    %    RetainData = true so the activation graph remains usable.
    % ---------------------------------------------------------------- %
    gradFeat = dlgradient(score, featMap, ...
        'EnableHigherDerivatives', true, ...
        'RetainData', true);

    % ---------------------------------------------------------------- %
    % 5. Grad-CAM: GAP over spatial dims -> per-channel weights;
    %    weighted sum across channels -> spatial map; ReLU.
    % ---------------------------------------------------------------- %
    weights = mean(gradFeat, [1 2]);      % 1x1xCx1 dlarray, 'SSCB'
    cam     = sum(featMap .* weights, 3); % HxWx1x1 dlarray, 'SSCB'
    cam     = max(cam, 0);                % ReLU (differentiable)

    % ---------------------------------------------------------------- %
    % 6. Differentiable upsample to 224x224.
    %    Uses dlresize when available (R2022a+); otherwise nearest-
    %    neighbour repeat via kron. DO NOT call imresize on dlarray —
    %    that path is not autograd-safe in older releases.
    % ---------------------------------------------------------------- %
    cam = local_differentiable_upsample(cam, [224 224]);

    % ---------------------------------------------------------------- %
    % 7. Drop the singleton C and B dims while keeping the autograd link.
    %    reshape on a dlarray preserves gradient; format is dropped, which
    %    is fine since downstream losses (cam_cosine_loss, tversky_*) use
    %    reductions that are format-agnostic.
    % ---------------------------------------------------------------- %
    cam = reshape(cam, [224 224]);

    % ---------------------------------------------------------------- %
    % 8. Differentiable [0,1] normalisation.
    %    No extractdata — everything stays tracked.
    % ---------------------------------------------------------------- %
    cmin = min(cam, [], 'all');
    cmax = max(cam, [], 'all');
    studCAM = (cam - cmin) ./ (cmax - cmin + single(1e-6));
end


function out = local_differentiable_upsample(x, outSize)
%LOCAL_DIFFERENTIABLE_UPSAMPLE  dlarray-safe upsample to outSize.
%   Prefers dlresize (R2022a+, bilinear); falls back to nearest-neighbour
%   repeat via kron for older releases.

    persistent useDlresize
    if isempty(useDlresize)
        useDlresize = exist('dlresize', 'file') == 2;
    end

    if useDlresize
        % dlresize expects a formatted dlarray and operates on S dims.
        % If x is 'SSCB' (HxWx1x1) we can pass directly; result keeps fmt.
        out = dlresize(x, 'OutputSize', outSize, 'Method', 'linear');
    else
        % Fallback: nearest-neighbour via kron. Differentiable (kron is
        % linear in its first argument). Works for any positive integer
        % scale factor; pads/crops to the exact outSize.
        Hin = size(x, 1);
        Win = size(x, 2);
        sH = max(1, ceil(outSize(1) / Hin));
        sW = max(1, ceil(outSize(2) / Win));
        rep = kron(x, ones(sH, sW, 'like', x));
        out = rep(1:outSize(1), 1:outSize(2), :, :);
    end
end
