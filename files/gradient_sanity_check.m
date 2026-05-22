function ok = gradient_sanity_check(net, classes, featureLayer, useGPU)
%GRADIENT_SANITY_CHECK  Verify that CAM loss actually back-props to weights.
%
%   ok = gradient_sanity_check(net, classes, featureLayer, useGPU)
%
%   Builds a synthetic image, runs student_cam_differentiable, takes the
%   gradient of a toy CAM-only loss w.r.t. net.Learnables, and prints the
%   gradient norm on the first conv layer. If the norm is > 0, the patch
%   is active and the auxiliary losses can train the network.
%
%   Run this ONCE right after applying the patch, before kicking off a
%   full training run. Expected output:
%
%       CAM-only gradient norm on conv1_1: 3.142e-04
%         PASS - gradient flows from CAM loss to weights.
%
%   If you see "FAIL - gradient is zero", student_cam_differentiable is
%   not on path, the wrong feature layer name was passed, or an
%   extractdata() call still lurks somewhere in the inner path.
%
%   INPUTS:
%     net           dlnetwork student model (e.g. taken from a checkpoint)
%     classes       cellstr or categorical from your imds
%     featureLayer  e.g. 'relu5_3' for VGG16
%     useGPU        true to run on GPU
%
%   OUTPUT:
%     ok            logical, true if gradient norm > 1e-9
%
%   EXAMPLE:
%       ckpt = load('models/checkpoints/checkpoint_none_fold_1_improved.mat');
%       gradient_sanity_check(ckpt.fold_model, ckpt.classes, 'relu5_3', true);

    if nargin < 4, useGPU = canUseGPU; end
    if nargin < 3, featureLayer = 'relu5_3'; end

    if ~isa(net, 'dlnetwork')
        error('gradient_sanity_check:NotDlnetwork', ...
            'Expected a dlnetwork; got %s. Convert with dlnetwork(layerGraph(net)) first.', ...
            class(net));
    end

    fprintf('--- Gradient Sanity Check ---\n');
    fprintf('  Network class : %s\n', class(net));
    fprintf('  Feature layer : %s\n', featureLayer);
    fprintf('  Use GPU       : %d\n', useGPU);
    fprintf('  Num classes   : %d\n', numel(classes));

    % Synthetic input matching VGG16 expectations
    img = single(rand(224, 224, 3, 1)) * 255;
    if useGPU
        img_dl = gpuArray(dlarray(img, 'SSCB'));
    else
        img_dl = dlarray(img, 'SSCB');
    end
    gtIdx = 1;   % arbitrary

    % Compute gradient inside dlfeval
    gradsTbl = dlfeval(@local_probe, net, img_dl, featureLayer, gtIdx);

    % Pull the gradient for conv1_1 weights, or fall back to first learnable
    layerName = 'conv1_1';
    sel = strcmp(gradsTbl.Layer, layerName) & strcmp(gradsTbl.Parameter, 'Weights');
    if any(sel)
        gValue = gradsTbl.Value{find(sel, 1, 'first')};
        whichLayer = layerName;
    else
        warning('gradient_sanity_check:NoConv11', ...
            'Layer %s/Weights not found; using first learnable instead.', layerName);
        gValue = gradsTbl.Value{1};
        whichLayer = sprintf('%s/%s', gradsTbl.Layer(1,:), gradsTbl.Parameter(1,:));
    end

    gNorm = sqrt(sum(extractdata(gValue) .^ 2, 'all'));
    fprintf('  CAM-only gradient norm on %s: %.3e\n', whichLayer, gNorm);

    ok = double(gNorm) > 1e-9;
    if ok
        fprintf('  PASS - gradient flows from CAM loss to weights.\n');
    else
        fprintf('  FAIL - gradient is zero. Patch is not active.\n');
        fprintf('  Checklist:\n');
        fprintf('    [ ] student_cam_differentiable.m is on the MATLAB path\n');
        fprintf('    [ ] You replaced the local function in final_working_mixed.m\n');
        fprintf('    [ ] featureLayer name matches a real conv-relu in net.Layers\n');
        fprintf('    [ ] Net has fc8 (or some fc layer) for the logit probe\n');
    end
end


function grads = local_probe(net, img_dl, featureLayer, gtIdx)
% Toy graph: compute differentiable CAM, sum it, and take dL/dW.
    studCAM = student_cam_differentiable(net, img_dl, featureLayer, gtIdx);
    camLoss = sum(studCAM, 'all');   % scalar dlarray
    grads = dlgradient(camLoss, net.Learnables, 'EnableHigherDerivatives', true);
end
