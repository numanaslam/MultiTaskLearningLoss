function verify_patch(teacherModelSpec)
%VERIFY_PATCH  End-to-end check that the gradient-flow patch is active.
%
%   verify_patch()                  % defaults to teacher_model_spec='roi'
%   verify_patch('mixed')           % use the mixed teacher checkpoint
%   verify_patch('models/foo.mat')  % explicit path
%
%   Steps:
%     1. Resolve and load the teacher backbone (same logic as in
%        final_working_mixed.m / load_data_and_network).
%     2. Convert to dlnetwork if necessary.
%     3. Build the same student head (fc8 + softmax) used in
%        train_with_kfold_preprocessed.
%     4. Run gradient_sanity_check on the student.
%
%   Run from the project root, with this patch directory on the MATLAB
%   path:
%
%       >> addpath('path/to/ptb_distillation_patch');
%       >> verify_patch('roi');

    if nargin < 1 || isempty(teacherModelSpec)
        teacherModelSpec = 'roi';
    end

    fprintf('=== verify_patch (teacher = %s) ===\n', teacherModelSpec);

    % --- Required helpers ---------------------------------------------- %
    required = {'student_cam_differentiable', ...
                'label_index_from_path', ...
                'gradient_sanity_check'};
    for k = 1:numel(required)
        if isempty(which(required{k}))
            error('verify_patch:MissingHelper', ...
                'Cannot find %s on MATLAB path. addpath(''ptb_distillation_patch'') first.', ...
                required{k});
        end
    end

    % --- Resolve teacher model path ------------------------------------ %
    if endsWith(teacherModelSpec, '.mat', 'IgnoreCase', true) && exist(teacherModelSpec, 'file')
        modelPath = teacherModelSpec;
    else
        candidates = {
            sprintf('models/vgg16_%s_teacher.mat', lower(teacherModelSpec));
            sprintf('models/checkpoints/checkpoint_none_fold_1_improved.mat');
            sprintf('vgg16_%s_teacher.mat', lower(teacherModelSpec));
        };
        modelPath = '';
        for k = 1:numel(candidates)
            if exist(candidates{k}, 'file')
                modelPath = candidates{k};
                break;
            end
        end
        if isempty(modelPath)
            error('verify_patch:NoModel', ...
                'Cannot locate a teacher .mat for spec ''%s''. Tried:\n  %s\n  %s\n  %s', ...
                teacherModelSpec, candidates{:});
        end
    end
    fprintf('  Loading: %s\n', modelPath);

    % --- Load and unpack ----------------------------------------------- %
    S = load(modelPath);
    if isfield(S, 'teacherNet'),     net = S.teacherNet;
    elseif isfield(S, 'fold_model'), net = S.fold_model;
    elseif isfield(S, 'net'),        net = S.net;
    elseif isfield(S, 'trainedNet'), net = S.trainedNet;
    else
        error('verify_patch:NoNetField', ...
            'No teacherNet / fold_model / net / trainedNet field in %s.', modelPath);
    end

    if isfield(S, 'classes')
        classes = S.classes;
    else
        classes = {'Normal', 'PTB'};   % project default
    end

    % --- Convert to dlnetwork ------------------------------------------ %
    if ~isa(net, 'dlnetwork')
        fprintf('  Converting %s -> dlnetwork...\n', class(net));
        try
            net = dag2dlnetwork(net);
        catch
            net = dlnetwork(layerGraph(net));
        end
    end

    % --- If this is a teacher (no fc8), graft a fresh head on ---------- %
    layerNames = {net.Layers.Name};
    if ~any(strcmp(layerNames, 'fc8'))
        fprintf('  Network has no fc8; grafting fresh student head.\n');
        baseLg = layerGraph(net);
        toDrop = intersect({'fc8', 'prob', 'output'}, {baseLg.Layers.Name});
        if ~isempty(toDrop)
            baseLg = removeLayers(baseLg, toDrop);
        end
        newHead = [fullyConnectedLayer(numel(classes), 'Name', 'fc8')
                   softmaxLayer('Name', 'prob')];
        baseLg = addLayers(baseLg, newHead);
        % Try to connect from drop7 (VGG16 convention); fall back gracefully
        try
            baseLg = connectLayers(baseLg, 'drop7', 'fc8');
        catch
            tailName = baseLg.Layers(end-2).Name;   % the layer before our head
            baseLg = connectLayers(baseLg, tailName, 'fc8');
        end
        net = dlnetwork(baseLg);
    end

    % --- Run the check -------------------------------------------------- %
    useGPU = canUseGPU;
    ok = gradient_sanity_check(net, classes, 'relu5_3', useGPU);

    if ok
        fprintf('\n=== PATCH ACTIVE — safe to start training ===\n');
    else
        fprintf('\n=== PATCH NOT ACTIVE — fix before training ===\n');
    end
end
