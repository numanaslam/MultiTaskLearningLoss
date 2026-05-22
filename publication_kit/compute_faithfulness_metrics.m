function metrics = compute_faithfulness_metrics(dlnet, teacherNet, files, labels, masks, targetLayer, studentFCName, classNames)
%COMPUTE_FAITHFULNESS_METRICS  Per-image CAM-quality metrics for a model.
%
%   metrics = compute_faithfulness_metrics(dlnet, teacherNet, files, labels, ...
%                                          masks, targetLayer, studentFCName, classNames)
%
%   Computes three CAM-quality metrics for every (file, label, mask) triple:
%
%     energy_ratio  = sum(CAM .* mask) / sum(CAM)
%                     Fraction of student-CAM mass inside the lung mask.
%                     1.0 = perfectly anatomical; 0.0 = attention is all
%                     outside the lung.
%
%     pointing_game = 1 if argmax(CAM) is inside the lung mask, else 0.
%                     Standard interpretability metric.
%
%     ts_cosine     = cos(student_CAM, teacher_CAM)
%                     What the alignment loss is supposed to maximise.
%                     Useful diagnostic even when alignment loss is off.
%
%   The CAM is taken on the GROUND-TRUTH class (not argmax), so the metrics
%   reflect what the model *would* attend to if it were correct.
%
%   OUTPUT struct fields:
%     energy_ratio   Nx1 vector
%     pointing_game  Nx1 logical
%     ts_cosine      Nx1 vector
%     files          cellstr of N input paths (echoed for traceability)
%     labels         Nx1 numeric (echoed)

    N = numel(files);
    metrics = struct();
    metrics.energy_ratio  = nan(N, 1);
    metrics.pointing_game = false(N, 1);
    metrics.ts_cosine     = nan(N, 1);
    metrics.files  = files;
    metrics.labels = labels;

    if ~isa(dlnet, 'dlnetwork'),    dlnet = dlnetwork(layerGraph(dlnet));    end
    if ~isa(teacherNet, 'dlnetwork'), teacherNet = dlnetwork(layerGraph(teacherNet)); end

    teacherFCName = findLastFC(teacherNet);
    useGPU = hasGPU(dlnet);

    fprintf('    Computing faithfulness over %d samples', N);
    for i = 1:N
        if mod(i, 25) == 0, fprintf('.'); end

        img = readImage224(files{i});
        mask = readMaskSafe(masks{i});
        if isempty(mask), continue; end

        classIdx = labels(i);
        if ~isfinite(classIdx) || classIdx < 1, continue; end

        % Student CAM
        sCAM = cam224(dlnet, img, classIdx, targetLayer, studentFCName, useGPU);
        % Teacher CAM
        tCAM = cam224(teacherNet, img, classIdx, targetLayer, teacherFCName, useGPU);

        % Energy ratio
        camMass = sum(sCAM, 'all') + eps;
        metrics.energy_ratio(i) = sum(sCAM(mask), 'all') / camMass;

        % Pointing game
        [~, maxLin] = max(sCAM(:));
        [pr, pc] = ind2sub(size(sCAM), maxLin);
        metrics.pointing_game(i) = mask(pr, pc);

        % Teacher-student cosine
        sv = sCAM(:); tv = tCAM(:);
        metrics.ts_cosine(i) = (sv' * tv) / ...
            (sqrt(sum(sv.^2)) * sqrt(sum(tv.^2)) + eps);
    end
    fprintf(' done\n');

    fprintf('    energy_ratio  : mean=%.3f  std=%.3f\n', ...
        nanmean(metrics.energy_ratio), nanstd(metrics.energy_ratio));
    fprintf('    pointing_game : %.3f (%d/%d)\n', ...
        mean(metrics.pointing_game), sum(metrics.pointing_game), N);
    fprintf('    ts_cosine     : mean=%.3f  std=%.3f\n', ...
        nanmean(metrics.ts_cosine), nanstd(metrics.ts_cosine));
end


%% =========================== HELPERS ====================================

function img = readImage224(filepath)
    img = imread(filepath);
    if size(img, 3) == 1, img = cat(3, img, img, img);
    elseif size(img, 3) > 3, img = img(:,:,1:3);
    end
    img = imresize(img, [224 224]);
end

function mask = readMaskSafe(maskPath)
    mask = [];
    if isempty(maskPath) || ~exist(maskPath, 'file'), return; end
    m = imread(maskPath);
    if size(m, 3) > 1, m = rgb2gray(m); end
    m = imresize(m, [224 224], 'nearest');
    mask = logical(single(m) / single(max(m(:)) + eps) > 0.5);
end

function fcName = findLastFC(net)
    layers = net.Layers;
    fcIdx = find(arrayfun(@(L) isa(L, 'nnet.cnn.layer.FullyConnectedLayer'), layers), 1, 'last');
    fcName = layers(fcIdx).Name;
end

function tf = hasGPU(net)
    tf = false;
    if isempty(net.Learnables) || isempty(net.Learnables.Value), return; end
    v = net.Learnables.Value{1};
    tf = isa(v, 'gpuArray') || (isa(v, 'dlarray') && isa(extractdata(v), 'gpuArray'));
end

function cam = cam224(net, img, classIdx, targetLayer, fcName, useGPU)
    XData = reshape(single(img), [size(img,1), size(img,2), 3, 1]);
    if useGPU, XData = gpuArray(XData); end
    X = dlarray(XData, 'SSCB');
    camDl = dlfeval(@cam_forward, net, X, classIdx, targetLayer, fcName);
    cam = squeeze(gather(extractdata(camDl)));
end

function cam = cam_forward(net, X, classIdx, targetLayer, fcName)
    [rawLogits, act] = forward(net, X, 'Outputs', {fcName, targetLayer});
    logitsVec = reshape(stripdims(rawLogits), [], 1);
    logits = dlarray(logitsVec, 'CB');
    tgt = zeros(size(logitsVec), 'like', logitsVec); tgt(classIdx) = 1;
    obj = sum(logits .* dlarray(tgt, 'CB'), 1);
    g = dlgradient(obj, act);
    w = mean(g, [1 2]);
    cam = relu(sum(act .* w, 3));
    cam = dlresize(cam, 'OutputSize', [size(X, 1) size(X, 2)]);
    cam = cam ./ (max(cam, [], [1 2]) + 1e-6);
end
