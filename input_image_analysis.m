function input_image_analysis()
root = fullfile(pwd,'input');
datasets = {'roi','cxr'};
for di=1:numel(datasets)
    ds = datasets{di};
    fprintf('DATASET %s\n', ds);
    classes = {'normal','ptb'};
    totalCount = 0;
    allSizes = containers.Map;
    means = [];
    stds = [];
    mins = [];
    maxs = [];
    meanPaths = {};
    for ci=1:numel(classes)
        classPath = fullfile(root, ds, classes{ci});
        files = dir(fullfile(classPath, '*.png'));
        fprintf('  class %s: %d images\n', classes{ci}, numel(files));
        totalCount = totalCount + numel(files);
        for fi=1:numel(files)
            if mod(fi,100)==0
                fprintf('    %d/%d\n', fi, numel(files));
            end
            img = imread(fullfile(files(fi).folder, files(fi).name));
            if size(img,3) > 1
                img = rgb2gray(img);
            end
            img = double(img);
            mins(end+1) = min(img(:));
            maxs(end+1) = max(img(:));
            means(end+1) = mean(img(:));
            stds(end+1) = std(img(:));
            sz = sprintf('%dx%d', size(img,1), size(img,2));
            if isKey(allSizes, sz)
                allSizes(sz) = allSizes(sz) + 1;
            else
                allSizes(sz) = 1;
            end
            meanPaths{end+1} = fullfile(files(fi).folder, files(fi).name);
        end
    end
    fprintf('  total %s images: %d\n', ds, totalCount);
    fprintf('  size counts:\n');
    keys = allSizes.keys;
    for ki=1:numel(keys)
        fprintf('    %s: %d\n', keys{ki}, allSizes(keys{ki}));
    end
    fprintf('  intensity mean across images: mean=%.3f std=%.3f median=%.3f min=%.3f max=%.3f\n', mean(means), std(means), median(means), min(means), max(means));
    fprintf('  intensity std across images: mean=%.3f std=%.3f median=%.3f min=%.3f max=%.3f\n', mean(stds), std(stds), median(stds), min(stds), max(stds));
    fprintf('  intensity min across images: mean=%.3f std=%.3f median=%.3f\n', mean(mins), std(mins), median(mins));
    fprintf('  intensity max across images: mean=%.3f std=%.3f median=%.3f\n', mean(maxs), std(maxs), median(maxs));
    if di == 1
        roiFiles = meanPaths; roiMeans = means; roiMins = mins; roiMaxs = maxs;
    else
        cxrFiles = meanPaths; cxrMeans = means; cxrMins = mins; cxrMaxs = maxs;
    end
end

% Mask stats
fprintf('DATASET masks\n');
maskPath = fullfile(root, 'masks');
maskSizes = containers.Map;
maskAreas = [];
maskImgSizes = [];
classes = {'normal','ptb'};
for ci=1:numel(classes)
    files = dir(fullfile(maskPath, classes{ci}, '*.png'));
    fprintf('  class %s: %d masks\n', classes{ci}, numel(files));
    for fi=1:numel(files)
        img = imread(fullfile(files(fi).folder, files(fi).name));
        if size(img,3) > 1
            img = rgb2gray(img);
        end
        if ~islogical(img)
            img = img > 0;
        end
        maskAreas(end+1) = sum(img(:));
        maskImgSizes(end+1) = numel(img);
        sz = sprintf('%dx%d', size(img,1), size(img,2));
        if isKey(maskSizes, sz)
            maskSizes(sz) = maskSizes(sz) + 1;
        else
            maskSizes(sz) = 1;
        end
    end
end
fprintf('  total masks: %d\n', numel(maskAreas));
fprintf('  mask size counts:\n');
keys = maskSizes.keys;
for ki=1:numel(keys)
    fprintf('    %s: %d\n', keys{ki}, maskSizes(keys{ki}));
end
ratios = maskAreas ./ maskImgSizes;
fprintf('  mask area ratio: mean=%.4f std=%.4f median=%.4f min=%.4f max=%.4f\n', mean(ratios), std(ratios), median(ratios), min(ratios), max(ratios));

% Paired ROI/CXR shape and intensity comparison
fprintf('PAIRWISE ROI/CXR comparison\n');
roiMap = containers.Map('KeyType','char','ValueType','char');
for i=1:numel(roiFiles)
    [~,n,~] = fileparts(roiFiles{i});
    roiMap(n) = roiFiles{i};
end
common = 0;
ratios = [];
absDiffMeans = [];
for i=1:numel(cxrFiles)
    [~,n,~] = fileparts(cxrFiles{i});
    if isKey(roiMap, n)
        common = common + 1;
        roiFile = roiMap(n);
        roiImg = imread(roiFile);
        if size(roiImg,3)>1
            roiImg = rgb2gray(roiImg);
        end
        cxrImg = imread(cxrFiles{i});
        if size(cxrImg,3)>1
            cxrImg = rgb2gray(cxrImg);
        end
        ratios(end+1) = numel(roiImg)/numel(cxrImg);
        absDiffMeans(end+1) = abs(mean(double(roiImg(:))) - mean(double(cxrImg(:))));
    end
end
fprintf('  common filename pairs: %d\n', common);
fprintf('  mean size ratio ROI/CXR=%.4f median=%.4f min=%.4f max=%.4f\n', mean(ratios), median(ratios), min(ratios), max(ratios));
fprintf('  mean abs mean-intensity diff between ROI and CXR=%.3f\n', mean(absDiffMeans));
end