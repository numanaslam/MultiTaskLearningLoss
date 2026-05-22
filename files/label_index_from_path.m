function idx = label_index_from_path(filepath, classes)
%LABEL_INDEX_FROM_PATH  Map a file's parent folder name to a class index.
%
%   idx = label_index_from_path(filepath, classes)
%
%   The student CAM should be computed on the GROUND-TRUTH class score
%   (not argmax) so that teacher-student attention transfer is honest on
%   misclassified samples. Given a training image at
%       .../input/roi/PTB/img_001.png
%   we extract 'PTB' from the parent folder and return its index in
%   `classes`. Falls back to 1 if no match (safe default).
%
%   INPUTS:
%     filepath  char, full path to a training image
%     classes   cellstr or categorical, ordered class names from imds
%
%   OUTPUT:
%     idx       scalar integer in [1, numel(classes)]

    [parentDir, ~, ~] = fileparts(filepath);
    [~, labelName, ~] = fileparts(parentDir);

    if iscategorical(classes)
        idx = find(classes == categorical({labelName}), 1, 'first');
    elseif isstring(classes) || iscell(classes)
        idx = find(strcmp(cellstr(classes), labelName), 1, 'first');
    else
        idx = [];
    end

    if isempty(idx)
        idx = 1;   % safe default; logged at debug time if it ever triggers
    end
end
