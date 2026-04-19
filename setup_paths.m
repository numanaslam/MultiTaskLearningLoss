function setup_paths()
%SETUP_PATHS Add organized folders to MATLAB path
%   This function adds the src/ and scripts/ folders to MATLAB path
%   for easy access to functions and scripts.
%
%   Usage:
%       setup_paths()
%
%   Run this from the project root directory (where this file is located)
%   or it will automatically find the project root

fprintf('=== SETTING UP MATLAB PATHS ===\n\n');

% Find project root (directory containing this script)
scriptPath = mfilename('fullpath');
[projectRoot, ~, ~] = fileparts(scriptPath);

% Change to project root if not already there
if ~strcmp(pwd, projectRoot)
    cd(projectRoot);
    fprintf('Changed to project root: %s\n\n', projectRoot);
else
    fprintf('Project root: %s\n\n', projectRoot);
end

% Folders to add to path
foldersToAdd = {
    'src',
    'src/training',
    'src/evaluation',
    'src/utils',          % Important: includes precompute_gradcam_and_masks
    'src/visualization',
    'src/loss_functions',
    'scripts',
    'scripts/data_preprocessing',
    'scripts/analysis'
};

addedCount = 0;
skippedCount = 0;

for i = 1:length(foldersToAdd)
    folderPath = fullfile(projectRoot, foldersToAdd{i});
    
    if exist(folderPath, 'dir')
        % Check if already in path
        if isempty(strfind(path, folderPath))
            addpath(folderPath);
            fprintf('  Added to path: %s\n', foldersToAdd{i});
            addedCount = addedCount + 1;
        else
            fprintf('  Already in path: %s\n', foldersToAdd{i});
            skippedCount = skippedCount + 1;
        end
    else
        fprintf('  Warning: Folder not found: %s\n', foldersToAdd{i});
    end
end

fprintf('\n=== PATH SETUP COMPLETE ===\n');
fprintf('Added: %d folders\n', addedCount);
fprintf('Skipped: %d folders (already in path)\n', skippedCount);
fprintf('\nNote: These paths are temporary for this session.\n');
fprintf('To make permanent, add to MATLAB startup.m or use savepath\n');

end

