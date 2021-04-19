function pathsets = getPathsets(disp_path, gt_path, indi_path, gt_in_subfolder, method_names, key_word_disp, key_word_conf)

imgDataDir  = dir(disp_path);
imgDir_gt = dir(gt_path);

pathsets = {};

for i = 1:length(imgDataDir)
    if(isequal(imgDataDir(i).name,'.')||... % Remove the two hidden folders that come with the system
       isequal(imgDataDir(i).name,'..')||...
       ~imgDataDir(i).isdir)                % Remove traversal files that are not folders
           continue;
    end
    
    curr_dir_path = strcat(disp_path, imgDataDir(i).name, '/');
    disp_map_path = strcat(curr_dir_path, key_word_disp, '.png');
    conf_map_paths = {};
    
    % Check if the current directory contains a disparity map and one
    % confidence map for each of the specified method names
    valid_dir = isfile(disp_map_path);
    
    for method = method_names      
        curr_conf_map_path = strcat(curr_dir_path, key_word_conf, '_', method, '.pfm');
        valid_dir = valid_dir & isfile(curr_conf_map_path);
        conf_map_paths{end+1} = curr_conf_map_path;
    end
    
    % Get GT path
    gt_path_ = strcat(gt_path, imgDataDir(i).name, '.png');    
    if (gt_in_subfolder)
        gt_path_ = strcat(gt_path_, '/disp0GT.png');
    end
    valid_dir = valid_dir & isfile(gt_path_)
    
    % get indi path
    if indi_path
        indi_path_ = strcat(indi_path, imgDataDir(i).name, '.png');
        valid_dir = valid_dir & isfile(indi_path_);
    end
    
    % Add the current sample to the pathlist, if all files exist
    if valid_dir
        if indi_path
            pathsets{end+1} = {disp_map_path, gt_path_, conf_map_paths, indi_path_};
        else
            pathsets{end+1} = {disp_map_path, gt_path_, conf_map_paths};
        end
    end
end


end

