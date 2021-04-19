function [opt_auc_list, conf_auc_list, roc_list] = ComputeSequenceAUC(pathsets, indicat, threshold, step, gt_norm_factor, conf_norm_factor, conf_sort)

conf_auc_list = {};
roc_list = {};
percentage_interval = (0:step:1);

% preallocation for opt auc
if indicat
    opt_auc_list = zeros(length(pathsets),2);
else
    opt_auc_list = zeros(length(pathsets),1);
end

% no indicator case
im_indi = '';

for i = 1:length(pathsets)
    pathset = pathsets{i};
    pathset_size = size(pathset);
    path2disp = pathset{1};
    path2GT = pathset{2};
    path2conf_list = pathset{3};
    if pathset_size(2) > 3
        path2indi = pathset{4};
    end
    
    im_disp = double(imread(path2disp));
    a = imread(path2GT);
    im_GT = double(imread(path2GT))/gt_norm_factor; % Change the format of the ground truth image
    if indicat
        im_indi = double(imread(path2indi));
    end
    
    % apply threshold 
    T = thresholdMap(threshold, im_GT);  % the threshold matrix
    delta = abs(im_GT - im_disp);   % the difference between ground truth and candidate image
    
    % preallocation
    if indicat
        auc = zeros(1,length(path2conf_list)*2);
    else
        auc = zeros(1,length(path2conf_list));
    end
    
    % used to plot roc curve
    %percentages = {};
    errors = {};
    
    % Traverse all the confidence map
    for k = 1:length(path2conf_list) 
        path2conf = path2conf_list{k}{1};
        im_conf = double(read_pfm(path2conf, 0));
        im_conf = im_conf/conf_norm_factor(k); % normalization
        im_conf = abs(im_conf);
        % create the confidence list
        [conf_list, conf_list_other] = confidenceList_indicator(im_conf,im_GT,im_indi,conf_sort{k});
        % create the percentage list
        [mark_point,percentage_list,sub_list] = percentageList(conf_list,percentage_interval,step);
        if indicat
            [mark_point_other,percentage_list_other,sub_list_other] = percentageList(conf_list_other,percentage_interval,step);
        end
        % calculate error rate
        if size(sub_list) ~= [0,0]
            % sub_list contains all sections of the percentage list that 
            % are larger than a certain threshold (here 10%) and should 
            % therefore be subdivided. The mean error is only computed for 
            % those sections.
            error_rate = getErrorrate(mark_point,conf_list,delta,T);
            [sub_list_best,sub_list_worst] = sub_error_rate(sub_list,conf_list,delta,T);
            [percentage_list, error_rate] = error_list_combine(sub_list_best,sub_list_worst,percentage_list,error_rate);
        else
            error_rate = getErrorrate(mark_point,conf_list,delta,T);
        end
        
        if indicat
            if size(sub_list_other) ~= [0,0]
                error_rate_other = getErrorrate(mark_point_other,conf_list_other,delta,T);
                [sub_list_best_other,sub_list_worst_other] = sub_error_rate(sub_list_other,conf_list_other,delta,T);
                [percentage_list_other, error_rate_other] = error_list_combine(sub_list_best_other,sub_list_worst_other,percentage_list_other,error_rate_other);
            else
                error_rate_other = getErrorrate(mark_point_other,conf_list_other,delta,T);
            end
        end
        
        % AUC calculation
        auc(k) = getAUC(percentage_list,error_rate);
        if indicat
            auc(k+length(path2conf_list)) = getAUC(percentage_list_other, error_rate_other);
        end
        
        % save roc curve
        if indicat
            %percentages{end+1} = [percentage_list; percentage_list_other];
            errors{end+1} = [error_rate'; error_rate_other'];
        else
            %percentages{end+1} = percentage_list;
            errors{end+1} = error_rate';
        end
    end
    Epsilon = error_rate(end);
    opt_auc_list(i, 1) = getOptAUC(Epsilon);
    if indicat
        Epsilon_other = error_rate_other(end);
        opt_auc_list(i, 2) = getOptAUC(Epsilon_other);
    end
    conf_auc_list = [conf_auc_list;auc];
    roc_list = [roc_list;errors];
end

end

