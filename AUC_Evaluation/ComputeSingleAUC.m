function auc = ComputeSingleAUC(disp_est_path, disp_gt_path, conf_map_path, threshold, step, gt_norm_factor)

percentage_interval = (0:step:1);
disp_est = double(imread(disp_est_path));
disp_gt = double(imread(disp_gt_path))/gt_norm_factor; % Change the format of the ground truth image
conf_map = double(imread(conf_map_path))/255; % normalization

% apply threshold 
T = thresholdMap(threshold, disp_gt);  % the threshold matrix
delta = abs(disp_gt - disp_est);   % the difference between ground truth and candidate image

conf_list = confidenceList(conf_map,disp_gt);
[mark_point,percentage_list,sub_list] = percentageList(conf_list,percentage_interval,step);
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
% AUC calculation
auc = getAUC(percentage_list,error_rate);