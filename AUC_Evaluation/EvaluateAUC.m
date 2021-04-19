clear
close all
clc

%% Parameter definitions
dataset = 'K15';
disp_method = 'Census-BM';
%conf_methods = {'CVA-Net-192', 'CVA-Net-Residual', 'CVA-Net-Prob-occ', 'CVA-Net-Prob-noc'};
%key_word_disp = 'DispMap_CVA-Net-192'; % key word of the name of disparity map

% need to be changed
conf_methods = {'CVA-Net_Probabilistic_paper', 'CVA-Net_Mixed_Uniform_paper', 'CVA-Net_GMM_paper', 'CVA-Net_Laplacian_Uniform_paper'}; % 
legends = {'Optimal', 'Laplacian', 'Geometry', 'GMM', 'Mixture'};
conf_norm_factor = [1.0, 1.0, 1.0, 1.0];
conf_sort = {'ascend', 'ascend', 'ascend' 'ascend'};
indicat = true;
save_roc = true;

key_word_disp = strcat('DispMap', '_', disp_method);
key_word_conf = 'ConfMap'; % key word of the name of confidence map

threshold = [3; 0.05];
step_size = 0.05;
gt_norm_factor = 1;
%conf_norm_factor = [255.0, 1.0, 1.0, 1.0];
%conf_sort = {'descend', 'ascend', 'ascend', 'ascend'};


write_log_file = false;
time_stamp = string(datetime('now','TimeZone','local','Format','yyyy-MM-dd_HH:mm:ss'));
log_file_path = '/home/max/SeaDrive/My Libraries/Meine Bibliothek/results/ConfidenceMaps/logs/';
log_file_path = strcat(log_file_path, disp_method, '_', dataset, '_', time_stamp, '/');

if (write_log_file)
    if ~exist(log_file_path, 'dir')
        mkdir(log_file_path);
    end
end


%% Create paths
if (strcmp(dataset, 'K12'))
    %disp_path = strcat('/media/max/Daten/mehltretter/results/cost-volumes/', disp_method, '/kitti-2012/');
    %gt_path = '/media/max/Daten/mehltretter/data/kitti-2012/training/disp_occ/';
    disp_path = '/home/zeyun/Projects/CVA/results/kitti-2012/';
    gt_path = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/disp_occ/';
    indi_path = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/mask_indicator/';
    gt_in_subfolder = false;
    gt_norm_factor = 255;
elseif (strcmp(dataset, 'K15'))   
    disp_path = '/home/zeyun/Projects/CVA/results/kitti-2015/';
    gt_path = '/media/zeyun/ZEYUN/MA/kitti-2015/disp_gt_occ/';
    indi_path = '/media/zeyun/ZEYUN/MA/kitti-2015/mask_indicator/';
    gt_in_subfolder = false;
    gt_norm_factor = 255;
elseif (strcmp(dataset, 'M3'))
    disp_path = '/home/zeyun/Projects/CVA/results/middlebury-v3/';
    gt_path = '/media/zeyun/ZEYUN/MA/middlebury-v3/disp_gt/';
    indi_path = '/media/zeyun/ZEYUN/MA/middlebury-v3/mask_indicator_wo_disc/';
    gt_in_subfolder = false; 
else
    error('Unknown dataset!')
end

if ~indicat
    indi_path = '';
end
pathsets = getPathsets(disp_path, gt_path, indi_path, gt_in_subfolder, conf_methods, key_word_disp, key_word_conf);
conf_methods = [{'Optimal'}, conf_methods];

%% Compute AUC values
[opt_auc_list, conf_auc_list, roc_list] = ComputeSequenceAUC(pathsets, indicat, threshold, step_size, gt_norm_factor, conf_norm_factor, conf_sort);
conf_auc_list = cell2mat(conf_auc_list);
if indicat
    combined_AUCs_good = [opt_auc_list(:, 1), conf_auc_list(:, 1:length(conf_methods)-1)];
    combined_AUCs_other = [opt_auc_list(:, 2), conf_auc_list(:, length(conf_methods):end)];
    mean_AUCs_good = mean(combined_AUCs_good, 1);
    mean_AUCs_other = mean(combined_AUCs_other, 1);
    AUC2Log(combined_AUCs_good, mean_AUCs_good, conf_methods, write_log_file, log_file_path);
    AUC2Log(combined_AUCs_other, mean_AUCs_other, conf_methods, write_log_file, log_file_path);
else
    combined_AUCs = [opt_auc_list, conf_auc_list];
    mean_AUCs = mean(combined_AUCs, 1);
    AUC2Log(combined_AUCs, mean_AUCs, conf_methods, write_log_file, log_file_path);
end

%% Plot ROC curve
if indicat
    error_rate_good = cellfun(@(x) x(1, :),roc_list,'UniformOutput',false);
    lim_y_good = [0 35];
    [percentage_opt_good,error_opt_good,percentage_good,mean_error_rate_good] = PlotSequenceROC(error_rate_good, conf_methods, lim_y_good, save_roc, legends, 'good');
    error_rate_other = cellfun(@(x) x(2, :),roc_list,'UniformOutput',false);
    lim_y_other = [0 60];
    [percentage_opt_other,error_opt_other,percentage_other,mean_error_rate_other] = PlotSequenceROC(error_rate_other, conf_methods, lim_y_other, save_roc, legends, 'hard');
else
    lim_y = [0 60];
    PlotSequenceROC(roc_list, conf_methods, lim_y, save_roc, legends);
end

%% Plot ROC curve thesis
PlotROCThesis(percentage_opt_good,error_opt_good,percentage_opt_other,error_opt_other,percentage_good,mean_error_rate_good,percentage_other,mean_error_rate_other,lim_y_good,lim_y_other,save_roc,legends);

%% Plot the AUC values for the whole sequence
% The values are sorted by the optimal values NOT by the frame ID
% if indicat
%     PlotSequenceAUC(combined_AUCs_good, conf_methods, write_log_file, log_file_path, 'good');
%     PlotSequenceAUC(combined_AUCs_other, conf_methods, write_log_file, log_file_path, 'other');
% else
%     PlotSequenceAUC(combined_AUCs, conf_methods, write_log_file, log_file_path);
% end