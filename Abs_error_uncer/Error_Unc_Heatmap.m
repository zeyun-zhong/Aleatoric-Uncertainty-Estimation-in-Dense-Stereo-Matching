clear
close all
clc

% Define paths and params
dataset = 'K15';
disp_method = 'Census-BM';
mode = 'Geometry';
indicator = true;
savefig = false; % save heat map
show_big_error = false;

unc_label = 'Standard Deviation [pixel]';

if strcmp(mode, 'Laplacian')
    %unc_name = 'AleaUncMap_CVA-Net_Laplace_Max_1.pfm';
    unc_name = 'ConfMap_CVA-Net_Laplacian_MC-CNN.pfm';
    unc_limits = [0 300];
elseif strcmp(mode, 'Weighted_Laplacian')
    unc_name = 'ConfMap_CVA-Net_Weighted_Laplacian_MC-CNN_lookup_std_4_thresh_5.pfm';
    unc_limits = [0 300];
elseif strcmp(mode, 'Geometry')
    unc_name = 'ConfMap_CVA-Net_Weighted_Geometry_Census-BM_lookup_std_4.pfm';
    unc_limits = [0 300];
elseif strcmp(mode, 'Mixture')
    unc_name = 'ConfMap_CVA-Net_Mixture_Census-BM.pfm';
    unc_limits = [0 400];
elseif strcmp(mode, 'GMM')
    unc_name = 'ConfMap_CVA-Net_GMM_paper_K3_new.pfm';
    unc_limits = [0 300];
else
    ME = MException('mode:noSuchVariable', 'Unknown mode ', mode);
    throw(ME);
end

%% create path
if (strcmp(dataset, 'K12'))
    gt_dir = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/disp_occ/';
    disp_dir = '/home/zeyun/Projects/CVA/results/kitti-2012/';
    disp_name = 'DispMap.png';
    indi_dir = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/mask_indicator/';
    gt_norm_factor = 256.0;
elseif (strcmp(dataset, 'K15'))   
    gt_dir = '/media/zeyun/ZEYUN/MA/kitti-2015/disp_gt_occ/';
    disp_dir = '/home/zeyun/Projects/CVA/results/kitti-2015/';
    disp_name = strcat('DispMap', '_', disp_method, '_192', '.png');
    indi_dir = '/media/zeyun/ZEYUN/MA/kitti-2015/mask_indicator/';
    gt_norm_factor = 256.0;
elseif (strcmp(dataset, 'M3'))
    gt_dir = '/media/zeyun/ZEYUN/MA/middlebury-v3/disp_gt_pfm/';
    disp_dir = '/home/zeyun/Projects/CVA/results/middlebury-v3/';
    disp_name = strcat('DispMap', '_', disp_method, '_192', '.png');
    indi_dir = '/media/zeyun/ZEYUN/MA/middlebury-v3/mask_indicator_wo_disc/';
    gt_norm_factor = 1;
elseif (strcmp(dataset, 'Sceneflow'))
    gt_dir = '/media/zeyun/ZEYUN/MA/sceneflow/disp_gt/';
    disp_dir = '/home/zeyun/Projects/CVA/results/sceneflow/';
    disp_name = strcat('DispMap', '_', disp_method, '.png');
    indi_dir = '/media/zeyun/ZEYUN/MA/sceneflow/mask_indicator/';
    gt_norm_factor = 1;
else
    error('Unknown dataset!')
end

%% read pfm for GC-Net
if (strcmp(disp_method, 'GC-Net'))
    disp_name = replace(disp_name, '.png', '.pfm');
end


%%
% Define list of images used for evaluation
img_list = dir(disp_dir);
img_list = {img_list(:).name};
img_list = img_list(3:end);


% Read images and extract points for the heatmap
points = [];
if indicator
    points_other = [];
end
for img_idx=1:length(img_list)
    % Load estimated and ground truth disparity maps and the corresponding
    % uncertainty maps
    if contains(gt_dir, 'pfm')
        gt = read_pfm(strcat(gt_dir, img_list{img_idx}, '.pfm'), 0);
    else
        gt = double(imread(strcat(gt_dir, img_list{img_idx}, '.png')))/gt_norm_factor;
    end
    gt(isinf(gt)|isnan(gt)) = 0;
    
    if contains(disp_name, 'pfm')
        disp = read_pfm(strcat(disp_dir, img_list{img_idx}, '/', disp_name), 0);
    else
        disp = double(imread(strcat(disp_dir, img_list{img_idx}, '/', disp_name)));
    end
    
    unc = abs(read_pfm(strcat(disp_dir, img_list{img_idx}, '/', unc_name), 0));
    
    % Compute error per pixel
    error = abs(gt - disp);
    curr_points = [error(:), unc(:)];
    if ~indicator
        curr_points = curr_points(gt(:)>=1.0,:);
        points = cat(1,points,curr_points);
    else
        indi = double(imread(strcat(indi_dir, img_list{img_idx}, '.png')));
        curr_points_good = curr_points(gt(:)>=1.0 & indi(:)>=1.0,:);
        curr_points_other = curr_points(gt(:)>=1.0 & indi(:)<1.0,:);
        points = cat(1,points,curr_points_good);
        points_other = cat(1,points_other,curr_points_other);
    end
end

if show_big_error
    points = points(points(:,1)>= 3.0, :);
    if indicator
        points_other = points_other(points_other(:,1)>=3.0, :);
    end
end

% Generate heat map
roh = corrcoef(points)
if ~indicator
    savename = strrep(unc_name, 'pfm', 'svg');
    Generate_Heatmap(points, savefig, savename, unc_limits, true, true);
else
    savename_good = strrep(unc_name, '.pfm', '_good.svg');
    Generate_Heatmap(points, savefig, savename_good, unc_limits, true, true);
    roh_other = corrcoef(points_other)
    savename_other = strrep(unc_name, '.pfm', '_other.svg');
    Generate_Heatmap(points_other, savefig, savename_other, unc_limits, true, false);
end