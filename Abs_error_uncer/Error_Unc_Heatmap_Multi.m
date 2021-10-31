clear
close all
clc

dataset = 'K15';
disp_method = 'MC-CNN';
indicator = true;
savefig = true; % save heat map
show_big_error = false;

if show_big_error
    save_name = strcat(disp_method, "_", dataset, "_big_error", ".svg");
else
    save_name = strcat(disp_method, "_", dataset, ".svg");
end

unc_limits = [0 150];

% unc_names = [strcat("ConfMap_CVA-Net_Laplacian_", disp_method, ".pfm"),
%     strcat("ConfMap_CVA-Net_Weighted_Laplacian_", disp_method, "_lookup.pfm"),
%     strcat("ConfMap_CVA-Net_Weighted_Laplacian_", disp_method, "_lookup_std_4.pfm"),
%     strcat("ConfMap_CVA-Net_Weighted_Laplacian_", disp_method, "_lookup_std_4_thresh_5.pfm")];

unc_names = [strcat("ConfMap_CVA-Net_Geometry_", disp_method, ".pfm"),
    strcat("ConfMap_CVA-Net_Weighted_Geometry_", disp_method, "_lookup_std_4.pfm")];

%% define path
[gt_dir, disp_dir, disp_name, indi_dir, gt_norm_factor] = define_path(dataset, disp_method);

%% read pfm for GC-Net
if (strcmp(disp_method, 'GC-Net'))
    disp_name = replace(disp_name, '.png', '.pfm');
end

%% load data
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
    
    % Compute error per pixel
    error = abs(gt - disp);
    curr_points = [error(:)];
    
    
    for unc_idx=1:length(unc_names)
        unc = abs(read_pfm(strcat(disp_dir, img_list{img_idx}, '/', unc_names(unc_idx)), 0));
        curr_points = cat(2,curr_points,unc(:));
    end
    
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

%% Generate heat map
figure('DefaultAxesFontSize', 11);
set(gcf,'position',[300,100,600,500]);

tiledlayout(indicator+1, length(unc_names), 'TileSpacing', 'none', 'Padding', 'none');

for idx=1:length(unc_names)
    label_y = false;
    clbar = false;
    
    curr_points = [points(:,1),points(:,idx+1)];
    roh = corrcoef(curr_points)
    if idx==1
        label_y = true;
        clbar = true;
    end
    
    if ~indicator
        nexttile
        Generate_Heatmap_Multi(curr_points, roh, unc_limits, true, label_y, clbar);
    else
        nexttile(idx)
        Generate_Heatmap_Multi(curr_points, roh, unc_limits, false, label_y, clbar);
        
        curr_points_other = [points_other(:,1),points_other(:,idx+1)];
        roh_other = corrcoef(curr_points_other)
        nexttile(idx + length(unc_names))
        Generate_Heatmap_Multi(curr_points_other, roh_other, unc_limits, true, label_y, false);
    end
end

if savefig
    saveas(gcf,save_name)
end

%%
function [gt_dir, disp_dir, disp_name, indi_dir, gt_norm_factor] = define_path(dataset, disp_method)
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
end