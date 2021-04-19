clear
close all
clc

% Define paths and params
dataset = 'M3';
disp_method = 'MC-CNN';
mode = 'Mixed_Uniform';
savefig = false; % save heat map


unc_label = 'Standard Deviation [pixel]';

if strcmp(mode, 'Probabilistic')
    unc_name = strcat('ConfMap_CVA-Net_Probabilistic_paper_MC-CNN_M3.pfm');
    unc_limits = [0 600];
elseif strcmp(mode, 'Mixed_Uniform')
    unc_name = 'ConfMap_CVA-Net_Mixed_Uniform_paper_MC-CNN_M3.pfm';
    unc_limits = [0 150];
elseif strcmp(mode, 'Laplacian_Uniform')
    unc_name = 'ConfMap_CVA-Net_Laplacian_Uniform_paper_MC-CNN_M3.pfm';
    unc_limits = [0 400];
elseif strcmp(mode, 'GMM')
    unc_name = 'ConfMap_CVA-Net_GMM_paper_K3_MC-CNN_M3.pfm';
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
    disc_dir = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/mask_discont/';
    gt_norm_factor = 255;
elseif (strcmp(dataset, 'K15'))   
    gt_dir = '/media/zeyun/ZEYUN/MA/kitti-2015/disp_gt_occ/';
    disp_dir = '/home/zeyun/Projects/CVA/results/kitti-2015/';
    disp_name = strcat('DispMap', '_', disp_method, '.png');
    disc_dir = '/media/zeyun/ZEYUN/MA/kitti-2015/mask_discont/';
    gt_norm_factor = 255;
elseif (strcmp(dataset, 'M3'))
    gt_dir = '/media/zeyun/ZEYUN/MA/middlebury-v3/disp_gt/';
    disp_dir = '/home/zeyun/Projects/CVA/results/middlebury-v3/';
    disp_name = strcat('DispMap', '_', disp_method, '.png');
    disc_dir = '/media/zeyun/ZEYUN/MA/middlebury-v3/mask_discont/';
    gt_norm_factor = 1;
else
    error('Unknown dataset!')
end


%%
% Define list of images used for evaluation
img_list = dir(disp_dir);
img_list = {img_list(:).name};
img_list = img_list(3:end);


% Read images and extract points for the heatmap
points = [];
for img_idx=6:length(img_list)
    % Load estimated and ground truth disparity maps and the corresponding
    % uncertainty maps
    % gt1 = read_pfm(strcat('/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/disp_occ_pfm/', img_list{img_idx}, '.pfm'), 0);
    gt = double(imread(strcat(gt_dir, img_list{img_idx}, '.png')))/gt_norm_factor;
    gt(isinf(gt)|isnan(gt)) = 0;
    
    disp = double(imread(strcat(disp_dir, img_list{img_idx}, '/', disp_name)));
    unc = abs(read_pfm(strcat(disp_dir, img_list{img_idx}, '/', unc_name), 0));
    
    % Compute error per pixel
    error = abs(gt - disp);
    curr_points = [error(:), unc(:)];
    disc = double(imread(strcat(disc_dir, img_list{img_idx}, '.png')));
    curr_points = curr_points(gt(:)>=1.0 & disc(:)==128,:);
    points = cat(1,points,curr_points);
end

% Generate heat map
roh = corrcoef(points)
Rsq = LinearRegression(points);
savename = strrep(unc_name, 'pfm', 'svg');
Generate_Heatmap(points, savefig, savename, unc_limits, true, true);