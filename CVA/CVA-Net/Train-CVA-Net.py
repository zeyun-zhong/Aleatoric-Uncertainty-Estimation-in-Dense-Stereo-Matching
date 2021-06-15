import time
import os
import sys
sys.path.insert(1, '../utils')

from opts import _parse_train_opts
import params
from data_generators import DataSample
from train import Train

args = _parse_train_opts()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# training params may need to be changed
parameter = params.Params()
parameter.epochs = 40
parameter.batch_size = args.batch_size
parameter.learning_rate = args.lr
parameter.nb_filter_size = args.nb_filter_size # 3 or 5
parameter.nb_filter_num = args.nb_filter_num
parameter.dense_layer_type = args.dense_layer_type # 'AP' or 'FC'
parameter.dense_filter_num = args.dense_filter_num
parameter.depth_filter_num = args.depth_filter_num
parameter.data_mode = 'cv'
parameter.loss_type = args.loss_type
# parameter for Geometry-aware model with mask prediction
parameter.geometry_loss_weight = args.geometry_loss_weight
# GMM parameter
parameter.gmm_loss_weight = args.gmm_loss_weight
parameter.num_gaussian_comp = args.num_gaussian_comp

if args.cluster:
    root_path = '/bigwork/nhgnzhon/Data/'
else:
    root_path = '/home/zeyun/Projects/CVA/'

print(root_path)
# ---------------------------
# Assemble datasets
# ---------------------------
parameter.training_data = []
parameter.validation_data = []

if args.dataset == 'K12':
    data_path = root_path + 'stimuli/kitti-2012/training/'
elif args.dataset == 'K15': # not used
    data_path = root_path + 'stimuli/kitti-2015/training/'
elif args.dataset == 'M3':
    data_path = root_path + 'stimuli/middlebury-v3/'
elif args.dataset == 'Sceneflow':
    data_path = root_path + 'stimuli/sceneflow/'
    parameter.abs_error_prop = {3: 0.9267, 10: 0.0289, 50: 0.0309, float('inf'): 0.0135}
else:
    raise Exception('Unknown dataset: %s' % args.dataset)

left_image_path = data_path + 'image_0/'
right_image_path = data_path + 'image_1/'
disp_gt_path = data_path + 'disp_occ/'
indic_path = data_path + 'mask_indicator/'

# In this work, we use Census_BM and MC-CNN to generate cost volumes
cv_method = args.cv_method
if cv_method == 'Census-BM':
    cv_path = data_path + 'cost_volumes/'
    disp_est_path = '' # add path if the estimated disparity map should be loaded
elif cv_method == 'GC-Net':
    cv_path = data_path + 'GC-Net/cost_volumes/'
    disp_est_path = data_path + 'GC-Net/disp_est/'
    parameter.cv_norm = [33.29, 0.0064] if 'sceneflow' in data_path else [37.37, 0.00817]
elif cv_method == 'MC-CNN':
    cv_path = data_path + 'MC-CNN/cost_volumes/'
    disp_est_path = '' # data_path + 'MC-CNN/disp_est/'
    parameter.cv_norm = [1, 0.5]
else:
    raise Exception('Unknown cv method: %s' % cv_method)

files = sorted(os.listdir(cv_path))

# number of cost volumes for training and valuation
if args.dataset == 'Sceneflow':
    n = 5
elif cv_method == 'MC-CNN' and args.dataset == 'K12':
    n = 21
else:
    n = 20
m = 8 if args.dataset == 'Sceneflow' else 23

for img_idx in range(0, n):
    if cv_method == 'MC-CNN' and img_idx == 1 and args.dataset == 'K12': continue
    sample_name = files[img_idx].replace('.bin', '')
    print(sample_name)

    indicator_path = indic_path + sample_name + '.png' if 'Geometry' in parameter.loss_type else ''
    est_path = disp_est_path + sample_name + '.png' if disp_est_path else ''
    cost_volume_path = cv_path + sample_name + '.bin' if cv_method == 'MC-CNN' else cv_path + sample_name
    gt_path = disp_gt_path + sample_name + '.pfm' if args.dataset == 'Sceneflow' else disp_gt_path + sample_name + '.png'
    parameter.training_data.append(DataSample(gt_path=gt_path,
                                              indicator_path=indicator_path,
                                              est_path=est_path,
                                              left_image_path=left_image_path + sample_name + '.png',
                                              right_image_path=right_image_path + sample_name + '.png',
                                              cost_volume_depth=256,
                                              cost_volume_path=cost_volume_path))

for img_idx in range(n, m):
    sample_name = files[img_idx].replace('.bin', '') if files else str(img_idx).rjust(6, '0') + '_10'
    print(sample_name)

    indicator_path = indic_path + sample_name + '.png' if 'Geometry' in parameter.loss_type else ''
    est_path = disp_est_path + sample_name + '.png' if disp_est_path else ''
    cost_volume_path = cv_path + sample_name + '.bin' if cv_method == 'MC-CNN' else cv_path + sample_name
    gt_path = disp_gt_path + sample_name + '.pfm' if args.dataset == 'Sceneflow' else disp_gt_path + sample_name + '.png'
    parameter.validation_data.append(DataSample(gt_path=gt_path,
                                                indicator_path=indicator_path,
                                                est_path=est_path,
                                                left_image_path=left_image_path + sample_name + '.png',
                                                right_image_path=right_image_path + sample_name + '.png',
                                                cost_volume_depth=256,
                                                cost_volume_path=cost_volume_path))

# ---------------------------
# Start training
# ---------------------------
root_dir = root_path + 'experiments'
experiment_series = 'dynamic-depth'
network_name = "CVA-Net_{0}".format(int(time.time())) if not args.name else "CVA-Net_" + args.name
# pretrained_network = root_path + 'experiments/dynamic-depth/CVA-Net_Mixed_Uniform_mask_paper_MC-CNN/models/weights_13_2.665.h5'
pretrained_network = ''

print('Initialising...')
trainer = Train(parameter=parameter, network_name=network_name, root_dir=root_dir,
                experiment_series=experiment_series, pretrained_network=pretrained_network)

print('Start training...')
trainer.train()
