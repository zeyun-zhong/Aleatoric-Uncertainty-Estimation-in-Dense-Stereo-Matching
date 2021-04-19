from image_io import read, write
import os
import pickle
import sys
sys.path.insert(1, '../CVA-Net')
import cost_volume
import numpy as np
import image_io
import matplotlib.pyplot as plt
import random


def load_cost_volume(cv_path, img_path=None):
    cv = cost_volume.CostVolume()
    if cv_path[-3:] == 'bin':
        img_shape = image_io.read(img_path).shape
        cv.load_bin(cv_path, img_shape[0], img_shape[1], 192)

    elif cv_path[-3:] == 'dat':
        cv.load_dat(cv_path)

    else:
        with open(cv_path, 'rb') as file:
            cv = pickle.load(file)

    cv.normalise(1, 0.5)
    return cv.get_data()


'''
img_idx = 3
sample = str(img_idx).rjust(6, '0') + '_10'
cv_path = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/MC-CNN/cost_volumes/' + sample + '.bin'
indi_path = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/mask_indicator/{}.png'.format(sample)
gt_path = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/disp_occ/{}.png'.format(sample)
cv = load_cost_volume(cv_path, gt_path)
mask_indi = read(indi_path)
gt = read(gt_path)
m = 5

region_index_good = np.where(np.logical_and(mask_indi == 1, gt != 0))
region_index_hard = np.where(np.logical_and(mask_indi == 0, gt != 0))
index_good_tmp = random.sample(range(len(region_index_good[0])), m)
index_good = [[region_index_good[0][i], region_index_good[1][i]] for i in index_good_tmp]
index_hard_tmp = random.sample(range(len(region_index_hard[0])), m)
index_hard = [[region_index_hard[0][i], region_index_hard[1][i]] for i in index_hard_tmp]
plt.figure()
for i in index_good:
    cost_curve = cv[i[0], i[1], :]
    plt.plot(cost_curve, label='({},{})'.format(i[0], i[1]))
plt.title('good')
plt.legend()
# plt.show()
plt.savefig(sample + '_good.png', bbox_inches='tight')

plt.figure()
for i in index_hard:
    cost_curve = cv[i[0], i[1], :]
    plt.plot(cost_curve, label='({},{})'.format(i[0], i[1]))
plt.title('hard')
plt.legend()
# plt.show()
plt.savefig(sample + '_hard.png', bbox_inches='tight')
'''


cv_path = '/media/zeyun/ZEYUN/MA/kitti-2015/cv_MC'
img_path = '/media/zeyun/ZEYUN/MA/kitti-2015/images/left'
cv_paths = sorted(os.listdir(cv_path))
img_paths = sorted(os.listdir(img_path))
MAX, MIN = 0, float('inf')
for i, path in enumerate(cv_paths):
    a = load_cost_volume(os.path.join(cv_path, path), os.path.join(img_path, img_paths[i]))
    b = np.max(a)
    c = np.min(a)
    print(b)
    print(c)
    MAX = max(b, MAX)
    MIN = min(c, MIN)
print('Max: ', MAX)
print('Min: ', MIN)


