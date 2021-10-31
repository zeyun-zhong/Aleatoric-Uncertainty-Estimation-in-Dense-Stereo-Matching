import cv2
import numpy as np
import os
from image_io import read, write
import matplotlib.pyplot as plt


class UncertaintySmallErrors:
    def __init__(self, dataset, cv_method):
        assert dataset in ['kitti-2015', 'middlebury-v3'], 'Unknown dataset name'
        assert cv_method in ['Census-BM', 'MC-CNN', 'GC-Net'], 'Unknown disparity method!'
        self.dataset = dataset
        self.cv_method = cv_method
        self.set_paths()

    def set_paths(self):
        root_path = '/media/zeyun/ZEYUN/MA/'
        est_paths = {
            'Census-BM': 'est_Census-BM',
            'MC-CNN': 'est_MC-CNN',
            'GC-Net': 'est_GC-Net_pfm'
        }

        gt_paths = {
            'kitti-2015': 'disp_gt_occ',
            'middlebury-v3': 'disp_gt_pfm'
        }

        self.disp_est_path = os.path.join(root_path, self.dataset, est_paths[self.cv_method])
        self.disp_gt_path = os.path.join(root_path, self.dataset, gt_paths[self.dataset])
        self.unc_path = os.path.join('/home/zeyun/Projects/CVA/results/', self.dataset)
        self.left_img_path = os.path.join(root_path, self.dataset, 'images/left/')
        self.indicator_path = os.path.join(root_path, self.dataset, 'mask_indicator/' if self.dataset == 'kitti-2015' else 'mask_indicator_wo_disc')
        self.folders = sorted(os.listdir(self.unc_path))

    @staticmethod
    def show_pixels(img, trans_idx=127, *conds):
        """
        only show specific pixels with index, other pixels are marked as transparent
        :param img: uint8 img with bgr order
        :param trans_idx: transparent level
        :param conds: a set of conditions
        :return: transparent image with bgra order
        """
        index = np.where(np.logical_not(np.logical_and(*conds)))
        img_trans = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        index_3d = tuple(list(index) + [np.ones(len(index[0]), dtype=int) * 3])
        img_trans[index_3d] = trans_idx
        return img_trans

    def generate_plots(self, model_name, img_index, trans_idx=127, error_thresh=3, unc_thresh=100):
        self.load_data(model_name, img_index)

        cond1 = self.gt > 0
        cond2 = self.indi == 1
        cond3 = self.indi == 0
        img_good_trans = self.show_pixels(self.left_img, trans_idx, *(cond1, cond2))
        img_hard_trans = self.show_pixels(self.left_img, trans_idx, *(cond1, cond3))

        # conds for pixels with small error and large uncertainty
        abs_disparity_errors = np.abs(self.gt - self.est)
        cond3 = abs_disparity_errors <= error_thresh
        cond4 = self.unc >= unc_thresh
        img_unc_trans = self.show_pixels(self.left_img, trans_idx, *(cond1, cond3, cond4))

        # plot
        self.fig, ax = plt.subplots(figsize=(15, 5.5), nrows=2, ncols=2)
        self.plot(self.left_img, ax[0, 0], 'Reference')
        self.plot(img_unc_trans, ax[0, 1], 'Small Error Large Unc')
        self.plot(img_good_trans, ax[1, 0], 'Good Region')
        self.plot(img_hard_trans, ax[1, 1], 'Hard Region')
        plt.tight_layout()
        # plt.show()

    @staticmethod
    def plot(img, ax, title):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB if title == 'Reference' else cv2.COLOR_BGRA2RGBA))
        ax.axis('off')
        ax.set_title(title)

    def load_data(self, model_name, img_index):
        img_path = self.folders[img_index]
        self.gt = self.load_file(os.path.join(self.disp_gt_path, img_path))
        self.est = self.load_file(os.path.join(self.disp_est_path, img_path))
        self.unc = self.load_file(os.path.join(self.unc_path, img_path, 'ConfMap_{}'.format(model_name)))
        self.left_img = cv2.imread(os.path.join(self.left_img_path, img_path + '.png'))
        self.indi = self.load_file(os.path.join(self.indicator_path, img_path))

    @staticmethod
    def load_file(file_path):
        try:
            file = read(file_path + '.png').astype(float)
        except:
            file = read(file_path + '.pfm')
        return file


if __name__ == '__main__':
    dataset = 'kitti-2015'
    cv_method = 'MC-CNN'
    save_path_root = '/home/zeyun/Dropbox/LUH/IPI/Meeting/21.10/images/'

    USE = UncertaintySmallErrors(dataset=dataset, cv_method=cv_method)

    for i in range(10):
        USE.generate_plots('CVA-Net_Weighted_Laplacian_Census-BM_lookup_std_4', img_index=i, trans_idx=30)
        img_name = USE.folders[i] + '.png'

        save_path = os.path.join(save_path_root, cv_method, dataset, img_name)
        USE.fig.savefig(save_path)
