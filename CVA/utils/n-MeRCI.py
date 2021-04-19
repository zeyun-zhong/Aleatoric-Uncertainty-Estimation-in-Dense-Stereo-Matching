import numpy as np
from image_io import read
import os


class n_MeRCI():
    def __init__(self, indicator_path, gt_path, est_path, unc_path, model_name, alpha=0.95, neglect_rate=0):
        self.indicator_path = indicator_path
        self.gt_path = gt_path
        self.est_path = est_path
        self.unc_path = unc_path
        self.model_name = model_name
        self.alpha = alpha if alpha <= 1 else alpha / 100
        self.neglect_rate = neglect_rate if neglect_rate < 1 else neglect_rate / 100
        self.data_good, self.data_hard = self.load_standard_abs_error()

    def load_standard_abs_error(self):
        model_name = 'ConfMap_CVA-Net_' + self.model_name + '.pfm'
        file_list = sorted(os.listdir(self.unc_path))
        file_list = [file + '.png' for file in file_list]

        error_abs_good, unc_good, gt_good, est_good = [], [], [], []
        error_abs_hard, unc_hard, gt_hard, est_hard = [], [], [], []

        for img_path in file_list:
            indi = read(self.indicator_path + img_path)
            gt = read(self.gt_path + img_path).astype(float)
            est = read(self.est_path + img_path).astype(float)
            error_abs = np.abs(gt - est)
            unc = read(self.unc_path + img_path.replace('.png', '/') + model_name)
            index_good = np.where(indi == 1)
            index_hard = np.where(np.logical_and(indi == 0, gt != 0))
            # data of good region
            error_abs_good.extend(error_abs[index_good])
            unc_good.extend(unc[index_good])
            gt_good.extend(gt[index_good])
            est_good.extend(est[index_good])
            # data of hard region
            error_abs_hard.extend(error_abs[index_hard])
            unc_hard.extend(unc[index_hard])
            gt_hard.extend(gt[index_hard])
            est_hard.extend(est[index_hard])
            # break

        error_abs_standard_good = np.array(error_abs_good) / np.array(unc_good)
        error_abs_standard_hard = np.array(error_abs_hard) / np.array(unc_hard)

        data_good = np.column_stack([error_abs_standard_good, gt_good, est_good, unc_good])
        data_hard = np.column_stack([error_abs_standard_hard, gt_hard, est_hard, unc_hard])

        return data_good, data_hard

    def extract_data_and_lambda(self, data):
        """
        data: ndarray, [standardized abs error, gt, est, unc]
        """
        # generates cumulative distribution of abs error
        # abs_error = np.abs(data[:, 1] - data[:, 2])
        # abs_error = np.sort(abs_error)
        # x = range(0, 255)
        # y = [np.argmax(abs_error > i) / len(abs_error) for i in x]
        # print(y)

        data_sort_unc = data[data[:, -1].argsort()]
        neglect_row = int(data_sort_unc.shape[0] * self.neglect_rate)
        data_extract_unc = data_sort_unc[neglect_row:data.shape[0]-neglect_row, :]

        data_sort_lam = data_extract_unc[data_extract_unc[:, 0].argsort()]

        extract_row = int(data_extract_unc.shape[0] * self.alpha)
        data_extract_lam = data_sort_lam[0:extract_row, :]
        lam = data_extract_lam[-1, 0]
        return data_extract_lam, lam

    def calc_n_MeRCI_single(self, data):
        """
        data: ndarray, [standardized abs error, gt, est, unc]
        """
        data_extract, lam = self.extract_data_and_lambda(data)
        abs_error = np.abs(data_extract[:, 1] - data_extract[:, 2])
        MAE = np.mean(abs_error)
        MeRCI = np.mean(lam * data_extract[:, -1])
        n_MeRCI = (MeRCI - MAE) / (np.max(abs_error) - MAE)
        return n_MeRCI

    def calc_n_MeRCI(self):
        n_MeRCI_good = self.calc_n_MeRCI_single(self.data_good)
        n_MeRCI_hard = self.calc_n_MeRCI_single(self.data_hard)
        return n_MeRCI_good, n_MeRCI_hard


if __name__ == '__main__':
    dataset = 'K15'
    dataset_name = 'middlebury-v3' if dataset == 'M3' else 'kitti-2015'
    disp_method = 'Census-BM'
    if disp_method == 'MC-CNN' and dataset == 'M3':
        suffix = '_MC-CNN_M3'
    elif disp_method == 'MC-CNN':
        suffix = '_MC-CNN'
    else:
        suffix = ''
    indicator_path = '/media/zeyun/ZEYUN/MA/' + dataset_name + '/mask_indicator/'
    gt_path = '/media/zeyun/ZEYUN/MA/' + dataset_name + '/disp_gt_occ/'
    est_path = '/media/zeyun/ZEYUN/MA/' + dataset_name + '/est_' + disp_method + '/'
    unc_path = '/home/zeyun/Projects/CVA/results/' + dataset_name + '/'
    model_names = ['Probabilistic_paper', 'Mixed_Uniform_paper', 'GMM_paper', 'Laplacian_Uniform_paper']
    model_names = [name + suffix for name in model_names]

    alphas = [x * 0.01 for x in range(96, 100, 5)]

    data = open("n_MeRCI_{}_{}.txt".format(dataset, disp_method), mode='a')

    for alpha in alphas:
        data.write('alpha= {:.2f}'.format(alpha))
        for model_name in model_names:
            if 'GMM' in model_name and disp_method == 'MC-CNN':
                if dataset == 'M3': model_name = 'GMM_paper_K3_MC-CNN_M3'
                else: model_name = 'GMM_paper_K3_MC-CNN_epoch_10'

            metric = n_MeRCI(indicator_path, gt_path, est_path, unc_path, model_name, alpha=alpha)
            n_MeRCI_good, n_MeRCI_hard = metric.calc_n_MeRCI()
            data.write(" {:.3f} {:.3f}".format(n_MeRCI_good, n_MeRCI_hard))
            data.flush()
            print(n_MeRCI_good)
            print(n_MeRCI_hard)
        data.write('\n')
