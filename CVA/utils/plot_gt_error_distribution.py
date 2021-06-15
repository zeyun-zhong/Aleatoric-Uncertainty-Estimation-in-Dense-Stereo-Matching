import numpy as np
import matplotlib.pyplot as plt
from image_io import read
import os


def load_abs_error(indicator_path, gt_path, est_path):
    file_list = sorted(os.listdir(gt_path))

    error_abs_list, error_abs_good, error_abs_hard = [], [], []

    for img_path in file_list:
        gt = read(gt_path + img_path).astype(float)
        img_path = img_path.replace('.pfm', '.png')
        est = read(est_path + img_path).astype(float)
        error_abs = np.abs(gt - est)
        error_abs_list.extend(error_abs[gt != 0])

        if indicator_path:
            indi = read(indicator_path + img_path)
            index_good = np.where(indi == 1)
            index_hard = np.where(np.logical_and(indi == 0, gt != 0))
            # data of good region
            error_abs_good.extend(error_abs[index_good])
            # data of hard region
            error_abs_hard.extend(error_abs[index_hard])

    return np.array(error_abs_list), np.array(error_abs_good), np.array(error_abs_hard)


def extract_abs_error(abs_error, x):
    """
    data: 1d array, abs error
    """
    # generates cumulative distribution of abs error
    if len(abs_error) == 0: return
    abs_error = np.sort(abs_error)
    a = max(abs_error)
    y = [np.argmax(abs_error > i) / len(abs_error) for i in x]
    return y


def plot_gt_error_sub(x, y, ax, xname='', yname=''):
    if y[-1] <= 1: y = [i*100 for i in y]
    # delete unexpected values
    if 0 in y:
        index_0 = y.index(0)
        x = x[:index_0]
        y = y[:index_0]
    ax.plot(x, y)
    ax.set_xlim([0, 255])
    ax.set_ylim([50, 100])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_yscale('log')


def main(indicator_path, gt_path, est_path, x=range(0,255), dataset='K15', disp_method='Census-BM'):
    error_abs, error_abs_good, error_abs_hard = load_abs_error(indicator_path, gt_path, est_path)
    y, y_good, y_hard = extract_abs_error(error_abs, x), extract_abs_error(error_abs_good, x), extract_abs_error(error_abs_hard, x)
    # plot gt abs error
    if y_good and y_hard:
        plt.style.use('seaborn-whitegrid')
        plt.rc('font', family='Times New Roman')
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharex='col', sharey='row')
        plot_gt_error_sub(x, y_good, ax[0], xname='disparity [pixel]', yname='CDF [%]')
        plot_gt_error_sub(x, y_hard, ax[1], xname='disparity [pixel]')
        fig.tight_layout()
        # plt.savefig('{}_{}.svg'.format(dataset, disp_method))
        # plt.show()
    print(y)
    print(y_good)
    print(y_hard)


if __name__ == '__main__':
    # dataset = 'K15'
    # dataset_name = 'middlebury-v3' if dataset == 'M3' else 'kitti-2015'
    # disp_method = 'MC-CNN'

    base_path = '/home/zeyun/Projects/CVA/stimuli/sceneflow/'

    indicator_path = None
    gt_path = base_path + 'disp_occ/'
    est_path = base_path + 'GC-Net/disp_est/'
    main(indicator_path, gt_path, est_path, x=[3, 10, 50])