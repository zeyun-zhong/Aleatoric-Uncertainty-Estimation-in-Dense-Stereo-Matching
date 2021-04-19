from util import compute_acc_bin, ECE, get_bin_info, standardize
import numpy as np
from image_io import read
import os
import matplotlib.pyplot as plt
import scipy.stats

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def load_standard_abs_error(indicator_path, gt_path, est_path, unc_path, model_names):
    model_names = ['ConfMap_CVA-Net_' + x + '.pfm' for x in model_names]
    file_list = sorted(os.listdir(unc_path))
    file_list = [file + '.png' for file in file_list]

    error_abs_good, error_abs_hard = [], []
    unc_good = [[] for _ in model_names]
    unc_hard = [[] for _ in model_names]

    for img_path in file_list:
        indi = read(indicator_path + img_path)
        gt = read(gt_path + img_path).astype(float)
        est = read(est_path + img_path).astype(float)
        error_abs = np.abs(gt - est)
        index_good = np.where(indi == 1)
        index_hard = np.where(np.logical_and(indi == 0, gt != 0))
        error_abs_good.extend(error_abs[index_good])
        error_abs_hard.extend(error_abs[index_hard])
        for i, model_name in enumerate(model_names):
            unc = read(unc_path + img_path.replace('.png','/') + model_name)
            unc_good[i].extend(unc[index_good])
            unc_hard[i].extend(unc[index_hard])
        # break

    error_abs_standard_good = np.array(error_abs_good) / np.array(unc_good)
    error_abs_standard_hard = np.array(error_abs_hard) / np.array(unc_hard)

    return error_abs_standard_good, error_abs_standard_hard


def filter_error_abs_standard(error_abs_standard, x):
    """
    select errors in range(min(x), max(x))
    """
    dimension = error_abs_standard.shape[0] if error_abs_standard.ndim > 1 else 1
    error_abs_standard_filtered = []
    for i in range(dimension):
        condi = np.logical_and(error_abs_standard[i] < max(x), error_abs_standard[i] > min(x))
        error_tmp = error_abs_standard[i][condi]
        error_abs_standard_filtered.append(error_tmp)
    return error_abs_standard_filtered


def hist_sub(error_abs_standard, ax, name='', xname='', yname='', binwidth=1):
    MIN, MAX = np.min(error_abs_standard), np.max(error_abs_standard)
    if MAX > 200: MAX = 200
    ax.hist(error_abs_standard, bins=range(int(MIN), int(MAX) + binwidth, binwidth))
    ax.set_ylabel(xname)
    ax.set_xlabel(yname)
    ax.set_title(name)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_hist(error_abs_standard_good, error_abs_standard_hard, save_name='hist.svg', binwidth=1):
    """
    plot histogram of standardized absolute error
    """
    plt.style.use('seaborn-whitegrid')
    plt.rc('font', family='Times New Roman')
    dimension = error_abs_standard_good.shape[0] if error_abs_standard_good.ndim > 1 else 1
    fig, ax = plt.subplots(nrows=dimension, ncols=2, figsize=(8, 6))
    for i in range(dimension):
        hist_sub(error_abs_standard_good[i], ax[i][0], binwidth=binwidth)
        ax[i][0].set_ylabel('Count [-]')
    for i in range(dimension):
        hist_sub(error_abs_standard_hard[i], ax[i][1], binwidth=binwidth)
    ax[dimension - 1][0].set_xlabel('Abs Error Standard [pixel]')
    ax[dimension - 1][1].set_xlabel('Abs Error Standard [pixel]')
    ax[0][0].set_title('Good')
    ax[0][1].set_title('Hard')

    fig.tight_layout()
    plt.savefig(save_name)
    plt.show()


def reliability_diagram_sub(error_abs_standard, cum_prob_half, x, ax, name='', legends='', colors=[], markers=[], xname='', yname=''):
    """
    error_abs_standard: list
    """
    y = [[] for _ in error_abs_standard]
    for idx in range(len(cum_prob_half)):
        for j in range(len(error_abs_standard)):
            error_abs_standard_tmp = np.array(error_abs_standard[j])
            cum_prob_cur = len(error_abs_standard_tmp[error_abs_standard_tmp >= x[idx]]) / len(error_abs_standard_tmp)
            y[j].append(cum_prob_cur)

    cum_prob = [prob * 2 for prob in cum_prob_half]
    ax.plot([0.0, 1.0], [0.0, 1.0], '--', label='Optimal', color=colors[0])
    for k in range(len(error_abs_standard)):
        ax.plot(cum_prob, y[k], label=legends[k], color=colors[k+1], marker=markers[k], markerfacecolor="None", markersize=8)
    ax.legend()
    ax.set_title(name)
    ax.set_aspect('equal')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)


def gen_plots(error_abs_standard_good, error_abs_standard_hard, legends, filter=False):
    cum_prob_half = [0.00001] + [x * 0.025 for x in range(1, 21)]
    x = [-scipy.stats.laplace(0, np.sqrt(2)/2).ppf(c) for c in cum_prob_half ]
    # x = [-scipy.stats.norm(0, 1).ppf(c) for c in cum_prob_half]
    error_abs_standard_good_filtered = filter_error_abs_standard(error_abs_standard_good, x) if filter else error_abs_standard_good
    error_abs_standard_hard_filtered = filter_error_abs_standard(error_abs_standard_hard, x) if filter else error_abs_standard_hard
    colors = ['blue','red','green','black']
    markers = ['*', 'o', '^']
    plt.style.use('seaborn-whitegrid')
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex='col', sharey='row')
    reliability_diagram_sub(error_abs_standard_good_filtered, cum_prob_half, x, ax[0], name='Good', legends=legends,
                            colors=colors, markers=markers, xname='Expected Cumulative Probability')
    reliability_diagram_sub(error_abs_standard_hard_filtered, cum_prob_half, x, ax[1], name='Hard', legends=legends,
                            colors=colors, markers=markers, xname='Expected Cumulative Probability')
    ax[0].set_ylabel('Observed Cumulative Probability')
    fig.tight_layout()
    plt.savefig('reliability_K15_MC-CNN_laplace.svg')
    plt.show()


def main(indicator_path, gt_path, est_path, unc_path, model_names, legends):
    error_abs_standard_good, error_abs_standard_hard = load_standard_abs_error(indicator_path, gt_path, est_path, unc_path, model_names)
    gen_plots(error_abs_standard_good, error_abs_standard_hard, legends)
    # plot_hist(error_abs_standard_good, error_abs_standard_hard, save_name='error_abs_standard_wo_zero.svg', binwidth=1)


if __name__ == '__main__':
    dataset = 'K15'
    disp_method = 'MC-CNN'
    dataset_name = 'middlebury-v3' if dataset == 'M3' else 'kitti-2015'
    indicator_path = '/media/zeyun/ZEYUN/MA/' + dataset_name + '/mask_indicator/'
    gt_path = '/media/zeyun/ZEYUN/MA/' + dataset_name + '/disp_gt_occ/'
    est_path = '/media/zeyun/ZEYUN/MA/' + dataset_name + '/est_MC-CNN/'
    unc_path = '/home/zeyun/Projects/CVA/results/' + dataset_name + '/'

    model_names = ['Probabilistic_paper', 'Mixed_Uniform_paper', 'Laplacian_Uniform_paper']
    model_names = [name + '_MC-CNN' for name in model_names]
    legends = ['Laplacian', 'Geometry', 'Mixture']
    main(indicator_path, gt_path, est_path, unc_path, model_names, legends)