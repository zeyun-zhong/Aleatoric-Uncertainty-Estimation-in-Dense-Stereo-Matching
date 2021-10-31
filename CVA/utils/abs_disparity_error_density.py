import numpy as np
import matplotlib.pyplot as plt
from image_io import read
import os
import pickle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

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


class AbsDisparityError:
    def __init__(self, gt_path, est_path, indicator_path):
        self.gt_path = gt_path
        self.est_path = est_path
        self.indicator_path = indicator_path
        self.error_abs_all, self.error_abs_good, self.error_abs_hard = self.load_abs_error()

    def load_abs_error(self):
        file_list = sorted(os.listdir(self.est_path))
        error_abs_list, error_abs_good, error_abs_hard = [], [], []

        for img_path in file_list:
            gt = read(os.path.join(self.gt_path, img_path.replace('pfm', 'png'))).astype(float)
            est = read(os.path.join(self.est_path, img_path)).astype(float)
            error_abs = np.abs(gt - est)
            error_abs_list.extend(error_abs[gt != 0])

            if self.indicator_path:
                indi = read(os.path.join(self.indicator_path, img_path.replace('pfm', 'png')))
                index_good = np.where(indi == 1)
                index_hard = np.where(np.logical_and(indi == 0, gt != 0))
                # data of good region
                error_abs_good.extend(error_abs[index_good])
                # data of hard region
                error_abs_hard.extend(error_abs[index_hard])
        return np.array(error_abs_list), np.array(error_abs_good), np.array(error_abs_hard)

    def calc_density(self):
        density_all = self.extract_abs_error(self.error_abs_all)
        density_good, density_hard = None, None
        if len(self.error_abs_good) > 0:
            density_good = self.extract_abs_error(self.error_abs_good)
        if len(self.error_abs_hard) > 0:
            density_hard = self.extract_abs_error(self.error_abs_hard)
        return np.array(density_all), np.array(density_good), np.array(density_hard)

    def save_density(self, save_path):
        density_all, density_good, density_hard = self.calc_density()
        print(density_all[:10], '\n', density_good[:10], '\n', density_hard[:10])
        self.save_data(density_all, save_path, 'density_all')
        self.save_data(density_good, save_path, 'density_good')
        self.save_data(density_hard, save_path, 'density_hard')

    @staticmethod
    def extract_abs_error(abs_error):
        '''
        calculate density for every abs error value
        :param abs_error: abs disparity errors
        :return: lists contains density value
        '''
        assert len(abs_error) > 0, 'abs error should not be empty'
        abs_error = np.sort(abs_error)
        x = range(0, int(abs_error[-1]) + 1)
        density_cum = [np.argmax(abs_error > i) / len(abs_error) if i < abs_error[-1] else 1 for i in x]
        density = [density_cum[i] - density_cum[i - 1] if i >= 1 else density_cum[i] for i in range(len(density_cum))]
        return density

    @staticmethod
    def save_data(data, save_path, save_name):
        with open(os.path.join(save_path, save_name), 'wb') as handle:
            pickle.dump(data, handle)


class PlotDensity:
    def __init__(self, density_path):
        self.density_all = self.read_density(density_path, 'density_all') * 100
        self.density_good = self.read_density(density_path, 'density_good') * 100
        self.density_hard = self.read_density(density_path, 'density_hard') * 100

        self.x_all = np.arange(len(self.density_all))
        self.x_good = np.arange(len(self.density_good))
        self.x_hard = np.arange(len(self.density_hard))

        eps = 1e-8
        self.inverse_density_all = 1 / (self.density_all + eps)
        self.inverse_density_good = 1 / (self.density_good + eps)
        self.inverse_density_hard = 1 / (self.density_hard + eps)

        plt.style.use('seaborn-whitegrid')
        plt.rc('font', family='Times New Roman')

    @staticmethod
    def read_density(density_path, name):
        with open(os.path.join(density_path, name), 'rb') as handle:
            density = pickle.load(handle)
        return density

    def plot_density(self):
        plt.plot(self.x_all, self.density_all)
        plt.xlim([0, 120])
        plt.xlabel('Abs Disparity Error [pixel]')
        plt.ylabel('Density [%]')
        plt.show()

    def plot_density_and_inverse_density(self, show_filtered=False):
        fig, ax1 = plt.subplots()
        color1 = 'tab:blue'
        ax1.set_xlabel('Abs Disparity Error [pixel]')
        ax1.set_ylabel('Density [%]', color=color1)
        ax1.plot(self.x_all, self.density_all, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlim([0, 120])
        ax1.set_ylim([0, 60])

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color2 = 'tab:red'
        ax2.set_ylabel('Inverse Density', color=color2)  # we already handled the x-label with ax1
        ax2.plot(self.x_all, self.inverse_density_all, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim([0, 10000000])

        if show_filtered:
            density_filter = gaussian_filter1d(self.density_all, 1)
            inverse_density_filter = 1 / (density_filter )

            ax1.plot(self.x_all, density_filter, color='m')
            ax2.plot(self.x_all, inverse_density_filter, color='g')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def plot_weighted_density(self):
        weight = 0.2337 * np.square(self.x_all) + 18.78 * self.x_all - 459.36
        weight[weight <= 0] = 0.1

        weighted_density = weight * self.density_all
        weighted_density_norm = (weighted_density / np.sum(weighted_density)) * 100

        plt.plot(self.x_all, self.density_all, label='density')
        plt.plot(self.x_all, weighted_density, label='weighted density')
        plt.plot(self.x_all, weighted_density_norm, label='weighted density norm')
        plt.xlabel('Abs Disparity Error [Pixel]')
        plt.ylabel('Density [%]')
        plt.xlim([0, 120])
        plt.ylim([0, 60])
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_filtered_inverse_density(self, gauss_std=1):
        # Todo needs to be updated with region and order
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex='col', sharey='row')
        self.plot_filtered_inverse_density_sub(self.x_all, self.inverse_density_all, ax[0], xname='Abs Error [pixel]',
                                               title=r'All ($\sigma$ = %d)' % gauss_std, yname='Inverse Density', gauss_std=gauss_std)
        self.plot_filtered_inverse_density_sub(self.x_good, self.inverse_density_good, ax[1], xname='Abs Error [pixel]',
                                               title=r'Good ($\sigma$ = %d)' % gauss_std, gauss_std=gauss_std)
        self.plot_filtered_inverse_density_sub(self.x_hard, self.inverse_density_hard, ax[2], xname='Abs Error [pixel]',
                                               title=r'Hard ($\sigma$ = %d)' % gauss_std, gauss_std=gauss_std)
        fig.tight_layout()
        plt.show()

    def plot_filtered_inverse_density_multi_std(self, gauss_stds, region='', order='inverse_first'):
        '''plot density of a specific region w.r.t. different gaussian stds'''
        assert type(gauss_stds) is list and len(gauss_stds) > 0, 'unallowed format of gaussian stds'
        assert order in ['inverse_first', 'filter_first'], 'unallowed order'

        if order == 'inverse_first':
            density, x = self.get_inverse_density_and_x(region)
        else:
            density, x = self.get_density_and_x(region)

        fig, ax = plt.subplots(nrows=1, ncols=len(gauss_stds), figsize=(4*len(gauss_stds), 4), sharex='col', sharey='row')
        for i, std in enumerate(gauss_stds):
            self.plot_filtered_inverse_density_sub(x, density, order, ax[i], xname='Abs Error [pixel]',
                                                   title=r'%s ($\sigma$ = %d)' % (region, std), gauss_std=std)
        ax[0].set_ylabel('Inverse Density')
        fig.tight_layout()
        plt.show()

    def get_inverse_density_and_x(self, region=''):
        if region in ['', 'all', 'All', None]:
            return self.inverse_density_all, self.x_all
        elif region in ['good', 'Good']:
            return self.inverse_density_good, self.x_good
        elif region in ['hard', 'Hard']:
            return self.inverse_density_hard, self.x_hard
        else:
            raise Exception('Unknown region name: %s' % region)

    def get_density_and_x(self, region=''):
        if region in ['', 'all', 'All', None]:
            return self.density_all, self.x_all
        elif region in ['good', 'Good']:
            return self.density_good, self.x_good
        elif region in ['hard', 'Hard']:
            return self.density_hard, self.x_hard
        else:
            raise Exception('Unknown region name: %s' % region)

    @staticmethod
    def plot_filtered_inverse_density_sub(x, y, order, ax, xname='', yname='', title='', gauss_std=1, label1='Inverse Density', label2='Filtered Inverse Density'):
        '''
        filter first: y == density
        inverse first: y == inverse density
        '''
        y_filtered = gaussian_filter1d(y, gauss_std)
        # y_filtered = medfilt(y, kernel_size=gauss_std)

        if order == 'filter_first':
            y = 1 / (y + 1e-8)
            y_filtered = 1 / (y_filtered + 1e-8)

        ax.plot(x, y, label=label1)
        ax.plot(x, y_filtered, label=label2)
        ax.legend()
        ax.set_xlim([0, x[-1]+1])
        ax.set_ylim([0, 10])
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(title)


if __name__ == '__main__':
    base_path = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/'
    cv_method = 'Census-BM'

    indicator_path = base_path + 'mask_indicator/'
    gt_path = base_path + 'disp_occ/'
    est_path = base_path + '{}/disp_est/'.format(cv_method)
    save_path = base_path + '{}/'.format(cv_method)

    PD = PlotDensity(save_path)
    PD.plot_filtered_inverse_density_multi_std(gauss_stds=[1,2,3,4], region='All', order='filter_first')

    # save density as pickle
    # ADE = AbsDisparityError(gt_path, est_path, indicator_path)
    # ADE.save_density(save_path)

