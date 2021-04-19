import matplotlib.pyplot as plt
plt.rc('font', family='serif')
import numpy as np
import image_io
import os
import pickle
import sys
from tqdm import tqdm

sys.path.insert(1, '../CVA-Net')
import cost_volume


class GMM():
    def __init__(self, data_path, results_path, model_name, index_range, m, save_path, save_fig=False):
        self.data_path = data_path
        self.results_path = results_path
        self.model_name = model_name
        self.index_range = index_range
        self.m = m
        self.save_path = os.path.join(save_path, self.model_name)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_fig = save_fig

    def plot(self, x, fig_size=(6, 4)):
        for index in tqdm(self.index_range):
            mask_indi = self.load_indicator(index)
            disp_gt = self.load_gt(index)
            disp_est = self.load_est(index)
            cv = self.load_cost_volume(index)
            # phi_map: 3 dimensional
            phi_map, mu_map, var_map = self.load_gmm_comp(index)
            for i in range(2):
                suffix = '_good' if i else '_other'
                region_index_all = np.where(np.logical_and(mask_indi == i, disp_gt != 0))
                region_index = (region_index_all[0][0:self.m], region_index_all[1][0:self.m])
                phi_region, mu_region, var_region = phi_map[region_index], mu_map[region_index], var_map[region_index]
                disp_gt_region, disp_est_region = disp_gt[region_index], disp_est[region_index]
                cv_region = cv[region_index]
                for j in range(self.m):
                    # shape should be 1 x 3
                    phi, mu, var = phi_region[j], mu_region[j], var_region[j]
                    gt, est = disp_gt_region[j], disp_est_region[j]
                    cv_cur = cv_region[j]

                    mean, variance = self.get_gmm_mix_mean_and_var(phi, mu, var)

                    mix_prob = self.get_gmm_mix_prob(x, phi, mu, var)
                    # self.plot_gmm(fig_size, x, gt, est, mean, variance, mix_prob, j, index, suffix)

                    comp_prob = self.get_gmm_comp_prob(x, phi, mu, var)
                    # self.plot_gmm_comp(fig_size, x, gt, est, phi, mu, var, comp_prob, j, index, suffix)
                    self.subplot_gmm(fig_size, x, gt, est, cv_cur, phi, mu, var, mean, variance, comp_prob, mix_prob, j, index, suffix)

    def subplot_gmm(self, fig_size, x, gt, est, cv_cur, phi, mu, var, mean, variance, comp_prob, mix_prob, j, index, suffix):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=fig_size)
        fig.suptitle('GMM Probability ' + str(j) + suffix.replace('_', ' '))

        ax0.plot(cv_cur)
        ax0.set_ylabel('cost')
        # ax0.grid(True, linestyle='--')

        for i in range(comp_prob.shape[1]):
            label = '$\phi$: {:5.3f} $\mu$: {:6.3f} $\sigma^2$: {:6.3f}'.format(phi[i], mu[i], var[i])
            ax1.plot(x, comp_prob[:,i], label=label)
        ax1.legend(prop={'family': 'monospace'})
        ax1.set_xlim([0, 255])
        ax1.set_ylim(bottom=0)
        ax1.set_ylabel('probability')
        # ax1.grid(True, linestyle='--')

        ax2.plot(x, mix_prob, label=r"mean: {:6.3f} $\sigma^2$: {:6.3f}".format(mean, variance))
        ax2.plot(est, 0, 'bo', label='$d_{est}$: %3d' % est)
        # ax2.text(est, 0, '$d_{est}$')
        ax2.plot(gt, 0, 'r*', label='$d_{gt}$: %4d' % gt)
        # ax2.text(gt, 0, '$d_{gt}$')
        ax2.legend()
        ax2.set_xlabel('disparity')
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel('probability')
        # ax2.grid(True, linestyle='--')
        plt.tight_layout()

        if self.save_fig:
            fig_name = self.model_name + '_' + str(j) + suffix + '.svg'
            self.save_img(index, fig_name)
        # plt.show()
        plt.close()

    def plot_gmm(self, fig_size, x, gt, est, mean, variance, prob, j, index, suffix):
        plt.figure(figsize=fig_size)
        plt.plot(x, prob, label=r"mean: {:6.3f} $\sigma^2$: {:6.3f}".format(mean, variance))
        plt.plot(est, 0, 'bo')
        plt.text(est, 0, '$d_{est}$')
        plt.plot(gt, 0, 'r*')
        plt.text(gt, 0, '$d_{gt}$')
        plt.tight_layout()
        plt.legend()
        title = 'GMM Mixture Probability ' + str(j) + suffix.replace('_', ' ')
        plt.title(title)
        plt.xlabel('disparity')
        plt.ylabel('probability')
        plt.xlim([0, 255])
        plt.ylim(bottom=0)
        if self.save_fig:
            fig_name = self.model_name + '_mix_' + str(j) + suffix + '.svg'
            self.save_img(index, fig_name)
        # plt.show()
        plt.close()

    def plot_gmm_comp(self, fig_size, x, gt, est, phi, mu, var, prob, j, index, suffix):
        plt.figure(figsize=fig_size)
        for i in range(prob.shape[1]):
            label = '$\phi$: {:5.3f} $\mu$: {:6.3f} $\sigma^2$: {:6.3f}'.format(phi[i], mu[i], var[i])
            plt.plot(x, prob[:,i], label=label)
        plt.plot(est, 0, 'bo')
        plt.text(est, 0, '$d_{est}$')
        plt.plot(gt, 0, 'r*')
        plt.text(gt, 0, '$d_{gt}$')
        plt.tight_layout()
        title = 'GMM Components Probability ' + str(j) + suffix.replace('_', ' ')
        plt.title(title)
        plt.legend(prop={'family': 'monospace'})
        plt.xlabel('disparity')
        plt.ylabel('probability')
        plt.xlim([0, 255])
        plt.ylim(bottom=0)
        if self.save_fig:
            fig_name = self.model_name + '_comp_' + str(j) + suffix + '.svg'
            self.save_img(index, fig_name)
        # plt.show()
        plt.close()

    def get_gmm_mix_mean_and_var(self, phi, mu, var):
        mean = np.sum(phi * mu, axis=-1)
        variance = np.sum(phi * (var + np.square(mu)), axis=-1) - np.square(mean)
        return mean, variance

    def get_gmm_comp_prob(self, x, phi, mu, var):
        """
        return a 2d probability array
        """
        k = len(mu)
        length = x.shape[0]
        prob = np.zeros((length, k))
        for i in range(k):
            phi_cur, mu_cur, var_cur = phi[i], mu[i], var[i]
            # broadcast x: 1 x n; phi, mu, var: 1, -> 1 x n
            phi_cur = np.repeat(phi_cur, length)
            mu_cur = np.repeat(mu_cur, length)
            var_cur = np.repeat(var_cur, length)
            eps = 1e-7
            prob_cur = 1 / (np.sqrt(2 * np.pi * var_cur) + eps) * np.exp(-np.square(x - mu_cur) / (2 * var_cur + eps))
            prob[:, i] = prob_cur
        return prob

    def get_gmm_mix_prob(self, x, phi, mu, var):
        # broadcast x: 1 x n -> n x 3; phi, mu, var: 1 x 3 -> n x 3
        k = len(mu)
        length = x.shape[0]
        x = np.repeat(x.reshape(length, 1), k, axis=1)
        phi = np.repeat(phi.reshape(1, k), length, axis=0)
        mu = np.repeat(mu.reshape(1, k), length, axis=0)
        var = np.repeat(var.reshape(1, k), length, axis=0)

        eps = 1e-7
        prob = phi / (np.sqrt(2 * np.pi * var) + eps) * np.exp(-np.square(x - mu) / (2 * var + eps))
        prob_sum = np.sum(prob, axis=1)
        return prob_sum.reshape(length)

    def save_img(self, index, img_name):
        path = os.path.join(self.save_path, str(index).rjust(6, '0') + '_10')
        if not os.path.exists(path):
            os.mkdir(path)
        img_path = os.path.join(path, img_name)
        plt.savefig(img_path, bbox_inches='tight')

    def load_indicator(self, index):
        path = self.data_path + 'mask_indicator/' + str(index).rjust(6, '0') + '_10.png'
        mask_indicator = image_io.read(path)
        return mask_indicator

    def load_gt(self, index):
        path = self.data_path + 'disp_occ/' + str(index).rjust(6, '0') + '_10.png'
        disp_gt = image_io.read(path)
        return disp_gt

    def load_est(self, index):
        path = self.results_path + str(index).rjust(6, '0') + '_10' + '/DispMap.png'
        disp_est = image_io.read(path)
        return disp_est

    def load_gmm_comp(self, index):
        path = self.results_path + str(index).rjust(6, '0') + '_10' + '/' + self.model_name
        phi = np.load(path + '_phi.npy')
        mu = np.load(path + '_mu.npy')
        var = np.load(path + '_sigma.npy')
        return phi, mu, var

    def load_cost_volume(self, index):
        path = self.data_path + 'cost_volumes/' + str(index).rjust(6, '0') + '_10'
        with open(path, 'rb') as file:
            cv = pickle.load(file)
        return cv.get_data()


if __name__ == '__main__':
    data_path = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/'
    results_path = '/home/zeyun/Projects/CVA/results/kitti-2012/'
    save_path = '/home/zeyun/Projects/CVA/GMM_plots/'

    # may need to be changed
    model_name = 'CVA-Net_GMM_gamma_1.0'
    index_range = np.arange(5, 15)
    m = 10
    save_fig = True
    x = np.arange(0, 255, 1)

    gmm = GMM(data_path, results_path, model_name, index_range, m, save_path, save_fig)
    gmm.plot(x, fig_size=(6, 6))