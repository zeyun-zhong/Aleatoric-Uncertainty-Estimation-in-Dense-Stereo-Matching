from abs_disparity_error_density import load_abs_error, extract_abs_error
import numpy as np
from abc import ABC, abstractmethod
import pickle
from matplotlib import pyplot as plt

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


class LinearRegression(ABC):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps1=1e-4, eps2=0.1, inverse_density_max=4000):
        """
        @param gt_path:
        @param est_path:
        @param indicator_path:
        @param save_path:
        @param eps1: coefficient for weighting function of density fitting
        @param eps2: coefficient for weighting function of inverse density fitting
        @param inverse_density_max: maximum of inverse density, for the clip function
        """
        self.gt_path = gt_path
        self.est_path = est_path
        self.indicator_path = indicator_path
        self.save_path = save_path
        self.eps1 = eps1
        self.eps2 = eps2
        self.inverse_density_max = inverse_density_max
        self.coefficients = None
        self.fitted_func = None
        self.weighting_func = None
        # inputs, outputs = self.prepare_data()
        with open('inputs', 'rb') as handle:
            self.inputs_raw = pickle.load(handle)
        with open('outputs', 'rb') as handle:
            self.outputs_raw = pickle.load(handle)
        self.inputs, self.outputs = self.preprocess_data()

    def prepare_data(self):
        """
        :return: disparity error and the corresponding density
        """
        error_abs, error_abs_good, error_abs_hard = load_abs_error(self.indicator_path, self.gt_path, self.est_path)
        inputs = range(0, int(np.max(error_abs)) + 1)
        density_cum = extract_abs_error(error_abs, inputs)
        outputs = [density_cum[i] - density_cum[i - 1] if i >= 1 else density_cum[i] for i in range(len(density_cum))]
        return np.array(inputs), np.array(outputs)

    @abstractmethod
    def preprocess_data(self):
        """
        @return: preprocessed data
        """
        raise NotImplementedError

    def gauss_markov(self, design_matrix, observation):
        coefficients = np.dot(np.dot(np.linalg.inv(np.dot(design_matrix.T, design_matrix)), design_matrix.T), observation)
        return coefficients

    def fit(self):
        """
        Linear regression using least square error
        outputs = design_matrix * inputs + v --> min(v.T * v)
        :return: coefficients
        """
        design_matrix = np.ones((len(self.inputs), 2))
        design_matrix[:, 0] = self.inputs
        self.coefficients = self.gauss_markov(design_matrix, self.outputs)
        self.fitted_func = lambda x: self.coefficients[0] * x + self.coefficients[1]

    @abstractmethod
    def get_weighting_func(self):
        """
        considers the reflected fitted model as the weight function
        :return: coefficients of weighting function
        """
        raise NotImplementedError

    def plot(self):
        plt.style.use('seaborn-whitegrid')
        plt.rc('font', family='Times New Roman')
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex='col', sharey='row')
        ax[0].plot(self.inputs_raw, self.outputs, label='Abs Error Density')
        ax[0].plot(self.inputs_raw, self.fitted_func(self.inputs), label='Fitted Model')
        ax[0].plot(self.inputs_raw, self.weighting_func(self.inputs), label='Weighting Model')
        ax[0].legend()
        ax[0].set_xlabel('Abs Disparity Error [pixel]')
        ax[0].set_ylabel('Density [%]')
        ax[0].set_xlim([0, 120])

        weighted_density = self.outputs * self.weighting_func(self.inputs)
        weighted_density_norm = (weighted_density / sum(weighted_density)) * 100

        ax[1].plot(self.inputs_raw, weighted_density, label='Weighted Density')
        ax[1].plot(self.inputs_raw, weighted_density_norm, label='Weighted Density Norm')
        ax[1].legend()
        ax[1].set_xlabel('Abs Disparity Error [pixel]')
        ax[1].set_ylim([0, 10])
        ax[1].set_xlim([0, 120])
        fig.tight_layout()
        if self.save_path:
            plt.savefig(self.save_path)
        plt.show()


class SimpleLinearRegression(LinearRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=1e-4):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path, eps1=eps)

    def preprocess_data(self):
        return self.inputs_raw, self.outputs_raw * 100

    def get_weighting_func(self):
        def f(x):
            tmp = self.fitted_func(x)
            tmp[tmp <= 0] = self.eps1
            return 1 / tmp
        self.weighting_func = f


class LogLinearRegression(LinearRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=1e-4):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path, eps1=eps)

    def preprocess_data(self):
        inputs = np.log(self.inputs_raw + 1e-8)
        return inputs, self.outputs_raw * 100

    def get_weighting_func(self):
        def f(x):
            tmp = self.fitted_func(x)
            tmp[tmp <= 0] = self.eps1
            return 1 / tmp
        self.weighting_func = f


class InverseDensityRegression(LinearRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=0.1, inverse_density_max=4000):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path,
                         eps2=eps, inverse_density_max=inverse_density_max)

    @abstractmethod
    def preprocess_data(self):
        raise NotImplementedError

    def get_weighting_func(self):
        # self.weighting_func = self.fitted_func
        def f(x):
            tmp = self.fitted_func(x)
            tmp[tmp <= 0] = self.eps2
            return tmp
        self.weighting_func = f

    def plot(self):
        plt.style.use('seaborn-whitegrid')
        plt.rc('font', family='Times New Roman')
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex='col', sharey='none')

        ax[0].plot(self.inputs_raw, self.outputs_raw * 100, label='Abs Error Density')
        ax[0].set_ylabel('Density [%]', color='tab:blue')
        ax[0].set_xlabel('Abs Disparity Error [pixel]')
        ax[0].set_ylabel('Density [%]')
        ax[0].set_xlim([0, 120])
        ax[0].set_ylim([0, 10])

        ax2 = ax[0].twinx()
        ax2.plot(self.inputs_raw, self.outputs, label='Inverse Density', color='tab:orange')
        ax2.plot(self.inputs_raw, self.weighting_func(self.inputs), label='Weighting Model', color='tab:green')
        ax2.set_ylim([0, 1e4])
        ax2.set_ylabel('Inverse Density')
        ax2.legend()

        print('weights for zero error: ', self.weighting_func(self.inputs)[0])

        weighted_density = self.outputs_raw * 100 * self.weighting_func(self.inputs)
        weighted_density_norm = (weighted_density / sum(weighted_density)) * 100

        ax[1].plot(self.inputs_raw, weighted_density, label='Weighted Density')
        ax[1].plot(self.inputs_raw, weighted_density_norm, label='Weighted Density Norm')
        ax[1].legend()
        ax[1].set_xlabel('Abs Disparity Error [pixel]')
        ax[1].set_xlim([0, 120])
        ax[1].set_ylim([0, 10])
        ax[1].set_ylabel('Density [%]')
        fig.tight_layout()
        if self.save_path:
            plt.savefig(self.save_path)
        plt.show()


class SimpleInverseRegression(InverseDensityRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=0.1, inverse_density_max=4000):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path,
                         eps=eps, inverse_density_max=inverse_density_max)

    def preprocess_data(self):
        inputs = self.inputs_raw
        outputs = 1 / (self.outputs_raw * 100 + 1e-8)
        outputs = np.clip(outputs, a_min=None, a_max=self.inverse_density_max)
        return inputs, outputs


class ExpInverseRegressionOld(InverseDensityRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=0.1, inverse_density_max=4000):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path,
                         eps=eps, inverse_density_max=inverse_density_max)

    def preprocess_data(self):
        inputs = np.exp(self.inputs_raw)
        outputs = 1 / (self.outputs_raw * 100 + 1e-8)
        outputs = np.clip(outputs, a_min=None, a_max=self.inverse_density_max)
        return inputs, outputs


class ExpInverseRegression(InverseDensityRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=0.1, inverse_density_max=4000):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path,
                         eps=eps, inverse_density_max=inverse_density_max)

    def preprocess_data(self):
        outputs = 1 / (self.outputs_raw * 100 + 1e-8)
        outputs = np.clip(outputs, a_min=None, a_max=self.inverse_density_max)
        return self.inputs_raw, outputs

    def fit(self):
        design_matrix = np.ones((len(self.inputs), 2))
        design_matrix[:, 0] = self.inputs
        self.coefficients = self.gauss_markov(design_matrix, np.log(self.outputs))
        self.fitted_func = lambda x: np.exp(self.coefficients[0] * x + self.coefficients[1])

    def get_weighting_func(self):
        self.weighting_func = self.fitted_func


class SpecialExpInverseRegression(InverseDensityRegression):
    # y = a * exp(b * x)
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=0.1, inverse_density_max=4000):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path,
                         eps=eps, inverse_density_max=inverse_density_max)

    def preprocess_data(self):
        outputs = 1 / (self.outputs_raw * 100 + 1e-8)
        outputs = np.clip(outputs, a_min=None, a_max=self.inverse_density_max)
        return self.inputs_raw, outputs

    def fit(self):
        design_matrix = np.ones((len(self.inputs), 1))
        design_matrix[:, 0] = self.inputs
        observation = np.log(self.outputs / self.eps2)
        self.coefficients = self.gauss_markov(design_matrix, observation)
        self.fitted_func = lambda x: np.exp(np.log(self.eps2) + self.coefficients * x)

    def get_weighting_func(self):
        self.weighting_func = self.fitted_func


class QuadraticInverseRegression(InverseDensityRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=0.1, inverse_density_max=4000):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path,
                         eps=eps, inverse_density_max=inverse_density_max)

    def preprocess_data(self):
        inputs = self.inputs_raw ** 2
        outputs = 1 / (self.outputs_raw * 100 + 1e-8)
        outputs = np.clip(outputs, a_min=None, a_max=self.inverse_density_max)
        return inputs, outputs


class PolynomialInverseRegression(InverseDensityRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=0.1, inverse_density_max=4000, degree=2):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path,
                         eps=eps, inverse_density_max=inverse_density_max)
        self.degree = degree

    def preprocess_data(self):
        outputs = 1 / (self.outputs_raw * 100 + 1e-8)
        outputs = np.clip(outputs, a_min=None, a_max=self.inverse_density_max)
        return self.inputs_raw, outputs

    def fit(self):
        design_matrix = np.ones((len(self.inputs), self.degree + 1))
        for i in range(self.degree):
            design_matrix[:, i] = self.inputs ** (self.degree - i)

        self.coefficients = self.gauss_markov(design_matrix, self.outputs)
        print('coefficients: %s' % self.coefficients)

        def f(x):
            res = np.zeros(x.shape)
            for i in range(self.degree):
                tmp = x ** (self.degree - i)
                res += self.coefficients[i] * tmp
            return res + self.coefficients[-1]
        self.fitted_func = f


class SpecialQuadraticInverseRegression(InverseDensityRegression):
    def __init__(self, gt_path, est_path, indicator_path=None, save_path=None, eps=0.1, inverse_density_max=4000):
        super().__init__(gt_path=gt_path, est_path=est_path, indicator_path=indicator_path, save_path=save_path,
                         eps=eps, inverse_density_max=inverse_density_max)

    def preprocess_data(self):
        outputs = 1 / (self.outputs_raw * 100 + 1e-8)
        outputs = np.clip(outputs, a_min=None, a_max=self.inverse_density_max)
        return self.inputs_raw, outputs

    def fit(self):
        design_matrix = np.ones((len(self.inputs), 2))
        for i in range(2):
            design_matrix[:, i] = self.inputs ** (2 - i)

        self.coefficients = self.gauss_markov(design_matrix, self.outputs)
        print('coefficients: %s' % self.coefficients)
        self.fitted_func = lambda x: self.coefficients[0] * (x ** 2) + self.coefficients[1] * x


if __name__ == "__main__":
    # base_path = '/home/zeyun/Projects/CVA/stimuli/kitti-2012/training/'
    # indicator_path = None
    # gt_path = base_path + 'disp_occ/'
    # est_path = base_path + 'GC-Net/disp_est/'
    #
    # regressor = PolynomialInverseRegression(gt_path=None, est_path=None, indicator_path=None, save_path=None, eps=0.1,
    #                                         inverse_density_max=4000, degree=2)
    # # regressor = SpecialQuadraticInverseRegression(gt_path=None, est_path=None, indicator_path=None, save_path=None, eps=1,
    # #                                        inverse_density_max=4000)
    regressor = SpecialExpInverseRegression(gt_path=None, est_path=None, indicator_path=None, save_path=None, eps=0.1,
                                     inverse_density_max=10000)
    regressor.fit()
    regressor.get_weighting_func()
    regressor.plot()

    # PlotDensity().plot_weighted_density()