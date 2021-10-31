import os
from data_generators import DataSample
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
import pickle
import params

parameter = params.Params()


class WeightLoss:
    def __init__(self, CVMethod, using_lookup=False, gauss_std=1):
        self.CVMethod = CVMethod
        self.using_lookup = using_lookup
        self.gauss_std = gauss_std

    def weighting_func(self, loss, abs_error, region=''):
        density = self.load_density(region)
        if self.using_lookup:
            if max(density) < 1:
                density *= 100
            density_filtered = gaussian_filter1d(density, self.gauss_std)
            table = 1 / (density_filtered + 1e-8)
            table[table < 5.0] = 5.0 # test
            weight = tf.cast(tf.gather(table, tf.cast(abs_error, dtype=tf.int32)), dtype=tf.float32)
        else:
            # Todo weighting func should not be hard coded
            weight = 0.2337 * tf.square(abs_error) + 18.78 * abs_error - 459.36
            weight = tf.where(weight <= 0, 0.1, weight) # the weight should be larger than zero

        weighted_loss = loss * weight
        return weighted_loss

    def load_density(self, region=''):
        '''
        load density in the corresponding region
        :param region: region name
        :return: density
        '''
        density_path = self.CVMethod.cv_path.replace('cost_volumes/', '')
        if region in ['', 'all', 'All', None]:
            return self.load_data(density_path, 'density_all')
        elif region in ['good', 'Good']:
            return self.load_data(density_path, 'density_good')
        elif region in ['hard', 'Hard']:
            return self.load_data(density_path, 'density_hard')
        else:
            raise Exception('Unknown region name: %s' % region)

    @staticmethod
    def load_data(path, name):
        with open(os.path.join(path, name), 'rb') as handle:
            data = pickle.load(handle)
        return data


class Dataset:
    def __init__(self, dataset_name, root_path, data_path_suffix=parameter.data_path_suffix):
        assert dataset_name in ['K12', 'K15', 'M3', 'Sceneflow'], 'Unknown Dataset: {}'.format(dataset_name)
        self.dataset_name = dataset_name
        self.data_path = os.path.join(root_path, data_path_suffix[dataset_name])
        self.disp_gt_path = os.path.join(self.data_path, 'disp_occ/')
        self.indic_path = os.path.join(self.data_path, 'mask_indicator/')


class CVMethod:
    def __init__(self, cv_name, Dataset, cv_path_suffix=parameter.cv_path_suffix, cv_norms=parameter.cv_norms):
        assert cv_name in ['Census-BM', 'MC-CNN', 'GC-Net'], 'Unknown CV Method: {}'.format(cv_name)
        self.cv_name = cv_name
        self.cv_path = os.path.join(Dataset.data_path, cv_path_suffix[cv_name])
        self.disp_est_path = os.path.join(Dataset.data_path, 'GC-Net/disp_est_pfm/') if cv_name == 'GC-Net' else ''
        if Dataset.dataset_name == 'Sceneflow' and cv_name == 'GC-Net':
            self.cv_norm = cv_norms[('GC-Net', 'Sceneflow')]
        else:
            self.cv_norm = cv_norms[cv_name]


class IDataListCreator:
    def __init__(self, args, Dataset, CVMethod, train_end_idx, val_end_idx, WeightLoss=None, parameter=parameter):
        assert (args.using_weighted_loss and WeightLoss) or (not args.using_weighted_loss), 'WeighteLoss is None!'
        self.args = args
        self.Dataset = Dataset
        self.CVMethod = CVMethod
        self.files = sorted(os.listdir(self.CVMethod.cv_path))
        self.train_end_idx = train_end_idx
        self.val_end_idx = val_end_idx
        self.WeightLoss = WeightLoss
        self.parameter = parameter
        self.update_parameter()
        # generate training and validation sample lists
        self.generate_training_samples()
        self.generate_validation_samples()

    def update_parameter(self):
        for key, value in vars(self.args).items():
            setattr(self.parameter, key, value)
        self.parameter.cv_norm = self.CVMethod.cv_norm
        self.parameter.training_data = []
        self.parameter.validation_data = []

        if self.parameter.using_weighted_loss:
            self.parameter.weighting_loss_func = self.WeightLoss.weighting_func

    def generate_training_samples(self):
        self.create_list(0, self.train_end_idx, 'train')

    def generate_validation_samples(self):
        self.create_list(self.train_end_idx, self.val_end_idx, 'val')

    def create_list(self, start_idx, end_idx, phase):
        for img_idx in range(start_idx, end_idx):
            if self.CVMethod.cv_name == 'MC-CNN' and img_idx == 1 and self.Dataset.dataset_name == 'K12': continue
            sample_name = self.files[img_idx].replace('.bin', '')
            print(sample_name)

            indicator_path = self.Dataset.indic_path + sample_name + '.png' if 'Geometry' in self.parameter.loss_type else ''
            est_path = self.CVMethod.disp_est_path + sample_name + '.pfm' if self.CVMethod.disp_est_path else ''
            cost_volume_path = self.CVMethod.cv_path + sample_name + '.bin' if self.CVMethod.cv_name == 'MC-CNN' \
                else self.CVMethod.cv_path + sample_name
            gt_path = self.Dataset.disp_gt_path + sample_name + '.pfm' if self.Dataset.dataset_name == 'Sceneflow' \
                else self.Dataset.disp_gt_path + sample_name + '.png'

            sample = DataSample(gt_path=gt_path, indicator_path=indicator_path, est_path=est_path, cost_volume_depth=256,
                                cost_volume_path=cost_volume_path) # cv depth should be 256, otherwise cv larger than 192 will not be reduced

            if phase == 'train':
                self.parameter.training_data.append(sample)
            else:
                self.parameter.validation_data.append(sample)