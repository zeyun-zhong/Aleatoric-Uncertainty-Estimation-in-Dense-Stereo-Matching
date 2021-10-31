import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from test import Test
from data_generators import DataSample


class Inference:
    def __init__(self, dataset, cv_method, sample_start_index=None, sample_end_index=None):
        assert dataset in ['kitti-2015', 'middlebury-v3'], 'Unknown dataset!'
        assert cv_method in ['Census-BM', 'MC-CNN', 'GC-Net'], 'Unknown disparity method!'
        self.dataset = dataset
        self.cv_method = cv_method
        self.set_paths()

        if sample_start_index is not None:
            self.sample_start_index = sample_start_index
        elif dataset == 'kitti-2015':
            self.sample_start_index = 1
        else:
            self.sample_start_index = 0
        self.sample_end_index = sample_end_index if sample_end_index else len(self.files)

        self.generate_test_samples()

    def set_paths(self):
        self.model_root_path = '/home/zeyun/Projects/CVA/experiments/dynamic-depth/'
        self.results_path = '/home/zeyun/Projects/CVA/results/{}/'.format(self.dataset)
        self.left_img_path = '/media/zeyun/ZEYUN/MA/{}/images/left/'.format(self.dataset)
        cv_dict = {'Census-BM': 'cv_census',
                   'MC-CNN': 'cv_MC',
                   'GC-Net': 'cv_GC'}
        self.cv_path = '/media/zeyun/ZEYUN/MA/{}/{}/'.format(self.dataset, cv_dict[self.cv_method])
        self.files = sorted(os.listdir(self.cv_path))

    def get_weight_path(self, model_name, weight_file):
        if not weight_file:
            # find the model with minimal val loss
            model_path = os.path.join(self.model_root_path, model_name, 'models/')
            files = os.listdir(model_path)
            files_ = [f.split('_')[-1].replace('.h5', '') for f in files]
            files1 = list(map(float, files_))
            weight_file = files[files1.index(min(files1))]
            print('finded the best model: ', weight_file)

            weight_file = os.path.join(model_path, weight_file)
        return weight_file

    def generate_test_samples(self):
        self.samples = []
        for img_idx in range(self.sample_start_index, self.sample_end_index):
            sample_name = self.files[img_idx].replace('.bin', '')
            result_path = self.results_path + sample_name + '/'
            if not os.path.exists(result_path):
                os.mkdir(result_path)

            cost_volume_path = self.cv_path + sample_name + '.bin' if self.cv_method == 'MC-CNN' else self.cv_path + sample_name
            left_image_path = self.left_img_path + sample_name + '.png'
            self.samples.append(DataSample(left_image_path=left_image_path,
                                           cost_volume_path=cost_volume_path,
                                           result_path=result_path))

    def infer(self, model_name, weight_file=None):
        weight_file = self.get_weight_path(model_name, weight_file)
        param_file = os.path.join(self.model_root_path, model_name, 'parameter')
        tester = Test(weights_file=weight_file, param_file=param_file, model_name=model_name, save_disp_map=False)
        tester.predict(self.samples)


if __name__ == '__main__':
    dataset = 'kitti-2015'

    # Infer1 = Inference(dataset=dataset, cv_method='Census-BM')
    # Infer1.infer('CVA-Net_Weighted_Laplacian_Census-BM_lookup_std_4_thresh_5')
    #
    # Infer2 = Inference(dataset=dataset, cv_method='MC-CNN')
    # Infer2.infer('CVA-Net_Weighted_Laplacian_MC-CNN_lookup_std_4_thresh_5')

    Infer3 = Inference(dataset=dataset, cv_method='GC-Net')
    Infer3.infer('CVA-Net_Weighted_Laplacian_GC-Net_lookup_std_4_thresh_5')