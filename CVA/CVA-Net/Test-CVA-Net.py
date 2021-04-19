import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from test import Test
from data_generators import DataSample
import params
parameter = params.Params()


def infer(model_name, cv_path, results_path, n, weight_file=None, folder=None, left_img_path=None):
    model_path = '/home/zeyun/Projects/CVA/experiments/dynamic-depth/' + model_name + '/'
    if not weight_file:
        # find the model with minimal val loss
        files = os.listdir(model_path + 'models/')
        files_ = [f.split('_')[-1].replace('.h5', '') for f in files]
        files1 = list(map(float, files_))
        weight_file = files[files1.index(min(files1))]
        print('finded the best model: ', weight_file)

        weight_file = model_path + 'models/' + weight_file

    samples = []
    start_index = 1 if 'kitti' in cv_path else 0
    for img_idx in range(start_index, n):
        sample_name = folder[img_idx].replace('.bin', '') if folder else str(img_idx).rjust(6, '0') + '_10'
        result_path = results_path + sample_name + '/'
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        cost_volume_path = cv_path + sample_name + '.bin' if 'MC-CNN' in model_name else cv_path + sample_name
        left_image_path = left_img_path + sample_name + '.png' if left_img_path else None
        print(left_image_path)
        samples.append(DataSample(left_image_path=left_image_path,
                                  cost_volume_depth=parameter.cost_volume_depth,
                                  cost_volume_path=cost_volume_path,
                                  result_path=result_path))

    # gamma_gmm = 0.5
    # model_name = model_name + '_' + str(gamma_gmm)
    # model_name = model_name + '_epoch_10'
    tester = Test(weights_file=weight_file, param_file=model_path + 'parameter',
                  model_name=model_name, save_disp_map=False, save_gmm_komponents=False)
    tester.predict(samples)


if __name__ == '__main__':
    dataset = 'kitti-2015' # or 'middlebury-v3'

    cv_path = '/media/zeyun/ZEYUN/MA/{}/cv_MC/'.format(dataset)
    results_path = '/home/zeyun/Projects/CVA/results/{}/'.format(dataset)

    folder = sorted(os.listdir(cv_path)) if 'middlebury' in cv_path else None
    n = 15 if 'middlebury' in cv_path else 100
    left_img_path = '/media/zeyun/ZEYUN/MA/{}/images/left/'.format(dataset) if 'MC' in cv_path else None

    # weight_file = '/home/zeyun/Projects/CVA/experiments/dynamic-depth/CVA-Net_GMM_paper_K3_MC-CNN/models/weights_10_246.219.h5'

    infer('CVA-Net_Mixed_Uniform_mask_pred_paper_MC-CNN', cv_path, results_path, n, weight_file=None, folder=folder, left_img_path=left_img_path)