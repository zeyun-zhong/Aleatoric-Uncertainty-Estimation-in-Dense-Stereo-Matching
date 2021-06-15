import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import pickle
import sys
import scipy.io

sys.path.insert(1, '../utils')
import image_io
import graph
import census_metric
import cost_volume
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


class Test:

    def __init__(self, weights_file, param_file, model_name, extract_width=100, extract_height=100,
                 file_extension='.pfm', save_png=True, save_disp_map=False, save_gmm_komponents=False):
        self.model_name = model_name
        self.save_png = save_png
        self.file_extension = file_extension
        self.save_disp_map = save_disp_map
        self.save_gmm_komponents = save_gmm_komponents

        # Depending on the amount of available memory, it may be necessary to process the cost volume block-wise
        self.extract_width = extract_width
        self.extract_height = extract_height

        # Load parameter
        with open(param_file, 'rb') as param_file:
            parameter = pickle.load(param_file)
        self.neighbourhood_size = parameter.nb_size
        self.cost_volume_depth = parameter.cost_volume_depth
        self.loss_type = parameter.loss_type
        self.cv_norm = parameter.cv_norm

        # Load trained model
        self.model = graph.CVANet().get_model(parameter)
        self.model.load_weights(weights_file)

    def create_cost_volume(self, sample):
        image_left = image_io.read(sample.left_image_path)
        image_right = image_io.read(sample.right_image_path)

        cm = census_metric.CensusMetric(5, 5)
        census_left = cm.__create_census_trafo__(image_left)
        census_right = cm.__create_census_trafo__(image_right)
        return cm.__compute_cost_volume__(census_left, census_right, self.cost_volume_depth)

    def load_cost_volume(self, sample):
        cv_path = sample.cost_volume_path
        cv = cost_volume.CostVolume()
        if cv_path[-3:] == 'bin':
            img_shape = image_io.read(sample.left_image_path).shape
            cv.load_bin(cv_path, img_shape[0], img_shape[1], self.cost_volume_depth)

        elif cv_path[-3:] == 'dat':
            cv.load_dat(cv_path)

        else:
            with open(cv_path, 'rb') as file:
                cv = pickle.load(file)

        if sample.cost_volume_depth > self.cost_volume_depth:
            cv.reduce_depth(self.cost_volume_depth)

        cv.normalise(self.cv_norm[0], self.cv_norm[1])
        return cv

    def predict(self, samples, gamma_gmm=0.01):
        for idx, sample in enumerate(samples):
            print('Started sample ' + str(idx+1) + ' of ' + str(len(samples)))

            # Compute / load the cost volume (in this example the cost volume is computed based on the Census metric)
            if sample.cost_volume_path:
                print('    Load cost volume...')
                cv = self.load_cost_volume(sample)
            else:
                print('    Compute cost volume...')
                cv = self.create_cost_volume(sample)

            border = int((self.neighbourhood_size - 1) / 2)
            cv_data = cv.get_data(border)
            cost_volume_dims = cv.dim()

            # Process cost volume block-wise to get the confidence map
            print('    Compute confidence map...')
            print('    Loss type: ', self.loss_type)

            num_of_predicted_values = 1
            if self.loss_type == 'Mixture':
                num_of_predicted_values = 3
            elif self.loss_type == 'Geometry-mask':
                num_of_predicted_values = 2

            # analysis of GMM properties
            if self.loss_type == 'GMM' and self.save_gmm_komponents:
                phi_map = np.zeros((cost_volume_dims[0], cost_volume_dims[1], 3))
                mu_map = np.zeros((cost_volume_dims[0], cost_volume_dims[1], 3))
                sigma_map = np.zeros((cost_volume_dims[0], cost_volume_dims[1], 3)) # variance of GMM komponents

            prediction = np.zeros((cost_volume_dims[0], cost_volume_dims[1], num_of_predicted_values))
            start_y = 0
            end_y = start_y + self.extract_height
            while (start_y < cost_volume_dims[0]):
                start_x = 0
                end_x = start_x + self.extract_width

                while (start_x < cost_volume_dims[1]):
                    if (end_x > cost_volume_dims[1]):
                        end_x = cost_volume_dims[1]
                    if (end_y > cost_volume_dims[0]):
                        end_y = cost_volume_dims[0]

                    extract = cv_data[start_y:end_y+2*border, start_x:end_x+2*border, :]
                    net_input = np.empty((1, extract.shape[0], extract.shape[1], extract.shape[2], 1))
                    net_input[0,:,:,:,0] = extract

                    prediction_extract = self.model.predict(net_input)

                    if self.loss_type == 'GMM':
                        phi = prediction_extract[0][0, :, :, 0, :]
                        mu = prediction_extract[1][0, :, :, 0, :]
                        s = prediction_extract[2][0, :, :, 0, :]

                        # try to solve overflow for sigma
                        index = np.where(phi < gamma_gmm)
                        s[index] = 0
                        sigma = np.exp(s) # gmm: sigma -> variance; lmm: sigma -> std
                        sigma[index] = 0
                        phi[index] = 0

                        mean = np.sum(phi * mu, axis=-1)
                        variance = np.sum(phi * (sigma + np.square(mu)), axis=-1) - np.square(mean)
                        prediction[start_y:end_y, start_x:end_x, 0] = np.sqrt(variance)

                        if self.save_gmm_komponents:
                            phi_map[start_y:end_y, start_x:end_x] = phi
                            mu_map[start_y:end_y, start_x:end_x] = mu
                            sigma_map[start_y:end_y, start_x:end_x] = sigma
                    else:
                        if num_of_predicted_values > 1:
                            for i in range(num_of_predicted_values):
                                prediction[start_y:end_y, start_x:end_x, i] = prediction_extract[i][0, :, :, 0, 0]
                        else:
                            prediction[start_y:end_y, start_x:end_x, 0] = prediction_extract[0, :, :, 0, 0]

                    start_x = start_x + self.extract_width
                    end_x = start_x + self.extract_width

                start_y = start_y + self.extract_height
                end_y = start_y + self.extract_height

            if self.loss_type == 'Binary_Cross_Entropy':
                unc_map = prediction[:, :, 0]

            elif self.loss_type == 'Laplacian' or self.loss_type == 'Geometry':
                unc_map = np.exp(prediction[:, :, 0])

            elif self.loss_type == 'Mixture':
                alpha = prediction[:, :, 0]
                s_l = prediction[:, :, 1]
                s_u = prediction[:, :, 2]

                variance_laplace = np.square(np.exp(s_l))
                variance_uniform = np.square(np.exp(s_u))
                variance_combined = np.multiply(alpha, variance_laplace) + \
                                    np.multiply(np.ones_like(alpha) - alpha, variance_uniform)
                unc_map = np.sqrt(variance_combined)

            elif self.loss_type == 'Geometry-mask':
                mask_map = prediction[:, :, 0]
                unc_map = np.exp(prediction[:, :, 1])
                np.save(sample.result_path + self.model_name + '_mask' + '.npy', mask_map.astype(np.float32))

            # save GMM properties
            elif self.loss_type == 'GMM':
                unc_map = prediction[:, :, 0]
                if self.save_gmm_komponents:
                    np.save(sample.result_path + self.model_name + '_phi' + '.npy', phi_map.astype(np.float32))
                    np.save(sample.result_path + self.model_name + '_mu' + '.npy', mu_map.astype(np.float32))
                    np.save(sample.result_path + self.model_name + '_sigma' + '.npy', sigma_map.astype(np.float32))

            else:
                raise Exception('Unknown loss type: %s' % self.loss_type)

            # Save confidence map
            if self.save_png:
                unc_map_visual = unc_map

                if self.loss_type == 'Binary_Cross_Entropy':
                    unc_map_visual = unc_map_visual * 255

                unc_map_visual = unc_map_visual.astype(int)
                image_io.write(sample.result_path + 'ConfMap_' + self.model_name + '.png', unc_map_visual)
            image_io.write(sample.result_path + 'ConfMap_' + self.model_name + '.pfm', unc_map.astype(np.float32))

            # Save disparity map
            if self.save_disp_map:
                disp_map = np.argmin(cv.get_data(), 2)
                image_io.write(sample.result_path + 'DispMap.png', disp_map)
