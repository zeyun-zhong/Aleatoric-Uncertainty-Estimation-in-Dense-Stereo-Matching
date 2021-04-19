import numpy as np
from numpy import inf

import tensorflow as tf
from tensorflow.keras.utils import Sequence
# from tqdm import tqdm

import pickle
import random
import cv2
from abc import abstractmethod
import sys
import metrics

sys.path.insert(1, '../utils')
import cost_volume
# import census_metric
import image_io
import utils


class DataSample:
    def __init__(self, gt_path='', indicator_path='', est_path='', cost_volume_path='', left_image_path='', right_image_path='', offset=[0,0],
                 step_size=[1,1], cost_volume_depth=256, result_path=''):
        self.gt_path = gt_path
        self.indicator_path = indicator_path
        self.est_path = est_path
        self.cost_volume_path = cost_volume_path
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        self.offset = offset
        self.step_size = step_size
        self.cost_volume_depth = cost_volume_depth
        self.result_path = result_path


class TrainingSample:
    def __init__(self, sample_name, row , col, gt_value, est_value=None, indicator=False, indicator_value=None):
        self.sample_name = sample_name
        self.row = row
        self.col = col
        self.gt_value = gt_value
        self.est_value = est_value
        self.indicator = indicator
        self.indicator_value = indicator_value


class IDataGenerator(Sequence):
    """ Abstract base class for generating batches of data. """
    
    def __init__(self, data_samples, batch_size, dim, shuffle, augment, mode='extract'):
        
        # Set member variables
        self.mode = mode
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = 1
        self.shuffle = shuffle
        self.augment = augment
        
        # Create sample list
        self.training_samples = self.create_training_samples(data_samples)
        self.indexes = np.arange(len(self.training_samples))
            
        # Shuffle sample list for initialisation
        if self.shuffle == True:
            self.shuffle_training_samples()

    @abstractmethod
    def create_training_samples(self, data_samples):
        """ Create sample IDs based on the provided file list

        @warning This is an abstract function that has to be implemented in any inherited class.

        @param data_samples: List containing data samples.
        @return: A list containing training samples.
        """
        raise NotImplementedError

    def get_number_of_samples(self):
        positives = 0
        overall = 0

        # for sample in tqdm(self.training_samples):
        for sample in self.training_samples:
            cv_extract, resize_factor = self.get_cv_extract(sample)
            disp_est =  cv_extract[int((self.dim[0] - 1) / 2), int((self.dim[1] - 1) / 2), :].argmin()
            disp_est = tf.convert_to_tensor((1.0 / resize_factor) * disp_est, dtype=tf.float32)
            disp_gt = tf.convert_to_tensor(sample.gt_value, dtype=tf.float32)
            positives += tf.dtypes.cast(metrics.compute_labels(disp_est, disp_gt), dtype=tf.float32)
            overall += 1.0

        return overall, positives

    def get_positive_weight(self, overall, positives, print_numbers=False):
        negatives = overall - positives
        if print_numbers:
            print('===========================')
            print('Number of training samples:')
            print('   Overall: ' + str(overall))
            print('   Positives: ' + str(positives))
            print('   Negatives: ' + str(negatives))
            print('===========================')
        return negatives / positives

    def __data_generation__(self, training_samples):
        """ Creates a batch of data with reference lables based on the specified IDs.

        @warning This is an abstract function that has to be implemented in any inherited class

        @param training_samples: List containing training samples which should be used to create this batch.
        @return: A batch of data with corresponding labels.
        """

        # Batch initialisation
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        if training_samples[0].indicator:
            Y = np.empty((self.batch_size, 3), dtype=float)
        else:
            Y = np.empty((self.batch_size, 2), dtype=float)

        # Generate data
        for i, training_sample in enumerate(training_samples):
            # ID structure: cost volume path, row, column, gt_disp
            cv_extract, resize_factor = self.get_cv_extract(training_sample)

            # generate disparity estimation if est is not available
            if training_sample.est_value:
                disp_est = training_sample.est_value
            else:
                disp_est = cv_extract[int((self.dim[0] - 1) / 2), int((self.dim[1] - 1) / 2), :].argmin()

            # extract cost volume and generate label
            X[i, :, :, :, 0] = cv_extract[:, :, :]
            Y[i, 0] = (1.0 / resize_factor) * disp_est
            Y[i, 1] = training_sample.gt_value
            if training_sample.indicator:
                Y[i, 2] = training_sample.indicator_value

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32)

    @abstractmethod
    def get_cv_extract(self):
        raise NotImplementedError

    def __len__(self):
        """ Denotes the number of batches per epoch.

        @return: Number of batches per epoch.
        """
        return int(np.floor(len(self.training_samples) / self.batch_size))

    def __getitem__(self, index):
        """ Creates a batch of data with reference lables for a specified batch index.

        @param index: Index of the batch to be created
        @return: A batch of data with corresponding labels.
        """

        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        training_samples = [self.training_samples[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation__(training_samples)
        return X, y

    def shuffle_training_samples(self):
        """ Shuffles the list of batch indices. """
        self.indexes = np.arange(len(self.training_samples))
        np.random.shuffle(self.indexes)

    def on_epoch_end(self):  
        # Updates indexes after each epoch
        if (self.shuffle):
            self.shuffle_training_samples()

    def training_samples_from_GT(self, sample_name, gt_path, step_size, offset, est_path=None, indicator_path=None):
        """ Creates a set of sample IDs based on a specified reference disparty map.

        Based on the specified step size and offset, the reference disparity map is sampled and for every pixel
        with a reference disparity available one sample ID is created.

        @param sample_name: Name of the current sample (e.g. left image path, cost volume path)
        @param gt_path: Path of the reference disparity map
        @param step_size: Specifies the distance between two sample points
        @param offset: Specifies the offset of the first sample point from the image origin
        @return: A list of sample IDs and the normalised reference disparity map
        """

        # Read ground truth and normalise values if necessary
        disp_gt = image_io.read(gt_path)
        dimensions = disp_gt.shape

        # Read region indicator
        if indicator_path:
            mask_indicator = image_io.read(indicator_path)

        # Read estimated disparity if necessary
        if est_path:
            disp_est = image_io.read(est_path)

        # Assure that there are no constructs like -inf, inf
        disp_gt[disp_gt == -inf] = 0
        disp_gt[disp_gt == inf] = 0

        # Check for available ground truth points
        training_samples = []
        indicator = True if indicator_path else False
        print('using indicators') if indicator else print('not using indicators')
        for row in range(offset[0], dimensions[0], step_size[0]):
            for col in range(offset[1], dimensions[1], step_size[1]):
                gt_value = disp_gt[row][col]
                if (gt_value != 0):
                    # Ground truth point is available -> Create sample for this pixel
                    est_value = disp_est[row][col] if est_path else None
                    indicator_value = mask_indicator[row][col] if indicator_path else 1
                    training_samples.append(TrainingSample(sample_name, row , col, gt_value, est_value,
                                                           indicator=indicator, indicator_value=indicator_value))
                    
        return training_samples, disp_gt


class DataGeneratorCV(IDataGenerator):
    """ Generates batches of data based on a previously computed cost volume """

    # @brief Initialises the data generator
    # @warning All specified cost volumes are loaded to memory before training can be started
    # @param data_file_paths One tuple per cost volume: 
    # [{cost volume path}, {gt image path}, gt_norm_factor, step_size, offset]
    def __init__(self, data_samples, batch_size=8, dim=(13, 13, 256), shuffle=False,
                 augment=False, cv_norm=[0.0, 1.0]):
        
        # Call constructor of abstract base class
        super(DataGeneratorCV, self).__init__(data_samples, batch_size, dim, shuffle, augment)
        self.cv_norm = cv_norm
        
        # Load cost volumes
        self.cv_dict = self.create_cv_dict(data_samples)

    def create_training_samples(self, data_samples):
        """ Implementation of the abstract function defined in the base class. """
        
        training_samples = []
        
        # Iterate over the provided cost volumes
        for data_sample in data_samples:
            
            # Get datasamples and normalised ground truth for current sample
            curr_data_samples, _ = self.training_samples_from_GT(sample_name=data_sample.cost_volume_path,
                                                                 gt_path=data_sample.gt_path,
                                                                 est_path=data_sample.est_path,
                                                                 indicator_path=data_sample.indicator_path,
                                                                 step_size=data_sample.step_size,
                                                                 offset=data_sample.offset)
            training_samples.extend(curr_data_samples)
                      
        return training_samples

    def create_cv_dict(self, data_samples):
        """ Loads all cost volumes to memory.

        @param data_samples: List of data samples to load.
        @return: A dictionary containing all specified cost volumes with their path as key attribute.
        """

        cv_dict = {}        
        for data_sample in data_samples:
                      
            # Load and store cost volume
            cv_path = data_sample.cost_volume_path
            cv = cost_volume.CostVolume()
            if (cv_path[-3:] == 'bin'): 
                # To get the cost volume dimensions the ground truth disparity map is used
                disp_gt = image_io.read(data_sample.gt_path)
                cv.load_bin(cv_path, disp_gt.shape[0], disp_gt.shape[1], data_sample.cost_volume_depth)
                             
            elif cv_path[-3:] == 'dat':
                cv.load_dat(cv_path)
                
            else:
                with open(cv_path, 'rb') as file:
                    cv = pickle.load(file)

            if data_sample.cost_volume_depth > self.dim[2]:
                cv.reduce_depth(self.dim[2])

            # Normalise the cost volume
            cv.normalise(self.cv_norm[0], self.cv_norm[1])
            print('sample cv max: ', np.max(cv.get_data()))
            print('sample cv min: ', np.min(cv.get_data()))
            cv_dict[cv_path] = cv
        return cv_dict

    def get_cv_extract(self, training_sample):
        """ Implementation of the abstract function defined in the base class. """
        return self.cv_dict[training_sample.sample_name].get_excerpt((training_sample.row, training_sample.col),
                                                                     self.dim[0]), 1.0
    

class DataGeneratorImage(IDataGenerator):
    """ Generates batches of data based on a stereo image pair using the Census metric. """
    
    # @brief Initialise the data generator
    # @param data_file_paths One tuple per image pair: 
    # [{left image path}, {right image path}, {gt image path}, gt_norm_factor, step_size, offset]
    def __init__(self, data_samples, batch_size=8, dim=(13, 13, 256), shuffle=False,
                 augment=False):
        
        # Set member variables        
        self.decrease_prob = 0.5
        self.resize_factor_range = [0.25, 2.0]       
        self.metric_filter_size = 5
        self.smooth_filter_size = 5
        self.metric = census_metric.CensusMetric(self.metric_filter_size, self.smooth_filter_size)
        self.resize_factor_dict = None
        
        # Call constructor of abstract base class
        super(DataGeneratorImage, self).__init__(data_samples, batch_size, dim, shuffle, augment)
        
        # Load images
        self.image_dict = self.create_image_dict(data_samples)
               
        # Create Census transformations
        self.census_dict = self.create_census_dict()

    def create_training_samples(self, data_samples):
        """ Implementation of the abstract function defined in the base class. """
        
        training_samples = []
        resize_factor_dict = {}
        
        # Iterate over the provided image pairs
        for data_sample in data_samples:

            # Get datasamples and normalised ground truth for current sample
            sample_name = data_sample.left_image_path
            curr_data_samples, disp_gt = self.training_samples_from_GT(sample_name=sample_name,
                                                                 gt_path=data_sample.gt_path,
                                                                 step_size=data_sample.step_size,
                                                                 offset=data_sample.offset)
            training_samples.extend(curr_data_samples)
            
            # Get max disparity and determine the resize factor range of the specific image           
            max_resize_factor = self.dim[2] / disp_gt.max()
            max_resize_factor = min(max_resize_factor, self.resize_factor_range[1])
            min_resize_factor = min(max_resize_factor, self.resize_factor_range[0])          
            resize_factor_dict[sample_name] = [min_resize_factor, max_resize_factor]
            
        self.resize_factor_dict = resize_factor_dict                        
        return training_samples

    def create_image_dict(self, data_samples):
        """ Loads images based on provided file list and stores them in a dictionary.

        @param data_samples: List containing data samples
        @return: A dictionary of type: key: left image path, data: tuple[left image, left census, right image, right census]
        """
        
        image_dict = {}        
        for data_sample in data_samples:
            # Load and store left and right image
            left_image = image_io.read(data_sample.left_image_path)
            right_image = image_io.read(data_sample.right_image_path)
            image_dict[data_sample.left_image_path] = [left_image, right_image]
        return image_dict

    def create_census_dict(self):
        """ Loads images based on provided file list and stores them in a dictionary.

        @return: A dictionary of type: key: left image path, data: tuple[left census, right census, resize_factor]
        """
        
        census_dict = {}
                
        for name, images in self.image_dict.items():
            
            # Generate random resize factor
            resize_factor_range = self.resize_factor_dict[name]
            
            if (self.augment and resize_factor_range[0] != resize_factor_range[1]):
                
                if (random.uniform(0.0, 1.0) > self.decrease_prob and resize_factor_range[1] > 1.0):
                    # Increase image dimensions
                    resize_factor = round(random.uniform(1.0, resize_factor_range[1]), 2)
                else:
                    # Decrease image dimensions
                    resize_factor = round(random.uniform(resize_factor_range[0], min(resize_factor_range[1], 1.0)), 2)      
            else:
                resize_factor = min(resize_factor_range[1], 1.0)

            left_image = images[0]
            right_image = images[1]
            
            # Resize images if requested
            if (resize_factor < 1.0):
                left_image = cv2.resize(left_image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
                right_image = cv2.resize(right_image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            elif (resize_factor > 1.0):
                left_image = cv2.resize(left_image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
                right_image = cv2.resize(right_image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
            
            # Transform images
            census_dict[name] = [self.metric.__create_census_trafo__(left_image),
                                          self.metric.__create_census_trafo__(right_image), resize_factor]
                
        return census_dict

    def on_epoch_end(self):  
        # Updates indexes after each epoch
        if (self.shuffle):
            self.shuffle_training_samples()
            
        # Create new Census transformations with different resize factors 
        # after each epoch if augmentation is required
        if (self.augment):
            self.create_census_dict()

    def get_cv_extract(self, training_sample):
        """ Implementation of the abstract function defined in the base class. """
        census_trafos = self.census_dict[training_sample.sample_name]
        resize_factor = census_trafos[2]

        # Generate cost volume extract
        row = int(training_sample.row * resize_factor)
        col = int(training_sample.col * resize_factor)

        cv_extract = self.metric.__compute_cv_extract__(census_trafos[0], census_trafos[1],
                                                        row, col, self.dim)

        return cv_extract, resize_factor
