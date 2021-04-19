import os
import pickle
import numpy as np
import random
import sys

from abc import ABCMeta, abstractmethod
from metric_tracking import MetricTrackerManager


def progressbar(it, prefix="", size=40, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


class ITrainingLoop:
    """ Implementation of a custom training loop interface. """

    def __init__(self, parameter, network_name, root_dir, experiment_series, pretrained_network):
        """ Default initialisation function.

        @param parameter: An parameter object containing all relevant training parameters.
        @param network_name: The name of the network to be trained.
        @param root_dir: Root directory used for every output produced during training.
        @param experiment_series: Path relative to the root dir used for every output produced during training.
        @param pretrained_network: Path to a file containing pretrained model weights.
        """
        self.network_name = network_name
        self.parameter = parameter
        np.random.seed(parameter.training_start_epoch)
        random.seed(parameter.training_start_epoch)

        # ---------------------------
        # Initialise data, metric tracking and graph
        # ---------------------------
        self.training_generator, self.validation_generator = self.setup_generators(parameter)

        self.model = self.setup_model(parameter)
        input, _ = self.training_generator.__getitem__(0)
        _ = self.model(input)

        self.metric_tracking = MetricTrackerManager()

        # ---------------------------
        # Load pretrained weights
        # ---------------------------
        if pretrained_network:
            self.model.load_weights(pretrained_network)
            print(pretrained_network + ' is loaded!')
            # TODO: Implement feature to load optimiser state

        # ---------------------------
        # Initialise training information storage
        # ---------------------------
        self.root_path, self.log_path, self.model_path = self.setup_directory_structure(root_dir, experiment_series,
                                                                                        network_name)
        self.write_info_file(file=self.root_path + 'Info.txt', network_name=network_name, parameters=parameter,
                             model=self.model)

        with open(self.root_path + 'parameter', 'wb') as param_file:
            pickle.dump(parameter, param_file)

    @abstractmethod
    def setup_model(self, parameters):
        """ Initialise a trainable model.

        @param parameters: An parameter object containing all relevant training parameters.
        @return: A trainable keras model object.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_generators(self, parameters):
        """ Initialise two data generators

        @param parameters: An parameter object containing all relevant training parameters.
        @return: A data generator for the training data and one for the validation data.
        """
        raise NotImplementedError

    @staticmethod
    def setup_directory_structure(root_dir, experiment_series, network_name):
        """ Define the directory structure for output generated during training.

        @param root_dir: Root directory used for every output produced during training.
        @param experiment_series: Path relative to the root dir used for every output produced during training.
        @param network_name: The name of the network to be trained.
        @return: Paths to the root, the log and the model directories.
        """
        root_path = root_dir + '/' + experiment_series + '/' + network_name + '/'
        log_path = root_path + 'logs/'
        model_path = root_path + 'models/'

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        return root_path, log_path, model_path

    @staticmethod
    def write_info_file(file, network_name, parameters, model):
        """ Writes parameter and model information to file.

        @param file: Path to the file to which the information should be saved.
        @param network_name: The name of the network to be trained.
        @param parameters: An parameter object containing all relevant training parameters.
        @param model: An initialised Keras model object.
        """
        with open(file, 'w') as text_file:
            text_file.write(network_name + '\n\n')
            text_file.write(parameters.to_string())
            text_file.write('\n')
            model.summary(print_fn=lambda x: text_file.write(x + '\n'))

    @staticmethod
    def write_log_file(file, network_name, epoch, metrics_tracking, is_training):
        """ Writes a log file containing metric information at a specific epoch.

        @param file: Path to the file to which the information should be saved.
        @param network_name: The name of the network to be trained.
        @param epoch: The epoch for which the metric information is specified.
        @param metrics_tracking: A metric tracking object.
        @param is_training: Specifies if the lof file should contain training or validation information.
        """
        file_content = {'network_name':network_name, 'epoch':epoch, 'is_training':is_training,
                        'data':metrics_tracking.get_all_data(is_training)}
        with open(file, 'wb') as fp:
            pickle.dump(file_content, fp)

    @staticmethod
    def write_log_files(log_path, network_name, epoch, metrics_tracking):
        """ Writes log files for training and validation data containing metric information at a specific epoch.

        @param log_path: Path to the directory in which the log files should be saved.
        @param network_name: The name of the network to be trained.
        @param epoch: The epoch for which the metric information is specified.
        @param metrics_tracking: A metric tracking object.
        """
        training_file = log_path + network_name + '_training_' + str(epoch)
        ITrainingLoop.write_log_file(training_file, network_name, epoch, metrics_tracking, True)

        validation_file = log_path + network_name + '_validation_' + str(epoch)
        ITrainingLoop.write_log_file(validation_file, network_name, epoch, metrics_tracking, False)

    @abstractmethod
    def train_on_batch(self, X, y):
        """ Train the initialised model on a single data batch.

        @param X: Data batch.
        @param y: Reference / ground truth data.
        @return: Loss and metric values for the specified data batch.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_on_batch(self, X, y):
        """ Validate the initialised model on a single data batch.

        @param X: Data batch.
        @param y: Reference / ground truth data.
        @return: Loss and metric values for the specified data batch.
        """
        raise NotImplementedError

    def train(self):
        """ Train and validate the initialised model using the specified training parameters. """
        print(self.network_name)
        print('Number of training samples: ' + str(self.training_generator.__len__()))
        print('Number of validation samples: ' + str(self.validation_generator.__len__()))

        loss = open(self.root_path + "loss.txt", mode='a')
        loss.write('train, val \n')

        for epoch in range(self.parameter.epochs):
            print('Epoch ' + str(epoch+1) + ' of ' + str(self.parameter.epochs))
            self.metric_tracking.reset()

            # Training loop
            for sample_idx in progressbar(range(self.training_generator.__len__()), prefix='Train: '):
                input, reference = self.training_generator.__getitem__(sample_idx)
                metric_values = self.train_on_batch(input, reference)
                self.metric_tracking.update(True, self.metric_names, metric_values)

            # Validation loop
            for sample_idx in progressbar(range(self.validation_generator.__len__()), prefix='Val: '):
                input, reference = self.validation_generator.__getitem__(sample_idx)
                metric_values = self.validate_on_batch(input, reference)
                self.metric_tracking.update(False, self.metric_names, metric_values)

            # End of epoch
            self.training_generator.on_epoch_end()
            self.validation_generator.on_epoch_end()

            self.write_log_files(self.log_path, self.network_name, epoch+1, self.metric_tracking)

            # TODO: Write optimiser state to file
            validation_loss = self.metric_tracking.get_data('Loss', False)
            train_loss = self.metric_tracking.get_data('Loss', True)
            weights_path = self.model_path + 'weights_{:d}_{:.3f}.h5'.format(epoch+1, validation_loss)
            self.model.save_weights(filepath=weights_path)

            # write loss to txt file
            loss.write("{:.3f}, {:.3f} \n".format(train_loss, validation_loss))
            loss.flush()

            print(self.metric_tracking.to_string(True))
            print(self.metric_tracking.to_string(False))
            print('---')
