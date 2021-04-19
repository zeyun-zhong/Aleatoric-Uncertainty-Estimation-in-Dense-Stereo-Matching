import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import time

import sys
sys.path.insert(1, '../utils')

import data_generators
import graph
import params
import metrics

# from training_loop import ITrainingLoop
from training_loop_luis import ITrainingLoop

class Train(ITrainingLoop):
    """ Implementation of a custom training loop used to train different versions of CVA-Net. """

    def __init__(self, parameter=params.Params(), network_name="CVANet_{0}".format(int(time.time())),
                 root_dir='experiments', experiment_series='', pretrained_network=''):

        """ Default initialisation function.

        @param parameter: An parameter object containing all relevant training parameters.
        @param network_name: The name of the network to be trained.
        @param root_dir: Root directory used for every output produced during training.
        @param experiment_series: Path relative to the root dir used for every output produced during training.
        @param pretrained_network: Path to a file containing pretrained model weights.
        """
        super(Train, self).__init__(parameter=parameter, network_name=network_name, root_dir=root_dir,
                                    experiment_series=experiment_series, pretrained_network=pretrained_network)

        # ---------------------------
        # Initialise optimiser and metrics
        # ---------------------------
        self.metrics_object = metrics.Metrics(basic_loss=parameter.loss_type, pos_class_weight=parameter.pos_class_weight,
                                              gamma=parameter.gamma, eta=parameter.eta)
        self.optimizer = tf.keras.optimizers.Adam(lr=parameter.learning_rate)

        if parameter.task_type == 'Classification':
            self.metric_names = ['Loss', 'Accuracy']
            metric_formats = ['{:.3f}', '{:.3f}']
        else:
            self.metric_names = ['Loss']
            metric_formats = ['{:.3f}']
        self.metric_tracking.add_metrics(self.metric_names, metric_formats)

    def setup_model(self, parameters):
        """ Implementation of the abstract function defined in the base class. """
        if 'MM' in parameters.loss_type:
            return graph.CVANet().get_model_gmm(parameters)
        if 'Gaussian_Uniform' in parameters.loss_type or 'Laplacian_Uniform' in parameters.loss_type:
            return graph.CVANet().get_model_laplacian_uniform(parameters)
        if 'mask' in parameters.loss_type:
            return graph.CVANet().get_model_geometry_mask(parameters)
        return graph.CVANet().get_model(parameters)

    def setup_generators(self, parameters):
        """ Implementation of the abstract function defined in the base class. """
        sample_dims = (parameters.nb_size, parameters.nb_size, parameters.cost_volume_depth)

        if parameters.data_mode == 'cv':
            training_generator = data_generators.DataGeneratorCV(data_samples=parameters.training_data,
                                                                 batch_size=parameters.batch_size, dim=sample_dims,
                                                                 shuffle=True, augment=False, cv_norm=parameters.cv_norm)
            validation_generator = data_generators.DataGeneratorCV(data_samples=parameters.validation_data,
                                                                 batch_size=parameters.batch_size, dim=sample_dims,
                                                                 shuffle=False, augment=False, cv_norm=parameters.cv_norm)
        elif parameters.data_mode == 'image':
            training_generator = data_generators.DataGeneratorImage(data_samples=parameters.training_data,
                                                                 batch_size=parameters.batch_size, dim=sample_dims,
                                                                 shuffle=True, augment=False)
            validation_generator = data_generators.DataGeneratorImage(data_samples=parameters.validation_data,
                                                                   batch_size=parameters.batch_size, dim=sample_dims,
                                                                   shuffle=False, augment=False)
        else:
            raise Exception('Unknown data mode: ' + parameters.data_mode)

        return training_generator, validation_generator

    @tf.function
    def train_on_batch(self, X, y):
        """ Implementation of the abstract function defined in the base class. """
        with tf.GradientTape() as tape:
            prediction = self.model(X, training=True)
            loss = self.metrics_object.generic_loss()(y_true=y, y_pred=prediction)

        # loss = tf.stop_gradient(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Compute additional metrics
        if self.parameter.task_type == 'Classification':
            accuracy = self.metrics_object.accuracy(y_true=y, y_pred=prediction)
            return [loss, accuracy]
        else:
            return [loss]

    @tf.function
    def validate_on_batch(self, X, y):
        """ Implementation of the abstract function defined in the base class. """
        prediction = self.model(X, training=False)
        loss = self.metrics_object.generic_loss()(y_true=y, y_pred=prediction)

        # Compute additional metrics
        if self.parameter.task_type == 'Classification':
            accuracy = self.metrics_object.accuracy(y_true=y, y_pred=prediction)
            return [loss, accuracy]
        else:
            return [loss]

