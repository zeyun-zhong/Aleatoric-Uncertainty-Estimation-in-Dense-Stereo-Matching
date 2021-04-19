import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, Concatenate, Add, BatchNormalization, GlobalAveragePooling3D

import params


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


class CVANet():
    def get_base_architecture(self, parameter):
        inputs = Input(shape=(None, None, parameter.cost_volume_depth, 1))
        inter_layer = inputs
        inter_layer = BatchNormalization()(inter_layer)

        # Neighbourhood layers
        nb_layer_num = int((parameter.nb_size - 1) / (parameter.nb_filter_size - 1))
        for nb_layers in range(0, nb_layer_num):
            inter_layer = Conv3D(parameter.nb_filter_num,
                                 (parameter.nb_filter_size, parameter.nb_filter_size, parameter.nb_filter_size),
                                 kernel_initializer='random_normal')(inter_layer)
            inter_layer = BatchNormalization()(inter_layer)
            inter_layer = Activation('relu')(inter_layer)

        # Depth layers
        depth = 8
        for depth_layers in range(0, parameter.depth_layer_num):
            if depth_layers < parameter.depth_layer_num - 1:
                dp_filter_num = parameter.depth_filter_num
            else:
                dp_filter_num = parameter.last_dp_filter_num
            inter_layer = Conv3D(dp_filter_num, (1, 1, depth), padding='same', kernel_initializer='random_normal')(
                inter_layer)
            inter_layer = BatchNormalization()(inter_layer)
            inter_layer = Activation('relu')(inter_layer)

            if (depth < 64):
                depth = depth * 2

        # Dense layer - Fully convolutional
        dense_depth = parameter.cost_volume_depth - ((parameter.nb_filter_size - 1) * nb_layer_num)
        # if parameter.dense_layer_type == 'GAP':
        #     inter_layer = GlobalAveragePooling3D()(inter_layer)
        if parameter.dense_layer_type == 'FC':
            inter_layer = Conv3D(parameter.dense_filter_num, (1, 1, dense_depth), padding='valid',
                                 kernel_initializer='glorot_normal', activation="relu")(inter_layer)
        elif parameter.dense_layer_type == 'AP':
            inter_layer = AveragePooling3D(pool_size=(1, 1, dense_depth), strides=None, padding="valid")(inter_layer)
        else:
            raise Exception('Unknown dense layer type: %s' % parameter.dense_layer_type)

        inter_layer = Dropout(0.5)(inter_layer)

        for dense_layer in range(0, parameter.dense_layer_num):
            inter_layer = Conv3D(parameter.dense_filter_num, (1, 1, 1), padding='valid',
                                 kernel_initializer='glorot_normal', activation="relu")(inter_layer)
            inter_layer = Dropout(0.5)(inter_layer)
        return inputs, inter_layer

    def get_model(self, parameter):
        inputs, inter_layer = self.get_base_architecture(parameter)
        inter_layer = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)

        if parameter.task_type == 'Classification':
            predictions = Activation('sigmoid')(inter_layer)
        elif parameter.task_type == 'Regression':
            predictions = inter_layer
        else:
            raise Exception('Unknown task type: %s' % parameter.task_type)

        return Model(inputs=inputs, outputs=predictions)

    def get_model_gmm(self, parameter):
        inputs, inter_layer = self.get_base_architecture(parameter)

        # phi: mixture coefficients; mu: mean; s: log variance
        K = parameter.K
        regularizer = None

        phi_layer = Conv3D(K, (1, 1, 1), padding='valid', kernel_initializer=tf.keras.initializers.Constant(value=1/K),
                           kernel_regularizer=regularizer)(inter_layer)
        phi_layer = Activation('softmax')(phi_layer)
        mu_layer = Conv3D(K, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal',
                          kernel_regularizer=regularizer)(inter_layer)
        s_layer = Conv3D(K, (1, 1, 1), padding='valid', kernel_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=regularizer,
                         activation=None)(inter_layer)

        return Model(inputs=inputs, outputs=[phi_layer, mu_layer, s_layer])

    def get_model_laplacian_uniform(self, parameter):
        inputs, inter_layer = self.get_base_architecture(parameter)

        alpha_layer = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)
        alpha_layer = Activation('sigmoid')(alpha_layer)
        s_layer = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)

        if parameter.lu_out == 3:
            s_layer_uniform = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)

        outputs = [alpha_layer, s_layer, s_layer_uniform] if parameter.gu_out == 3 else [alpha_layer, s_layer]

        return Model(inputs=inputs, outputs=outputs)

    def get_model_geometry_mask(self, parameter):
        inputs, inter_layer = self.get_base_architecture(parameter)

        c_layer = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)
        c_layer = Activation('sigmoid')(c_layer)
        s_layer = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)

        return Model(inputs=inputs, outputs=[c_layer, s_layer])