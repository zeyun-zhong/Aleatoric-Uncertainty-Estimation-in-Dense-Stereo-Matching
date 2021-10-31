import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.layers import MaxPooling3D, AveragePooling3D, Concatenate, Add, BatchNormalization

import params


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


class CVANet():
    def get_model(self, parameter):
        inputs, inter_layer = self.get_base_architecture(parameter)
        if parameter.loss_type == 'Laplacian' or parameter.loss_type == 'Weighted_Laplacian':
            predictions = self.get_head_laplacian(parameter, inter_layer)
        elif 'Geometry' in parameter.loss_type:
            predictions = self.get_head_geometry(parameter, inter_layer)
        elif parameter.loss_type == 'Mixture':
            predictions = self.get_head_mixture(parameter, inter_layer)
        elif parameter.loss_type == 'GMM':
            predictions = self.get_head_gmm(parameter, inter_layer)
        else:
            raise Exception('Unknown loss type: %s' % parameter.loss_type)
        return Model(inputs=inputs, outputs=predictions)

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
            dp_filter_num = parameter.depth_filter_num
            inter_layer = Conv3D(dp_filter_num, (1, 1, depth), padding='same', kernel_initializer='random_normal')(
                inter_layer)
            inter_layer = BatchNormalization()(inter_layer)
            inter_layer = Activation('relu')(inter_layer)

            if (depth < 64):
                depth = depth * 2

        # Dense layer - Fully convolutional
        dense_depth = parameter.cost_volume_depth - ((parameter.nb_filter_size - 1) * nb_layer_num)
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

    def get_head_laplacian(self, parameter, inter_layer):
        unc = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)
        return unc

    def get_head_geometry(self, parameter, inter_layer):
        unc = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)
        if 'mask' in parameter.loss_type:
            region_index = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)
            region_index = Activation('sigmoid')(region_index)
            return [region_index, unc]
        return unc

    def get_head_mixture(self, parameter, inter_layer):
        alpha = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)
        alpha = Activation('sigmoid')(alpha)
        unc_laplacian = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)
        unc_uniform = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)
        return [alpha, unc_laplacian, unc_uniform]

    def get_head_gmm(self, parameter, inter_layer):
        # phi: mixture coefficients; mu: mean; s: log variance
        K = parameter.K
        regularizer = None

        phi_layer = Conv3D(K, (1, 1, 1), padding='valid',
                           kernel_initializer=tf.keras.initializers.Constant(value=1 / K),
                           kernel_regularizer=regularizer)(inter_layer)
        phi_layer = Activation('softmax')(phi_layer)
        mu_layer = Conv3D(K, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal',
                          kernel_regularizer=regularizer)(inter_layer)
        s_layer = Conv3D(K, (1, 1, 1), padding='valid', kernel_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=regularizer,
                         activation=None)(inter_layer)
        return [phi_layer, mu_layer, s_layer]