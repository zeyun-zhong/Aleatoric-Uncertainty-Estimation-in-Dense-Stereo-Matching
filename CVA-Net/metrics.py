import tensorflow as tf
import utils
import math
# from tensorflow_probability import distributions as tfd


def compute_labels(disp_est, disp_gt, treshold_abs=3.0, threshold_rel=0.05):
    """ Computes binary labels based on estimated and reference disparity for interpreting the uncertainty
    prediction as classification task.

    @details: For computing the labels, the error definition of the KITTI 2015 dataset is used, utilising an
    absolute and a relative error threshold to decide whether an estimate is correct or not.

    @param disp_est: Estimated disparity.
    @param disp_gt: Ground truth disparity.
    @param treshold_abs: Absolute error threshold.
    @param threshold_rel: Error threshold defined relative to the reference disparity.
    @return: Tensor containing binary labels.
    """
    diff = tf.abs(disp_est - disp_gt)
    label_abs = tf.greater(diff, tf.constant([treshold_abs]))
    label_rel = tf.greater(diff, tf.constant([threshold_rel]) * tf.abs(disp_gt))
    return tf.math.logical_not(tf.math.logical_and(label_abs, label_rel))


class Metrics:
    """ Contains and manages loss functions and further metrics for training CVA-Net based networks.

    If the generic loss function is used, the Metrics object takes care of the correct configuration according to the
    specified member variables.
    """

    def __init__(self, basic_loss, pos_class_weight=1.0, gamma=1.0, eta=1.0):
        """ Initialisation for the metrics class.

        @param basic_loss: String that specifies which kind of loss function should be used.
        @param pos_class_weight: Weighting of positive samples, while negative samples are weightes with 1.0.
                                 (Only used for classification-based losses.)
        This information is of special importance for the additional metrics since their computation may
        vary based on the utilised loss function.
        """
        self.basic_loss = basic_loss
        self.pos_class_weight = pos_class_weight
        self.gamma = gamma
        self.eta = eta

    def binary_crossentropy(self, y_true, y_pred):
        """ Computes the binary cross-entropy interpreting the uncertainty prediction as classification task.

        @param y_true: Tensor containing estimated and reference disparity.
        @param y_pred: The predicted label.
        @param pos_weight: Weighting of positive samples, while negative samples are weightes with 1.0.
        @return: The binary cross entropy with respect to the predicted labels.
        """
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)
        return tf.keras.losses.binary_crossentropy(tf.squeeze(label_gt), tf.squeeze(y_pred))

    def weighted_binary_crossentropy(self, y_true, y_pred):

        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        label_gt = tf.squeeze(tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32))
        weights = (label_gt * (self.pos_class_weight - 1.0)) + 1.0

        y_pred = tf.squeeze(y_pred)

        label_gt = tf.stack([label_gt, 1.0 - label_gt], axis=1)
        y_pred = tf.stack([y_pred, 1.0 - y_pred], axis=1)

        #bce = -label_gt * tf.math.log(y_pred) - (1.0 - label_gt) * tf.math.log(1.0 - y_pred)

        #bce = tf.keras.losses.BinaryCrossentropy(
        #    from_logits=False, label_smoothing=0, reduction=tf.keras.losses.Reduction.NONE)
        losses = tf.keras.losses.binary_crossentropy(label_gt, y_pred)

        #tf.print(tf.shape(label_gt))
        #tf.print(tf.shape(y_pred))
        #tf.print(tf.shape(losses))


        weighted_bce = weights * losses
        mean_bce = tf.math.reduce_mean(weighted_bce)

        '''
        if math.isnan(mean_bce):
            tf.print('y_true: ', y_true, summarize=-1)
            tf.print('y_pred: ', y_pred, summarize=-1)
            tf.print('Est: ', disp_est, summarize=-1)
            tf.print('GT: ', disp_gt, summarize=-1)
            tf.print('label_gt: ', label_gt, summarize=-1)
            tf.print('weights: ', weights, summarize=-1)
            tf.print('bce: ', losses, summarize=-1)
            tf.print('weighted_bce: ', weighted_bce, summarize=-1)
            tf.print('mean_bce: ', mean_bce, summarize=-1)
            raise Exception('Problem with loss!')
        '''
        return mean_bce

    def accuracy(self, y_true, y_pred):
        """ Computes the binary accuracy if uncertainty prediction is interpreted as classification task.

        @param y_true: Tensor containing estimated and reference disparity.
        @param y_pred: The predicted label.
        @return: The accuracy of the estimated binary labels.
        """
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)
        return tf.metrics.binary_accuracy(tf.squeeze(label_gt), tf.squeeze(y_pred))

    def residual_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        loss = tf.abs(tf.squeeze(disp_est) + tf.squeeze(y_pred) - tf.squeeze(disp_gt))
        return tf.math.reduce_mean(loss)

    def residual_loss_abs(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        loss = tf.abs(tf.abs(tf.squeeze(disp_est) - tf.squeeze(disp_gt)) - tf.abs(tf.squeeze(y_pred)))
        return tf.math.reduce_mean(loss)

    def probabilistic_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        loss = (math.sqrt(2) * tf.math.abs(tf.squeeze(disp_gt) - tf.squeeze(disp_est)) *
                tf.math.exp(-tf.squeeze(y_pred))) + tf.squeeze(y_pred)
        return tf.math.reduce_mean(loss)

    def mixed_uniform_l1(self, y_true, y_pred):
        disp_est, disp_gt, indicators = tf.split(y_true, num_or_size_splits=3, axis=1)
        # set all variables to one dimensional
        disp_est = tf.squeeze(disp_est)
        disp_gt = tf.squeeze(disp_gt)
        indicators = tf.squeeze(indicators)
        y_pred = tf.squeeze(y_pred)

        diff = tf.abs(disp_gt - disp_est)
        diff_1 = diff - math.sqrt(3) * tf.math.exp(y_pred)

        loss_good = math.sqrt(2) * diff * tf.math.exp(-y_pred) + y_pred
        loss = tf.where(tf.equal(indicators, 1), loss_good, tf.abs(diff_1))
        return tf.math.reduce_mean(loss)

    def mixed_uniform_l2(self, y_true, y_pred):
        disp_est, disp_gt, indicators = tf.split(y_true, num_or_size_splits=3, axis=1)
        # set all variables to one dimensional
        disp_est = tf.squeeze(disp_est)
        disp_gt = tf.squeeze(disp_gt)
        indicators = tf.squeeze(indicators)
        y_pred = tf.squeeze(y_pred)

        diff = tf.abs(disp_gt - disp_est)
        diff_1 = diff - math.sqrt(3) * tf.math.exp(y_pred)

        loss_good = math.sqrt(2) * diff * tf.math.exp(-y_pred) + y_pred
        loss = tf.where(tf.equal(indicators, 1), loss_good, tf.square(diff_1))
        return tf.math.reduce_mean(loss)

    def mixed_uniform_loss(self, y_true, y_pred):
        disp_est, disp_gt, indicators = tf.split(y_true, num_or_size_splits=3, axis=1)
        # set all variables to one dimensional
        disp_est = tf.squeeze(disp_est)
        disp_gt = tf.squeeze(disp_gt)
        indicators = tf.squeeze(indicators)
        y_pred = tf.squeeze(y_pred)

        diff = tf.abs(disp_gt - disp_est)
        diff_1 = diff - math.sqrt(3) * tf.math.exp(y_pred)
        diff_1_abs = tf.abs(diff_1)

        loss_good = math.sqrt(2) * diff * tf.math.exp(-y_pred) + y_pred
        loss_tmp = tf.where(tf.equal(indicators, 1), loss_good, diff_1_abs - 0.5)
        condition = tf.logical_and(tf.equal(indicators, 0), tf.less(diff_1_abs, 1.0))
        loss = tf.where(condition, 0.5 * tf.square(diff_1), loss_tmp)
        return tf.math.reduce_mean(loss)

    def mixed_uniform_mask_loss(self, y_true, y_pred):
        # the mask index is also predicted
        disp_est, disp_gt, indicators = tf.split(y_true, num_or_size_splits=3, axis=1)
        # set all variables to one dimensional
        disp_est = tf.squeeze(disp_est)
        disp_gt = tf.squeeze(disp_gt)
        indicators = tf.squeeze(indicators)
        c = tf.squeeze(y_pred[0])
        s = tf.squeeze(y_pred[1])

        # negative log likelihood
        diff = tf.abs(disp_gt - disp_est)
        diff_1 = diff - math.sqrt(3) * tf.math.exp(s)
        diff_1_abs = tf.abs(diff_1)

        loss_good = math.sqrt(2) * diff * tf.math.exp(-s) + s
        loss_tmp = tf.where(tf.equal(indicators, 1), loss_good, diff_1_abs - 0.5)
        condition = tf.logical_and(tf.equal(indicators, 0), tf.less(diff_1_abs, 1.0))
        nll_loss = tf.math.reduce_mean(tf.where(condition, 0.5 * tf.square(diff_1), loss_tmp))

        # binary cross-entropy
        cross_h_loss = tf.keras.losses.binary_crossentropy(indicators, c)
        return nll_loss + self.eta * cross_h_loss

    def mixed_uniform_mask_pred_loss(self, y_true, y_pred):
        # the mask index is also predicted and the nll loss depends on the predicted mask index
        disp_est, disp_gt, indicators = tf.split(y_true, num_or_size_splits=3, axis=1)
        # set all variables to one dimensional
        disp_est = tf.squeeze(disp_est)
        disp_gt = tf.squeeze(disp_gt)
        indicators = tf.squeeze(indicators)
        c = tf.squeeze(y_pred[0])
        s = tf.squeeze(y_pred[1])

        # negative log likelihood
        diff = tf.abs(disp_gt - disp_est)
        diff_1 = diff - math.sqrt(3) * tf.math.exp(s)
        diff_1_abs = tf.abs(diff_1)

        loss_good = math.sqrt(2) * diff * tf.math.exp(-s) + s
        loss_tmp = tf.where(tf.greater(c, 0.5), loss_good, diff_1_abs - 0.5)
        condition = tf.logical_and(tf.less_equal(c, 0.5), tf.less(diff_1_abs, 1.0))
        nll_loss = tf.math.reduce_mean(tf.where(condition, 0.5 * tf.square(diff_1), loss_tmp))

        # binary cross-entropy
        cross_h_loss = tf.keras.losses.binary_crossentropy(indicators, c)
        return nll_loss + self.eta * cross_h_loss

    def gmm_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        # mixture parameter
        phi = tf.squeeze(y_pred[0])
        # shape: (128, 3)
        mu = tf.squeeze(y_pred[1])
        s = tf.squeeze(y_pred[2])
        # negative log likelihood loss
        eps = 1e-7

        likelihood = tf.math.log(phi + eps) - 0.5 * tf.math.log(2 * math.pi) - 0.5 * s - 0.5 * tf.square(disp_gt - mu) * tf.exp(-s)
        nll_loss = - tf.reduce_mean(tf.reduce_logsumexp(likelihood, axis=1))

        # mode shift error
        mode_shift_loss = tf.reduce_mean(tf.square(tf.squeeze(disp_est) - tf.reduce_sum(phi * mu, axis=1)))
        return nll_loss + self.gamma * mode_shift_loss

    def lmm_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        # mixture parameter
        phi = tf.squeeze(y_pred[0])
        mu = tf.squeeze(y_pred[1])
        s = tf.squeeze(y_pred[2])
        # negative log likelihood loss
        eps = 1e-7

        likelihood = tf.math.log(phi + eps) - tf.math.log(2.0) - s - tf.abs(disp_gt - mu) * tf.exp(-s)
        nll_loss = - tf.reduce_mean(tf.reduce_logsumexp(likelihood, axis=1))

        # mode shift error
        mode_shift_loss = tf.reduce_mean(tf.abs(tf.squeeze(disp_est) - tf.reduce_sum(phi * mu, axis=1)))
        # tf.print('nll loss: ', nll_loss, summarize=-1)
        # tf.print('mode shift error: ', mode_shift_loss, summarize=-1)
        return nll_loss + mode_shift_loss

    def lmm_nnelu_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        # mixture parameter
        phi = tf.squeeze(y_pred[0])
        mu = tf.squeeze(y_pred[1])
        sigma = tf.squeeze(y_pred[2])
        # negative log likelihood loss
        eps = 1e-7

        likelihood = tf.math.log(phi + eps) - tf.math.log(2 * sigma + eps) - tf.abs(disp_gt - mu) / (sigma + eps)
        nll_loss = - tf.reduce_mean(tf.reduce_logsumexp(likelihood, axis=1))

        # mode shift error
        mode_shift_loss = tf.reduce_mean(tf.abs(tf.squeeze(disp_est) - tf.reduce_sum(phi * mu, axis=1)))
        # tf.print('nll loss: ', nll_loss, summarize=-1)
        # tf.print('mode shift error: ', mode_shift_loss, summarize=-1)
        return nll_loss + mode_shift_loss

    def gaussian_uniform_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        alpha = tf.squeeze(y_pred[0])
        s = tf.squeeze(y_pred[1])
        s_u = s if len(y_pred) < 3 else tf.squeeze(y_pred[2])
        disp_est = tf.squeeze(disp_est)
        disp_gt = tf.squeeze(disp_gt)

        diff = tf.abs(disp_gt - disp_est)
        diff_1 = diff - tf.sqrt(3 * tf.exp(s_u))
        diff_1_abs = tf.abs(diff_1)

        gaussian_loss = 0.5 * (tf.square(diff) * tf.math.exp(-s) + s)
        uniform_loss_tmp = 0.5 * tf.square(diff_1)
        uniform_loss = tf.where(tf.less(diff_1_abs, 1.0), uniform_loss_tmp, diff_1_abs - 0.5)

        loss = alpha * gaussian_loss + (1.0 - alpha) * uniform_loss
        return tf.reduce_mean(loss)

    def laplacian_uniform_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        alpha = tf.squeeze(y_pred[0])
        s = tf.squeeze(y_pred[1])
        s_u = s if len(y_pred) < 3 else tf.squeeze(y_pred[2])
        disp_est = tf.squeeze(disp_est)
        disp_gt = tf.squeeze(disp_gt)

        diff = tf.abs(disp_gt - disp_est)
        diff_1 = diff - math.sqrt(3) * tf.exp(s_u)
        diff_1_abs = tf.abs(diff_1)

        laplacian_loss = math.sqrt(2) * diff * tf.math.exp(-s) + s
        uniform_loss_tmp = 0.5 * tf.square(diff_1)
        uniform_loss = tf.where(tf.less(diff_1_abs, 1.0), uniform_loss_tmp, diff_1_abs - 0.5)

        loss = alpha * laplacian_loss + (1.0 - alpha) * uniform_loss
        return tf.reduce_mean(loss)

    def laplacian_uniform_l1(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        alpha = tf.squeeze(y_pred[0])
        s = tf.squeeze(y_pred[1])
        s_u = s if len(y_pred) < 3 else tf.squeeze(y_pred[2])
        disp_est = tf.squeeze(disp_est)
        disp_gt = tf.squeeze(disp_gt)

        diff = tf.abs(disp_gt - disp_est)
        diff_1 = diff - math.sqrt(3) * tf.exp(s_u)

        laplacian_loss = math.sqrt(2) * diff * tf.math.exp(-s) + s
        uniform_loss = tf.abs(diff_1)

        loss = alpha * laplacian_loss + (1.0 - alpha) * uniform_loss
        return tf.reduce_mean(loss)

    def generic_loss(self):
        """ Wrapper for a generic loss functions, which computes the loss based on its Metrics object configuration.

        @return: Float value representing the computed loss.
        """
        def generic_loss_wrapped(y_true, y_pred):
            if self.basic_loss == 'Binary_Cross_Entropy':
                loss = self.weighted_binary_crossentropy(y_true, y_pred)
                #loss_tf = self.binary_crossentropy(y_true, y_pred)
            elif self.basic_loss == 'Probabilistic':
                loss = self.probabilistic_loss(y_true, y_pred)
            elif self.basic_loss == 'Residual':
                loss = self.residual_loss(y_true, y_pred)
            elif self.basic_loss == 'Residual_Abs':
                loss = self.residual_loss_abs(y_true, y_pred)
            elif self.basic_loss == 'Mixed_Uniform':
                loss = self.mixed_uniform_loss(y_true, y_pred)
            elif self.basic_loss == 'Mixed_Uniform_l1':
                loss = self.mixed_uniform_l1(y_true, y_pred)
            elif self.basic_loss == 'Mixed_Uniform_mask':
                loss = self.mixed_uniform_mask_loss(y_true, y_pred)
            elif self.basic_loss == 'Mixed_Uniform_mask_pred':
                loss = self.mixed_uniform_mask_pred_loss(y_true, y_pred)
            elif self.basic_loss == 'GMM':
                loss = self.gmm_loss(y_true, y_pred)
            elif self.basic_loss == 'LMM':
                loss = self.lmm_loss(y_true, y_pred)
            elif self.basic_loss == 'LMM_nnelu':
                loss = self.lmm_nnelu_loss(y_true, y_pred)
            elif self.basic_loss == 'Gaussian_Uniform':
                loss = self.gaussian_uniform_loss(y_true, y_pred)
            elif self.basic_loss == 'Laplacian_Uniform':
                loss = self.laplacian_uniform_loss(y_true, y_pred)
            elif self.basic_loss == 'Laplacian_Uniform_l1':
                loss = self.laplacian_uniform_l1(y_true, y_pred)
            else:
                raise Exception('Unknown loss type: %s' % self.basic_loss)

            return loss

        return generic_loss_wrapped
