import numpy as np
import cv2
import image_io

def normalise_image(image, src_min = 0.0, src_max = 255.0, dest_min = -1.0, dest_max = 1.0):
    """ Transforms the values within an image from a specified source range to a specified destination range.

    @param image: Numpy array representing an image
    @param src_min: Minimum value of the source range
    @param src_max: Maximum value of the source range
    @param dest_min: Minimum value of the destination range
    @param dest_max: Maximum value of the destination range
    @return: The transformed image
    """
    scale_factor = (src_max - src_min) / (dest_max - dest_min)
    normalised_image = (image - src_min) * (1.0 / scale_factor) + dest_min
    return normalised_image


def resize_image(image, resize_factor, divisibility = 1):
    """ Resizes an image to a format which is suitable for a specific network architecture.

    @param image: Input image.
    @param resize_factor: Factor which is applied to the image first in order to change it size.
    @param divisibility: The image is shrunk independently in x- and y- direction so that it's dimensions are dividable
    by the specified factor.
    @return: The resized image, and the scale factors in x- and y- direction that were applied in total.
    """
    orig_size = image.shape
    dest_x = resize_factor * orig_size[0]
    dest_y = resize_factor * orig_size[1]
    diff_x = dest_x % divisibility
    diff_y = dest_y % divisibility
    scale_factor_height = (dest_x - diff_x) / orig_size[0]
    scale_factor_width = (dest_y - diff_y) / orig_size[1]
    resized_image = cv2.resize(image, (0, 0), fy=scale_factor_height, fx=scale_factor_width,
                            interpolation=cv2.INTER_AREA)
    return resized_image, scale_factor_height, scale_factor_width


def save_disparity_map(image, file_path, scale_factor_height = 1.0, scale_factor_width = 1.0, save_png = True,
                       file_extension = '.pfm'):
    """ Saves a disparity map to file.

    @param image: Image to be written to file.
    @param file_path: Path of the resulting file.
    @param scale_factor_height: Height factor that was applied to shrink the input images.
    @param scale_factor_width: Width factor that was applied to shrink the input images.
    @param save_png: Specifies if the disparity map should additionally be saved in png format with rounded values.
    @param file_extension: Specifies the file extension used to save the results
    """
    image = cv2.resize(image, (0, 0), fy=(1.0 / scale_factor_height), fx=(1.0 / scale_factor_width))
    image = image * (1.0 / scale_factor_width)
    image_io.write(file_path + file_extension, image.astype(np.float32))

    if save_png:
        image_io.write(file_path + '.png', disp_to_img(image))


def disp_to_img(mat):
    sample = np.round(mat)
    return sample.astype(np.uint8)


# TODO: Fuse metric definitions with the ones in metrics.py to avoid double definitions
def mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff_nz = diff[y_true.astype(dtype=bool)]
    return np.mean(diff_nz)


def rmse(y_true, y_pred):
    sqr_diff = (y_true - y_pred) ** 2
    sqr_diff_nz = sqr_diff[y_true.astype(dtype=bool)]
    mean = np.mean(sqr_diff_nz)
    return np.sqrt(mean)


def pixel_error(y_true, y_pred, threshold):
    diff = np.abs(y_true - y_pred)
    diff_nz = diff[y_true.astype(dtype=bool)]
    return (np.count_nonzero(np.greater(diff_nz, threshold)) / diff_nz.size)


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, label):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        label (numpy.ndarray): list of labels (0 or 1)

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    accuracy, avg_conf, len_bin = [], [], []
    if conf.ndim > 1:
        for i in range(conf.shape[0]):
            cur = conf[i]
            index = np.where(np.logical_and(cur > conf_thresh_lower, cur <= conf_thresh_upper))
            filtered_tuples = cur[index]
            if len(filtered_tuples) < 1:
                accuracy.append(0)
                avg_conf.append(0)
                len_bin.append(0)
            else:
                len_b = len(filtered_tuples)
                avg_c = np.sum(filtered_tuples) / len_b
                acc = np.sum(label[index]) / len_b
                accuracy.append(acc)
                avg_conf.append(avg_c)
                len_bin.append(len_b)
    else:
        index = np.where(np.logical_and(conf > conf_thresh_lower, conf <= conf_thresh_upper))
        filtered_tuples = conf[index]
        if len(filtered_tuples) < 1:
            accuracy.append(0)
            avg_conf.append(0)
            len_bin.append(0)
        else:
            len_b = len(filtered_tuples)
            avg_c = np.sum(filtered_tuples) / len_bin
            acc = np.sum(label[index]) / len_bin
            accuracy.append(acc)
            avg_conf.append(avg_c)
            len_bin.append(len_b)
    return accuracy, avg_conf, len_bin


def ECE(conf, label, bin_size=0.1):
    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)

    Returns:
        ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf, label)
        ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE

    return ece


def get_bin_info(conf_good, conf_hard, label_good, label_hard, bin_size=0.1):
    """
    Get accuracy, confidence and elements in bin information for all the bins.

    Args:
        conf (numpy.ndarray): list of confidences
        label (numpy.ndarray): list of labels (0 or 1), whether an estimation is correct or not
        bin_size: (float): size of one bin (0,1)

    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    n = 1 if conf_good.ndim == 1 else conf_good.shape[0]

    accuracies_good = [[] for _ in range(n)]
    accuracies_hard = [[] for _ in range(n)]
    confidences_good = [[] for _ in range(n)]
    confidences_hard = [[] for _ in range(n)]
    bin_lengths_good = [[] for _ in range(n)]
    bin_lengths_hard = [[] for _ in range(n)]

    for conf_thresh in upper_bounds:
        acc_good, avg_conf_good, len_bin_good = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf_good, label_good)
        acc_hard, avg_conf_hard, len_bin_hard = compute_acc_bin(conf_thresh - bin_size, conf_thresh, conf_hard, label_hard)
        for i in range(n):
            accuracies_good[i].append(acc_good[i])
            confidences_good[i].append(avg_conf_good[i])
            bin_lengths_good[i].append(len_bin_good[i])
            accuracies_hard[i].append(acc_hard[i])
            confidences_hard[i].append(avg_conf_hard[i])
            bin_lengths_hard[i].append(len_bin_hard[i])

    return accuracies_good, accuracies_hard, confidences_good, confidences_hard, bin_lengths_good, bin_lengths_hard


def standardize(data):
    """
    data: 1d numpy array
    """
    if data.ndim > 1:
        raise ValueError('not a 1d data')
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

