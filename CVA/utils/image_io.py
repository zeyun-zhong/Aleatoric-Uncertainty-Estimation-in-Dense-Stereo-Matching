import re
import numpy as np
import sys
import cv2


def read(file):
    """Loads an image from file

    @param file: Path to the file to load
    @return: A numpy array containing the loaded image
    """
    if file.endswith('.pfm'):
        return read_pfm(file)[0]
    else:
        return read_image(file)


def write(file, data):
    """Writes an image to file

    @param file: ath to the file to write
    @param data: Numpy array containing the image to write
    @return: Return value of the called subfunctions
    """
    if file.endswith('.pfm'):
        return write_pfm(file, data)
    else:
        return write_image(file, data)


def read_pfm(file):
    """Loads an image from a pfm-file

    @param file: Path to the file to load
    @return: A numpy array containing the loaded image and the scale
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def write_pfm(file, image, scale=1):
    """Writes an image to a pfm-file

    @param file: Path to the file to write
    @param image: Numpy array containing the image to write
    @param scale: The image scale
    """
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def to_grayscale(image):
    """Converts a gray-scale image from a 3-channel to a single-channel representation

    @param image: The image to convert
    @return: The converted single-channel gray-scale image
    """
    grayscale_image = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
    grayscale_image[:, :] = image[:, :, 0]
    return grayscale_image


def read_image(file):
    """Loads an image from a common file format (e.g. jpg, png)

    @param file: Path to the file to load
    @return: A numpy array containing the loaded image and the scale
    """
    # Read image
    image = cv2.imread(file)

    # Check if it was read successfully
    if (image.size < 1):
        raise Exception('Error while loading the image: ' + file)

    # Check if it's a colour or a grayscale image
    if (image.ndim == 2):
        # It's a grayscale image
        return image

    elif (image.ndim == 3 and image.shape[2] == 1):
        # It's a grayscale image, but with 3 dimensions -> Convert!
        return to_grayscale(image)

    elif (image.ndim == 3 and image.shape[2] == 3 and np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(
            image[:, :, 0], image[:, :, 2])):
        # It's a grayscale image, but with 3 channels -> Convert!
        return to_grayscale(image)

    elif (image.ndim == 3 and image.shape[2] == 3):
        # It's a colour image
        return image

    else:
        # This is a unknown format
        raise Exception('Unknown format for image: ' + file)


def write_image(file, data):
    """Writes an image to file

    @param file: Path to the file to write
    @param data: Data that is to be written to file
    """
    if file.endswith('.pfm') or file.endswith('.PFM'):
        return write_pfm(file, data, 1)
    cv2.imwrite(file, data)
