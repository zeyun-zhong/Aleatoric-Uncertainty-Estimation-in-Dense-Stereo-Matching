import os
import csv
import numpy as np


class CostVolume:

    def __init__(self):
        self.dimensions = (0, 0, 0)
        self.data = np.zeros((0, 0, 0))

    def dim(self):
        """ Getter for dimensions.

        @return: The dimensions of the cost volume: [height, width, depth]
        """
        return self.dimensions

    def normalise(self, additive_factor=0.0, multiplicative_factor=1.0):
        """ Normalises the cost volume inplace, using the specified factors.

        @param additive_factor: Factor that is added to all entries of the cost volume.
        @param multiplicative_factor: Factor that is multiplicated with all entries of the cost volume.
        """
        self.data = (self.data + additive_factor) * multiplicative_factor

    def get_data(self, border=0):
        """ Getter for data.

        @param border: Adds a zero border around the volume at axis 0 and 1 (height and width), not in depth.
        @return: The cost volume as 3D numpy array.
        """
        if border != 0:
            cv = np.zeros((self.dimensions[0] + (border * 2), self.dimensions[1] + (border * 2), self.dimensions[2]))
            cv[border:self.dimensions[0] + border, border:self.dimensions[1] + border, :] = self.data
            return cv
        else:
            return self.data

    def to_3d(self, data, dim):
        """ Transfers a cost volume from 1D to 3D representation.

        @param data: A 1D numpy array.
        @param dim: Dimensions of the desired 3D array.
        @return: A 3D numpy array.
        """

        if (dim[0] * dim[1] * dim[2]) != data.shape[0]:
            raise ValueError('Dimensions and data do not match!')

        data_3d = np.zeros((dim[0], dim[1], dim[2]))
        for row in range(0, dim[0]):
            for col in range(0, dim[1]):
                start_index = col * dim[2] + row * dim[1] * dim[2]
                end_index = start_index + dim[2]
                data_3d[row, col, :] = data[start_index:end_index]
        return data_3d

    def to_1d(self, data):
        """ Transfers a cost volume from 3D to 1D representation.

        @param data: A 3D numpy array.
        @return: A 1D numpy array.
        """

        dim = data.dim
        if len(dim) != 3:
            raise ValueError('Input data has to be 3D!')

        data_1d = np.zeros(dim[0] * dim[1] * dim[2])
        for row in range(0, dim[0]):
            for col in range(0, dim[1]):
                start_index = int(col * dim[2] + row * dim[1] * dim[2])
                end_index = start_index + dim[2]
                data_1d[start_index:end_index] = data[:, row, col]
        return data_1d

    def load_dat(self, path):
        """ Loads a cost volume and its dimensions from a text file

        @param path: Path to a cost volume.
        """
        # Read cost volume from file
        # dimensions: [height, width, depth]
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            dimensions = next(reader)
            data = next(reader)

        # Remove empty strings
        data = list(filter(None, data))
        
        # Transform from string to float
        dimensions = [int(i) for i in dimensions]
        data = np.asarray([float(i) for i in data])
        
        self.dimensions = dimensions
        self.data = self.to_3d(data, dimensions)

    def load_bin(self, path, height, width, depth):
        """ Load cost volume from a .bin-file.

        The format of the .bin-file follows the definition of the MC-CNN algorithm.

        @param path: Path of the .bin-file
        @param height: Heigth of the cost volume
        @param width: Width of the cost volume
        @param depth: epth of the cost volume
        """
        # Import bin-file - could contain nan and inf!
        data = np.memmap(path, dtype=np.float32, shape=(depth, height, width))
        data = np.nan_to_num(data)

        self.dimensions = (height, width, depth)
        self.data = np.zeros(self.dimensions)
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                self.data[x,y,:] = data[:,x,y]

    def write_dat(self, path):
        """ Writes data to a .dat-file.

        @param path: Path of the resulting file
        """
        data_1d = self.to_1d(self.data)

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(self.dimensions)
            writer.writerow(data_1d)

    def set_data(self, data):
        """ Set data of the cost volume.

        @warning Overwrites all previously saved data

        @param data: The data to set.
        """

        self.dimensions = data.shape
        self.data = data

    def get_excerpt(self, position, size):
        """ Extracts an excerpt of a cost volume around a specified pixel.

        @param position: Position of the pixel of interest
        @param size: Size of the excerpt (note: size = width = height)
        @return: A cost volume excerpt as 3D numpy array
        """

        # Create empty excerpt
        excerpt = np.zeros((size, size, self.dimensions[2]))
        nb_offset = ((size - 1) / 2)
        
        # Copy values from cost volume to excerpt pixel-wise
        for excerpt_row in range(0, size):
            for excerpt_col in range(0, size):
                image_row = int(position[0] + excerpt_row - nb_offset)
                image_col = int(position[1] + excerpt_col - nb_offset)

                # Check if the current pixel is within the image dimensions
                if (image_row >= 0 and image_row < self.dimensions[0] and image_col >= 0 and image_col < self.dimensions[1]):
                    # Neighnourhood pixel is within the image dimensions -> Copy to excerpt
                    excerpt[excerpt_row, excerpt_col, :] = self.get_cost_function(image_row, image_col)[:]
                
        return excerpt

    def get_cost_function(self, image_row, image_col):
        """ Extract a single cost funtion from a cost volume.

        @param image_row: Row of the pixel for which the cost function should be extracted
        @param image_col: Column of the pixel for which the cost function should be extracted
        @return: The specified cost function as 1D numpy array
        """
        return self.data[image_row, image_col, :]

    def reduce_depth(self, new_depth):
        if new_depth > self.dimensions[2]:
            raise ValueError('New depth is larger than the original one!')

        self.set_data(self.data[:,:,0:new_depth])
