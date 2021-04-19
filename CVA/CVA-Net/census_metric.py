import numpy as np
from bitarray import bitarray
import struct
import cv2
import cost_volume


# @brief Implements the Census matching metric
class CensusMetric:
    
    
    # @param metric_filter_size   Size of the census filter (is used for width and height)
    # @param smooth_filter_size   Size of the smoothness filter (is used for width and height)
    def __init__(self, metric_filter_size, smooth_filter_size):
        self.smooth_filter_size = smooth_filter_size
        self.metric_filter_size = metric_filter_size
        self.metric_entries = (self.metric_filter_size * self.metric_filter_size) - 1
  

    # @brief Transforms an image extract using Census transformation
    # @param image_extract   Image extract that should be transformed
    # @return The integer encoded census masks for the center pixel of the provided extract
    def __census_trafo__(self, image_extract):
                              
        # Check if census mask fits in 32 bit
        extract_size = image_extract.size
        if (extract_size > 32):
            raise Exception('Unsupported metric neighbourhood size!')
        
        # Seperate neighbourhood and center pixel
        nb = np.zeros(extract_size - 1, dtype=image_extract.dtype)
        center_pos = int((extract_size - 1) / 2)
        extract_idx = 0
        nb_idx = 0
        center = 0
               
        for extract_entry in np.nditer(image_extract, order = 'C'):
            if (extract_idx != center_pos):
                nb[nb_idx] = extract_entry
                nb_idx += 1
            else:
                center = extract_entry
            extract_idx += 1
                      
        # Compute census transformation
        trafo_bit = bitarray('0' * 32, endian='little')           
        for bit_idx in range(0, nb.size):
            trafo_bit[bit_idx] = (nb[bit_idx] <= center)          
        return int(struct.unpack("<L", trafo_bit)[0])       
    
    
    # @brief Transforms an image using Census transformation
    # @param image   Image that should be transformed
    # @return The image transformation in form of pixel-wise integer encoded census masks
    def __create_census_trafo__(self, image):
        
        # Create array to store transformation
        image_size = image.shape
        census_trafo = np.zeros(image_size, dtype=int)
        
        # Add a padding to the image to also get the trasformation for the borders
        padding = int((self.metric_filter_size - 1) / 2)
        padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
        
        # Compute transformation
        for y in range(0, image_size[0]):
            for x in range(0, image_size[1]):
                census_trafo[y,x] = self.__census_trafo__(padded_image[y:y+2*padding+1, x:x+2*padding+1])               
                    
        return census_trafo
    
    
    # @brief Computes a cost curve based on hamming distance using two census transformations
    # @param cost_curve   Cost curve that should be computed
    # @param left_trafo   Census transformation of left image
    # @param right_trafo  Census transformation of right image
    # @param row          Row of pixel of interest in left image
    # @param col          Column of pixel of interest in left image
    def __compute_cost_curve__(self, cost_curve, left_trafo, right_trafo, row, col):
        
        # Determine max index for which the metric should be computed
        disp_levels = cost_curve.shape[0]
        max_idx = min(col+1, disp_levels)     
        
        # Compute metric values
        left_trafo_pixel = left_trafo[row, col]
        right_trafo_line = right_trafo[row, :]        
        for d in range(0, max_idx):
            cost_curve[d] = (bin(left_trafo_pixel ^ right_trafo_line[col - d]).count("1") 
                             / self.metric_entries)
                      
        # Fill positions of the cost curve which are out of the right image
        cost_curve[range(max_idx, disp_levels)] = cost_curve[max_idx - 1]
    
    
    # @brief Creates a cost volume extract based on the provided image transformations
    # @param left_trafo   Left census trafo
    # @param right_trafo  Right census trafo
    # @param row          Row of the point of interest in the left image
    # @param col          Column of the point of interest in the left image
    # @param extract_dim  Array containing the 3 dimensions of the resulting extract
    # @return A cost volume extract
    def __compute_cv_extract__(self, left_trafo, right_trafo, row, col, extract_dim):
        
        # Create initial extract (bigger than final one because of ability to smooth)     
        margin_y = int((self.smooth_filter_size - 1) / 2)
        margin_x = int((self.smooth_filter_size - 1) / 2)
        
        cv_extract = np.zeros((extract_dim[0] + self.smooth_filter_size - 1, 
                              extract_dim[1] + self.smooth_filter_size - 1,
                              extract_dim[2]), dtype=float)
        
        # Compute the metric values
        img_dim = left_trafo.shape
        cv_dims = cv_extract.shape
        y_offset = int(row - ((cv_dims[0] - 1) / 2))
        x_offset = int(col - ((cv_dims[1] - 1) / 2))
        
        for y in range(0, cv_dims[0]):
            for x in range(0, cv_dims[1]):
                
                y_abs = y_offset + y
                x_abs = x_offset + x               
                if (y_abs >= 0 and x_abs >= 0 and x_abs < img_dim[1] and y_abs < img_dim[0]):         
                    self.__compute_cost_curve__(cv_extract[y, x, :], left_trafo, right_trafo,
                                                y_abs, x_abs)
              
        # Apply 5x5 box filter
        if (self.smooth_filter_size > 1):
            cv_extract = cv2.boxFilter(cv_extract, -1, (self.smooth_filter_size, self.smooth_filter_size))        
        return cv_extract[margin_y:-margin_y, margin_x:-margin_x, :]
    
    
    # @brief Creates a cost volume object based on the provided image transformations
    # @param left_trafo         Left census trafo
    # @param right_trafo        Right census trafo
    # @param disparity_levels   Number of disparity levels that should be computed (= depth of cost volume)
    # @return A cost volume object
    def __compute_cost_volume__(self, left_trafo, right_trafo, disparity_levels):
        
        # Compute cost volume
        cost_volume_data = np.zeros((left_trafo.shape[0], left_trafo.shape[1], disparity_levels), dtype=float)        
        for y in range(0, cost_volume_data.shape[0]):
            for x in range(0, cost_volume_data.shape[1]):
                self.__compute_cost_curve__(cost_volume_data[y, x, :], left_trafo, right_trafo, y, x)
                
        # Smooth it (to apply block matching) if smoothing filter is set
        if (self.smooth_filter_size > 1):
            cost_volume_data = cv2.boxFilter(cost_volume_data, -1, 
                                             (self.smooth_filter_size, self.smooth_filter_size))
            
        # Transfer data to cost volume object
        cv = cost_volume.CostVolume()
        cv.set_data(cost_volume_data)
        return cv