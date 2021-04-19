class Params():
    def __init__(self):
        # Training parameters
        self.epochs = 20
        self.learning_rate = 0.0001
        self.batch_size = 128
        self.data_mode = 'cv'                       # 'cv', 'image' - defines the used data generator
        self.training_start_epoch = 0
        self.loss_type = 'Binary_Cross_Entropy'
        self.pos_class_weight = 1.0
        self.cv_norm = [0.0, 1.0]
        
        # Architecture parameters
        self.nb_size = 13     
        self.cost_volume_depth = 192
        self.nb_filter_num = 32
        self.depth_filter_num = 32
        self.depth_layer_num = 10
        self.dense_filter_num = 16     
        self.dense_layer_num = 0
        self.task_type = 'Classification'           # 'Classification', 'Regression' - defines the network output
        self.nb_filter_size = 3 # Zeyun added param, origin is 3, changed to 5 to save training time
        self.dense_layer_type = 'FC' # Zeyun added param, origin is 'FC', 'AP'
        self.last_dp_filter_num = 32 # Zeyun added param, origin is 32, changed to 64 to increase the nodes connected
        # to the last regression node

        # coefficient for the binary cross-entropy loss term of the geometry-aware model
        self.eta = 1.0

        # GMM components
        self.K = 3
        self.gamma = 1.0

        # Laplacian Uniform parameters
        self.lu_out = 2 # 2: stds of Laplacian and Uniform are the same 3: different std for Laplacian and Uniform

        # Data
        self.training_data = ''
        self.validation_data = ''

    def to_string(self):
        param_string = ''
        param_string += 'Learning rate: ' + str(self.learning_rate) + '\n'
        param_string += 'Loss type: ' + self.loss_type + '\n'
        param_string += 'Task type: ' + self.task_type + '\n'
        param_string += 'Neighbourhood size: ' + str(self.nb_size) + '\n'
        param_string += 'Cost volume depth: ' + str(self.cost_volume_depth) + '\n'
        param_string += '# Neighbourhood filter: ' + str(self.nb_filter_num) + '\n'
        param_string += '# Depth filter: ' + str(self.depth_filter_num) + '\n'
        param_string += '# Depth layers: ' + str(self.depth_layer_num) + '\n'
        param_string += '# Dense filter: ' + str(self.dense_filter_num) + '\n'
        param_string += '# Additional dense layers: ' + str(self.dense_layer_num) + '\n'
        param_string += '# Neighbourhood filter size: ' + str(self.nb_filter_size) + '\n'
        param_string += '# Dense layer type: ' + str(self.dense_layer_type) + '\n'
        param_string += '# Last Depth layer filter num: ' + str(self.last_dp_filter_num) + '\n'
        return param_string