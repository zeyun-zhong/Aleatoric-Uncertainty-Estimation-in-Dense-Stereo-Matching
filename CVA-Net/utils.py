class KITTI():
    # @brief Checks if a disparity estimation is correct using the KITTI stereo metric.
    # @param [in]   est   Disparity estimation of a single pixel
    # @param [in]   gt    Corresponding ground truth disparity
    # @return  True, if the estimation is correct. Otherwise, false.
    @staticmethod
    def is_correct(est, gt):
        est = float(est)
        gt = float(gt)    
        abs_dff = abs(gt - est)
        return not((abs_dff > 3) and (abs_dff/abs(gt) > 0.05))
    
    
    # @brief Creates the KITTI file name out of the image number
    # @param img_number Number of the image, a name should be created for
    # @return Image file name
    def get_name(self, img_number):
        img_name = str(img_number)
        while (len(img_name) < 6):
            img_name = '0' + img_name
        return img_name + '_10'
    
    
    # @brief Creates a KITTI sample that can be used for training/validation of CVA-Net
    # @details A sample consists of: left image path, right image path, reference disparity map path, factor to normalise the reference disparity map to the intervall [0, 255]
    # @param dataset_path Root path to the KITTI dataset
    # @param left_cam Name of the left camera
    # @param right_cam Name of the right camera
    # @param disp_gt Name of the reference disparity map
    # @param img_idx Image number for which the sample should be created
    # @return A KIITI sample for the specified camera and image details
    def get_sample(self, dataset_path, left_cam, right_cam, disp_gt, img_idx):
        gt_norm_factor = (1.0/256.0)
        img_name = self.get_name(img_idx)
        left_image = dataset_path + left_cam + img_name + '.png'
        right_image = dataset_path + right_cam + img_name + '.png'
        gt_disp = dataset_path + disp_gt + img_name + '.png'
        return [left_image, right_image, gt_disp, gt_norm_factor]
    
    
class Middlebury():
    # @brief Creates a Middlebury sample that can be used for training/validation of CVA-Net
    # @details A sample consists of: left image path, right image path, reference disparity map path, factor to normalise the reference disparity map to the intervall [0, 255]
    # @param dataset_path Root path to the Middlebury dataset
    # @param img_name Image name for which the sample should be created
    # @param gt_name Naming of the reference disparity map
    # @return A Middlebury sample for the specified image details
    def get_sample(self, dataset_path, img_name, gt_name):
        gt_norm_factor = 1.0
        left_image = dataset_path + img_name + 'im0.png'
        right_image = dataset_path + img_name + 'im1.png'
        gt_disp = dataset_path + img_name + gt_name
        return [left_image, right_image, gt_disp, gt_norm_factor]
    
    
class Flyingthings():
    # @brief Creates a Flyingthings sample that can be used for training/validation of CVA-Net
    # @details A sample consists of: left image path, right image path, reference disparity map path, factor to normalise the reference disparity map to the intervall [0, 255]
    # @param dataset_path Root path to the Flyingthings dataset
    # @param mode Mode for which the sample should be created (TRAIN, TEST)
    # @param set_name Set number for which the sample should be created (A, B, C)
    # @param scene_num Scene number for which the sample should be created
    # @param sample_num Image number for which the sample should be created
    # @return A Flyingthings sample for the specified image details
    def get_sample(self, dataset_path, mode, set_name, scene_num, sample_num):
        gt_norm_factor = 1.0
        scene_name = str(scene_num)
        while (len(scene_name) < 4):
            scene_name = '0' + scene_name 
        scene_name = mode + '/' + set_name + '/' + scene_name + '/'

        left_image = dataset_path + 'frames_finalpass/' + scene_name + 'left/' + sample_num + '.png'
        right_image = dataset_path + 'frames_finalpass/' + scene_name + 'right/' + sample_num + '.png'
        gt_disp = dataset_path + 'disparity/' + scene_name + 'left/' + sample_num + '.pfm'
        return [left_image, right_image, gt_disp, gt_norm_factor]