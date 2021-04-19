import numpy as np
from image_io import write, read
import os
from sklearn.metrics import confusion_matrix


def calc_mask_pred_acc(model_name, disp_gt_path, mask_gt_path, mask_pred_path, thresh=0.5):
    number_of_corr_list, number_of_total_list = [], []
    TP_list, TN_list, FP_list, FN_list = [], [], [], []
    image_names = sorted(os.listdir(mask_pred_path))
    for image_name in image_names:
        disp_gt = read(os.path.join(disp_gt_path, image_name + '.png'))
        mask_gt = read(os.path.join(mask_gt_path, image_name + '.png'))
        mask_pred = np.load(os.path.join(mask_pred_path, image_name, model_name + '.npy'))
        mask_pred_thresh = np.ones_like(mask_pred)
        mask_pred_thresh[mask_pred <= thresh] = 0

        # select only pixels with gt disparities
        gt_index = np.where(disp_gt > 0)
        mask_pred_thresh_valid = mask_pred_thresh[gt_index]
        mask_gt_valid = mask_gt[gt_index]

        # compute accuracy and confusion matrix
        number_of_corr = (mask_pred_thresh_valid == mask_gt_valid).sum()
        TN, FP, FN, TP = confusion_matrix(mask_gt_valid, mask_pred_thresh_valid).ravel()

        number_of_corr_list.append(number_of_corr)
        number_of_total_list.append(len(mask_gt[gt_index]))
        TN_list.append(TN)
        FP_list.append(FP)
        FN_list.append(FN)
        TP_list.append(TP)

    acc = sum(number_of_corr_list) / sum(number_of_total_list)
    TP_res, FP_res, FN_res, TN_res = sum(TP_list), sum(FP_list), sum(FN_list), sum(TN_list)
    return acc, TP_res, FP_res, FN_res, TN_res


if __name__ == '__main__':
    dataset = 'middlebury-v3'  # or 'middlebury-v3'

    disp_gt_path = '/media/zeyun/ZEYUN/MA/{}/disp_gt/'.format(dataset)
    mask_gt_path = '/media/zeyun/ZEYUN/MA/{}/mask_indicator_wo_disc/'.format(dataset)
    mask_pred_path = '/home/zeyun/Projects/CVA/results/{}/'.format(dataset)

    acc, TP, FP, FN, TN = calc_mask_pred_acc('CVA-Net_Mixed_Uniform_mask_paper_MC-CNN_M3_mask', disp_gt_path, mask_gt_path, mask_pred_path)
    print(acc)
    print([TP, FP, FN, TN])