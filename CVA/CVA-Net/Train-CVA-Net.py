import time
import os
import sys
import logging
logging.getLogger().setLevel(logging.INFO)

sys.path.insert(1, '../utils')
from opts import parse_train_opts
from train import Train
import data_list


args = parse_train_opts()
logging.info(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

root_path = '/bigwork/nhgnzhon/Data/' if args.cluster else '/home/zeyun/Projects/CVA/'
logging.debug(root_path)

# ---------------------------
# initialization
# ---------------------------
Dataset = data_list.Dataset(args.dataset, root_path)
CVMethod = data_list.CVMethod(args.cv_method, Dataset)
WeightLoss = None
if args.using_weighted_loss:
    WeightLoss = data_list.WeightLoss(CVMethod=CVMethod, using_lookup=args.using_lookup, gauss_std=4)

# ---------------------------
# determine the number of training and validation samples
# ---------------------------
if args.dataset == 'Sceneflow':
    train_end_idx = 5
elif args.cv_method == 'MC-CNN' and args.dataset == 'K12':
    train_end_idx = 21
else:
    train_end_idx = 20
val_end_idx = 8 if args.dataset == 'Sceneflow' else 23

# ---------------------------
# create data list
# ---------------------------
DataList = data_list.IDataListCreator(args, Dataset, CVMethod, train_end_idx=train_end_idx, val_end_idx=val_end_idx, WeightLoss=WeightLoss)

# ---------------------------
# start training
# ---------------------------
root_dir = root_path + 'experiments'
network_name = "CVA-Net_{0}".format(int(time.time())) if not args.name else "CVA-Net_" + args.name
pretrained_network = ''

print('Initialising...')
trainer = Train(parameter=DataList.parameter, network_name=network_name, root_dir=root_dir,
                experiment_series='dynamic-depth', pretrained_network=pretrained_network)

print('Start training...')
trainer.train()