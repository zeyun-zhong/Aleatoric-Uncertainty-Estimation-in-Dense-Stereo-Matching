#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --mail-user=zeyun.zhong@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=NONE

# Change to my work dir
cd $BIGWORK/CVA/CVA-Net

# Load modules
module load fosscuda/2019b TensorFlow/2.2.0-Python-3.7.4 matplotlib/3.1.1-Python-3.7.4 OpenCV/4.2.0-Python-3.7.4

# Run GPU application
python Train-CVA-Net.py --cluster --name 'Mixed_Uniform_mask_pred_paper' --cv_method 'Census-BM' --eta 1.0 --loss_type 'Mixed_Uniform_mask_pred'
