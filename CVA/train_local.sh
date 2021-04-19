#!/bin/bash

cd CVA-Net

python Train-CVA-Net.py --no-cluster --dataset 'M3' --cv_method 'MC-CNN' --loss_type 'Mixed_Uniform'
