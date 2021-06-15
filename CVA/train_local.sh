#!/bin/bash

cd CVA-Net

python Train-CVA-Net.py --no-cluster --cv_method 'MC-CNN' --loss_type 'Mixture'