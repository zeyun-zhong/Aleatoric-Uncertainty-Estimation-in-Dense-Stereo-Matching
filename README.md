## Aleatoric-Uncertainty-Estimation-in-Dense-Stereo-Matching
### Abstract
The ability to identify erroneous depth estimates is of fundamental interest. Information regarding the aleatoric uncertainty of depth
estimates can be, for example, used to support the process of depth reconstruction itself. Consequently, various methods for the
estimation of aleatoric uncertainty in the context of dense stereo matching have been presented in recent years, with deep learning-
based approaches being particularly popular. Among these deep learning-based methods, probabilistic strategies are increasingly
attracting interest, because the estimated uncertainty can be quantified in pixels or in metric units due to the consideration of
real error distributions. However, existing probabilistic methods usually assume a unimodal distribution to describe the error
distribution while simply neglecting cases in real-world scenarios that could violate this assumption. To overcome this limitation, we
propose two novel mixed probability models consisting of Laplacian and Uniform distributions for the task of aleatoric uncertainty
estimation. In this way, we explicitly address commonly challenging regions in the context of dense stereo matching and outlier
measurements, respectively. To allow a fair comparison, we adapt a common neural network architecture to investigate the effects
of the different uncertainty models. In an extensive evaluation using two datasets and two common dense stereo matching methods,
the proposed methods demonstrate state-of-the-art accuracy.

### Installation
Environment is described [here](CVA/environment.yml). If your anaconda env is already activated, use:
```
cd CVA
conda env update --file environment.yml
```
Or update a specific environment without activating it:
```
cd CVA
conda env update --name envname --file environment.yml
```


### Data structure
The code of this repository uses the following data structure.
#### Train
```
train-folder    
│
└───disp_gt
│   │   ...
│
└───mask_indicator
│   │   ...
│
└───cv_Census-BM
│   │   ...
│
└───cv_MC-CNN
    │   ...
```

#### Test
```
test-folder    
│
└───disp_gt
│   │   ...
│
└───mask_indicator
│   │   ...
│
└───left-image-folder
│   │   ...
│
└───cv_Census-BM
│   │   ...
│
└───cv_MC-CNN
    │   ...
```

#### Evaluation
```
results-folder    
│
└───sample folder
│   │   uncertainty_maps (.pfm)
│   │   est_disparity_maps (.png)
│
└───...
    │   ...
    │   ...
```

### How to use the code
#### Train
Please note that the opts are defined [here](CVA/utils/opts.py). Three opts are required for the training process: 1) whether the code runs local or in cluster 2) cv_method: stereo matching method to generate cost volumes 3) loss_type. 
```
cd CVA
./train_local.sh
```

#### Test
```
cd CVA/CVA-Net
python Test-CVA-Net.py
```

#### Evaluation
Run the following two files in Matlab to evaluate the uncertainty values generated by the network.
```
Abs_error_uncer/Error_Unc_Heatmap.m
AUC_Evaluation/EvaluateAUC.m
```

### Publication
Please cite the following paper if you use the idea of this paper or parts of this code in your own work.
```
@inproceedings{zhong2021unc,
  title={{Mixed Probability Models for Aleatoric Uncertainty Estimation in the Context of Dense Stereo Matching}},
  author={Zhong, Zeyun and Mehltretter, Max},
  booktitle={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  year={2021}
}
```
