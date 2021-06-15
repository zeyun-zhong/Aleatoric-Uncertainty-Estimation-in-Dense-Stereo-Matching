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

### Folder structure
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