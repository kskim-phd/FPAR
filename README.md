# FPAR

# Paper Title Pseudonymized (Under Review)

We release FPAR evaluation code.

Collaborators: Kyungsu Kim*, Seongje Oh* (Equal contribution)

Detailed instructions for testing the image are as follows.

------

# Implementation

A PyTorch implementation of FPAR based on original pytorch-gradcam code.

pytorch-gradcam [https://github.com/jacobgil/pytorch-grad-cam] (Thanks for Jacob Gildenblat and contributors.)

Attention U-Net [https://github.com/ozan-oktay/Attention-Gated-Networks] (Thanks for Ozan Oktay and contributors.)

------
## Environments

The setting of the virtual environment we used is described as packagelist.txt.

------
## CXR dataset

Please send me a request email (kskim.doc@gmail.com) for that inference sample data and pre-trained weight (As this work is under review, so it is open to reviewers only).

Put the data you received into the 'data' folder ('./data/')

------
## Our pretrained weight


Put the weight files you received into the 'checkpoint' folder ('./checkpoint/')

## Segmentation

Please run "segmentation_shoulder.py".

```
python segmentation_shoulder.py 
```

The segment mask (file name : same name) is stored in the './data/train or test/shoulder_mask/' folder.

------

## Pre-processing

Please run "preprocessing.py".

```
python preprocessing.py 
```

All data in the './data/' folder is preprocessed and stored in the './preprocessing_data/' folder.

------


## Train FPAR


The following code is used to train models with FPAR.

```
python train_FPAR.py 
```

Please refer to the code('train_FPAR.py') for detailed arguments.

------


## Validation FPAR


We provide weights of the learned EfficientNet and ShuffleNetV2 to validate against the test sample.


```
python validation_FPAR.py 
```

Please refer to the code('validation_FPAR.py') for detailed arguments.

------

## GradCAM

We provide visualization code based on ShuffleNetv2 with the highest performance visually.

Please run "Heatmap.py" and "Overlay_Heatmap_and_image.py"

```
python Heatmap.py
python Overlay_Heatmap_and_image.py
```

If you run two things sequentially, you will see that a "Full_CAM" folder is created, storing the Heatmap

------
