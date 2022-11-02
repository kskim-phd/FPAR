# FPAR

# Artificial Intelligence-assisted Analysis to Facilitate Effective Detection of Humeral Lesions in Chest Radiograph

We release FPAR evaluation code.

Collaborators: Kyung-su Kim*, Seongje Oh* (Equal contribution)

Detailed instructions for testing the image are as follows.

------

# Implementation

A PyTorch implementation of FPAR based on original pytorch-gradcam code.

pytorch-gradcam [https://github.com/jacobgil/pytorch-grad-cam] (Thanks for Jacob Gildenblat and contributors.)
Attention U-Net [] (Thanks for - and contributors.)

------
## Environments

The setting of the virtual environment we used is described as packagelist.txt.

------
## CXR dataset

Please send me a request email (kskim.doc@gmail.com) for that inference sample data (As this work is under review, so it is open to reviewers only).

------
## N/A diagnosis (5/2)

Please downloading the pre-trained weight file [here](https://drive.google.com/file/d/198TmyO5YtXlO-Acn5VE16n_52s5bscSb/view?usp=sharing). 
Please run "Classification/N_A_inference.py"

```
python N_A_inference.py 
```
You will see result of baseline and proposed(N/A)

------
## Segmentation

Put the test data in the "dataset" folder to create a split mask. please downloading the pre-trained weight file [here](https://drive.google.com/file/d/1Mqs8HA8vjrClaVNMvUbEL__cPPm90scX/view?usp=sharing).  
Please run "Segmentation/mask_maker.py".

```
python mask_maker.py 
```
The segment mask (file name : same name+"mask.jpg") is stored in the same folder.

------

## GradCAM

Please run "Heatmap.py"

```
python Heatmap.py
python Overlay_Heatmap_and_image.py
```

If you run four things sequentially, you will see that a "Full_CAM" folder is created, storing the Heatmap

------
