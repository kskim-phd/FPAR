from __future__ import print_function
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision import models, transforms
import matplotlib.pyplot as plt
import argparse
import cv2,os
from utils import *
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM, \
                             LayerCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

import torch.nn as nn
import warnings
warnings.filterwarnings(action='ignore')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam++',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


test_transform=transforms.Compose([

        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])
if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    num_class=2
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM}


    weightfile='./checkpoint/ShuffleNetV2_weight.pth'

    output_dir='./CAM/'
    os.makedirs(output_dir,exist_ok=True)
    model = TEST_FPAR_shuffleNET_v2(num_class)
    model = nn.DataParallel(model).to(device)
    checkpoint = torch.load(weightfile)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    target_layer = 'module.net.conv5.2'
    target_category = 1

    for name, module in model.named_modules():

        if name == target_layer:
            target_layer = module
            break

    _, image_paths = get_train_validation_files()
    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=False)
    for nu in tqdm(range(len(image_paths))):
        raw_image = cv2.imread(image_paths[nu][0], 0)

        slices = raw_image.shape[0]

        raw_image = cv2.equalizeHist(raw_image)

        raw_image = Image.fromarray(raw_image, 'L')
        raw_image = test_transform(raw_image)
        raw_image = torch.cat([raw_image, raw_image, raw_image], dim=0)
        input_tensor = torch.unsqueeze(raw_image, dim=0)
        cam.batch_size = 1
        #
        grayscale_cam, label = cam(input_tensor=input_tensor,
                                   target_category=target_category,
                                   aug_smooth=args.aug_smooth,
                                   eigen_smooth=args.eigen_smooth
                                   )
        grayscale_cam = grayscale_cam[0, :]
        if label != 'normal':
            plt.imsave(
                output_dir + image_paths[nu][0].split('/')[-1].replace('.png', '') + '_' + label + '_gradcam.jpg',
                grayscale_cam, cmap='gray')
        else:
            print(image_paths[nu][0].split('/')[-1].replace('.png', ''),'  Prediction : Normal save x ')
