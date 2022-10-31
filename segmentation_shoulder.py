# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 2020
@author: Yujin Oh (yujin.oh@kaist.ac.kr)
"""

import sys
# common
import torch, torchvision
import numpy as np

# dataset
from torch.utils.data import DataLoader
from PIL import Image

# model
import os

import cv2
from utils import AttU_Net,MyInferenceClass,one_hot

def main():


    ##############################################################################################################################
    # Semantic segmentation (inference)


    # GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        num_worker = 8
    else:
        device = torch.device("cpu")
        num_worker = 0

    # Model initialization
    net =AttU_Net(img_ch=1,output_ch=2)
    # Load model
    model_dir = './checkpoint/model_last.pth'
    if os.path.isfile(model_dir):
        print('\n>> Load Segmentation model ' )
        checkpoint = torch.load(model_dir)
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('[Err] Model does not exist in %s' % (model_dir))
        exit()

    # network to GPU
    net.to(device)

    # loop dataset class
    imgdir='./data/*/shoulder_image/'
    num_masks=2
    num_batch_test=8
    # Dataset
    print('\n>> Load dataset -',  imgdir)

    testset = MyInferenceClass(imgdir)
    testloader = DataLoader(testset, batch_size=num_batch_test, shuffle=False, num_workers=num_worker,pin_memory=True)
    print("  >>> Total # of sampler : %d" % (len(testset)))

    # inference

    sys.stdout.write("\r" '>> segmentaion..')
    with torch.no_grad():

        # initialize
        net.eval()
        for i, data in enumerate(testloader, 0):

            # forward
            outputs = net(data['input'].to(device))

            outputs = torch.argmax(outputs.detach(), dim=1)

            # one hot
            outputs_max = torch.stack(
                [one_hot(outputs[k], num_masks) for k in range(len(data['input']))])

            # each case
            for k in range(len(data['input'])):
                cv2.imwrite(data['name'][k].replace('/shoulder_image/','/shoulder_mask/') , np.array(torch.argmax(outputs_max[k], dim=0).numpy()/(num_masks-1) * 255,dtype='uint8'))
    sys.stdout.flush()
    print('.done')



if __name__ == '__main__':
    main()