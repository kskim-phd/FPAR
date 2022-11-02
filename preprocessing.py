import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
import cv2
import numpy as np


for phase in ['train','test']:
    datas = glob.glob("./data/"+phase+"/shoulder_image/*.png")
    savedir = './preprocessing_data/'+phase+'/'
    os.makedirs(savedir+'1',exist_ok=True)
    os.makedirs(savedir+'0',exist_ok=True)
    for data in tqdm(datas):
        img=cv2.imread(data,0)
        maskimg = cv2.imread(data.replace('/shoulder_image/', '/shoulder_mask/') , 0)
        maskimg = cv2.resize(maskimg,img.shape[::-1])


        maskimg[maskimg>0]=1
        maskimg[int(maskimg.shape[0] / 2):, :] = 0

        w,h=maskimg.shape[::-1]
        padding_val = int(w/2) if w>h else int(h/2)
        maskimg = cv2.copyMakeBorder(maskimg, padding_val, padding_val, padding_val, padding_val, cv2.BORDER_CONSTANT, value=0)
        maskimg = full_fill(maskimg)
        img = cv2.copyMakeBorder(img, padding_val, padding_val, padding_val, padding_val, cv2.BORDER_CONSTANT, value=0)

        try:
            annotation_img = cv2.imread(data.replace('/shoulder_image/', '/shoulder_annotation/') , 0)
        except:
            annotation_img=np.zeros(img.shape[::-1])
        annotation_img = cv2.copyMakeBorder(annotation_img, padding_val, padding_val, padding_val, padding_val, cv2.BORDER_CONSTANT, value=0)
        annotation_mask = full_fill(annotation_img)

        leftannotation = annotation_mask.copy()
        leftannotation[:, int(annotation_mask.shape[0] / 2):] = 0
        rightannotation = annotation_mask.copy()
        rightannotation[:, :int(annotation_mask.shape[0] / 2)] = 0

        leftmask = maskimg.copy()
        leftmask[:, int(maskimg.shape[0]/2):] = 0
        rightmask = maskimg.copy()
        rightmask[:, :int(maskimg.shape[0]/2)] = 0

        right_label = 1 if rightannotation.sum() > 0 else 0
        left_label = 1 if leftannotation.sum() > 0 else 0

        if rightmask.sum()>0:
            cropshape, croplist=find_center(rightmask)
            shape_size = np.max(cropshape.shape)
            shape_size = 512 if shape_size<512 else int(shape_size * 2 / 3)
            center_point = [int((croplist[0]+croplist[1])/2),int((croplist[2]+croplist[3])/2)]
            rightseg = rightmask[center_point[0]-shape_size:center_point[0]+shape_size,center_point[1]-shape_size:center_point[1]+shape_size]
            img_right = img[center_point[0] - shape_size:center_point[0] + shape_size,
                       center_point[1] - shape_size:center_point[1] + shape_size]
            rightannotation = rightannotation[center_point[0] - shape_size:center_point[0] + shape_size,
                        center_point[1] - shape_size:center_point[1] + shape_size]

            plt.imsave(savedir+str(right_label)+'/'+data.split('/')[-1]+'_r_mask.png',rightseg,cmap='gray')
            cv2.imwrite(savedir+str(right_label)+'/'+data.split('/')[-1]+'_r_img.png',img_right)
            plt.imsave(savedir+str(right_label)+'/'+data.split('/')[-1]+'_r_annotation.png',rightannotation,cmap='gray')

        if leftmask.sum()>0:
            cropshape, croplist=find_center(leftmask)
            shape_size = np.max(cropshape.shape)
            shape_size = 512 if shape_size<512 else int(shape_size * 2 / 3)
            center_point = [int((croplist[0]+croplist[1])/2),int((croplist[2]+croplist[3])/2)]
            leftseg = leftmask[center_point[0]-shape_size:center_point[0]+shape_size,center_point[1]-shape_size:center_point[1]+shape_size]
            img_left = img[center_point[0] - shape_size:center_point[0] + shape_size,
                       center_point[1] - shape_size:center_point[1] + shape_size]
            leftannotation = leftannotation[center_point[0] - shape_size:center_point[0] + shape_size,
                             center_point[1] - shape_size:center_point[1] + shape_size]

            plt.imsave(savedir+str(left_label)+'/' + data.split('/')[-1]+'_l_mask.png',cv2.flip(leftseg,1),cmap='gray')
            cv2.imwrite(savedir+str(left_label)+'/' + data.split('/')[-1]+'_l_img.png',cv2.flip(img_left,1))
            plt.imsave(savedir+str(left_label)+'/' + data.split('/')[-1] + '_l_annotation.png', cv2.flip(leftannotation, 1),cmap='gray')
