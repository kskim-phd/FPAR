import os
from tqdm import tqdm
from utils import *

cam_dir = './CAM/'
savedir = './Full_CAM/'
os.makedirs(savedir,exist_ok=True)
cropdata_dir='./preprocessing_data/test/'
datas=glob.glob('./data/test/shoulder_image/*')
for data in tqdm(datas):
    img=cv2.imread(data,0)

    cams_r = glob.glob(cam_dir+data.split('/')[-1].replace('.png','')+'_r_*gradcam*')
    cams_l = glob.glob(cam_dir+data.split('/')[-1].replace('.png','')+'_l_*gradcam*')

    maskimg = cv2.imread(data.replace('/shoulder_image/', '/shoulder_mask/'), 0)

    maskimg = cv2.resize(maskimg,img.shape[::-1])
    maskimg[maskimg>0]=1
    maskimg[int(maskimg.shape[0] / 2):, :] = 0

    w, h=maskimg.shape[::-1]
    padding_val = int(w/2) if w>h else int(h/2)
    maskimg = cv2.copyMakeBorder(maskimg, padding_val, padding_val, padding_val, padding_val, cv2.BORDER_CONSTANT, value=0)
    img = cv2.copyMakeBorder(img, padding_val, padding_val, padding_val, padding_val, cv2.BORDER_CONSTANT, value=0)

    full_cam = maskimg.copy()
    full_cam[full_cam!=0]=0
    leftmask = maskimg.copy()
    leftmask[:, int(maskimg.shape[0]/2):] = 0
    rightmask = maskimg.copy()
    rightmask[:, :int(maskimg.shape[0]/2)] = 0


    if len(cams_r)>0:

        cropshape, croplist=find_center(rightmask)
        shape_size = np.max(cropshape.shape)
        shape_size = 512 if shape_size < 512 else int(shape_size * 2 / 3)
        center_point = [int((croplist[0]+croplist[1])/2),int((croplist[2]+croplist[3])/2)]
        full_cam[center_point[0] - shape_size:center_point[0] + shape_size,
                   center_point[1] - shape_size:center_point[1] + shape_size]= cv2.resize(cv2.imread(cams_r[0],0),(shape_size*2,shape_size*2))

    if len(cams_l) > 0:

        cropshape, croplist=find_center(leftmask)
        shape_size = np.max(cropshape.shape)
        shape_size = 512 if shape_size < 512 else int(shape_size * 2 / 3)
        center_point = [int((croplist[0] + croplist[1]) / 2), int((croplist[2] + croplist[3]) / 2)]
        full_cam[center_point[0] - shape_size:center_point[0] + shape_size,
                       center_point[1] - shape_size:center_point[1] + shape_size]=cv2.resize(cv2.flip(cv2.imread(cams_l[0],0),1),(shape_size*2,shape_size*2))

    img=cv2.merge([img,img,img])
    full_cam=cv2.applyColorMap(full_cam,colormap=cv2.COLORMAP_JET)
    img=img[padding_val:-padding_val,padding_val:-padding_val]
    full_cam = full_cam[padding_val:-padding_val, padding_val:-padding_val]
    cam_img=cv2.addWeighted(img,0.6,full_cam,0.4,0)
    cv2.imwrite(savedir+data.split('/')[-1],cam_img)

