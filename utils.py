import torch
import torch.nn.functional as F
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet,shufflenetv2
from torch.utils.data import Dataset, DataLoader
import glob
from skimage import measure
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
import cv2
import numpy as np
from torch.nn import init


def get_train_validation_files():

    train_files=[]
    validation_files=[]
    datapath = './preprocessing_data/'
    for idxs, dataset in enumerate(['train', 'test']):
        name_0 = glob.glob(datapath +dataset+ '/0/*_*_img.png')
        name_1 = glob.glob(datapath +dataset+ '/1/*_*_img.png')

        for file in name_0:
            if idxs == 0:
                train_file = (file, [1, 0])
                train_files.append(train_file)
            else:
                validation_file = (file, [1, 0])
                validation_files.append(validation_file)

        for file in name_1:
            if idxs == 0:
                train_file = (file, [0, 1])
                train_files.append(train_file)

            else:
                validation_file = (file, [0, 1])
                validation_files.append(validation_file)

    print('trainset : ', len(train_files), '&  testset : ', len(validation_files))

    return train_files, validation_files

class FPAR_shuffleNET_v2(nn.Module):
    def __init__(self,n_classes):
        super(FPAR_shuffleNET_v2, self).__init__()

        self.net = shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
        self.relu=nn.ReLU()
        self.fc = nn.Linear(1000, n_classes)
    def forward(self, x,lb):

        x = self.net.conv1(x)
        x = self.net.maxpool(x)
        x = self.net.stage2(x)
        x = self.net.stage3(x)
        x = self.net.stage4(x)
        x = self.net.conv5(x)
        if lb != None:
            activate_mask = self.relu(
                torch.unsqueeze(
                    (torch.matmul(self.fc.weight, self.net.fc.weight)[torch.argmax(lb, dim=1)][:, :, None,
                     None] * x).mean(axis=1), dim=1))
        x = x.mean([2, 3])  # globalpool
        x = self.net.fc(x)
        x = self.fc(x)
        if lb!=None:
            return x, activate_mask
        else:
            return x



class FPAR_EfficientNet(nn.Module):
    def __init__(self,n_classes):
        super(FPAR_EfficientNet, self).__init__()

        self.net = EfficientNet.from_pretrained('efficientnet-b7',in_channels=1)

        self.relu=nn.ReLU()
        self.fc = nn.Linear(1000, n_classes)
    def forward(self, x,lb):
        x = self.net.extract_features(x)
        if lb != None:
            activate_mask = self.relu(
                torch.unsqueeze(
                    (torch.matmul(self.fc.weight, self.net._fc.weight)[torch.argmax(lb, dim=1)][:, :, None,
                     None] * x).mean(axis=1), dim=1))
        x = self.net._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.net._dropout(x)
        x = self.net._fc(x)
        x = self.fc(x)

        if lb != None:
            return x, activate_mask
        else:
            return x



class FPAR_ResNet50(nn.Module):
    def __init__(self,n_classes):
        super(FPAR_ResNet50, self).__init__()

        self.net = resnet.resnet50(pretrained=True)
        self.relu=nn.ReLU()
        self.fc = nn.Linear(1000, n_classes)

    def forward(self, x,lb):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        if lb != None:
            activate_mask = self.relu(
                torch.unsqueeze(
                    (torch.matmul(self.fc.weight, self.net.fc.weight)[torch.argmax(lb, dim=1)][:, :, None,
                     None] * x).mean(axis=1), dim=1))

        x = self.net._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.net._dropout(x)
        x = self.net.fc(x)
        x = self.fc(x)

        if lb != None:
            return x, activate_mask
        else:
            return x


class TEST_FPAR_shuffleNET_v2(nn.Module):
    def __init__(self,n_classes):
        super(TEST_FPAR_shuffleNET_v2, self).__init__()

        self.net = shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
        self.relu=nn.ReLU()
        self.fc = nn.Linear(1000, n_classes)
    def forward(self, x):

        x = self.net.conv1(x)
        x = self.net.maxpool(x)
        x = self.net.stage2(x)
        x = self.net.stage3(x)
        x = self.net.stage4(x)
        x = self.net.conv5(x)

        x = x.mean([2, 3])  # globalpool
        x = self.net.fc(x)
        x = self.fc(x)

        return x

def l2loss(actilayer,mask):
    loss_l2=torch.sqrt(torch.square((actilayer*mask)+1e-9).mean())
    return loss_l2

def find_center(mask):
    mask_copy=mask.copy()

    contours = measure.find_contours(mask_copy, 0.95)
    vol_contours = []
    for contour in contours:
        hull = ConvexHull(contour)

        vol_contours.append([contour, hull.volume])

    contours = sorted(vol_contours, key=lambda s: s[1])
    contours.reverse()
    if len(contours)>3:
        contours=contours[:3]
    mask_copy=np.zeros(mask_copy.shape)
    for contour in contours:
        area = contour[0]
        x = area[:, 0]
        y = area[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', mask.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask_copy += np.array(img).T
    mask_copy[mask_copy>0] = 1
    for idx in range(0, mask_copy.shape[1]):
        if mask_copy[:, idx].sum() != 0:
            l_idx12 = idx
            break
    for idx in range(mask_copy.shape[1] - 1, 0, -1):
        if mask_copy[:, idx].sum() != 0:
            r_idx12 = idx
            break

    mask_copy = mask_copy[:, l_idx12:r_idx12 + 1]
    for idx in range(0, mask_copy.shape[0]):
        if mask_copy[idx, :].sum() != 0:
            l_idx23 = idx
            break
    for idx in range(mask_copy.shape[0] - 1, 0, -1):
        if mask_copy[idx, :].sum() != 0:
            r_idx23 = idx
            break

    mask_copy = mask_copy[l_idx23:r_idx23 + 1, :]
    return mask_copy,[l_idx23,r_idx23 + 1,  l_idx12,r_idx12 + 1]

def full_fill(mask_img):
    contours = measure.find_contours(mask_img, 0.95)
    vol_contours = []
    for contour in contours:
        hull = ConvexHull(contour)

        vol_contours.append([contour, hull.volume])

    contours = sorted(vol_contours, key=lambda s: s[1])
    contours.reverse()
    masks = []
    for contour in contours:
        area = contour[0]
        x = area[:, 0]
        y = area[:, 1]
        polygon_tuple = list(zip(x, y))
        img_d = Image.new('L', mask_img.shape, 0)
        ImageDraw.Draw(img_d).polygon(polygon_tuple, outline=0, fill=1)
        masks.append(np.array(img_d).T)
    masks = np.sum(masks, 0)
    masks = np.array(masks, dtype='uint8')
    masks[masks > 0] = 1
    return masks


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class MyInferenceClass(Dataset):

    def __init__(self, image_path):


        self.images = glob.glob(str(image_path) + "/*.png")

        self.data_len = len(self.images)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        images = cv2.imread(self.images[index], 0)

        original_image_size = images.shape

        images = cv2.equalizeHist(images)

        images = cv2.resize(images, (512, 512))
        images = np.array(images, dtype='float32')

        return {'input': np.expand_dims(images, 0),
                'im_size': original_image_size, 'label': self.images[index].split('/')[-2] , 'name':self.images[index]}



def one_hot(x, class_count):
    return torch.eye(class_count)[:, x]
