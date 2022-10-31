import os
import pdb

import pandas as pd

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import shufflenetv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import auc, roc_curve, roc_auc_score
import numpy as np
import cv2, sys, time, glob, warnings,pickle
from sklearn.metrics import classification_report
from PIL import Image
from utils import *
warnings.filterwarnings(action='ignore')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

savedir='./checkpoint/'
select_model=0#1,2
#0 : shuffleNetV2
#1 : resnet50
#2 : efficientnetb7
os.makedirs(savedir,exist_ok=True)
save=True
num_workers =0
n_epochs =50
batch_size= 16
img_wsize = 512
img_hsize = 512
gpu = 'T'
pretrained=True
lr = 0.1
class_number = 2
epoch_lr = [20,40]
L2_lambda=10



class IntracranialDataset(Dataset):
    def __init__(self, df,model_select):
        self.data = df
        self.transform =transforms.Compose([
                        transforms.Resize((img_hsize, img_wsize)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                        ])
        self.model_ch = 3 if model_select<2 else 1
        self.mask_trans = transforms.Compose([
            transforms.Resize((int(img_wsize/32),int(img_wsize/32))),
            transforms.ToTensor(), ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0]
        img = cv2.imread(img_name,0)
        labels = torch.tensor(self.data[idx][1])
        if torch.argmax(labels)==0:
            mask=cv2.imread(img_name.replace('_img.png','_mask.png'),0)
        else:
            mask = cv2.imread(img_name.replace('_img.png', '_label.png'), 0)
        img=cv2.equalizeHist(img)
        img = Image.fromarray(img,'L')
        mask = Image.fromarray(mask, 'L')
        if self.transform:
            img = self.transform(img)
            mask = self.mask_trans(mask)
        if self.model_ch==3:
            img = torch.cat((img, img, img), axis=0)
        return {'image': img, 'labels': labels,'masks':1-mask}


def multi_acc(y_pred, y_test):
    _, y_pred_tags = torch.max(y_pred, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    trueval = correct_pred.sum().tolist()

    return [int(trueval),len(trueval)]


def train_phase(net,trnloader,scheduler,optimizer,criterion,valloader):
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        net.train().cuda()
        tr_loss = 0
        tr_acc = 0

        for step, batch in enumerate(trnloader):
            inputs = batch["image"].float().cuda()
            labels = batch["labels"].long().cuda()
            masks = batch["masks"].float().cuda()
            outputs,at_masks = net(inputs,labels)

            loss = criterion(F.softmax(outputs, dim=1), torch.max(labels, 1)[1])+ (l2loss(at_masks,masks)*L2_lambda)
            acc = multi_acc(F.softmax(outputs, dim=1), torch.max(labels, 1)[1])
            tr_loss += loss.item()
            tr_acc += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del inputs, labels, outputs
        scheduler.step()
        tr_loss = tr_loss / len(trnloader)
        tr_acc = tr_acc[0] / tr_acc[1]
        print('Trn loss: {}, trn acc: {}'.format(round(tr_loss, 4), round(tr_acc, 4), ))
        Validation_phase(net, valloader)

def Validation_phase(model, dataLoader):
    model.eval()
    valprediction = []
    valy = []
    with torch.no_grad():
        for i, batch in enumerate(dataLoader):
            input = batch["image"].float().cuda()
            target = batch["labels"].long().cuda()

            output = model(input,None)

            valprediction += F.softmax(output, dim=1).tolist()
            valy += target.tolist()
            del input, target, output

    print(classification_report(np.argmax(valy, axis=1),np.argmax(valprediction, axis=1),
                                target_names=['normal', 'abnormal']))  # ,'Viral','COVID-19']))


def main():
    if select_model ==0:
        net = FPAR_shuffleNET_v2(n_classes=class_number)
    elif select_model ==1:
        net = FPAR_ResNet50(n_classes=class_number)
    else:
        net = FPAR_EfficientNet(n_classes=class_number)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net).cuda()
    traindata, testdata = get_train_validation_files()

    trndataset = IntracranialDataset(traindata, select_model)
    valdataset = IntracranialDataset(testdata, select_model)
    trnloader = DataLoader(trndataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    valloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    plist = [{'params': net.parameters(), 'lr': lr}]
    optimizer = optim.SGD(plist, lr=lr)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=epoch_lr, gamma=0.1)

    os.makedirs(savedir, exist_ok=True)
    train_phase(net, trnloader, scheduler, optimizer, criterion, valloader)
    torch.save({'model_state_dict': copy.deepcopy(net.state_dict())},
           os.path.join(savedir +'model_weight.pth'))

main()
