#!/usr/bin/env python
# coding: utf-8




import os, glob, sys
import re, shutil
from tqdm import tqdm
import time
import copy

from collections import defaultdict
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
import zipfile
import random



treeType = "cedar"

workDir = "./04_trainingModel"

checkpoint_path = f"04_trainingModel\\data2016_modelResize-0256_batch32_epoch200_class-cedar\\chkpoint_"
best_model_path = os.path.join(os.path.dirname(checkpoint_path),f'bestmodel.pt')
modelID = checkpoint_path.split("\\")[1]#"classcypress_data168_batch2_epoch100"#checkpoint_path.split("\\\\")[1]



# orgDir = sys.argv[2]#f"C:\\datas\\uav_cnn_{treeType}\\org_crop4Corner_5120_3072_Size1024_lap512"
orgDir = f"C:\\datas\\uav_cnn_{treeType}\\org_crop4Corner_5120_3072_Size1024_lap512"


def calculate_statistics(folder_path):
    image_files = os.listdir(folder_path)
    num_images = len(image_files)
    
    # 初期化
    sum_values = np.zeros(3)
    sum_squares = np.zeros(3)
    
    print("Calc Means, Stds.")
    for image_file in tqdm(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)  # 画像を読み込む
        
        # 画像のピクセル値を正規化する（範囲を0から1にスケーリング）
        normalized_image = image / 255.0
        
        # ピクセル値の合計を計算
        sum_values += np.sum(normalized_image, axis=(0, 1))
        
        # ピクセル値の2乗の合計を計算
        sum_squares += np.sum(normalized_image ** 2, axis=(0, 1))
    
    # 平均値を計算
    mean_values = sum_values / (num_images * image.shape[0] * image.shape[1])
    
    # 標準偏差を計算
    variance = (sum_squares / (num_images * image.shape[0] * image.shape[1])) - (mean_values ** 2)
    std_deviation = np.sqrt(variance)
    
    return mean_values, std_deviation


# フォルダのパスを指定して統計量を計算
# mean_values, std_deviation = calculate_statistics(orgDir)
# print(mean_values, std_deviation)
mean_values, std_deviation = np.array([0.29874884,0.53215061,0.40219877]), np.array([0.14500768, 0.21275673, 0.1953213 ])



def mask2single(mask,values:list):
    for value in values:
        mask[mask==value] = 255
    mask[mask!=255] = 0
    return mask

from pprint import pprint


def get_mskPath(orgPath):
    mskPath = orgPath.replace("org","msk").replace(".jpg",".png")
    if os.path.exists(mskPath):
        return mskPath
    else:
        return ""

#画像データ拡張の関数
def get_train_transform(resizeValue):
   return A.Compose(
       [
        #リサイズ(こちらはすでに適用済みなのでなくても良いです)
        A.Resize(resizeValue, resizeValue),
        #正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Normalize(),
        #水平フリップ（pはフリップする確率）
        # A.HorizontalFlip(p=0.25),
        #垂直フリップ
        # A.VerticalFlip(p=0.25),
        ToTensor()
        ])


class LoadPredDataSet(Dataset):
        def __init__(self,imgPaths,resizeValue, transform=None):
            self.path = imgPaths
            self.folders = [os.path.basename(imgPath) for imgPath in imgPaths]
            self.transforms = get_train_transform(resizeValue)
            self.resizeValue = resizeValue

        def __len__(self):
            return len(self.folders)


        def __getitem__(self, idx):
            image_path = self.path[idx]
            # mask_path = get_mskPath(self.path[idx])
            resizeValue = self.resizeValue

            #画像データの取得
            img = io.imread(image_path)[:,:,0:3].astype('float32')
            img = transform.resize(img,(resizeValue,resizeValue))

            # mask = self.get_mask(mask_path, resizeValue, resizeValue ).astype('float32')


            augmented = self.transforms(image=img)
            # augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            # mask = augmented['mask']
            # mask = mask.permute(2, 0, 1)


            # # 可視化
            # figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 8))
            # ax[0,0].imshow(img.permute(1, 2, 0))#img画像は正規化しているため色味がおかしい
            # ax[0,1].imshow(mask[0,:,:])

            return img, image_path

def format_image(img):
    # img torch.Size([1, 3, 1024, 1024]) を変換
    if isinstance(img, torch.Tensor):
        img = np.transpose(np.squeeze(img.cpu().numpy()), (1,2,0))
    
    #下は画像拡張での正規化を元に戻しています
    mean = mean_values
    std= std_deviation
    img  = std * img + mean
    return img

def deformat_image(img):
    # img = np.array(np.transpose(img, (1,2,0)))
    #下は画像拡張での正規化を元に戻しています
    mean = mean_values
    std= std_deviation
    img  = img / std  - mean
    return img

def format_mask(mask):
    if isinstance(mask, torch.Tensor):
    # msk torch.Size([1, 1, 1024, 1024]) を変換
        mask = mask.cpu().numpy()
    return np.squeeze(mask)






os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

from unet_model import UNet, DiceBCELoss, DiceLoss, IoU, save_ckp, load_ckp


model_init = UNet(3,1).cuda()

# from pre_segmentation_model import UnetModel, calculate_iou
# encoder_name = "resnet34"
# encoder_weight = "imagenet"
# model_init = UnetModel(encoder_name,encoder_weight,3,1)

optimizer = torch.optim.Adam(model_init.parameters(),lr = 1e-3)

print("loading Model checkPoint")
print(checkpoint_path)
model, optimizer, start_epoch, valid_loss_min = load_ckp(checkpoint_fpath=checkpoint_path, model=model_init, optimizer=optimizer)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available(),device)
model.to(device)

predictDir = "05_predicted"
outImagePath = os.path.join(predictDir,modelID,"OutImages")
os.makedirs(outImagePath,exist_ok=True)





def visualize_predict(model, pred_loader, imgSave=False):
    n_images = pred_loader.__len__()
    figure, ax = plt.subplots(nrows=n_images, ncols=3, figsize=(15, 5*n_images))
    with torch.no_grad():
        for img,msk in pred_loader:
            # img = img.cuda()#modelの重みは、cuda.Float.Tensorなので、cuda変換は不要
            # msk = msk.cuda()
            if torch.cuda.is_available():
                img = img.cuda()
            output = model(img)
            # break
            
            for img_no in tqdm(range(0, n_images)):
                tm=output[0][0].data.cpu().numpy()
                img = img
                msk = msk
                # print(tm.shape, img.shape, msk.shape)
                img = format_image(img)
                msk = format_mask(msk)


                outImagePath_pred = os.path.join(workDir,modelID,"predOutImages",f"{img_no}_pred.png")
                outImagePath_img = os.path.join(workDir,modelID,"predOutImages",f"{img_no}_org.png")
                outImagePath_msk = os.path.join(workDir,modelID,"predOutImages",f"{img_no}_msk.png")
                os.makedirs(os.path.dirname(outImagePath_pred) ,exist_ok=True)

                # print(tm.shape, img.shape, msk.shape)
                cv2.imwrite(outImagePath_pred,tm)
                cv2.imwrite(outImagePath_img, img)
                cv2.imwrite(outImagePath_msk,np.array(msk))


                ax[img_no, 0].imshow(img)
                ax[img_no, 1].imshow(msk, interpolation="nearest", cmap="gray")
                ax[img_no, 2].imshow(tm, interpolation="nearest", cmap="gray")
                ax[img_no, 0].set_title("Input Image")
                ax[img_no, 1].set_title("Labeled Mask Conifer")
                ax[img_no, 2].set_title("Predicted Mask Conifer")
                ax[img_no, 0].set_axis_off()
                ax[img_no, 1].set_axis_off()
                ax[img_no, 2].set_axis_off()
            plt.tight_layout()
            if imgSave:
                plt.savefig(os.path.join(os.path.dirname(outImagePath_pred),f"predictedSet_{modelID}.png"))
            # plt.show()
            plt.close()

predict_dataset = LoadPredictDataSet(orgDir, transform=get_train_transform())
pred_loader = DataLoader(dataset=predict_dataset, batch_size=1)

print(predict_dataset.__len__())

visualize_predict(model, pred_loader, imgSave=True)
# visualize_predict(model, train_dataset.__len__()//20-1, imgSave=True)


def visualize_images(outputImagesPaths, LabesName:str):
    n_images = len(outputImagesPaths)
    figure, ax = plt.subplots(nrows=n_images, ncols=3, figsize=(15, 5*n_images))
    print(n_images)

    for orgPath in tqdm(outputImagesPaths):
        img_no = int(re.findall(".*/(\d+)_org.png",orgPath)[0])
        
        predPath = orgPath.replace("org","pred")
        mskPath = orgPath.replace("org","msk")
        tm = cv2.imread(predPath)*255
        img = cv2.imread(orgPath)
        msk = cv2.imread(mskPath)

        # msk = cv2.medianBlur(np.array(msk),11)

        ax[img_no, 0].imshow(img)
        ax[img_no, 1].imshow(msk, interpolation="nearest", cmap="gray")
        ax[img_no, 2].imshow(tm, interpolation="nearest", cmap="gray")
        ax[img_no, 0].set_title("Input Image")
        ax[img_no, 1].set_title(f"Labeled Mask {LabesName}")
        ax[img_no, 2].set_title(f"Predicted Mask {LabesName}")
        ax[img_no, 0].axes.xaxis.set_visible(False)
        ax[img_no, 1].axes.xaxis.set_visible(False)
        ax[img_no, 2].axes.xaxis.set_visible(False)
        ax[img_no, 0].axes.yaxis.set_visible(False)
        ax[img_no, 1].axes.yaxis.set_visible(False)
        ax[img_no, 2].axes.yaxis.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(orgPath), f"predictedSet_{LabesName}.png"))
    plt.show()
    plt.close()


# visualize_images(outImagePath, treeType)

# outputImagesPaths = glob.glob("./training/classBamboo_data880_batch44_epoch100/classBamboo_data880_batch44_epoch100/OutImages/*_org.png")
