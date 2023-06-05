#!/usr/bin/env python
# coding: utf-8



import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_properties = torch.cuda.get_device_properties(device)
    gpu_memory_size = gpu_properties.total_memory / 1024 ** 3  # Convert bytes to gigabytes
    print(f"GPU Memory Size: {gpu_memory_size:.2f} GB")
else:
    print("GPU is not available.")



# !pip install -r requirements_uavUnet.txt



import os, glob, sys
import re, shutil
from tqdm import tqdm

from collections import defaultdict
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

import cv2

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
import zipfile
import random




def get_mskPath(orgPath):
    mskPath = orgPath.replace("org","msk").replace(".jpg",".png")
    if os.path.exists(mskPath):
        return mskPath
    else:
        return ""


def copyLocaliImages(orgDir, copyDir):
    orgDirName = os.path.basename(orgDir)
    
    orgPaths = glob.glob(orgDir + "/*.*")
    mskPath = get_mskPath(orgPaths[0])
    mskDirName = os.path.basename(os.path.dirname(mskPath))

    os.makedirs(os.path.join(copyDir,orgDirName), exist_ok=True)
    os.makedirs(os.path.join(copyDir,mskDirName), exist_ok=True)

    for orgPath in orgPaths:
        mskPath = get_mskPath(orgPath)
        new_orgPath = os.path.join(copyDir,orgDirName,os.path.basename(orgPath))
        if not os.path.exists(new_orgPath):
            shutil.copy(orgPath,new_orgPath)
        
        new_mskPath = os.path.join(copyDir,mskDirName,os.path.basename(mskPath))
        if not os.path.exists(new_mskPath):
            shutil.copy(mskPath,new_mskPath)
    return os.path.join(copyDir,orgDirName)




def addSavePath(Path,addKeyword):
    Path_added = os.path.dirname(Path) + "/" + os.path.basename(Path).split(".")[0] + "_" + str(addKeyword) + "." + os.path.basename(Path).split(".")[1]
    return Path_added

def convMskPath(imgPath):
    mskPath = imgPath.replace("org","msk").replace(".jpg",".png")
    return mskPath


def flipMirrorSave(imgPath, SaveDir):
    os.makedirs(SaveDir,exist_ok=True)

    imgPil = Image.open(imgPath)
    imgPil_flip = ImageOps.flip(imgPil)
    imgPil_mirror = ImageOps.mirror(imgPil)

    SavePath = os.path.join(SaveDir, os.path.basename(imgPath))

    SavePath_flip = addSavePath(SavePath,"_flip")
    SavePath_mirror = addSavePath(SavePath,"_mirror")

    os.makedirs(os.path.dirname(SavePath_flip), exist_ok=True)
    os.makedirs(os.path.dirname(SavePath_mirror), exist_ok=True)

    if not os.path.exists(SavePath):
        imgPil.save(SavePath)
    if not os.path.exists(SavePath_flip):
        imgPil_flip.save(SavePath_flip)
    if not os.path.exists(SavePath_mirror):
        imgPil_mirror.save(SavePath_mirror)
#     return imgPil_filp, imgPil_mirror

def rotateSave(imgPath, SaveDir, angleMax:int, angleInterval:90):
    os.makedirs(SaveDir,exist_ok=True)

    imgPil = Image.open(imgPath)
    #　画像を回転角度の設定値　90, 180, 270
    angleMax = angleMax + 90
    # angleInterval = 90

    SavePath = os.path.join(SaveDir, os.path.basename(imgPath))

    for angle in range(0,angleMax,angleInterval):
        angleStr = str(angle).zfill(3)
        
        #画像の保存名を定義。
        SavePath_rotate = addSavePath(SavePath, f"rotate{angleStr}")
        os.makedirs(os.path.dirname(SavePath_rotate), exist_ok=True)
        if not os.path.exists(SavePath_rotate):
            imgPil_rotate = imgPil.rotate(angle)
            imgPil_rotate.save(SavePath_rotate)
            print(SavePath_rotate,imgPil_rotate.mode)


def argmentDataset(orgDir, rotate=True, flipMirror=True):
    #Rotate処理
    imgPaths = glob.glob(os.path.join(orgDir,"*.jpg"))
    print("image Rotate")
    for imgPath in tqdm(imgPaths):
        if rotate:
            rotateSave(imgPath, SaveDir=os.path.dirname(imgPath)+"_rotate", angleMax=270, angleInterval=90)
            mskPath = convMskPath(imgPath)
            rotateSave(mskPath, SaveDir=os.path.dirname(mskPath)+"_rotate", angleMax=270, angleInterval=90)
                
    #フリップ、ミラー処理
    print("image flipMirror")
    imgPaths = glob.glob(os.path.join(os.path.dirname(imgPaths[0])+"_rotate","*.jpg"))
    for imgPath in tqdm(imgPaths):
        if flipMirror:
            flipMirrorSave(imgPath, SaveDir=os.path.dirname(imgPath)+"_flipMirror")
            mskPath = convMskPath(imgPath)
            flipMirrorSave(mskPath, SaveDir=os.path.dirname(mskPath)+"_flipMirror")

    imgPaths = glob.glob(os.path.join(os.path.dirname(imgPaths[0])+"_flipMirror","*.jpg"))
    orgDir = os.path.dirname(imgPaths[0])
    return orgDir


def calculate_statistics(image_paths):
    num_images = len(image_paths)
    
    # 初期化
    sum_values = np.zeros(3)
    sum_squares = np.zeros(3)
    
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)  # 画像を読み込む
        
        # 画像のピクセル値を正規化する
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


def calculate_pil_statistics(image_paths):
    red_values = []
    green_values = []
    blue_values = []

    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)
        rgb_array = rgb_array / 255.0
        red_values.extend(rgb_array[:, :, 0].flatten())
        green_values.extend(rgb_array[:, :, 1].flatten())
        blue_values.extend(rgb_array[:, :, 2].flatten())

    red_mean = np.mean(red_values)
    green_mean = np.mean(green_values)
    blue_mean = np.mean(blue_values)

    red_std = np.std(red_values)
    green_std = np.std(green_values)
    blue_std = np.std(blue_values)

    return [red_mean, green_mean, blue_mean], [red_std, green_std, blue_std]





#画像データ拡張の関数
def get_train_transform(mean_values,std_deviation):
   return A.Compose(
       [
        # #リサイズ(こちらはすでに適用済みなのでなくても良いです)
        #正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
        # A.Normalize(),
        A.Normalize(mean=mean_values, std=std_deviation),
        #水平フリップ（pはフリップする確率）
        # A.HorizontalFlip(p=0.5),
        # #垂直フリップ
        # A.VerticalFlip(p=0.5),
        # A.Rotate(limit=[90,180,270], p=0.5),
        ToTensor()
        ])

def mask2single(mask,values:list):
    for value in values:
        mask[mask==value] = 255
    mask[mask!=255] = 0
    return mask

import psql_connect 

#Datasetクラスの定義
class LoadDataSet(Dataset):
        def __init__(self,path, transform=None):
            self.path = path
            self.folders = os.listdir(path)
            mean_values, std_deviation = psql_connect.getMeanStd(path)
            self.transforms = get_train_transform(mean_values, std_deviation)
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_folder = self.path
            mask_folder = self.path.replace("org","msk")
            orgPath = os.path.join(image_folder,os.listdir(image_folder)[idx])
            mskPath = get_mskPath(orgPath)
            # print("\nimageName\t",os.path.basename(orgPath))
            #画像データの取得
            img = io.imread(orgPath)[:,:,0:3].astype('float32')
            # img = transform.resize(img,(256,256))
            
            height, width, _ = img.shape
            
            mask = self.get_mask(mskPath, height, width ).astype('float32')

            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
            # augmented key名　image, mask
            img = augmented['image']
            mask = augmented['mask']
            mask = mask.permute(2, 0, 1)

            # # 可視化
            # figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 8))
            # ax[0,0].imshow(img.permute(1, 2, 0))#img画像は正規化しているため色味がおかしい
            # ax[0,1].imshow(mask[0,:,:])

            return (img,mask) 

        #マスクデータの取得
        def get_mask(self, mskPath, IMG_HEIGHT, IMG_WIDTH):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
            mask_ = io.imread(mskPath)
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_,axis=-1)
            mask = np.maximum(mask, mask_)
              
            return mask


# def tensor2img(imgTensor):
#     # チャンネルの順番を変更するために、データの形状を変換します
#     img = imgTensor.permute(1, 2, 0)

#     # データを0から1の範囲に正規化します
#     img = (img - img.min()) / (img.max() - img.min())
#     return img

# def showOrgMsk(org,msk):
#     # 2つのサブプロットを作成し、それぞれのサブプロットに画像を表示します
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#     # 左側のサブプロットに画像1を表示します
#     axs[0].imshow(org)
#     axs[0].axis('off')

#     # 右側のサブプロットに画像2を表示します
#     axs[1].imshow(msk)
#     axs[1].axis('off')

#     # グラフを表示します
#     plt.show()
#     plt.clf()

# for i in range(10,13):
#     org, msk = train_dataset.__getitem__(i)
#     showOrgMsk(tensor2img(org),tensor2img(msk))



def format_image(img,mean_values,std_deviation):
    img = np.array(np.transpose(img, (1,2,0)))
    #下は画像拡張での正規化を元に戻しています
    mean = mean_values
    std= std_deviation
    img  = std * img + mean
    # img = img.astype(np.uint8)
    return img

def deformat_image(img,mean_values,std_deviation):
    img = np.array(np.transpose(img, (1,2,0)))
    mean = mean_values
    std= std_deviation
    img  = img - mean
    return img / std


def format_mask(mask):
    mask = np.squeeze(np.transpose(mask, (1,2,0)))
    return mask

def visualize_dataset(n_images, predict=None):
    images = random.sample(range(0, train_dataset.__len__()), n_images)
    figure, ax = plt.subplots(nrows=len(images), ncols=2, figsize=(5, 8))
    # print(images)
    for i in range(0, len(images)):
        img_no = images[i]
        # print(img_no)
        image, mask = train_dataset.__getitem__(img_no)
        image = format_image(image)
        mask = format_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest", cmap="gray")
        ax[i, 0].set_title("Input Image")
        ax[i, 1].set_title("Label Mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()

# visualize_dataset(2)






def get_gpu_memory_usage():
    """GPUメモリの使用量を取得する関数"""
    return torch.cuda.memory_allocated() / 1024**3  # GiB単位で返す

def print_allocated_tensors():
    """GPUメモリを占有している変数の一覧を表示する関数"""
    tensor_list = torch.cuda.memory_summary()  # 現在GPUメモリに割り当てられているテンソルの一覧を取得
    print(tensor_list)


# 可視化
def showPred(img_show, msk_show, pred,mean_values:list,std_deviation:list,workDir:str, epoch:int, modelID:str, imgSave=True):  
    
    try:
        treeType = re.search(".*class-(.*)_data.*" ,modelID).group(1)
    except:
        treeType = "treeType"
    
    img_show = img_show.cpu().numpy().copy()
    img_show = format_image(img_show,mean_values,std_deviation)
    # img_show = np.transpose(img_show, (1,2,0))
    print("img_show.shape",img_show.shape)
    msk_show = np.squeeze(msk_show).cpu().numpy().copy()
    print("msk_show.shape",msk_show.shape)
    print("output.shape",pred.shape)
    
    print("pred.shape",pred.shape)    
    pred = np.squeeze(pred.data.cpu())
    print("pred.shape",pred.shape,type(pred))    
    cmap = plt.cm.get_cmap('inferno')
    print("1 pred.unique()",np.unique(pred)) 
    pred_msk = pred.clone().numpy()#テンソルをNumPy配列に変換し、メモリを分けて別の配列を作成
    print("pred_msk.unique()",np.unique(pred_msk),len(np.unique(pred_msk))) 
    # mskThValue = 0
    # pred_msk[pred_msk>=mskThValue] = 1
    # pred_msk[pred_msk<=mskThValue] = 0
    # Int8型に変換
    print("pred.unique()",np.unique(pred_msk))        
    pred_msk = pred_msk.astype(np.uint8)
    print("org.unique()",np.unique(img_show)[0:10])        
    print("msk.unique()",np.unique(msk_show))        
    print("pred.unique()",np.unique(pred_msk))        
    
    print("img_show.shape",img_show.shape)
    print("msk_show.shape",msk_show.shape)
    figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    ax[0].imshow(img_show)
    ax[1].imshow(msk_show, interpolation="nearest", cmap="gray")
    ax[2].imshow(pred_msk, interpolation="nearest", cmap="gray")
    im = ax[3].imshow(pred,cmap=cmap, vmin=pred.min(), vmax=pred.max(), interpolation="nearest")
    figure.colorbar(im, ax=ax[3])
    ax[0].set_title("Input Image")
    ax[1].set_title(f"Labeled Mask {treeType}")
    ax[2].set_title(f"Predicted Mask {treeType}")
    ax[3].set_title("Predicted Ratio")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    ax[3].set_axis_off()
    plt.tight_layout()
    if imgSave:
        plt.savefig(workDir + f"traingSet_Epoch{str(epoch+1)}_{modelID}.png")
    plt.clf()
    plt.close()
    del figure, ax, im, img_show, msk_show, pred, pred_msk


def savePred(img_show, msk_show, pred,mean_values:list,std_deviation:list,workDir:str, epoch:int, modelID:str):  
    try:
        treeType = re.search(".*class-(.*)_data.*" ,modelID).group(1)
    except:
        treeType = "treeType"
    img_show = img_show.cpu().numpy().copy()
    img_show = format_image(img_show,mean_values,std_deviation)
    msk_show = np.squeeze(msk_show).cpu().numpy().copy()
    
    pred = np.squeeze(pred.data.cpu())
    cmap = plt.cm.get_cmap('inferno')
    pred_msk = pred.clone().numpy()#テンソルをNumPy配列に変換し、メモリを分けて別の配列を作成
    # Int8型に変換
    # print("pred.unique()",np.unique(pred_msk))        
    pred_msk = pred_msk.astype(np.uint8)
    # print("org.unique()",np.unique(img_show)[0:10])        
    # print("msk.unique()",np.unique(msk_show))        
    # print("pred.unique()",np.unique(pred_msk))        
    
    figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    ax[0].imshow(img_show)
    ax[1].imshow(msk_show, interpolation="nearest", cmap="gray")
    ax[2].imshow(pred_msk, interpolation="nearest", cmap="gray")
    im = ax[3].imshow(pred,cmap=cmap, vmin=pred.min(), vmax=pred.max(), interpolation="nearest")
    figure.colorbar(im, ax=ax[3])
    ax[0].set_title("Input Image")
    ax[1].set_title(f"Labeled Mask {treeType}")
    ax[2].set_title(f"Predicted Mask {treeType}")
    ax[3].set_title("Predicted Ratio")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    ax[3].set_axis_off()
    plt.tight_layout()
    plt.savefig(workDir + f"traingSet_Epoch{str(epoch+1)}_{modelID}.png")
    plt.clf()
    plt.close()
    del figure, ax, im, img_show, msk_show, pred, pred_msk



def main(treeType:str, epoch:int):

    print(f"GPU Memory Usage: {get_gpu_memory_usage():.2f} GiB\n")
    # print("Allocated Tensors:")
    # print_allocated_tensors()


    workDir = "./04_trainingModel"
    TRAIN_PATH = workDir + '/Forest tsumura 2 50m P4Pv2_{treeType}/'
    os.makedirs(TRAIN_PATH, exist_ok=True)

    orgDir = f"./03_datasetforModel/Forest tsumura 2 50m P4Pv2_{treeType}/org_crop4Corner_5120_3072_Size1024_lap512"

    orgPaths = glob.glob(orgDir + "/*.*")
    mskPaths = [get_mskPath(orgPath) for orgPath in orgPaths]
    print(len(orgPaths),len(mskPaths))

    orgDir = copyLocaliImages(orgDir, f"C:\\datas\\uav_cnn_{treeType}")
    orgPaths = glob.glob(orgDir + "/*.*")

    # フォルダのパスを指定して統計量を計算
    mean_values, std_deviation = calculate_statistics(orgDir)
    # mean_values, std_deviation = calculate_statistics(orgDir.replace("org","msk"))

    print("平均値:", mean_values)
    print("標準偏差:", std_deviation)

            
    train_dataset = LoadDataSet(orgDir, transform=get_train_transform(mean_values=mean_values, std_deviation=std_deviation))
    print("images count\t",train_dataset.__len__())

    org, msk = train_dataset.__getitem__(1)

    print("org.shape",org.shape)
    print("msk.shape",msk.shape)
    del org, msk


    #データ前処理
    split_ratio = 0.2
    train_size=int(np.round(train_dataset.__len__()*(1 - split_ratio),0))
    valid_size=int(np.round(train_dataset.__len__()*split_ratio,0))

    BATCHSIZE = 2#train_dataset.__len__()//80
    train_data, valid_data = random_split(train_dataset, [train_size, valid_size])
    train_loader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
    val_loader = DataLoader(dataset=valid_data, batch_size=BATCHSIZE)

    num_epochs=epoch


    modelID = "class{}_data{}_batch{}_epoch{}".format(treeType,train_dataset.__len__(), BATCHSIZE, num_epochs)

    workDir = os.path.join(workDir, modelID) + "\\"
    os.makedirs(workDir, exist_ok=True)


    print("Length of train　data:\t\t{}".format(len(train_data)))
    print("Length of validation　data:\t{}".format(len(valid_data)))
    print("Length of ALL　data:\t\t{}".format(train_dataset.__len__()))

    print("batchSize",BATCHSIZE)
    print("epoch",num_epochs)



    #<---------------各インスタンス作成---------------------->
    # model = UNet(3,1).cuda() 自作モデル
    from unet_model import UNet, DiceBCELoss, save_ckp, load_ckp
    from pre_segmentation_model import UnetModel, calculate_iou, validateModel

    encoder_name = "resnet34"
    encoder_weight = "imagenet"

    model = UnetModel(encoder_name,encoder_weight,3,1)
    modelID = "class{}_data{}_batch{}_epoch{}_model{}".format(treeType,train_dataset.__len__(), BATCHSIZE, num_epochs, encoder_name + "-" + encoder_weight)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
    # criterion = DiceLoss()
    # accuracy_metric = IoU()
    valid_loss_min = np.Inf


    torch.cuda.empty_cache()

    # CuDNNを使用したネットワークのベンチマークを有効にし、パフォーマンスを最適化します。
    torch.backends.cudnn.benchmark = True


    checkpoint_path = os.path.join(workDir,f'model/{treeType}_chkpoint_')
    best_model_path = os.path.join(workDir,f'model/{treeType}_bestmodel.pt')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    total_train_loss = []
    total_train_score = []
    total_valid_loss = []
    total_valid_score = []

    losses_value = 0
    for epoch in range(num_epochs):
    #<---------------トレーニング---------------------->
        train_loss = []
        train_score = []
        valid_loss = []
        valid_score = []
        train_loader_pbar = tqdm(train_loader, desc = 'description')


        #<---------------評価---------------------->
        losses_value, score = validateModel(model, train_loader_pbar)

        # for x_train, y_train in pbar:



        #     x_train = torch.autograd.Variable(x_train).cuda()
        #     y_train = torch.autograd.Variable(y_train).cuda()
        #     optimizer.zero_grad()
        #     output = model(x_train)
        #     #   shape (batchsize,ch,width,height)
        #     ## 損失計算
        #     loss = criterion(output, y_train)
        #     losses_value = loss.item()
        #     ## 精度評価
        #     score = accuracy_metric(output,y_train)
        #     loss.backward()
        #     optimizer.step()
        train_loss.append(losses_value)
        train_score.append(score.item())
        
        train_loader_pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")

        losses_value, score = validateModel(model, val_loader)

        valid_loss.append(losses_value)
        valid_score.append(score.item())



        # with torch.no_grad():
        #     for img,msk in val_loader:
        #         img = torch.autograd.Variable(img).cuda()
        #         msk = torch.autograd.Variable(msk).cuda()
        #         output = model(img)
        #         ## 損失計算
        #         loss = criterion(output, msk)
        #         losses_value = loss.item()
        #         ## 精度評価
        #         # score = calculate_iou(output=output, target=msk)
        #         score = accuracy_metric(output,msk)
        #         valid_loss.append(losses_value)
        #         valid_score.append(score.item())

        #学習過程のモデルの予測結果を可視化
        img_show,msk_show = val_loader.next()
        img_show = img_show[0]
        msk_show = msk_show[0]
        pred = model(img_show)[0][0]

        showPred(img_show,msk_show,pred,imgSave=True)

        total_train_loss.append(np.mean(train_loss))
        total_train_score.append(np.mean(train_score))
        total_valid_loss.append(np.mean(valid_loss))
        total_valid_score.append(np.mean(valid_score))
        print(f"Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}")
        print(f"Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}")

        
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': total_valid_loss[-1],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        

        # checkpointの保存
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        # 評価データにおいて最高精度のモデルのcheckpointの保存
        if total_valid_loss[-1] <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,total_valid_loss[-1]))
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = total_valid_loss[-1]
        

        print("")
        # print("epoch",epoch)
        # print_allocated_tensors()
        torch.cuda.empty_cache()


    score = {
        # "epoch" : range(1,num_epochs+1),
        "train_Loss" : total_train_loss,
        "valid__Loss" : total_valid_loss,
        "train_scoreIoU" : total_train_score,
        "valid__scoreIoU" : total_valid_score,
        }

    import pandas as pd
    df_score = pd.DataFrame(score, index=range(1,num_epochs+1))
    df_score.to_csv(workDir + f"scoreSheet_{modelID}.csv")

    data = {'dataset_Mean': mean_values, 'dataset_Std Deviation': std_deviation}
    df_statics = pd.DataFrame(data)

    # DataFrameをCSVファイルとして保存
    df_statics.to_csv(workDir + f"statistics_{modelID}.csv", index=False)
    
    