#!/usr/bin/env python
# coding: utf-8

# # U-Netを用いたUAV画像セグメンテーションについて


# ①データの確認、探索  
# ②データの前処理  
# ③U-Netのモデルの定義、トレーニング  
# ④U-Netモデルの性能評価の確認

import os, glob, sys, re
from tqdm import tqdm
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from tqdm import tqdm as tqdm
import cv2
from fpathutils import copyLocaliImages, trainPairCheck
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available(),device)


className =sys.argv[1]
num_epochs=int(sys.argv[2])
BATCHSIZE = int(sys.argv[3])
resizeValue = int(sys.argv[4])
datasetDirName = sys.argv[5]

# orgDir = f"03_datasetforModel/Forest tsumura 2 50m P4Pv2_{className}/org_crop4Corner_5120_3072_Size1024_lap512_rotate_flipMirror"
# orgDir = f"03_datasetforModel/Forest tsumura 2 50m P4Pv2_{className}/org_crop4Corner_5120_3072_Size1024_lap512"
orgDir = f"03_datasetforModel\\Forest tsumura 2 50m P4Pv2_{className}\\{datasetDirName}"

imageSize = re.search(".*_Size(\d+)_lap.*",datasetDirName).group(1)

# os.makedirs(orgDir,exist_ok=True)

from u_net_pytorch import get_train_transform, LoadDataSet

orgDir = copyLocaliImages(orgDir, copyDir=f"C:\\datas\\uav_cnn_{className}")
trainPairCheck(orgDir=orgDir)

train_dataset = LoadDataSet(orgDir, resizeValue, transform=get_train_transform(resizeValue))
print("datasets count\t",train_dataset.__len__())

image, mask = train_dataset.__getitem__(3)
print(image.shape)
print(mask.shape)



from genericpath import exists
split_valid_ratio = 0.2
train_size=int(np.round(train_dataset.__len__()*(1 - split_valid_ratio),0))
valid_size=int(np.round(train_dataset.__len__()*split_valid_ratio,0))

# BATCHSIZE = train_dataset.__len__()//20
# BATCHSIZE = 8

train_data, valid_data = random_split(train_dataset, [train_size, valid_size])
train_loader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
val_loader = DataLoader(dataset=valid_data, batch_size=BATCHSIZE)

if num_epochs<=20:
    modelID = f"Test_data{train_dataset.__len__()}_modelResize-{str(resizeValue).zfill(4)}_batch{BATCHSIZE}_epoch{num_epochs}_class-{className}"
else:
    modelID = f"data{train_dataset.__len__()}_imageSize-{imageSize}_modelResize-{str(resizeValue).zfill(4)}_batch{BATCHSIZE}_epoch{num_epochs}_class-{className}"


workDir = "04_trainingModel"
workDir = os.path.join(workDir, modelID)
os.makedirs(workDir, exist_ok=True)
print(workDir)

print("Length of train　data:\t\t{}".format(len(train_data)))
print("Length of validation　data:\t{}".format(len(valid_data)))
print("Length of ALL　data:\t\t{}".format(train_dataset.__len__()))


import u_net_pytorch
import importlib
importlib.reload(u_net_pytorch)
from u_net_pytorch import UNet, IoU, DiceBCELoss, DiceLoss, save_ckp, load_ckp, format_image, format_mask, EarlyStopping



#<---------------各インスタンス作成---------------------->
model = UNet(3,1).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
criterion = DiceLoss()
accuracy_metric = IoU()
valid_loss_min = np.Inf


checkpoint_path = os.path.join(workDir,'chkpoint_')
best_model_path = os.path.join(workDir,'bestmodel.pt')


total_train_loss = []
total_train_score = []
total_valid_loss = []
total_valid_score = []

# EarlyStoppingの初期化

early_stopping = EarlyStopping(patience=10, delta=0.2)
'''deltaを0.2に設定した場合、EarlyStoppingは検証損失が前回の最小値から
0.2以上改善しない場合にカウンタをインクリメントします。
つまり、前回の最小値が0.5だった場合、
新しい損失が0.3以上であれば改善とみなされ、カウンタはリセットされます。
しかし、新しい損失が0.5から0.7までの範囲であれば改善とみなされず、
カウンタがインクリメントされます。我慢エポック数(patience)に達した時点で学習が停止します。
'''

importlib.reload(u_net_pytorch)
from u_net_pytorch import visualize_training_predict

losses_value = 0
for epoch in range(num_epochs):
    #<---------------トレーニング---------------------->
    train_loss = []
    train_score = []
    valid_loss = []
    valid_score = []
    pbar = tqdm(train_loader, desc = 'description')
    for x_train, y_train in pbar:
        x_train = torch.autograd.Variable(x_train).cuda()
        y_train = torch.autograd.Variable(y_train).cuda()
        optimizer.zero_grad()
        output = model(x_train)
        ## 損失計算
        loss = criterion(output, y_train)
        losses_value = loss.item()
        ## 精度評価
        score = accuracy_metric(output,y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(losses_value)
        train_score.append(score.item())
        pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")
      #<---------------評価---------------------->
        with torch.no_grad():
            for image,mask in val_loader:
                image = torch.autograd.Variable(image).cuda()
                mask = torch.autograd.Variable(mask).cuda()
                output = model(image)
                ## 損失計算
                loss = criterion(output, mask)
                losses_value = loss.item()
                ## 精度評価
                score = accuracy_metric(output,mask)
                valid_loss.append(losses_value)
                valid_score.append(score.item())

    
            if epoch%10==0:
                visualize_training_predict(image,mask,output,workDir,True,True)

            
        total_train_loss.append(np.mean(train_loss))
        total_train_score.append(np.mean(train_score))
        total_valid_loss.append(np.mean(valid_loss))
        total_valid_score.append(np.mean(valid_score))
    print(f"Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}",f"Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}")


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
        print(checkpoint_path)

    
    if epoch%10==0:

        csvPath = os.path.join(workDir,f"scoreSheet_{modelID}_unet.csv")
        score = {
        # "epoch" : range(1,num_epochs+1),
        "train_Loss" : total_train_loss,
        "valid__Loss" : total_valid_loss,
        "train_scoreIoU" : total_train_score,
        "valid__scoreIoU" : total_valid_score,
        }

        df_score = pd.DataFrame(score, index=range(1,len(total_train_score)+1))
        df_score.to_csv(csvPath)
        print("saved Score\n",csvPath)

    # EarlyStoppingの評価
    early_stopping(np.mean(valid_loss))

    # EarlyStoppingが有効化された場合は学習を停止
    if early_stopping.early_stop:
        print("Epoch EarlyStop",epoch)
        

        
        break

    print("Epoch End",epoch)



csvPath = os.path.join(workDir,f"scoreSheet_{modelID}_unet.csv")
score = {
# "epoch" : range(1,num_epochs+1),
"train_Loss" : total_train_loss,
"valid__Loss" : total_valid_loss,
"train_scoreIoU" : total_train_score,
"valid__scoreIoU" : total_valid_score,
}


df_score = pd.DataFrame(score, index=range(1,len(total_train_score)+1))
df_score.to_csv(csvPath)
print("saved Score\n",csvPath)

import seaborn as sns

plt.figure(1)
plt.figure(figsize=(15,5))
sns.set_style(style="darkgrid")
plt.subplot(1, 2, 1)
sns.lineplot(x=range(1,num_epochs+1), y=total_train_loss, label="Train Loss")
sns.lineplot(x=range(1,num_epochs+1), y=total_valid_loss, label="Valid Loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("DiceLoss")

plt.subplot(1, 2, 2)
sns.lineplot(x=range(1,num_epochs+1), y=total_train_score, label="Train Score")
sns.lineplot(x=range(1,num_epochs+1), y=total_valid_score, label="Valid Score")
plt.title("Score (IoU)")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xlabel("epochs",fontsize=18)
plt.ylabel("IoU",fontsize=18)
plt.tick_params(labelsize=18)

plt.savefig(os.path.join(workDir,f"Unet_score_{modelID}.png"))
# plt.show()
plt.close()


# ## ④U-Netモデルの性能評価の確認
best_model_path = os.path.join(workDir,'bestmodel.pt')
model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)





# 続いて入力画像と教師データ、モデルによる出力を表示する関数を用意し、出力を行います。

outImagePath = os.path.join(workDir,modelID,"OutImages")
os.makedirs(outImagePath,exist_ok=True)

predict_orgPaths = glob.glob(os.path.join(orgDir,"DJI_0065*.jpg"))
predict_orgPaths = random.sample(predict_orgPaths, 3)


def visualize_predict(model, n_images, imgSave=False):
    figure, ax = plt.subplots(nrows=n_images, ncols=3, figsize=(15, 5*n_images))
    with torch.no_grad():
        for data,mask in val_loader:
            data = torch.autograd.Variable(data, volatile=True).cuda()
            mask = torch.autograd.Variable(mask, volatile=True).cuda()
            o = model(data)
            break
    for img_no in tqdm(range(0, n_images)):
        tm=o[img_no][0].data.cpu().numpy()
        img = data[img_no].data.cpu()
        msk = mask[img_no].data.cpu()
        img = format_image(img)
        msk = format_mask(msk)
        
        

        # msk = cv2.medianBlur(np.array(msk),11)

        outImagePath_pred = os.path.join(workDir,"OutImages",f"{img_no}_pred.png")
        outImagePath_img = os.path.join(workDir,"OutImages",f"{img_no}_org.png")
        outImagePath_msk = os.path.join(workDir,"OutImages",f"{img_no}_msk.png")
        os.makedirs(os.path.dirname(outImagePath_pred),exist_ok=True)

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
        plt.savefig(os.path.join(workDir, f"predictedSet_{modelID}.png"))
    # plt.show()

visualize_predict(model, 5, imgSave=True)
# visualize_predict(model, train_dataset.__len__()//20-1, imgSave=True)

