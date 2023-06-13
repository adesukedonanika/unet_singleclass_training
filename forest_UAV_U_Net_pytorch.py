#!/usr/bin/env python
# coding: utf-8

# # U-Netを用いたUAV画像セグメンテーションについて

# 実装の流れとしては、ネット上でも良く取り扱われている他のチュートリアルとほぼ同様に、以下の流れで進めます。　　
# 
# ①データの確認、探索  
# ②データの前処理  
# ③U-Netのモデルの定義、トレーニング  
# ④U-Netモデルの性能評価の確認

# また、画像のアップロードや画像の処理やモデルのトレーニングを行う際に長い時間を要しますので、十分な時間を確保した上で以下を実行お願いします。（実際に、以下のコード実行は2時間ほどかかりました。）
# 
# また、モデルのトレーニングを行う場合はGPUの使用をお勧めします。Google Colabを使う場合は上部『ランタイム』タブから『ランタイムのタイプを変更』を選択し、ハードウェアアクセラレータをGPUに変更をお願いします。

# ## ①データの確認、探索

# 今回用いるデータセットはkaggleのコンペティションで用いられたデータセットを用います。
# （https://www.kaggle.com/c/data-science-bowl-2018/data）　
# 
# もし本記事に記載されているコードを実行する場合は、上のサイトから一度お使いのパソコンにデータセットをダウンロードし、そのデータセットをこのcolabノートブック上にアップロードする必要があります。

# トレーニングデータのzipファイルであるstage1_train.zipをアップロードしてから以下を実行してください。(アップロードは10分ぐらい時間がかかりました。)

# まず、zipファイルの解凍を行います。

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


# %cd /content/drive/MyDrive/Forest/src


# In[3]:


# !nvidia-smi


# In[4]:


# !ls


# In[5]:


# ! unzip U_Net_tutorial_pytorch.ipynb_data/stage1_train.zip -d U_Net_tutorial_pytorch.ipynb_data/stage1_train


# 必要なライブラリをインポートしておきます。

# In[6]:


import os, glob, sys
import re, shutil
from tqdm import tqdm
import time
import copy
from collections import defaultdict
import torch
import shutil
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch import nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from tqdm import tqdm as tqdm

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import cv2

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
from PIL import Image
from torch import nn
import zipfile

import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available(),device)


# In[ ]:





# In[7]:



className =sys.argv[1]
TRAIN_PATH = f"03_datasetforModel/Forest tsumura 2 50m P4Pv2_{className}/org_crop4Corner_5120_3072_Size1024_lap512_rotate_flipMirror"

# os.makedirs(TRAIN_PATH,exist_ok=True)


# In[8]:


# remakeFolder = "./training/hikariTestArea/org/256rap192__2445_50cm_hikariTestArea"

# for fileName in tqdm(glob.glob(remakeFolder + "/*.tif")):

# # print(fileNames)
#     imageID = re.findall(r'.*hikariTestArea_(\d+_\d+)\.tif', fileName)[0]

#     imageID = imageID + f"_{className}"

#     orgIamgeDirPath = os.path.join(TRAIN_PATH,imageID,"org")
#     os.makedirs(orgIamgeDirPath,exist_ok=True) 
#     mskImageDirPath = orgIamgeDirPath.replace("org","msk")
#     os.makedirs(mskImageDirPath,exist_ok=True) 

#     orgPath = os.path.join(orgIamgeDirPath,os.path.basename(fileName))
#     mskPath = os.path.join(mskImageDirPath,os.path.basename(fileName.replace("org","msk")))

#     if not os.path.exists(orgPath):
#         shutil.copy(fileName, orgPath)
#     if not os.path.exists(mskPath):
#         shutil.copy(fileName.replace("org","msk"), mskPath)


# 続いてデータセットの読み込みを行います。
# 
# ここでは前処理の一部である画像のリサイズと画像データ拡張も同時に行います。
# 
# 画像のリサイズは様々なサイズの画像を全て固定のサイズに調整することにより1つのモデルによる対応が可能になります。
# 
# ここでは全ての画像を256×256の画像にリサイズしています。
# 
# また画像データ拡張は最近の画像系のディープラーニングの前処理とは一般的な処理で、画像に処理を加えることによりモデルの汎用化（予測精度を高める）ことが可能になります。
# 
# ここでは画像の正規化と水平垂直方向に画像をフリップさせる処理を追加しています。

# 「mask」はマスキング、つまり細胞のセグメンテーションがなされているデータであり、これが教師データとなります。
# 
# マスクに関しては今回は１つの細胞ごとに1つのファイルとなっているので、複数の画像を1つにまとめています。

# トレーニングデータは一般的に入力画像と教師データ(mask)をペアとしてまとめ、このペアにより学習を行います。pytorchではこれらをDatasetクラスを用いてまとめます。

# In[9]:


from fpathutils import get_mskPath


# In[10]:


resizeValue = 512

#画像データ拡張の関数
def get_train_transform():
   return A.Compose(
       [
        #リサイズ(こちらはすでに適用済みなのでなくても良いです)
        A.Resize(resizeValue, resizeValue),
        #正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.Normalize(),
        #水平フリップ（pはフリップする確率）
        A.HorizontalFlip(p=0.25),
        #垂直フリップ
        A.VerticalFlip(p=0.25),
        ToTensor()
        ])

# def mask2single(mask,values:list):
#     for value in values:
#         mask[mask==value] = 255
#     mask[mask!=255] = 0
#     return mask



#Datasetクラスの定義
class LoadDataSet(Dataset):
        def __init__(self,path, transform=None):
            self.path = path
            self.folders = os.listdir(path)
            self.transforms = get_train_transform()
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path, self.folders[idx])
            mask_path = get_mskPath(os.path.join(self.path, self.folders[idx]))
            # image_path = os.path.join(image_folder,os.listdir(image_folder)[idx])
            
            #画像データの取得
            img = io.imread(image_path)[:,:,0:3].astype('float32')
            img = transform.resize(img,(resizeValue,resizeValue))
            
            mask = self.get_mask(mask_path, resizeValue, resizeValue ).astype('float32')


            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            mask = mask.permute(2, 0, 1)


            # # 可視化
            # figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 8))
            # ax[0,0].imshow(img.permute(1, 2, 0))#img画像は正規化しているため色味がおかしい
            # ax[0,1].imshow(mask[0,:,:])

            return (img,mask) 

        #マスクデータの取得
        def get_mask(self,mask_path,IMG_HEIGHT, IMG_WIDTH):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
            # for mask_ in os.listdir(mask_folder):
            mask_ = io.imread(mask_path)
            mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_ = np.expand_dims(mask_,axis=-1)
            mask = np.maximum(mask, mask_)              
            return mask


# In[11]:


train_dataset = LoadDataSet(TRAIN_PATH, transform=get_train_transform())
len(train_dataset)

image, mask = train_dataset.__getitem__(3)
print(image.shape)
print(mask.shape)


# 一枚の画像データとマスクの次元を確認します。

# 画像枚数を確認します。

# In[12]:


#Print total number of unique images.
train_dataset.__len__()


# 次に、入力画像とマスクのデータがどうなっているのか確認してみます。

# In[13]:


def format_image(img):
    img = np.array(np.transpose(img, (1,2,0)))
    #下は画像拡張での正規化を元に戻しています
    mean=np.array((0.485, 0.456, 0.406))
    std=np.array((0.229, 0.224, 0.225))
    img  = std * img + mean
    img = img*255
    img = img.astype(np.uint8)
    return img


def format_mask(mask):
    mask = np.squeeze(np.transpose(mask, (1,2,0)))
    return mask


plt.clf()

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
    # plt.show()
# visualize_dataset(4)


# In[ ]:





# 左列が入力画像、右列がマスクデータとなっています。左列で細胞がある箇所に右列でマスクがなされていることが確認できます。U-Netは左の画像を入力した際に右のようなマスクされた画像データが出力できれば良いということになります。

# ## ②データの前処理

# 続いて評価データ作成のため、トレーニングデータの一部を評価データとして分割します。またpytorchではミニバッチ処理ができるようにDataLoaderクラスを作成します。

# In[14]:


from genericpath import exists
split_valid_ratio = 0.2
train_size=int(np.round(train_dataset.__len__()*(1 - split_valid_ratio),0))
valid_size=int(np.round(train_dataset.__len__()*split_valid_ratio,0))

# BATCHSIZE = train_dataset.__len__()//20
BATCHSIZE = 4
num_epochs=int(sys.argv[2])

# BATCHSIZE = 8
train_data, valid_data = random_split(train_dataset, [train_size, valid_size])
train_loader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
val_loader = DataLoader(dataset=valid_data, batch_size=BATCHSIZE)


modelID = f"data{train_dataset.__len__()}_modelResize-{str(resizeValue).zfill(4)}_batch{BATCHSIZE}_epoch{num_epochs}_class-{className}"

workDir = "04_trainingModel"
workDir = os.path.join(workDir, modelID)
os.makedirs(workDir, exist_ok=True)
print(workDir)

print("Length of train　data:\t\t{}".format(len(train_data)))
print("Length of validation　data:\t{}".format(len(valid_data)))
print("Length of ALL　data:\t\t{}".format(train_dataset.__len__()))


# 続いてU-Netのモデルを実装します。モデルについては解説記事か以下のサイトをご参照ください。
# https://www.researchgate.net/figure/U-net-Convolutional-Neural-Network-model-The-U-net-model-contains-two-parts_fig6_317493482
# 　　　 
# 
# こちらを元に実装をしていきます。

# U-Netモデルにおいては細かい構成というよりはモデルの全体構成から把握していった方が理解がしやすいと思いました。
# 
# U-Net解説記事にも記載してあります通り、
# 
# ①FCNにあたる部分、
# 
# ②Up Samplingにあたる部分、
# 
# ③Skip Connectionにあたる部分
# 
# をまず把握します。
# 
# 以下のコードコメント文にそれぞれがどこに該当するかを記載しています。
# 
# Skip Connectionはtorch.catによりFCN時の出力と合わせています。
# 
# conv_bn_relu関数は畳み込みとバッチ正規化と活性化関数Reluをまとめています。

# セマンティックセグメンテーションの損失関数としてはBCELoss(Binary Cross Entropy)をベースとしたDiceBCELossがよく用いられます。詳細な説明とコードは以下に記載があります。https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# 
# 考え方としてはIoUに近く、予測した範囲が過不足なく教師データとなる領域を捉えているほど損失が低くなります。

# In[15]:


import u_net_pytorch
import importlib
importlib.reload(u_net_pytorch)
from u_net_pytorch import UNet, down_pooling, conv_bn_relu, up_pooling, IoU, DiceBCELoss, DiceLoss, save_ckp, load_ckp


#<---------------各インスタンス作成---------------------->
model = UNet(3,1).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
criterion = DiceLoss()
accuracy_metric = IoU()
valid_loss_min = np.Inf


# In[ ]:





# In[16]:


checkpoint_path = os.path.join(workDir,'chkpoint_')
best_model_path = os.path.join(workDir,'bestmodel.pt')


total_train_loss = []
total_train_score = []
total_valid_loss = []
total_valid_score = []


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

      print("")

csvPath = os.path.join(workDir,f"scoreSheet_{modelID}_unet.csv")
score = {
    # "epoch" : range(1,num_epochs+1),
    "train_Loss" : total_train_loss,
    "valid__Loss" : total_valid_loss,
    "train_scoreIoU" : total_train_score,
    "valid__scoreIoU" : total_valid_score,
    }

import pandas as pd

df_score = pd.DataFrame(score, index=range(1,num_epochs+1))
df_score.to_csv(csvPath)
print("saved Score\n",csvPath)


# In[17]:


# !pip install seaborn
import seaborn as sns

def plotScore(csvPath:str, strSize:int):

    df = pd.read_csv(csvPath)
    plt.figure(1)
    plt.figure(figsize=(15,5))
    sns.set_style(style="darkgrid")
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(1,len(df)+1), y=df["train_Loss"], label="Train Loss")
    sns.lineplot(x=range(1,len(df)+1), y=df["valid__Loss"], label="Valid Loss")
    # plt.title("Loss")
    plt.legend(fontsize=strSize)

    lossYrangeMax = df[["train_Loss",	"valid__Loss"]].max().max()
    lossYrangeMin = df[["train_Loss",	"valid__Loss"]].min().min()
    lossYrangeMax = round(lossYrangeMax, 2)
    lossYrangeMin = round(lossYrangeMin, 2)
    print(lossYrangeMin, lossYrangeMax)
    plt.yticks(np.arange(lossYrangeMin, lossYrangeMax+0.1, step=0.2))
    plt.title("Loss")
    plt.xlabel("epochs",fontsize=strSize)
    plt.ylabel("DiceLoss",fontsize=strSize,labelpad=-20)
    plt.tick_params(labelsize=strSize)


    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(1,len(df)+1), y=df["train_scoreIoU"], label="Train Score")
    sns.lineplot(x=range(1,len(df)+1), y=df["valid__scoreIoU"], label="Valid Score")
    # plt.title("Score (IoU)")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.legend(fontsize=strSize)
    plt.title("Score (IoU)")
    plt.xlabel("epochs",fontsize=strSize)
    plt.ylabel("IoU",fontsize=strSize,labelpad=-30)
    plt.tick_params(labelsize=strSize)

    figPath = os.path.join(workDir,f"Unet_score_{os.path.basename(csvPath)}.png")
    plt.savefig(figPath,facecolor="azure", bbox_inches='tight', pad_inches=0)
    print(figPath)
    # plt.show()

plotScore(csvPath,10)


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

df_score


# ## ④U-Netモデルの性能評価の確認

# ### 出力したcsvで可視化

# 学習と評価が終了しましたので、エポックごとの損失、精度の変化をグラフ化します。

# 以上の通り、エポックが進むにつれて損失が減り、精度が向上していることがわかります。これは機械学習においてはモデルの学習が進み、より汎化性能（予測性能）が増して行っていることを意味しています。

# 次に、作成した学習したモデルを利用して、実際のモデルによるセマンティックセグメンテーションの結果を表示してみましょう。
# 
# まず作成したモデルを読み込みます。

# 

# In[18]:


# !ls training/classConifer_data440_batch22_epoch100


# In[19]:


best_model_path = os.path.join(workDir,'bestmodel.pt')
model, optimizer, start_epoch, valid_loss_min = load_ckp(best_model_path, model, optimizer)


# In[20]:




# 続いて入力画像と教師データ、モデルによる出力を表示する関数を用意し、出力を行います。

# In[21]:


outImagePath = os.path.join(workDir,modelID,"OutImages")
os.makedirs(outImagePath,exist_ok=True)

predict_orgPaths = glob.glob(os.path.join(TRAIN_PATH,"DJI_0065*.jpg"))
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
