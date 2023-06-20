
import os, glob, re, datetime
from tqdm import tqdm
import torch
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from tqdm import tqdm as tqdm
from PIL import Image
import random

mean_values, std_deviation = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])#A.Normalize()のデフォルト値


#画像データ拡張の関数
def get_train_transform(resizeValue):
   return A.Compose(
       [
        #リサイズ(こちらはすでに適用済みなのでなくても良いです)
        A.Resize(resizeValue, resizeValue),
        #正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # A.Normalize(),
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
            # img = transform.resize(img,(resizeValue,resizeValue))

            # mask = self.get_mask(mask_path, resizeValue, resizeValue ).astype('float32')

            print("Read",image_path)
            p_calc(img)

            augmented = self.transforms(image=img)
            # augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']

            print("Transformed")
            p_calc(img)
            # mask = augmented['mask']
            # mask = mask.permute(2, 0, 1)


            # # 可視化
            # figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 8))
            # ax[0,0].imshow(img.permute(1, 2, 0))#img画像は正規化しているため色味がおかしい
            # ax[0,1].imshow(mask[0,:,:])

            return img, image_path


# !pip install segmentation_models_pytorch
from segmentation_models_pytorch.utils.metrics import IoU

def calculate_iou(output, target):

    iou_metric = IoU(threshold=0.5)
    iou = iou_metric(output, target)
    return iou

from matplotlib.font_manager import FontProperties


def p_calc(img):
    print(img.max(),img.min(),img.mean())

fontFpath = r'meiryob.ttc'
if os.path.exists(fontFpath):
    fp = FontProperties(fname=fontFpath, size=16)


from segment_model_training import get_mskPath, format_image
from preprocess import norm

def predictImage(model,resizeValue,imgPaths, lowThreshold:int, pickMaskPath=False):
    failedImgPaths = []
    pred_dataset = LoadPredDataSet(imgPaths, resizeValue, transform=get_train_transform(resizeValue))

    image,image_path = pred_dataset.__getitem__(1)

    # GPUを使用する場合
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("torchDevice",device)
    # model = model.to(device)

    pred_loader = DataLoader(dataset=pred_dataset, batch_size=1, shuffle=False)
    for image,image_path in pred_loader:
        image_path = image_path[0]#DataLoaderから取得するとtupleで返るので0で取る
        image = image.to(device)

        # 予測の実行
        with torch.no_grad():
            predict = model(image)

        image = image.data.cpu().numpy()#(1,3,imgsize,imgsize)
        image = np.squeeze(image)#(3,imgsize,imgsize)
        predict = predict.data.cpu().numpy()#(1,1,imgsize,imgsize)
        predict = np.squeeze(predict)#(imgsize,imgsize)



        predict = np.expand_dims(predict,axis=-1)#(1,imgsize,imgsize)

        # print(np.unique(predict))

        # img = format_image(image,mean_values=mean_values, std_deviation=std_deviation)
        img = np.array(np.transpose(image, (1,2,0)))
        # pred = format_mask(predict)
        
        img = norm(img)
        
        pred = predict
        if pickMaskPath==True:
            # print(image_path)
            mskPath = get_mskPath(image_path)
            # print(mskPath)
            msk = np.array(Image.open(mskPath).resize([resizeValue,resizeValue]).convert("L"))
            msk_iou = msk.copy()
            msk_iou[msk_iou!=0]=1.0

            # print(np.unique(msk_iou),np.unique(pred))
            iouVal = calculate_iou(torch.from_numpy(np.squeeze(pred)),torch.from_numpy(msk_iou)).item()
            # print(int(iouVal))
            if int(iouVal*100)<=lowThreshold:
                failedImgPaths.append(image_path)
                # continue
            print(image_path)
            figure, ax = plt.subplots(1,3)
            # print(images)
            ax[0].imshow(img)
            ax[1].imshow(pred, interpolation="nearest", cmap="gray")
            ax[2].imshow(msk, interpolation="nearest", cmap="gray")
            # try:
            #     ax[0].set_title("元画像", fontproperties=fp)
            #     ax[1].set_title("AI予測", fontproperties=fp)
            #     ax[2].set_title("正解画像", fontproperties=fp)
            #     ax[1].set_xlabel(f"正解率:{str(round(iouVal*100,2))}", fontproperties=fp)  # x軸のラベルを設定
            # except:
            ax[0].set_title("Original")
            ax[1].set_title("Predict")
            ax[2].set_title("Mask")
            ax[1].set_xlabel(f"IoU:{str(round(iouVal*100,2))}")  # x軸のラベルを設定
            ax[1].set_yticks([])  # y軸の目盛りを非表示にする
            ax[1].set_xticks([])  # x軸の目盛りを非表示にする
            ax[0].set_axis_off()
            # ax[1].set_axis_off()
            ax[2].set_axis_off()


        else:
            figure, ax = plt.subplots(1,2)
            # print(images)
            ax[0].imshow(img)
            ax[1].imshow(pred, interpolation="nearest", cmap="gray")
            ax[0].set_title("Oeiginal Image")
            ax[1].set_title("Predict Mask")
            ax[0].set_axis_off()
            ax[1].set_axis_off()
        # plt.tight_layout()
        plt.show()
        plt.close()

        # print(f"正解率:{str(round(iouVal*100,2))}%")
    return failedImgPaths

