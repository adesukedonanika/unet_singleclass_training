
import os, glob, re, datetime, shutil
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


# def _convert_to_wandb_image(image: Union[np.ndarray, torch.Tensor], channels_last: bool):
#     if isinstance(image, torch.Tensor):
#         image = image.data.cpu().numpy()

#     # Error out for empty arrays or weird arrays of dimension 0.
#     if np.any(np.equal(image.shape, 0)):
#         raise ValueError(f'Got an image (shape {image.shape}) with at least one dimension being 0! ')

#     # Squeeze any singleton dimensions and then add them back in if image dimension
#     # less than 3.
#     image = image.squeeze()

#     # Add in length-one dimensions to get back up to 3
#     # putting channels last.
#     if image.ndim == 1:
#         image = np.expand_dims(image, (1, 2))
#         channels_last = True
#     if image.ndim == 2:
#         image = np.expand_dims(image, 2)
#         channels_last = True

#     if image.ndim != 3:
#         raise ValueError(
#             textwrap.dedent(f'''Input image must be 3 dimensions, but instead
#                             got {image.ndim} dims at shape: {image.shape}
#                             Your input image was interpreted as a batch of {image.ndim}
#                             -dimensional images because you either specified a
#                             {image.ndim + 1}D image or a list of {image.ndim}D images.
#                             Please specify either a 4D image of a list of 3D images'''))

#     if not channels_last:
#         assert isinstance(image, np.ndarray)
#         image = image.transpose(1, 2, 0)
#     return image


# def im_detect_keypoints(model, im_scale, boxes, blob_conv):
#     """Infer instance keypoint poses. This function must be called after
#     im_detect_bbox as it assumes that the Caffe2 workspace is already populated
#     with the necessary blobs.

#     Arguments:
#         model (DetectionModelHelper): the detection model to use
#         im_scale (list): image blob scales as returned by im_detect_bbox
#         boxes (ndarray): R x 4 array of bounding box detections (e.g., as
#             returned by im_detect_bbox)

#     Returns:
#         pred_heatmaps (ndarray): R x J x M x M array of keypoint location
#             logits (softmax inputs) for each of the J keypoint types output
#             by the network (must be processed by keypoint_results to convert
#             into point predictions in the original image coordinate space)
#     """
#     M = cfg.KRCNN.HEATMAP_SIZE
#     if boxes.shape[0] == 0:
#         pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
#         return pred_heatmaps

    # inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scale)}

    # # Add multi-level rois for FPN
    # if cfg.FPN.MULTILEVEL_ROIS:
    #     _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    # pred_heatmaps = model.module.keypoint_net(blob_conv, inputs)
    # pred_heatmaps = pred_heatmaps.data.cpu().numpy().squeeze()

    # # In case of 1
    # if pred_heatmaps.ndim == 3:
    #     pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    # return pred_heatmaps



#画像データ拡張の関数
def get_train_transform(resizeValue,normalize):
    if normalize:
        return A.Compose(
            [
                #リサイズ
                A.Resize(resizeValue, resizeValue),
                #正規化(こちらの細かい値はalbumentations.augmentations.transforms.Normalizeのデフォルトの値を適用)
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Normalize(),
                ToTensor()
                ])
    else:
        return A.Compose(
            [
                #リサイズ
                A.Resize(resizeValue, resizeValue),
                ToTensor()
                ])
        


class LoadPredDataSet(Dataset):
        def __init__(self,imgPaths,resizeValue, normalize, transform=None):
            self.path = imgPaths
            self.folders = [os.path.basename(imgPath) for imgPath in imgPaths]
            self.transforms = get_train_transform(resizeValue, normalize=normalize)
            self.resizeValue = resizeValue
            self.normalize = normalize

        def __len__(self):
            return len(self.folders)


        def __getitem__(self, idx):
            image_path = self.path[idx]
            resizeValue = self.resizeValue

            #画像データの取得
            img = io.imread(image_path)[:,:,0:3].astype('float32')
            # img = transform.resize(img,(resizeValue,resizeValue))
            # mask = self.get_mask(mask_path, resizeValue, resizeValue ).astype('float32')

            # print("Read",image_path)
            # p_calc(img)

            augmented = self.transforms(image=img)
            img = augmented['image']

            # print("Transformed")
            # p_calc(img)

            return img, image_path


# !pip install segmentation_models_pytorch
from segmentation_models_pytorch.utils.metrics import IoU

def calculate_iou(output, target):
    iou_metric = IoU(threshold=0.5)
    iou = iou_metric(output, target)
    return iou

from matplotlib.font_manager import FontProperties


def p_calc(img):
    print("Max",img.max(),"Min",img.min(),"Mean",img.mean())

fontFpath = r'meiryob.ttc'
if os.path.exists(fontFpath):
    fp = FontProperties(fname=fontFpath, size=16)


from segment_model_training import get_mskPath, format_image
from preprocess import norm

def predictImage(model,resizeValue,imgPaths, lowThreshold:int,saveDir:str, pickMaskPath=False,normalize=False, imgShow=False):
    failedImgPaths = []
    predictedImages = {}
    pred_dataset = LoadPredDataSet(imgPaths, resizeValue, normalize=False, transform=get_train_transform(resizeValue=resizeValue, normalize=normalize))

    image,image_path = pred_dataset.__getitem__(0)

    if isinstance(image, torch.Tensor):
        image = image.data.cpu().numpy()

    # GPUを使用する場合
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("torchDevice",device)
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
        # Threshold = lowThreshold/100
        # predict[predict>=Threshold]=1.0
        # predict[predict<=Threshold]=0.0

        # print(np.unique(predict))

        # img = format_image(image,mean_values=mean_values, std_deviation=std_deviation)
        img = np.array(np.transpose(image, (1,2,0)))#(imgsize,imgsize,3)
        # pred = format_mask(predict)
        
        img = norm(img)
        
        pred = predict
        predictedImages[image_path] = np.squeeze(pred)#(imgsize,imgsize,1)->(imgsize,imgsize)
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

            if imgShow:
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
                # ax[1].set_xlabel(f"IoU:{str(round(iouVal*100,2))}")  # x軸のラベルを設定
                ax[1].set_yticks([])  # y軸の目盛りを非表示にする
                ax[1].set_xticks([])  # x軸の目盛りを非表示にする
                ax[0].set_axis_off()
                # ax[1].set_axis_off()
                ax[2].set_axis_off()
                plt.savefig(os.path.join(saveDir,"pred_" + os.path.basename(image_path)))
                plt.show()
                plt.close()

        else:
            if imgShow:
                figure, ax = plt.subplots(1,2)
                # print(images)
                ax[0].imshow(img)
                ax[1].imshow(pred, interpolation="nearest", cmap="gray")
                ax[0].set_title("Oeiginal Image")
                ax[1].set_title("Predict Mask")
                ax[0].set_axis_off()
                ax[1].set_axis_off()
                # plt.tight_layout()
                plt.savefig(os.path.join(saveDir,"pred_" + os.path.basename(image_path)))
                plt.show()
                plt.close()

        # print(f"正解率:{str(round(iouVal*100,2))}%")
    return failedImgPaths,predictedImages

def getOrgImgInfo(imgName:str):
    if not os.path.exists(imgName):
        pattern = f"./03_datasetforModel/Forest_tsumura_2_50m_P4Pv2_cypress/org/{imgName}.JPG"
        # print(pattern)
        imagePaths = glob.glob(pattern)
    else:
        imagePaths = [imgName]

    if len(imagePaths)==1:
        with Image.open(imagePaths[0]) as img:
            width, height = img.size
        return imagePaths[0], width, height
    else:
        print("Not match fileName",pattern)
        return ""


def getUAVImageName(imgPath):
    splits = os.path.basename(imgPath).split("_")
    return f"{splits[0]}_{splits[1]}"


def getModelResizeSizeClassName(checkpoint_path):
    trainedModelPath = os.path.dirname(checkpoint_path)
    checkpoint_path = f'{trainedModelPath}/bestmodel.pt'
    match_modelResize = re.search(".*modelResize-(\d+)_batch.*",checkpoint_path)
    match_className = re.search(".*class-(.+)/.*",checkpoint_path)

    if match_modelResize and match_className:
        # 数値に変換
        resizeSize = int(match_modelResize.group(1))
        print("model training size\t",resizeSize)
        className = str(match_className.group(1))
        print(className)
    else:
        print("Match not found")
    return resizeSize, className

def getCropLapSize(imageDir):

    match_cropSize = re.search(".*_Size(\d+).*",imageDir)
    match_lapSize = re.search(".*_lap(\d+).*",imageDir)

    if match_cropSize and match_lapSize:
        # 数値に変換
        cropSize = int(match_cropSize.group(1))
        print("model Cropsize\t",cropSize)
        lapSize = str(match_lapSize.group(1))
        print("model Lapsize\t",lapSize)
    else:
        print("Match not found")
    return cropSize,lapSize

import orgImageCrop
import time 

# #画像ファイルパスからcropSize, lapSizeを取得
# datasetDirName = os.path.basename(os.path.dirname(imgPaths[0]))
# cropSize = int(re.search(".*_Size(\d+)_lap.*",datasetDirName).group(1))
# lapSize = int(re.search(".*_lap(\d+).*",datasetDirName).group(1))
# lapSize = 0


def predictUAVImageCropLap(UAVimgPath,saveDir,model,resizeSize,cropSize, lapSize,normalize, className):
    org_UAV_pil = Image.open(UAVimgPath)
    width_uav, height_uav = org_UAV_pil.size

    orgPath_uav = UAVimgPath

    UAVImageName = os.path.basename(orgPath_uav).split(".")[0]

    x_step, y_step, HW = orgImageCrop.getXYtileInfo(orgPath_uav, int(cropSize), int(lapSize))
    x_step, y_step = x_step+1, y_step+1

    os.makedirs("./temp/",exist_ok=True)
    tempImgPath = "./temp/predict_canvas.png"

    predCanvas = Image.new('L',[width_uav, height_uav])
    predCanvas.save(tempImgPath)
    predCanvas = np.array(predCanvas)
    org_UAV = np.array(org_UAV_pil)

    cropPositions = []
    for y in range(y_step):
        for x in range(x_step):
            cropPositions.append([x, y])

    uavDir = os.path.join(saveDir,UAVImageName)
    os.makedirs(uavDir,exist_ok=True)

    for x,y in tqdm(cropPositions):
        # print("\n","x, y =", x, y )

        if lapSize==0:
            x_start = x*cropSize
            x_end = x*cropSize + cropSize
            y_start = y*cropSize
            y_end = y*cropSize + cropSize
        else:
            x_start , x_end = x*lapSize, x*lapSize + cropSize
            y_start , y_end = y*lapSize, y*lapSize + cropSize
        # print(x_start , x_end, y_start , y_end)

        # imgNp_resized[height : width : ch]の構造　 imgNp_true.shape = (h, w, ch)
        # (predictImgSize, predictImgSize, 3) で取り出す　他サイズでは出ない

        if x_end >= width_uav:
            x_start , x_end = width_uav-cropSize, width_uav
        if y_end >= height_uav:
            y_start , y_end = height_uav-cropSize, height_uav
        # if x_end >= width_uav:
        #     x_start , x_end = width_uav-cropSize, width_uav
        # if y_end >= height_uav:
        #     y_start , y_end = height_uav-cropSize, height_uav
        # print("HW",height_pred, width_pred)
        # print(x_start , x_end, y_start , y_end)

        xyStr = "X" + str(x_start).zfill(5) + "to" + str(x_end).zfill(5) + "_" + "Y" + str(y_start).zfill(5) + "to" + str(y_end).zfill(5)
        # print("Makedict imgNp", imgNp.shape, xyStr)
        saveImgDir = "temp/"#os.path.dirname(imgPath) + "_Size" +str(cropSize).zfill(4) + "_lap" + str(lapSize) + "\\"
        os.makedirs(saveImgDir, exist_ok=True)
        saveImgPath = saveImgDir + os.path.basename(orgPath_uav).split(".")[-2] + "_" + xyStr + "." + os.path.basename(orgPath_uav).split(".")[-1].replace("JPG","jpg").replace("PNG","png")

        imgNp = org_UAV[y_start : y_end,x_start : x_end, :].copy()
        # print(imgNp.shape,f"{org_UAV.shape}\n[{x_start} : {x_end}, {y_start} : {y_end}]")
        if imgNp.shape[0]==imgNp.shape[1]:
            crpoImg_org = Image.fromarray(imgNp)
            crpoImg_org.save(saveImgPath)
        predImagePaths = glob.glob(saveImgPath)
        # print("predImagePaths",predImagePaths)
        if len(predImagePaths)==0:
            continue
        
        # normalize = False
        low80ImgPaths, predImages = predictImage(model,
                                                 resizeSize,
                                                 imgPaths=predImagePaths,
                                                 lowThreshold=50,
                                                 saveDir=saveImgDir,
                                                 pickMaskPath=False,
                                                 normalize=normalize,
                                                 imgShow=False)
        # print(predImages,predImages[predImagePaths[0]].shape)
        pred = Image.fromarray(predImages[predImagePaths[0]]).resize([cropSize,cropSize])
        pred = np.array(pred)
        predCanvas[y_start : y_end, x_start : x_end] = pred

        savePredPath = os.path.join(uavDir,os.path.basename(orgPath_uav).split(".")[-2] + "_" + xyStr + "." + os.path.basename(orgPath_uav).split(".")[-1].replace("JPG","jpg").replace("PNG","png"))
        pred = Image.fromarray((pred*100).astype(np.uint8)).convert('L')
        pred.save(savePredPath+".png")
        # time.sleep(0.5)
        shutil.copyfile(saveImgPath,os.path.join(uavDir, os.path.basename(saveImgPath)))
        
        # time.sleep(0.5)
        # twoImgShow(Image.fromarray(pred),Image.fromarray(imgNp))

        # twoImgShow(Image.fromarray(predCanvas),org_UAV_pil)
        # plt.imshow(predCanvas)
        # plt.show()
        # plt.close()

    predCanvas = predCanvas[0:height_uav,0:width_uav]
    # predCanvas[predCanvas!=0]=128

    Image.fromarray(predCanvas).save(os.path.join(saveDir,UAVImageName+"_"+className+".PNG"))

