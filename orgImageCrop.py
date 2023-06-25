
import rasterio
import os,glob,sys,shutil,re,json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import albumentations as A
from tqdm import tqdm
import rasterio
from fpathutils import addStrBeforeExt

def getXYtileInfo(imgPath,cropSize,lapSize):
    img_io = rasterio.open(imgPath)
    org_height, org_width, ch = img_io.height, img_io.width, img_io.count
    img_io.close()
    print("OriginalSize\t",org_height, org_width, ch)

    predictImgSize = cropSize

    if lapSize == 0:
        step_x_col = org_width//predictImgSize
        step_y_row = org_height//predictImgSize
    else:
        step_x_col = (org_width - (predictImgSize - lapSize))// lapSize
        step_y_row = (org_height - (predictImgSize - lapSize))// lapSize
    # step_x_col = org_width - (org_width % lapSize))
    # step_y_row = org_height - (org_height % lapSize)


    resizeSetSlide = ((org_height // cropSize) * cropSize, org_width // cropSize * cropSize)
    
    print(step_x_col, step_y_row, resizeSetSlide)
    return step_x_col, step_y_row, resizeSetSlide



def getCrop4CornerPositions(imgPath:str, cropSize:int):
    '''
    UAV画像のcropサイズを元に、４隅の切り取り画像の位置を返却する。
    ４隅の画像サイズはすべて、cropサイズの倍数のサイズになる。
    '''
    img = Image.open(imgPath)
    img = np.array(img)

    width = img.shape[1]
    height = img.shape[0]


    resizeSetSlide = ((height // cropSize) * cropSize, width // cropSize * cropSize)


    x_start = 0
    x_end = resizeSetSlide[1]
    y_start = 0
    y_end = resizeSetSlide[0]

    upperLeftRect =(x_start, x_end, y_start, y_end)
    print("upperLeftRect",upperLeftRect)


    x_start = width - cropSize
    x_end = width
    y_start = 0
    y_end = resizeSetSlide[0]

    upperRightRect =(x_start, x_end, y_start, y_end)
    print("upperRightRect",upperRightRect)

    x_start = 0
    x_end = resizeSetSlide[1]
    y_start = height - cropSize
    y_end = height

    lowerLeftRect =(x_start, x_end, y_start, y_end)
    print("lowerLeftRect",lowerLeftRect)


    x_start = width - cropSize
    x_end = width
    y_start = height - cropSize
    y_end = height

    lowerRightRect =(x_start, x_end, y_start, y_end)
    print("lowerRightRect",lowerRightRect)
    
    crop4corners_dict = {}
    crop4corners_dict["01upperL"]=upperLeftRect
    crop4corners_dict["02upperR"]=upperRightRect
    crop4corners_dict["03lowerL"]=lowerLeftRect
    crop4corners_dict["04lowerR"]=lowerRightRect

    return crop4corners_dict

def crop4CornersOrgImageBysize(imgPath:str, cropSize:int):
    img = Image.open(imgPath)
    img = np.array(img)

    width = img.shape[1]
    height = img.shape[0]

    resizeSetSlide = ((height // cropSize) * cropSize, width // cropSize * cropSize)

    saveImgDIr = os.path.dirname(imgPath) + f"_crop4Corner_{resizeSetSlide[1]}_{resizeSetSlide[0]}\\"
    os.makedirs(saveImgDIr, exist_ok=True)

    upperLeftRect, upperRightRect, lowerLeftRect, lowerRightRect = getCrop4CornerPositions(imgPath, cropSize)

    
    imgSavePath = saveImgDIr + os.path.basename(addStrBeforeExt(imgPath,"01upperL_rect"))
    x_start, x_end, y_start, y_end = upperLeftRect
    Image.fromarray(img[y_start : y_end, x_start : x_end].copy()).save(imgSavePath)
    
    
    imgSavePath = saveImgDIr + os.path.basename(addStrBeforeExt(imgPath,"02upperR_rect"))
    x_start, x_end, y_start, y_end = upperRightRect
    Image.fromarray(img[y_start : y_end, x_start : x_end].copy()).save(imgSavePath)
    
    imgSavePath = saveImgDIr + os.path.basename(addStrBeforeExt(imgPath,"03lowerL_rect"))
    x_start, x_end, y_start, y_end = lowerLeftRect
    Image.fromarray(img[y_start : y_end, x_start : x_end].copy()).save(imgSavePath)
    
    imgSavePath = saveImgDIr + os.path.basename(addStrBeforeExt(imgPath,"04lowerR_rect"))
    x_start, x_end, y_start, y_end = lowerRightRect
    Image.fromarray(img[y_start : y_end, x_start : x_end].copy()).save(imgSavePath)
    
    return saveImgDIr


def getCropAndLapPositions(imgPath:str,cropSize:int,lapSize:int):

    step_x_col, step_y_row, resizeSetSlide = getXYtileInfo(imgPath, cropSize, lapSize)
    # np.array[height : width : ch]の構造　
    # imgNp_true_resized.shape = (h, w, ch)

    CropAndLapPositions = []

    print("Create Tiles ImageSize:{0}, LapSize:{1}".format(cropSize, lapSize))
    for y in range(step_y_row):
        # print("\n\n")
        for x in range(step_x_col):

            # print("\n","x, y =", x, y )
            if lapSize==0:
                x_start = x*cropSize
                x_end = x*cropSize + cropSize
                y_start = y*cropSize
                y_end = y*cropSize + cropSize
            else:
                x_start = x*lapSize
                x_end = x*lapSize + cropSize
                y_start = y*lapSize
                y_end = y*lapSize + cropSize
            
            cropPositon =(x_start, x_end, y_start, y_end)
            CropAndLapPositions.append(cropPositon)
    return CropAndLapPositions
    

def cropImgBylapsize(imgPath,cropSize,lapSize):

    img_pil = Image.open(imgPath)

    img = np.array(img_pil)
    # imgNp_true_resized[height : width : ch]の構造　 imgNp_true_resized.shape = (h, w, ch)

    # print("Resize Image SIze", img.shape)
    # print("Create Tiles ImageSize:{0}, LapSize:{1}".format(cropSize, lapSize))

    CropAndLapPositions = getCropAndLapPositions(imgPath,cropSize,lapSize)
    for cropPositon in CropAndLapPositions:
        x_start, x_end, y_start, y_end = cropPositon

        xyStr = "X" + str(x_start).zfill(5) + "to" + str(x_end).zfill(5) + "_" + "Y" + str(y_start).zfill(5) + "to" + str(y_end).zfill(5)
        saveImgDIr = os.path.dirname(imgPath) + "_Size" +str(cropSize).zfill(4) + "_lap" + str(lapSize).zfill(4) + "\\"
        os.makedirs(saveImgDIr, exist_ok=True)
        saveImgPath = saveImgDIr + os.path.basename(imgPath).split(".")[-2] + "_" + xyStr + "." + os.path.basename(imgPath).split(".")[-1].replace("JPG","jpg").replace("PNG","png")

        # org処理            
        if img_pil.mode == 'RGB':
            imgNp = img[x_start : x_end, y_start : y_end, :].copy()
            if imgNp.shape[0]==imgNp.shape[1]:
                crpoImg_org = Image.fromarray(imgNp)
                crpoImg_org.save(saveImgPath)
            
        # msk処理
        else:
            imgNp = img[x_start : x_end, y_start : y_end].copy()
            if imgNp.shape[0]==imgNp.shape[1]:
                cropImg_msk = Image.fromarray(imgNp)
                cropImg_msk.save(saveImgPath)
