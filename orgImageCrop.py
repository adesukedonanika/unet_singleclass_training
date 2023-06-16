
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
    
    return upperLeftRect, upperRightRect, lowerLeftRect, lowerRightRect

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