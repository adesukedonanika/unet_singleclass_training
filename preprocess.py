

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint


# import albumentations as A

# Image.open
#  numpy.asarray numpy.ndarrayに変換する。形は(h, w, c)。

# cv2.imread
# numpy.ndarray (h, w, c)  BGR dtype=uint8 [0, 255]

# matplotlib.image.imread
# (h, w, c) RGB、dtype=float32、[0.0, 1.0]。

# rasterio.open(path).read()
# (ch, h, w)


def simpleNorm(array):
    max = np.max(array)
    min = np.min(array)
    return ((array - min) / (max - min))

def npImgResize(nparray, size):
    img = Image.fromarray(nparray)
    img_resize = img.resize([size, size])
    nparray_resize = np.array(img_resize)
    return nparray_resize

def npImgResizeCR(nparray, column, row):
    img = Image.fromarray(nparray)# print(pil_img.size) # (w, h)
    img_resize = img.resize([column, row])# (width, height)
    nparray_resize = np.array(img_resize)
    return nparray_resize




import rasterio

def getXYtileInfo(predictImgPath,cropSize,lapSize):
    
    img_io = rasterio.open(predictImgPath)
    org_height, org_width, ch = img_io.height, img_io.width, img_io.count
    img_io.close()

    predictImgSize = cropSize


    if lapSize == 0:
        step_x_col = org_width//predictImgSize
        step_y_row = org_height//predictImgSize
    else:
        step_x_col = (org_width - (predictImgSize - lapSize))// lapSize
        step_y_row = (org_height - (predictImgSize - lapSize))// lapSize
    # step_x_col = org_width - (org_width % lapSize))
    # step_y_row = org_height - (org_height % lapSize)


    if lapSize==0:
        resizeSetSlide = ((org_height // cropSize) * cropSize, org_width // cropSize * cropSize)
    else:
        resizeSetSlide = (org_height // lapSize * lapSize, org_width // lapSize * lapSize)

    return step_x_col, step_y_row, resizeSetSlide





def resizeNpBaseImgSize(imgNp,BaseImgsize):
    if len(imgNp.shape) == 3:
        imgNp_resize = np.zeros(shape=(BaseImgsize, BaseImgsize, imgNp.shape[-1]))    
        imgNp_resize[:imgNp.shape[0],:imgNp.shape[1],:] = imgNp
        return imgNp_resize
    else:
        imgNp_resize = np.zeros(shape=(BaseImgsize,BaseImgsize))
        imgNp_resize[:imgNp.shape[0],:imgNp.shape[1]] = imgNp
        return imgNp_resize



    
# helper function for data visualization 
# 正規化 最大値 255と 0　は使用しない　max:98% min: 2%
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)# 0以下は　すべて0 1以上はすべて1へ丸める
    return x

def norm(array):
    Max = np.max(array)
    Min = np.min(array)

    # (array - Min)/(Max - Min)
    A = (array - Min)
    B = (Max - Min)

    if B==0:
        array_norm = array
    else:
        array_norm = A/B

    return array_norm

def band2Norm(array_3bandimg):
    img = array_3bandimg
    b1 = img[:,:,0]
    b2 = img[:,:,1]
    b3 = img[:,:,2]

    b1 = norm(b1)
    b2 = norm(b2)
    b3 = norm(b3)
    
    bit = 1
    
    imgNorm = np.zeros(img.shape)
    imgNorm[:,:,0] = b1 * (2**bit - 1)
    imgNorm[:,:,1] = b2 * (2**bit - 1)
    imgNorm[:,:,2] = b3 * (2**bit - 1)

    return imgNorm


