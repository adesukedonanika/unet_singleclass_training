#!/usr/bin/env python
# coding: utf-8


import os,glob,sys,shutil,re,json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import albumentations as A
from tqdm import tqdm
import rasterio


# input
orgDir = sys.argv[1]#"01_LabelboxExport/Forest tsumura 2 50m P4Pv2/original"
mskDir = sys.argv[2]#"02_MargedMaskImages/Forest tsumura 2 50m P4Pv2/msk"


# Opening JSON file
with open('treeTypeValues.json', 'r') as openfile: 
    # Reading from json file
    treeTypeValues = json.load(openfile)
    
treeTypes = list(treeTypeValues.keys())

orgPaths = glob.glob(orgDir + "/*.JPG")

for treeType in treeTypes:
    mskPaths = glob.glob(mskDir + f"*{treeType}/*.PNG")

    # Output
    cropArgmentDir = f"{sys.argv[3]}_{treeType}"
    os.makedirs(cropArgmentDir,exist_ok=True)

    try:
        shutil.copytree(os.path.dirname(orgPaths[0]),os.path.join(cropArgmentDir,"org"))
        shutil.copytree(os.path.dirname(mskPaths[0]),os.path.join(cropArgmentDir,"msk"))
    except:
        print("already copy")
        
    orgPaths = glob.glob(os.path.join(cropArgmentDir,"org") + "/*.JPG")
    mskPaths = glob.glob(os.path.join(cropArgmentDir,"msk") + "/*.PNG")

    print(len(orgPaths))
    print(len(orgPaths)==len(mskPaths))
    print(orgPaths[0],mskPaths[0])

    cropSize = 1024
    lapSize = (cropSize//2)
    print(lapSize)

    for path in orgPaths:
        if os.path.basename(path) == "Thumbs.db":
            orgPaths.remove(path)

    for path in mskPaths:
        if os.path.basename(path) == "Thumbs.db":
            mskPaths.remove(path)


    def getXYtileInfo(predictImgPath,cropSize,lapSize):
        img_io = rasterio.open(predictImgPath)
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

    print("resize\t",getXYtileInfo(orgPaths[0],cropSize,lapSize))

    def addStrBeforeExt(filePath:str,addStr:str):
        filePath_list = filePath.split(".")
        return f"{filePath_list[-2]}_{addStr}.{filePath_list[-1]}"



    def crop4CornersOrgImageByLapsize(imgPath:str, cropSize:int, lapSize:int):

        img = Image.open(imgPath)
        img = np.array(img)

        width = img.shape[1]
        height = img.shape[0]

        step_x_col, step_y_row, resizeSetSlide = getXYtileInfo(imgPath, cropSize, lapSize)
        # resizeSetSlide = (y, x)
        # img.shape = (y, x, ch)

        x_amari = img.shape[1]-resizeSetSlide[1]
        y_amari = img.shape[0]-resizeSetSlide[0]

        saveImgDIr = os.path.dirname(imgPath) + f"_crop4Corner_{resizeSetSlide[1]}_{resizeSetSlide[0]}\\"
        os.makedirs(saveImgDIr, exist_ok=True)

        x_start = 0
        x_end = resizeSetSlide[1]
        y_start = 0
        y_end = resizeSetSlide[0]

        # print(x_start,x_end,y_start,y_end,x_amari,y_amari)

        # upperLeftRect =(x_start, x_end, y_start, y_end)
        imgSavePath = saveImgDIr + os.path.basename(addStrBeforeExt(imgPath,"01_upperL_rect"))
        Image.fromarray(img[y_start : y_end, x_start : x_end].copy()).save(imgSavePath)

        x_start = width - cropSize
        x_end = width
        y_start = 0
        y_end = resizeSetSlide[0]

        # upperRightRect =(x_start, x_end, y_start, y_end)
        imgSavePath = saveImgDIr + os.path.basename(addStrBeforeExt(imgPath,"02upperR_rect"))
        Image.fromarray(img[y_start : y_end, x_start : x_end].copy()).save(imgSavePath)

        x_start = 0
        x_end = resizeSetSlide[1]
        y_start = height - cropSize
        y_end = height

        lowerLeftRect =(x_start, x_end, y_start, y_end)
        imgSavePath = saveImgDIr + os.path.basename(addStrBeforeExt(imgPath,"03lowerL_rect"))
        Image.fromarray(img[y_start : y_end, x_start : x_end].copy()).save(imgSavePath)


        x_start = width - cropSize
        x_end = width
        y_start = height - cropSize
        y_end = height

        lowerRightRect =(x_start, x_end, y_start, y_end)
        imgSavePath = saveImgDIr + os.path.basename(addStrBeforeExt(imgPath,"04lowerR_rect"))
        Image.fromarray(img[y_start : y_end, x_start : x_end].copy()).save(imgSavePath)
        
        return saveImgDIr

    orgPaths = glob.glob(os.path.join(cropArgmentDir,"org") + "/*.JPG")
    mskPaths = glob.glob(os.path.join(cropArgmentDir,"msk") + "/*.PNG")

    for orgPath in tqdm(orgPaths):
        orgDir = crop4CornersOrgImageByLapsize(orgPath,cropSize,lapSize)
        mskPath = orgPath.replace("org","msk").replace(".JPG",".PNG")
        mskDir = crop4CornersOrgImageByLapsize(mskPath,cropSize,lapSize)
    orgPaths = glob.glob(orgDir + "/*_*Rect.JPG")
    mskPaths = glob.glob(mskDir + "/*_*Rect.PNG")

    print(len(orgPaths),len(mskPaths))


    def cropImg_lapsize(imgPath,cropSize,lapSize):
        # imgNp_true = imgNp_true.reshape([org_height, org_width, ch]).copy()

        predictImgSize = cropSize
        # print("imgNp_true.shape", imgNp_true.shape)

        img = Image.open(imgPath)

        step_x_col, step_y_row, resizeSetSlide = getXYtileInfo(imgPath, cropSize, lapSize)
        # print("col_step, row_step, resizeSetSlide ",step_x_col, step_y_row, resizeSetSlide)
        print(resizeSetSlide," <= orgSize",img.size[1],img.size[0])
        # img_resize = img.resize(resizeSetSlide)
        imgNp_resized = np.array(img)
        # imgNp_true_resized[height : width : ch]の構造　 imgNp_true_resized.shape = (h, w, ch)
        print("Resize Image SIze", imgNp_resized.shape)


        print("Create Tiles ImageSize:{0}, LapSize:{1}".format(cropSize, lapSize))
        for y in range(step_y_row):
            # print("\n\n")
            for x in range(step_x_col):

                # print("\n","x, y =", x, y )
                if lapSize==0:
                    x_start , x_end = x*predictImgSize, x*cropSize + cropSize
                    y_start , y_end = y*predictImgSize, y*cropSize + cropSize
                else:
                    x_start , x_end = x*lapSize , x*lapSize + cropSize
                    y_start , y_end = y*lapSize , y*lapSize + cropSize

                # imgNp_resized[height : width : ch]の構造　 imgNp_true.shape = (h, w, ch)
                # (predictImgSize, predictImgSize, 3) で取り出す　他サイズでは出ない

                xyStr = "X" + str(x_start).zfill(5) + "to" + str(x_end).zfill(5) + "_" + "Y" + str(y_start).zfill(5) + "to" + str(y_end).zfill(5)
                # print("Makedict imgNp", imgNp.shape, xyStr)
                saveImgDIr = os.path.dirname(imgPath) + "_Size" +str(cropSize).zfill(4) + "_lap" + str(lapSize) + "\\"
                os.makedirs(saveImgDIr, exist_ok=True)
                saveImgPath = saveImgDIr + os.path.basename(imgPath).split(".")[-2] + "_" + xyStr + "." + os.path.basename(imgPath).split(".")[-1].replace("JPG","jpg").replace("PNG","png")

                # org処理            
                if img.mode == 'RGB':
                    imgNp = imgNp_resized[x_start : x_end, y_start : y_end, :].copy()
                    if imgNp.shape[0]==imgNp.shape[1]:
                        crpoImg_org = Image.fromarray(imgNp)
                        crpoImg_org.save(saveImgPath)
                    
                # msk処理
                else:
                    imgNp = imgNp_resized[x_start : x_end, y_start : y_end].copy()
                    if imgNp.shape[0]==imgNp.shape[1]:
    #                 cropImg_msk = convIndexColor_fromNp(imgNp)
                        cropImg_msk = Image.fromarray(imgNp)
                        cropImg_msk.save(saveImgPath)

    for orgPath in tqdm(orgPaths):
        cropImg_lapsize(orgPath,cropSize,lapSize)
        mskPath = orgPath.replace("org","msk").replace(".JPG",".PNG")
        cropImg_lapsize(mskPath,cropSize,lapSize)



print(orgDir)
orgPaths = glob.glob(orgDir[:-1] + f"_Size{cropSize}_lap{lapSize}/*.jpg")
mskPaths = glob.glob(mskDir[:-1] + f"_Size{cropSize}_lap{lapSize}/*.png")

print(len(orgPaths),len(mskPaths))

