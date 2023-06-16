#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -r requirements_uavUnet.txt --user

import os,glob,sys,shutil,re,json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from orgImageCrop import crop4CornersOrgImageBysize, getXYtileInfo


# In[24]:


# input
orgDir = "01_LabelboxExport/Forest tsumura 2 50m P4Pv2/original"
mskDir = "02_MargedMaskImages/Forest tsumura 2 50m P4Pv2/msk"

# Opening JSON file
with open('treeTypeValues.json', 'r') as openfile:
 
    # Reading from json file
    treeTypeValues = json.load(openfile)
    
treeTypeValues
treeTypes = list(treeTypeValues.keys())


orgPaths = glob.glob(orgDir + "/*.JPG")

for treeType in treeTypes:
    mskPaths = glob.glob(mskDir + f"*{treeType}/*.PNG")


    # def getTreeType_fromPath(Path:str):
    #     pattern = r'msk_merged_(.*)\\DJI_.*'
    #     match = re.search(pattern, Path)
    #     if match:
    #         treeType = match.group(1)
    #         return treeType
    #     else:
    #         return Path

    # treeType = getTreeType_fromPath(mskPaths[0])


    # Output
    cropArgmentDir = f"03_datasetforModel/Forest tsumura 2 50m P4Pv2_{treeType}"
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




    orgPaths = glob.glob(os.path.join(cropArgmentDir,"org") + "/*.JPG")
    mskPaths = glob.glob(os.path.join(cropArgmentDir,"msk") + "/*.PNG")

    for orgPath in tqdm(orgPaths):
        orgDir = crop4CornersOrgImageBysize(orgPath,cropSize)
        mskPath = orgPath.replace("org","msk").replace(".JPG",".PNG")
        mskDir = crop4CornersOrgImageBysize(mskPath,cropSize)
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


# In[25]:


print(orgDir)
orgPaths = glob.glob(orgDir[:-1] + f"_Size{cropSize}_lap{lapSize}/*.jpg")
mskPaths = glob.glob(mskDir[:-1] + f"_Size{cropSize}_lap{lapSize}/*.png")

print(len(orgPaths),len(mskPaths))


# In[ ]:




