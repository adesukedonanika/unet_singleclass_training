#!/usr/bin/env python
# coding: utf-8


from PIL import Image
import os,glob, re, json



# input
orgDir = "01_LabelboxExport/Forest tsumura 2 50m P4Pv2/original"
mskDir = "01_LabelboxExport/Forest tsumura 2 50m P4Pv2/masked"

orgPaths = glob.glob(orgDir + "/*.JPG")
mskPaths = glob.glob(mskDir + "_seg_*/*.png")

# Output
mskImagesDir = "02_MargedMaskImages/Forest tsumura 2 50m P4Pv2/msk"



def getTreeType_fromPath(Path:str):
    pattern = r'masked_seg_(.*)\\DJI_.*'
    match = re.search(pattern, Path)
    if match:
        treeType = match.group(1)
    return treeType



import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



# 定義
# treeTypeValues = {
#     "cedar":1,
#     "cypress":2
# }
# 書き込み
# with open("treeTypeValues.json","w") as file:
#     json.dump(treeTypeValues, file)


# Opening JSON file
with open('treeTypeValues.json', 'r') as openfile:
 
    # Reading from json file
    treeTypeValues = json.load(openfile)
    
treeTypeValues
treeTypes = list(treeTypeValues.keys())


# 単一樹種分類用Mask画像作成
def makeSingleMaskSets(orgPaths:list,mskPaths:list,treeTypes:dict):

    for orgPath in orgPaths:
        orgImgName = os.path.basename(orgPath).split(".")[0]

        for treeType in treeTypes:

            org = np.array(Image.open(orgPath))
            msk_base = np.zeros(org.shape[:2],dtype=np.uint8)

            # オリジナル画像名と、樹種名が含まれているmskパス群を抽出
            for mskPath in filter(lambda x: (orgImgName in x and treeType in x), mskPaths):
                # print(mskPath)

                mergedMasksDir = f"{mskImagesDir}_merged_{treeType}"
                os.makedirs(mergedMasksDir, exist_ok=True)

                msk = np.array(Image.open(mskPath).convert("L"))

                # # 値の置換 255->1　単一分類では行わない
                # msk = np.where(msk == 255, treeTypeValues[treeType], 0).astype(np.uint8)
                msk_base = msk_base + msk

            cv2.imwrite(os.path.join(mergedMasksDir,orgImgName+".PNG"), msk_base)

            # 2つの画像を横に並べる
            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(org)
            # axs[1].imshow(msk_base, cmap='gray')
            # # 表示する
            # # plt.show()
            del org,msk_base,msk


makeSingleMaskSets(orgPaths,mskPaths,treeTypes)



# マルチ樹種の分類用マスク画像作成

def makeMultiMaskSets(orgPaths:list,mskPaths:list,treeTypes:dict):

    for orgPath in orgPaths:
        orgImgName = os.path.basename(orgPath).split(".")[0]

        org = np.array(Image.open(orgPath))
        msk_base = np.zeros(org.shape[:2],dtype=np.uint8)
        
        for treeType in treeTypes:
            # オリジナル画像名と、樹種名が含まれているmskパス群を抽出
            for mskPath in filter(lambda x: (orgImgName in x and treeType in x), mskPaths):
                # print(mskPath)

                mergedMasksDir = f"{mskImagesDir}_merged_MultiChannel_{'+'.join(treeTypes)}"
                os.makedirs(mergedMasksDir, exist_ok=True)

                msk = np.array(Image.open(mskPath).convert("L"))

                msk = np.where(msk == 255, int(treeTypeValues[treeType])*(255//len(treeTypes)), 0).astype(np.uint8)
                msk_base = msk_base + msk
    
        
        mskPath_merged = f"{mergedMasksDir}/{orgImgName}.PNG"
        print(mskPath_merged)
        cv2.imwrite(mskPath_merged, msk_base)

        # # 2つの画像を横に並べる
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(org)
        # axs[1].imshow(msk_base, cmap='gray')
        # # 表示する
        # # plt.show()
        del org,msk
        del msk_base


makeMultiMaskSets(orgPaths,mskPaths,treeTypes)
