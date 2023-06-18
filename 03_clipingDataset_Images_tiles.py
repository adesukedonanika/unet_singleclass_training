#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -r requirements_uavUnet.txt --user

import os,glob,shutil,json,sys
from tqdm import tqdm
from orgImageCrop import crop4CornersOrgImageBysize, cropImgBylapsize


def main():
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

        # print(len(orgPaths))
        # print(len(orgPaths)==len(mskPaths))
        # print(orgPaths[0],mskPaths[0])

        cropSize = int(sys.argv[1])
        if cropSize == 256:
            lapSize=0
        else:
            lapSize = (cropSize//2)
        # print(lapSize)

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


        for orgPath in tqdm(orgPaths):
            cropImgBylapsize(orgPath,cropSize,lapSize)
            mskPath = orgPath.replace("org","msk").replace(".JPG",".PNG")
            cropImgBylapsize(mskPath,cropSize,lapSize)



    print(orgDir)
    orgPaths = glob.glob(orgDir[:-1] + f"_Size{cropSize}_lap{lapSize}/*.jpg")
    mskPaths = glob.glob(mskDir[:-1] + f"_Size{cropSize}_lap{lapSize}/*.png")

    print(len(orgPaths),len(mskPaths))

# スクリプトが直接実行された場合にのみmain()関数を呼び出す
if __name__ == "__main__":
    main()





