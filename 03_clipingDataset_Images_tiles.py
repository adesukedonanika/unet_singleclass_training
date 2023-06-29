#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -r requirements_uavUnet.txt --user

import os,glob,shutil,json,sys
from tqdm import tqdm
from orgImageCrop import crop4CornersOrgImageBysize, cropImgBylapsize
from fpathutils import get_mskPath

def main():
    # input

    # Opening JSON file
    with open('treeTypeValues.json', 'r') as openfile:
    
        # Reading from json file
        treeTypeValues = json.load(openfile)
        
    treeTypes = list(treeTypeValues.keys())



    for treeType in treeTypes:
        orgDir = f"C:/datas/uav_cnn_{treeType}/org"
        mskDir = f"C:/datas/uav_cnn_{treeType}/msk"

        orgPaths = glob.glob(orgDir + "/*.JPG")
        mskPaths = glob.glob(mskDir+"/*.PNG")


        # Output
        cropArgmentDir = orgDir#f"03_datasetforModel/Forest tsumura 2 50m P4Pv2_{treeType}"
        os.makedirs(cropArgmentDir,exist_ok=True)

        try:
            shutil.copytree(os.path.dirname(orgPaths[0]),os.path.join(cropArgmentDir,"org"))
            shutil.copytree(os.path.dirname(mskPaths[0]),get_mskPath(cropArgmentDir))
        except:
            print("already copy")
            
        orgPaths = glob.glob(os.path.join(cropArgmentDir,"org") + "/*.JPG")
        mskPaths = glob.glob(get_mskPath(cropArgmentDir) + "/*.PNG")

        print(len(orgPaths))
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


        orgPaths = glob.glob(cropArgmentDir + "/*.JPG")
        mskPaths = glob.glob(get_mskPath(cropArgmentDir) + "/*.PNG")

        for orgPath in tqdm(orgPaths):
            orgDir = crop4CornersOrgImageBysize(orgPath, cropSize)
            mskPath = get_mskPath(orgPath)
            mskDir = crop4CornersOrgImageBysize(mskPath, cropSize)


        orgPaths = glob.glob(orgDir + "/*_*rect.JPG")
        mskPaths = glob.glob(mskDir + "/*_*rect.PNG")

        print(len(orgPaths),len(mskPaths))


        for orgPath in tqdm(orgPaths):
            cropImgBylapsize(orgPath,cropSize,lapSize)
            mskPath = get_mskPath(orgPath)
            cropImgBylapsize(mskPath,cropSize,lapSize)



    print(orgDir)
    orgPaths = glob.glob(orgDir[:-1] + f"_Size{cropSize}_lap{lapSize}/*.jpg")
    mskPaths = glob.glob(mskDir[:-1] + f"_Size{cropSize}_lap{lapSize}/*.png")

    print(len(orgPaths),len(mskPaths))

# スクリプトが直接実行された場合にのみmain()関数を呼び出す
if __name__ == "__main__":
    main()





