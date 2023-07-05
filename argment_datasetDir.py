import os, glob
from tqdm import tqdm
from PIL import Image,ImageOps
from fpathutils import get_mskPath

from fpathutils import addSavePath, get_mskPath

def flipMirrorSave(imgPath, SaveDir):
    os.makedirs(SaveDir,exist_ok=True)

    imgPil = Image.open(imgPath)
    imgPil_flip = ImageOps.flip(imgPil)
    imgPil_mirror = ImageOps.mirror(imgPil)

    SavePath = os.path.join(SaveDir, os.path.basename(imgPath))

    SavePath_flip = addSavePath(SavePath,"_flip")
    SavePath_mirror = addSavePath(SavePath,"_mirror")

    os.makedirs(os.path.dirname(SavePath_flip), exist_ok=True)
    os.makedirs(os.path.dirname(SavePath_mirror), exist_ok=True)

    if not os.path.exists(SavePath):
        imgPil.save(SavePath)
    if not os.path.exists(SavePath_flip):
        imgPil_flip.save(SavePath_flip)
    if not os.path.exists(SavePath_mirror):
        imgPil_mirror.save(SavePath_mirror)
#     return imgPil_filp, imgPil_mirror

def rotateSave(imgPath, SaveDir, angleMax:int, angleInterval:90):
    os.makedirs(SaveDir,exist_ok=True)

    imgPil = Image.open(imgPath)
    #　画像を回転角度の設定値　90, 180, 270
    angleMax = angleMax + 90
    # angleInterval = 90

    SavePath = os.path.join(SaveDir, os.path.basename(imgPath))

    for angle in range(0,angleMax,angleInterval):
        angleStr = str(angle).zfill(3)
        
        #画像の保存名を定義。
        SavePath_rotate = addSavePath(SavePath, f"rotate{angleStr}")
        os.makedirs(os.path.dirname(SavePath_rotate), exist_ok=True)
        if not os.path.exists(SavePath_rotate):
            imgPil_rotate = imgPil.rotate(angle)
            imgPil_rotate.save(SavePath_rotate)
            print(SavePath_rotate,imgPil_rotate.mode)


def argmentDataset(orgDir, rotate=True, flipMirror=True):
    #Rotate処理
    dirPath = os.path.join(orgDir,"*.jpg")
    print(dirPath)
    imgPaths = glob.glob(dirPath)
    for imgPath in tqdm(imgPaths):
        if rotate:
            print("image Rotate")
            rotateSave(imgPath, SaveDir=os.path.dirname(imgPath)+"_rotate", angleMax=270, angleInterval=90)
            mskPath = get_mskPath(imgPath)
            rotateSave(mskPath, SaveDir=os.path.dirname(mskPath)+"_rotate", angleMax=270, angleInterval=90)
    
    if not (rotate==False and flipMirror==True):
        #フリップ、ミラー処理
        imgPaths = glob.glob(os.path.join(os.path.dirname(imgPaths[0])+"_rotate","*.jpg"))
    
    for imgPath in tqdm(imgPaths):
        if flipMirror:
            print("image flipMirror")
            flipMirrorSave(imgPath, SaveDir=os.path.dirname(imgPath)+"_flipMirror")
            mskPath = get_mskPath(imgPath)
            flipMirrorSave(mskPath, SaveDir=os.path.dirname(mskPath)+"_flipMirror")

    print(os.path.join(os.path.dirname(imgPaths[0])+"_flipMirror","*.jpg"))
    imgPaths = glob.glob(os.path.join(os.path.dirname(imgPaths[0])+"_flipMirror","*.jpg"))
    orgDir = os.path.dirname(imgPaths[0])
    return orgDir


orgDir = "C:\\datas\\uav_cnn_cedar\\org_crop4Corner_5120_3584_Size0512_lap0256"
argmentDataset(orgDir=orgDir,rotate=True,flipMirror=True)

orgDir = "C:\\datas\\uav_cnn_cedar\\org_crop4Corner_5376_3584_Size0256_lap0000"
argmentDataset(orgDir=orgDir,rotate=True,flipMirror=True)
