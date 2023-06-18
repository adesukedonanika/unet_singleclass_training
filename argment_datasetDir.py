from segment_model_training import rotateSave, flipMirrorSave
import os, glob
from tqdm import tqdm

from fpathutils import get_mskPath

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


orgDir = ".\\03_datasetforModel\\Forest tsumura 2 50m P4Pv2_cedar\\org_crop4Corner_5120_3584_Size0512_lap0256"

argmentDataset(orgDir=orgDir,rotate=True,flipMirror=True)
