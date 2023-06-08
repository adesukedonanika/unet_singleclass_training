import os, glob, shutil




def get_mskPath(orgPath):
    mskPath = orgPath.replace("org","msk").replace(".jpg",".png")
    if os.path.exists(mskPath):
        return mskPath
    else:
        return ""


def copyLocaliImages(orgDir, copyDir):
    orgDirName = os.path.basename(orgDir)
    
    orgPaths = glob.glob(orgDir + "/*.*")
    mskPath = get_mskPath(orgPaths[0])
    mskDirName = os.path.basename(os.path.dirname(mskPath))

    os.makedirs(os.path.join(copyDir,orgDirName), exist_ok=True)
    os.makedirs(os.path.join(copyDir,mskDirName), exist_ok=True)

    for orgPath in orgPaths:
        mskPath = get_mskPath(orgPath)
        new_orgPath = os.path.join(copyDir,orgDirName,os.path.basename(orgPath))
        if not os.path.exists(new_orgPath):
            shutil.copy(orgPath,new_orgPath)
        
        new_mskPath = os.path.join(copyDir,mskDirName,os.path.basename(mskPath))
        if not os.path.exists(new_mskPath):
            shutil.copy(mskPath,new_mskPath)
    return os.path.join(copyDir,orgDirName)




def addSavePath(Path,addKeyword):
    Path_added = os.path.dirname(Path) + "/" + os.path.basename(Path).split(".")[0] + "_" + str(addKeyword) + "." + os.path.basename(Path).split(".")[1]
    return Path_added

