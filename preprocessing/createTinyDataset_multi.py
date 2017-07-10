# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:12:27 2016
@author: kraken
"""
import os
import time
from PIL import Image
from multiprocessing import Process
from multiprocessing import Pool



def resizeandsave(fullname, outfilename, f):
    img = Image.open(fullname)
    width, height = img.size
    ratioX = maxWidth /width
    ratioY = maxHeight/height
    ratio = max(ratioX, ratioY)
    newWidth = width * ratio
    newHeight = height * ratio          
    img = img.resize((int(newWidth),int(newHeight)), Image.BILINEAR)
    img.save(outfilename)
    if(f % 1000 == 0):
        print("Done: " + str(f))
    
    return
        

poolSize = 6
srcDirTrain = "/home/remi/Deep_Learning/objRecTestbench/dataset/testbench_corr/" 
destDirTrain = "/home/remi/Deep_Learning/objRecTestbench/dataset/tiny_testbench_corr/"

#srcDirVal = "/raid/Datasets/ImageNet/val" 
#destDirVal = "/raid/Datasets/TinyImageNet/val"

maxHeight = 256.0
maxWidth = 256.0

d = 0
f = 0

p = Pool(processes=poolSize)
#os.makedirs(destDirTrain)

for subdir, dirs, files in os.walk(srcDirTrain): 
    dirName = ""
    for dir in dirs:
        d = d+1
        dirName = dir
        #print(dir)
        os.makedirs(os.path.join(destDirTrain, dir))
        for imgdir, imgdirs, imgfiles in os.walk(os.path.join(srcDirTrain, dir)):
            for file in imgfiles:
                fullname = os.path.join(imgdir, file)
                outfilename = os.path.join(destDirTrain, dir, file )
                p.apply_async(resizeandsave, (fullname,outfilename,f))
                f = f + 1
                if(f%128==0):
                    time.sleep(0.100)
                if (f%1000==0):
                    print("Started:" + str(f))
p.close()
p.join()
