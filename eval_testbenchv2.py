import os
import numpy as np
import scipy as sp
from scipy import misc
import lutorpy as lua
import argparse
from PIL import Image
from skimage import img_as_float

require('nn')
require('cunn')
require('cudnn')
require('image')

parser = argparse.ArgumentParser(description = 'Horus object recognition testbench')
parser.add_argument('-m', '--model', help = 'Model to test', required = True)
parser.add_argument('-t', '--testbench', help = 'raw or horus', required = True)
args = parser.parse_args()

if(args.testbench == 'horus'):
    path = "/home/remi/Deep_Learning/objRecTestbench/dataset/tiny_test/"
else:
    path = "/home/remi/Deep_Learning/objRecTestbench/dataset/tiny_hv02_testbench/"

top1 = 0
top3 = 0
top5 = 0

file = open('./testbench_results/testbench_'+args.model[:-3]+'.txt', 'w')

file.write('Testbench results for model :'+args.model+'\n')
file.write('\n')


def imgPreProcess(im):
    im = misc.imresize(im, (224,224))
    im = img_as_float(im.transpose(2,0,1))
    for i in range(0,3):
        if(model.img_mean):
            im[i] -= model.img_mean[i]
        if(model.img_std):
            im[i] /= model.img_std[i]
    if(model.tensor_dim == 4):
        np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
    return im

def isTop3(X_hat, X):
    b = False
    for i in range(0,3):
        if(X_hat[i] == X):
            b = True
    return b

def isTop5(X_hat, X):
    b = False
    for i in range(0,5):
        if(X_hat[i] == X):
            b = True
    return b

model = torch.load(args.model)
print '==> Loading Model'
model.convnet._add(nn.SoftMax())
model.convnet._evaluate(model.convnet)
model.convnet = model.convnet._cuda()

print model.convnet

top1_classes = []
top3_classes = []
top5_classes = []
nTest = 0

for subdir, _, files in os.walk(path):
    
    top1_class = 0
    top3_class = 0
    n_class = 0
    
    for f in files:
        nTest += 1
        n_class += 1

        img = misc.imread(os.path.join(subdir, f))
        img_proc = imgPreProcess(img)
        im_t = torch.fromNumpyArray(img_proc)
        im_cuda = im_t._cuda()
        outputs = model.convnet._forward(im_cuda)
        preds = outputs._float()
        preds = preds.asNumpyArray()
        pred_sorted = np.sort(preds)
        
        label = subdir
        print label

file.close()
