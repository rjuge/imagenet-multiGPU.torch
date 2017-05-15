import os
import numpy as np
import scipy as sp
from scipy import misc
import lutorpy as lua
import argparse
import itertools
from PIL import Image
from skimage import img_as_float
from skimage import transform
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pylab import *
import json

np.set_printoptions(precision=2)

require('torch')
require('nn')
require('cunn')
require('cudnn')
require('image')

parser = argparse.ArgumentParser(description = 'Horus object recognition testbench')
parser.add_argument('-m', '--model', help = 'Model to test', required = True)
parser.add_argument('-t', '--testbench', help = 'raw or horus', required = True)
parser.add_argument('-jc2l', '--hv02_c2l', 
                    help = 'horus json mapping file to test on subset of classes', 
                    required = False)
parser.add_argument('-jl2c', '--hv02_l2c', 
                    help = 'horus json mapping file hv02_ to names', 
                    required = False)
parser.add_argument('-jext', '--js_ext', 
                    help = 'json external mapping file', 
                    required = False)
parser.add_argument('-l', '--list', help = 'text file, list subset of classes to test on', 
                    required = False)
parser.add_argument('-M', '--map', 
                    help = 'mapping file .t7 from output number to label', 
                    required = False)
args = parser.parse_args()

if(args.testbench == 'horus'):
    path = "/home/remi/Deep_Learning/objRecTestbench/dataset/tiny_testbench_corr/"
elif(args.testbench == 'raw'):
    path = "/home/remi/Deep_Learning/objRecTestbench/dataset/tiny_testbench_raw/"

if(args.hv02_c2l and args.list):
    jc2l = json.load(open(args.hv02_c2l,'r'))
    f = open(args.list, 'r')
    listNames = f.readlines()
    listDirs = [jc2l[i[:-1]] for i in listNames]
    listForTest = [os.path.join(path,d) for d in listDirs]
elif(args.hv02_c2l and not args.list):
    raise ValueError('List of classes is needed')
elif(args.list and not args.hv02_c2l):
    raise ValueError('Json mapping file is needed')
else:
    listDirs = [d for d in os.listdir(path) if not os.path.isfile(os.path.join(path, d))]
    listForTest = [os.path.join(path,d) for d in listDirs]

if(args.hv02_l2c and args.js_ext and args.map):
    jl2c = json.load(open(args.hv02_l2c,'r'))
    jext = json.load(open(args.js_ext,'r'))
    map_lua = torch.load(args.map)
elif(args.hv02_l2c and not (args.js_ext or args.map)):
    raise ValueError('External json and mapping file are needed')
elif(args.js_ext and not (args.hv02_l2c or args.map)):
    raise ValueError('Horus labels to classes and mapping file are needed')

TOP1 = 0.0
TOP3 = 0.0
TOP5 = 0.0

N_CLASSES = len(listForTest)
print 'Number of classes to test on : ', N_CLASSES

file = open('./testbench_results/py_testbench_'+args.model[:-3]+'.txt', 'w')

file.write('Testbench results for model :'+args.model+'\n')
file.write('\n')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def imgPreProcess(im):
    im = transform.resize(im, (3,224,224), preserve_range=True)
    for i in range(0,3):
        if(model.img_mean):
            im[i] -= model.img_mean[i]
        if(model.img_std):
            im[i] /= model.img_std[i]
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

TOP1_CLASSES = np.ndarray(shape=(400,))
TOP3_CLASSES = np.ndarray(shape=(400,))
TOP5_CLASSES = np.ndarray(shape=(400,))
nTest = 0

predictions = []
truth = []
classes = []
for subdir, _, files in os.walk(path):
    if(subdir == path or subdir not in listForTest):
        continue

    TOP1_CLASS = 0.0
    TOP3_CLASS = 0.0
    TOP5_CLASS = 0.0
    n_class = 0
    classes.append(int(subdir[-3:]))
    for f in files:
        nTest += 1
        n_class += 1
        
        img = image.load(os.path.join(subdir, f),3)
        img = img.asNumpyArray()
        img_proc = imgPreProcess(img)
        im_t = torch.fromNumpyArray(img_proc)
        im_cuda = im_t._cuda()
        outputs = model.convnet._forward(im_cuda)
        preds = outputs._float()
        preds = preds.asNumpyArray().transpose()
        pred = preds.tolist()
        
        pred_sorted = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        pred_sorted = [i+1 for i in pred_sorted] #python offset
        predictions.append(pred_sorted[0])
        if(args.js_ext):
            label = subdir[-3:]
            truth.append(int(label))
            l = jext[jl2c['hv02_'+str(label)]]
            pred_sorted = [map_lua[i] for i in pred_sorted]
            if(pred_sorted[0] == l):
                TOP1 += 1 
                TOP1_CLASS += 1
            if(isTop3(pred_sorted, l) == True): 
                TOP3 += 1
                TOP3_CLASS += 1
            if(isTop5(pred_sorted, l) == True): 
                TOP5 += 1
                TOP5_CLASS += 1
        else:
            label = int(subdir[-3:])
            truth.append(label)
            if(pred_sorted[0] == label):
                TOP1 += 1 
                TOP1_CLASS += 1
            if(isTop3(pred_sorted, label) == True): 
                TOP3 += 1
                TOP3_CLASS += 1
            if(isTop5(pred_sorted, label) == True): 
                TOP5 += 1
                TOP5_CLASS += 1

    TOP1_CLASS *= 100 / n_class
    TOP1_CLASSES[int(label)-1] = TOP1_CLASS
    TOP3_CLASS *= 100 / n_class
    TOP3_CLASSES[int(label)-1] = TOP3_CLASS
    TOP5_CLASS *= 100 / n_class
    TOP5_CLASSES[int(label)-1] = TOP5_CLASS


TOP1 = TOP1 * 100 / nTest
TOP3 = TOP3 * 100 / nTest
TOP5 = TOP5 * 100 / nTest

file.write('Global Top1: '+str(TOP1)+'\n')
file.write('Global Top3: '+str(TOP3)+'\n')
file.write('Global Top5: '+str(TOP5)+'\n')
file.write('\n')
file.write('Scores per class:'+'\n')
file.write('hv02_ Top1 Top3 Top5'+'\n')
file.write('---------------------'+'\n')
for i in sorted(range(len(TOP1_CLASSES)), key=lambda k: TOP1_CLASSES[k], reverse=True):
    file.write(str(i+1)+' '+str(TOP1_CLASSES[i])+' '
               +str(TOP3_CLASSES[i])+' '+str(TOP5_CLASSES[i])+' '+'\n')

cm = confusion_matrix(truth, predictions, labels=classes)

f = plt.figure(figsize=(40,40))
plot_confusion_matrix(cm, classes, normalize=False,
                      title='Confusion matrix')

plt.savefig('CM.png', format='png')

TOP1_sorted = sorted(TOP1_CLASSES, reverse=True)
TOP1_id = sorted(range(len(TOP1_CLASSES)), key=lambda k: TOP1_CLASSES[k], reverse=True)
pos = np.arange(400)+.5
f, ax = plt.subplots()
ax.barh(pos, TOP1_sorted, align='center')
ax.set_yticks(pos)
ax.set_yticklabels(TOP1_id)
ax.invert_yaxis()
ax.set_xlabel('Top1')
ax.set_title('Top1 chart')
ax.set_ylim(bottom=0, top=500)
#grid(True)
plt.savefig('hist.png', format='png')
print('==> Test ran on '+str(nTest)+' samples')

print 'Top1 ',TOP1
print('Top3 '+str(TOP3))
print('Top5 '+str(TOP5))

file.close()
