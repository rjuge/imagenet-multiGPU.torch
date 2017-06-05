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
import operator

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
parser.add_argument('-jh2a', '--js_h2a', 
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

jc2l = json.load(open(args.hv02_c2l,'r'))
jc2l['other'] = 'hv02_000'
jl2c = {v: k for k,v in jc2l.iteritems()}    

if(args.list):
    f = open(args.list, 'r')
    listNames = f.readlines()
    listDirs = [jc2l[i[:-1]] for i in listNames]
    listForTest = [os.path.join(path,d) for d in listDirs]
    FONT = 12
else:
    listDirs = [d for d in os.listdir(path) if not os.path.isfile(os.path.join(path, d))]
    listForTest = [os.path.join(path,d) for d in listDirs]
    FONT = 8

if(args.js_h2a and args.map):
    jh2a = json.load(open(args.js_h2a,'r'))
    ja2h = {v: k for k,v in jh2a.iteritems()} 
    map_lua = torch.load(args.map)
elif(args.js_h2a and not args.map):
    raise ValueError('External mapping file is needed')
elif(not args.js_h2a and args.map):
    raise ValueError('Mapping file between models classes is needed')

folder = './testbench_results/'+args.model[:-3]+'/'+str(len(listForTest))+'classes/'

if not(os.path.exists(folder)):
    os.makedirs(folder)

TOP1 = 0.0
TOP3 = 0.0
TOP5 = 0.0

N_CLASSES = len(listForTest)
print 'Number of classes to test on : ', N_CLASSES

file = open(folder+'testbench.txt', 'w')
file.write('Testbench results for model :'+args.model+'\n')
file.write('\n')

def imgPreProcess(im):
    im = transform.resize(im, (3,224,224), preserve_range=True)
    for i in range(0,3):
        if(model.img_mean):
            im[i] -= model.img_mean[i]
        if(model.img_std):
            im[i] /= model.img_std[i]
    im = np.expand_dims(im, axis=0)
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

TOP1_CLASSES = {}
TOP3_CLASSES = {}
TOP5_CLASSES = {}
nTest = 0

predictions = []
truth = []
classes = []

correct=[]
incorrect=[]

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
        #print 'file : ', os.path.join(subdir, f)
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
        if(args.js_h2a):
            label = subdir[-3:]
            truth.append(int(label))
            l = jh2a[jl2c['hv02_'+str(label)]]
            pred_lua = [map_lua[i-1] for i in pred_sorted]
            a = pred_lua[0]
            try:
                h = ja2h[a]
            except:
                predictions.append(0)
            else:
                hl = jc2l[h][-3:]
                predictions.append(int(hl))
            if(pred_lua[0] == l):
                TOP1 += 1 
                TOP1_CLASS += 1
                correct.append(pred[pred_sorted[0]-1])
	    else:
            	incorrect.append(pred[pred_sorted[0]-1])
            if(isTop3(pred_lua, l) == True): 
                TOP3 += 1
                TOP3_CLASS += 1
            if(isTop5(pred_lua, l) == True): 
                TOP5 += 1
                TOP5_CLASS += 1
        else:
            predictions.append(pred_sorted[0])
            label = int(subdir[-3:])
            truth.append(label)
            if(pred_sorted[0] == label):
                TOP1 += 1 
                TOP1_CLASS += 1
                correct.append(pred[pred_sorted[0]-1])
            else:
                incorrect.append(pred[pred_sorted[0]-1])
            if(isTop3(pred_sorted, label) == True): 
                TOP3 += 1
                TOP3_CLASS += 1
            if(isTop5(pred_sorted, label) == True): 
                TOP5 += 1
                TOP5_CLASS += 1

    TOP1_CLASS *= 100 / n_class
    TOP1_CLASSES[int(label)] = TOP1_CLASS
    TOP3_CLASS *= 100 / n_class
    TOP3_CLASSES[int(label)] = TOP3_CLASS
    TOP5_CLASS *= 100 / n_class
    TOP5_CLASSES[int(label)] = TOP5_CLASS


TOP1 = TOP1 * 100 / nTest
TOP3 = TOP3 * 100 / nTest
TOP5 = TOP5 * 100 / nTest

#confusion matrix
classes.append(0)
cm = confusion_matrix(truth, predictions, labels=classes)
f, ax = plt.subplots(figsize=(48, 48))
ax.set_title('Confusion Matrix', fontsize=45)
res = ax.imshow(np.array(cm), cmap=plt.cm.jet, interpolation='nearest')
width, height = cm.shape
for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=FONT)

names = [jl2c['hv02_'+(str(i).zfill(3))] for i in classes]
plt.xticks(range(width), names, rotation=90, fontsize=FONT)
plt.yticks(range(height), names, rotation=0, fontsize=FONT)
plt.savefig(folder+'confusion_matrix.png', format='png')

#Misclassified histogram
np.fill_diagonal(cm, 0)
val = cm[np.nonzero(cm)]
fig = plt.figure()
plt.hist(val, 50, edgecolor='black')
plt.title('Median misclassified')
plt.xlabel('Number of misclassified images')
plt.ylabel('Count')
plt.grid(True)
uni = np.unique(val)
med = np.median(uni)
plt.axvline(med, color='r', linestyle='dashed', linewidth=2)
plt.savefig(folder+'median.png', format='png')
f = open(folder+'misclassified.txt', 'w')
f.write('Misclassified pairs for model :'+args.model+'\n')
f.write('\n')
f.write('Nb   Ground truth   Prediction'+'\n')
f.write('------------------------------'+'\n')
for i in range(int(np.floor(med)), int(np.amax(uni))+1):
    idx = np.where(cm==i)
    for x,y in zip(idx[0], idx[1]):
        f.write(str(i)+'  '+names[x]+'  '+names[y]+'\n')
f.close()

#Write Top and score per class
file.write('Global Top1: '+str(TOP1)+'\n')
file.write('Global Top3: '+str(TOP3)+'\n')
file.write('Global Top5: '+str(TOP5)+'\n')
file.write('\n')
#file.write('TP:',tp,'\n')
#file.write('TN:',tn,'\n')
#file.write('FP:',fp,'\n')
#file.write('FN:',fn,'\n')
#file.write('\n')
file.write('Scores per class:'+'\n')
file.write(' Class  Top1  Top3  Top5'+'\n')
file.write('------------------------'+'\n')

TOP1_id = sorted(TOP1_CLASSES.items(), key=operator.itemgetter(1), reverse=True)
for k,v in TOP1_id:
    file.write(jl2c['hv02_'+(str(k).zfill(3))]+' '+str(v)+' '
               +str(TOP3_CLASSES[k])+' '+str(TOP5_CLASSES[k])+' '+'\n')
file.close()

#TOP1 chart
bar_sorted = sorted(TOP1_CLASSES)
bar_id = sorted(TOP1_CLASSES.items(), key=operator.itemgetter(1))
bar_id = ['hv02_'+(str(k).zfill(3)) for k,_ in bar_id]
bar_names = [jl2c[l] for l in bar_id]
pos = np.arange(len(bar_id))
f, ax  = plt.subplots(figsize=(48, 48))
ax.barh(pos, bar_sorted, height=0.8, align='center')
ax.set_yticks(pos)
ax.set_yticklabels(bar_names, fontsize=FONT)
ax.set_xlabel('Top1')
ax.set_title('Top1 chart', fontsize=45)
ax.set_ylim(bottom=0, top=len(bar_id)+1)
plt.savefig(folder+'TOP1_chart.png', format='png')

#histograms
fig = plt.figure()
plt.hist(np.asarray(correct), 50, facecolor=(0, 0, 1, 0.5))
plt.hist(np.asarray(incorrect), 50, facecolor=(1, 0, 0, 0.5))
plt.axis([0.0, 1.0, 0, 250])
plt.title('Correct/Incorrect labels')
plt.xlabel('Confidences')
plt.ylabel('Count')
plt.grid(True)
plt.savefig(folder+'hist.png', format='png')


print('==> Test ran on '+str(nTest)+' samples')

print 'Top1',TOP1
print 'Top3',TOP3
print 'Top5',TOP5
