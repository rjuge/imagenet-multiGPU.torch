import os
import numpy as np
import lutorpy as lua
import argparse
import itertools
from skimage import transform
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pylab import *

np.set_printoptions(precision=2)

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
elif(args.testbench == 'raw'):
    path = "/home/remi/Deep_Learning/objRecTestbench/dataset/tiny_hv02_testbench/"

file = open('./testbench_results/py_testbench_'+args.model[:-3]+'.txt', 'w')

file.write('Testbench results for model :'+args.model+'\n')
file.write('\n')

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

top1 = 0.0
top3 = 0.0
top5 = 0.0

top1_classes = np.ndarray(shape=(400,))
top3_classes = np.ndarray(shape=(400,))
top5_classes = np.ndarray(shape=(400,))
nTest = 0

predictions = []
truth = []
classes = []

correct=[]
incorrect=[]

for subdir, _, files in os.walk(path):
    if(subdir == path):
        continue

    top1_class = 0.0
    top3_class = 0.0
    top5_class = 0.0
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
        label = int(subdir[-3:])
        truth.append(label)
        
        if(pred_sorted[0] == label):
            top1 = top1 + 1 
            top1_class = top1_class + 1
            correct.append(pred[pred_sorted[0]-1])
        else:
            incorrect.append(pred[pred_sorted[0]-1])
        if(isTop3(pred_sorted, label) == True): 
            top3 = top3 + 1
            top3_class = top3_class + 1
        if(isTop5(pred_sorted, label) == True): 
            top5 = top5 + 1
            top5_class = top5_class + 1

    top1_class = top1_class * 100 / n_class
    top1_classes[(label)-1] = top1_class
    top3_class = top3_class * 100 / n_class
    top3_classes[(label)-1] = top3_class
    top5_class = top5_class * 100 / n_class
    top5_classes[(label)-1] = top5_class


top1 = top1 * 100 / nTest
top3 = top3 * 100 / nTest
top5 = top5 * 100 / nTest

#confusion matrix
cm = confusion_matrix(truth, predictions, labels=classes)

file.write('Global Top1: '+str(top1)+'\n')
file.write('Global Top3: '+str(top3)+'\n')
file.write('Global Top5: '+str(top5)+'\n')
file.write('\n')
file.write('TP:',tp,'\n')
file.write('TN:',tn,'\n')
file.write('FP:',fp,'\n')
file.write('FN:',fn,'\n')
file.write('\n')
file.write('Scores per class:'+'\n')
file.write(' Class  Top1  Top3  Top5'+'\n')
file.write('------------------------'+'\n')

top1_id = sorted(range(len(top1_classes)), key=lambda k: top1_classes[k], reverse=True)
for i in top1_id:
    file.write('hv02_'+(str(i+1).zfill(3))+' '+str(top1_classes[i])+' '
               +str(top3_classes[i])+' '+str(top5_classes[i])+' '+'\n')

#top1 chart
bar_sorted = sorted(top1_classes)
bar_id = sorted(range(len(top1_classes)), key=lambda k: top1_classes[k])
bar_id = ['hv02_'+(str(i+1).zfill(3)) for i in bar_id]
pos = np.arange(len(bar_id))
f, ax  = plt.subplots(figsize=(50, 50))
ax.barh(pos, bar_sorted, height=0.8, align='center')
ax.set_yticks(pos)
ax.set_yticklabels(bar_id)
ax.set_xlabel('Top1')
ax.set_title('Top1 chart')
ax.set_ylim(bottom=0, top=401)
plt.savefig('./testbench_results/top1_chart_'+args.model[:-3]+'.png', format='png')

#histograms
fig = plt.figure()
plt.hist(np.asarray(correct), 50, facecolor=(0, 0, 1, 0.5))
plt.hist(np.asarray(incorrect), 50, facecolor=(1, 0, 0, 0.5))
plt.axis([0.0, 1.0, 0, 250])
plt.title('Correct/Incorrect labels')
plt.xlabel('Confidences')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('./testbench_results/hist_'+args.model[:-3]+'.png', format='png')


print('==> Test ran on '+str(nTest)+' samples')

print 'Top1',top1
print 'Top3',top3
print 'Top5',top5

file.close()
