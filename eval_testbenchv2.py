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

folder = './testbench_results/'+args.model[:-3]+'/'

if not(os.path.exists(folder)):
    os.mkdir(folder)

with open('hv02_labels2classes.json') as json_data:
    mapping = json.load(json_data)

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

width, height = cm.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=5)

names = [mapping['hv02_'+(str(i).zfill(3))] for i in classes]
plt.xticks(range(width), names, rotation=90, fontsize=6)
plt.yticks(range(height), names, rotation=0, fontsize=6)
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
    file.write(mapping['hv02_'+(str(i+1).zfill(3))]+' '+str(top1_classes[i])+' '
               +str(top3_classes[i])+' '+str(top5_classes[i])+' '+'\n')
file.close()

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
plt.savefig(folder+'top1_chart.png', format='png')

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

print 'Top1',top1
print 'Top3',top3
print 'Top5',top5
