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

np.set_printoptions(precision=2)

#require('torch')
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

top1 = 0.0
top3 = 0.0
top5 = 0.0

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

top1_classes = np.ndarray(shape=(400,))
top3_classes = np.ndarray(shape=(400,))
top5_classes = np.ndarray(shape=(400,))
nTest = 0

predictions = []
truth = []
classes = []
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

file.write('Global Top1: '+str(top1)+'\n')
file.write('Global Top3: '+str(top3)+'\n')
file.write('Global Top5: '+str(top5)+'\n')
file.write('\n')
file.write('Scores per class:'+'\n')
file.write('hv02_ Top1 Top3 Top5'+'\n')
file.write('---------------------'+'\n')
for i in sorted(range(len(top1_classes)), key=lambda k: top1_classes[k], reverse=True):
    file.write(str(i+1)+' '+str(top1_classes[i])+' '
               +str(top3_classes[i])+' '+str(top5_classes[i])+' '+'\n')

cm = confusion_matrix(truth, predictions, labels=classes)

#f = plt.figure()
plot_confusion_matrix(cm, classes, normalize=False,
                      title='Normalized confusion matrix')

#plt.savefig('CM.png', format='png')

top1_sorted = sorted(top1_classes, reverse=True)
top1_id = sorted(range(len(top1_classes)), key=lambda k: top1_classes[k], reverse=True)
pos = np.arange(400)+.5
f, ax = plt.subplots()
ax.barh(pos, top1_sorted, align='center')
ax.set_yticks(pos)
ax.set_yticklabels(top1_id)
ax.invert_yaxis()
ax.set_xlabel('Top1')
ax.set_title('Top1 chart')
ax.set_ylim(bottom=0, top=500)
grid(True)
plt.savefig('hist.png', format='png')
print('==> Test ran on '+str(nTest)+' samples')

print 'Top1 ',top1
print('Top3 '+str(top3))
print('Top5 '+str(top5))

file.close()
