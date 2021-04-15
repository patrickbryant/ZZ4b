import time, os, sys
from pathlib import Path
#import multiprocessing
from glob import glob
from copy import copy
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
import torch.multiprocessing as mp
#from nadam import NAdam
from sklearn.metrics import roc_curve, roc_auc_score, auc # pip/conda install scikit-learn
from roc_auc_with_negative_weights import roc_auc_with_negative_weights
#from sklearn.preprocessing import StandardScaler
#from sklearn.externals import joblib
from scipy import interpolate
from scipy.stats import ks_2samp, chisquare
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlibHelpers as pltHelper
from networks import *
np.random.seed(0)#always pick the same training sample
torch.manual_seed(1)#make training results repeatable 
from multiClassifier import classInfo, nameTitle, roc_data, plotROC

import argparse


d4 = classInfo(abbreviation='d4', name=  'FourTag Data',       index=0, color='red')
d3 = classInfo(abbreviation='d3', name= 'ThreeTag Data',       index=1, color='orange')
t4 = classInfo(abbreviation='t4', name= r'FourTag $t\bar{t}$', index=2, color='green')
t3 = classInfo(abbreviation='t3', name=r'ThreeTag $t\bar{t}$', index=3, color='cyan')

zz = classInfo(abbreviation='zz', name= 'ZZ MC',          index=0, color='red')
zh = classInfo(abbreviation='zh', name= 'ZH MC',          index=1, color='orange')
tt = classInfo(abbreviation='tt', name=r'$t\bar{t}$ MC',  index=2, color='green')
mj = classInfo(abbreviation='mj', name= 'Multijet Model', index=3, color='cyan')

sg = classInfo(abbreviation='sg', name='Signal',     index=0, color='blue')
bg = classInfo(abbreviation='bg', name='Background', index=1, color='brown')


bs_milestones=[2,6,14]#[3,6,9]#[1,5,21]#[5,10,15]
lr_milestones=[16,18]#[12,15,18]#[25,30,35]#[20,25,30]

def plotByEpoch(train=None, valid=None, contr=None, ylabel='', plotName='', loc='best', labels=[]):
    fig = plt.figure(figsize=(10,7))

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    #plt.ylim(yMin,yMax)
    x = np.arange(1,20+1)
    
    colors = ['#d34031', 'b', 'g', 'm']

    if labels:
        plt.plot([], [],
                 marker="o",
                 linestyle="-",
                 linewidth=1, alpha=1.0,
                 color="black",
                 label="Training Set")
        plt.plot([], [],
                 marker="o",
                 linestyle="-",
                 linewidth=2, alpha=0.5,
                 color="black",
                 label="Validation Set")
        if contr is not None:
            plt.plot([], [],
                     marker="o",
                     linestyle="--",
                     linewidth=2, alpha=0.5,
                     color="black",
                     label="Control Region")

        for i in range(len(train)):
            plt.plot(x, train[i][1:],
                     marker="o",
                     linestyle="-",
                     linewidth=1, alpha=1.0,
                     color=colors[i],
                     label=labels[i])
            plt.plot(x, valid[i][1:],
                     marker="o",
                     linestyle="-",
                     linewidth=2, alpha=0.5,
                     color=colors[i],
                     label='')
            if contr is not None:
                plt.plot(x, contr[i][:],
                         marker="o",
                         linestyle="--",
                         linewidth=2, alpha=0.5,
                         color=colors[i],
                         label='')

    else:
        plt.plot(x, train,
                 marker="o",
                 linestyle="-",
                 linewidth=1, alpha=1.0,
                 color="#d34031",
                 label="Training")
        plt.plot(x, valid,
                 marker="o",
                 linestyle="-",
                 linewidth=2, alpha=0.5,
                 color="#d34031",
                 label="Validation")
        if contr is not None:
            plt.plot(x, contr,
                     marker="o",
                     linestyle="--",
                     linewidth=2, alpha=0.5,
                     color="#d34031",
                     label="Control Region")
    plt.xticks(x)


    plt.legend(loc=loc)

    ylim = plt.gca().get_ylim()
    xlim = plt.gca().get_xlim()

    for e in bs_milestones:
        plt.plot([e+0.5,e+0.5], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
    for e in lr_milestones:
        plt.plot([e+0.5,e+0.5], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
    if '/' in ylabel:
        plt.plot(xlim, [1,1], color='k', alpha=0.5, linestyle='-', linewidth=1)
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)

    plotName = 'ZZ4b/nTupleAnalysis/pytorchModels/%s.pdf'%(plotName)
    try:
        print(plotName)
        fig.savefig(plotName)
    except:
        print("Cannot save fig: ",plotName)
    plt.close(fig)


def getNPs(fileName):
    return int( fileName[fileName.find('_np')+3: fileName.find('_lr')] )
def getArchitecture(fileName):
    if "ResNet" in fileName: return "HCR"
    if "BasicCNN" in fileName: return "Basic CNN"
    if "BasicDNN" in fileName: return "Basic DNN"

classifiers = {'SvB': ["ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.01_epochs20_offset1_epoch20.pkl",
                       "ZZ4b/nTupleAnalysis/pytorchModels/SvB_BasicCNN_8_8_8_np375_lr0.01_epochs20_offset1_epoch20.pkl",
                       "ZZ4b/nTupleAnalysis/pytorchModels/SvB_BasicDNN_128_128_128_pdrop0.4_np38429_lr0.01_epochs20_offset1_epoch20.pkl",
                   ],
               'FvT': ["ZZ4b/nTupleAnalysis/pytorchModels/FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.01_epochs20_offset1_epoch20.pkl",
                       "ZZ4b/nTupleAnalysis/pytorchModels/FvT_BasicCNN+multijetAttention_8_8_8_np478_lr0.01_epochs20_offset1_epoch20.pkl",
                       "ZZ4b/nTupleAnalysis/pytorchModels/FvT_BasicDNN_128_128_128_pdrop0.4_np38429_lr0.01_epochs20_offset1_epoch20.pkl",
                   ]
           }

classifiers = {'FvT': ["ZZ4b/nTupleAnalysis/pytorchModels/FvT_ResNet+multijetAttention_6_np986_lr0.01_epochs20_offset1_epoch20.pkl",
                       "ZZ4b/nTupleAnalysis/pytorchModels/FvT_ResNet+multijetAttention_8_np1572_lr0.01_epochs20_offset1_epoch20.pkl",
                       "ZZ4b/nTupleAnalysis/pytorchModels/FvT_ResNet+multijetAttention_10_np2294_lr0.01_epochs20_offset1_epoch20.pkl",
                   ]
           }


for classifier, files in classifiers.items():
    items = [nameTitle(name='loss', title='Loss', aux='upper right'),
             nameTitle(name='auc',  title='AUC',  aux='lower right'),
         ]
    if classifier in ['FvT']:
        items.append( nameTitle(name='stat', title='Data / Background', aux='best', abbreviation='norm') )
        classes = [d4, d3, t4, t3]
    if classifier in ['SvB', 'SvB_MA']:
        items.append( nameTitle(name='stat', title='Sensitivity Estimate', aux='lower right', abbreviation='sigma') )
        classes = [zz, zh, tt, mj]

    archs = [torch.load(f)['training history'] for f in files]

    labels = ['{} ({:,d})'.format(getArchitecture(f), getNPs(f) ) for f in files] 

    for item in items:
        train = [arch[  'training.'+item.name] for arch in archs]
        valid = [arch['validation.'+item.name] for arch in archs]
        try:
            contr = [arch['control.'+item.name] for arch in archs]
        except:
            contr = None
            
        plotByEpoch(train, valid, contr, 
                    ylabel=item.title, 
                    plotName='architectureComparison_%s_%s'%(classifier,item.abbreviation), 
                    loc=item.aux, 
                    labels=labels)

    for c in classes:
        train = [[class_loss[c.index] for class_loss in arch[  'training.class_loss']] for arch in archs]
        valid = [[class_loss[c.index] for class_loss in arch['validation.class_loss']] for arch in archs]
        try:
            contr = [[class_loss[c.index] for class_loss in arch['control.class_loss']] for arch in archs]
        except:
            contr = None
            
        plotByEpoch(train, valid, contr, 
                    ylabel='Loss (%s)'%c.name, 
                    plotName='architectureComparison_%s_%s_%s'%(classifier, 'loss', c.abbreviation), 
                    loc=item.aux, 
                    labels=labels)
