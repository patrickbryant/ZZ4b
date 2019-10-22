import time, os, sys
from glob import glob
from copy import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from sklearn.metrics import roc_curve, roc_auc_score, auc # pip/conda install scikit-learn
from roc_auc_with_negative_weights import roc_auc_with_negative_weights
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlibHelpers as pltHelper
from networks import *
np.random.seed(0)#always pick the same training sample
torch.manual_seed(1)#make training results repeatable 

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--data', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-t', '--ttbar',      default='',    type=str, help='Input MC ttbar file in hdf5 format')
parser.add_argument('-s', '--signal',     default='', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-c', '--classifier', default='', type=str, help='Which classifier to train: FvT, ZHvB, ZZvB, M1vM2.')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-l', '--lrInit', default=1e-3, type=float, help='Initial learning rate.')
parser.add_argument('-p', '--pDropout', default=0.4, type=float, help='p(drop) for dropout.')
parser.add_argument(      '--layers', default=3, type=int, help='N of fully-connected layers.')
parser.add_argument('-n', '--nodes', default=32, type=int, help='N of fully-connected nodes.')
parser.add_argument('--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
#parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

n_queue = 20
eval_batch_size = 16384
print_step = 2
rate_StoS, rate_BtoB = None, None
barScale=200
barMin=0.5
nClasses=1
#trigger="HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5"
trigger="passHLT"

class cycler:
    def __init__(self,options=['-','\\','|','/']):
        self.cycle=0
        self.options=options
        self.m=len(self.options)
    def next(self):
        self.cycle = (self.cycle + 1)%self.m
        return self.options[self.cycle]

class nameTitle:
    def __init__(self,name,title):
        self.name = name
        self.title= title

class classInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color

loadCycler = cycler()

classifier = args.classifier


if classifier in ['ZHvB', 'ZZvB']:
    barMin=0.8
    barScale=400
    signalName='Signal'
    backgroundName='Background'
    weight = 'weight'
    yTrueLabel = ['y_true']
    ZB = classifier[:2]
    for fileName in sorted(glob(args.signal)):
        if "ZH4b" in fileName: 
            print("Signal is ZH")
        else: 
            print("Signal is ZZ")
 
    updateAttributes = ['y_pred']

    if not args.update:
        train_fraction = 0.7
        train_batch_size_small =  64
        train_batch_size_large = 512

        # Read .h5 files
        frames = []
        for fileName in sorted(glob(args.data)):
            print("Reading",fileName)
            frames.append(pd.read_hdf(fileName, key='df'))
        dfDB = pd.concat(frames, sort=False)
        dfDB = dfDB.loc[ (dfDB[trigger]==True) & (dfDB['fourTag']==False) & ((dfDB['SB']==True)|(dfDB['CR']==True)|(dfDB['SR']==True)) ]
        nD3b = dfDB.shape[0]
        wD3b = np.sum( dfDB[weight] )
        print("nD3b",nD3b)
        print("wD3b",wD3b)
        #print(dfDB[weight],dfDB['fourTag'])

        if args.ttbar:
            frames = []
            for fileName in sorted(glob(args.ttbar)):
                print("Reading",fileName)
                frames.append(pd.read_hdf(fileName, key='df'))
            dfT = pd.concat(frames, sort=False)
            print("Multiply ttbar weights by -1")
            dfT.loc[dfT.fourTag==False, weight] *= -5
            dfT3b = dfT.loc[ (dfT[trigger]==True) & (dfT['fourTag']==False) & ((dfT['SB']==True)|(dfT['CR']==True)|(dfT['SR']==True)) ]
            dfT4b = dfT.loc[ (dfT[trigger]==True) & (dfT['fourTag']==True ) & ((dfT['SB']==True)|(dfT['CR']==True)|(dfT['SR']==True)) ]
            dfT3b = dfT3b.sample(frac=1.0/5, random_state=0)
            nT3b = dfT3b.shape[0]
            nT4b = dfT4b.shape[0]
            wT3b = np.sum(dfT3b[weight])
            wT4b = np.sum(dfT4b[weight])
            print("nT3b",nT3b)
            print("nT4b",nT4b)
            print("wT3b",wT3b)
            print("wT4b",wT4b)

            dfB = pd.concat([dfDB, dfT3b, dfT4b], sort=False)

        else:
            print("WARNING: No ttbar sample specified")
            dfB = dfDB
            
            
        frames = []
        for fileName in sorted(glob(args.signal)):
            print("Reading",fileName)
            frames.append(pd.read_hdf(fileName, key='df'))
        dfS = pd.concat(frames, sort=False)
        #classifier = ZB+'vB'


        #select events in desired region for training/validation/test
        #dfB = dfB.loc[ (dfB[trigger]==True) & (dfB['fourTag']==False) & ((dfB['SB']==True)|(dfB['CR']==True)|(dfB['SR']==True)) ]
        dfS = dfS.loc[ (dfS[trigger]==True) & (dfS['fourTag']==True ) & ((dfS['SB']==True)|(dfS['CR']==True)|(dfS['SR']==True)) ]
        #dfB = dfB.loc[ (dfB[trigger]==True) & (dfB['fourTag']==False) ]
        #dfS = dfS.loc[ (dfS[trigger]==True) & (dfS['fourTag']==True ) ]

        # add y_true values to dataframes
        dfB['y_true'] = pd.Series(np.zeros(dfB.shape[0], dtype=np.uint8), index=dfB.index)
        dfS['y_true'] = pd.Series(np.ones( dfS.shape[0], dtype=np.uint8), index=dfS.index)

        nS      = dfS.shape[0]
        nB      = dfB.shape[0]
        print("nS",nS)
        print("nB",nB)

        # compute relative weighting for S and B
        sum_wS = np.sum(np.float32(dfS[weight]))
        sum_wB = np.sum(np.float32(dfB[weight]))
        print("sum_wS",sum_wS)
        print("sum_wB",sum_wB)

        sum_wStoS = np.sum(np.float32(dfS.loc[dfS[ZB+'SR']==True ][weight]))
        sum_wBtoB = np.sum(np.float32(dfB.loc[dfB[ZB+'SR']==False][weight]))
        print("sum_wStoS",sum_wStoS)
        print("sum_wBtoB",sum_wBtoB)
        rate_StoS = sum_wStoS/sum_wS
        rate_BtoB = sum_wBtoB/sum_wB
        print("Cut Based WP:",rate_StoS,"Signal Eff.", rate_BtoB,"1-Background Eff.")

        dfS[weight] *= sum_wB/sum_wS #normalize signal to background

        #
        # Split into training and validation sets
        #
        nTrainS = int(nS*train_fraction)
        nTrainB = int(nB*train_fraction)
        nValS   = nS-nTrainS
        nValB   = nB-nTrainB

        #random ordering to mix up which data is used for training or validation
        idxS    = np.random.permutation(nS)
        idxB    = np.random.permutation(nB)

        #define dataframes for trainging and validation
        dfS_train = dfS.iloc[idxS[:nTrainS]]
        dfS_val   = dfS.iloc[idxS[nTrainS:]]
        dfB_train = dfB.iloc[idxB[:nTrainB]]
        dfB_val   = dfB.iloc[idxB[nTrainB:]]
    
        df_train = pd.concat([dfB_train, dfS_train], sort=False)
        df_val   = pd.concat([dfB_val,   dfS_val  ], sort=False)

if classifier in ['FvT','DvT3', 'DvT4', 'M1vM2']:
    barMin = 0.34
    barScale=1000
    if classifier == 'M1vM2': barMin, barScale = 0.5, 500
    if classifier == 'DvT3' : barMin, barScale = 0.8, 100
    signalName='FourTag'
    backgroundName='ThreeTag'
    weight = 'mcPseudoTagWeight'
    yTrueLabel = 'target'

    d4 = classInfo(abbreviation='d4', name=  'FourTag Data',       index=0, color='red')
    d3 = classInfo(abbreviation='d3', name= 'ThreeTag Data',       index=1, color='orange')
    t4 = classInfo(abbreviation='t4', name= r'FourTag $t\bar{t}$', index=2, color='green')
    t3 = classInfo(abbreviation='t3', name=r'ThreeTag $t\bar{t}$', index=3, color='cyan')
    classes = [d4,d3,t4,t3]
    eps = 0.0001

    nClasses = len(classes)
    if classifier in ['M1vM2']: yTrueLabel = 'y_true'
    if classifier == 'M1vM2':
        signalName, backgroundName = 'Mixed', 'Unmixed'
        weight = 'weight'
    if classifier in ['DvT3']:
        signalName = r'ThreeTag $t\bar{t} MC$'
        backgroundName = 'ThreeTag Data'
    ZB = ''

    if classifier in ['FvT', 'DvT3', 'DvT4']: 
        updateAttributes = [
            nameTitle('r',   classifier),
            nameTitle('pd4', classifier+'_pd4'),
            nameTitle('pd3', classifier+'_pd3'),
            nameTitle('pt4', classifier+'_pt4'),
            nameTitle('pt3', classifier+'_pt3'),
            nameTitle('pm4', classifier+'_pm4'),
            nameTitle('pm3', classifier+'_pm3'),
            nameTitle('p4',  classifier+'_p4'),
            nameTitle('p3',  classifier+'_p3'),
            nameTitle('pd',  classifier+'_pd'),
            nameTitle('pt',  classifier+'_pt'),
            ]
    wC = torch.FloatTensor([1, 1, 1, 1]).to("cuda")

    if not args.update:
        train_numerator = 7
        train_denominator = 10
        train_fraction = 7/10
        train_offset = 0
        train_batch_size_small =  64
        # train_batch_size_small = 128
        train_batch_size_large = 256
        # Read .h5 files
        frames = []
        for fileName in sorted(glob(args.data)):
            print("Reading",fileName)
            frames.append(pd.read_hdf(fileName, key='df'))
        dfD = pd.concat(frames, sort=False)

        print("Add true class labels to data")
        dfD['d4'] =  dfD.fourTag
        dfD['d3'] = (dfD.fourTag+1)%2
        dfD['t4'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
        dfD['t3'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)

        frames = []
        for fileName in sorted(glob(args.ttbar)):
            print("Reading",fileName)
            frames.append(pd.read_hdf(fileName, key='df'))
        dfT = pd.concat(frames, sort=False)

        print("Add true class labels to ttbar MC")
        dfT['t4'] =  dfT.fourTag
        dfT['t3'] = (dfT.fourTag+1)%2
        dfT['d4'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
        dfT['d3'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
        #dfT[weight] *= 0.5 #749.5/831.76
        #dfT.loc[dfT.weight<0, weight] *= 0
        wtn = dfT.loc[dfT.weight<0, weight].sum()

        print("concatenate data and ttbar dataframes")
        df = pd.concat([dfD, dfT], sort=False)

        print("add encoded target")
        df['target'] = d4.index*df.d4 + d3.index*df.d3 + t4.index*df.t4 + t3.index*df.t3 # classes are mutually exclusive so the target computed in this way is 0,1,2 or 3.

        print("Apply event selection")
        if classifier == 'FvT':
            df = df.loc[ (df[trigger]==True) & (df.SB==True) & (df.xWt>2) ]# & (df[weight]>0) ]
        if classifier == 'DvT3':
            df = df.loc[ (df[trigger]==True) & ((df.d3==True)|(df.t3==True)|(df.t4==True)) & ((df.SB==True)|(df.CR==True)|(df.SR==True)) & (df.xWt>2) ]# & (df[weight]>0) ]
        if classifier == 'DvT4':
            df = df.loc[ (df[trigger]==True) & (df.SB==True) & (df.xWt>2) ]# & (df[weight]>0) ]

        n = df.shape[0]

        nd4, wd4 = df.d4.sum(), df.loc[df.d4==1].mcPseudoTagWeight.sum()
        nd3, wd3 = df.d3.sum(), df.loc[df.d3==1].mcPseudoTagWeight.sum()
        nt4, wt4 = df.t4.sum(), df.loc[df.t4==1].mcPseudoTagWeight.sum()
        nt3, wt3 = df.t3.sum(), df.loc[df.t3==1].mcPseudoTagWeight.sum()

        w = wd4+wd3+wt4+wt3

        fC =      torch.FloatTensor([wd4/w, wd3/w, wt4/w, wt3/w])
        #wC = 0.25*torch.FloatTensor([w/wd4, w/wd3, w/wt4, w/wt3]).to("cuda")
        #fC =      torch.FloatTensor([0.25, 0.25, 0.25, 0.25])
        if classifier == 'DvT3': 
            wC[d4.index] *= 0
            #wC[t4.index] *= 0

        print("nd4 = %7d, wd4 = %6.1f"%(nd4,wd4))
        print("nd3 = %7d, wd3 = %6.1f"%(nd3,wd3))
        print("nt4 = %7d, wt4 = %6.1f"%(nt4,wt4))
        print("nt3 = %7d, wt3 = %6.1f"%(nt3,wt3))
        print("wtn = %6.1f"%(wtn))
        print("fC:",fC)
        print("wC:",wC)

        train_batch_size_small = 512
        train_batch_size_large = 2*train_batch_size_small

        #
        # Split into training and validation sets
        #
        idx_train, idx_val = [], []
        print("build idx with offset %i, modulus %i, and train/val split %i"%(train_offset, train_denominator, train_numerator))
        for e in range(n):
            if (e+train_offset)%train_denominator < train_numerator: 
                idx_train.append(e)
            else:
                idx_val  .append(e)
        idx_train, idx_val = np.array(idx_train), np.array(idx_val)

        print("Split into training and validation sets")
        df_train, df_val = df.iloc[idx_train], df.iloc[idx_val]




# Run on gpu if available
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
print('torch.cuda.is_available()',torch.cuda.is_available())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device",device,torch.cuda.device_count(),torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU:",device)

#from networkTraining import *

class roc_data:
    def __init__(self, y_true, y_pred, weights, trueName, falseName):
        self.fpr, self.tpr, self.thr = roc_curve(y_true, y_pred, sample_weight=weights)
        self.auc = roc_auc_with_negative_weights(y_true, y_pred, weights=weights)
        self.trueName  = trueName
        self.falseName = falseName


class loaderResults:
    def __init__(self, name, alpha=1.0, linewidth=1):
        self.name = name
        self.trainLoader = None
        self. evalLoader = None
        self.smallBatchLoader = None
        self.largeBatchLoader = None
        self.y_true = None
        self.y_pred = None
        self.n      = None
        self.w      = None
        self.fpr    = None
        self.tpr    = None
        self.thr    = None
        self.roc_auc= 0 #[0 for cl in classes]
        self.roc_auc_prev= None #[None for cl in classes]
        self.loss = 1e6
        self.loss_prev = None
        self.loss_best = 1e6
        self.combined_auc = 0
        self.combined_auc_prev = None
        self.roc_auc_decreased=0
        self.roc_auc_best = None
        self.y_pred_S = None
        self.y_pred_B = None
        self.w_S = None
        self.w_B = None
        self.sum_w_S = None
        self.sum_w_B = None
        self.probNorm_StoB = None
        self.probNorm_BtoS = None
        self.probNormRatio_StoB = None
        self.norm_d4_over_B = None
        self.alpha = alpha
        self.linewidth = linewidth
        self.A = None

    def splitAndScale(self):
        self.pd3 = self.y_pred[:,d3.index]
        self.pt3 = self.y_pred[:,t3.index]
        self.pd4 = self.y_pred[:,d4.index]
        self.pt4 = self.y_pred[:,t4.index]

        #renormalize regressed probabilities such that their mean is as expected from the relative fraction of the samples
        self.pd3_ave     = (self.pd3 * self.w).sum()/self.w_sum
        self.pd3_ave_exp = self.wd3.sum()/self.w_sum
        self.sd3 = self.pd3_ave_exp/self.pd3_ave
        self.pt3_ave     = (self.pt3 * self.w).sum()/self.w_sum
        self.pt3_ave_exp = self.wt3.sum()/self.w_sum
        self.st3 = self.pt3_ave_exp/self.pt3_ave
        self.pd4_ave     = (self.pd4 * self.w).sum()/self.w_sum
        self.pd4_ave_exp = self.wd4.sum()/self.w_sum
        self.sd4 = self.pd4_ave_exp/self.pd4_ave
        self.pt4_ave     = (self.pt4 * self.w).sum()/self.w_sum
        self.pt4_ave_exp = self.wt4.sum()/self.w_sum
        self.st4 = self.pt4_ave_exp/self.pt4_ave

        # Compute multijet probabilities
        self.pm4 = self.pd4 - self.pt4
        self.pm3 = self.pd3 - self.pt3

        self.p4 = self.pd4 + self.pt4
        self.p3 = self.pd3 + self.pt3
        self.pd = self.pd4 + self.pd3
        self.pt = self.pt4 + self.pt3

        # # fix divide by zero
        # self.pm4[self.pm4>=0] += eps
        # self.pm4[self.pm4< 0] -= eps
        # self.pm3[self.pm3>=0] += eps
        # self.pm3[self.pm3< 0] -= eps

        # Compute reweight factor
        # self.r = self.pm4/self.pm3
        self.r = self.pm4/self.pd3

        #regressed probabilities for fourTag data to be each class
        self.pd4d4 = self.y_pred[self.y_true==d4.index][:,d4.index]
        self.pd4t4 = self.y_pred[self.y_true==d4.index][:,t4.index]
        self.pd4d3 = self.y_pred[self.y_true==d4.index][:,d3.index]
        self.pd4t3 = self.y_pred[self.y_true==d4.index][:,t3.index]

        #regressed probabilities for threeTag data to be each class
        self.pd3d4 = self.y_pred[self.y_true==d3.index][:,d4.index]
        self.pd3t4 = self.y_pred[self.y_true==d3.index][:,t4.index]
        self.pd3d3 = self.y_pred[self.y_true==d3.index][:,d3.index]
        self.pd3t3 = self.y_pred[self.y_true==d3.index][:,t3.index]

        #regressed probabilities for fourTag ttbar MC to be each class
        self.pt4d4 = self.y_pred[self.y_true==t4.index][:,d4.index]
        self.pt4t4 = self.y_pred[self.y_true==t4.index][:,t4.index]
        self.pt4d3 = self.y_pred[self.y_true==t4.index][:,d3.index]
        self.pt4t3 = self.y_pred[self.y_true==t4.index][:,t3.index]

        #regressed probabilities for threeTag ttbar MC to be each class
        self.pt3d4 = self.y_pred[self.y_true==t3.index][:,d4.index]
        self.pt3t4 = self.y_pred[self.y_true==t3.index][:,t4.index]
        self.pt3d3 = self.y_pred[self.y_true==t3.index][:,d3.index]
        self.pt3t3 = self.y_pred[self.y_true==t3.index][:,t3.index]


        #Define regressed probabilities for each class to be multijet
        self.pd4m4 = self.pm4[self.y_true==d4.index] # self.pd4d4 - self.pd4t4
        self.pd4m3 = self.pm3[self.y_true==d4.index] # self.pd4d3 - self.pd4t3
        self.pd3m4 = self.pm4[self.y_true==d3.index] # self.pd3d4 - self.pd3t4
        self.pd3m3 = self.pm3[self.y_true==d3.index] # self.pd3d3 - self.pd3t3
        self.pt3m4 = self.pm4[self.y_true==t3.index] # self.pt3d4 - self.pt3t4
        self.pt3m3 = self.pm3[self.y_true==t3.index] # self.pt3d3 - self.pt3t3
        self.pt4m4 = self.pm4[self.y_true==t4.index] # self.pt4d4 - self.pt4t4
        self.pt4m3 = self.pm3[self.y_true==t4.index] # self.pt4d3 - self.pt4t3

        #Compute multijet weights for each class
        self.rd4 = self.r[self.y_true==d4.index] # self.pd4m4/self.pd4m3
        self.rd3 = self.r[self.y_true==d3.index] # self.pd3m4/self.pd3m3
        self.rt4 = self.r[self.y_true==t4.index] # self.pt4m4/self.pt4m3
        self.rt3 = self.r[self.y_true==t3.index] # self.pt3m4/self.pt3m3

        #Compute normalization of the reweighted background model
        # self.normB = ( self.wd3 * self.rd3 ).sum() - ( self.wt3 * self.rt3 ).sum() + self.wt4.sum()
        self.normB = ( self.wd3 * self.rd3 ).sum() + self.wt4.sum()
        self.norm_d4_over_B = self.wd4.sum()/self.normB

    def update(self, y_pred, y_true, w_ordered, loss, doROC=False):
        self.y_pred = y_pred
        self.y_true = y_true
        self.w      = w_ordered
        self.loss   = loss
        self.w_sum  = self.w.sum()
        self.wB = np.copy(self.w)
        self.wB[self.y_true==t3.index] *= -1

        # Weights for each class
        self.wd4 = self.w[self.y_true==d4.index]
        self.wt4 = self.w[self.y_true==t4.index]
        self.wd3 = self.w[self.y_true==d3.index]
        self.wt3 = self.w[self.y_true==t3.index]

        self.wt4n = self.wt4[self.wt4<0]
        self.wt3n = self.wt3[self.wt3<0]
        #print()
        #print("t3 negative fraction:",abs(self.wt3n.sum())/self.wt3.sum())
        #print("t4 negative fraction:",abs(self.wt4n.sum())/self.wt4.sum())

        self.splitAndScale()

        if doROC:
            if classifier in ['DvT3']:
                self.roc_t3 = roc_data(np.array(self.y_true==t3.index, dtype=np.float), 
                                       self.y_pred[:,t3.index], 
                                       self.w,
                                       r'ThreeTag $t\bar{t}$ MC',
                                       'ThreeTag Data')
            if classifier in ['FvT','DvT4']:
                self.roc_t3 = roc_data(np.array(self.y_true==t3.index, dtype=np.float), 
                                       self.y_pred[:,t3.index], 
                                       self.w,
                                       r'ThreeTag $t\bar{t}$ MC',
                                       'Other')
                self.roc_td = roc_data(np.array((self.y_true==t3.index)|(self.y_true==t4.index), dtype=np.float), 
                                       self.y_pred[:,t3.index]+self.y_pred[:,t4.index], 
                                       self.w,
                                       r'$t\bar{t} MC$',
                                       'Data')
                self.roc_43 = roc_data(np.array((self.y_true==t4.index)|(self.y_true==d4.index), dtype=np.float), 
                                       self.y_pred[:,t4.index]+self.y_pred[:,d4.index], 
                                       self.w,
                                       'FourTag',
                                       'ThreeTag')


class modelParameters:
    def __init__(self, fileName=''):
        self.xVariables=['canJet0_pt', 'canJet1_pt', 'canJet2_pt', 'canJet3_pt',
                         'dRjjClose', 'dRjjOther', 
                         'aveAbsEta', 'xWt1',
                         'nSelJets', 'm4j',
                         ]
        #             |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        self.layer1Pix = "012302130312"
        # if classifier ==   "FvT": self.fourVectors=[['canJet%s_pt'%i, 'canJet%s_eta'%i, 'canJet%s_phi'%i, 'canJet%s_m'%i, 'm'+('0123'.replace(i,''))] for i in self.layer1Pix] #index[pixel][color]
        # if classifier == ZB+"vB": self.fourVectors=[['canJet%s_pt'%i, 'canJet%s_eta'%i, 'canJet%s_phi'%i, 'canJet%s_m'%i, 'm'+('0123'.replace(i,''))] for i in self.layer1Pix] #index[pixel][color]
        self.fourVectors=[['canJet%s_pt'%i, 'canJet%s_eta'%i, 'canJet%s_phi'%i, 'canJet%s_m'%i] for i in self.layer1Pix] #index[pixel][color]
        #if classifier == ZB+"vB": self.fourVectors=[['canJet%s_pt'%i, 'canJet%s_eta'%i, 'canJet%s_phi'%i, 'canJet%s_m'%i] for i in self.layer1Pix] #index[pixel][color]
        self.othJets = [['notCanJet%i_pt'%i, 'notCanJet%i_eta'%i, 'notCanJet%i_phi'%i, 'notCanJet%i_m'%i, 'notCanJet%i_isSelJet'%i] for i in range(12)]#, 'notCanJet'+i+'_isSelJet'
        self.jetFeatures = len(self.fourVectors[0])
        self.othJetFeatures = len(self.othJets[0])

        self.dijetAncillaryFeatures=[ 'm01',  'm23',  'm02',  'm13',  'm03',  'm12',
                                     'dR01', 'dR23', 'dR02', 'dR13', 'dR03', 'dR12',
                                    #'pt01', 'pt23', 'pt02', 'pt13', 'pt03', 'pt12',
                                      #m012
                                      ]

        self.quadjetAncillaryFeatures=['dR0123', 'dR0213', 'dR0312',
                                       'm4j',    'm4j',    'm4j',
                                       #'m012', 'm012', 'm012',
                                       #'m123', 'm123', 'm123',
                                       ]
        #if classifier == "FvT": self.quadjetAncillaryFeatures += ['m4j', 'm4j', 'm4j']
        #else: self.quadjetAncillaryFeatures += ['m'+ZB+'0123', 'm'+ZB+'0213', 'm'+ZB+'0312']

        self.ancillaryFeatures = ['nSelJets', 'xWt']
        #if classifier in ['M1vM2']: self.ancillaryFeatures[1] = 'xWt1'
        #if classifier == "FvT":   self.ancillaryFeatures += ['stNotCan', 'xWt1', 'aveAbsEtaOth', 'nPVsGood']#, 'dRjjClose', 'dRjjOther', 'aveAbsEtaOth']#, 'nPSTJets']
        #if classifier == "FvT":   self.ancillaryFeatures += ['xWt']#, 'dRjjClose', 'dRjjOther', 'aveAbsEtaOth']#, 'nPSTJets']
        #if classifier == ZB+"vB": self.ancillaryFeatures += ['xWt']#, 'nPSTJets']
        self.useOthJets = ''
        if classifier in ["FvT", 'DvT3', 'DvT4', "M1vM2"]: self.useOthJets = 'multijetAttention'

        self.validation = loaderResults("validation")
        self.training   = loaderResults("training")

        if fileName:
            self.classifier           = classifier #    fileName.split('_')[0] if not 
            if "FC" in fileName:
                self.nodes         =   int(fileName[fileName.find(      'x')+1 : fileName.find('_pdrop')])
                self.layers        =   int(fileName[fileName.find(     'FC')+2 : fileName.find('x')])
                self.dijetFeatures = None
                self.quadjetFeatures = None
                self.combinatoricFeatures = None
                self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_np')])
            if "ResNet" in fileName:
                self.dijetFeatures        = int(fileName.split('_')[2])
                self.quadjetFeatures      = int(fileName.split('_')[3])
                self.combinatoricFeatures = int(fileName.split('_')[4])
                self.nodes    = None
                self.pDropout = None
            self.lrInit        = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_loss')])
            self.validation.loss_best  = float(fileName[fileName.find(   '_loss')+5 : fileName.find('.pkl')])
            self.scalers = torch.load(fileName)['scalers']

        else:
            self.dijetFeatures = 6
            self.quadjetFeatures = 6
            self.combinatoricFeatures = 6 #ZZ4b/nTupleAnalysis/pytorchModels/FvT_ResNet+LSTM_8_6_8_np2409_lr0.001_epochs20_stdscale_epoch9_auc0.5934.pkl
            self.nodes         = args.nodes
            self.layers        = args.layers
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0           
            self.validation.loss_best  = 0.87 if args.signal else 0.163
            if classifier in ['M1vM2']: self.validation.roc_auc_best = 0.5
            self.scalers = {}

        self.epoch = self.startingEpoch

        #self.net = basicCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        #self.net = dijetCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        self.net = ResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, len(self.ancillaryFeatures), self.useOthJets, device=device, nClasses=nClasses).to(device)
        #self.net.debug=True
        #self.net = ResNetZero(self.jetFeatures, len(self.dijetAncillaryFeatures)//6, len(self.quadjetAncillaryFeatures)//3, len(self.ancillaryFeatures), self.useOthJets).to(device)
        #self.net = basicDNN(len(self.xVariables), self.layers, self.nodes, self.pDropout).to(device)
        #self.net = PhiInvResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures).to(device)
        #self.net = PresResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        #self.net = deepResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        self.nTrainableParameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.name = classifier+'_'+self.net.name+'_np%d_lr%s_epochs%d_stdscale'%(self.nTrainableParameters, str(self.lrInit), args.epochs+self.startingEpoch)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrInit, amsgrad=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.2, threshold=0.0001, threshold_mode='rel', patience=1, cooldown=2, min_lr=0, verbose=True)

        self.foundNewBest = False
        
        self.dump()

        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state
            self.optimizer.load_state_dict(torch.load(fileName)['optimizer'])

            if args.update:
                for sample in [args.data, args.ttbar, args.signal]:
                    for sampleFile in sorted(glob(sample)):
                        self.update(sampleFile)
                exit()


    def epochString(self):
        return ('>> %'+str(len(str(args.epochs+self.startingEpoch)))+'d/%d <<')%(self.epoch, args.epochs+self.startingEpoch)

    def dfToTensors(self, df, y_true=None):
        n = df.shape[0]
        #basicDNN variables
        X=torch.FloatTensor( np.float32(df[self.xVariables]) )

        #Convert to list np array
        P=[np.float32(df[jet]) for jet in self.fourVectors]
        O=[np.float32(df[jet]) for jet in self.othJets]
        #make 3D tensor with correct axes [event][color][pixel] = [event][mu (4-vector component)][jet]
        P=torch.FloatTensor( [np.float32([[P[jet][event][mu] for jet in range(len(self.fourVectors))] for mu in range(self.jetFeatures)]) for event in range(n)] )
        O=torch.FloatTensor( [np.float32([[O[jet][event][mu] for jet in range(len(self.othJets    ))] for mu in range(self.othJetFeatures)]) for event in range(n)] )

        # take log of jet pt's, m's
        #P[:,(0,3),:] = torch.log(P[:,(0,3),:])
        #O[:,(0,3),:][O[:,(0,3),:]==0] = 0.1
        #O[:,(0,3),:] = torch.log(O[:,(0,3),:])

        #extra features 
        D=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.dijetAncillaryFeatures], 1 )
        Q=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.quadjetAncillaryFeatures], 1 )
        A=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.ancillaryFeatures], 1 ) 

        if y_true:
            y=torch.LongTensor( np.array(df[y_true], dtype=np.uint8).reshape(-1) )
        else:#assume all zero. y_true not needed for updating classifier output values in .h5 files for example.
            y=torch.LongTensor( np.zeros(df.shape[0], dtype=np.uint8).reshape(-1) )

        w=torch.FloatTensor( np.float32(df[weight]).reshape(-1) )

        #print('P.shape, A.shape, y.shape, w.shape:', P.shape, A.shape, y.shape, w.shape)
        return X, P, O, D, Q, A, y, w

    def update(self, fileName):
        print("Add",classifier,"output to",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')
 
        n = df.shape[0]
        print("n",n)

        X, P, O, D, Q, A, y, w = self.dfToTensors(df)
        print('P.shape', P.shape)

        X = torch.FloatTensor(self.scalers['xVariables'].transform(X))
        for jet in range(P.shape[2]):
            P[:,:,jet] = torch.FloatTensor(self.scalers[0].transform(P[:,:,jet]))
        for jet in range(O.shape[2]):
            O[:,:,jet] = torch.FloatTensor(self.scalers['othJets'].transform(O[:,:,jet]))
        D = torch.FloatTensor(self.scalers['dijetAncillary'].transform(D))
        Q = torch.FloatTensor(self.scalers['quadjetAncillary'].transform(Q))
        A = torch.FloatTensor(self.scalers['ancillary'].transform(A))

        # Set up data loaders
        dset   = TensorDataset(X, P, O, D, Q, A, y, w)
        updateResults = loaderResults("update")
        updateResults.evalLoader = DataLoader(dataset=dset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        updateResults.n = n
        print('Batches:', len(updateResults.evalLoader))

        self.evaluate(updateResults, doROC = False)

        for attribute in updateAttributes:
            print(attribute.name, attribute.title, getattr(updateResults, attribute.name))
            df[attribute.title] = pd.Series(np.float32(getattr(updateResults, attribute.name)), index=df.index)
        #print("df.dtypes")
        #print(df.dtypes)
        print("df.shape", df.shape)
        df.to_hdf(fileName, key='df', format='table', mode='w')

        del dset
        del updateResults

    def trainSetup(self, df_train, df_val):
        print("Convert df_train to tensors")
        X_train, P_train, O_train, D_train, Q_train, A_train, y_train, w_train = self.dfToTensors(df_train, y_true=yTrueLabel)
        print("Convert df_val to tensors")
        X_val,   P_val  , O_val  , D_val  , Q_val  , A_val  , y_val  , w_val   = self.dfToTensors(df_val  , y_true=yTrueLabel)
        print('P_train.shape, O_train.shape, A_train.shape, y_train.shape, w_train.shape:', P_train.shape, O_train.shape, A_train.shape, y_train.shape, w_train.shape)
        print('P_val  .shape, O_val  .shape, A_val  .shape, y_val  .shape, w_val  .shape:', P_val  .shape, O_val  .shape, A_val  .shape, y_val  .shape, w_val  .shape)

        # Standardize inputs
        if not args.model:
            self.scalers['xVariables'] = StandardScaler()
            self.scalers['xVariables'].fit(X_train)
            self.scalers[0] = StandardScaler()
            self.scalers[0].fit(P_train[:,:,1])
            self.scalers[0].scale_[1], self.scalers[0].mean_[1] =   2.4,  0 # eta max
            self.scalers[0].scale_[2], self.scalers[0].mean_[2] = np.pi,  0 # pi
            self.scalers['othJets'] = StandardScaler()
            self.scalers['othJets'].fit(O_train[:,:,0])
            self.scalers['othJets'].scale_[0], self.scalers['othJets'].mean_[0] = self.scalers[0].scale_[0], self.scalers[0].mean_[0] #use same pt scale as candidate jets
            self.scalers['othJets'].scale_[1], self.scalers['othJets'].mean_[1] =   2.4,  0 # eta max
            self.scalers['othJets'].scale_[2], self.scalers['othJets'].mean_[2] = np.pi,  0 # pi
            self.scalers['othJets'].scale_[3], self.scalers['othJets'].mean_[3] = self.scalers[0].scale_[3], self.scalers[0].mean_[3] #use same mass scale as candidate jets
            self.scalers['othJets'].scale_[4], self.scalers['othJets'].mean_[4] =     1,  0 # isSelJet
            print("self.scalers[0].scale_",self.scalers[0].scale_)
            print("self.scalers[0].mean_",self.scalers[0].mean_)
            self.scalers['dijetAncillary'], self.scalers['quadjetAncillary'], self.scalers['ancillary'] = StandardScaler(), StandardScaler(), StandardScaler()
            self.scalers['dijetAncillary'].fit(D_train)
            self.scalers['quadjetAncillary'].fit(Q_train)
            self.scalers['ancillary'].fit(A_train)

            #dijet masses
            self.scalers['dijetAncillary'].scale_[0], self.scalers['dijetAncillary'].mean_[0] = 100, 130
            self.scalers['dijetAncillary'].scale_[1], self.scalers['dijetAncillary'].mean_[1] = 100, 130
            self.scalers['dijetAncillary'].scale_[2], self.scalers['dijetAncillary'].mean_[2] = 100, 130
            self.scalers['dijetAncillary'].scale_[3], self.scalers['dijetAncillary'].mean_[3] = 100, 130
            self.scalers['dijetAncillary'].scale_[4], self.scalers['dijetAncillary'].mean_[4] = 100, 130
            self.scalers['dijetAncillary'].scale_[5], self.scalers['dijetAncillary'].mean_[5] = 100, 130

            #dijet pts
            # self.scalers['dijetAncillary'].scale_[ 6], self.scalers['dijetAncillary'].mean_[ 6] = self.scalers['dijetAncillary'].scale_[7], self.scalers['dijetAncillary'].mean_[7]
            # self.scalers['dijetAncillary'].scale_[ 7], self.scalers['dijetAncillary'].mean_[ 7] = self.scalers['dijetAncillary'].scale_[7], self.scalers['dijetAncillary'].mean_[7]
            # self.scalers['dijetAncillary'].scale_[ 8], self.scalers['dijetAncillary'].mean_[ 8] = self.scalers['dijetAncillary'].scale_[7], self.scalers['dijetAncillary'].mean_[7]
            # self.scalers['dijetAncillary'].scale_[ 9], self.scalers['dijetAncillary'].mean_[ 9] = self.scalers['dijetAncillary'].scale_[7], self.scalers['dijetAncillary'].mean_[7]
            # self.scalers['dijetAncillary'].scale_[10], self.scalers['dijetAncillary'].mean_[10] = self.scalers['dijetAncillary'].scale_[7], self.scalers['dijetAncillary'].mean_[7]
            # self.scalers['dijetAncillary'].scale_[11], self.scalers['dijetAncillary'].mean_[11] = self.scalers['dijetAncillary'].scale_[7], self.scalers['dijetAncillary'].mean_[7]

            #dijet dRjj's
            self.scalers['dijetAncillary'].scale_[ 6], self.scalers['dijetAncillary'].mean_[ 6] = np.pi/2, np.pi/2
            self.scalers['dijetAncillary'].scale_[ 7], self.scalers['dijetAncillary'].mean_[ 7] = np.pi/2, np.pi/2
            self.scalers['dijetAncillary'].scale_[ 8], self.scalers['dijetAncillary'].mean_[ 8] = np.pi/2, np.pi/2
            self.scalers['dijetAncillary'].scale_[ 9], self.scalers['dijetAncillary'].mean_[ 9] = np.pi/2, np.pi/2
            self.scalers['dijetAncillary'].scale_[10], self.scalers['dijetAncillary'].mean_[10] = np.pi/2, np.pi/2
            self.scalers['dijetAncillary'].scale_[11], self.scalers['dijetAncillary'].mean_[11] = np.pi/2, np.pi/2

            #quadjet dRBB's
            self.scalers['quadjetAncillary'].scale_[0], self.scalers['quadjetAncillary'].mean_[0] = np.pi/2, np.pi
            self.scalers['quadjetAncillary'].scale_[1], self.scalers['quadjetAncillary'].mean_[1] = np.pi/2, np.pi
            self.scalers['quadjetAncillary'].scale_[2], self.scalers['quadjetAncillary'].mean_[2] = np.pi/2, np.pi

            # #quadjet mZH's
            self.scalers['quadjetAncillary'].scale_[3], self.scalers['quadjetAncillary'].mean_[3] = self.scalers['quadjetAncillary'].scale_[4], self.scalers['quadjetAncillary'].mean_[4]
            self.scalers['quadjetAncillary'].scale_[4], self.scalers['quadjetAncillary'].mean_[4] = self.scalers['quadjetAncillary'].scale_[4], self.scalers['quadjetAncillary'].mean_[4]
            self.scalers['quadjetAncillary'].scale_[5], self.scalers['quadjetAncillary'].mean_[5] = self.scalers['quadjetAncillary'].scale_[4], self.scalers['quadjetAncillary'].mean_[4]

            # #nSelJets
            # self.scalers['ancillary'].scale_[0], self.scalers['ancillary'].mean_[0] =   4,   8

            print("self.scalers['dijetAncillary'].scale_",self.scalers['dijetAncillary'].scale_)
            print("self.scalers['dijetAncillary'].mean_",self.scalers['dijetAncillary'].mean_)
            print("self.scalers['quadjetAncillary'].scale_",self.scalers['quadjetAncillary'].scale_)
            print("self.scalers['quadjetAncillary'].mean_",self.scalers['quadjetAncillary'].mean_)
            print("self.scalers['ancillary'].scale_",self.scalers['ancillary'].scale_)
            print("self.scalers['ancillary'].mean_",self.scalers['ancillary'].mean_)

        X_train = torch.FloatTensor(self.scalers['xVariables'].transform(X_train))
        X_val   = torch.FloatTensor(self.scalers['xVariables'].transform(X_val))
        for jet in range(P_train.shape[2]):
            P_train[:,:,jet] = torch.FloatTensor(self.scalers[0].transform(P_train[:,:,jet]))
            P_val  [:,:,jet] = torch.FloatTensor(self.scalers[0].transform(P_val  [:,:,jet]))
        for jet in range(O_train.shape[2]):
            O_train[:,:,jet] = torch.FloatTensor(self.scalers['othJets'].transform(O_train[:,:,jet]))
            O_val  [:,:,jet] = torch.FloatTensor(self.scalers['othJets'].transform(O_val  [:,:,jet]))
        D_train = torch.FloatTensor(self.scalers['dijetAncillary'].transform(D_train))
        D_val   = torch.FloatTensor(self.scalers['dijetAncillary'].transform(D_val))
        Q_train = torch.FloatTensor(self.scalers['quadjetAncillary'].transform(Q_train))
        Q_val   = torch.FloatTensor(self.scalers['quadjetAncillary'].transform(Q_val))
        A_train = torch.FloatTensor(self.scalers['ancillary'].transform(A_train))
        A_val   = torch.FloatTensor(self.scalers['ancillary'].transform(A_val))

        # Set up data loaders
        dset_train   = TensorDataset(X_train, P_train, O_train, D_train, Q_train, A_train, y_train, w_train)
        dset_val     = TensorDataset(X_val,   P_val,   O_val,   D_val,   Q_val,   A_val,   y_val,   w_val)
        self.training.smallBatchLoader = DataLoader(dataset=dset_train, batch_size=train_batch_size_small, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True)
        #self.training.largeBatchLoader = DataLoader(dataset=dset_train, batch_size=train_batch_size_large, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True)
        self.training  .evalLoader     = DataLoader(dataset=dset_train, batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.validation.evalLoader     = DataLoader(dataset=dset_val,   batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.training.n, self.validation.n = w_train.shape[0], w_val.shape[0]
        self.training .trainLoader     = self.training.smallBatchLoader
        print("Training Batch Size:",train_batch_size_small)
        print("Training Batches:",len(self.training.trainLoader))

        #model initial state
        epochSpaces = max(len(str(args.epochs))-2, 0)
        print(">> "+(epochSpaces*" ")+"Epoch"+(epochSpaces*" ")+" <<   Data Set |  Loss  | Norm | % AUC | AUC Bar Graph ^ ROC % ABC * Output Model")
        self.trainEvaluate()
        self.validate(doROC=True)
        #plotClasses(self.training, self.validation, 'test.pdf')
        print()
        self.scheduler.step(self.validation.loss)
        #self.validation.roc_auc_prev = copy(self.validation.roc_auc)


    def evaluate(self, results, doROC=True):
        self.net.eval()
        y_pred, y_true, w_ordered = np.ndarray((results.n,nClasses), dtype=np.float), np.zeros(results.n, dtype=np.float), np.zeros(results.n, dtype=np.float)
        A_ordered = np.ndarray((results.n,self.net.nAv), dtype=np.float)
        print_step = len(results.evalLoader)//200+1
        nProcessed = 0
        loss = 0
        for i, (X, P, O, D, Q, A, y, w) in enumerate(results.evalLoader):
            nBatch = w.shape[0]
            X, P, O, D, Q, A, y, w = X.to(device), P.to(device), O.to(device), D.to(device), Q.to(device), A.to(device), y.to(device), w.to(device)
            logits = self.net(X, P, O, D, Q, A)
            loss += (w * F.cross_entropy(logits, y, weight=wC, reduction='none')).sum(dim=0).cpu().item()
            y_pred[nProcessed:nProcessed+nBatch] = F.softmax(logits, dim=-1).detach().cpu().numpy()
            y_true[nProcessed:nProcessed+nBatch] = y.cpu()
            w_ordered[nProcessed:nProcessed+nBatch] = w.cpu()
            A_ordered[nProcessed:nProcessed+nBatch] = A.cpu()
            # if self.epoch >= 1: 
            #     print(y_pred[nProcessed],y_true[nProcessed],w[nProcessed])
            #     input()
            nProcessed+=nBatch
            if int(i+1) % print_step == 0:
                percent = float(i+1)*100/len(results.evalLoader)
                sys.stdout.write('\rEvaluating %3.0f%%     '%(percent))
                sys.stdout.flush()

        loss = loss/results.n   
        #print("<loss>:",loss,"<y_pred>:",y_pred.mean(axis=0))
        results.update(y_pred, y_true, w_ordered, loss, doROC)
        results.A = A_ordered


    def validate(self, doROC=True):
        self.evaluate(self.validation, doROC)
        bar=1-self.validation.loss*nClasses
        bar=int((bar-barMin)*barScale) if bar > barMin else 0
        # bar=int((self.validation.roc_auc-barMin)*barScale) if self.validation.roc_auc > barMin else 0

        # roc_abc=None
        overtrain=""
        # if self.training.roc_auc: 
        #     n = self.validation.fpr.shape[0]
        #     roc_val = interpolate.interp1d(self.validation.fpr[np.arange(0,n,100)], self.validation.tpr[np.arange(0,n,100)], fill_value="extrapolate")
        #     tpr_val = roc_val(self.training.fpr)#validation tpr estimated at training fpr
        #     roc_abc = auc(self.training.fpr, np.abs(self.training.tpr-tpr_val)) #area between curves
        #     #roc_abc = auc(self.training.fpr, (self.training.tpr-tpr_val)**2)**0.5 #quadrature sum of area between curves. Points where curves are close contribute very little to this metric of overtraining
        #     overtrain="^ %1.1f%%"%(roc_abc*100/(self.training.roc_auc-0.5))
        print('\r'+self.epochString()+' Validation | %0.4f | %0.2f | %2.2f'%(self.validation.loss, self.validation.norm_d4_over_B, self.validation.roc_auc*100),"|"+("#"*bar)+"|",overtrain, end = " ")


    def train(self):
        self.net.train()
        #if self.epoch == 5: self.net.debug=True
        print_step = len(self.training.trainLoader)//200+1
        accuracy = 0.0
        totalLoss = 0
        totalttError = 0
        totalLargeReweightLoss = 0
        rMax=0
        for i, (X, P, O, D, Q, A, y, w) in enumerate(self.training.trainLoader):
            X, P, O, D, Q, A, y, w = X.to(device), P.to(device), O.to(device), D.to(device), Q.to(device), A.to(device), y.to(device), w.to(device)
            self.optimizer.zero_grad()
            logits = self.net(X, P, O, D, Q, A)
            w_sum = w.sum()

            #compute classification loss
            loss  = (w * F.cross_entropy(logits, y, weight=wC, reduction='none')).mean(dim=0)

            # compute loss term to account for failure to always give data higher prob than ttbar
            y_pred = F.softmax(logits, dim=-1)
            t3d3 = y_pred[:,t3.index] - y_pred[:,d3.index]
            t4d4 = y_pred[:,t4.index] - y_pred[:,d4.index]
            t3d3 = F.relu(t3d3)
            t4d4 = F.relu(t4d4)
            ttbarOverPredictionError = 1*(w*t3d3 + w*t4d4).mean()
            totalttError += ttbarOverPredictionError
            # add amount pt>pd to loss
            # if self.epoch >= 2:
            #     loss += ttbarOverPredictionError

            # compute loss term to penalize reweight factors greater than some threshold
            m3 = (y_pred[:,d3.index] - y_pred[:,t3.index]).abs() + eps
            m4 = (y_pred[:,d4.index] - y_pred[:,t4.index]).abs() # + eps
            r = m4/y_pred[:,d3.index] # m4/m3
            # only penlize three-tag events because fourtag events do not get reweighted
            r[y==t4.index] *= 0
            r[y==d4.index] *= 0
            largeReweightLoss = 1*(w*torch.log1p(F.relu(r-10))).mean()
            totalLargeReweightLoss += largeReweightLoss
            # if self.epoch >= 4:
            #     loss += largeReweightLoss

            #r[y==t4.index] *= 0
            #r[y==d4.index] *= 0
            rMax = torch.max(r) if torch.max(r)>rMax else rMax

            #perform backprop
            loss.backward()
            self.optimizer.step()

            totalLoss+=loss.item()
            binary_pred = logits[:,d4.index].ge(0.).byte()
            #w[y==t3.index]*=-1
            binary_result = binary_pred.eq((y==0).byte()).float()*w
            accuracy += binary_result.sum().item()/sum(w)
            if (i+1) % print_step == 0:
                l = totalLoss/print_step#/(i+1)
                t = totalttError/print_step * 1e4
                r = totalLargeReweightLoss/print_step
                a =  accuracy/print_step#/(i+1)
                accuracy, totalLoss, totalttError, totalLargeReweightLoss = 0, 0, 0, 0
                if a > 1: a = 0
                bar=1-loss*nClasses
                bar=int((bar-barMin)*barScale) if bar > barMin else 0
                #bar=int((1-loss*nClasses)*200)
                sys.stdout.write(str(('\rTraining %3.0f%% ('+loadCycler.next()+') Loss: %0.3f (ttbar>data %0.3f/1e4, r>10 %0.3f, rMax %0.1f)')%(float(i+1)*100/len(self.training.trainLoader),l,t,r,rMax))+' '+('-'*bar)+"|")
                sys.stdout.flush()

                self.net.out.gradStats.dump()
                self.net.out.gradStats.reset()
        self.trainEvaluate()

    def trainEvaluate(self):
        self.evaluate(self.training)
        sys.stdout.write(' '*200)
        sys.stdout.flush()
        bar=1-self.training.loss*nClasses
        bar=int((bar-barMin)*barScale) if bar > barMin else 0
        #bar=int((self.training.roc_auc-barMin)*barScale) if self.training.roc_auc > barMin else 0
        print('\r'+' '*len(self.epochString())+'   Training | %0.4f | %0.2f | %2.2f'%(self.training.loss, self.training.norm_d4_over_B, self.training.roc_auc*100),"|"+("-"*bar)+"|")


    def runEpoch(self):
        self.epoch += 1

        self.train()
        self.validate()

        if self.validation.loss < self.validation.loss_best or abs(self.validation.norm_d4_over_B-1)<0.01:
            if self.validation.loss < self.validation.loss_best:
                self.foundNewBest = True
                self.validation.loss_best = copy(self.validation.loss)
    
            modelPkl = 'ZZ4b/nTupleAnalysis/pytorchModels/%s_epoch%d_loss%.4f.pkl'%(self.name, self.epoch, self.validation.loss)
            print("*", modelPkl)
            model_dict = {'model': model.net.state_dict(), 'optimizer': model.optimizer.state_dict(), 'scalers': model.scalers}
            torch.save(model_dict, modelPkl)

            if classifier in ['DvT3', 'DvT4']:
                plotROC(self.training.roc_t3, self.validation.roc_t3, plotName=modelPkl.replace('.pkl', '_ROC_t3.pdf'))
            if classifier in ['DvT4']:
                plotROC(self.training.roc_td, self.validation.roc_td, plotName=modelPkl.replace('.pkl', '_ROC_td.pdf'))
                plotROC(self.training.roc_43, self.validation.roc_43, plotName=modelPkl.replace('.pkl', '_ROC_43.pdf'))
            plotClasses(self.training, self.validation, modelPkl.replace('.pkl', '.pdf'))
        
        else:
            print()

        if self.epoch in [5,10,15]:
            self.scheduler.step(1e6)
        else:
            self.scheduler.step(self.validation.loss)
        # if self.epoch == 5:
        #     print("Start using larger batches for training:",train_batch_size_small,"->",train_batch_size_large)
        #     self.trainLoader = self.training.largeBatchLoader
        # if self.validation.roc_auc_decreased > 1:# and self.training.trainLoader == self.training.largeBatchLoader:
        #     if self.training.trainLoader == self.training.largeBatchLoader and False:
        #         print("Start using smaller batches for training:", train_batch_size_large, "->", train_batch_size_small)
        #         self.training.trainLoader = self.training.smallBatchLoader
        #     else:
        #         print("Decrease learning rate:",self.optimizer.lr,"->",self.optimizer.lr/5)
        #         self.optimizer.lr = self.optimizer.lr/5
        #     self.validation.roc_auc_decreased = 0
        
    
    def dump(self):
        print(self.net)
        print(self.name)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('loss_best:',self.validation.loss_best)
        print('N trainable params:',self.nTrainableParameters)
        #print('useAncillary:',self.net.useAncillary)




#Simple ROC Curve plot function
def plotROC(train, val, plotName='test.pdf'): #fpr = false positive rate, tpr = true positive rate
    if classifier in ['ZHvB','ZZvB']:
        lumiRatio = 140/59.6
        S = val.tpr*sum_wS*lumiRatio
        S[S<(0.10*sum_wS*lumiRatio)] = 0 # require at least 10% signal acceptance 
        B = val.fpr*sum_wB*lumiRatio
        B[B<(0.001*sum_wB*lumiRatio)] = 1.0e9 # require at least 0.1% background acceptance 
        sigma = S / np.sqrt(S+B+2.5)
        iMaxSigma = np.argmax(sigma)
        maxSigma = sigma[iMaxSigma]

    f = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    ax.plot([0,1], [1,0], color='k', alpha=0.5, linestyle='--', linewidth=1)

    plt.xlabel('Rate( '+val.trueName+' to '+val.trueName+' )')
    plt.ylabel('Rate( '+val.falseName+' to '+val.falseName+' )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    ax.plot(train.tpr, 1-train.fpr, color='#d34031', linestyle='-', linewidth=1, alpha=1.0, label="Training")
    ax.plot(val  .tpr, 1-val  .fpr, color='#d34031', linestyle='-', linewidth=2, alpha=0.5, label="Validation")
    ax.legend(loc='lower left')
    ax.text(0.73, 1.07, "Validation AUC = %0.4f"%(val.auc))

    if classifier in ['ZHvB','ZZvB']:
        ax.scatter(rate_StoS, rate_BtoB, marker='o', c='k')
        ax.text(rate_StoS+0.03, rate_BtoB-0.100, ZB+"SR \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)
        tprMaxSigma, fprMaxSigma, thrMaxSigma = val.tpr[iMaxSigma], val.fpr[iMaxSigma], val.thr[iMaxSigma]
        ax.scatter(tprMaxSigma, (1-fprMaxSigma), marker='o', c='#d34031')
        ax.text(tprMaxSigma+0.03, (1-fprMaxSigma)-0.025, ("(%0.3f, %0.3f), "+classifier+" $>$ %0.2f \n $%1.2f\sigma$ with 140fb$^{-1}$")%(tprMaxSigma, (1-fprMaxSigma), thrMaxSigma, maxSigma), bbox=bbox)

    f.savefig(plotName)
    plt.close(f)

def plotClasses(train, valid, name):
    # Make place holder datasets to add the training/validation set graphical distinction to the legend
    trainLegend=pltHelper.dataSet(name=  'Training Set', color='black', alpha=1.0, linewidth=1)
    validLegend=pltHelper.dataSet(name='Validation Set', color='black', alpha=0.5, linewidth=2)

    for cl1 in classes: # loop over classes
        cl1cl2_args = {'dataSets': [trainLegend,validLegend],
                       'bins': [b/20.0 for b in range(-5,21)],
                       'xlabel': 'P( '+cl1.name+r' $\rightarrow$ Class )',
                       'ylabel': 'Arb. Units',
                       }
        cl2cl1_args = {'dataSets': [trainLegend,validLegend],
                       'bins': [b/20.0 for b in range(-5,21)],
                       'xlabel': r'P( Class $\rightarrow$ '+cl1.name+' )',
                       'ylabel': 'Arb. Units',
                       }
        reweight_args = {'dataSets': [trainLegend,validLegend],
                         'bins': [b/20.0 for b in range(-10,121)],
                         'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Multijet )',
                         'ylabel': 'Arb. Units',
                         }
        for cl2 in classes: # loop over classes
        # Make datasets to be plotted
            cl1cl2_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(train,'w'+cl1.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl1cl2_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(valid,'w'+cl1.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl1cl2_args['dataSets'] += [cl1cl2_valid, cl1cl2_train]

            cl2cl1_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl2.abbreviation+cl1.abbreviation), weights=getattr(train,'w'+cl2.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl2cl1_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl2.abbreviation+cl1.abbreviation), weights=getattr(valid,'w'+cl2.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl2cl1_args['dataSets'] += [cl2cl1_valid, cl2cl1_train]

        #multijet probabilities well defined but no multijet class labels. Therefore cl1cl2 plot can include multijet but not cl2cl1 plot.
        m4 = classInfo(abbreviation='m4', name= 'FourTag Multijet', color='blue')
        m3 = classInfo(abbreviation='m3', name='ThreeTag Multijet', color='violet')
        for cl2 in [m4,m3]:
            cl1cl2_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(train,'w'+cl1.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl1cl2_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(valid,'w'+cl1.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl1cl2_args['dataSets'] += [cl1cl2_train, cl1cl2_valid]

        reweight_train = pltHelper.dataSet(name=cl1.name, points=getattr(train,'r'+cl1.abbreviation), weights=getattr(train, 'w'+cl1.abbreviation)/train.w_sum, color=cl1.color, alpha=1.0, linewidth=1)
        reweight_valid = pltHelper.dataSet(               points=getattr(valid,'r'+cl1.abbreviation), weights=getattr(valid, 'w'+cl1.abbreviation)/valid.w_sum, color=cl1.color, alpha=0.5, linewidth=2)
        reweight_args['dataSets'] += [reweight_valid, reweight_train]

        #make the plotter
        cl1cl2 = pltHelper.histPlotter(**cl1cl2_args)
        cl2cl1 = pltHelper.histPlotter(**cl2cl1_args)
        reweight = pltHelper.histPlotter(**reweight_args)

        #remove the lines from the trainLegend/validLegend placeholders
        cl1cl2.artists[0].remove()
        cl1cl2.artists[1].remove()
        cl2cl1.artists[0].remove()
        cl2cl1.artists[1].remove()
        reweight.artists[0].remove()
        reweight.artists[1].remove()

        #save the pdf
        cl1cl2.savefig(name.replace('.pdf','_'+cl1.abbreviation+'_to_class.pdf'))
        cl2cl1.savefig(name.replace('.pdf','_class_to_'+cl1.abbreviation+'.pdf'))
        reweight.savefig(name.replace('.pdf','_reweight_'+cl1.abbreviation+'.pdf'))


    bm_vs_d4_args = {'dataSets': [trainLegend,validLegend],
                     'bins': [b/20.0 for b in range(-10,121)],
                     'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Multijet )',
                     'ylabel': 'Arb. Units',
                     }
    d4_train = pltHelper.dataSet(name=d4.name, points=train.rd4, weights=   train.wd4/train.w_sum, color=d4.color, alpha=1.0, linewidth=1)
    d4_valid = pltHelper.dataSet(              points=valid.rd4, weights=   valid.wd4/valid.w_sum, color=d4.color, alpha=0.5, linewidth=2)
    bm_train = pltHelper.dataSet(name='Background Model', 
                                 points=np.concatenate((train.rd3,train.rt3,train.rt4),axis=None), 
                                 weights=np.concatenate((train.wd3,-train.wt3,train.wt4)/train.w_sum,axis=None), 
                                 color='brown', alpha=1.0, linewidth=1)
    bm_valid = pltHelper.dataSet(points=np.concatenate((valid.rd3,valid.rt3,valid.rt4),axis=None), 
                                 weights=np.concatenate((valid.wd3,-valid.wt3,valid.wt4)/valid.w_sum,axis=None), 
                                 color='brown', alpha=0.5, linewidth=2)
    t4_train = pltHelper.dataSet(name=t4.name, points=train.rt4, weights= train.wt4/train.w_sum, color=t4.color, alpha=1.0, linewidth=1)
    t4_valid = pltHelper.dataSet(              points=valid.rt4, weights= valid.wt4/valid.w_sum, color=t4.color, alpha=0.5, linewidth=2)
    t3_train = pltHelper.dataSet(name=t3.name, points=train.rt3, weights=-train.wt3/train.w_sum, color=t3.color, alpha=1.0, linewidth=1)
    t3_valid = pltHelper.dataSet(              points=valid.rt3, weights=-valid.wt3/valid.w_sum, color=t3.color, alpha=0.5, linewidth=2)
    bm_vs_d4_args['dataSets'] += [d4_valid, d4_train, bm_valid, bm_train, t4_valid, t4_train, t3_valid, t3_train]
    bm_vs_d4 = pltHelper.histPlotter(**bm_vs_d4_args)
    bm_vs_d4.artists[0].remove()
    bm_vs_d4.artists[1].remove()
    bm_vs_d4.savefig(name.replace('.pdf','_bm_vs_d4.pdf'))


    rbm_vs_d4_args = {'dataSets': [trainLegend,validLegend],
                     'bins': [b/20.0 for b in range(-10,121)],
                     'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Multijet )',
                     'ylabel': 'Arb. Units',
                     }
    rbm_train = pltHelper.dataSet(name='Background Model', 
                                 points=np.concatenate((train.rd3,train.rt4),axis=None), 
                                 weights=np.concatenate((train.rd3*train.wd3,train.wt4)/train.w_sum,axis=None), 
                                 color='brown', alpha=1.0, linewidth=1)
    rbm_valid = pltHelper.dataSet(points=np.concatenate((valid.rd3,valid.rt4),axis=None), 
                                 weights=np.concatenate((valid.rd3*valid.wd3,valid.wt4)/valid.w_sum,axis=None), 
                                 color='brown', alpha=0.5, linewidth=2)
    rt3_train = pltHelper.dataSet(name=t3.name, points=train.rt3, weights=-train.rt3*train.wt3/train.w_sum, color=t3.color, alpha=1.0, linewidth=1)
    rt3_valid = pltHelper.dataSet(              points=valid.rt3, weights=-valid.rt3*valid.wt3/valid.w_sum, color=t3.color, alpha=0.5, linewidth=2)
    rbm_vs_d4_args['dataSets'] += [d4_valid, d4_train, rbm_valid, rbm_train, t4_valid, t4_train, rt3_valid, rt3_train]
    rbm_vs_d4 = pltHelper.histPlotter(**rbm_vs_d4_args)
    rbm_vs_d4.artists[0].remove()
    rbm_vs_d4.artists[1].remove()
    rbm_vs_d4.savefig(name.replace('.pdf','_rbm_vs_d4.pdf'))





model = modelParameters(args.model)

#model initial state
print("Setup training/validation tensors")
model.trainSetup(df_train, df_val)

# Training loop
for _ in range(args.epochs): 
    model.runEpoch()

print()
print(">> DONE <<")
if model.foundNewBest: print("Minimum Loss =", model.validation.loss_best)
