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

class cycler:
    def __init__(self,options=['-','\\','|','/']):
        self.cycle=0
        self.options=options
        self.m=len(self.options)
    def next(self):
        self.cycle = (self.cycle + 1)%self.m
        return self.options[self.cycle]

loadCycler = cycler()

classifier = args.classifier

if classifier in ['ZHvB', 'ZZvB']:
    barMin=0.7
    barScale=400
    signalName='Signal'
    backgroundName='Background'
    weight = 'weight'
    yTrueLabel = 'y_true'
    ZB = ''
    for fileName in glob(args.signal):
        if "ZH4b" in fileName: 
            print("Signal is ZH")
            ZB="ZH"
        else: 
            print("Signal is ZZ")
            ZB="ZZ"
 
    if not args.update:
        train_fraction = 0.7
        train_batch_size_small =  64
        train_batch_size_large = 256

        # Read .h5 files
        frames = []
        for fileName in glob(args.data):
            print("Reading",fileName)
            frames.append(pd.read_hdf(fileName, key='df'))
        dfB = pd.concat(frames, sort=False)
            
        frames = []
        for fileName in glob(args.signal):
            print("Reading",fileName)
            frames.append(pd.read_hdf(fileName, key='df'))
        dfS = pd.concat(frames, sort=False)
        classifier = ZB+'vB'

        #select events in desired region for training/validation/test
        dfB = dfB.loc[ (dfB['passHLT']==True) & (dfB['fourTag']==False) & ((dfB[ZB+'SB']==True)|(dfB[ZB+'CR']==True)|(dfB[ZB+'SR']==True)) ]
        dfS = dfS.loc[ (dfS['passHLT']==True) & (dfS['fourTag']==True ) & ((dfS[ZB+'SB']==True)|(dfS[ZB+'CR']==True)|(dfS[ZB+'SR']==True)) ]
        #dfB = dfB.loc[ (dfB['passHLT']==True) & (dfB['fourTag']==False) ]
        #dfS = dfS.loc[ (dfS['passHLT']==True) & (dfS['fourTag']==True ) ]

        # add y_true values to dataframes
        dfB['y_true'] = pd.Series(np.zeros(dfB.shape[0], dtype=np.uint8), index=dfB.index)
        dfS['y_true'] = pd.Series(np.ones( dfS.shape[0], dtype=np.uint8), index=dfS.index)

        nS      = dfS.shape[0]
        nB      = dfB.shape[0]
        print("nS",nS)
        print("nB",nB)

        # compute relative weighting for S and B
        sum_wS = np.sum(np.float32(dfS['weight']))
        sum_wB = np.sum(np.float32(dfB['weight']))
        print("sum_wS",sum_wS)
        print("sum_wB",sum_wB)

        sum_wStoS = np.sum(np.float32(dfS.loc[dfS[ZB+'SR']==True ]['weight']))
        sum_wBtoB = np.sum(np.float32(dfB.loc[dfB[ZB+'SR']==False]['weight']))
        print("sum_wStoS",sum_wStoS)
        print("sum_wBtoB",sum_wBtoB)
        rate_StoS = sum_wStoS/sum_wS
        rate_BtoB = sum_wBtoB/sum_wB
        print("Cut Based WP:",rate_StoS,"Signal Eff.", rate_BtoB,"1-Background Eff.")

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
        dfS['weight'] = dfS['weight']*sum_wB/sum_wS #normalize signal to background
        dfS_train = dfS.iloc[idxS[:nTrainS]]
        dfS_val   = dfS.iloc[idxS[nTrainS:]]
        dfB_train = dfB.iloc[idxB[:nTrainB]]
        dfB_val   = dfB.iloc[idxB[nTrainB:]]
    
        df_train = pd.concat([dfB_train, dfS_train], sort=False)
        df_val   = pd.concat([dfB_val,   dfS_val  ], sort=False)

if classifier in ['FvT', 'M1vM2']:
    barMin = 0.55
    barScale=1000
    if classifier == 'M1vM2': barMin, barScale = 0.5, 500
    signalName='FourTag'
    backgroundName='ThreeTag'
    weight = 'weight'
    yTrueLabel = 'fourTag'
    if classifier in ['M1vM2']: yTrueLabel = 'y_true'
    if classifier == 'M1vM2':
        signalName, backgroundName = 'Mixed', 'Unmixed'
        weight = 'weight'
    ZB = ''

    if not args.update:
        train_numerator = 7
        train_denominator = 10
        train_fraction = 7/10
        train_offset = 0
        train_batch_size_small =  64
        # train_batch_size_small = 128
        # train_batch_size_large = 256
        # Read .h5 files
        frames = []
        for fileName in glob(args.data):
            print("Reading",fileName)
            frames.append(pd.read_hdf(fileName, key='df'))
        dfD = pd.concat(frames, sort=False)

        if classifier in ['M1vM2']:
            frames = []
            for fileName in glob(args.signal):
                print("Reading",fileName)
                frames.append(pd.read_hdf(fileName, key='df'))
            dfS = pd.concat(frames, sort=False)

        # keep events passing trigger and in region of interest
        if classifier in ['FvT']:
            print("Splitting data into three and four tag selections")
            dfDB = dfD.loc[ (dfD['passHLT']==True) & (dfD['fourTag']==False) & (dfD[ZB+'SB']==True) ]
            dfDS = dfD.loc[ (dfD['passHLT']==True) & (dfD['fourTag']==True ) & (dfD[ZB+'SB']==True) ]
        if classifier in ['M1vM2']:
            print("Add y_true values to dataframes")
            dfD['y_true'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
            dfS['y_true'] = pd.Series(np.ones( dfS.shape[0], dtype=np.uint8), index=dfS.index)            
            print("M1vM2 definitions of dfDB, dfDS")
            dfDB = dfD.loc[ (dfD['passHLT']==True) & (dfD['fourTag']==False) ]#& ((dfD[ZB+'SB']==True) | (dfD[ZB+'CR']==True) | (dfD[ZB+'SR']==True)) ]
            dfDS = dfS.loc[ (dfS['passHLT']==True) & (dfS['fourTag']==False) ]#& ((dfS[ZB+'SB']==True) | (dfS[ZB+'CR']==True) | (dfS[ZB+'SR']==True)) ]

        nDS, nDB = dfDS.shape[0], dfDB.shape[0]
        print("nDS, nDB",nDS,nDB)

        if args.ttbar:
            frames = []
            for fileName in glob(args.ttbar):
                print("Reading",fileName)
                frames.append(pd.read_hdf(fileName, key='df'))
            dfT = pd.concat(frames, sort=False)
            print("Multiply ttbar weights by -1")
            dfT[weight] = dfT[weight] * -1

            #select events in desired region for training/validation/test
            print("Splitting ttbar into three and four tag selections")
            dfTB = dfT.loc[ (dfT['passHLT']==True) & (dfT['fourTag']==False) & (dfT[ZB+'SB']==True) ]
            dfTS = dfT.loc[ (dfT['passHLT']==True) & (dfT['fourTag']==True ) & (dfT[ZB+'SB']==True) ]

            nTS, nTB = dfTS.shape[0], dfTB.shape[0]

            dfB = pd.concat([dfDB, dfTB], sort=False)
            dfS = pd.concat([dfDS, dfTS], sort=False)

            nB = nDB + nTB
            nS = nDS + nTS

        else:
            print("WARNING: No ttbar sample specified")
            dfB = dfDB
            dfS = dfDS
            
            nB = nDB
            nS = nDS
            
            
        train_batch_size_large = 20*nB//nS

        # compute relative weighting for S and B
        sum_wS = np.sum(np.float32(dfS[weight]))
        sum_wB = np.sum(np.float32(dfB[weight]))
        print("sum_wS",sum_wS)
        print("sum_wB",sum_wB)

        #
        # Split into training and validation sets
        #
        idxS_train, idxS_val = [], []
        idxB_train, idxB_val = [], []
        print("build idxS with offset %i, modulus %i, and train/val split %i"%(train_offset, train_denominator, train_numerator))
        for e in range(nS):
            if (e+train_offset)%train_denominator < train_numerator: 
                idxS_train.append(e)
            else:
                idxS_val  .append(e)
        print("build idxB with offset %i, modulus %i, and train/val split %i"%(train_offset, train_denominator, train_numerator))
        for e in range(nB):
            if (e+train_offset)%train_denominator < train_numerator: 
                idxB_train.append(e)
            else:
                idxB_val  .append(e)
        idxS_train, idxS_val = np.array(idxS_train), np.array(idxS_val)
        idxB_train, idxB_val = np.array(idxB_train), np.array(idxB_val)

        print("Split into training and validation sets")
        dfS_train, dfS_val = dfS.iloc[idxS_train], dfS.iloc[idxS_val]
        dfB_train, dfB_val = dfB.iloc[idxB_train], dfB.iloc[idxB_val]

        df_train = pd.concat([dfB_train, dfS_train], sort=False)
        df_val   = pd.concat([dfB_val,   dfS_val  ], sort=False)


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

class loaderResults:
    def __init__(self, name):
        self.name = name
        self.trainLoader = None
        self. evalLoader = None
        self.smallBatchLoader = None
        self.largeBatchLoader = None
        self.y_true = None
        self.y_pred = None
        self.w      = None
        self.fpr    = None
        self.tpr    = None
        self.thr    = None
        self.roc_auc= 0
        self.roc_auc_prev=None
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
        self.probNormRatio_BtoS = None

    def update(self, y_pred, y_true, w_ordered, doROC=False):
        self.y_pred = np.transpose(np.concatenate(y_pred))[0]
        self.y_true = np.transpose(np.concatenate(y_true))[0]
        self.w      = np.transpose(np.concatenate(w_ordered))[0]
        self.y_pred_S = self.y_pred[self.y_true==1]
        self.y_pred_B = self.y_pred[self.y_true==0]
        self.w_S = self.w[self.y_true==1]
        self.w_B = self.w[self.y_true==0]
        self.sum_w_S = np.sum(self.w_S)
        self.sum_w_B = np.sum(self.w_B)
        self.probNorm_StoB = np.sum( self.w_S * (1-self.y_pred_S)/self.y_pred_S )
        self.probNorm_BtoS = np.sum( self.w_B * self.y_pred_B/(1-self.y_pred_B) )
        self.probNormRatio_StoB = self.sum_w_B/self.probNorm_StoB
        self.probNormRatio_BtoS = self.sum_w_S/self.probNorm_BtoS
        if doROC:
            self.fpr, self.tpr, self.thr = roc_curve(self.y_true, self.y_pred, sample_weight=self.w)
            self.fpr = np.sort(self.fpr, kind='mergesort')
            self.tpr = np.sort(self.tpr, kind='mergesort')
            self.roc_auc_prev = copy(self.roc_auc)
            self.roc_auc = auc(self.fpr, self.tpr)
            if self.roc_auc_prev:
                if self.roc_auc_prev > self.roc_auc: self.roc_auc_decreased += 1


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
        if classifier in ['M1vM2']: self.ancillaryFeatures[1] = 'xWt1'
        #if classifier == "FvT":   self.ancillaryFeatures += ['stNotCan', 'xWt1', 'aveAbsEtaOth', 'nPVsGood']#, 'dRjjClose', 'dRjjOther', 'aveAbsEtaOth']#, 'nPSTJets']
        #if classifier == "FvT":   self.ancillaryFeatures += ['xWt']#, 'dRjjClose', 'dRjjOther', 'aveAbsEtaOth']#, 'nPSTJets']
        #if classifier == ZB+"vB": self.ancillaryFeatures += ['xWt']#, 'nPSTJets']
        self.useOthJets = ''
        if True: self.useOthJets = 'multijetAttention'

        self.validation = loaderResults("validation")
        self.training   = loaderResults("training")

        if fileName:
            self.classifier           =     fileName.split('_')[0]
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
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_auc')])
            self.validation.roc_auc_best  = float(fileName[fileName.find(   '_auc')+4 : fileName.find('.pkl')])
            self.scalers = torch.load(fileName)['scalers']

        else:
            self.dijetFeatures = 8
            self.quadjetFeatures = 8
            self.combinatoricFeatures = 8 #ZZ4b/nTupleAnalysis/pytorchModels/FvT_ResNet+LSTM_8_6_8_np2409_lr0.001_epochs20_stdscale_epoch9_auc0.5934.pkl
            self.nodes         = args.nodes
            self.layers        = args.layers
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0           
            self.validation.roc_auc_best  = 0.87 if args.signal else 0.583
            if classifier in ['M1vM2']: self.validation.roc_auc_best = 0.5
            self.scalers = {}

        self.epoch = self.startingEpoch

        #self.net = basicCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        #self.net = dijetCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        self.net = ResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, len(self.ancillaryFeatures), self.useOthJets, device=device).to(device)
        #self.net.debug=True
        #self.net = ResNetZero(self.jetFeatures, len(self.dijetAncillaryFeatures)//6, len(self.quadjetAncillaryFeatures)//3, len(self.ancillaryFeatures), self.useOthJets).to(device)
        #self.net = basicDNN(len(self.xVariables), self.layers, self.nodes, self.pDropout).to(device)
        #self.net = PhiInvResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures).to(device)
        #self.net = PresResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        #self.net = deepResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        self.nTrainableParameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.name = classifier+'_'+self.net.name+'_np%d_lr%s_epochs%d_stdscale'%(self.nTrainableParameters, str(self.lrInit), args.epochs+self.startingEpoch)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrInit, amsgrad=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.2, threshold=0.0001, threshold_mode='rel', patience=1, cooldown=2, min_lr=0, verbose=True)

        self.foundNewBest = False
        
        self.dump()

        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state
            self.optimizer.load_state_dict(torch.load(fileName)['optimizer'])

            if args.update:
                for sample in [args.data, args.ttbar, args.signal]:
                    for sampleFile in glob(sample):
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
            y=torch.FloatTensor( np.array(df[y_true], dtype=np.uint8).reshape(-1,1) )
        else:#assume all zero. y_true not needed for updating classifier output values in .h5 files for example.
            y=torch.FloatTensor( np.zeros(df.shape[0], dtype=np.uint8).reshape(-1,1) )

        w=torch.FloatTensor( np.float32(df[weight]).reshape(-1,1) )

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
        print('Batches:', len(updateResults.evalLoader))

        self.evaluate(updateResults, doROC = False)

        print(updateResults.y_pred)
        df[classifier] = pd.Series(np.float32(updateResults.y_pred), index=df.index)
        print("df.dtypes")
        print(df.dtypes)
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
        #self.training.smallBatchLoader = DataLoader(dataset=dset_train, batch_size=train_batch_size_small, shuffle=True,  num_workers=n_queue, pin_memory=True)
        self.training.largeBatchLoader = DataLoader(dataset=dset_train, batch_size=train_batch_size_large, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True)
        self.training  .evalLoader     = DataLoader(dataset=dset_train, batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.validation.evalLoader     = DataLoader(dataset=dset_val,   batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.training .trainLoader     = self.training.largeBatchLoader
        print("Training Batch Size:",train_batch_size_large)
        print("Training Batches:",len(self.training.trainLoader))

        #model initial state
        epochSpaces = max(len(str(args.epochs))-2, 0)
        print(">> "+(epochSpaces*" ")+"Epoch"+(epochSpaces*" ")+" <<   Data Set | Norm | % AUC | AUC Bar Graph ^ ROC % ABC * Output Model")
        self.validate(doROC=True)
        print()
        self.scheduler.step(self.validation.roc_auc)
        #self.validation.roc_auc_prev = copy(self.validation.roc_auc)


    def evaluate(self, results, doROC=True):
        self.net.eval()
        y_pred, y_true, w_ordered = [], [], []
        print_step = len(results.evalLoader)//200+1
        for i, (X, P, O, D, Q, A, y, w) in enumerate(results.evalLoader):
            X, P, O, D, Q, A, y, w = X.to(device), P.to(device), O.to(device), D.to(device), Q.to(device), A.to(device), y.to(device), w.to(device)
            logits = self.net(X, P, O, D, Q, A)
            prob_pred = torch.sigmoid(logits)
            y_pred.append(prob_pred.tolist())
            y_true.append(y.tolist())
            w_ordered.append(w.tolist())
            if int(i+1) % print_step == 0:
                percent = float(i+1)*100/len(results.evalLoader)
                sys.stdout.write('\rEvaluating %3.0f%%     '%(percent))
                sys.stdout.flush()
                
        results.update(y_pred, y_true, w_ordered, doROC)

    def validate(self, doROC=True):
        self.evaluate(self.validation, doROC)
        bar=int((self.validation.roc_auc-barMin)*barScale) if self.validation.roc_auc > barMin else 0
        #overtrain="^ %1.1f%%"%((self.training.roc_auc-self.validation.roc_auc)*100/(self.training.roc_auc - 0.5)) if self.training.roc_auc else ""
        roc_abc=None
        overtrain=""
        if self.training.roc_auc: 
            roc_val = interpolate.interp1d(self.validation.fpr, self.validation.tpr, fill_value="extrapolate")
            tpr_val = roc_val(self.training.fpr)#validation tpr estimated at training fpr
            roc_abc = auc(self.training.fpr, np.abs(self.training.tpr-tpr_val)) #area between curves
            #roc_abc = auc(self.training.fpr, (self.training.tpr-tpr_val)**2)**0.5 #quadrature sum of area between curves. Points where curves are close contribute very little to this metric of overtraining
            overtrain="^ %1.1f%%"%(roc_abc*100/(self.training.roc_auc-0.5))
        print('\r'+self.epochString()+' Validation | %0.2f | %2.2f'%(self.validation.probNormRatio_BtoS, self.validation.roc_auc*100),"|"+("#"*bar)+"|",overtrain, end = " ")


    def train(self):
        self.net.train()
        #if self.epoch == 5: self.net.debug=True
        print_step = len(self.training.trainLoader)//200+1
        accuracy = 0.0
        #totalLoss = 0
        for i, (X, P, O, D, Q, A, y, w) in enumerate(self.training.trainLoader):
            X, P, O, D, Q, A, y, w = X.to(device), P.to(device), O.to(device), D.to(device), Q.to(device), A.to(device), y.to(device), w.to(device)
            self.optimizer.zero_grad()
            logits = self.net(X, P, O, D, Q, A)
            #prob_pred = torch.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=w) # binary classification
            loss.backward()
            self.optimizer.step()
            #totalLoss += loss.item()
            binary_pred = logits.ge(0.).byte()
            binary_result = binary_pred.eq(y.byte()).float()*w
            accuracy += binary_result.sum().item()/sum(w)
            if (i+1) % print_step == 0:
                #l = totalLoss/(i+1)
                a = accuracy/(i+1)
                bar=int((a-barMin)*barScale) if a > barMin else 0
                sys.stdout.write(str(('\rTraining %3.0f%% ('+loadCycler.next()+') Accuracy: %2.1f%%')%(float(i+1)*100/len(self.training.trainLoader), a*100))+' '+('-'*bar)+"|")#+str(' <loss> = %f'%l))
                sys.stdout.flush()

        self.evaluate(self.training)
        sys.stdout.write(' '*200)
        sys.stdout.flush()
        bar=int((self.training.roc_auc-barMin)*barScale) if self.training.roc_auc > barMin else 0
        print('\r'+' '*len(self.epochString())+'   Training | %0.2f | %2.2f'%(self.training.probNormRatio_BtoS, self.training.roc_auc*100),"|"+("-"*bar)+"|")


    def runEpoch(self):
        self.epoch += 1

        self.train()
        self.validate()

        if self.validation.roc_auc > self.validation.roc_auc_best:
            self.foundNewBest = True
            self.validation.roc_auc_best = copy(self.validation.roc_auc)
    
            modelPkl = 'ZZ4b/nTupleAnalysis/pytorchModels/%s_epoch%d_auc%.4f.pkl'%(self.name, self.epoch, self.validation.roc_auc_best)
            print("*", modelPkl)
            plotROC(self.training, self.validation, modelPkl.replace('.pkl', '_ROC.pdf'))
            #plotROC(self.validation, modelPkl.replace('.pkl', '_ROC_val.pdf'))
            plotNet(self.training, self.validation, modelPkl.replace('.pkl','_NetOutput.pdf'))
            #plotNet(self.validation, modelPkl.replace('.pkl','_NetOutput_val.pdf'))
        
            model_dict = {'model': model.net.state_dict(), 'optimizer': model.optimizer.state_dict(), 'scalers': model.scalers}
            torch.save(model_dict, modelPkl)
        else:
            print()

        self.scheduler.step(self.validation.roc_auc)
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
        print('roc_auc_best:',self.validation.roc_auc_best)
        print('N trainable params:',self.nTrainableParameters)
        #print('useAncillary:',self.net.useAncillary)




#Simple ROC Curve plot function
def plotROC(train, val, name): #fpr = false positive rate, tpr = true positive rate
    lumiRatio = 140/59.6
    S = val.tpr*sum_wS*lumiRatio
    B = val.fpr*sum_wB*lumiRatio + 2.5
    sigma = S / np.sqrt(S+B)
    iMaxSigma = np.argmax(sigma)
    maxSigma = sigma[iMaxSigma]
    f = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    ax.plot([0,1], [1,0], color='k', alpha=0.5, linestyle='--', linewidth=1)

    plt.xlabel('Rate( '+signalName+' to '+signalName+' )')
    plt.ylabel('Rate( '+backgroundName+' to '+backgroundName+' )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    ax.plot(train.tpr, 1-train.fpr, color='#d34031', linestyle='-', linewidth=1, alpha=1.0, label="Training")
    ax.plot(val  .tpr, 1-val  .fpr, color='#d34031', linestyle='-', linewidth=2, alpha=0.5, label="Validation")
    ax.legend(loc='lower left')
    ax.text(0.73, 1.07, "Validation AUC = %0.4f"%(val.roc_auc))

    if rate_StoS and rate_BtoB:
        ax.scatter(rate_StoS, rate_BtoB, marker='o', c='k')
        ax.text(rate_StoS+0.03, rate_BtoB-0.025, "Cut Based WP \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)

        tprMaxSigma, fprMaxSigma, thrMaxSigma = val.tpr[iMaxSigma], val.fpr[iMaxSigma], val.thr[iMaxSigma]
        ax.scatter(tprMaxSigma, (1-fprMaxSigma), marker='o', c='#d34031')
        ax.text(tprMaxSigma+0.03, (1-fprMaxSigma)-0.025, "Optimal WP, "+classifier+" $>$ %0.2f \n (%0.2f, %0.2f), $%1.2f\sigma$ with 140fb$^{-1}$"%(thrMaxSigma, tprMaxSigma, (1-fprMaxSigma), maxSigma), bbox=bbox)

    f.savefig(name)
    plt.close(f)


def plotNet(train, val, name):
    orange='#ef8636'
    blue='#3b75af'
    yS_val  , yB_val   = val  .y_pred[val  .y_true==1], val  .y_pred[val  .y_true==0]
    wS_val  , wB_val   = val  .w     [val  .y_true==1], val  .w     [val  .y_true==0]
    sumW_val   = np.sum(wS_val  )+np.sum(wB_val  )
    yS_train, yB_train = train.y_pred[train.y_true==1], train.y_pred[train.y_true==0]
    wS_train, wB_train = train.w     [train.y_true==1], train.w     [train.y_true==0]
    sumW_train = np.sum(wS_train)+np.sum(wB_train)
    wS_val  , wB_val   = wS_val  /sumW_val  , wB_val  /sumW_val
    wS_train, wB_train = wS_train/sumW_train, wB_train/sumW_train
    fig = pltHelper.plot([yS_val, yB_val, yS_train, yB_train], 
                         [b/20.0 for b in range(21)],
                         "NN Output ("+classifier+")", "Arb. Units", 
                         weights=[wS_val, wB_val, wS_train, wB_train],
                         samples=["Validation "+signalName,"Validation "+backgroundName,"Training "+signalName,"Training "+backgroundName],
                         colors=[blue,orange,blue,orange],
                         alphas=[0.5,0.5,1,1],
                         linews=[2,2,1,1],
                         ratio=True,
                         ratioTitle=signalName+' / '+backgroundName,
                         ratioRange=[0,5])
    fig.savefig(name)
    plt.close(fig)


    
model = modelParameters(args.model)

#model initial state
print("Setup training/validation tensors")
model.trainSetup(df_train, df_val)

# Training loop
for _ in range(args.epochs): 
    model.runEpoch()

print()
print(">> DONE <<")
if model.foundNewBest: print("Best ROC AUC =", model.validation.roc_auc_best)
