import time, os, sys
from copy import copy
import numpy as np
import pandas as pd
np.random.seed(0)#always pick the same training sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from sklearn.metrics import roc_curve, auc # pip/conda install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlibHelpers as pltHelper
from networks import *

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-b', '--background', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD1.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-s', '--signal',     default='/uscms/home/bryantp/nobackup/ZZ4b/bothZH4b2018/picoAOD0.h5', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-l', '--lrInit', default=1e-3, type=float, help='Initial learning rate.')
parser.add_argument('-p', '--pDropout', default=0.2, type=float, help='p(drop) for dropout.')
parser.add_argument('-c', '--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

n_queue = 20
train_batch_size_small =  32 #64 #32 #36
train_batch_size_large = 128
eval_batch_size = 16384
print_step = 100
train_fraction = 0.7

# Read .h5 files
dfB = pd.read_hdf(args.background, key='df')
dfS = pd.read_hdf(args.signal,     key='df')

# add y_true values to dataframes
dfB['y_true'] = pd.Series(np.zeros(dfB.shape[0], dtype=np.uint8), index=dfB.index)
dfS['y_true'] = pd.Series(np.ones( dfS.shape[0], dtype=np.uint8), index=dfS.index)

#select events in desired region for training/validation/test
dfB = dfB.loc[ (dfB['passHLT']==True) & (dfB['fourTag']==False) & ((dfB['ZHSB']==True)|(dfB['ZHCR']==True)|(dfB['ZHSR']==True)) ]
dfS = dfS.loc[ (dfS['passHLT']==True) & (dfS['fourTag']==True ) & ((dfS['ZHSB']==True)|(dfS['ZHCR']==True)|(dfS['ZHSR']==True)) ]
#dfB = dfB.loc[ (dfB['passHLT']==True) & (dfB['fourTag']==False) ]
#dfS = dfS.loc[ (dfS['passHLT']==True) & (dfS['fourTag']==True ) ]

nS      = dfS.shape[0]
nB      = dfB.shape[0]
print("nS",nS)
print("nB",nB)

# compute relative weighting for S and B
sum_wS = np.sum(np.float32(dfS['weight']))
sum_wB = np.sum(np.float32(dfB['weight']))
print("sum_wS",sum_wS)
print("sum_wB",sum_wB)

sum_wStoS = np.sum(np.float32(dfS.loc[dfS['ZHSR']==True ]['weight']))
sum_wBtoB = np.sum(np.float32(dfB.loc[dfB['ZHSR']==False]['weight']))
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
dfS['weight'] = dfS['weight']*sum_wB/sum_wS
dfS_train = dfS.iloc[idxS[:nTrainS]]
dfS_val   = dfS.iloc[idxS[nTrainS:]]
dfB_train = dfB.iloc[idxB[:nTrainB]]
dfB_val   = dfB.iloc[idxB[nTrainB:]]

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
        self.roc_auc= None
        self.roc_auc_prev=None
        self.roc_auc_decreased=0
        self.roc_auc_best = None


class modelParameters:
    def __init__(self, fileName=''):
        # self.fourVectors=['canJet0_pt', 'canJet1_pt', 'canJet2_pt', 'canJet3_pt',
        #                  'canJet0_eta', 'canJet1_eta', 'canJet2_eta', 'canJet3_eta',
        #                  'canJet0_phi', 'canJet1_phi', 'canJet2_phi', 'canJet3_phi',
        #                  'canJet0_e', 'canJet1_e', 'canJet2_e', 'canJet3_e',
        #                  ]
        #             |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        self.layer1Pix = "012302130312"
        #self.layer1Col = ['_pt', '_eta', '_phi', '_e']
        #self.fourVectors=[['canJet'+i+'_pt', 'canJet'+i+'_eta', 'canJet'+i+'_phi', 'canJet'+i+'_e'] for i in self.layer1Pix] #index[pixel][color]
        self.fourVectors=[['canJet'+i+'_pt', 'canJet'+i+'_eta', 'canJet'+i+'_phi'] for i in self.layer1Pix] #index[pixel][color]
        self.jetFeatures = len(self.fourVectors[0])
        self.ancillaryFeatures=['d01', 'd23', 'd02', 'd13', 'd03', 'd12', 'nSelJets', 'm4j', 'xWt1']#, 'xWt1']
        #self.nAncillaryFeatures = len(self.ancillaryFeatures)
        #self.useAncillaryROCAUCMin = 0.82
        #self.fourVectors=[['canJet'+jet+mu for jet in self.layer1Pix] for mu in self.layer1Col] #index[color][pixel]
        #self.fourVectors[color][pixel]
        self.validation = loaderResults("validation")
        self.training   = loaderResults("training")

        if fileName:
            self.dijetFeatures        = int(fileName.split('_')[2])
            self.quadjetFeatures      = int(fileName.split('_')[3])
            self.combinatoricFeatures = int(fileName.split('_')[4])
            self.nodes                = None#int(fileName.split('_')[5])
            self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_lr')]) if '_pdrop' in fileName else None
            self.lrInit        = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_auc')])
            self.validation.roc_auc_best  = float(fileName[fileName.find(   '_auc')+4 : fileName.find('.pkl')])
            self.scalers = torch.load(fileName)['scalers']

        else:
            self.dijetFeatures = 4
            self.quadjetFeatures = 4
            self.combinatoricFeatures = 20
            self.nodes = 128
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0           
            self.validation.roc_auc_best  = 0.87
            self.scalers = {}

        self.epoch = self.startingEpoch

        #self.net = basicCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        #self.net = dijetCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        self.net = ResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures).to(device)
        #self.net = PresResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        #self.net = deepResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        self.name = 'SvsB_'+self.net.name+'_lr%s_epochs%d_stdscale'%(str(self.lrInit), args.epochs+self.startingEpoch)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrInit, amsgrad=False)

        self.foundNewBest = False
        
        self.dump()

        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state

            if args.update:
                for sample in [args.background, args.signal]:
                    self.update(sample)
                exit()


    def epochString(self):
        return ('>> %'+str(len(str(args.epochs+self.startingEpoch)))+'d/%d <<')%(self.epoch, args.epochs+self.startingEpoch)

    def dfToTensors(self, df, y_true=None):
        n = df.shape[0]
        #Convert to list np array
        P=[np.float32(df[jet]) for jet in self.fourVectors]
        #make 3D tensor with correct axes [event][color][pixel] = [event][mu (4-vector component)][jet]
        P=torch.FloatTensor([np.float32([[P[jet][event][mu] for jet in range(len(self.fourVectors))] for mu in range(self.jetFeatures)]) for event in range(n)])
        #extra features for use with output of CNN layers
        A=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.ancillaryFeatures], 1 )

        if y_true:
            y=torch.FloatTensor( np.array(df[y_true], dtype=np.uint8).reshape(-1,1) )
        else:#assume all zero. y_true not needed for updating classifier output values in .h5 files for example.
            y=torch.FloatTensor( np.zeros(df.shape[0], dtype=np.uint8).reshape(-1,1) )

        w=torch.FloatTensor( np.float32(df['weight']).reshape(-1,1) )

        #print('P.shape, A.shape, y.shape, w.shape:', P.shape, A.shape, y.shape, w.shape)
        return P, A, y, w

    def update(self, fileName):
        print("Add classifier output to",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')
 
        n = df.shape[0]
        print("n",n)

        P, A, y, w = self.dfToTensors(df)
        print('P.shape', P.shape)

        for jet in range(P.shape[2]):
            P[:,:,jet] = torch.FloatTensor(scalers[0].transform(P[:,:,jet]))
        A = torch.FloatTensor(scalers['ancillary'].transform(A))

        # Set up data loaders
        dset   = TensorDataset(P, A, y, w)
        updateResults = loaderResults("update")
        updateResults.evalLoader = DataLoader(dataset=dset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        print('Batches:', len(updateResults.evalLoader))

        self.evaluate(updateResults)

        print(updateResults.y_pred)
        df['ZHvsBackgroundClassifier'] = pd.Series(updateResults.y_pred, index=df.index)
        print("df.dtypes")
        print(df.dtypes)
        print("df.shape", df.shape)
        df.to_hdf(fileName, key='df', format='table', mode='w')

        del dset
        del loader

    def trainSetup(self, df_train, df_val):
        P_train, A_train, y_train, w_train = self.dfToTensors(df_train, y_true='y_true')
        P_val  , A_val  , y_val  , w_val   = self.dfToTensors(df_val  , y_true='y_true')
        print('P_train.shape, A_train.shape, y_train.shape, w_train.shape:', P_train.shape, A_train.shape, y_train.shape, w_train.shape)
        print('P_val  .shape, A_val  .shape, y_val  .shape, w_val  .shape:', P_val  .shape, A_val  .shape, y_val  .shape, w_val  .shape)

        # Standardize inputs
        if not args.model:
            self.scalers[0] = StandardScaler()
            self.scalers[0].fit(P_train[:,:,1])
            self.scalers[0].scale_[1], self.scalers[0].mean_[1] =   2.5,  0 # eta max
            self.scalers[0].scale_[2], self.scalers[0].mean_[2] = np.pi,  0 # pi
            print("self.scalers[0].scale_",self.scalers[0].scale_)
            print("self.scalers[0].mean_",self.scalers[0].mean_)
            self.scalers['ancillary'] = StandardScaler()
            self.scalers['ancillary'].fit(A_train)
            self.scalers['ancillary'].scale_[0], self.scalers['ancillary'].mean_[0] = 100, 200
            self.scalers['ancillary'].scale_[1], self.scalers['ancillary'].mean_[1] = 100, 200
            self.scalers['ancillary'].scale_[2], self.scalers['ancillary'].mean_[2] = 100, 200
            self.scalers['ancillary'].scale_[3], self.scalers['ancillary'].mean_[3] = 100, 200
            self.scalers['ancillary'].scale_[4], self.scalers['ancillary'].mean_[4] = 100, 200
            self.scalers['ancillary'].scale_[5], self.scalers['ancillary'].mean_[5] = 100, 200
            self.scalers['ancillary'].scale_[6], self.scalers['ancillary'].mean_[6] =   3,   6
            print("self.scalers['ancillary'].scale_",self.scalers['ancillary'].scale_)
            print("self.scalers['ancillary'].mean_",self.scalers['ancillary'].mean_)

        for jet in range(P_train.shape[2]):
            P_train[:,:,jet] = torch.FloatTensor(self.scalers[0].transform(P_train[:,:,jet]))
            P_val  [:,:,jet] = torch.FloatTensor(self.scalers[0].transform(P_val  [:,:,jet]))
        A_train = torch.FloatTensor(self.scalers['ancillary'].transform(A_train))
        A_val   = torch.FloatTensor(self.scalers['ancillary'].transform(A_val))

        # Set up data loaders
        dset_train   = TensorDataset(P_train, A_train, y_train, w_train)
        dset_val     = TensorDataset(P_val,   A_val,   y_val,   w_val)
        self.training.smallBatchLoader = DataLoader(dataset=dset_train, batch_size=train_batch_size_small, shuffle=True,  num_workers=n_queue, pin_memory=True)
        self.training.largeBatchLoader = DataLoader(dataset=dset_train, batch_size=train_batch_size_large, shuffle=True,  num_workers=n_queue, pin_memory=True)
        self.training  .evalLoader     = DataLoader(dataset=dset_train, batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.validation.evalLoader     = DataLoader(dataset=dset_val,   batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.training .trainLoader     = self.training.largeBatchLoader

        #model initial state
        self.validate()
        print()
        self.validation.roc_auc = copy(self.validation.roc_auc)


    def evaluate(self, results):
        self.net.eval()
        y_pred, y_true, w_ordered = [], [], []
        for i, (P, A, y, w) in enumerate(results.evalLoader):
            P, A, y, w = P.to(device), A.to(device), y.to(device), w.to(device)
            logits = self.net(P, A)
            prob_pred = torch.sigmoid(logits)
            y_pred.append(prob_pred.tolist())
            y_true.append(y.tolist())
            w_ordered.append(w.tolist())
            if (i+1) % print_step == 0:
                sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(loader)))
                sys.stdout.flush()
                
        results.y_pred = np.transpose(np.concatenate(y_pred))[0]
        results.y_true = np.transpose(np.concatenate(y_true))[0]
        results.w      = np.transpose(np.concatenate(w_ordered))[0]
        results.fpr, results.tpr, results.thr = roc_curve(results.y_true, results.y_pred, sample_weight=results.w)
        results.roc_auc_prev = copy(results.roc_auc)
        results.roc_auc = auc(results.fpr, results.tpr)
        if results.roc_auc_prev:
            if results.roc_auc_prev > results.roc_auc: results.roc_auc_decreased += 1


    def validate(self):
        self.evaluate(self.validation)
        bar=int((self.validation.roc_auc-0.5)*200) if self.validation.roc_auc > 0.5 else 0
        print('\r'+self.epochString()+' ROC Validation: %2.1f%%'%(self.validation.roc_auc*100),("#"*bar)+"|", end = " ")


    def train(self):
        self.net.train()
        for i, (P, A, y, w) in enumerate(self.training.trainLoader):
            P, A, y, w = P.to(device), A.to(device), y.to(device), w.to(device)
            self.optimizer.zero_grad()
            logits = self.net(P, A)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=w) # binary classification
            loss.backward()
            self.optimizer.step()
            if (i+1) % print_step == 0:
                sys.stdout.write('\rTraining %3.0f%%     '%(float(i+1)*100/len(self.training.trainLoader)))
                sys.stdout.flush()

        self.evaluate(self.training)
        bar=int((self.training.roc_auc-0.5)*200) if self.training.roc_auc > 0.5 else 0
        print('\r'+' '*len(self.epochString())+'       Training: %2.1f%%'%(self.training.roc_auc*100),("-"*bar)+"|")


    def runEpoch(self):
        self.epoch += 1

        self.train()
        self.validate()

        if self.validation.roc_auc > self.validation.roc_auc_best:
            self.foundNewBest = True
            self.validation.roc_auc_best = copy(self.validation.roc_auc)
    
            modelPkl = 'ZZ4b/NtupleAna/pytorchModels/%s_epoch%d_auc%.4f.pkl'%(self.name, self.epoch, self.validation.roc_auc_best)
            print("*", modelPkl)
            plotROC(self.training,   modelPkl.replace('.pkl', '_ROC_train.pdf'))
            plotROC(self.validation, modelPkl.replace('.pkl', '_ROC_val.pdf'))
            plotNet(self.training,   modelPkl.replace('.pkl','_NetOutput_train.pdf'))
            plotNet(self.validation, modelPkl.replace('.pkl','_NetOutput_val.pdf'))
        
            model_dict = {'model': model.net.state_dict(), 'optim': model.optimizer.state_dict(), 'scalers': model.scalers}
            torch.save(model_dict, modelPkl)
        else:
            print("^ %1.1f%%"%((self.training.roc_auc-self.validation.roc_auc)*100))

        if self.validation.roc_auc_decreased > 1 and self.training.trainLoader == self.training.largeBatchLoader:
            print("Start using smaller batches for training:", train_batch_size_large, "->", train_batch_size_small)
            self.training.trainLoader = self.training.smallBatchLoader
            self.optimizer.lr = self.lrInit/5
        
    
    def dump(self):
        print(self.net)
        print(self.name)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('roc_auc_best:',self.validation.roc_auc_best)
        print('N trainable params:',sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        #print('useAncillary:',self.net.useAncillary)




#Simple ROC Curve plot function
def plotROC(results, name): #fpr = false positive rate, tpr = true positive rate
    lumiRatio = 10
    sigma = (results.tpr*sum_wS*lumiRatio) / np.sqrt(results.fpr*sum_wB*lumiRatio + 1)
    iMaxSigma = np.argmax(sigma)
    maxSigma = sigma[iMaxSigma]
    f = plt.figure()
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    plt.plot([0,1], [1,0], color='0.8', linestyle='--')

    plt.xlabel('Rate( Signal to Signal )')
    plt.ylabel('Rate( Background to Background )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    plt.plot(results.tpr, 1-results.fpr)
    plt.text(0.80, 1.07, "ROC AUC = %0.4f"%(results.roc_auc))

    plt.scatter(rate_StoS, rate_BtoB, marker='o', c='r')
    plt.text(rate_StoS+0.03, rate_BtoB-0.025, "Cut Based WP \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)

    tprMaxSigma, fprMaxSigma, thrMaxSigma = results.tpr[iMaxSigma], results.fpr[iMaxSigma], results.thr[iMaxSigma]
    plt.scatter(tprMaxSigma, (1-fprMaxSigma), marker='o', c='k')
    plt.text(tprMaxSigma+0.03, (1-fprMaxSigma)-0.025, "Optimal WP, SvB $>$ %0.2f \n (%0.2f, %0.2f), $%1.2f\sigma$ with 140fb$^{-1}$"%(thrMaxSigma, tprMaxSigma, (1-fprMaxSigma), maxSigma), bbox=bbox)

    f.savefig(name)
    plt.close(f)


def plotNet(results, name):
    yS_pred, yB_pred = results.y_pred[results.y_true==1], results.y_pred[results.y_true==0]
    wS,      wB      = results.w     [results.y_true==1], results.w     [results.y_true==0]
    fig = pltHelper.plot([yS_pred, yB_pred], 
                         [b/20.0 for b in range(21)],
                         "NN Output (SvB)", "Events / Bin", 
                         weights=[wS, wB],
                         samples=['Signal','Background'],
                         ratio=True,
                         ratioRange=[0,5])
    fig.savefig(name)
    plt.close(fig)


    
model = modelParameters(args.model)

#model initial state
model.trainSetup(df_train, df_val)

# Training loop
for _ in range(args.epochs): 
    model.runEpoch()

print()
print(">> DONE <<")
if model.foundNewBest: print("Best ROC AUC =", model.validation.roc_auc_best)
