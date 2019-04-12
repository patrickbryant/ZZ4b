import time, os, sys
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
import matplotlib.pyplot as plt
import matplotlibHelpers as pltHelper
class Lin_View(nn.Module):
    def __init__(self):
        super(Lin_View, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

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


class modelParameters:
    def __init__(self, fileName=''):
        # self.xVariables=['canJet0_pt', 'canJet1_pt', 'canJet2_pt', 'canJet3_pt',
        #                  'canJet0_eta', 'canJet1_eta', 'canJet2_eta', 'canJet3_eta',
        #                  'canJet0_phi', 'canJet1_phi', 'canJet2_phi', 'canJet3_phi',
        #                  'canJet0_e', 'canJet1_e', 'canJet2_e', 'canJet3_e',
        #                  ]
        #             |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        self.layer1 = "012302130312"
        self.xVariables=[['canJet'+i+'_pt', 'canJet'+i+'_eta', 'canJet'+i+'_phi', 'canJet'+i+'_e'] for i in self.layer1]
        if fileName:
            self.dijetFeatures = 12
            self.quadjetFeatures = 12
            self.combinatoricFeatures = 12
            self.nodes = 12
            self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_lr')])
            self.lrInit        = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_auc')])
            self.roc_auc_best  = float(fileName[fileName.find(   '_auc')+4 : fileName.find('.pkl')])
            self.scalers = torch.load(fileName)['scalers']

        else:
            self.dijetFeatures = 12
            self.quadjetFeatures = 24
            self.combinatoricFeatures = 48
            self.nodes = 128
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0
            self.roc_auc_best  = 0.5#0.7365461288230212 --layers 3 -n 128 with 1.6 scale factor, epoch 74 #0.7341273506654001 --layers 5, epoch 58 #0.7338770736660832 --layers 4, epoch 60 #0.7384488659621641 # -l 5e-4 -p 0.2, epoch 70
            self.scalers = {}

        #self.name = 'SvsB_FC%dx%d_pdrop%.2f_lr%s_epochs%d_stdscale'%(self.nodes, self.layers, self.pDropout, str(self.lrInit), args.epochs+self.startingEpoch)
        self.name = 'SvsB_CNN_%d_%d_%d_%d_pdrop%.2f_lr%s_epochs%d_stdscale'%(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout, str(self.lrInit), args.epochs+self.startingEpoch)

        # # Set up NN model and optimizer
        # self.dump()
        # self.fc = []
        # self.fc.append(nn.Linear(len(self.xVariables), self.nodes))
        # self.fc.append(nn.ReLU())
        # self.fc.append(nn.Dropout(p=self.pDropout))
        # outNodes = self.nodes
        # for i in range(self.layers):
        #     inNodes  = outNodes
        #     outNodes = int(inNodes/2)
        #     print(inNodes,"->",outNodes)
        #     self.fc.append(nn.Linear(inNodes, outNodes))
        #     self.fc.append(nn.ReLU())
        #     self.fc.append(nn.Dropout(p=self.pDropout))
        # inNodes  = outNodes
        # outNodes = 1
        # print(inNodes,"->",outNodes)
        # self.fc.append(nn.Linear(inNodes, outNodes))
        # self.fcnet = nn.Sequential(*self.fc).to(device)

        # colors are 4-vector components. 4-vectors are arranged in ID "image"
        # | 2 | 1 | 3 | 1 | 4 | 3 | 2 | 4 | 1 |  ##This order ensures kernel size 2 gets all possible dijets. 
        #   |1,2|1,3|1,3|1,4|3,4|2,3|2,4|1,4|    ##The dijets |1,3| and |1,4| are repeated
        #    |1,2,3| |1,3,4| |2,3,4| |1,2,4|     ##All four trijets are represented
        #       |1,2,3,4|1,2,3,4|1,2,3,4|        ##The quadjet is represented three times

        #
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  ##kernel=3 -> DNN

        self.dump()
        self.fc = []

        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        self.fc.append(nn.Conv1d(4, self.dijetFeatures, 2, stride=2))
        self.fc.append(nn.ReLU())
        #self.fc.append(nn.Dropout(p=self.pDropout))

        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        self.fc.append(nn.Conv1d(self.dijetFeatures, self.quadjetFeatures, 2, stride=2))
        self.fc.append(nn.ReLU())
        #self.fc.append(nn.Dropout(p=self.pDropout))

        # |1,2,3,4|1,2,3,4|1,2,3,4|  ##kernel=3
        self.fc.append(nn.Conv1d(self.quadjetFeatures, self.combinatoricFeatures, 3))
        self.fc.append(Lin_View())
        self.fc.append(nn.ReLU())
        #self.fc.append(nn.Dropout(p=self.pDropout))

        # DNN for S vs B classification 
        self.fc.append(nn.Linear(self.combinatoricFeatures, self.nodes))
        self.fc.append(nn.ReLU())
        #self.fc.append(nn.Dropout(p=self.pDropout))
        self.fc.append(nn.Linear(self.nodes, self.nodes))
        self.fc.append(nn.ReLU())
        self.fc.append(nn.Dropout(p=self.pDropout))
        self.fc.append(nn.Linear(self.nodes, self.nodes))
        self.fc.append(nn.ReLU())
        self.fc.append(nn.Dropout(p=self.pDropout))
        self.fc.append(nn.Linear(self.nodes, 1))
        self.fcnet = nn.Sequential(*self.fc).to(device)
        print(self.fcnet)

        if fileName:
            print("Load Model:", fileName)
            self.fcnet.load_state_dict(torch.load(fileName)['model']) # load model from previous state
    
    def dump(self):
        print(self.name)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('roc_auc_best:',self.roc_auc_best)


# Run on gpu if available
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
print('torch.cuda.is_available()',torch.cuda.is_available())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device",device,torch.cuda.device_count(),torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU:",device)

model = modelParameters(args.model)

n_queue = 10
batch_size = 32 #36
foundNewBest = False
print_step = 100
train_fraction = 0.5

if args.model and args.update:
    for fileName in [args.background, args.signal]:
        print("Add classifier output to",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')

        n = df.shape[0]
        print("n",n)

        X = torch.FloatTensor( [np.float32(df[jet]) for jet in model.xVariables] ).resize_([n, 4, len(model.xVariables)])
        y = np.zeros(n, dtype=np.uint8).reshape(-1,1)
        print('X.shape', X.shape)

        for i in range(X.shape[2]):
            X[:,:,i] = torch.FloatTensor(model.scalers[i].transform(X[:,:,i]))

        # Set up data loaders
        batch_size = 2048
        dset   = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        print('Batches:', len(loader))

        model.fcnet.eval()
        y_pred = []
        for i, (X, y) in enumerate(loader):
            X = X.to(device)
            logits = model.fcnet(X).view(-1,1)
            binary_pred = logits.ge(0.).byte()
            prob_pred = torch.sigmoid(logits)
            y_pred.append(prob_pred.tolist())
            if (i+1) % 1000 == 0:
                print('Evaluate Batch %d/%d'%(i+1, len(loader)))

        y_pred = np.float32(np.concatenate(y_pred).reshape(1,df.shape[0])[0])
        print(y_pred)
        df['ZHvsBackgroundClassifier'] = pd.Series(y_pred, index=df.index)
        print("df.dtypes")
        print(df.dtypes)
        print("df.shape", df.shape)
        df.to_hdf(fileName, key='df', format='table', mode='w')

        del dset
        del loader

    exit()

# Read .h5 file
dfB = pd.read_hdf(args.background, key='df')
dfS = pd.read_hdf(args.signal,     key='df')

#select events in desired region for training/validation/test
dfB = dfB.loc[ (dfB['fourTag']==False) & ((dfB['ZHSB']==True)|(dfB['ZHCR']==True)|(dfB['ZHSR']==True)) & (dfB['passDEtaBB']==True) ]
dfS = dfS.loc[ (dfS['fourTag']==True ) & ((dfS['ZHSB']==True)|(dfS['ZHCR']==True)|(dfS['ZHSR']==True)) & (dfS['passDEtaBB']==True) ]

print("dfS.shape",dfS.shape)

nS      = dfS.shape[0]
nB      = dfB.shape[0]
print("nS",nS)
print("nB",nB)
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

# compute relative weighting for S and B
sum_wS = np.sum(np.float32(dfS['weight']))
sum_wB = np.sum(np.float32(dfB['weight']))
print("sum_wS",sum_wS)
print("sum_wB",sum_wB)

sum_wStoS = np.sum(np.float32(dfS.loc[ dfS['ZHSR']==True ]['weight']))
sum_wBtoB = np.sum(np.float32(dfB.loc[ dfB['ZHSR']==False]['weight']))
print("sum_wStoS",sum_wStoS)
print("sum_wBtoB",sum_wBtoB)
rate_StoS = sum_wStoS/sum_wS
rate_BtoB = sum_wBtoB/sum_wB
print("Cut Based WP:",rate_StoS,"Signal Eff.", rate_BtoB,"1-Background Eff.")

# |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
X_train=[np.concatenate( (np.float32(dfB_train[jet]), np.float32(dfS_train[jet])) ) for jet in model.xVariables]
X_val  =[np.concatenate( (np.float32(dfB_val  [jet]), np.float32(dfS_val  [jet])) ) for jet in model.xVariables]
X_train=torch.FloatTensor([np.float32([[X_train[jet][event][mu] for jet in range(len(model.xVariables))] for mu in range(4)]) for event in range(nTrainB+nTrainS)])
X_val  =torch.FloatTensor([np.float32([[X_val  [jet][event][mu] for jet in range(len(model.xVariables))] for mu in range(4)]) for event in range(nValB  +nValS  )])
# X_train=torch.cat(  ( torch.FloatTensor([np.float32(dfB_train[jet]) for jet in model.xVariables]).resize_([nTrainB, 4, len(model.xVariables)]),
#                       torch.FloatTensor([np.float32(dfS_train[jet]) for jet in model.xVariables]).resize_([nTrainS, 4, len(model.xVariables)]) )  )
# X_val  =torch.cat(  ( torch.FloatTensor([np.float32(dfB_val  [jet]) for jet in model.xVariables]).resize_([nValB,   4, len(model.xVariables)]),
#                       torch.FloatTensor([np.float32(dfS_val  [jet]) for jet in model.xVariables]).resize_([nValS,   4, len(model.xVariables)]) )  )

y_train=torch.FloatTensor(  np.concatenate( (np.zeros(nTrainB, dtype=np.uint8).reshape(-1,1), 
                                             np.ones( nTrainS, dtype=np.uint8).reshape(-1,1)) )  )
y_val  =torch.FloatTensor(  np.concatenate( (np.zeros(nValB,   dtype=np.uint8).reshape(-1,1), 
                                             np.ones( nValS,   dtype=np.uint8).reshape(-1,1)) )  )

w_train=torch.FloatTensor(  np.concatenate( (np.float32(dfB_train['weight']).reshape(-1,1),   
                                             np.float32(dfS_train['weight']).reshape(-1,1)*sum_wB/sum_wS) )  )
w_val  =torch.FloatTensor(  np.concatenate( (np.float32(dfB_val  ['weight']).reshape(-1,1),   
                                             np.float32(dfS_val  ['weight']).reshape(-1,1)*sum_wB/sum_wS) )  )

# X_train = np.concatenate( (np.float32(dfB_train[model.xVariables]),         np.float32(dfS_train[model.xVariables])) )
# X_val   = np.concatenate( (np.float32(dfB_val  [model.xVariables]),         np.float32(dfS_val  [model.xVariables])) )
# y_train = np.concatenate( (np.zeros(nTrainB, dtype=np.uint8).reshape(-1,1), np.ones(nTrainS, dtype=np.uint8).reshape(-1,1)) )
# y_val   = np.concatenate( (np.zeros(nValB  , dtype=np.uint8).reshape(-1,1), np.ones(nValS  , dtype=np.uint8).reshape(-1,1)) )
# w_train = np.concatenate( (np.float32(dfB_train['weight']).reshape(-1,1),   np.float32(dfS_train[  'weight']).reshape(-1,1)*sum_wB/sum_wS) )
# w_val   = np.concatenate( (np.float32(dfB_val  ['weight']).reshape(-1,1),   np.float32(dfS_val  [  'weight']).reshape(-1,1)*sum_wB/sum_wS) )
print('X_train.shape, y_train.shape, w_train.shape:', X_train.shape, y_train.shape, w_train.shape)
print('X_val  .shape, y_val  .shape, w_val  .shape:', X_val  .shape, y_val  .shape, w_val  .shape)

# Standardize inputs

if not args.model:
    # model.scalers[0] = StandardScaler(with_mean=False)
    # model.scalers[0].fit(X_train[:,:,1].index_select(1,torch.LongTensor([0,3]))) ##only fit the scalar to one jet spectra. Don't want each pt ordered jet scale to be different

    model.scalers[0] = StandardScaler(with_mean=False)
    model.scalers[0].fit(X_train[:,:,1])
    model.scalers[0].scale_[1] = 2.5   # eta max
    model.scalers[0].scale_[2] = np.pi # pi
    model.scalers[0].scale_[3] = model.scalers[0].scale_[0]
    print("scale_",model.scalers[0].scale_)

print("Before Scale")
print(X_train[0])
for jet in range(X_train.shape[2]):
    X_train[:,:,jet] = torch.FloatTensor(model.scalers[0].transform(X_train[:,:,jet]))
    X_val  [:,:,jet] = torch.FloatTensor(model.scalers[0].transform(X_val  [:,:,jet]))
print("After Scale")
print(X_train[0])


# Set up data loaders
dset_train   = TensorDataset(X_train, y_train, w_train)
dset_val     = TensorDataset(X_val,   y_val,   w_val)
train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True,  num_workers=n_queue, pin_memory=True)
val_loader   = DataLoader(dataset=dset_val,   batch_size=batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
print('len(train_loader), len(val_loader):', len(train_loader), len(val_loader))
print('N trainable params:',sum(p.numel() for p in model.fcnet.parameters() if p.requires_grad))

optimizer = optim.Adam(model.fcnet.parameters(), lr=model.lrInit)

#Function to perform training epoch
def train(s):
    #print('-------------------------------------------------------------')
    model.fcnet.train()
    now = time.time()
    y_pred, y_true, w_ordered = [], [], []
    for i, (X, y, w) in enumerate(train_loader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model.fcnet(X).view(-1,1)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w) # binary classification
        #loss = F.binary_cross_entropy_with_logits(logits, y) # binary classification
        #loss = F.mse_loss(logits, y) # regression
        #break
        loss.backward()
        optimizer.step()
        #break
        prob_pred = torch.sigmoid(logits)
        y_pred.append(prob_pred.tolist())
        y_true.append(y.tolist())
        w_ordered.append(w.tolist())
        if (i+1) % print_step == 0:
            binary_pred = logits.ge(0.).byte()
            accuracy = binary_pred.eq(y.byte()).float().mean().item()
            sys.stdout.write('\rTraining %3.0f%%     '%(float(i+1)*100/len(train_loader)))
            sys.stdout.flush()

    now = time.time() - now
    #print(s+' Train time: %.2fs'%(now))

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    w_ordered = np.concatenate(w_ordered)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    bar=int((roc_auc-0.5)*200) if roc_auc > 0.5 else 0
    print('\r'+s+' ROC AUC: %0.4f   (Training Set)'%(roc_auc),("-"*bar)+"|")
    #print()
    return y_pred, y_true, w_ordered, fpr, tpr, roc_auc


def evaluate(loader):
    now = time.time()
    model.fcnet.eval()
    loss, accuracy = [], []
    y_pred, y_true, w_ordered = [], [], []
    for i, (X, y, w) in enumerate(loader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        logits = model.fcnet(X)#.view(-1,1)
        binary_pred = logits.ge(0.).byte()
        prob_pred = torch.sigmoid(logits)
        batch_loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction='none') # binary classification
        #batch_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none') # binary classification
        accuracy.append(binary_pred.eq(y.byte()).float().tolist())
        loss.append(batch_loss.tolist())
        y_pred.append(prob_pred.tolist())
        y_true.append(y.tolist())
        w_ordered.append(w.tolist())
        if (i+1) % print_step == 0:
            sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(loader)))
            sys.stdout.flush()

    now = time.time() - now
    #print('Evaluate time: %.2fs'%(now))

    accuracy = np.concatenate(accuracy)
    loss = np.concatenate(loss)
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    w_ordered = np.concatenate(w_ordered)

    return y_pred, y_true, w_ordered, accuracy, loss


#function to check performance on validation set
def validate(s):
    y_pred, y_true, w_ordered, accuracy, loss = evaluate(val_loader)
    #print(s+' Val loss: %f, accuracy: %f'%(loss.mean(), accuracy.mean()))

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    bar=int((roc_auc-0.5)*200) if roc_auc > 0.5 else 0
    print('\r'+s+' ROC AUC: %0.4f (Validation Set)'%(roc_auc),("#"*bar)+"|", end = " ")
    return y_pred, y_true, w_ordered, fpr, tpr, roc_auc


#Simple ROC Curve plot function
def plotROC(fpr, tpr, name): #fpr = false positive rate, tpr = true positive rate
    roc_auc = auc(fpr, tpr)
    f = plt.figure()
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    plt.plot([0,1], [1,0], color='0.8', linestyle='--')

    #plt.title(name.split("/")[-1].replace(".pdf","").replace("_"," "))
    plt.xlabel('Rate( Signal to Signal )')
    plt.ylabel('Rate( Background to Background )')

    plt.plot(tpr, 1-fpr)
    plt.text(0.72, 0.98, "ROC AUC = %0.4f"%(roc_auc))
    plt.scatter(rate_StoS, rate_BtoB, marker='o', c='r')
    plt.text(rate_StoS+0.03, rate_BtoB+0.02, "Cut Based WP")
    plt.text(rate_StoS+0.03, rate_BtoB-0.03, "(%0.2f, %0.2f)"%(rate_StoS, rate_BtoB))
    #print("plotROC:",name)
    f.savefig(name)
    plt.close(f)


def plotDNN(y_pred, y_true, w, name):
    fig = pltHelper.plot([y_pred[y_true==1], y_pred[y_true==0]], 
                         [b/20.0 for b in range(21)],
                         "DNN Output", "Events / Bin", 
                         weights=[w[y_true==1],w[y_true==0]],
                         samples=['Signal','Background'],
                         ratio=True,
                         ratioRange=[0,5])
    #print("plotDNN:",name)
    fig.savefig(name)
    plt.close(fig)
    

#model initial state
y_pred_val, y_true_val, w_ordered_val, fpr, tpr, roc_auc = validate(">> Epoch %3d/%d <<<<<<<<"%(model.startingEpoch, args.epochs+model.startingEpoch))
print()
if args.model:
    plotROC(fpr, tpr, args.model.replace('.pkl', '_ROC_val.pdf'))
    plotDNN(y_pred_val, y_true_val, w_ordered_val, args.model.replace('.pkl','_DNN_output_val.pdf'))

# Training loop
for epoch in range(model.startingEpoch+1, model.startingEpoch+args.epochs+1):
    epochString = '>> Epoch %3d/%d <<<<<<<<'%(epoch, args.epochs+model.startingEpoch)

    # Run training
    y_pred_train, y_true_train, w_ordered_train, fpr_train, tpr_train, roc_auc_train =    train(epochString)

    # Run Validation
    y_pred_val,   y_true_val,   w_ordered_val,   fpr_val,   tpr_val,   roc_auc_val   = validate(epochString)

    #print(epochString+" Overtraining %1.1f"%((roc_auc_train-roc_auc_val)*100))

    # Save model file:
    #roc_auc = (roc_auc_val + roc_auc_train)/2
    #roc_auc = roc_auc_train
    roc_auc = roc_auc_val
    if roc_auc > model.roc_auc_best:
        foundNewBest = True
        model.roc_auc_best = roc_auc
    
        filename = 'ZZ4b/NtupleAna/pytorchModels/%s_epoch%d_auc%.4f.pkl'%(model.name, epoch, model.roc_auc_best)
        #print("New Best AUC:", model.roc_auc_best, filename)
        print("*", filename)
        #print("Evaluate on training set for plots")
        y_pred_train, y_true_train, w_ordered_train, _, _ = evaluate(train_loader)
        fpr_train, tpr_train, _ = roc_curve(y_true_train, y_pred_train)
        plotROC(fpr_train, tpr_train, filename.replace('.pkl', '_ROC_train.pdf'))
        plotROC(fpr_val,   tpr_val,   filename.replace('.pkl', '_ROC_val.pdf'))
        plotDNN(y_pred_train, y_true_train, w_ordered_train, filename.replace('.pkl','_DNN_output_train.pdf'))
        plotDNN(y_pred_val,   y_true_val,   w_ordered_val,   filename.replace('.pkl','_DNN_output_val.pdf'))
        
        model_dict = {'model': model.fcnet.state_dict(), 'optim': optimizer.state_dict(), 'scalers': model.scalers}
        torch.save(model_dict, filename)
        #joblib.dump(scaler, filename)
    else:
        print()

print()
print(">> DONE <<<<<<<<")
if foundNewBest: print("Best ROC AUC =", model.roc_auc_best)
