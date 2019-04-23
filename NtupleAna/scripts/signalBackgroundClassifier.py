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
import matplotlib
matplotlib.use('Agg')
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

# Run on gpu if available
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
print('torch.cuda.is_available()',torch.cuda.is_available())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device",device,torch.cuda.device_count(),torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU:",device)

from networks import *

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
        self.ancillaryFeatures=['d01', 'd23', 'd02', 'd13', 'd03', 'd12', 'nSelJets', 'm4j', 'xWt0', 'xWt1']
        #self.nAncillaryFeatures = len(self.ancillaryFeatures)
        #self.useAncillaryROCAUCMin = 0.82
        #self.fourVectors=[['canJet'+jet+mu for jet in self.layer1Pix] for mu in self.layer1Col] #index[color][pixel]
        #self.fourVectors[color][pixel]
        if fileName:
            self.dijetFeatures        = int(fileName.split('_')[2])
            self.quadjetFeatures      = int(fileName.split('_')[3])
            self.combinatoricFeatures = int(fileName.split('_')[4])
            self.nodes                = None#int(fileName.split('_')[5])
            self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_lr')]) if '_pdrop' in fileName else None
            self.lrInit        = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_auc')])
            self.roc_auc_best  = float(fileName[fileName.find(   '_auc')+4 : fileName.find('.pkl')])
            self.scalers = torch.load(fileName)['scalers']

        else:
            self.dijetFeatures = 4
            self.quadjetFeatures = 4
            self.combinatoricFeatures = 20
            self.nodes = 128
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_12_22_32_lr0.001_epochs50_stdscale_epoch48_auc0.8363.pkl started taking off at epoch ~26 batch size 20, amsgrad=false
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_12_22_32_lr0.001_epochs50_stdscale_epoch42_auc0.8429.pkl batch size 20 amsgrad=true not using jet energy, significant overtraining
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_12_22_32_lr0.001_epochs50_stdscale_epoch11_auc0.8653.pkl manually adding dijet masses to quadjet feature space
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_12_22_32_lr0.001_epochs50_stdscale_epoch10_auc0.8764.pkl 8192 trainable. also manually add nSelJets to input of view selector, get rid of fancy training tricks with ROC thresholds overtrain epoch 13
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_8_12_20_lr0.001_epochs50_stdscale_epoch8_auc0.8707.pkl 2948 trainable
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_8_12_20_lr0.001_epochs50_stdscale_epoch20_auc0.8760.pkl dynamic training batch size
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_4_4_20_lr0.001_epochs20_stdscale_epoch19_auc0.8698.pkl 792 trainable
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_4_4_20_lr0.001_epochs20_stdscale_epoch17_auc0.8736.pkl 852 trainable
            # 789 trainable, moved dijet masses to dijetResNetBlock, bad performance. Maybe not enough features? 
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_7_11_20_lr0.001_epochs20_stdscale_epoch13_auc0.8755.pkl 2472 trainable
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_6_10_20_lr0.001_epochs20_stdscale_epoch17_auc0.8752.pkl 2598 trainable moved back to adding dijet masses at quadjet block
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_6_5_20_lr0.001_epochs20_stdscale_epoch17_auc0.8713.pkl 1192 trainable
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_12_5_32_lr0.001_epochs20_stdscale_epoch16_auc0.8710.pkl 2440 trainable started overtraining
            # 4_4_32 1128 trainable, never beat 0.87
            # 4_4_15 737 trainable, never beat 0.87
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_4_4_21_lr0.001_epochs20_stdscale_epoch19_auc0.8731.pkl 875 trainable,
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_4_4_21_lr0.001_epochs20_stdscale_epoch18_auc0.8781.pkl 1001 trainable added xWt0, xWt1 to viewSelector input features
            # 4_4_27 85.1% 1175 trainable
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_4_4_18_lr0.001_epochs20_stdscale_epoch19_auc0.8771.pkl 914 trainable
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_4_4_22_lr0.001_epochs20_stdscale_epoch20_auc0.8748.pkl 1030 trainable
            # ZZ4b/NtupleAna/pytorchModels/SvsB_ResNet_4_4_20_lr0.001_epochs20_stdscale_epoch20_auc0.8777.pkl 972 trainable
            
            self.roc_auc_best  = 0.87
            self.scalers = {}

        #self.name = 'SvsB_FC%dx%d_pdrop%.2f_lr%s_epochs%d_stdscale'%(self.nodes, self.layers, self.pDropout, str(self.lrInit), args.epochs+self.startingEpoch)

        #self.net = basicCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        #self.net = dijetCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        #self.name = 'SvsB_dijetCNN_%d_%d_%d_%d_pdrop%.2f_lr%s_epochs%d_stdscale'%(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout, str(self.lrInit), args.epochs+self.startingEpoch)
        self.net = ResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures).to(device)
        #self.net = PresResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        #self.net = deepResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nAncillaryFeatures).to(device)
        self.name = 'SvsB_'+self.net.name+'_lr%s_epochs%d_stdscale'%(str(self.lrInit), args.epochs+self.startingEpoch)

        #if fileName and self.roc_auc_best > self.useAncillaryROCAUCMin: self.net.useAncillary = True 

        self.dump()

        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state
    
    def dump(self):
        print(self.net)
        print(self.name)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('roc_auc_best:',self.roc_auc_best)
        #print('useAncillary:',self.net.useAncillary)



model = modelParameters(args.model)

n_queue = 20
train_batch_size_small =  32 #64 #32 #36
train_batch_size_large = 128
eval_batch_size = 16384
foundNewBest = False
print_step = 100
train_fraction = 0.7

if args.model and args.update:
    for fileName in [args.background, args.signal]:
        print("Add classifier output to",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')

        n = df.shape[0]
        print("n",n)

        P = [np.float32(df[jet]) for jet in model.fourVectors]
        P = torch.FloatTensor([np.float32([[P[jet][event][mu] for jet in range(len(model.fourVectors))] for mu in range(model.jetFeatures)]) for event in range(n)])
        A = torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in model.ancillaryFeatures], 1 )
        y = torch.FloatTensor( np.zeros(n, dtype=np.uint8).reshape(-1,1) )
        print('P.shape', P.shape)

        for jet in range(P.shape[2]):
            P[:,:,jet] = torch.FloatTensor(model.scalers[0].transform(P[:,:,jet]))
        A = torch.FloatTensor(model.scalers['ancillary'].transform(A))

        # Set up data loaders
        dset   = TensorDataset(P, A, y)
        loader = DataLoader(dataset=dset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        print('Batches:', len(loader))

        model.net.eval()
        y_pred = []
        for i, (P, A, y) in enumerate(loader):
            P, A = P.to(device), A.to(device)
            logits = model.net(P, A)
            binary_pred = logits.ge(0.).byte()
            prob_pred = torch.sigmoid(logits)
            y_pred.append(prob_pred.tolist())
            if (i+1) % print_step == 0:
                sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(loader)))
                sys.stdout.flush()

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
#dfB = dfB.loc[ (dfB['passHLT']==True) & (dfB['fourTag']==False) & ((dfB['ZHSB']==True)|(dfB['ZHCR']==True)|(dfB['ZHSR']==True)) & (dfB['passDEtaBB']==True) ]
#dfS = dfS.loc[ (dfS['passHLT']==True) & (dfS['fourTag']==True ) & ((dfS['ZHSB']==True)|(dfS['ZHCR']==True)|(dfS['ZHSR']==True)) & (dfS['passDEtaBB']==True) ]
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
nTrain   = df_train.shape[0]
df_val   = pd.concat([dfB_val,   dfS_val  ], sort=False)
nVal     = df_val  .shape[0]

#Convert to list np array
P_train=[np.float32(df_train[jet]) for jet in model.fourVectors]
P_val  =[np.float32(df_val  [jet]) for jet in model.fourVectors]
#make 3D tensor with correct axes [event][color][pixel] = [event][mu (4-vector component)][jet]
P_train=torch.FloatTensor([np.float32([[P_train[jet][event][mu] for jet in range(len(model.fourVectors))] for mu in range(model.jetFeatures)]) for event in range(nTrain)])
P_val  =torch.FloatTensor([np.float32([[P_val  [jet][event][mu] for jet in range(len(model.fourVectors))] for mu in range(model.jetFeatures)]) for event in range(nVal  )])
#extra features for use with output of CNN layers
A_train=torch.cat( [torch.FloatTensor( np.float32(df_train[feature]).reshape(-1,1) ) for feature in model.ancillaryFeatures], 1 )
A_val  =torch.cat( [torch.FloatTensor( np.float32(df_val  [feature]).reshape(-1,1) ) for feature in model.ancillaryFeatures], 1 )

y_train=torch.FloatTensor(  np.concatenate( (np.zeros(nTrainB, dtype=np.uint8).reshape(-1,1), 
                                             np.ones( nTrainS, dtype=np.uint8).reshape(-1,1)) )  )
y_val  =torch.FloatTensor(  np.concatenate( (np.zeros(nValB,   dtype=np.uint8).reshape(-1,1), 
                                             np.ones( nValS,   dtype=np.uint8).reshape(-1,1)) )  )

w_train=torch.FloatTensor( np.float32(df_train['weight']).reshape(-1,1) )
w_val  =torch.FloatTensor( np.float32(df_val  ['weight']).reshape(-1,1) )

print('P_train.shape, A_train.shape, y_train.shape, w_train.shape:', P_train.shape, A_train.shape, y_train.shape, w_train.shape)
print('P_val  .shape, A_val  .shape, y_val  .shape, w_val  .shape:', P_val  .shape, A_val  .shape, y_val  .shape, w_val  .shape)

# Standardize inputs
# for jet in range(P_train.shape[2]):
#     P_train[:,0,jet] = torch.FloatTensor(np.log(P_train[:,0,jet]))
#     P_train[:,3,jet] = torch.FloatTensor(np.log(P_train[:,3,jet]))
#     P_val  [:,0,jet] = torch.FloatTensor(np.log(P_val  [:,0,jet]))
#     P_val  [:,3,jet] = torch.FloatTensor(np.log(P_val  [:,3,jet]))

if not args.model:
    # model.scalers[0] = StandardScaler(with_mean=False)
    # model.scalers[0].fit(P_train[:,:,1].index_select(1,torch.LongTensor([0,3]))) ##only fit the scalar to one jet spectra. Don't want each pt ordered jet scale to be different

    model.scalers[0] = StandardScaler()
    model.scalers[0].fit(P_train[:,:,1])
    #model.scalers[0].scale_[0], model.scalers[0].mean_[0] =   100, 80
    model.scalers[0].scale_[1], model.scalers[0].mean_[1] =   2.5,  0 # eta max
    model.scalers[0].scale_[2], model.scalers[0].mean_[2] = np.pi,  0 # pi
    #model.scalers[0].scale_[3], model.scalers[0].mean_[3] = model.scalers[0].scale_[0], model.scalers[0].mean_[0]
    #model.scalers[0].scale_[3], model.scalers[0].mean_[3] =   200, 80
    print("model.scalers[0].scale_",model.scalers[0].scale_)
    print("model.scalers[0].mean_",model.scalers[0].mean_)
    model.scalers['ancillary'] = StandardScaler()
    model.scalers['ancillary'].fit(A_train)
    model.scalers['ancillary'].scale_[0], model.scalers['ancillary'].mean_[0] = 100, 200
    model.scalers['ancillary'].scale_[1], model.scalers['ancillary'].mean_[1] = 100, 200
    model.scalers['ancillary'].scale_[2], model.scalers['ancillary'].mean_[2] = 100, 200
    model.scalers['ancillary'].scale_[3], model.scalers['ancillary'].mean_[3] = 100, 200
    model.scalers['ancillary'].scale_[4], model.scalers['ancillary'].mean_[4] = 100, 200
    model.scalers['ancillary'].scale_[5], model.scalers['ancillary'].mean_[5] = 100, 200
    model.scalers['ancillary'].scale_[6], model.scalers['ancillary'].mean_[6] =   3,   6
    print("model.scalers['ancillary'].scale_",model.scalers['ancillary'].scale_)
    print("model.scalers['ancillary'].mean_",model.scalers['ancillary'].mean_)

for jet in range(P_train.shape[2]):
    P_train[:,:,jet] = torch.FloatTensor(model.scalers[0].transform(P_train[:,:,jet]))
    P_val  [:,:,jet] = torch.FloatTensor(model.scalers[0].transform(P_val  [:,:,jet]))
A_train = torch.FloatTensor(model.scalers['ancillary'].transform(A_train))
A_val   = torch.FloatTensor(model.scalers['ancillary'].transform(A_val))


# Set up data loaders
dset_train   = TensorDataset(P_train, A_train, y_train, w_train)
dset_val     = TensorDataset(P_val,   A_val,   y_val,   w_val)
train_loader_small_batches = DataLoader(dataset=dset_train, batch_size=train_batch_size_small, shuffle=True,  num_workers=n_queue, pin_memory=True)
train_loader_large_batches = DataLoader(dataset=dset_train, batch_size=train_batch_size_large, shuffle=True,  num_workers=n_queue, pin_memory=True)
eval_train_loader = DataLoader(dataset=dset_train, batch_size=eval_batch_size, shuffle=False,  num_workers=n_queue, pin_memory=True)
val_loader   = DataLoader(dataset=dset_val,   batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
print('len(train_loader_large_batches), len(train_loader_small_batches), len(val_loader):', len(train_loader_large_batches), len(train_loader_small_batches), len(val_loader))
print('N trainable params:',sum(p.numel() for p in model.net.parameters() if p.requires_grad))

optimizer = optim.Adam(model.net.parameters(), lr=model.lrInit, amsgrad=False)

def evaluate(loader):
    now = time.time()
    model.net.eval()
    #loss, accuracy = [], []
    y_pred, y_true, w_ordered = [], [], []
    for i, (P, A, y, w) in enumerate(loader):
        P, A, y, w = P.to(device), A.to(device), y.to(device), w.to(device)
        logits = model.net(P, A)#.view(-1,1)
        binary_pred = logits.ge(0.).byte()
        prob_pred = torch.sigmoid(logits)
        batch_loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction='none') # binary classification
        #batch_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none') # binary classification
        #accuracy.append(binary_pred.eq(y.byte()).float().tolist())
        #loss.append(batch_loss.tolist())
        y_pred.append(prob_pred.tolist())
        y_true.append(y.tolist())
        w_ordered.append(w.tolist())
        if (i+1) % print_step == 0:
            sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(loader)))
            sys.stdout.flush()

    now = time.time() - now

    #accuracy = np.concatenate(accuracy)
    #loss = np.concatenate(loss)
    y_pred = np.transpose(np.concatenate(y_pred))[0]
    y_true = np.transpose(np.concatenate(y_true))[0]
    w_ordered = np.transpose(np.concatenate(w_ordered))[0]
    
    fpr, tpr, thr = roc_curve(y_true, y_pred, sample_weight=w_ordered)
    roc_auc = auc(fpr, tpr)

    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


#Function to perform training epoch
def train(s, loader):
    #print('-------------------------------------------------------------')
    model.net.train()
    now = time.time()
    #y_pred, y_true, w_ordered = [], [], []
    for i, (P, A, y, w) in enumerate(loader):
        P, A, y, w = P.to(device), A.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model.net(P, A)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w) # binary classification
        #loss = F.binary_cross_entropy_with_logits(logits, y) # binary classification
        #loss = F.mse_loss(logits, y) # regression
        #break
        loss.backward()
        optimizer.step()
        #break
        #prob_pred = torch.sigmoid(logits)
        #y_pred.append(prob_pred.tolist())
        #y_true.append(y.tolist())
        #w_ordered.append(w.tolist())
        if (i+1) % print_step == 0:
            #binary_pred = logits.ge(0.).byte()
            #accuracy = binary_pred.eq(y.byte()).float().mean().item()
            sys.stdout.write('\rTraining %3.0f%%     '%(float(i+1)*100/len(loader)))
            sys.stdout.flush()

    now = time.time() - now

    y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc = evaluate(eval_train_loader)

    # y_pred = np.concatenate(y_pred)
    # y_true = np.concatenate(y_true)
    # w_ordered = np.concatenate(w_ordered)
    # fpr, tpr, thr = roc_curve(y_true, y_pred, sample_weight=w_ordered)
    # roc_auc = auc(fpr, tpr)
    bar=int((roc_auc-0.5)*200) if roc_auc > 0.5 else 0
    print('\r'+' '*len(s)+'       Training: %2.1f%%'%(roc_auc*100),("-"*bar)+"|")
    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


#function to check performance on validation set
def validate(s):
    y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc = evaluate(val_loader)

    bar=int((roc_auc-0.5)*200) if roc_auc > 0.5 else 0
    print('\r'+s+' ROC Validation: %2.1f%%'%(roc_auc*100),("#"*bar)+"|", end = " ")
    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


#Simple ROC Curve plot function
def plotROC(fpr, tpr, thr, name): #fpr = false positive rate, tpr = true positive rate
    roc_auc = auc(fpr, tpr)
    lumiRatio = 10
    sigma = (tpr*sum_wS*lumiRatio) / np.sqrt(fpr*sum_wB*lumiRatio + 1)
    iMaxSigma = np.argmax(sigma)
    maxSigma = sigma[iMaxSigma]
    f = plt.figure()
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    plt.plot([0,1], [1,0], color='0.8', linestyle='--')

    plt.xlabel('Rate( Signal to Signal )')
    plt.ylabel('Rate( Background to Background )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    plt.plot(tpr, 1-fpr)
    plt.text(0.80, 1.07, "ROC AUC = %0.4f"%(roc_auc))
    plt.scatter(rate_StoS, rate_BtoB, marker='o', c='r')
    plt.text(rate_StoS+0.03, rate_BtoB-0.025, "Cut Based WP \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)
    #plt.text(rate_StoS-0.20, rate_BtoB-0.03, "(%0.2f, %0.2f)"%(rate_StoS, rate_BtoB))
    plt.scatter(tpr[iMaxSigma], (1-fpr[iMaxSigma]), marker='o', c='k')
    plt.text(tpr[iMaxSigma]+0.03, (1-fpr[iMaxSigma])-0.025, "Optimal WP, SvB $>$ %0.2f \n (%0.2f, %0.2f), $%1.2f\sigma$ with 140fb$^{-1}$"%(thr[iMaxSigma], tpr[iMaxSigma], (1-fpr[iMaxSigma]), maxSigma), bbox=bbox)
    #plt.text(tpr[iMaxSigma]+0.03, (1-fpr[iMaxSigma])+0.01, "(%0.2f, %0.2f) $%1.2f\sigma$ with 140fb$^{-1}$"%(tpr[iMaxSigma], (1-fpr[iMaxSigma]), maxSigma))
    f.savefig(name)
    plt.close(f)


def plotNet(y_pred, y_true, w, name):
    yS_pred, yB_pred = y_pred[y_true==1], y_pred[y_true==0]
    wS,      wB      = w     [y_true==1], w     [y_true==0]
    fig = pltHelper.plot([yS_pred, yB_pred], 
                         [b/20.0 for b in range(21)],
                         "NN Output (SvB)", "Events / Bin", 
                         weights=[wS, wB],
                         samples=['Signal','Background'],
                         ratio=True,
                         ratioRange=[0,5])
    fig.savefig(name)
    plt.close(fig)
    
def epochString(epoch):
    return ('>> %'+str(len(str(args.epochs+model.startingEpoch)))+'d/%d <<')%(epoch, args.epochs+model.startingEpoch)


#model initial state
y_pred_val, y_true_val, w_ordered_val, fpr_val, tpr_val, thr_val, roc_auc = validate(epochString(0))
print()
if args.model:
    plotROC(fpr_val, tpr_val, thr_val, args.model.replace('.pkl', '_ROC_val.pdf'))
    plotNet(y_pred_val, y_true_val, w_ordered_val, args.model.replace('.pkl','_NetOutput_val.pdf'))

# Training loop
reducedLearningRate = False
train_loader = train_loader_large_batches
roc_auc_val_prev = roc_auc
nValDecrease = 0
for epoch in range(model.startingEpoch+1, model.startingEpoch+args.epochs+1):
    # # Start using ancillary information if ROC AUC above some threshold
    # if roc_auc > model.useAncillaryROCAUCMin and not model.net.useAncillary: 
    #     model.net.useAncillary = True
    #     print("Start using ancillary information")

    # # Reduce learning rate
    # if roc_auc > 0.75 and not reducedLearningRate:
    #     reducedLearningRate = True
    #     optimizer.lr = args.lrInit/2
    #     print("Reduce Learning Rate",optimizer.lr)

    # Run training
    y_pred_train, y_true_train, w_ordered_train, fpr_train, tpr_train, thr_train, roc_auc_train =    train(epochString(epoch), train_loader)

    # Run Validation
    y_pred_val,   y_true_val,   w_ordered_val,   fpr_val,   tpr_val, thr_val,     roc_auc_val   = validate(epochString(epoch))

    roc_auc = roc_auc_val
    if roc_auc > model.roc_auc_best:
        foundNewBest = True
        model.roc_auc_best = roc_auc
    
        filename = 'ZZ4b/NtupleAna/pytorchModels/%s_epoch%d_auc%.4f.pkl'%(model.name, epoch, model.roc_auc_best)
        print("*", filename)
        #y_pred_train, y_true_train, w_ordered_train, fpr_train, tpr_train, thr_train, roc_auc_train = evaluate(eval_train_loader)
        #fpr_train, tpr_train, thr_train = roc_curve(y_true_train, y_pred_train, sample_weight=w_ordered_train)
        plotROC(fpr_train, tpr_train, thr_train, filename.replace('.pkl', '_ROC_train.pdf'))
        plotROC(fpr_val,   tpr_val,   thr_val,   filename.replace('.pkl', '_ROC_val.pdf'))
        plotNet(y_pred_train, y_true_train, w_ordered_train, filename.replace('.pkl','_NetOutput_train.pdf'))
        plotNet(y_pred_val,   y_true_val,   w_ordered_val,   filename.replace('.pkl','_NetOutput_val.pdf'))
        
        model_dict = {'model': model.net.state_dict(), 'optim': optimizer.state_dict(), 'scalers': model.scalers}
        torch.save(model_dict, filename)
    else:
        print("^ %1.1f%%"%((roc_auc_train-roc_auc_val)*100))

    #if (roc_auc_train - roc_auc_val) > 0.0045 and train_loader == train_loader_large_batches: 
    if roc_auc_val < roc_auc_val_prev: nValDecrease += 1
    roc_auc_val_prev = roc_auc_val
    if nValDecrease > 1 and train_loader == train_loader_large_batches: 
        print("Start using smaller batches for training:",train_batch_size_large,"->",train_batch_size_small)
        train_loader = train_loader_small_batches
        optimizer.lr = args.lrInit/5


print()
print(">> DONE <<")
if foundNewBest: print("Best ROC AUC =", model.roc_auc_best)
