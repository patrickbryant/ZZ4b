import time, os
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

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--infile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-l', '--lrInit', default=5e-4, type=float, help='Initial learning rate.')
parser.add_argument('-b', '--layers', default=3, type=int, help='N of fully-connected layers.')
parser.add_argument('-n', '--nodes', default=128, type=int, help='N of fully-connected nodes.')
parser.add_argument('-p', '--pDropout', default=0.2, type=float, help='p(drop) for dropout.')
parser.add_argument('-c', '--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")

args = parser.parse_args()


class modelParameters:
    def __init__(self, fileName=''):
        self.xVariables=['canJet1_pt', 'canJet3_pt',
                         'dRjjClose', 'dRjjOther', 
                         'aveAbsEta',
                         ]
        if fileName:
            self.nodes         =   int(fileName[fileName.find(     'FC')+2 : fileName.find('x')])
            self.layers        =   int(fileName[fileName.find(      'x')+1 : fileName.find('_pdrop')])
            self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_lr')])
            self.lrInit        = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_auc')])
            self.roc_auc_best  = float(fileName[fileName.find(   '_auc')+4 : fileName.find('.pkl')])
            self.scaler = torch.load(fileName)['scaler']

        else:
            self.nodes         = args.nodes
            self.layers        = args.layers
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0
            self.roc_auc_best  = 0.5365222388741322 #0.5432693564061828 #batch 128, l=1e-3, p=0.4   #0.539474367272775 #batch 512, default others
            self.scaler = StandardScaler()

        self.name = 'FC%dx%d_pdrop%.2f_lr%s_epochs%d_stdscale'%(self.nodes, self.layers, self.pDropout, str(self.lrInit), args.epochs+self.startingEpoch)

        # Set up NN model and optimizer
        self.dump()
        self.fc = []
        self.fc.append(nn.Linear(len(self.xVariables), self.nodes))
        self.fc.append(nn.ReLU())
        self.fc.append(nn.Dropout(p=self.pDropout))
        for _ in range(self.layers):
            self.fc.append(nn.Linear(self.nodes, self.nodes))
            self.fc.append(nn.ReLU())
            self.fc.append(nn.Dropout(p=self.pDropout))
        self.fc.append(nn.Linear(self.nodes, 1))
        self.fcnet = nn.Sequential(*self.fc).to(device)

        if fileName:
            print("Load Model:", fileName)
            self.fcnet.load_state_dict(torch.load(fileName)['model']) # load model from previous state
    
    def dump(self):
        print(self.name)
        print('nodes:',self.nodes)
        print('layers:',self.layers)
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
batch_size = 128 #36
foundNewBest = False
print_step = 10000
train_fraction = 0.5

# Read .h5 file
df = pd.read_hdf(args.infile, key='df')

if args.model and args.update:
    X_all = np.float32(df[model.xVariables])
    y_all = np.  uint8(df['fourTag']).reshape(-1,1)
    print('X_all.shape', X_all.shape)

    X_all = model.scaler.transform(X_all)

    # Set up data loaders
    batch_size = 2048
    dset   = TensorDataset(torch.FloatTensor(X_all), torch.FloatTensor(y_all))
    loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
    print('Batches:', len(loader))

    model.fcnet.eval()
    y_pred = []
    for i, (X, y) in enumerate(loader):
        X = X.to(device)
        logits = model.fcnet(X)
        binary_pred = logits.ge(0.).byte()
        prob_pred = torch.sigmoid(logits)
        y_pred.append(prob_pred.tolist())
        if (i+1) % 1000 == 0:
            print('Evaluate Batch %d/%d'%(i+1, len(loader)))

    y_pred = np.float32(np.concatenate(y_pred).reshape(1,df.shape[0])[0])
    print(y_pred)
    df['nTagClassifier'] = pd.Series(y_pred, index=df.index)
    print("df.dtypes")
    print(df.dtypes)
    print("df.shape", df.shape)
    df.to_hdf(args.infile, key='df', format='table', mode='w')

    exit()


#select events in desired region for training/validation/test
df_selected = df.loc[ (df['ZHSB'] == True) & (df['passDEtaBB'] == True) ]
print("df_selected.shape",df_selected.shape)

n      = df_selected.shape[0]
nTrain = int(n*train_fraction)
idxs   = np.random.permutation(n)
print(idxs)

#define dataframes for trainging and validation
df_train = df_selected.iloc[idxs[:nTrain]]
df_val   = df_selected.iloc[idxs[nTrain:]]

# #compute weights for weighted random sampler used in the training dataloader to ensure equal statistical representation of three and four tag events
# n4b = df_train.loc[df_train['fourTag'] == True ].shape[0]
# n3b = df_train.loc[df_train['fourTag'] == False].shape[0]
# print('nThreeTag',n3b)
# print('nFourTag',n4b)

# samplerWeight3b = float(n)/n3b #float(n4b/n3b) #float(n)/n3b
# samplerWeight4b = float(n)/n4b #1.0 #float(n)/n4b
# samplerWeights = [samplerWeight4b if row['fourTag'] == True else samplerWeight3b for index, row in df_train.iterrows()]
# samplerWeights = np.float32(samplerWeights)

# Convert to numpy
X_train = np.float32(df_train[model.xVariables])
y_train = np.  uint8(df_train[ 'fourTag']).reshape(-1,1)
w_train = np.float32(df_train[  'weight']).reshape(-1,1) #/samplerWeights
X_val   = np.float32(df_val  [model.xVariables])
y_val   = np.  uint8(df_val  [ 'fourTag']).reshape(-1,1)
w_val   = np.float32(df_val  [  'weight']).reshape(-1,1)
print('X_train.shape, y_train.shape, w_train.shape:', X_train.shape, y_train.shape, w_train.shape)
print('X_val  .shape, y_val  .shape, w_val  .shape:', X_val  .shape, y_val  .shape, w_val  .shape)

# Standardize inputs
if not args.model:
    model.scaler.fit(X_train)
X_train = model.scaler.transform(X_train)
X_val   = model.scaler.transform(X_val)

# Set up data loaders
dset_train   = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(w_train))
dset_val     = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val),   torch.FloatTensor(w_val))
#sampler = sampler.WeightedRandomSampler(torch.FloatTensor(samplerWeights), nTrain)
#sampler = sampler.RandomSampler(range(nTrain))
train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True,  num_workers=n_queue, pin_memory=True)
#train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler = sampler)
val_loader   = DataLoader(dataset=dset_val,   batch_size=batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
print('len(train_loader), len(val_loader):', len(train_loader), len(val_loader))
print('N trainable params:',sum(p.numel() for p in model.fcnet.parameters() if p.requires_grad))

optimizer = optim.Adam(model.fcnet.parameters(), lr=model.lrInit)

#Function to perform training epoch
def train(s):
    print('-------------------------------------------------------------')
    model.fcnet.train()
    now = time.time()
    y_pred, y_true, w_ordered = [], [], []
    for i, (X, y, w) in enumerate(train_loader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model.fcnet(X)
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
            print(s+' (%d/%d) Train loss: %f, accuracy: %f'%(i+1, len(train_loader), loss.item(), accuracy))

    now = time.time() - now
    #print(s+' Train time: %.2fs'%(now))

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    w_ordered = np.concatenate(w_ordered)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(s+' ROC AUC: %f (Training Set)'%(roc_auc))
    #print()
    return y_pred, y_true, w_ordered, fpr, tpr, roc_auc


def evaluate(loader):
    now = time.time()
    model.fcnet.eval()
    loss, accuracy = [], []
    y_pred, y_true, w_ordered = [], [], []
    for i, (X, y, w) in enumerate(loader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        logits = model.fcnet(X)
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
            print('Evaluate Batch %d/%d'%(i+1, len(loader)))

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
    print(s+' ROC AUC: %f (Validation Set)'%(roc_auc))
    return y_pred, y_true, w_ordered, fpr, tpr, roc_auc


#Simple ROC Curve plot function
def plotROC(fpr, tpr, name): #fpr = false positive rate, tpr = true positive rate
    roc_auc = auc(fpr, tpr)
    f = plt.figure()
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    plt.plot([0,1], [1,0], color='0.8', linestyle='--')

    #plt.title(name.split("/")[-1].replace(".pdf","").replace("_"," "))
    plt.xlabel('Rate( fourTag to fourTag )')
    plt.ylabel('Rate( threeTag to threeTag )')

    plt.plot(tpr, 1-fpr)
    plt.text(0.72, 0.98, "ROC AUC = %0.4f"%(roc_auc))
    print("plotROC:",name)
    f.savefig(name)


def plotDNN(y_pred, y_true, w, name):
    fig = pltHelper.plot([y_pred[y_true==1], y_pred[y_true==0]], 
                         [b/20.0 for b in range(21)],
                         "DNN Output", "Events / Bin", 
                         weights=[w[y_true==1],w[y_true==0]],
                         samples=['fourTag','threeTag'],
                         ratio=True)
    print("plotDNN:",name)
    fig.savefig(name)
    

#model initial state
y_pred_val, y_true_val, w_ordered_val, fpr, tpr, roc_auc = validate(">> Initial State Epoch %d <<<<<<<<"%(model.startingEpoch))
if args.model:
    plotROC(fpr, tpr, args.model.replace('.pkl', '_ROC_val.pdf'))
    plotDNN(y_pred_val, y_true_val, w_ordered_val, args.model.replace('.pkl','_DNN_output_val.pdf'))

# Training loop
for epoch in range(model.startingEpoch+1, model.startingEpoch+args.epochs+1):
    epochString = '>> Epoch %d/%d <<<<<<<<'%(epoch, args.epochs+model.startingEpoch)

    # Run training
    y_pred_train, y_true_train, w_ordered_train, fpr_train, tpr_train, roc_auc_train =    train(epochString)

    # Run Validation
    y_pred_val,   y_true_val,   w_ordered_val,   fpr_val,   tpr_val,   roc_auc_val   = validate(epochString)

    print(epochString+" Overtraining %1.1f"%((roc_auc_train-roc_auc_val)*100))

    # Save model file:
    #roc_auc = (roc_auc_val + roc_auc_train)/2
    #roc_auc = roc_auc_train
    roc_auc = roc_auc_val
    if roc_auc > model.roc_auc_best:
        foundNewBest = True
        model.roc_auc_best = roc_auc
    
        filename = 'ZZ4b/NtupleAna/pytorchModels/%s_epoch%d_auc%.4f.pkl'%(model.name, epoch, model.roc_auc_best)
        print("New Best AUC:", model.roc_auc_best, filename)
        fpr_train, tpr_train, _ = roc_curve(y_true_train, y_pred_train)
        plotROC(fpr_train, tpr_train, filename.replace('.pkl', '_ROC_train.pdf'))
        plotROC(fpr_val,   tpr_val,   filename.replace('.pkl', '_ROC_val.pdf'))
        plotDNN(y_pred_train, y_true_train, w_ordered_train, filename.replace('.pkl','_DNN_output_train.pdf'))
        plotDNN(y_pred_val,   y_true_val,   w_ordered_val,   filename.replace('.pkl','_DNN_output_val.pdf'))
        
        model_dict = {'model': model.fcnet.state_dict(), 'optim': optimizer.state_dict(), 'scaler': model.scaler}
        torch.save(model_dict, filename)
        #joblib.dump(scaler, filename)

print()
print(">> DONE <<<<<<<<")
if foundNewBest: print("Best ROC AUC =", model.roc_auc_best)
