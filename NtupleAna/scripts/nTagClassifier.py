import time, os
import numpy as np
import pandas as pd
np.random.seed(0)
#import pyarrow.parquet as pq # pip install pyarrow==0.7.1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from sklearn.metrics import roc_curve, auc # pip/conda install scikit-learn
from sklearn.preprocessing import StandardScaler

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--infile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-l', '--lr_init', default=5.e-4, type=float, help='Initial learning rate.')
parser.add_argument('-b', '--layers', default=3, type=int, help='N of fully-connected layers.')
parser.add_argument('-n', '--nodes', default=128, type=int, help='N of fully-connected nodes.')
parser.add_argument('-p', '--p_dropout', default=0.2, type=float, help='p(drop) for dropout.')
parser.add_argument('-c', '--cuda', default=1, type=int, help='Which gpuid to use.')
args = parser.parse_args()

lr_init = args.lr_init
epochs = args.epochs
nodes  = args.nodes
layers = args.layers
p_dropout = args.p_dropout
n_queue = 10
batch_size = 36
#n_train = 2*180*1000

# Run on gpu if available
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

expt_name = 'FC%dx%d_pdrop%.2f_lr%s_epochs%d_stdscale'%(nodes, layers, p_dropout, str(lr_init), epochs)
#sample = 'DiPhotonJet_GluGluHToGG_Eta14_MGG90_FC'
#expt_name = '%s_%s'%(sample, expt_name)
print('Experiment:',expt_name)

# Read parquet file
#pqIn = pq.ParquetFile(args.infile)
#pqIn = args.infile
# Read .h5 file
df = pd.read_hdf(args.infile, key='df')
print("df.shape",df.shape)
rows=df.shape[0]
n_train = int(rows*0.7)
columns=df.shape[1]
idxs = np.random.permutation(rows)
#data = pqIn.read().to_pydict() # Reads whole dataset into memory

# Convert to numpy and split to training and validation sets
xVariables=['canJet1_pt', 'canJet3_pt', 'dRjjClose', 'dRjjOther', 'aveAbsEta']
#print(df[xVariables].iloc[idxs[:n_train]])
X_train = np.float64(df[xVariables].iloc[idxs[:n_train]])
y_train = np.  int64(df[ 'fourTag'].iloc[idxs[:n_train]]).reshape(-1,1)
X_val   = np.float64(df[xVariables].iloc[idxs[n_train:]])
y_val   = np.  int64(df[ 'fourTag'].iloc[idxs[n_train:]]).reshape(-1,1)
print('X_train.shape, y_train.shape:', X_train.shape, y_train.shape)
print('X_val.shape, y_val.shape:', X_val.shape, y_val.shape)

# Standardize inputs
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# Set up data loaders
dset_train = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(dataset=dset_train, batch_size=batch_size, shuffle=True, num_workers=n_queue, pin_memory=True)
dset_val = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
val_loader = DataLoader(dataset=dset_val, batch_size=batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
print('len(train_loader), len(val_loader):', len(train_loader), len(val_loader))

# Set up NN model and optimizer
fc = []
fc.append(nn.Linear(X_train.shape[-1], nodes))
fc.append(nn.ReLU())
fc.append(nn.Dropout(p=p_dropout))
for _ in range(layers):
    fc.append(nn.Linear(nodes, nodes))
    fc.append(nn.ReLU())
    fc.append(nn.Dropout(p=p_dropout))
fc.append(nn.Linear(nodes, 1))
fcnet = nn.Sequential(*fc).to(device)
#fcnet.load_state_dict(torch.load('MODELS/%s.pkl'%model_name)['model']) # load model from previous state
print('N trainable params:',sum(p.numel() for p in fcnet.parameters() if p.requires_grad))
optimizer = optim.Adam(fcnet.parameters(), lr=lr_init)

# Training loop
print_step = 1000
print(">> Training <<<<<<<<")
for e in range(epochs):

    epoch = e+1
    s = '>> Epoch %d/%d <<<<<<<<'%(epoch, epochs)

    # Run training
    fcnet.train()
    now = time.time()
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = fcnet(X)
        loss = F.binary_cross_entropy_with_logits(logits, y) # binary classification
        #loss = F.mse_loss(logits, y) # regression
        loss.backward()
        optimizer.step()
        #break
        if i % print_step == 0:
            binary_pred = logits.ge(0.).byte()
            accuracy = binary_pred.eq(y.byte()).float().mean().item()
            print(s+' (%d/%d) Train loss:%f, accuracy:%f'%(i, len(train_loader), loss.item(), accuracy))
    now = time.time() - now
    print(s+' Train time:%.2fs in %d steps'%(now, len(train_loader)))

    # Run Validation
    fcnet.eval()
    loss, accuracy = [], []
    y_pred, y_true = [], []
    now = time.time()
    for i, (X, y) in enumerate(val_loader):
        X, y = X.to(device), y.to(device)
        logits = fcnet(X)
        binary_pred = logits.ge(0.).byte()
        prob_pred = torch.sigmoid(logits)
        batch_loss = F.binary_cross_entropy_with_logits(logits, y, reduce=False) # binary classification
        #batch_loss = F.mse_loss(logits, y, reduce=False) # regression
        # Store metrics:
        accuracy.append(binary_pred.eq(y.byte()).float().tolist())
        loss.append(batch_loss.tolist())
        y_pred.append(prob_pred.tolist())
        y_true.append(y.tolist())

    now = time.time() - now
    accuracy = np.concatenate(accuracy)
    loss = np.concatenate(loss)
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    print(s+' Val time:%.2fs in %d steps'%(now, len(val_loader)))
    print(s+' Val loss:%f, accuracy:%f'%(loss.mean(), accuracy.mean()))

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(s+' Val ROC AUC: %f'%(roc_auc))

    # Save model file:
    #if roc_auc > roc_auc_best:
    #
    #    roc_auc_best = roc_auc
    #
    #    filename = 'MODELS/%s_epoch%d_auc%.4f.pkl'%(expt_name, epoch, roc_auc)
    #    model_dict = {'model': fcnet.state_dict(), 'optim': optimizer.state_dict()}
    #    torch.save(model_dict, filename)
