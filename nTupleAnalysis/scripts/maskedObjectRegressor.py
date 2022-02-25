import time, sys, gc
import uproot # https://github.com/scikit-hep/uproot3 is in lcg_99cuda
import uproot_methods
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from networks import *

class loaders:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name
        self.lv = dataset[:][0]
        self.w  = dataset[:][1]
        self.lv_reg = torch.zeros_like(self.lv)
        self.error = torch.zeros([self.lv.shape[0], self.lv.shape[2]])
        self.w_sum = self.w.sum()
        self.train = None
        self.infer = None
        self.loss = 0
        self.bs_scale = 2
        self.num_workers = 4

    def changeTrainBatchSize(self, newBatchSize=None):
        n = self.dataset.tensors[0].shape[0]
        currentBatchSize = self.train.batch_size
        if newBatchSize is None: newBatchSize = currentBatchSize*self.bs_scale
        if newBatchSize == currentBatchSize: return
        batchString = 'Change training batch size: %i -> %i (%i batches)'%(currentBatchSize, newBatchSize, n//newBatchSize )
        print(batchString)
        del self.train
        torch.cuda.empty_cache()
        gc.collect()
        self.train = DataLoader(dataset=self.dataset, batch_size=newBatchSize, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
        #self.bs_change.append(self.epoch)

    def update(self):
        self.PtEtaPhiM_error = self.lv_reg-self.lv
        self.PtEtaPhiM_error[:,2:3] = calcDeltaPhi(self.lv_reg, self.lv)

        # fractional error
        self.PtEtaPhiM_error[:,0:1] /= self.lv[:,0:1]
        self.PtEtaPhiM_error[:,1:2] /= 2.4
        self.PtEtaPhiM_error[:,2:3] /= np.pi
        # self.PtEtaPhiM_error[:,3:4] /= self.lv[:,3:4]
        
        self.PtEtaPhiM_error_mean = (self.PtEtaPhiM_error.mean(dim=2)*self.w.view(-1,1)).sum(dim=0)/self.w_sum
        self.PtEtaPhiM_error_abs_mean = (self.PtEtaPhiM_error.abs().mean(dim=2)*self.w.view(-1,1)).sum(dim=0)/self.w_sum

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', default='~/nobackup/ZZ4b/toyTrees/threeTag_toyTree.root',    type=str, help='Input dataset for training')
parser.add_argument('-k', '--kfold', default='0', help='training k-fold. use 0,1,2 for offset in selecting the third of data to reserve for validation')
args = parser.parse_args()

lr_init = 1e-2
epochs = 20

train_batch_size = 2**10
infer_batch_size = 2**15
num_workers = 4

kfold = int(args.kfold)
train_modulus = 3
train_portion = 2
t=uproot.open(args.train)['Tree']

pt, eta, phi, w = t.arrays(['jetPt', 'jetEta', 'jetPhi', 'weight'], outputtype=tuple) # mass set to zero in toyTrees
lv = uproot_methods.TLorentzVectorArray.from_ptetaphim(pt,eta,phi,0)
lv, w = torch.FloatTensor([pt, eta, phi, lv.mass]), torch.FloatTensor(w)
# lv is [feature, event, jet], want [event, feature, jet]
lv = lv.transpose(0,1)

print("Split into training and validation sets")
n = lv.shape[0]
idx = np.arange(n)
is_train = (idx+kfold)%train_modulus < train_portion
is_valid = ~is_train

dataset_train = TensorDataset(lv[is_train], w[is_train])
dataset_valid = TensorDataset(lv[is_valid], w[is_valid])

train_loaders = loaders(dataset_train, 'Training  ')
train_loaders.train = DataLoader(dataset=dataset_train, batch_size=train_batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
train_loaders.infer = DataLoader(dataset=dataset_train, batch_size=infer_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
valid_loaders = loaders(dataset_valid, 'Validation')
valid_loaders.infer = DataLoader(dataset=dataset_valid, batch_size=infer_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

model = missingObjectRegressor().to('cuda')
nTrainableParameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Parameters: %d'%nTrainableParameters)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, amsgrad=False)

loadCycler = cycler()
torch.autograd.set_detect_anomaly(True)
def train():
    model.train()

    print_step = 1+len(train_loaders.train)//200
    startTime = time.time()
    backpropTime = 0
    lossEstimate = 0
    for i, (X, w) in enumerate(train_loaders.train):
        X, w = X.to('cuda'), w.to('cuda')
        bs = X.shape[0]
        n  = X.shape[2]
        idx = torch.cat([torch.randperm(n).view(1,1,n) for _ in range(bs)])
        mask = (idx==0).repeat(1,n,1) 
        x = X[ mask].view(bs,4) # masked object is the target for regression
        X = X[~mask].view(bs,4,n-1) # unmasked objects

        optimizer.zero_grad()

        x_reg = model(X)

        x = PxPyPzM(x)
        error = ((x-x_reg)**2).sum(dim=1).sqrt()
        loss = (error.log10()*w).sum()/w.sum()

        # print(x[0])
        # print(x_reg[0])
        # print(error[0])
        # print((error*w).max(),'max error')
        # print((error*w).mean(),'mean error')
        # print((error*w).sum(),'(error*w).sum()')
        # print(w.sum(),'w.sum()')
        # print(w[0])
        # print(loss)
        # input()
        
        #perform backprop
        backpropStart = time.time()
        loss.backward()
        optimizer.step()
        backpropTime += time.time() - backpropStart

        thisLoss = loss.item()
        if not lossEstimate: lossEstimate = thisLoss
        lossEstimate = lossEstimate*0.98 + thisLoss*(1-0.98) # running average with 0.98 exponential decay rate

        if (i+1) % print_step == 0:
            elapsedTime = time.time() - startTime
            fractionDone = (i+1)/len(train_loaders.train)
            percentDone = fractionDone*100
            estimatedEpochTime = elapsedTime/fractionDone
            timeRemaining = estimatedEpochTime * (1-fractionDone)
            estimatedBackpropTime = backpropTime/fractionDone

            progressString  = str('\r%d Training %3.0f%% ('+loadCycler.next()+')  ')%(kfold, percentDone)
            progressString += str(('Loss: %0.4f | Time Remaining: %3.0fs | Estimated Epoch Time: %3.0fs | Estimated Backprop Time: %3.0fs ')%
                                 (lossEstimate, timeRemaining, estimatedEpochTime, estimatedBackpropTime))

            sys.stdout.write(progressString)
            sys.stdout.flush()


@torch.no_grad()
def evaluate(loader, epoch):
    model.eval()
    loader.loss = 0
    print_step = 1+len(loader.infer)//200
    startTime = time.time()
    nProcessed = 0
    for i, (X, w) in enumerate(loader.infer):
        X, w = X.to('cuda'), w.to('cuda')
        bs = X.shape[0]
        n  = X.shape[2]
        # idx = torch.cat([torch.randperm(n).view(1,1,n) for _ in range(bs)])
        # mask = (idx==0).repeat(1,n,1) 
        # x = X[ mask].view(bs,4) # masked object is the target for regression
        # X = X[~mask].view(bs,4,n-1) # unmasked objects

        x = X.clone().transpose(1,2).transpose(0,1).contiguous().view(n*bs,4)
        #x = torch.cat([X[:,idx] for idx in range(n)], dim=0)
        X = torch.cat([torch.cat([X[:,:,:idx],X[:,:,idx+1:]], dim=2) for idx in range(n)], dim=0)
        # print(X.shape)
        
        x_reg = model(X)
        # print(x_reg.shape)
        x = PxPyPzM(x)
        error = ((x-x_reg)**2).sum(dim=1).sqrt()
        loss = (error.log10()*w.repeat(n)).sum()/n
        
        loader.loss += loss.cpu()

        x_reg = PtEtaPhiM(x_reg)
        X_reg = x_reg.view(n,bs,4).transpose(0,1).transpose(1,2)
        error = error.view(n,bs).transpose(0,1)

        loader.lv_reg[nProcessed:nProcessed+bs] = X_reg.cpu()
        loader.error [nProcessed:nProcessed+bs] = error.cpu()

        nProcessed += bs

        if (i+1) % print_step == 0:
            elapsedTime = time.time() - startTime
            fractionDone = (i+1)/len(train_loaders.train)
            percentDone = fractionDone*100
            estimatedEpochTime = elapsedTime/fractionDone
            timeRemaining = estimatedEpochTime * (1-fractionDone)

            progressString  = str('\r%d Evaluating %3.0f%% ('+loadCycler.next()+')  ')%(kfold, percentDone)
            progressString += str(('| Time Remaining: %3.0fs | Estimated Time: %3.0fs ')%
                                 (timeRemaining, estimatedEpochTime))

            sys.stdout.write(progressString)
            sys.stdout.flush()

    loader.loss /= loader.w_sum
    loader.update()
    sys.stdout.write('\r'+' '*200)
    sys.stdout.flush()
    #print('\r', end='')
    print('\r%d >> %02d/%02d << %s | %1.2f |'%(kfold, epoch, epochs, loader.name, loader.loss), end='')
    print('| %3.0f | %3.0f | %3.0f | %3.0f |'%tuple(100*loader.PtEtaPhiM_error_mean), end='')
    print('| %3.0f | %3.0f | %3.0f | %3.0f |'%tuple(100*loader.PtEtaPhiM_error_abs_mean))


# previous=1e6
print()
for epoch in range(1,epochs+1):
    if epoch in [2,4,8,16]: 
        train_loaders.changeTrainBatchSize()
        gb_decay = 4 #2 if self.epoch in bs_milestones else 4
        print('setGhostBatches(%d)'%(model.nGhostBatches//gb_decay))
        model.setGhostBatches(model.nGhostBatches//gb_decay)
    train()
    evaluate(train_loaders, epoch)
    evaluate(valid_loaders, epoch)
    # if train_loaders.loss > previous: train_loaders.changeTrainBatchSize()
    # previous = train_loaders.loss
