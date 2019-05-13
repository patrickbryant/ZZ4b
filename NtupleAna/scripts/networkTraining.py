import time, os, sys
import numpy as np
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

print_step = 100

def evaluate(model, loader, device):
    model.net.eval()
    y_pred, y_true, w_ordered = [], [], []
    for i, (P, A, y, w) in enumerate(loader):
        P, A, y, w = P.to(device), A.to(device), y.to(device), w.to(device)
        logits = model.net(P, A)#.view(-1,1)
        binary_pred = logits.ge(0.).byte()
        prob_pred = torch.sigmoid(logits)
        batch_loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction='none') # binary classification
        y_pred.append(prob_pred.tolist())
        y_true.append(y.tolist())
        w_ordered.append(w.tolist())
        if (i+1) % print_step == 0:
            sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(loader)))
            sys.stdout.flush()

    y_pred = np.transpose(np.concatenate(y_pred))[0]
    y_true = np.transpose(np.concatenate(y_true))[0]
    w_ordered = np.transpose(np.concatenate(w_ordered))[0]
    
    fpr, tpr, thr = roc_curve(y_true, y_pred, sample_weight=w_ordered)
    roc_auc = auc(fpr, tpr)

    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


#Function to perform training epoch
def train(model, train_loader, eval_train_loader, device, epochString):
    model.net.train()
    for i, (P, A, y, w) in enumerate(train_loader):
        P, A, y, w = P.to(device), A.to(device), y.to(device), w.to(device)
        model.optimizer.zero_grad()
        logits = model.net(P, A)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w) # binary classification
        loss.backward()
        model.optimizer.step()
        if (i+1) % print_step == 0:
            sys.stdout.write('\rTraining %3.0f%%     '%(float(i+1)*100/len(train_loader)))
            sys.stdout.flush()

    y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc = evaluate(model, eval_train_loader, device)

    bar=int((roc_auc-0.5)*200) if roc_auc > 0.5 else 0
    print('\r'+' '*len(epochString)+'       Training: %2.1f%%'%(roc_auc*100),("-"*bar)+"|")
    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


#function to check performance on validation set
def validate(model, loader, device, epochString):
    y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc = evaluate(model, loader, device)

    bar=int((roc_auc-0.5)*200) if roc_auc > 0.5 else 0
    print('\r'+epochString+' ROC Validation: %2.1f%%'%(roc_auc*100),("#"*bar)+"|", end = " ")
    return y_pred, y_true, w_ordered, fpr, tpr, thr, roc_auc


# #Simple ROC Curve plot function
# def plotROC(fpr, tpr, thr, name): #fpr = false positive rate, tpr = true positive rate
#     roc_auc = auc(fpr, tpr)
#     lumiRatio = 10
#     sigma = (tpr*sum_wS*lumiRatio) / np.sqrt(fpr*sum_wB*lumiRatio + 1)
#     iMaxSigma = np.argmax(sigma)
#     maxSigma = sigma[iMaxSigma]
#     f = plt.figure()
#     plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

#     #y=-x diagonal reference curve for zero mutual information ROC
#     plt.plot([0,1], [1,0], color='0.8', linestyle='--')

#     plt.xlabel('Rate( Signal to Signal )')
#     plt.ylabel('Rate( Background to Background )')
#     bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
#     plt.plot(tpr, 1-fpr)
#     plt.text(0.80, 1.07, "ROC AUC = %0.4f"%(roc_auc))
#     plt.scatter(rate_StoS, rate_BtoB, marker='o', c='r')
#     plt.text(rate_StoS+0.03, rate_BtoB-0.025, "Cut Based WP \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)
#     #plt.text(rate_StoS-0.20, rate_BtoB-0.03, "(%0.2f, %0.2f)"%(rate_StoS, rate_BtoB))
#     plt.scatter(tpr[iMaxSigma], (1-fpr[iMaxSigma]), marker='o', c='k')
#     plt.text(tpr[iMaxSigma]+0.03, (1-fpr[iMaxSigma])-0.025, "Optimal WP, SvB $>$ %0.2f \n (%0.2f, %0.2f), $%1.2f\sigma$ with 140fb$^{-1}$"%(thr[iMaxSigma], tpr[iMaxSigma], (1-fpr[iMaxSigma]), maxSigma), bbox=bbox)
#     #plt.text(tpr[iMaxSigma]+0.03, (1-fpr[iMaxSigma])+0.01, "(%0.2f, %0.2f) $%1.2f\sigma$ with 140fb$^{-1}$"%(tpr[iMaxSigma], (1-fpr[iMaxSigma]), maxSigma))
#     f.savefig(name)
#     plt.close(f)


# def plotNet(y_pred, y_true, w, name):
#     yS_pred, yB_pred = y_pred[y_true==1], y_pred[y_true==0]
#     wS,      wB      = w     [y_true==1], w     [y_true==0]
#     fig = pltHelper.plot([yS_pred, yB_pred], 
#                          [b/20.0 for b in range(21)],
#                          "NN Output (SvB)", "Events / Bin", 
#                          weights=[wS, wB],
#                          samples=['Signal','Background'],
#                          ratio=True,
#                          ratioRange=[0,5])
#     fig.savefig(name)
#     plt.close(fig)
    
# def epochString(epoch):
#     return ('>> %'+str(len(str(args.epochs+model.startingEpoch)))+'d/%d <<')%(epoch, args.epochs+model.startingEpoch)


