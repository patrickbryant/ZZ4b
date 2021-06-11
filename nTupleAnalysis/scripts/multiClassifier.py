import time, os, sys
os.environ['CUDA_LAUNCH_BLOCKING']='1'
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
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp
#from nadam import NAdam
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
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
from functools import partial


import argparse

class classInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color

d4 = classInfo(abbreviation='d4', name=  'FourTag Data',       index=0, color='red')
d3 = classInfo(abbreviation='d3', name= 'ThreeTag Data',       index=1, color='orange')
t4 = classInfo(abbreviation='t4', name= r'FourTag $t\bar{t}$', index=2, color='green')
t3 = classInfo(abbreviation='t3', name=r'ThreeTag $t\bar{t}$', index=3, color='cyan')

zz = classInfo(abbreviation='zz', name= 'ZZ MC',          index=0, color='red')
zh = classInfo(abbreviation='zh', name= 'ZH MC',          index=1, color='orange')
tt = classInfo(abbreviation='tt', name=r'$t\bar{t}$ MC',  index=2, color='green')
mj = classInfo(abbreviation='mj', name= 'Multijet Model', index=3, color='cyan')

sg = classInfo(abbreviation='sg', name='Signal',     index=[zz.index, zh.index], color='blue')
bg = classInfo(abbreviation='bg', name='Background', index=[tt.index, mj.index], color='brown')


def getFrame(fileName, PS=None):
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])-2010
    thisFrame = pd.read_hdf(fileName, key='df')
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    n = thisFrame.shape[0]

    if PS:
        PS = int(PS)
        print("Cutting on Trigger and SB|CR|SR ...was ",n)
        thisFrame = thisFrame.loc[ (thisFrame[trigger] == True) & ((thisFrame.SB==True)|(thisFrame.CR==True)|(thisFrame.SR==True)) ]

        print("getFrame::PS is ",PS)
        PSOffset = 0
        idx_pass = []
        n = thisFrame.shape[0]
        for e in range(n):
            if (e+PSOffset)%PS < 1: 
                idx_pass.append(e)

        idx_pass = np.array(idx_pass)
        print("Prescaling by factor of ",PS,"...size was...",n)
        thisFrame = thisFrame.iloc[idx_pass]
        thisFrame[args.weightName] = thisFrame[args.weightName] * PS
    
    n = thisFrame.shape[0]
    print("Read",fileName,year,n)

    return thisFrame


def getFramesHACK(fileReaders,getFrame,dataFiles,PS=None):
    largeFiles = []
    print("dataFiles was:",dataFiles)
    for d in dataFiles:
        if Path(d).stat().st_size > 2e9:
            print("Large File",d)
            largeFiles.append(d)
            # dataFiles.remove(d) this caused problems because it modifies the list being iterated over
    for d in largeFiles:
        dataFiles.remove(d)
    results = fileReaders.map_async(partial(getFrame, PS=PS), sorted(dataFiles))
    frames = results.get()

    for f in largeFiles:
        print("read large file:",f)
        frames.append(getFrame(f,PS))

    return frames


trigger="passHLT"
def getFrameSvB(fileName):
    #print("Reading",fileName)    
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])-2010
    thisFrame = pd.read_hdf(fileName, key='df')
    fourTag = False if "data201" in fileName else True

    FvTName = args.FvTName
    if "ZZ4b201" in fileName: FvTName = "FvT"
    if "ZH4b201" in fileName: FvTName = "FvT"

    thisFrame = thisFrame.loc[ (thisFrame[trigger]==True) & (thisFrame['fourTag']==fourTag) & ((thisFrame['SB']==True)|(thisFrame['CR']==True)|(thisFrame['SR']==True)) & (thisFrame[FvTName]>0) ]#& (thisFrame.passXWt) ]
    #thisFrame = thisFrame.loc[ (thisFrame[trigger]==True) & (thisFrame['fourTag']==fourTag) & ((thisFrame['SR']==True)) & (thisFrame.FvT>0) ]#& (thisFrame.passXWt) ]
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    if "ZZ4b201" in fileName: 
        index = zz.index
        #index = sg.index
        thisFrame['zz'] = pd.Series(np. ones(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['zh'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['tt'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['mj'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
    if "ZH4b201" in fileName: 
        index = zh.index
        #index = sg.index
        thisFrame['zz'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['zh'] = pd.Series(np. ones(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['tt'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['mj'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
    if "TTTo" in fileName:
        index = tt.index
        #index = bg.index
        thisFrame['zz'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['zh'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['tt'] = pd.Series(np. ones(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['mj'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
    if "data201" in fileName:
        index = mj.index
        #index = bg.index
        thisFrame['zz'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['zh'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['tt'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
        thisFrame['mj'] = pd.Series(np. ones(thisFrame.shape[0], dtype=bool), index=thisFrame.index)
    thisFrame['target']  = pd.Series(index*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    n = thisFrame.shape[0]
    print("Read",fileName,n,year)
    return thisFrame


class cycler:
    def __init__(self,options=['-','\\','|','/']):
        self.cycle=0
        self.options=options
        self.m=len(self.options)
    def next(self):
        self.cycle = (self.cycle + 1)%self.m
        return self.options[self.cycle]

class nameTitle:
    def __init__(self,name='',title='',aux='',abbreviation=''):
        self.name = name
        self.title= title
        self.aux  = aux
        self.abbreviation = abbreviation if abbreviation else name


def increaseBatchSize(loader, factor=4):
    currentBatchSize = loader.batch_sampler.batch_size
    loader.batch_sampler = BatchSampler(loader.sampler, currentBatchSize*factor, loader.drop_last)
    #loader.batch_size = loader.batch_size*2


queue = mp.Queue()
def runTraining(offset, df, df_control, modelName=''):
    model = modelParameters(modelName, offset)
    print("Setup training/validation tensors")
    model.trainSetup(df, df_control)
    #model initial state
    #model.makePlots()
    # Training loop
    for e in range(model.startingEpoch, model.epochs): 
        model.runEpoch()

    print()
    print(offset,">> DONE <<")
    if model.foundNewBest: print(offset,"Minimum Loss =", model.training.loss_best)
    queue.put(model.modelPkl)


@torch.no_grad()
def averageModels(models, results):
    for model in models: model.net.eval()

    y_pred, y_true, w_ordered = np.ndarray((results.n, models[0].nClasses), dtype=np.float), np.zeros(results.n, dtype=np.float), np.zeros(results.n, dtype=np.float)
    cross_entropy = np.zeros(results.n, dtype=np.float)
    q_score = np.ndarray((results.n, 3), dtype=np.float)
    print_step = len(results.evalLoader)//200+1
    nProcessed = 0

    if models[0].classifier in ['FvT']:
        r_std = np.zeros(results.n, dtype=np.float)
    else:
        r_std = None

    for i, (J, O, A, y, w, R) in enumerate(results.evalLoader):
        nBatch = w.shape[0]
        J, O, A, y, w = J.to(models[0].device), O.to(models[0].device), A.to(models[0].device), y.to(models[0].device), w.to(models[0].device)
        R = R.to(models[0].device)

        outputs = [model.net(J, O, A) for model in models]
        logits   = torch.stack([output[0] for output in outputs])
        q_scores = torch.stack([output[1] for output in outputs])
        y_preds  = F.softmax(logits, dim=-1)
        
        if r_std is not None:
            # get reweight for each offset
            rs = (y_preds[:,:,d4.index] - y_preds[:,:,t4.index]) / y_preds[:,:,d3.index]
            # get variance of the reweights across offsets
            r_var = rs.var(dim=0).cpu() # *3/2 inflation term to account for overlap of training sets?
            r_std[nProcessed:nProcessed+nBatch] = r_var.sqrt()

        logits   = logits  .mean(dim=0)
        q_scores = q_scores.mean(dim=0)
        cross_entropy [nProcessed:nProcessed+nBatch] = F.cross_entropy(logits, y, weight=models[0].wC, reduction='none').cpu().numpy()
        y_pred        [nProcessed:nProcessed+nBatch] = F.softmax(logits, dim=-1).cpu().numpy()
        y_true        [nProcessed:nProcessed+nBatch] = y.cpu()
        q_score       [nProcessed:nProcessed+nBatch] = q_scores.cpu().numpy()
        w_ordered     [nProcessed:nProcessed+nBatch] = w.cpu()
        nProcessed+=nBatch

        if int(i+1) % print_step == 0:
            percent = float(i+1)*100/len(results.evalLoader)
            sys.stdout.write('\rEvaluating %3.0f%%     '%(percent))
            sys.stdout.flush()

    loss = (w_ordered * cross_entropy).sum()/w_ordered.sum()

    results.update(y_pred, y_true, q_score, w_ordered, cross_entropy, loss, doROC=False)
    if r_std is not None:
        results.r_std = r_std





parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--data', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('--data4b',     default='', help="Take 4b from this file if given, otherwise use --data for both 3-tag and 4-tag")
parser.add_argument('--data3bWeightSF',     default=None, help="Take 4b from this file if given, otherwise use --data for both 3-tag and 4-tag")
parser.add_argument('-t', '--ttbar',      default='',    type=str, help='Input MC ttbar file in hdf5 format')
parser.add_argument('--ttbar4b',          default=None, help="Take 4b ttbar from this file if given, otherwise use --ttbar for both 3-tag and 4-tag")
parser.add_argument('--ttbarPS',          default=None, help="")
parser.add_argument('-s', '--signal',     default='', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-c', '--classifier', default='', type=str, help='Which classifier to train: FvT, ZHvB, ZZvB, M1vM2.')
parser.add_argument(      '--architecture', default='ResNet', type=str, help='classifier architecture to use')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-o', '--outputName', default='', type=str, help='Prefix to output files.')
#parser.add_argument('-l', '--lrInit', default=4e-3, type=float, help='Initial learning rate.')
parser.add_argument('-p', '--pDropout', default=0.4, type=float, help='p(drop) for dropout.')
parser.add_argument(      '--layers', default=3, type=int, help='N of fully-connected layers.')
parser.add_argument('-n', '--nodes', default=32, type=int, help='N of fully-connected nodes.')
parser.add_argument('--cuda', default=0, type=int, help='Which gpuid to use.')
parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
parser.add_argument(      '--onnx', dest="onnx",  default=False, action="store_true", help='Export model to onnx')
parser.add_argument(      '--train',  dest="train",  action="store_true", default=False, help="Train the model(s)")
parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
parser.add_argument(      '--storeEvent',     dest="storeEvent",     default="0", help="store the network response in a numpy file for the specified event")
parser.add_argument(      '--storeEventFile', dest="storeEventFile", default=None, help="store the network response in this file for the specified event")
parser.add_argument('--weightName', default="mcPseudoTagWeight", help='Which weights to use for JCM.')
parser.add_argument('--FvTName', default="FvT", help='Which FvT weights to use for SvB Training.')
parser.add_argument('--trainOffset', default='1', help='training offset. Use comma separated list to train with multiple offsets in parallel.')
parser.add_argument('--updatePostFix', default="", help='Change name of the classifier weights stored .')
#parser.add_argument('--updatePostFix', default="", help='Change name of the classifier weights stored .')

#parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

n_queue = 4
eval_batch_size = 2**15

# https://arxiv.org/pdf/1711.00489.pdf much larger training batches and learning rate inspired by this paper
train_batch_size = 2**10#9#10#11
max_train_batch_size = train_batch_size*64
lrInit = 1.0e-2#4e-3
max_patience = 1
fixedSchedule = True

bs_scale=2
lr_scale=0.5
bs_milestones=[1,3,6,10]
lr_milestones= bs_milestones + [15,16,17,18,19,20,21,22,23,24]
#lr_milestones=                 [15,16,17,18,19,20,21,22,23,24]

train_numerator = 2
train_denominator = 3
train_fraction = train_numerator/train_denominator
valid_fraction = 1-train_fraction
train_offset = [int(offset) for offset in args.trainOffset.split(',')] #int(args.trainOffset)

print_step = 2
rate_StoS, rate_BtoB = None, None
barScale=200
barMin=0.5

#fileReaders = multiprocessing.Pool(10)
fileReaders = mp.Pool(10)

loadCycler = cycler()

classifier = args.classifier
weightName = args.weightName
FvTForSvBTrainingName = args.FvTName
df_control = None

#wC = torch.FloatTensor([1, 1, 1, 1])#.to("cuda")

if classifier in ['SvB', 'SvB_MA']:
    #eval_batch_size = 2**15
    #train_batch_size = 2**10
    #lrInit = 0.8e-2

    barMin=0.84
    barScale=1000
    weight = 'weight'
    print("Using weight:",weight,"for classifier:",classifier) 
    yTrueLabel = 'target'

    classes = [zz,zh,tt,mj]
    # set class index
    for i,c in enumerate(classes): 
        c.index=i
    sg.index = [zz.index,zh.index]
    bg.index = [tt.index,mj.index]

    eps = 0.0001

    updateAttributes = [
        nameTitle('pzz',    classifier+args.updatePostFix+'_pzz'),
        nameTitle('pzh',    classifier+args.updatePostFix+'_pzh'),
        nameTitle('ptt',    classifier+args.updatePostFix+'_ptt'),
        nameTitle('pmj',    classifier+args.updatePostFix+'_pmj'),
        nameTitle('psg',    classifier+args.updatePostFix+'_ps'),
        nameTitle('pbg',    classifier+args.updatePostFix+'_pb'),
        nameTitle('q_1234', classifier+args.updatePostFix+'_q_1234'),
        nameTitle('q_1324', classifier+args.updatePostFix+'_q_1324'),
        nameTitle('q_1423', classifier+args.updatePostFix+'_q_1423'),
        ]

    if args.train or (not args.update and not args.storeEventFile and not args.onnx):
        # Read .h5 files
        dataFiles = glob(args.data)

        for d4b in args.data4b.split(","):
            dataFiles += glob(args.data4b)    

        results = fileReaders.map_async(getFrameSvB, sorted(dataFiles))
        frames = results.get()
        dfDB = pd.concat(frames, sort=False)
        dfDB[weight] = dfDB[weightName] * dfDB[FvTForSvBTrainingName]

        print("Setting dfDB weight:",weight,"to: ",weightName," * ",FvTForSvBTrainingName) 
        nDB = dfDB.shape[0]
        wDB = np.sum( dfDB[weight] )
        print("nDB",nDB)
        print("wDB",wDB)

        # Read .h5 files
        ttbarFiles = glob(args.ttbar)
        if args.ttbar4b:
            ttbarFiles += glob(args.ttbar4b)    


        results = fileReaders.map_async(getFrameSvB, sorted(ttbarFiles))
        frames = results.get()
        dfT = pd.concat(frames, sort=False)

        nT = dfT.shape[0]
        wT = dfT[weight].sum()
        print("nT",nT)
        print("wT",wT)

        dfB = pd.concat([dfDB, dfT], sort=False)

        results = fileReaders.map_async(getFrameSvB, sorted(glob(args.signal)))
        frames = results.get()
        dfS = pd.concat(frames, sort=False)

        nS      = dfS.shape[0]
        nB      = dfB.shape[0]
        print("nS",nS)
        print("nB",nB)

        # compute relative weighting for S and B
        nzz, wzz = dfS.zz.sum(), dfS[dfS.zz][weight].sum()
        nzh, wzh = dfS.zh.sum(), dfS[dfS.zh][weight].sum()
        sum_wS = dfS[weight].sum()
        sum_wB = dfB[weight].sum()
        print("sum_wS",sum_wS)
        print("sum_wB",sum_wB)
        print("nzz = %7d, wzz = %6.1f"%(nzz,wzz))
        print("nzh = %7d, wzh = %6.1f"%(nzh,wzh))

        # sum_wStoS = np.sum(np.float32(dfS.loc[dfS[ZB+'SR']==True ][weight]))
        # sum_wBtoB = np.sum(np.float32(dfB.loc[dfB[ZB+'SR']==False][weight]))
        # print("sum_wStoS",sum_wStoS)
        # print("sum_wBtoB",sum_wBtoB)
        # rate_StoS = sum_wStoS/sum_wS
        # rate_BtoB = sum_wBtoB/sum_wB
        # print("Cut Based WP:",rate_StoS,"Signal Eff.", rate_BtoB,"1-Background Eff.")

        #normalize signal to background
        dfS.loc[dfS.zz, weight] = dfS[dfS.zz][weight]*sum_wB/wzz
        dfS.loc[dfS.zh, weight] = dfS[dfS.zh][weight]*sum_wB/wzh

        df = pd.concat([dfB, dfS], sort=False)


if classifier in ['FvT','DvT3', 'DvT4', 'M1vM2']:
    barMin = 0.5
    barScale=100
    if classifier == 'M1vM2': barMin, barScale = 0.50,  500
    if classifier == 'DvT3' : barMin, barScale = 0.80,  100
    if classifier == 'DvT4' : barMin, barScale = 0.80,  100
    if classifier == 'FvT'  : barMin, barScale = 0.62, 1000
    weight = weightName

    yTrueLabel = 'target'

    classes = [d4,d3,t4,t3]
    if classifier in ['DvT3']:
        classes = [d3,t3]

    if classifier in ['DvT4']:
        classes = [d4,t4]


    # set class index
    for i,c in enumerate(classes): 
        c.index=i
        

    eps = 0.0001

    if classifier in ['M1vM2']: yTrueLabel = 'y_true'
    if classifier == 'M1vM2'  :  weight = 'weight'


    print("Using weight:",weight,"for classifier:",classifier)

    if classifier in ['FvT']: 

        if args.updatePostFix == "":
            updateAttributes = [
                nameTitle('r',      classifier+args.updatePostFix),
                nameTitle('r_std',  classifier+args.updatePostFix+'_std'),
                nameTitle('pd4',    classifier+args.updatePostFix+'_pd4'),
                nameTitle('pd3',    classifier+args.updatePostFix+'_pd3'),
                nameTitle('pt4',    classifier+args.updatePostFix+'_pt4'),
                nameTitle('pt3',    classifier+args.updatePostFix+'_pt3'),
                nameTitle('pm4',    classifier+args.updatePostFix+'_pm4'),
                nameTitle('pm3',    classifier+args.updatePostFix+'_pm3'),
                nameTitle('p4',     classifier+args.updatePostFix+'_p4'),
                nameTitle('p3',     classifier+args.updatePostFix+'_p3'),
                nameTitle('pd',     classifier+args.updatePostFix+'_pd'),
                nameTitle('pt',     classifier+args.updatePostFix+'_pt'),
                nameTitle('q_1234', classifier+args.updatePostFix+'_q_1234'),
                nameTitle('q_1324', classifier+args.updatePostFix+'_q_1324'),
                nameTitle('q_1423', classifier+args.updatePostFix+'_q_1423'),
            ]

        else:
            updateAttributes = [
                nameTitle('r',      classifier+args.updatePostFix),
                nameTitle('pt4',    classifier+args.updatePostFix+'_pt4'),
                nameTitle('pt3',    classifier+args.updatePostFix+'_pt3'),
                nameTitle('pd3',    classifier+args.updatePostFix+'_pd3'),
            ]

    if classifier in ['DvT3']:
        updateAttributes = [
            nameTitle('r',      classifier+args.updatePostFix),
            nameTitle('pt3',    classifier+args.updatePostFix+'_pt3'),
            nameTitle('pd3',    classifier+args.updatePostFix+'_pd3'),
        ]

    if classifier in ['DvT4']:
        updateAttributes = [
            nameTitle('r',      classifier+args.updatePostFix),
            nameTitle('pt4',    classifier+args.updatePostFix+'_pt4'),
            nameTitle('pd4',    classifier+args.updatePostFix+'_pd4'),
        ]


            
    if args.train or (not args.update and not args.storeEventFile and not args.onnx):


        # Read .h5 files
        dataFiles = glob(args.data)
        frames = getFramesHACK(fileReaders,getFrame,dataFiles)
        dfD = pd.concat(frames, sort=False)

        if args.data4b:
            dfD.fourTag = False
            #dfD = dfD.loc[~dfD.fourTag] # this line does nothing since dfD.fourTag was set to False for all entries on the previous line...
            data4bFiles = []
            for d4b in args.data4b.split(","):
                data4bFiles += glob(d4b)

            frames = getFramesHACK(fileReaders,getFrame,data4bFiles)
            frames = pd.concat(frames, sort=False)
            frames.fourTag = True
            frames.mcPseudoTagWeight /= frames.pseudoTagWeight
            dfD = pd.concat([dfD,frames], sort=False)


        # Read .h5 files
        ttbarFiles = glob(args.ttbar)
        frames = getFramesHACK(fileReaders,getFrame,ttbarFiles,PS=args.ttbarPS)
        dfT = pd.concat(frames, sort=False)

        if args.ttbar4b:
            dfT.fourTag = False
            #dfT = dfT.loc[~dfT.fourTag] # this line does nothing since dfT.fourTag is False for all entries... (see previous line)
            ttbar4bFiles = glob(args.ttbar4b)
            frames = getFramesHACK(fileReaders,getFrame,ttbar4bFiles)
            frames = pd.concat(frames, sort=False)
            frames.fourTag = True
            frames.mcPseudoTagWeight /= frames.pseudoTagWeight
            dfT = pd.concat([dfT,frames], sort=False)


        negative_ttbar = dfT.weight<0
        df_negative_ttbar = dfT.loc[negative_ttbar]
        wtn = df_negative_ttbar[weight].sum()
        # print("Move negative weight ttbar events (sum_w = %f) to data"%wtn)
        # dfT = dfT.loc[~negative_ttbar] # tilde negates boolean series, ie it is a NOT logical operator
        # df_negative_ttbar[weight] *= -1
        # dfD = pd.concat([dfD, df_negative_ttbar], sort=False)

        print("Add true class labels to data")
        dfD['d4'] =  dfD.fourTag
        dfD['d3'] = ~dfD.fourTag
        dfD['t4'] = False #pd.Series(np.zeros(dfD.shape[0], dtype=bool), index=dfD.index)
        dfD['t3'] = False #pd.Series(np.zeros(dfD.shape[0], dtype=bool), index=dfD.index)

        if args.data3bWeightSF:
            print("Scaling data3b weights by",float(args.data3bWeightSF))
            print("was", dfD.loc[dfD.d3, weight])
            dfD.loc[df.d3, weight] = dfD[df.d3][weight]*float(args.data3bWeightSF)
            print("now", dfD.loc[dfD.d3, weight])


        print("Add true class labels to ttbar MC")
        dfT['t4'] =  dfT.fourTag
        dfT['t3'] = ~dfT.fourTag
        dfT['d4'] = False #pd.Series(np.zeros(dfT.shape[0], dtype=bool), index=dfT.index)
        dfT['d3'] = False #pd.Series(np.zeros(dfT.shape[0], dtype=bool), index=dfT.index)
        #dfT[weight] *= 0.5 #749.5/831.76
        #dfT.loc[dfT.weight<0, weight] *= 0

        print("concatenate data and ttbar dataframes")
        df = pd.concat([dfD, dfT], sort=False)


        target_string = ', '.join(['%s=%d'%(c.abbreviation,c.index) for c in classes])
        print("add encoded target: "+target_string)
        if classifier in ['FvT']:
            df['target'] = d4.index*df.d4 + d3.index*df.d3 + t4.index*df.t4 + t3.index*df.t3 # classes are mutually exclusive so the target computed in this way is 0,1,2 or 3.
        if classifier in ['DvT3']:
            df['target'] = d3.index*df.d3 + t3.index*df.t3 # classes are mutually exclusive so the target computed in this way is 0,1,2 or 3.
        if classifier in ['DvT4']:
            df['target'] = d4.index*df.d4 + t4.index*df.t4 # classes are mutually exclusive so the target computed in this way is 0,1,2 or 3.


        
        #print("add passXWt")
        #df['passXWt'] = (pow(df.xbW - 0.25,2) + pow(df.xW - 0.5,2)) > 3
        #passXWt = t->rWbW > 3;
        #rWbW = sqrt(pow((xbW-0.25),2) + pow((W->xW-0.5),2)); // after minimizing, the ttbar distribution is centered around ~(0.5, 0.25) with surfaces of constant density approximiately constant radii

        print("Apply event selection")
        if classifier == 'FvT':
            df_control = df.loc[ df[trigger] & df.CR ]
            df = df.loc[ df[trigger] & (df.SB | ((df.CR|df.SR) & (~df.d4))) ]
        if classifier == 'DvT3':
            df = df.loc[ df[trigger] & (df.d3|df.t3|df.t4) & (df.SB|df.CR|df.SR) ]#& (df.passXWt) ]# & (df[weight]>0) ]
        if classifier == 'DvT4':
            df = df.loc[ df[trigger] & df.SB ]#& (df.passXWt) ]# & (df[weight]>0) ]

        keep_fraction = 1/10
        print("Only keep %f of t3 so that it has comparable stats to the d3 sample"%keep_fraction)
        keep = (~df.t3) | (np.random.rand(df.shape[0]) < keep_fraction) # a random third of t3 events will be kept set
        keep_fraction = (keep & df.t3).sum()/df.t3.sum() # update keep_fraction with actual fraction instead of target fraction
        print("keep fraction",keep_fraction)
        df = df[keep]
        df.loc[df.t3, weight] = df[df.t3][weight] / keep_fraction

        n = df.shape[0]

        nd4, wd4 = df.d4.sum(), df[df.d4][weight].sum()
        nd3, wd3 = df.d3.sum(), df[df.d3][weight].sum()
        nt4, wt4 = df.t4.sum(), df[df.t4][weight].sum()
        nt3, wt3 = df.t3.sum(), df[df.t3][weight].sum()

        awd4 = wd4/nd4
        awd3 = wd3/nd3
        awt4 = wt4/nt4
        awt3 = wt3/nt3

        w = wd4+wd3+wt4+wt3

        fC =      torch.FloatTensor([wd4/w, wd3/w, wt4/w, wt3/w])
        #wC = 0.25*torch.FloatTensor([w/wd4, w/wd3, w/wt4, w/wt3]).to("cuda")
        #fC =      torch.FloatTensor([0.25, 0.25, 0.25, 0.25])
        # if classifier == 'DvT3': 
        #    wC[d4.index] *= 0
        #    #wC[t4.index] *= 0

        print("nd4 = %7d, wd4 = %6.1f, <w> = %5.3f"%(nd4,wd4,awd4))
        print("nd3 = %7d, wd3 = %6.1f, <w> = %5.3f"%(nd3,wd3,awd3))
        print("nt4 = %7d, wt4 = %6.1f, <w> = %5.3f"%(nt4,wt4,awt4))
        print("nt3 = %7d, wt3 = %6.1f, <w> = %5.3f"%(nt3,wt3,awt3))
        print("wtn = %6.1f"%(wtn))
        print("fC:",fC)
        #print("wC:",wC)
        
        wd4_SB = df[df.d4 & df.SB][weight].sum()
        wd3_SB = df[df.d3 & df.SB][weight].sum()
        wt4_SB = df[df.t4 & df.SB][weight].sum()
        wt3_SB = df[df.t3 & df.SB][weight].sum()
        
        print("SB Normalization = wd4_SB/(wd3_SB-wt3_SB+wt4_SB)")
        print("                 = %0.0f/(%0.0f-%0.0f+%0.0f)"%(wd4_SB,wd3_SB,wt3_SB,wt4_SB))
        print("                 = %4.2f +/- %5.3f (%5.3f validation stat uncertainty, norm should converge to about this precision)"%(wd4_SB/(wd3_SB-wt3_SB+wt4_SB), wd4_SB**-0.5, (wd4_SB/valid_fraction)**-0.5))

        #df = df.loc[(df.nSelJets==4)]
        #df = df.loc[(df.year==2018)]




#from networkTraining import *

class roc_data:
    def __init__(self, y_true, y_pred, weights, trueName, falseName, title=''):
        self.fpr, self.tpr, self.thr = roc_curve(y_true, y_pred, sample_weight=weights)
        self.auc = roc_auc_with_negative_weights(y_true, y_pred, weights=weights)
        self.title = title
        self.trueName  = trueName
        self.falseName = falseName
        wS = None
        self.maxSigma=None
        if "Background" in self.falseName:
            lumiRatio = 1#140/59.6
            if "ZZ" in self.trueName: 
                wS = wzz
                self.pName = "P($ZZ$)"
            if "ZH" in self.trueName: 
                wS = wzh
                self.pName = "P($ZH$)"
            if "Signal" in self.trueName: 
                wS = wzz+wzh
                self.pName = "P(Signal)"
            self.S = self.tpr*wS*lumiRatio
            self.S[self.tpr<0.10] = 0 # require at least 10% signal acceptance 
            self.B = self.fpr*sum_wB*lumiRatio
            self.B[self.fpr<0.001] = 1.0e9 # require at least 0.1% background acceptance 
            sigma = self.S / np.sqrt( self.S+self.B + (0.02*self.B)**2 + (0.1*self.S)**2 ) # include 2% background systematic and 10% signal systematic
            self.iMaxSigma = np.argmax(sigma)
            self.maxSigma = sigma[self.iMaxSigma]
            self.S = self.S[self.iMaxSigma]
            self.B = self.B[self.iMaxSigma]
            self.tprMaxSigma, self.fprMaxSigma, self.thrMaxSigma = self.tpr[self.iMaxSigma], self.fpr[self.iMaxSigma], self.thr[self.iMaxSigma]


class loaderResults:
    def __init__(self, name, classes):
        self.name = name
        self.classes = classes
        self.extra_classes = []
        if classifier in ['SvB', 'SvB_MA']:
            self.extra_classes += [sg, bg]
        self.class_abbreviations = [cl.abbreviation for cl in self.classes]
        self.trainLoader = None
        self. evalLoader = None
        self.smallBatchLoader = None
        self.largeBatchLoader = None
        self.y_true = None
        self.y_pred = None
        self.q_score = None
        self.q_1234, self.q_1324, self.q_1423 = None, None, None
        self.n      = None
        self.w      = None
        self.w_sum  = None
        self.roc1, self.roc2 = None, None #[0 for cl in classes]
        self.loss = 1e6
        self.loss_min = 1e6
        self.loss_prev = None
        self.loss_best = 1e6
        self.roc_auc_best = None
        self.sum_w_S = None
        self.probNorm_StoB = None
        self.probNorm_BtoS = None
        self.probNormRatio_StoB = None
        self.norm_data_over_model = None
        self.r_std = None
        self.r_max = None

        for cl in self.classes:
            setattr(self, 'p'+cl.abbreviation, None)
            setattr(self, 'w'+cl.abbreviation, None)
            setattr(self, 'ce'+cl.abbreviation, None)

    def update(self, y_pred, y_true, q_score, w_ordered, cross_entropy, loss, doROC=False):
        self.y_pred = y_pred
        self.y_true = y_true
        self.q_score =  q_score
        self.w      = w_ordered
        self.cross_entropy = cross_entropy
        self.loss   = loss
        if loss is not None:
            self.loss_min = loss if loss < (self.loss_min - 1e-4) else self.loss_min
        self.w_sum  = self.w.sum()

        if q_score is not None:
            self.q_1234 = self.q_score[:,0]
            self.q_1324 = self.q_score[:,1]
            self.q_1423 = self.q_score[:,2]

        self.class_loss = []
        for cl in self.classes:
            setattr(self, 'w'+cl.abbreviation, self.w[self.y_true==cl.index])
            setattr(self, 'w%sn'%cl.abbreviation, getattr(self, 'w'+cl.abbreviation)[getattr(self, 'w'+cl.abbreviation)<0])
            setattr(self, 'p'+cl.abbreviation, self.y_pred[:,cl.index])

            if cross_entropy is not None:
                setattr(self, 'ce'+cl.abbreviation, self.cross_entropy[self.y_true==cl.index])
                self.class_loss.append( (getattr(self, 'w'+cl.abbreviation)*getattr(self, 'ce'+cl.abbreviation)).sum()/self.w_sum )

        
        if 'd3' in self.class_abbreviations and 't3' in self.class_abbreviations:
            self.pm3 = self.pd3 - self.pt3
            self.p3  = self.pd3 + self.pt3
            for cl in self.classes:
                setattr(self, 'p%sm3'%cl.abbreviation, self.pm3[self.y_true==cl.index])
        if 'd4' in self.class_abbreviations and 't4' in self.class_abbreviations:
            self.pm4 = self.pd4 - self.pt4
            self.p4  = self.pd4 + self.pt4
            for cl in self.classes:
                setattr(self, 'p%sm4'%cl.abbreviation, self.pm4[self.y_true==cl.index])
        if 'd3' in self.class_abbreviations and 'd4' in self.class_abbreviations:
            self.pd  = self.pd3 + self.pd4
        if 't3' in self.class_abbreviations and 't4' in self.class_abbreviations:
            self.pt  = self.pt3 + self.pt4
        if 'tt' in self.class_abbreviations and 'mj' in self.class_abbreviations:
            self.wbg = self.w[(self.y_true==tt.index)|(self.y_true==mj.index)]
            self.pbg = self.ptt + self.pmj
        if 'zz' in self.class_abbreviations and 'zh' in self.class_abbreviations:
            self.wsg = self.w[(self.y_true==zz.index)|(self.y_true==zh.index)]
            self.psg = self.pzz + self.pzh


        # Compute reweight factor
        if   'd4' in self.class_abbreviations and 't4' in self.class_abbreviations and 'd3' in self.class_abbreviations:
            self.r = np.divide(self.pm4, self.pd3, out=np.zeros_like(self.pm4), where=self.pd3!=0) # self.pm4/self.pd3
        elif 'd3' in self.class_abbreviations and 't3' in self.class_abbreviations:
            self.r = self.pm3/self.pd3
        elif 'd4' in self.class_abbreviations and 't4' in self.class_abbreviations:
            self.r = self.pm4/self.pd4


        if hasattr(self, 'r'):
            # clip reweight
            r_large = abs(self.r)>1000
            r_large = r_large & (self.y_true==d3.index)
            if r_large.any():
                print()
                print('self.r[r_large]\n',self.r[r_large])
                print('self.y_true[r_large]\n',self.y_true[r_large])
                print('np.argmax(self.y_pred[r_large], axis=1)\n',np.argmax(self.y_pred[r_large], axis=1))
                print('self.w[r_large]\n',self.w[r_large])
                if 'd4' in self.class_abbreviations and 't4' in self.class_abbreviations:
                    print('self.pd4[r_large]\n',self.pd4[r_large])
                    print('self.pt4[r_large]\n',self.pt4[r_large])
                    print('self.pm4[r_large]\n',self.pm4[r_large])
                if 'd3' in self.class_abbreviations and 't3' in self.class_abbreviations:
                    print('self.pd3[r_large]\n',self.pd3[r_large])
                    print('self.pt3[r_large]\n',self.pt3[r_large])
                    print('self.pm3[r_large]\n',self.pm3[r_large])
                print('self.cross_entropy[r_large]\n',self.cross_entropy[r_large])
            self.r = np.clip(self.r, -20, 20)
            #Compute multijet weights for each class
            for cl in self.classes:
                setattr(self, 'r'+cl.abbreviation, self.r[self.y_true==cl.index])

        #regressed probabilities for each class to be each class
        for cl1 in self.classes+self.extra_classes:
            for cl2 in self.classes+self.extra_classes:
                try:
                    mask = (self.y_true==cl1.index[0]) | (self.y_true==cl1.index[1])
                except TypeError:
                    mask = (self.y_true==cl1.index)
                try:
                    pred = self.y_pred[mask][:,cl2.index[0]] + self.y_pred[mask][:,cl2.index[1]]
                except TypeError:
                    pred = self.y_pred[mask][:,cl2.index]
                setattr(self, 'p'+cl1.abbreviation+cl2.abbreviation, pred)

        #Compute normalization of the reweighted background model
        try:
            self.r_max = self.rd3.max() if self.rd3.max() > abs(self.rd3.min()) else self.rd3.min()
            if   'd4' in self.class_abbreviations: # reweighting three-tag data to four-tag multijet
                self.norm_model = ( self.wd3 * self.rd3 ).sum() + self.wt4.sum()
                self.norm_data_over_model = self.wd4.sum()/self.norm_model if self.norm_model>0 else 0
            elif 'd3' in self.class_abbreviations: # reweighting three-tag data to three-tag multijet
                self.norm_model = ( self.wd3 * self.rd3 ).sum() 
                self.norm_data_over_model = (self.wd3.sum()-self.wt3.sum())/self.norm_model if self.norm_model>0 else 0
        except:
            self.r_max = 0
            self.norm_model = 0
            self.norm_data_over_model = 0

        if doROC:
            if classifier in ['DvT3']:
                self.roc_t3 = roc_data(np.array(self.y_true==t3.index, dtype=np.float), 
                                       self.y_pred[:,t3.index], 
                                       self.w,
                                       r'ThreeTag $t\bar{t}$ MC',
                                       'ThreeTag Data')
                self.roc1 = self.roc_t3


            if classifier in ['DvT4']:
                self.roc_t4 = roc_data(np.array(self.y_true==t4.index, dtype=np.float), 
                                       self.y_pred[:,t4.index], 
                                       self.w,
                                       r'fourTag $t\bar{t}$ MC',
                                       'FourTag Data')
                
                self.roc1 = self.roc_t4


            if classifier in ['FvT']:
                isData = (self.y_true==d3.index)|(self.y_true==d4.index)

                self.roc_d43 = roc_data(np.array(self.y_true[isData]==d4.index, dtype=np.float), 
                                        self.y_pred[isData,t4.index]+self.y_pred[isData,d4.index], 
                                        self.w[isData],
                                        'FourTag',
                                        'ThreeTag',
                                        title='Data Only')

                self.roc_43 = roc_data( np.array((self.y_true==t4.index)|(self.y_true==d4.index), dtype=np.float), 
                                       self.y_pred[:,t4.index]+self.y_pred[:,d4.index], 
                                       self.w,
                                       'FourTag',
                                       'ThreeTag',
                                       title=r'Data and $t\bar{t}$ MC')

                self.roc_td = roc_data(np.array((self.y_true==t3.index)|(self.y_true==t4.index), dtype=np.float), 
                                       self.y_pred[:,t3.index]+self.y_pred[:,t4.index], 
                                       self.w,
                                       r'$t\bar{t}$ MC',
                                       'Data')

                self.roc1 = self.roc_d43
                self.roc2 = self.roc_td

            if classifier in ['SvB', 'SvB_MA']:
                self.roc1 = roc_data(np.array((self.y_true==zz.index)|(self.y_true==zh.index), dtype=np.float), 
                                     self.y_pred[:,zz.index]+self.y_pred[:,zh.index], 
                                     self.w,
                                     'Signal',
                                     'Background')
                isSignal = (self.y_true==zz.index)|(self.y_true==zh.index)
                self.roc2 = roc_data(np.array(self.y_true[isSignal]==zz.index, dtype=np.float), 
                                     (self.y_pred[isSignal,zz.index]-self.y_pred[isSignal,zh.index])/2+0.5, 
                                     self.w[isSignal],
                                     '$ZZ$',
                                     '$ZH$')

                zhIndex = self.y_true!=zz.index
                self.roc_zh = roc_data(np.array(self.y_true[zhIndex]==zh.index, dtype=np.float), 
                                       self.y_pred[zhIndex][:,zh.index], 
                                       self.w[zhIndex],
                                       '$ZH$',
                                       'Background')
                zzIndex = self.y_true!=zh.index
                self.roc_zz = roc_data(np.array(self.y_true[zzIndex]==zz.index, dtype=np.float), 
                                       self.y_pred[zzIndex][:,zz.index], 
                                       self.w[zzIndex],
                                       '$ZZ$',
                                       'Background')


class modelParameters:
    def __init__(self, fileName='', offset=0):
        self.classifier = classifier
        self.xVariables=['canJet0_pt', 'canJet1_pt', 'canJet2_pt', 'canJet3_pt',
                         'dRjjClose', 'dRjjOther', 
                         'aveAbsEta', 'xWt',
                         'nSelJets', 'm4j',
                         ]
        #             |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        #self.layer1Pix = "012302130312"
        self.layer1Pix = "0123"
        self.canJets = ['canJet%s_pt' %i for i in self.layer1Pix]
        self.canJets+= ['canJet%s_eta'%i for i in self.layer1Pix]
        self.canJets+= ['canJet%s_phi'%i for i in self.layer1Pix]
        self.canJets+= ['canJet%s_m'  %i for i in self.layer1Pix]

        self.nOthJets = 8
        self.othJets = ['notCanJet%s_pt' %i for i in range(self.nOthJets)]
        self.othJets+= ['notCanJet%s_eta'%i for i in range(self.nOthJets)]
        self.othJets+= ['notCanJet%s_phi'%i for i in range(self.nOthJets)]
        self.othJets+= ['notCanJet%s_m'  %i for i in range(self.nOthJets)]
        self.othJets+= ['notCanJet%s_isSelJet'%i for i in range(self.nOthJets)]

        self.ancillaryFeatures = ['nSelJets', 'xW', 'xbW', 'year'] 
        #self.ancillaryFeatures = ['nSelJets', 'year'] 
        self.nA = len(self.ancillaryFeatures)

        self.useOthJets = ''
        if classifier in ["FvT", 'DvT3', 'DvT4', "M1vM2", 'SvB_MA']: self.useOthJets = 'attention'
        if args.architecture in ['BasicDNN']: self.useOthJets = ''
        #self.useOthJets = 'multijetAttention'

        self.trainingHistory = {}

        self.validation = loaderResults("validation", classes)
        self.training   = loaderResults("training", classes)
        self.control    = None
        if classifier in ['FvT']:
            self.control = loaderResults("control", classes)

        self.train_losses, self.train_aucs, self.train_stats = [], [], []
        self.valid_losses, self.valid_aucs, self.valid_stats = [], [], []
        self.control_losses, self.control_aucs, self.control_stats = [], [], []
        self.bs_change = []
        self.lr_change = []
        self.dataset_train = None

        lossDict = {'FvT': 0.88,#0.1485,
                    'DvT3': 0.065,
                    'DvT4': 0.88,
                    'ZZvB': 1,
                    'ZHvB': 1,
                    'SvB': 0.74,
                    'SvB_MA': 0.74,
                    }

        if fileName:
            self.classifier           = classifier #    fileName.split('_')[0] if not 
            if "FC" in fileName:
                self.nodes         =   int(fileName[fileName.find(      'x')+1 : fileName.find('_pdrop')])
                self.layers        =   int(fileName[fileName.find(     'FC')+2 : fileName.find('x')])
                self.dijetFeatures = None
                self.quadjetFeatures = None
                self.combinatoricFeatures = None
                self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_np')])
            if "HCR" in fileName:
                name = fileName.replace(classifier, "")
                nFeatures     = int(name.split('_')[2])
                self.dijetFeatures = nFeatures
                self.quadjetFeatures = nFeatures
                self.combinatoricFeatures = nFeatures
                # self.dijetFeatures        = int(name.split('_')[2])
                # self.quadjetFeatures      = int(name.split('_')[3])
                # self.combinatoricFeatures = int(name.split('_')[4])
                self.nodes    = None
                self.pDropout = None
            self.lrInit             = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.offset             =   int(fileName[fileName.find('_offset')+7 : fileName.find('_offset')+8])
            self.startingEpoch      =   int(fileName[fileName.find('_offset')+14: fileName.find('.pkl')])
            #self.training.loss_best = float(fileName[fileName.find(  '_loss')+5 : fileName.find('.pkl')])

        else:
            nFeatures = 14
            self.dijetFeatures  = nFeatures
            self.quadjetFeatures = nFeatures
            self.combinatoricFeatures = nFeatures
            self.nodes         = args.nodes
            self.layers        = args.layers
            self.pDropout      = args.pDropout
            self.lrInit        = lrInit
            self.offset = offset
            self.startingEpoch = 0           
            self.training.loss_best  = lossDict[classifier]
            if classifier in ['M1vM2']: self.validation.roc_auc_best = 0.5

        self.modelPkl = fileName
        self.epochs = args.epochs+self.startingEpoch
        self.epoch = self.startingEpoch

        # Run on gpu if available
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Found CUDA device",self.device,torch.cuda.device_count(),torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            print("Using CPU:",self.device)

        self.nClasses = len(classes)
        self.wC = torch.FloatTensor([1 for i in range(self.nClasses)]).to(self.device)

        if args.architecture == 'BasicCNN':
            self.net = BasicCNN(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.useOthJets, device=self.device, nClasses=self.nClasses).to(self.device)
        elif args.architecture == 'BasicDNN':
            self.net = BasicDNN(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.useOthJets, device=self.device, nClasses=self.nClasses).to(self.device)
        else:
            self.net = HCR(self.dijetFeatures, self.quadjetFeatures, self.ancillaryFeatures, self.useOthJets, device=self.device, nClasses=self.nClasses).to(self.device)

        self.nTrainableParameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.name = args.outputName+classifier+'_'+self.net.name+'_np%d_lr%s_epochs%d_offset%d'%(self.nTrainableParameters, str(self.lrInit), self.epochs, self.offset)
        self.logFileName = 'ZZ4b/nTupleAnalysis/pytorchModels/'+self.name+'.log'
        print("Set log file:", self.logFileName)
        self.logFile = open(self.logFileName, 'a', 1)

        self.lr_current = copy(self.lrInit)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrInit, amsgrad=False)
        #self.optimizer = NAdam(self.net.parameters(), lr=self.lrInit)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.4, momentum=0.95, nesterov=True)
        self.patience = 0
        self.max_patience = max_patience
        if fixedSchedule:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, lr_milestones, gamma=lr_scale, last_epoch=-1)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=lr_scale, threshold=1e-4, patience=self.max_patience, cooldown=1, min_lr=2e-4, verbose=True)
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.25, threshold=1e-4, patience=self.max_patience, cooldown=1, min_lr=1e-4, verbose=True)

        self.foundNewBest = False

        self.dump()

        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state
            self.optimizer.load_state_dict(torch.load(fileName)['optimizer'])
            self.trainingHistory = torch.load(fileName)['training history']

            if args.storeEventFile:
                files = []
                for sample in [args.data, args.ttbar, args.signal]:
                    files += sorted(glob(sample))

                if args.data4b:
                    files += sorted(glob(args.data4b))

                if args.ttbar4b:
                    files += sorted(glob(args.ttbar4b))


                self.storeEvent(files, args.storeEvent)


    def logprint(self, s, end='\n'):
        print(s,end=end)
        self.logFile.write(s+end)

    def epochString(self):
        return ('%d >> %'+str(len(str(self.epochs)))+'d/%d <<')%(self.offset, self.epoch, self.epochs)

    def dfToTensors(self, df, y_true=None):
        n = df.shape[0]
        #basicDNN variables
        #X=torch.FloatTensor( np.float32(df[self.xVariables]) )

        #jet features
        J=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.canJets], 1 )
        O=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.othJets], 1 )

        #extra features 
        #D=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.dijetAncillaryFeatures], 1 )
        #Q=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.quadjetAncillaryFeatures], 1 )
        A=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.ancillaryFeatures], 1 ) 

        if y_true:
            y=torch.LongTensor( np.array(df[y_true], dtype=np.uint8).reshape(-1) )
        else:#assume all zero. y_true not needed for updating classifier output values in .h5 files for example.
            y=torch.LongTensor( np.zeros(df.shape[0], dtype=np.uint8).reshape(-1) )

        R  = torch.LongTensor( 1*np.array(df['SB'], dtype=np.uint8).reshape(-1) )
        R += torch.LongTensor( 2*np.array(df['CR'], dtype=np.uint8).reshape(-1) )
        R += torch.LongTensor( 3*np.array(df['SR'], dtype=np.uint8).reshape(-1) )

        w=torch.FloatTensor( np.float32(df[weight]).reshape(-1) )

        dataset   = TensorDataset(J, O, A, y, w, R)
        return dataset

    def storeEvent(self, files, event):
        #print("Store network response for",classifier,"from file",fileName)
        # Read .h5 file
        frames = getFramesHACK(fileReaders, getFrame, files)
        df = pd.concat(frames, sort=False)
        # df = pd.read_hdf(fileName, key='df')
        # yearIndex = fileName.find('201')
        # year = float(fileName[yearIndex:yearIndex+4])-2010
        # print("Add year to dataframe",year)#,"encoded as",(year-2016)/2)
        # df['year'] = pd.Series(year*np.ones(df.shape[0], dtype=np.float32), index=df.index)

        try:
            eventRow = int(event)
            print("Grab event from row",eventRow)
            i = pd.RangeIndex(df.shape[0])
            df.set_index(i, inplace=True)
            df = df.iloc[eventRow:eventRow+1,:]
        except ValueError:
            print("Grab first event such that",event)
            df = df[ eval(event) ]

        print(df)
        n = df.shape[0]
        print("Convert df to tensors",n)

        dataset = self.dfToTensors(df)

        # Set up data loaders
        print("Make data loader")
        updateResults = loaderResults("update")
        updateResults.evalLoader = DataLoader(dataset=dataset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        updateResults.n = n

        self.net.store=args.storeEventFile

        self.evaluate(updateResults, doROC = False)

        self.net.writeStore()


    def update(self, fileName):
        print("Add",classifier+args.updatePostFix,"output to",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')
        yearIndex = fileName.find('201')
        year = float(fileName[yearIndex:yearIndex+4])-2010
        print("Add year to dataframe",year)#,"encoded as",(year-2016)/2)
        df['year'] = pd.Series(year*np.ones(df.shape[0], dtype=np.float32), index=df.index)

        n = df.shape[0]
        print("Convert df to tensors",n)

        dataset = self.dfToTensors(df)

        # Set up data loaders
        print("Make data loader")
        updateResults = loaderResults("update", classes)
        updateResults.evalLoader = DataLoader(dataset=dataset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        updateResults.n = n

        self.evaluate(updateResults, doROC = False)

        for attribute in updateAttributes:
            df[attribute.title] = pd.Series(np.float32(getattr(updateResults, attribute.name)), index=df.index)

        df.to_hdf(fileName, key='df', format='table', mode='w')
        del df
        del dataset
        del updateResults
        print("Done")

    @torch.no_grad()
    def exportONNX(self):
        self.net.inputGBN.print()
        self.net.layers.print(batchNorm=True)
        self.net.onnx = True # apply softmax to class scores
        # Create a random input for the network. The onnx export will use this to trace out all the operations done by the model.
        # We can later check that the model output is the same with onnx and pytorch evaluation.
        # test_input = (torch.ones(1, 4, 12, requires_grad=True).to('cuda'),
        #               torch.ones(1, 5, 12, requires_grad=True).to('cuda'),
        #               torch.ones(1, self.net.nAd, 6, requires_grad=True).to('cuda'),
        #               torch.ones(1, self.net.nAq, 3, requires_grad=True).to('cuda'),
        #               )
        J = torch.tensor([182.747, 141.376, 109.942, 50.8254, 182.747, 109.942, 141.376, 50.8254, 182.747, 50.8254, 141.376, 109.942, 
                          0.772827, 1.2832, 1.44385, 2.06543, 0.772827, 1.44385, 1.2832, 2.06543, 0.772827, 2.06543, 1.2832, 1.44385, 
                          2.99951, -0.797241, 0.561157, -2.83203, 2.99951, 0.561157, -0.797241, -2.83203, 2.99951, -2.83203, -0.797241, 0.561157, 
                          14.3246, 10.5783, 13.1129, 7.70751, 14.3246, 13.1129, 10.5783, 7.70751, 14.3246, 7.70751, 10.5783, 13.1129],
                         requires_grad=False).to('cuda').view(1,48)
        O = torch.tensor([22.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                          0.0322418, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          -0.00404358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                          4.01562, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         requires_grad=False).to('cuda').view(1,60)
        Q = torch.tensor([3.18101, 2.74553, 2.99015, 
                          525.526, 525.526, 525.526, 
                          4.51741, 4.51741, 4.51741, 
                          0.554433, 0.554433, 0.554433, 
                          4, 4, 4, 
                          2016, 2016, 2016],
                         requires_grad=False).to('cuda').view(1,18)
        # Export the model
        self.net.eval()
        torch_out = self.net(J, O, A)
        print("test output:",torch_out)
        self.modelONNX = self.modelPkl.replace('.pkl','.onnx')
        print("Export ONNX:",self.modelONNX)
        torch.onnx.export(self.net,                                        # model being run
                          (J, O, A),                                       # model input (or a tuple for multiple inputs)
                          self.modelONNX,                                  # where to save the model (can be a file or file-like object)
                          export_params=True,                              # store the trained parameter weights inside the model file
                          #opset_version= 7,                               # the ONNX version to export the model to
                          #do_constant_folding=True,                       # whether to execute constant folding for optimization
                          input_names  = ['J','O','A'],                    # the model's input names
                          output_names = ['c_score', 'q_score'],           # the model's output names
                          #dynamic_axes={ 'input' : {0 : 'batch_size'},    # variable lenght axes
                          #              'output' : {0 : 'batch_size'}}
                          verbose = False
                          )

        # import onnx
        # onnx_model = onnx.load(self.modelONNX)
        # # Check that the IR is well formed
        # onnx.checker.check_model(onnx_model)


    def trainSetup(self, df, df_control=None): #df_train, df_valid):
        # Split into training and validation sets
        print("build idx with offset %i, modulus %i, and train/val split %i"%(self.offset, train_denominator, train_numerator))
        n = df.shape[0]
        idx = np.arange(n)
        is_train = (idx+self.offset)%train_denominator < train_numerator
        is_valid = ~is_train
        # if self.classifier in ['FvT']:
        #     print("Only keep 1/3 of t3 in training set so that it has comparable stats to the d3 sample")
        #     keep_in_train = (~df.t3) | (np.random.rand(n)<1/3) # a random third of t3 events will be kept in the training set
        #     keep_fraction = (df.t3 & keep_in_train).sum()/(df.t3).sum()
        #     print(keep_fraction)
        #     is_train = is_train & keep_in_train
        #     is_valid = ~is_train
        #     df.loc[is_train & df.t3, weight] = df[is_train & df.t3][weight] /      keep_fraction
        #     df.loc[is_valid & df.t3, weight] = df[is_valid & df.t3][weight] / (1 - keep_fraction)

        print("Split into training and validation sets")
        df_train, df_valid = df[is_train], df[is_valid]


        print("Convert df_train to tensors")
        self.dataset_train = self.dfToTensors(df_train, y_true=yTrueLabel)
        print("Convert df_valid to tensors")
        dataset_valid = self.dfToTensors(df_valid, y_true=yTrueLabel)

        print('Set mean and standard deviation of input GBNs using full training set stats instead of using running mean and standard deviation')
        #dataset   = TensorDataset(J, O, A, y, w, R)
        self.net.setMeanStd(self.dataset_train.tensors[0], self.dataset_train.tensors[1], self.dataset_train.tensors[2])

        # Set up data loaders
        # https://arxiv.org/pdf/1711.00489.pdf increase training batch size instead of decaying learning rate
        self.training .trainLoader = DataLoader(dataset=self.dataset_train, batch_size=train_batch_size, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True)
        self.training  .evalLoader = DataLoader(dataset=self.dataset_train, batch_size=eval_batch_size,  shuffle=False, num_workers=n_queue, pin_memory=True)
        self.validation.evalLoader = DataLoader(dataset=     dataset_valid, batch_size=eval_batch_size,  shuffle=False, num_workers=n_queue, pin_memory=True)
        if df_control is not None:
            print("Convert df_control to tensors")
            dataset_control = self.dfToTensors(df_control, y_true=yTrueLabel)
            self.control.evalLoader = DataLoader(dataset=dataset_control, batch_size=eval_batch_size,    shuffle=False, num_workers=n_queue, pin_memory=True)
            self.control.n = df_control.shape[0]
        self.training.n, self.validation.n = df_train.shape[0], df_valid.shape[0]
        print("Training Batch Size:",train_batch_size)
        print("Training Batches:",len(self.training.trainLoader))

        #model initial state
        epochSpaces = max(len(str(self.epochs))-2, 0)
        stat1 = 'Norm ' if classifier in ['FvT'] else 'Sig. '
        stat2 = 'r_max' if classifier in ['FvT'] else '     '
        items = (self.offset, ' '*epochSpaces, ' '*epochSpaces)+tuple([c.abbreviation for c in classes])+(stat1, stat2)
        class_loss_string = ', '.join(['%2s']*self.nClasses)
        legend = ('%d >> %sEpoch%s <<   Data Set |  Loss %%('+class_loss_string+') | %s | %s | %% AUC | %% AUC | AUC Bar Graph ^ (ABC, Max Loss, chi2/bin, p-value) * Output Model')%items
        self.logprint(legend)

        #self.fitRandomForest()
        #self.trainEvaluate(doROC=True)#, doEvaluate=False)
        #self.validate(doROC=True)#, doEvaluate=False)

        #self.logprint('')
        # if fixedSchedule:
        #     self.scheduler.step()
        # else:
        #     self.scheduler.step(self.training.loss)


    @torch.no_grad()
    def evaluate(self, results, doROC=True, evalOnly=False, zeroOutNotSB=True):
        self.net.eval()
        y_pred, y_true, w_ordered = np.ndarray((results.n,self.nClasses), dtype=np.float), np.zeros(results.n, dtype=np.float), np.zeros(results.n, dtype=np.float)
        cross_entropy = np.zeros(results.n, dtype=np.float)
        q_score = np.ndarray((results.n, 3), dtype=np.float)
        print_step = len(results.evalLoader)//200+1
        nProcessed = 0
        #loss = 0
        for i, (J, O, A, y, w, R) in enumerate(results.evalLoader):
            nBatch = w.shape[0]
            J, O, A, y, w = J.to(self.device), O.to(self.device), A.to(self.device), y.to(self.device), w.to(self.device)
            R = R.to(self.device)
            logits, quadjet_scores = self.net(J, O, A)

            if classifier in ['FvT'] and zeroOutNotSB:
                notSB = (R!=1)
                w[notSB] *= 0

            w_swapped, y_swapped = w.clone(), y.clone()
            if classifier in ['FvT']:
                w_neg = w<0
                w_swapped[w_neg] *= -1
                y_swapped[w_neg] = (y_swapped[w_neg]+2)%4

            #loss += (w * F.cross_entropy(logits, y, weight=wC, reduction='none')).sum(dim=0).cpu().item()
            cross_entropy[nProcessed:nProcessed+nBatch] = F.cross_entropy(logits, y_swapped, weight=self.wC, reduction='none').cpu().numpy()
            # this_y_pred = F.softmax(logits, dim=-1).cpu().numpy()
            # if classifier in ['FvT']:
            #     this_y_pred[:,d3.index] = this_y_pred[:,d3.index].clamp(0.1,1) # prevents weights from exceeding 10
            y_pred[nProcessed:nProcessed+nBatch] = F.softmax(logits, dim=-1).cpu().numpy()
            y_true[nProcessed:nProcessed+nBatch] = y.cpu()

            if quadjet_scores is not None:
                q_score[nProcessed:nProcessed+nBatch] = quadjet_scores.cpu().numpy()

            w_ordered[nProcessed:nProcessed+nBatch] = w.cpu()
            nProcessed+=nBatch
            if int(i+1) % print_step == 0:
                percent = float(i+1)*100/len(results.evalLoader)
                sys.stdout.write('\r%d Evaluating %3.0f%%     '%(self.offset, percent))
                sys.stdout.flush()

        loss = (w_ordered * cross_entropy).sum()/w_ordered.sum()#results.n
        #loss = loss/results.n   
        results.update(y_pred, y_true, q_score, w_ordered, cross_entropy, loss, doROC)


    def validate(self, doROC=True, doEvaluate=True):
        if doEvaluate: self.evaluate(self.validation, doROC)
        bar=self.validation.roc1.auc
        bar=int((bar-barMin)*barScale) if bar > barMin else 0

        # roc_abc=None
        overtrain=""

        if self.training.roc1: 
            try:
                n = self.validation.roc1.fpr.shape[0]
                roc_val = interpolate.interp1d(self.validation.roc1.fpr[np.arange(0,n,n//100)], self.validation.roc1.tpr[np.arange(0,n,n//100)], fill_value="extrapolate")
                tpr_val = roc_val(self.training.roc1.fpr)#validation tpr estimated at training fpr
                n = self.training.roc1.fpr.shape[0]
                roc_abc = auc(self.training.roc1.fpr[np.arange(0,n,n//100)], np.abs(self.training.roc1.tpr-tpr_val)[np.arange(0,n,n//100)]) #area between curves
                abcPercent = 100*roc_abc/(roc_abc + (self.validation.roc1.auc-0.5 if self.validation.roc1.auc > 0.5 else 0))

                w_train_notzero = (self.training  .w!=0)
                w_valid_notzero = (self.validation.w!=0)
                ce = np.concatenate((self.training.cross_entropy[w_train_notzero], self.validation.cross_entropy[w_valid_notzero]))
                w  = np.concatenate((self.training.w            [w_train_notzero], self.validation.w            [w_valid_notzero]))
                bins = np.quantile(ce*w, np.arange(0,1.05,0.05), interpolation='linear')
                ce_hist_validation, _    = np.histogram(self.validation.cross_entropy[w_valid_notzero]*self.validation.w[w_valid_notzero], bins=bins)#, weights=self.validation.w[w_valid_notzero])
                ce_hist_training  , bins = np.histogram(self.training  .cross_entropy[w_train_notzero]*self.training  .w[w_train_notzero], bins=bins)#, weights=self.training  .w[w_train_notzero])
                ce_hist_training = ce_hist_training * self.validation.w.sum()/self.training.w.sum() #self.validation.n/self.training.n
                # # remove bins where f_exp is less than 10 for chisquare test (assumes gaussian rather than poisson stats). Use validation as f_obs and training as f_exp
                # ce_hist_validation = ce_hist_validation[ce_hist_training>10]
                # ce_hist_training   = ce_hist_training  [ce_hist_training>10]
                chi2 = chisquare(ce_hist_validation, ce_hist_training)
                ndf = len(ce_hist_validation)

                if chi2.statistic/ndf > 5:
                    print('chi2/ndf > 5')
                    print('bins\n',bins)
                    print('pulls\n',(ce_hist_validation - ce_hist_training)/ce_hist_training**0.5)

                overtrain="^ (%1.1f%%, %1.2f, %2.1f, %1.0f%%)"%(abcPercent, bins[-1], chi2.statistic/ndf, chi2.pvalue*100)

            except:
               overtrain="NaN"

        stat1 = self.validation.norm_data_over_model if classifier in ['FvT', 'DvT3'] else self.validation.roc1.maxSigma
        if stat1 == None: stat1 = -99
        stat2 = self.validation.r_max if classifier in ['FvT', 'DvT3', 'DvT4'] else 0.
        stat2 = '%5.1f'%stat2 if abs(stat2)<100 else '%5.0e'%stat2
        print('\r', end = '')
        s =str(self.offset)+' '*(len(self.epochString())-1)
        auc1 = self.validation.roc1.auc*100 if self.validation.roc1 is not None else 0
        auc2 = self.validation.roc2.auc*100 if self.validation.roc2 is not None else 0
        items = (self.validation.loss,)+tuple([100*l/self.validation.loss for l in self.validation.class_loss])+(stat1, stat2, auc2, auc1, '#'*bar, overtrain)
        class_loss_string = ', '.join(['%2.0f']*self.nClasses)
        s+=(' Validation | %6.4f ('+class_loss_string+') | %5.3f | %s | %5.2f | %5.2f |%s| %s')%items
        self.logprint(s, end=' ')

        try:
            self.trainingHistory['validation.stat'].append(copy(stat1))
            self.trainingHistory['validation.loss'].append(copy(self.validation.loss))
            self.trainingHistory['validation.auc'].append(copy(self.validation.roc1.auc))
            self.trainingHistory['validation.class_loss'].append(copy(self.validation.class_loss))
        except KeyError:
            self.trainingHistory['validation.stat'] = [copy(stat1)]
            self.trainingHistory['validation.loss'] = [copy(self.validation.loss)]
            self.trainingHistory['validation.auc'] = [copy(self.validation.roc1.auc)]
            self.trainingHistory['validation.class_loss'] = [copy(self.validation.class_loss)]


    def train(self):
        #self.net.dijetResNetBlock.multijetAttention.attention.debug=False
        #if self.epoch==2: self.net.attention.debug=True
        self.net.train()
        print_step = len(self.training.trainLoader)//200+1

        totalLoss = 0
        totalttError = 0
        totalLargeReweightLoss = 0
        rMax=0
        startTime = time.time()
        backpropTime = 0
        for i, (J, O, A, y, w, R) in enumerate(self.training.trainLoader):
            J, O, A = J.to(self.device), O.to(self.device), A.to(self.device)
            y, w, R = y.to(self.device), w.to(self.device), R.to(self.device)

            self.optimizer.zero_grad()
            logits, quadjet_scores = self.net(J, O, A)
            
            if classifier in ['FvT']:
                # Use d3, t3, t4 in CR and SR to add loss term in that phase space            
                notSB = (R!=1) # Region==1,2,3 is SB,CR,SR
                notSBisD3   = notSB & (y==d3.index) # get mask of events that are d3
                notSBisntD3 = notSB & (y!=d3.index) # get mask of events that aren't d3 so they can be downweighted by half
                w[notSBisntD3] = 0.5*w[notSBisntD3]
                weightToD4 = notSBisD3 & torch.randint(2,(y.shape[0],), dtype=torch.bool).to(self.device) # make a mask where ~half of the d3 events outside the SB are selected at random

                y_pred = F.softmax(logits.detach(), dim=-1) # compute the class probability estimates with softmax
                #y_pred = F.softmax(logits, dim=-1) # It is critical to detatch the reweight factor from the gradient graph, fails to train badly otherwise, weights diverge to infinity
                D4overD3 = y_pred[weightToD4,d4.index] / y_pred[weightToD4,d3.index] # compute the reweight for d3 -> d4
                D4overD3 = D4overD3.clip(0,20)

                w[weightToD4] = w[weightToD4]*D4overD3 # weight the random d3 events outside the SB to the estimated d4 PDF
                y[weightToD4] = 0*y[weightToD4] # d4.index is zero so multiplying by zero sets these true labels to d4
                #w[notSB] = w[notSB] * max(0, min((self.epoch-3)/4.0, 2.0)) # slowly turn on this loss term so that it isn't large when the PDFs have not started converging
                w[notSB] = w[notSB] * (0. if self.epoch<4 else 1.)
                w_notSB_sum = w[notSB].sum()

            if classifier in ['DvT3','DvT4']:
                y_pred = F.softmax(logits.detach(), dim=-1) # compute the class probability estimates with softmax
                w_notSB_sum = w.sum()

            w_sum = w.sum()

            w_swapped, y_swapped = w.clone(), y.clone()
            if classifier in ['FvT']:
                w_neg = w<0
                w_swapped[w_neg] *= -1
                y_swapped[w_neg] = (y_swapped[w_neg]+2)%4
            #compute classification loss
            cross_entropy = F.cross_entropy(logits, y_swapped, weight=self.wC, reduction='none')
            loss  = (w_swapped * cross_entropy).sum(dim=0)/w_swapped.sum()#.mean(dim=0)

            #perform backprop
            backpropStart = time.time()
            loss.backward()
            #print(loss)
            self.optimizer.step()
            backpropTime += time.time() - backpropStart

            if classifier in ["FvT"]:
                # t3d3 = y_pred[:,t3.index] - y_pred[:,d3.index]
                # t4d4 = y_pred[:,t4.index] - y_pred[:,d4.index]
                # t3d3 = F.relu(t3d3)
                # t4d4 = F.relu(t4d4)
                # # compute loss term to account for failure to always give data higher prob than ttbar
                # ttbarOverPredictionError = 1*(w*t3d3 + w*t4d4).mean()
                # totalttError += ttbarOverPredictionError
                # largeReweightLoss = 1*(w*torch.log1p(F.relu(r-10))).mean()
                # totalLargeReweightLoss += largeReweightLoss
                is_d3 = (y==d3.index)
                r = (y_pred[:,d4.index] - y_pred[:,t4.index])/y_pred[:,d3.index] # m4/d3
                rMax = torch.max(r[is_d3]) if torch.max(r[is_d3])>rMax else rMax

                r_large = r.abs()>1000
                r_large = r_large & is_d3
                if r_large.any():
                    print("r[r_large]\n",r[r_large])
                    print("R[r_large]\n",R[r_large])
                    print("w[r_large]\n",w[r_large])
                    print("y[r_large]\n",y[r_large])
                    print("y_pred[r_large].argmax(1)\n",y_pred[r_large].argmax(1))
                    print("y_pred[r_large]\n",y_pred[r_large])
                    print("weightToD4[r_large]\n",weightToD4[r_large])
                    print('cross_entropy[r_large]\n',cross_entropy[r_large])

            #print(loss)
            thisLoss = loss.item()
            if not totalLoss: totalLoss = thisLoss
            totalLoss = totalLoss*0.98 + thisLoss*(1-0.98) # running average with 0.98 exponential decay rate
            if (i+1) % print_step == 0:
                elapsedTime = time.time() - startTime
                fractionDone = float(i+1)/len(self.training.trainLoader)
                percentDone = fractionDone*100
                estimatedEpochTime = elapsedTime/fractionDone
                timeRemaining = estimatedEpochTime * (1-fractionDone)
                estimatedBackpropTime = backpropTime/fractionDone
                progressString  = str('\r%d Training %3.0f%% ('+loadCycler.next()+')  ')%(self.offset, percentDone)
                progressString += str(('Loss: %0.4f | Time Remaining: %3.0fs | Estimated Epoch Time: %3.0fs | Estimated Backprop Time: %3.0fs ')%
                                     (totalLoss, timeRemaining, estimatedEpochTime, estimatedBackpropTime))


                if classifier in ['FvT', 'DvT3','DvT4']:

                    t = totalttError/print_step * 1e4
                    r = totalLargeReweightLoss/print_step
                    totalttError, totalLargeReweightLoss = 0, 0
                    #progressString += str(('| (ttbar>data %0.3f/1e4, r>10 %0.3f, rMax %0.1f, not SB %2.0f%%) ')%(t,r,rMax,100*w_notSB_sum/w_sum)) 
                    progressString += str(('| (r_max %0.1f, not SB %2.0f%%) ')%(rMax,100*w_notSB_sum/w_sum)) 

                if quadjet_scores is not None:
                    q_1234, q_1324, q_1423 = quadjet_scores[-1,0], quadjet_scores[-1,1], quadjet_scores[-1,2]
                    quadjet_scores, _ = quadjet_scores.sort(dim=1)
                    q_ave_min = quadjet_scores[:,0].mean()
                    q_ave_mid = quadjet_scores[:,1].mean()
                    q_ave_max = quadjet_scores[:,2].mean()
                    progressString += str(('| <q_score> min,mid,max = (%0.2f, %0.2f, %0.2f)   ')%(q_ave_min, q_ave_mid, q_ave_max))

                sys.stdout.write(progressString)
                sys.stdout.flush()
                #print(progressString)

            #checkMemory()

        self.trainEvaluate()

    def trainEvaluate(self, doROC=True, doEvaluate=True):
        if doEvaluate: self.evaluate(self.training, doROC=doROC)
        sys.stdout.write(' '*200)
        sys.stdout.flush()
        bar=self.training.roc1.auc
        bar=int((bar-barMin)*barScale) if bar > barMin else 0
        stat1 = self.training.norm_data_over_model if classifier in ['FvT'] else self.training.roc1.maxSigma
        if stat1 == None: stat1 = -99
        stat2 = self.training.r_max if classifier in ['FvT','DvT3','DvT4'] else 0.
        stat2 = '%5.1f'%stat2 if stat2<1000 else '%5.0e'%stat2
        print('\r',end='')
        auc1 = self.training.roc1.auc*100 if self.training.roc1 is not None else 0
        auc2 = self.training.roc2.auc*100 if self.training.roc2 is not None else 0
        items = (self.epochString(), self.training.loss)+tuple([100*l/self.training.loss for l in self.training.class_loss])+(stat1, stat2, auc2, auc1, "-"*bar)
        class_loss_string = ', '.join(['%2.0f']*self.nClasses)

        s=('%s   Training | %6.4f ('+class_loss_string+') | %5.3f | %s | %5.2f | %5.2f |%s|')%items
        self.logprint(s)

        try:
            self.trainingHistory['training.stat'].append(copy(stat1))
            self.trainingHistory['training.loss'].append(copy(self.training.loss))
            self.trainingHistory['training.auc'].append(copy(self.training.roc1.auc))
            self.trainingHistory['training.class_loss'].append(copy(self.training.class_loss))
        except KeyError:
            self.trainingHistory['training.stat'] = [copy(stat1)]
            self.trainingHistory['training.loss'] = [copy(self.training.loss)]
            self.trainingHistory['training.auc'] = [copy(self.training.roc1.auc)]
            self.trainingHistory['training.class_loss'] = [copy(self.training.class_loss)]

    def controlEvaluate(self, doROC=True, doEvaluate=True):
        if doEvaluate: self.evaluate(self.control, doROC=doROC, zeroOutNotSB=False)
        # sys.stdout.write(' '*200)
        # sys.stdout.flush()
        bar=self.control.roc1.auc
        bar=int((bar-barMin)*barScale) if bar > barMin else 0
        stat1 = self.control.norm_data_over_model if classifier in ['FvT'] else self.control.roc1.maxSigma
        stat2 = self.control.r_max if classifier in ['FvT'] else 0.
        stat2 = '%5.1f'%stat2 if stat2<1000 else '%5.0e'%stat2
        print('\r',end='')
        auc1 = self.control.roc1.auc*100 if self.control.roc1 is not None else 0
        auc2 = self.control.roc2.auc*100 if self.control.roc2 is not None else 0
        items = (self.offset, ' '*(len(self.epochString())-1), self.control.loss)+tuple([100*l/self.control.loss for l in self.control.class_loss])+(stat1, stat2, auc2, auc1, "$"*bar)
        class_loss_string = ', '.join(['%2.0f']*self.nClasses)
        s=('%d%s    Control | %6.4f ('+class_loss_string+') | %5.3f | %s | %5.2f | %5.2f |%s|')%items
        self.logprint(s, end=' ')

        try:
            self.trainingHistory['control.stat'].append(copy(stat1))
            self.trainingHistory['control.loss'].append(copy(self.control.loss))
            self.trainingHistory['control.auc'].append(copy(self.control.roc1.auc))
            self.trainingHistory['control.class_loss'].append(copy(self.control.class_loss))
        except KeyError:
            self.trainingHistory['control.stat'] = [copy(stat1)]
            self.trainingHistory['control.loss'] = [copy(self.control.loss)]
            self.trainingHistory['control.auc'] = [copy(self.control.roc1.auc)]
            self.trainingHistory['control.class_loss'] = [copy(self.control.class_loss)]


    def saveModel(self,writeFile=True):
        self.model_dict = {'model': deepcopy(self.net.state_dict()), 
                           'optimizer': deepcopy(self.optimizer.state_dict()), 
                           'epoch': self.epoch, 
                           'training history': copy(self.trainingHistory),
                       }
            
        if writeFile:
            self.modelPkl = 'ZZ4b/nTupleAnalysis/pytorchModels/%s_epoch%02d.pkl'%(self.name, self.epoch)
            self.logprint('* '+self.modelPkl)
            torch.save(self.model_dict, self.modelPkl)

    def loadModel(self):
        self.net.load_state_dict(self.model_dict['model']) # load model from previous saved state
        self.optimizer.load_state_dict(self.model_dict['optimizer'])
        self.epoch = self.model_dict['epoch']
        self.logprint("Revert to epoch %d"%self.epoch)


    def makePlots(self, baseName='', suffix=''):
        self.modelPkl = 'ZZ4b/nTupleAnalysis/pytorchModels/%s_epoch%02d.pkl'%(self.name, self.epoch)
        if not baseName: baseName = self.modelPkl.replace('.pkl', '')
        if classifier in ['SvB','SvB_MA']:
            plotROC(self.training.roc1,    self.validation.roc1,    plotName=baseName+suffix+'_ROC_sb.pdf')
            plotROC(self.training.roc2,    self.validation.roc2,    plotName=baseName+suffix+'_ROC_zz_zh.pdf')
            plotROC(self.training.roc_zz,  self.validation.roc_zz,  plotName=baseName+suffix+'_ROC_zz.pdf')
            plotROC(self.training.roc_zh,  self.validation.roc_zh,  plotName=baseName+suffix+'_ROC_zh.pdf')
        if classifier in ['DvT3']:
            plotROC(self.training.roc_t3, self.validation.roc_t3, plotName=baseName+suffix+'_ROC_t3.pdf')
        if classifier in ['DvT4']:
            plotROC(self.training.roc_t4, self.validation.roc_t4, plotName=baseName+suffix+'_ROC_t4.pdf')
        if classifier in ['FvT']:
            plotROC(self.training.roc_td, self.validation.roc_td, control=self.control.roc_td, plotName=baseName+suffix+'_ROC_td.pdf')
            plotROC(self.training.roc_43, self.validation.roc_43, control=self.control.roc_43, plotName=baseName+suffix+'_ROC_43.pdf')
            plotROC(self.training.roc_d43, self.validation.roc_d43, control=self.control.roc_d43, plotName=baseName+suffix+'_ROC_d43.pdf')
        plotClasses(self.training, self.validation, baseName+suffix+'.pdf', contr=self.control)

        if self.training.cross_entropy is not None:
            plotCrossEntropy(self.training, self.validation, baseName+suffix+'.pdf')

    def runEpoch(self):
        self.epoch += 1

        self.train()
        self.validate()
        if classifier in ['FvT']:
            self.logprint('')
            self.controlEvaluate()

        self.train_losses.append(copy(self.training  .loss))
        self.valid_losses.append(copy(self.validation.loss))
        self.train_aucs.append(copy(self.training  .roc1.auc))
        self.valid_aucs.append(copy(self.validation.roc1.auc))
        if self.control is not None:
            self.control_losses.append(copy(self.control.loss))
            self.control_aucs.append(copy(self.control.roc1.auc))
        if classifier in ['FvT']:
            self.train_stats.append(copy(self.training  .norm_data_over_model))
            self.valid_stats.append(copy(self.validation.norm_data_over_model))
            if self.control is not None:
                self.control_stats.append(copy(self.control.norm_data_over_model))
        if classifier in ['SvB', 'SvB_MA']:
            self.train_stats.append(copy(self.training  .roc1.maxSigma))
            self.valid_stats.append(copy(self.validation.roc1.maxSigma))

        self.plotTrainingProgress()

        saveModel = False
        if classifier in ['FvT']:
            maxNormGap = 0.01
            saveModel = (abs(self.training.norm_data_over_model-1)<maxNormGap) #and (abs(self.validation.norm_data_over_model-1)<2*maxNormGap)
        else:
            saveModel = self.training.loss < self.training.loss_best
        if self.epoch == self.epochs: 
            saveModel = True
        elif fixedSchedule:
            saveModel = False

        if self.training.loss < self.training.loss_best:
            self.foundNewBest = True
            self.training.loss_best = copy(self.training.loss)

        #self.makePlots()

        if saveModel:
            self.saveModel()
            self.makePlots()        
        else:
            self.logprint('')

        if fixedSchedule:
            self.scheduler.step()
            if self.epoch in bs_milestones or self.epoch in lr_milestones:
                gb_decay = 2 if self.epoch in bs_milestones else 4
                self.logprint('setGhostBatches(%d)'%(self.net.nGhostBatches//gb_decay))
                self.net.setGhostBatches(self.net.nGhostBatches//gb_decay)
            if self.epoch in bs_milestones:
                self.incrementTrainLoader()
            if self.epoch in lr_milestones:
                self.logprint("Decay learning rate: %f -> %f"%(self.lr_current, self.lr_current*lr_scale))
                self.lr_current *= lr_scale
                self.lr_change.append(self.epoch+0.5)
        elif bs_scale*self.training.trainLoader.batch_size > max_train_batch_size:
            self.scheduler.step(self.training.loss)
        elif self.training.loss > self.training.loss_min:
            if self.patience == self.max_patience:
                self.patience = 0
                self.incrementTrainLoader()
            else:
                self.patience += 1
        else:
            self.patience = 0

    def incrementTrainLoader(self):
        try:
            currentBatchSize = self.training.trainLoader.batch_size
            batchString = 'Increase training batch size: %i -> %i (%i batches)'%(currentBatchSize, currentBatchSize*bs_scale, len(self.training.trainLoader)//bs_scale )
            self.logprint(batchString)
            del self.training.trainLoader
            torch.cuda.empty_cache()
            self.training.trainLoader = DataLoader(dataset=self.dataset_train, batch_size=currentBatchSize*bs_scale, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True)
            self.bs_change.append(self.epoch+0.5)
        except:
            batchString = 'Ran out of training data loaders'
            self.logprint(batchString)

    def dump(self, batchNorm=False):
        print(self.net)
        self.net.layers.print(batchNorm=batchNorm)
        print(self.name)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('loss_best:',self.training.loss_best)
        self.nTrainableParameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('N trainable params:',self.nTrainableParameters)

    def plotByEpoch(self, train, valid, ylabel, suffix, loc='best', control=None):
        fig = plt.figure(figsize=(10,7))

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        #plt.ylim(yMin,yMax)
        x = np.arange(1,self.epoch+1)
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
        if control:
            plt.plot(x, control,
                     marker="o",
                     linestyle="--",
                     linewidth=2, alpha=0.5,
                     color="#d34031",
                     label="Control Region")

        plt.xticks(x)
        #plt.yticks(np.linspace(-1, 1, 5))
        plt.legend(loc=loc)

        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        if 'norm' in suffix:
            ylim=[0.8, 1.2]

        for e in self.bs_change:
            plt.plot([e,e], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1, zorder=1)
        for e in self.lr_change:
            plt.plot([e,e], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1, zorder=1)
        if 'norm' in suffix:
            plt.plot(xlim, [1,1], color='k', alpha=1.0, linestyle='-', linewidth=0.75, zorder=0)
        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)

        plotName = 'ZZ4b/nTupleAnalysis/pytorchModels/%s_%s.pdf'%(self.name, suffix)
        try:
            fig.savefig(plotName)
        except:
            print("Cannot save fig: ",plotName)
        plt.close(fig)

    def plotTrainingProgress(self):
        self.plotByEpoch(self.train_losses, self.valid_losses, "Loss", 'loss', loc='upper right', control=self.control_losses)
        self.plotByEpoch(self.train_aucs,   self.valid_aucs,   "AUC",  'auc',  loc='lower right', control=self.control_aucs)
        if classifier in ['FvT']:
            self.plotByEpoch(self.train_stats,  self.valid_stats, "Data / Background",    'norm',  loc='best', control=self.control_stats)
        if classifier in ['SvB', 'SvB_MA']:
            self.plotByEpoch(self.train_stats,  self.valid_stats, "Sensitivity Estimate", 'sigma', loc='lower right')

    def fitRandomForest(self):
        self.RFC = RandomForestClassifier(n_estimators=80, max_depth=3, random_state=0, verbose=1, max_features=3, n_jobs=4)

        y_train, w_train = np.zeros(self.training.n, dtype=np.float), np.zeros(self.training.n, dtype=np.float)
        X_train = np.ndarray((self.training.n, 4*4 + 6*2 + 3+5), dtype=np.float)

        for i, (J, O, D, A, y, w, R) in enumerate(self.training.evalLoader):
            nBatch = w.shape[0]
            nProcessed = nBatch*i

            y_train[nProcessed:nProcessed+nBatch] = y
            w_train[nProcessed:nProcessed+nBatch] = w
            X_train[nProcessed:nProcessed+nBatch,  0:16] = J.view(nBatch,4,12)[:,:,0:4].contiguous().view(nBatch,16) # remove duplicate jets
            # X_train[nProcessed:nProcessed+nBatch, 16:28] = D
            # X_train[nProcessed:nProcessed+nBatch, 28:31] = Q[:, 0: 3] # the three dR's
            # X_train[nProcessed:nProcessed+nBatch, 31:32] = Q[:, 3: 4] # m4j
            # X_train[nProcessed:nProcessed+nBatch, 32:33] = Q[:, 6: 7] # xW
            # X_train[nProcessed:nProcessed+nBatch, 33:34] = Q[:, 9:10] # xbW
            # X_train[nProcessed:nProcessed+nBatch, 34:35] = Q[:,12:13] # nSelJets
            # X_train[nProcessed:nProcessed+nBatch, 35:36] = Q[:,15:16] # year

        print("Fit Random Forest")
        self.RFC.fit(X_train, y_train, w_train)
        print(self.RFC.feature_importances_)

        y_pred_train = self.RFC.predict_proba(X_train)
        self.training.update(y_pred_train, y_train, None, w_train, None, 0, True)

        y_valid, w_valid = np.zeros(self.training.n, dtype=np.float), np.zeros(self.training.n, dtype=np.float)
        X_valid = np.ndarray((self.training.n, 4*4 + 6*2 + 3+5), dtype=np.float)

        for i, (J, O, D, A, y, w, R) in enumerate(self.validation.evalLoader):
            nBatch = w.shape[0]
            nProcessed = nBatch*i

            y_valid[nProcessed:nProcessed+nBatch] = y
            w_valid[nProcessed:nProcessed+nBatch] = w
            X_valid[nProcessed:nProcessed+nBatch,  0:16] = J.view(nBatch,4,12)[:,:,0:4].contiguous().view(nBatch,16) # remove duplicate jets
            # X_valid[nProcessed:nProcessed+nBatch, 16:28] = D
            # X_valid[nProcessed:nProcessed+nBatch, 28:31] = Q[:, 0: 3] # the three dR's
            # X_valid[nProcessed:nProcessed+nBatch, 31:32] = Q[:, 3: 4] # m4j
            # X_valid[nProcessed:nProcessed+nBatch, 32:33] = Q[:, 6: 7] # xW
            # X_valid[nProcessed:nProcessed+nBatch, 33:34] = Q[:, 9:10] # xbW
            # X_valid[nProcessed:nProcessed+nBatch, 34:35] = Q[:,12:13] # nSelJets
            # X_valid[nProcessed:nProcessed+nBatch, 35:36] = Q[:,15:16] # year

        y_pred_valid = self.RFC.predict_proba(X_valid)
        self.validation.update(y_pred_valid, y_valid, None, w_valid, None, 0, True)

        self.makePlots(baseName='ZZ4b/nTupleAnalysis/pytorchModels/'+self.classifier+'_random_forest')



#Simple ROC Curve plot function
def plotROC(train, valid, control=None, plotName='test.pdf'): #fpr = false positive rate, tpr = true positive rate
    f = plt.figure()
    ax = plt.subplot(1,1,1)
    ax.set_title(train.title)
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    ax.plot([0,1], [1,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
    plt.xlabel('Rate( '+valid.trueName+' to '+valid.trueName+' )')
    plt.ylabel('Rate( '+valid.falseName+' to '+valid.falseName+' )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    ax.plot(train.tpr, 1-train.fpr, color='#d34031', linestyle='-', linewidth=1, alpha=1.0, label="Training (%0.4f)"%train.auc)
    ax.plot(valid.tpr, 1-valid.fpr, color='#d34031', linestyle='-', linewidth=2, alpha=0.5, label="Validation (%0.4f)"%valid.auc)
    if control is not None:
        ax.plot(control.tpr, 1-control.fpr, color='#d34031', linestyle='--', linewidth=2, alpha=0.5, label="Control (%0.4f)"%control.auc)
    ax.legend(loc='lower left')
    #ax.text(0.73, 1.07, "Validation AUC = %0.4f"%(valid.auc))

    if valid.maxSigma is not None:
        #ax.scatter(rate_StoS, rate_BtoB, marker='o', c='k')
        #ax.text(rate_StoS+0.03, rate_BtoB-0.100, ZB+"SR \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)
        ax.scatter(valid.tprMaxSigma, (1-valid.fprMaxSigma), marker='o', c='#d34031')
        ax.text(valid.tprMaxSigma+0.03, (1-valid.fprMaxSigma)-0.025, 
                ("(%0.3f, %0.3f), "+valid.pName+" $>$ %0.2f \n S=%0.1f, B=%0.1f, $%1.2f\sigma$")%(valid.tprMaxSigma, (1-valid.fprMaxSigma), valid.thrMaxSigma, valid.S, valid.B, valid.maxSigma), 
                bbox=bbox)

    try:
        f.savefig(plotName)
    except:
        print("Cannot save fig: ",plotName)

    plt.close(f)

def plotClasses(train, valid, name, contr=None):
    # Make place holder datasets to add the training/validation set graphical distinction to the legend
    trainLegend=pltHelper.dataSet(name=  'Training', color='black', alpha=1.0, linewidth=1)
    validLegend=pltHelper.dataSet(name='Validation', color='black', alpha=0.5, linewidth=2)
    contrLegend=pltHelper.dataSet(name='Control Region', color='black', alpha=0.5, linewidth=1, fmt='o') if contr is not None else None

    extraClasses = []
    binWidth = 0.05
    if classifier in ["SvB",'SvB_MA']:
        extraClasses = [sg,bg]
        bins = np.arange(-binWidth, 1+2*binWidth, binWidth)
    else:
        bins = np.arange(-2*binWidth, 1+2*binWidth, binWidth)

    for cl1 in classes+extraClasses: # loop over classes
        cl1cl2_args = {'dataSets': [trainLegend,validLegend],
                       'bins': bins,
                       'xlabel': 'P( '+cl1.name+r' $\rightarrow$ Class )',
                       'ylabel': 'Arb. Units',
                       }
        cl2cl1_args = {'dataSets': [trainLegend,validLegend],
                       'bins': bins,
                       'xlabel': r'P( Class $\rightarrow$ '+cl1.name+' )',
                       'ylabel': 'Arb. Units',
                       }
        for cl2 in classes+extraClasses: # loop over classes
        # Make datasets to be plotted
            cl1cl2_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(train,'w'+cl1.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl1cl2_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(valid,'w'+cl1.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl1cl2_args['dataSets'] += [cl1cl2_valid, cl1cl2_train]

            cl2cl1_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl2.abbreviation+cl1.abbreviation), weights=getattr(train,'w'+cl2.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl2cl1_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl2.abbreviation+cl1.abbreviation), weights=getattr(valid,'w'+cl2.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl2cl1_args['dataSets'] += [cl2cl1_valid, cl2cl1_train]

        if classifier in ['FvT']:
            # multijet probabilities well defined but no multijet class labels. Therefore cl1cl2 plot can include multijet but not cl2cl1 plot.
            m4 = classInfo(abbreviation='m4', name= 'FourTag Multijet', color='blue')
            m3 = classInfo(abbreviation='m3', name='ThreeTag Multijet', color='violet')
            for cl2 in [m4,m3]:
                cl1cl2_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(train,'w'+cl1.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
                cl1cl2_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(valid,'w'+cl1.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
                cl1cl2_args['dataSets'] += [cl1cl2_train, cl1cl2_valid]

        #make the plotter
        cl1cl2 = pltHelper.histPlotter(**cl1cl2_args)
        cl2cl1 = pltHelper.histPlotter(**cl2cl1_args)
        #remove the lines from the trainLegend/validLegend placeholders
        cl1cl2.artists[0].remove()
        cl1cl2.artists[1].remove()
        cl2cl1.artists[0].remove()
        cl2cl1.artists[1].remove()

        #save the pdf
        try:
            cl1cl2.savefig(name.replace('.pdf','_'+cl1.abbreviation+'_to_class.pdf'))
        except:
            print("cannot save", name.replace('.pdf','_'+cl1.abbreviation+'_to_class.pdf'))

        try:
            cl2cl1.savefig(name.replace('.pdf','_class_to_'+cl1.abbreviation+'.pdf'))
        except:
            print("cannot save",name.replace('.pdf','_class_to_'+cl1.abbreviation+'.pdf'))


    if classifier in ['FvT']:
        #bins = np.arange(-0.5,5,0.1)
        bins = np.quantile(train.rd4, np.arange(0,1.05,0.05), interpolation='linear')
        bm_vs_d4_args = {'dataSets': [trainLegend,validLegend,contrLegend],
                         'bins': bins,
                         'divideByBinWidth': True,
                         'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Data )',
                         'ylabel': 'Arb. Units',
                         }
        d4_train = pltHelper.dataSet(name=d4.name, points=train.rd4, weights= train.wd4/train.w_sum, color=d4.color, alpha=1.0, linewidth=1)
        d4_valid = pltHelper.dataSet(              points=valid.rd4, weights= valid.wd4/valid.w_sum, color=d4.color, alpha=0.5, linewidth=2)
        d4_contr = pltHelper.dataSet(              points=contr.rd4, weights= contr.wd4/contr.w_sum, color=d4.color, alpha=0.5, linewidth=1, fmt='o')
        bm_train = pltHelper.dataSet(name='Background Model', 
                                     points=np.concatenate((train.rd3,train.rt3,train.rt4),axis=None), 
                                     weights=np.concatenate((train.wd3,-train.wt3,train.wt4)/train.w_sum,axis=None), 
                                     color='brown', alpha=1.0, linewidth=1)
        bm_valid = pltHelper.dataSet(points=np.concatenate((valid.rd3,valid.rt3,valid.rt4),axis=None), 
                                     weights=np.concatenate((valid.wd3,-valid.wt3,valid.wt4)/valid.w_sum,axis=None), 
                                     color='brown', alpha=0.5, linewidth=2)
        bm_contr = pltHelper.dataSet(points=np.concatenate((contr.rd3,contr.rt3,contr.rt4),axis=None), 
                                     weights=np.concatenate((contr.wd3,-contr.wt3,contr.wt4)/contr.w_sum,axis=None), 
                                     color='brown', alpha=0.5, linewidth=1, fmt='o')
        t4_train = pltHelper.dataSet(name=t4.name, points=train.rt4, weights= train.wt4/train.w_sum, color=t4.color, alpha=1.0, linewidth=1)
        t4_valid = pltHelper.dataSet(              points=valid.rt4, weights= valid.wt4/valid.w_sum, color=t4.color, alpha=0.5, linewidth=2)
        t4_contr = pltHelper.dataSet(              points=contr.rt4, weights= contr.wt4/contr.w_sum, color=t4.color, alpha=0.5, linewidth=1, fmt='o')
        t3_train = pltHelper.dataSet(name=t3.name, points=train.rt3, weights=-train.wt3/train.w_sum, color=t3.color, alpha=1.0, linewidth=1)
        t3_valid = pltHelper.dataSet(              points=valid.rt3, weights=-valid.wt3/valid.w_sum, color=t3.color, alpha=0.5, linewidth=2)
        t3_contr = pltHelper.dataSet(              points=contr.rt3, weights=-contr.wt3/contr.w_sum, color=t3.color, alpha=0.5, linewidth=1, fmt='o')
        bm_vs_d4_args['dataSets'] += [d4_contr, d4_valid, d4_train, 
                                      bm_contr, bm_valid, bm_train, 
                                      t4_contr, t4_valid, t4_train, 
                                      t3_contr, t3_valid, t3_train]
        bm_vs_d4 = pltHelper.histPlotter(**bm_vs_d4_args)
        bm_vs_d4.artists[0].remove()
        bm_vs_d4.artists[1].remove()
        bm_vs_d4.artists[2].remove()
        try:
            bm_vs_d4.savefig(name.replace('.pdf','_bm_vs_d4.pdf'))
        except:
            print("cannot save",name.replace('.pdf','_bm_vs_d4.pdf'))

        rbm_vs_d4_args = {'dataSets': [trainLegend,validLegend,contrLegend],
                         'bins': bins,
                         'divideByBinWidth': True,
                         'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Data )',
                         'ylabel': 'Arb. Units',
                         }
        rbm_train = pltHelper.dataSet(name='Background Model', 
                                     points=np.concatenate((train.rd3,train.rt4),axis=None), 
                                     weights=np.concatenate((train.rd3*train.wd3,train.wt4)/train.w_sum,axis=None), 
                                     color='brown', alpha=1.0, linewidth=1)
        rbm_valid = pltHelper.dataSet(points=np.concatenate((valid.rd3,valid.rt4),axis=None), 
                                     weights=np.concatenate((valid.rd3*valid.wd3,valid.wt4)/valid.w_sum,axis=None), 
                                     color='brown', alpha=0.5, linewidth=2)
        rbm_contr = pltHelper.dataSet(points=np.concatenate((contr.rd3,contr.rt4),axis=None), 
                                     weights=np.concatenate((contr.rd3*contr.wd3,contr.wt4)/contr.w_sum,axis=None), 
                                     color='brown', alpha=0.5, linewidth=1, fmt='o')
        rt3_train = pltHelper.dataSet(name=t3.name, points=train.rt3, weights=-train.rt3*train.wt3/train.w_sum, color=t3.color, alpha=1.0, linewidth=1)
        rt3_valid = pltHelper.dataSet(              points=valid.rt3, weights=-valid.rt3*valid.wt3/valid.w_sum, color=t3.color, alpha=0.5, linewidth=2)
        rt3_contr = pltHelper.dataSet(              points=contr.rt3, weights=-contr.rt3*contr.wt3/contr.w_sum, color=t3.color, alpha=0.5, linewidth=1, fmt='o')
        rbm_vs_d4_args['dataSets'] += [ d4_contr,  d4_valid,  d4_train, 
                                       rbm_contr, rbm_valid, rbm_train, 
                                        t4_contr,  t4_valid,  t4_train,
                                       rt3_contr, rt3_valid, rt3_train]
        rbm_vs_d4 = pltHelper.histPlotter(**rbm_vs_d4_args)
        rbm_vs_d4.artists[0].remove()
        rbm_vs_d4.artists[1].remove()
        rbm_vs_d4.artists[2].remove()
        try:
            rbm_vs_d4.savefig(name.replace('.pdf','_rbm_vs_d4.pdf'))
        except:
            print("cannot save",name.replace('.pdf','_rbm_vs_d4.pdf'))


def plotCrossEntropy(train, valid, name):
    cross_entropy_train = pltHelper.dataSet(name=  'Training Set', points=train.cross_entropy*train.w, weights=train.w/train.w_sum, color='black', alpha=1.0, linewidth=1)
    cross_entropy_valid = pltHelper.dataSet(name='Validation Set', points=valid.cross_entropy*valid.w, weights=valid.w/valid.w_sum, color='black', alpha=0.5, linewidth=2)

    w_train_notzero = (train.w!=0)
    bins = np.quantile(train.cross_entropy[w_train_notzero]*train.w[w_train_notzero], np.arange(0,1.05,0.05), interpolation='linear')

    cross_entropy_args = {'dataSets': [cross_entropy_train, cross_entropy_valid],
                          'bins': bins,#[b/50.0 for b in range(0,76)],
                          'xlabel': r'Cross Entropy * Event Weight',
                          'ylabel': 'Arb. Units',
                          'divideByBinWidth': True,
                          }

    for cl1 in classes: # loop over classes
        w_train = getattr(train,'w'+cl1.abbreviation)
        w_valid = getattr(valid,'w'+cl1.abbreviation)
        ce_train = getattr(train,'ce'+cl1.abbreviation)
        ce_valid = getattr(valid,'ce'+cl1.abbreviation)
        cl1_train = pltHelper.dataSet(name=cl1.name, points=ce_train*w_train, weights=w_train/train.w_sum, color=cl1.color, alpha=1.0, linewidth=1)
        cl1_valid = pltHelper.dataSet(               points=ce_valid*w_valid, weights=w_valid/valid.w_sum, color=cl1.color, alpha=0.5, linewidth=2)
        cross_entropy_args['dataSets'] += [cl1_valid, cl1_train]

    cross_entropy = pltHelper.histPlotter(**cross_entropy_args)
    try:
        cross_entropy.savefig(name.replace('.pdf','_cross_entropy.pdf'))
    except:
        print("cannot save",name.replace('.pdf','_cross_entropy.pdf'))



if __name__ == '__main__':

    models = []
    if args.train:
        if len(train_offset)>1:
            print("Train Models in parallel")
            processes = [mp.Process(target=runTraining, args=(offset, df, df_control)) for offset in train_offset]
            for p in processes: p.start()
            for p in processes: p.join()
            models = [queue.get() for p in processes]
        else:
            runTraining(train_offset[0], df, df_control)
            models = [queue.get()]
        print(models)

    if args.update:
        if not models:
            paths = args.model.split(',')
            for path in paths:
                models += glob(path)
        models.sort()
        models = [modelParameters(name) for name in models]

        files = []
        for sample in [args.data, args.ttbar, args.signal]:
            files += sorted(glob(sample))

        if args.data4b:
            for d4b in args.data4b.split(","):
                files += sorted(glob(d4b))

        if args.ttbar4b:
            files += sorted(glob(args.ttbar4b))

        for sampleFile in files:
            print(sampleFile)

        print("Average over %d models:"%len(models))
        for model in models:
            print("   ",model.modelPkl)

        for i, fileName in enumerate(files):
            print("Add",classifier+args.updatePostFix,"output to",fileName)
            # Read .h5 file
            df = pd.read_hdf(fileName, key='df')
            yearIndex = fileName.find('201')
            year = float(fileName[yearIndex:yearIndex+4])-2010
            #print("Add year to dataframe",year)#,"encoded as",(year-2016)/2)
            df['year'] = pd.Series(year*np.ones(df.shape[0], dtype=np.float32), index=df.index)

            n = df.shape[0]
            #print("Convert df to tensors",n)
            dataset = models[0].dfToTensors(df)

            # Set up data loaders
            #print("Make data loader")
            results = loaderResults("update", classes)
            results.evalLoader = DataLoader(dataset=dataset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
            results.n = n

            averageModels(models, results)

            for attribute in updateAttributes:
                df[attribute.title] = pd.Series(np.float32(getattr(results, attribute.name)), index=df.index)

            df.to_hdf(fileName, key='df', format='table', mode='w')
            del df
            del dataset
            del results
            print("File %2d/%d updated all %7d events from %s"%(i+1,len(files),n,fileName))


    if args.onnx:
        print("Export models to ONNX Runtime")
        if not models:
            paths = args.model.split(',')
            for path in paths:
                models += glob(path)
            models.sort()
            models = [modelParameters(name) for name in models]
        
        for model in models:
            print()
            model.exportONNX()
        modelEnsemble = HCREnsemble([model.net for model in models])
        modelEnsemble.exportONNX(models[0].modelPkl.replace("_offset0","").replace(".pkl",".onnx"))
