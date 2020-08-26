import time, os, sys
from pathlib import Path
import multiprocessing
from glob import glob
from copy import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'ZZ4b/nTupleAnalysis/scripts/') #https://github.com/patrickbryant/PlotTools
import matplotlibHelpers as pltHelper

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--data', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('--data4b',     default=None, help="Take 4b from this file if given, otherwise use --data for both 3-tag and 4-tag")
parser.add_argument('-t', '--ttbar',      default='',    type=str, help='Input MC ttbar file in hdf5 format')
parser.add_argument('--ttbar4b',          default=None, help="Take 4b ttbar from this file if given, otherwise use --ttbar for both 3-tag and 4-tag")
parser.add_argument('-s', '--signal',     default='', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-o', '--outdir',     default='', type=str, help='outputDirectory')
parser.add_argument('--weightName', default="mcPseudoTagWeight", help='Which weights to use for JCM.')
parser.add_argument('--FvTName', default="FvT", help='Which weights to use for FvT.')
args = parser.parse_args()

def getFrame(fileName):
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])
    print("Reading",fileName)
    thisFrame = pd.read_hdf(fileName, key='df')
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    return thisFrame



def getFramesHACK(fileReaders,getFrame,dataFiles):
    largeFiles = []
    print("dataFiles was:",dataFiles)
    for d in dataFiles:
        if Path(d).stat().st_size > 2e9:
            print("Large File",d)
            largeFiles.append(d)
            dataFiles.remove(d)

    results = fileReaders.map_async(getFrame, sorted(dataFiles))
    frames = results.get()

    for f in largeFiles:
        frames.append(getFrame(f))

    return frames





outputDir = args.outdir
if not os.path.isdir(outputDir):
    print("Making output dir",outputDir)
    os.mkdir(outputDir)

fileReaders = multiprocessing.Pool(10)

weightName = args.weightName
print("Using JCM weight with name: ",weightName)

FvTName = args.FvTName
print("Using FvT weight with name: ",FvTName)

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

d4 = classInfo(abbreviation='d4', name=  'FourTag Data',       index=0, color='red')
d3 = classInfo(abbreviation='d3', name= 'ThreeTag Data',       index=1, color='orange')
t4 = classInfo(abbreviation='t4', name= r'FourTag $t\bar{t}$', index=2, color='green')
t3 = classInfo(abbreviation='t3', name=r'ThreeTag $t\bar{t}$', index=3, color='cyan')
zz = classInfo(abbreviation='zz', name=r'$ZZ$ MC $\times100$', index=4, color='blue')
zh = classInfo(abbreviation='zh', name=r'$ZH$ MC $\times100$', index=5, color='violet')

dfs = []

# Read .h5 files
dataFiles = glob(args.data)
if args.data4b:
    dataFiles += glob(args.data4b)    


frames = getFramesHACK(fileReaders,getFrame,dataFiles)
dfD = pd.concat(frames, sort=False)

for k in dfD.keys():
    print(k)

print("Add true class labels to data")
dfD['d4'] =  dfD.fourTag
dfD['d3'] = (dfD.fourTag+1)%2
dfD['t4'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
dfD['t3'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
dfD['zz'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
dfD['zh'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)

dfs.append(dfD)

# Read .h5 files
ttbarFiles = glob(args.ttbar)
if args.ttbar4b:
    ttbarFiles += glob(args.ttbar4b)    

frames = getFramesHACK(fileReaders,getFrame,ttbarFiles)
dfT = pd.concat(frames, sort=False)

print("Add true class labels to ttbar MC")
dfT['t4'] =  dfT.fourTag
dfT['t3'] = (dfT.fourTag+1)%2
dfT['d4'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
dfT['d3'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
dfT['zz'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
dfT['zh'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)

dfs.append(dfT)

if args.signal:
    frames = []
    for fileName in sorted(glob(args.signal)):
        yearIndex = fileName.find('201')
        year = float(fileName[yearIndex:yearIndex+4])
        print("Reading",fileName)
        thisFrame = pd.read_hdf(fileName, key='df')
        print("Add year to dataframe",year)#,"encoded as",(year-2016)/2)
        thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
        print("Add true class labels to signal")
        if "ZZ4b201" in fileName: 
            index = zz.index
            thisFrame['zz'] = thisFrame.fourTag
            thisFrame['zh'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=np.uint8), index=thisFrame.index)
        if "ZH4b201" in fileName: 
            index = zh.index
            thisFrame['zz'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=np.uint8), index=thisFrame.index)
            thisFrame['zh'] = thisFrame.fourTag
        thisFrame['t4'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=np.uint8), index=thisFrame.index)
        thisFrame['t3'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=np.uint8), index=thisFrame.index)
        thisFrame['d4'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=np.uint8), index=thisFrame.index)
        thisFrame['d3'] = pd.Series(np.zeros(thisFrame.shape[0], dtype=np.uint8), index=thisFrame.index)
        frames.append(thisFrame)
    dfS = pd.concat(frames, sort=False)
    dfs.append(dfS)


print("concatenate dataframes")
df = pd.concat(dfs, sort=False)



class dataFrameOrganizer:
    def __init__(self, dataFrame):
        self.df = dataFrame
        self.dfSelected = dataFrame
        self.dfd4 = None
        self.dfd3 = None
        self.dft4 = None
        self.dft3 = None
        self.dfbg = None
        self.dfzz = None
        self.dfzh = None
        self.dfsg = None

    def applySelection(self, selection):
        print("Apply selection")
        self.dfSelected = self.df.loc[ selection ]
        
        self.dfd4 = self.dfSelected.loc[ self.dfSelected.d4==True ]
        self.dfd3 = self.dfSelected.loc[ self.dfSelected.d3==True ]
        self.dft4 = self.dfSelected.loc[ self.dfSelected.t4==True ]
        self.dft3 = self.dfSelected.loc[ self.dfSelected.t3==True ]
        self.dfbg = self.dfSelected.loc[ (self.dfSelected.d3==True) | (self.dfSelected.t4==True) ]
        if args.signal:
            self.dfzz = self.dfSelected.loc[ self.dfSelected.zz==True ]
            self.dfzh = self.dfSelected.loc[ self.dfSelected.zh==True ]
            self.dfsg = self.dfSelected.loc[ (self.dfSelected.zz==True) | (self.dfSelected.zh==True) ]

    def plotVar(self, var, FvTName, outName, bins=None, xmin=None, xmax=None, regName=""):

        if type(FvTName) == list:
            ttbarWeights = -getattr(self.dft3,weightName)   * 1./3 * (getattr(self.dft3,FvTName[0])+getattr(self.dft3,FvTName[1])+getattr(self.dft3,FvTName[2]))
            multijetWeights = getattr(self.dfd3,weightName) * 1./3 * (getattr(self.dfd3,FvTName[0])+getattr(self.dfd3,FvTName[1])+getattr(self.dfd3,FvTName[2]))
            backgroundWeights = np.concatenate((getattr(self.dfd3,weightName) * 1./3 * (getattr(self.dfd3,FvTName[0])+getattr(self.dfd3,FvTName[1])+getattr(self.dfd3,FvTName[2])), getattr(self.dft4,weightName)))
        else:
            ttbarWeights = -getattr(self.dft3,weightName) * getattr(self.dft3,FvTName)
            multijetWeights = getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName)
            backgroundWeights = np.concatenate((getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName), getattr(self.dft4,weightName)))



        multijet = self.dfd3[var] if type(var) != list else self.dfd3[var[0]] - self.dfd3[var[1]]
        #dfd3var  = self.dfd3[var] if type(var) != list else self.dfd3[var[0]] - self.dfd3[var[1]]
        dft4var  = self.dft4[var] if type(var) != list else self.dft4[var[0]] - self.dft4[var[1]]
        dfd4var  = self.dfd4[var] if type(var) != list else self.dfd4[var[0]] - self.dfd4[var[1]]
        dft3var  = self.dft3[var] if type(var) != list else self.dft3[var[0]] - self.dft3[var[1]]

        background = np.concatenate((multijet, dft4var))


        self.dsd4 = pltHelper.dataSet(name=d4.name, 
                                      points = dfd4var,
                                      weights=getattr(self.dfd4,weightName), 
                                      color=d4.color, alpha=1.0, linewidth=1)
        self.bkgd = pltHelper.dataSet(name='Background Model', 
                                      points =background,
                                      weights=backgroundWeights, 
                                      color='brown', alpha=1.0, linewidth=1)
        self.dst4 = pltHelper.dataSet(name=t4.name, 
                                      points =dft4var,
                                      weights=getattr(self.dft4,weightName),
                                      color=t4.color, alpha=1.0, linewidth=1)
        self.dsm3 = pltHelper.dataSet(name='ThreeTag Multijet', 
                                      points =multijet,
                                      weights=multijetWeights,
                                      color=d3.color, alpha=1.0, linewidth=1)
        self.dst3 = pltHelper.dataSet(name=t3.name, 
                                      points=dft3var,
                                      weights=ttbarWeights,
                                      color=t3.color, alpha=1.0, linewidth=1)
        datasets = [self.dsd4,self.bkgd,self.dst4,self.dsm3,self.dst3]
        if self.dfzz is not None:
            self.dszz = pltHelper.dataSet(name=zz.name,
                                          points=self.dfzz[var],
                                          weights=getattr(self.dfzz,weightName)*100,
                                          color=zz.color, alpha=1.0, linewidth=1)
            datasets += [self.dszz]
        if self.dfzh is not None:
            self.dszh = pltHelper.dataSet(name=zh.name,
                                          points=self.dfzh[var],
                                          weights=getattr(self.dfzh,weightName)*100,
                                          color=zh.color, alpha=1.0, linewidth=1)
            datasets += [self.dszh]

        if type(bins)!=list:
            if not bins: bins=50
            
            if type(var) == list:            
                if type(xmin)==type(None): xmin = self.dfSelected[var[0]].min()
                if type(xmax)==type(None): xmax = self.dfSelected[var[0]].max()
            else:
                if type(xmin)==type(None): xmin = self.dfSelected[var].min()
                if type(xmax)==type(None): xmax = self.dfSelected[var].max()
                    
            width = (xmax-xmin)/bins
            bins = [xmin + b*width for b in range(-1,bins+1)]

        if type(var) == list:            
            xlabel = var[0]+"_minus_"+var[1]
        else:
            xlabel = var

        args = {'dataSets': datasets,
                'ratio': [0,1],
                'ratioRange': [0.5,1.5],
                'ratioTitle': 'Data / Model',
                'bins': bins,
                'xlabel': xlabel.replace('_',' '),
                'ylabel': 'Events / Bin',
                }

        fig = pltHelper.histPlotter(**args)
        figName = outputDir + "/"+regName+"_"+xlabel+outName+'.pdf'
        fig.savefig(figName)

        print(figName)

    def hist2d(self, regName, dfName, xvar, yvar ,bins=50,histRange=None,reweight=False):
        df = getattr(self,dfName)

        x = df[xvar]

        if type(yvar) == list:
            y = df[yvar[0]] - df[yvar[1]]     
        else:
            y = df[yvar]

        if reweight:
            weights = getattr(df,weightName) * (getattr(df,FvTName) * (1-df.fourTag) + df.fourTag)
        else:
            weights = getattr(df,weightName)

        xlabel = xvar.replace('_',' ')

        if type(yvar) == list:
            ylabel = yvar[0]+"_minus_"+yvar[1]
        else:
            ylabel = yvar


        args = {'x':x, 'y':y, 'weights':weights,
                'xlabel': xlabel,
                'ylabel': ylabel.replace('_',' '),
                'zlabel': 'Events / Bin',
                'bins': bins,
                'range': histRange,
                }
        fig = pltHelper.hist2d(**args)
        figName = outputDir +"/"+regName+dfName+"_"+xvar+"_vs_"+ylabel+("_reweight" if reweight else "")+".pdf"
        fig.savefig(figName)
        print(figName)


def doRegion(regName):

    varsToPlot = [FvTName, 'SvB_ps', 'SvB_pzz', 'SvB_pzh', 'nSelJets','dR0123', 'dR0213', 'dR0312']
    
    for v in varsToPlot:
        xmax = None
        if not v.find('SvB') == -1: xmax = 1.0
        if not v.find('FvT') == -1: xmax = 3.0
    
        dfo.plotVar(v, FvTName=FvTName+"_offset0", outName="offSet0", regName=regName, xmin=0.0, xmax=xmax)
        dfo.plotVar(v, FvTName=FvTName+"_offset1", outName="offSet1", regName=regName, xmin=0.0, xmax=xmax)
        dfo.plotVar(v, FvTName=FvTName+"_offset2", outName="offSet2", regName=regName, xmin=0.0, xmax=xmax)
        dfo.plotVar(v, FvTName=[FvTName+"_offset0", FvTName+"_offset1", FvTName+"_offset2"],
                    outName="offSetComb", regName=regName, xmin=0.0, xmax=xmax)


        dfo.plotVar([FvTName+"_offset0",FvTName+"_offset1"], FvTName=FvTName+"_offset0", outName="", regName=regName, xmin=-1, xmax=1)
        dfo.plotVar([FvTName+"_offset0",FvTName+"_offset2"], FvTName=FvTName+"_offset0", outName="", regName=regName, xmin=-1, xmax=1)
        dfo.plotVar([FvTName+"_offset1",FvTName+"_offset2"], FvTName=FvTName+"_offset0", outName="", regName=regName, xmin=-1, xmax=1)

        dfo.plotVar([FvTName+"_offset0",FvTName+"_offset1"], FvTName=FvTName+"_offset1", outName="wrtOffset1", regName=regName, xmin=-1, xmax=1)
        dfo.plotVar([FvTName+"_offset0",FvTName+"_offset1"], FvTName=FvTName+"_offset2", outName="wrtOffset2", regName=regName, xmin=-1, xmax=1)

        dfo.plotVar([FvTName+"_offset0",FvTName+"_offset2"], FvTName=FvTName+"_offset1", outName="wrtOffse1", regName=regName, xmin=-1, xmax=1)

        histRange = [[0,2],[0,2]]
        dfo.hist2d(regName, 'dfd3',FvTName+"_offset0",FvTName+"_offset1" ,histRange=histRange, bins=100)
        dfo.hist2d(regName, 'dfd3',FvTName+"_offset0",FvTName+"_offset2" ,histRange=histRange, bins=100)
        dfo.hist2d(regName, 'dfd3',FvTName+"_offset1",FvTName+"_offset2" ,histRange=histRange, bins=100)

        histRange = [[0,1],[-1,1]]
        dfo.hist2d(regName, 'dfd3', "SvB_pzh", [FvTName+"_offset0",FvTName+"_offset1"],histRange=histRange, bins=100)
        dfo.hist2d(regName, 'dfd3', "SvB_pzz", [FvTName+"_offset0",FvTName+"_offset1"],histRange=histRange, bins=100)

        dfo.hist2d(regName, 'dfd3', "SvB_pzh", [FvTName+"_offset0",FvTName+"_offset2"],histRange=histRange, bins=100)
        dfo.hist2d(regName, 'dfd3', "SvB_pzz", [FvTName+"_offset0",FvTName+"_offset2"],histRange=histRange, bins=100)

        dfo.hist2d(regName, 'dfd3', "SvB_pzh", [FvTName+"_offset1",FvTName+"_offset2"],histRange=histRange, bins=100)
        dfo.hist2d(regName, 'dfd3', "SvB_pzz", [FvTName+"_offset1",FvTName+"_offset2"],histRange=histRange, bins=100)




if "Nominal" in args.FvTName:
    print("Blind 4 tag SR")
    df = df.loc[ (df.SR==False) | (df.d4==False) ]

dfo = dataFrameOrganizer(df)
#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) )

doRegion("SB")


#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) )

doRegion("CR")


#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) )

doRegion("SR")


    
    
