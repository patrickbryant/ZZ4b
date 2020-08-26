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

#d4 = classInfo(abbreviation='d4', name=  'FourTag Data',       index=0, color='red')
d3 = classInfo(abbreviation='d3', name= 'ThreeTag Data',       index=1, color='orange')
#t4 = classInfo(abbreviation='t4', name= r'FourTag $t\bar{t}$', index=2, color='green')
#t3 = classInfo(abbreviation='t3', name=r'ThreeTag $t\bar{t}$', index=3, color='cyan')
#zz = classInfo(abbreviation='zz', name=r'$ZZ$ MC $\times100$', index=4, color='blue')
#zh = classInfo(abbreviation='zh', name=r'$ZH$ MC $\times100$', index=5, color='violet')

dfs = []

# Read .h5 files
dataFiles = glob(args.data)
#if args.data4b:
#    dataFiles += glob(args.data4b)    
#

frames = getFramesHACK(fileReaders,getFrame,dataFiles)
dfD = pd.concat(frames, sort=False)

#for k in dfD.keys():
#    print(k)

print("Add true class labels to data")
#dfD['d4'] =  dfD.fourTag
dfD['d3'] = (dfD.fourTag+1)%2
#dfD['t4'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
#dfD['t3'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
#dfD['zz'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
#dfD['zh'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)

dfs.append(dfD)

# Read .h5 files
#ttbarFiles = glob(args.ttbar)
#if args.ttbar4b:
#    ttbarFiles += glob(args.ttbar4b)    

#frames = getFramesHACK(fileReaders,getFrame,ttbarFiles)
#dfT = pd.concat(frames, sort=False)

print("Add true class labels to ttbar MC")
#dfT['t4'] =  dfT.fourTag
#dfT['t3'] = (dfT.fourTag+1)%2
#dfT['d4'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
#dfT['d3'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
#dfT['zz'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
#dfT['zh'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)

#dfs.append(dfT)


print("concatenate dataframes")
df = pd.concat(dfs, sort=False)



class dataFrameOrganizer:
    def __init__(self, dataFrame):
        self.df = dataFrame
        self.dfSelected = dataFrame
        #self.dfd4 = None
        self.dfd3 = None
        #self.dft4 = None
        #self.dft3 = None
        #self.dfbg = None
        #self.dfzz = None
        #self.dfzh = None
        #self.dfsg = None

    def applySelection(self, selection):
        print("Apply selection")
        self.dfSelected = self.df.loc[ selection ]
        
        #self.dfd4 = self.dfSelected.loc[ self.dfSelected.d4==True ]
        self.dfd3 = self.dfSelected.loc[ self.dfSelected.d3==True ]
        #self.dft4 = self.dfSelected.loc[ self.dfSelected.t4==True ]
        #self.dft3 = self.dfSelected.loc[ self.dfSelected.t3==True ]
        #self.dfbg = self.dfSelected.loc[ (self.dfSelected.d3==True) | (self.dfSelected.t4==True) ]

    def plotVar(self, var, bins=None, xmin=None, xmax=None, regName=""):

        #ttbarWeights = -getattr(self.dft3,weightName) * getattr(self.dft3,FvTName)
        if type(var) == list:
            multijet = self.dfd3[var[0]] - self.dfd3[var[1]]
        else:
            multijet = self.dfd3[var]
        multijetWeights = getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName)
        #background = np.concatenate((self.dfd3[var], self.dft4[var]))
        #backgroundWeights = np.concatenate((getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName), getattr(self.dft4,weightName)))

        #self.dsd4 = pltHelper.dataSet(name=d4.name, 
        #                              points =self.dfd4[var],
        #                              weights=getattr(self.dfd4,weightName), 
        #                              color=d4.color, alpha=1.0, linewidth=1)
        #self.bkgd = pltHelper.dataSet(name='Background Model', 
        #                              points =background,
        #                              weights=backgroundWeights, 
        #                              color='brown', alpha=1.0, linewidth=1)
        #self.dst4 = pltHelper.dataSet(name=t4.name, 
        #                              points =self.dft4[var],
        #                              weights=getattr(self.dft4,weightName),
        #                              color=t4.color, alpha=1.0, linewidth=1)
        self.dsm3 = pltHelper.dataSet(name='ThreeTag Multijet', 
                                      points =multijet,
                                      weights=multijetWeights,
                                      color=d3.color, alpha=1.0, linewidth=1)
        #self.dst3 = pltHelper.dataSet(name=t3.name, 
        #                              points=self.dft3[var],
        #                              weights=ttbarWeights,
        #                              color=t3.color, alpha=1.0, linewidth=1)

        #datasets = [self.dsd4,self.bkgd,self.dst4,self.dsm3,self.dst3]
        datasets = [self.dsm3]


        if not bins: bins=50
        if type(var)!=list:
            if type(xmin)==type(None): xmin = self.dfSelected[var].min()
            if type(xmax)==type(None): xmax = self.dfSelected[var].max()
        else:
            xmin = -1
            xmax = 1

        width = (xmax-xmin)/bins
        bins = [xmin + b*width for b in range(-1,bins+1)]

        if type(var) == list:            
            xlabel = var[0]+"_minus_"+var[1]
        else:
            xlabel = var

        args = {'dataSets': datasets,
                #'ratio': [0,1],
                #'ratioRange': [0.5,1.5],
                #'ratioTitle': 'Data / Model',
                'bins': bins,
                'xlabel': xlabel.replace('_',' '),
                'ylabel': 'Events / Bin',
                }
        fig = pltHelper.histPlotter(**args)
        figName = outputDir + "/"+regName+"_"+xlabel+'.pdf'
        fig.savefig(figName)
        print(figName)


    def plotCompVar(self, var, legName, bins=None, xmin=None, xmax=None, regName=""):
        
        #ttbarWeights = -getattr(self.dft3,weightName) * getattr(self.dft3,FvTName)
        plotVar1 = self.dfd3[var[0]]
        plotVar2 = self.dfd3[var[1]]

        multijetWeights = getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName)

        #self.dsd4 = pltHelper.dataSet(name=d4.name, 
        #                              points =self.dfd4[var],
        #                              weights=getattr(self.dfd4,weightName), 
        #                              color=d4.color, alpha=1.0, linewidth=1)
        #self.bkgd = pltHelper.dataSet(name='Background Model', 
        #                              points =background,
        #                              weights=backgroundWeights, 
        #                              color='brown', alpha=1.0, linewidth=1)
        #self.dst4 = pltHelper.dataSet(name=t4.name, 
        #                              points =self.dft4[var],
        #                              weights=getattr(self.dft4,weightName),
        #                              color=t4.color, alpha=1.0, linewidth=1)

        self.dv1 = pltHelper.dataSet(name=legName[0], 
                                      points =plotVar1,
                                      weights=multijetWeights,
                                      color='red', alpha=1.0, linewidth=1)

        self.dv2 = pltHelper.dataSet(name=legName[1], 
                                      points =plotVar2,
                                      weights=multijetWeights,
                                      color='blue', alpha=1.0, linewidth=1)


        #datasets = [self.dsd4,self.bkgd,self.dst4,self.dsm3,self.dst3]
        datasets = [self.dv1,self.dv2]


        if not bins: bins=50
        if type(xmin)==type(None): xmin = self.dfSelected[var[0]].min()
        if type(xmax)==type(None): xmax = self.dfSelected[var[0]].max()
        width = (xmax-xmin)/bins
        bins = [xmin + b*width for b in range(-1,bins+1)]

        xlabel = var[0]+"_vs_"+var[1]

        args = {'dataSets': datasets,
                'ratio': [0,1],
                'ratioRange': [0.5,1.5],
                'ratioTitle': legName[0]+' / '+legName[1],
                'bins': bins,
                'xlabel': xlabel.replace('_',' '),
                'ylabel': 'Events / Bin',
                }
        fig = pltHelper.histPlotter(**args)
        figName = outputDir + "/"+regName+"_"+xlabel+'.pdf'
        fig.savefig(figName)
        print(figName)


    def plotRWVar(self, var, FvT1Name, FvT2Name, legName1, legName2, bins=None, xmin=None, xmax=None, regName=""):

        #ttbarWeights = -getattr(self.dft3,weightName) * getattr(self.dft3,FvTName)
        if type(var) == list:
            multijet    = self.dfd3[var[0]] - self.dfd3[var[1]]
        else:
            multijet    = self.dfd3[var]

        multijetWeights_1 = getattr(self.dfd3,weightName) * getattr(self.dfd3,FvT1Name)
        multijetWeights_2 = getattr(self.dfd3,weightName) * getattr(self.dfd3,FvT2Name)

        self.dsm3_1 = pltHelper.dataSet(name=legName1, 
                                        points =multijet,
                                        weights=multijetWeights_1,
                                        color='red', alpha=1.0, linewidth=1)

        self.dsm3_2 = pltHelper.dataSet(name=legName2, 
                                        points =multijet,
                                        weights=multijetWeights_2,
                                        color='blue', alpha=1.0, linewidth=1)


        #datasets = [self.dsd4,self.bkgd,self.dst4,self.dsm3,self.dst3]
        datasets = [self.dsm3_1, self.dsm3_2]


        if not bins: bins=50
        if type(var)!=list:
            if type(xmin)==type(None): xmin = self.dfSelected[var].min()
            if type(xmax)==type(None): xmax = self.dfSelected[var].max()
        else:
            xmin = -1
            xmax = 1

        width = (xmax-xmin)/bins
        bins = [xmin + b*width for b in range(-1,bins+1)]

        if type(var) == list:            
            xlabel = var[0]+"_minus_"+var[1]
        else:
            xlabel = var

        args = {'dataSets': datasets,
                'ratio': [0,1],
                'ratioRange': [0.8,1.2],
                'ratioTitle': legName1+' / '+legName2,
                'bins': bins,
                'xlabel': xlabel.replace('_',' '),
                'ylabel': 'Events / Bin',
                }

        fig = pltHelper.histPlotter(**args)
        figName = outputDir + "/"+regName+"_"+xlabel+'_rw'+legName1+'_vs_'+legName2+'.pdf'
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
        figName = outputDir +"/"+regName+"_"+dfName+"_"+xvar+"_vs_"+ylabel+("_reweight" if reweight else "")+".pdf"
        fig.savefig(figName)
        print(figName)


if "Nominal" in args.FvTName:
    print("Blind 4 tag SR")
    df = df.loc[ (df.SR==False) | (df.d4==False) ]


def makeRegionPlots(regName):
    
    for epochPair in [("e2","e50"),
                      ("e9","e50"),                  
                      ("e39","e50"),                  
                      ("e45","e50"),                  
                      ("e49","e50"),                
                      ]:
        histRange = [[0,2],[0,2]]
        dfo.hist2d(regName, 'dfd3', "FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[1], "FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[0],histRange=histRange, bins=100)

        histRange = [[0,1],[-1,1]]
        dfo.hist2d(regName, 'dfd3', "SvB_pzh", ["FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[1], "FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[0]],histRange=histRange, bins=100)
        dfo.hist2d(regName, 'dfd3', "SvB_pzz", ["FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[1], "FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[0]],histRange=histRange, bins=100)

        dfo.plotRWVar('SvB_pzh', FvT1Name="FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[0], FvT2Name="FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[1], legName1 = epochPair[0], legName2 = epochPair[1], regName=regName, xmin=0.0, xmax=1)
        dfo.plotRWVar('SvB_pzz', FvT1Name="FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[0], FvT2Name="FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[1], legName1 = epochPair[0], legName2 = epochPair[1], regName=regName, xmin=0.0, xmax=1)

        dfo.plotCompVar(["FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[0],"FvT_3bMix4b_rWbW2_v0_comb_"+epochPair[1]], legName = [epochPair[0],epochPair[1]], regName=regName, xmin=0.0, xmax=2)
    
    for v in varsToPlot:
        xmax = None
    
        if type(v) != list:
            if not v.find('SvB') == -1: xmax = 1.0
            if not v.find('FvT') == -1: xmax = 3.0
    
        dfo.plotVar(v, regName=regName, xmin=0.0, xmax=xmax)



varsToPlot = [FvTName, 
              "FvT_3bMix4b_rWbW2_v0_comb_e50",
              "FvT_3bMix4b_rWbW2_v0_comb_e49",
              "FvT_3bMix4b_rWbW2_v0_comb_e45",
              "FvT_3bMix4b_rWbW2_v0_comb_e39",
              "FvT_3bMix4b_rWbW2_v0_comb_e9",
              "FvT_3bMix4b_rWbW2_v0_comb_e2",
              [FvTName,"FvT_3bMix4b_rWbW2_v0_comb_e50"],
              [FvTName,"FvT_3bMix4b_rWbW2_v0_comb_e49"],
              [FvTName,"FvT_3bMix4b_rWbW2_v0_comb_e45"],
              [FvTName,"FvT_3bMix4b_rWbW2_v0_comb_e39"],
              [FvTName,"FvT_3bMix4b_rWbW2_v0_comb_e9"],
              [FvTName,"FvT_3bMix4b_rWbW2_v0_comb_e2"],
              'SvB_ps', 'SvB_pzz', 'SvB_pzh', 'nSelJets','dR0123', 'dR0213', 'dR0312']


dfo = dataFrameOrganizer(df)
#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) )
makeRegionPlots("SB")
#
##dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) )
makeRegionPlots("CR")

#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) )
makeRegionPlots("SR")

