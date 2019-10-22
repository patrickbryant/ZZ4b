import time, os, sys
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
parser.add_argument('-t', '--ttbar',      default='',    type=str, help='Input MC ttbar file in hdf5 format')
parser.add_argument('-s', '--signal',     default='', type=str, help='Input dataset file in hdf5 format')
args = parser.parse_args()

class cycler:
    def __init__(self,options=['-','\\','|','/']):
        self.cycle=0
        self.options=options
        self.m=len(self.options)
    def next(self):
        self.cycle = (self.cycle + 1)%self.m
        return self.options[self.cycle]

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

loadCycler = cycler()


d4 = classInfo(abbreviation='d4', name=  'FourTag Data',       index=0, color='red')
d3 = classInfo(abbreviation='d3', name= 'ThreeTag Data',       index=1, color='orange')
t4 = classInfo(abbreviation='t4', name= r'FourTag $t\bar{t}$', index=2, color='green')
t3 = classInfo(abbreviation='t3', name=r'ThreeTag $t\bar{t}$', index=3, color='cyan')
classes = [d4,d3,t4,t3]

nClasses = len(classes)
updateAttributes = [
    nameTitle('r',   'FvT'),
    nameTitle('pd4', 'FvT_pd4'),
    nameTitle('pd3', 'FvT_pd3'),
    nameTitle('pt4', 'FvT_pt4'),
    nameTitle('pt3', 'FvT_pt3'),
    nameTitle('pm4', 'FvT_pm4'),
    nameTitle('pm3', 'FvT_pm3'),
    ]

# Read .h5 files
frames = []
for fileName in sorted(glob(args.data)):
    print("Reading",fileName)
    frames.append(pd.read_hdf(fileName, key='df'))
dfD = pd.concat(frames, sort=False)

print("Add true class labels to data")
dfD['d4'] =  dfD.fourTag
dfD['d3'] = (dfD.fourTag+1)%2
dfD['t4'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)
dfD['t3'] = pd.Series(np.zeros(dfD.shape[0], dtype=np.uint8), index=dfD.index)

frames = []
for fileName in sorted(glob(args.ttbar)):
    print("Reading",fileName)
    frames.append(pd.read_hdf(fileName, key='df'))
dfT = pd.concat(frames, sort=False)

print("Add true class labels to ttbar MC")
dfT['t4'] =  dfT.fourTag
dfT['t3'] = (dfT.fourTag+1)%2
dfT['d4'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)
dfT['d3'] = pd.Series(np.zeros(dfT.shape[0], dtype=np.uint8), index=dfT.index)


print("concatenate data and ttbar dataframes")
df = pd.concat([dfD, dfT], sort=False)

class dataFrameOrganizer:
    def __init__(self, dataFrame):
        self.df = dataFrame
        self.dfSelected = dataFrame
        self.dfd4 = None
        self.dfd3 = None
        self.dft4 = None
        self.dft3 = None

    def applySelection(self, selection):
        print("Apply selection")
        self.dfSelected = self.df.loc[ selection ]
        
        self.dfd4 = self.dfSelected.loc[ self.dfSelected.d4==True ]
        self.dfd3 = self.dfSelected.loc[ self.dfSelected.d3==True ]
        self.dft4 = self.dfSelected.loc[ self.dfSelected.t4==True ]
        self.dft3 = self.dfSelected.loc[ self.dfSelected.t3==True ]

    def plotVar(self, var, bins=None, xmin=None, xmax=None, reweight=False):

        if reweight:
            ttbarWeights = -self.dft3.mcPseudoTagWeight * self.dft3.FvT
            # multijetWeights = np.concatenate((self.dfd3.mcPseudoTagWeight * self.dfd3.FvT, -self.dft3.mcPseudoTagWeight * self.dft3.FvT))
            multijet = self.dfd3[var]
            multijetWeights = self.dfd3.mcPseudoTagWeight * self.dfd3.FvT
            # backgroundWeights = np.concatenate((self.dfd3.mcPseudoTagWeight * self.dfd3.FvT, -self.dft3.mcPseudoTagWeight * self.dft3.FvT, self.dft4.mcPseudoTagWeight))
            background = np.concatenate((self.dfd3[var], self.dft4[var]))
            backgroundWeights = np.concatenate((self.dfd3.mcPseudoTagWeight * self.dfd3.FvT, self.dft4.mcPseudoTagWeight))
        else:
            ttbarWeights = -self.dft3.mcPseudoTagWeight
            multijet = np.concatenate((self.dfd3[var], self.dft3[var]))
            multijetWeights = np.concatenate((self.dfd3.mcPseudoTagWeight, -self.dft3.mcPseudoTagWeight))
            # multijetWeights = self.dfd3.mcPseudoTagWeight
            # background = np.concatenate((self.dfd3[var], self.dft3[var], self.dft4[var]))
            # backgroundWeights = np.concatenate((self.dfd3.mcPseudoTagWeight, -self.dft3.mcPseudoTagWeight, self.dft4.mcPseudoTagWeight))
            background = np.concatenate((self.dfd3[var], self.dft3[var], self.dft4[var]))
            backgroundWeights = np.concatenate((self.dfd3.mcPseudoTagWeight, -self.dft3.mcPseudoTagWeight, self.dft4.mcPseudoTagWeight))
            # backgroundWeights = np.concatenate((self.dfd3.mcPseudoTagWeight, self.dft4.mcPseudoTagWeight))

        self.dsd4 = pltHelper.dataSet(name=d4.name, 
                                      points =self.dfd4[var],
                                      weights=self.dfd4.mcPseudoTagWeight, 
                                      color=d4.color, alpha=1.0, linewidth=1)
        self.bkgd = pltHelper.dataSet(name='Background Model', 
                                      points =background,
                                      weights=backgroundWeights, 
                                      color='brown', alpha=1.0, linewidth=1)
        self.dst4 = pltHelper.dataSet(name=t4.name, 
                                      points =self.dft4[var],
                                      weights=self.dft4.mcPseudoTagWeight, 
                                      color=t4.color, alpha=1.0, linewidth=1)
        self.dsm3 = pltHelper.dataSet(name='ThreeTag Multijet', 
                                      points =multijet,
                                      weights=multijetWeights,
                                      color=d3.color, alpha=1.0, linewidth=1)
        self.dst3 = pltHelper.dataSet(name=t3.name, 
                                      points=self.dft3[var],
                                      weights=ttbarWeights,
                                      color=t3.color, alpha=1.0, linewidth=1)

        if type(bins)!=list:
            if not bins: bins=50
            if type(xmin)==type(None): xmin = self.dfSelected[var].min()
            if type(xmax)==type(None): xmax = self.dfSelected[var].max()
            width = (xmax-xmin)/bins
            bins = [xmin + b*width for b in range(-1,bins+1)]

        args = {'dataSets': [self.dsd4,self.bkgd,self.dst4,self.dsm3,self.dst3],
                'ratio': [0,1],
                'ratioRange': [0.5,1.5],
                'ratioTitle': 'Data / Model',
                'bins': bins,
                'xlabel': var.replace('_',' '),
                'ylabel': 'Events / Bin',
                }
        fig = pltHelper.histPlotter(**args)
        figName = var+('_reweight' if reweight else '')+'.pdf'
        fig.savefig(figName)
        print(figName)

    def hist2d(self, dfName, xvar, yvar ,bins=50,range=None,reweight=False):
        df = getattr(self,dfName)
        x,y = df[xvar],df[yvar]
        if reweight:
            weights = df.mcPseudoTagWeight * df.FvT
        else:
            weights = df.mcPseudoTagWeight
        xlabel = xvar.replace('_',' ')
        ylabel = yvar.replace('_',' ')
        args = {'x':x, 'y':y, 'weights':weights,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'zlabel': 'Events / Bin',
                'bins': bins,
                'range': range,
                }
        fig = pltHelper.hist2d(**args)
        figName = dfName+"_"+xvar+"_vs_"+yvar+("_reweight" if reweight else "")+".pdf"
        fig.savefig(figName)
        print(figName)



dfo = dataFrameOrganizer(df)
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) & (dfo.df.xWt > 2) )

dfo.plotVar('dRjjOther')
dfo.plotVar('dRjjOther', reweight=True)
