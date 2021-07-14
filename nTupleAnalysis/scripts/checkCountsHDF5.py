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
parser.add_argument('-i', '--inputFile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
args = parser.parse_args()

def getFrame(fileName):
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])
    #print("Reading",fileName)
    thisFrame = pd.read_hdf(fileName, key='df')
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    return thisFrame



def getFramesHACK(fileReaders,getFrame,dataFiles):
    largeFiles = []
    #print("dataFiles was:",dataFiles)
    for d in dataFiles:
        if Path(d).stat().st_size > 2e9:
            #print("Large File",d)
            largeFiles.append(d)
            dataFiles.remove(d)

    results = fileReaders.map_async(getFrame, sorted(dataFiles))
    frames = results.get()

    for f in largeFiles:
        frames.append(getFrame(f))

    return frames



class dataFrameOrganizer:
    def __init__(self, dataFrame):
        self.df = dataFrame
        self.dfSelected = dataFrame
        self.dfd4 = None
        self.dfd3 = None

    def applySelection(self, selection):
        #print("Apply selection")
        self.dfSelected = self.df.loc[ selection ]
        
        self.dfd4 = self.dfSelected.loc[ self.dfSelected.d4==True ]
        self.dfd3 = self.dfSelected.loc[ self.dfSelected.d3==True ]

    def printCounts(self,name):
        print("\t",name," ",np.sum(getattr(self.dfd4,weightName)),self.dfd4.shape[0])
        #print("d3 weights",self.dfd3.shape[0])#,np.sum(getattr(self.dfd3,weightName)))

    def printCounts3b(self,name):
        #print(name," ",np.sum(getattr(self.dfd4,weightName)),self.dfd4.shape[0])
        multijetWeights = getattr(self.dfd3,weightName) * getattr(self.dfd3,FvTName)
        print("\t",name," ",np.sum(multijetWeights),self.dfd3.shape[0])

fileReaders = multiprocessing.Pool(10)

print("Data 4b")


for y in ["2016","2017","2018"]:
    print(y)
    for s in range(10):
        sStr = str(s)
    

        inFile = "closureTests/UL/mixed"+y+"_3bDvTMix4bDvT_v"+sStr+"/picoAOD_3bDvTMix4bDvT_4b_wJCM_v"+sStr+".h5"
        weightName = "mcPseudoTagWeight_3bDvTMix4bDvT_v"+sStr
    
        # Read .h5 files
        dataFiles = glob(inFile)
    
        frames = getFramesHACK(fileReaders,getFrame,dataFiles)
        dfD = pd.concat(frames, sort=False)
        dfD['d4'] =  dfD.fourTag
        dfD['d3'] = (dfD.fourTag+1)%2
    
        df = pd.concat([dfD], sort=False)
        dfo = dataFrameOrganizer(df)
    
        dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) )
        dfo.printCounts(sStr)

if True:    
    print("Multi jet")
    
    for s in range(10):
        sStr = str(s)

        inFile = "closureTests/UL/data"+y+"_3b/picoAOD_3b_wJCM.h5"
        weightName = "mcPseudoTagWeight_3bDvTMix4bDvT_v"+sStr    

        FvTName = "FvT_3bDvTMix4bDvT_v"+sStr
    
        # Read .h5 files
        dataFiles = glob(inFile)
    
        frames = getFramesHACK(fileReaders,getFrame,dataFiles)
        dfD = pd.concat(frames, sort=False)
        dfD['d4'] =  dfD.fourTag
        dfD['d3'] = (dfD.fourTag+1)%2
    
        df = pd.concat([dfD], sort=False)
        dfo = dataFrameOrganizer(df)
    
        dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) )
        dfo.printCounts3b(sStr)
    
