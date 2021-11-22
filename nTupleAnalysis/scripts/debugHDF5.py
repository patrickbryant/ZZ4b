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
dataFiles = glob(args.inputFile)

frames = getFramesHACK(fileReaders,getFrame,dataFiles)
dfD = pd.concat(frames, sort=False)

for k in dfD.keys():
    print(k)

print("Add true class labels to data")
dfD['d4'] =  dfD.fourTag
dfD['d3'] = (dfD.fourTag+1)%2

dfs.append(dfD)


print("concatenate dataframes")
df = pd.concat(dfs, sort=False)
print(df.head())
for i in range(10):
    print( df["nSelJets"][0:10].values[i],
           df["SB"][0:10].values[i],
           df["CR"][0:10].values[i],
           df["SR"][0:10].values[i],
           df[args.FvTName][0:10].values[i],
           df[args.weightName][0:10].values[i]
    )



class dataFrameOrganizer:
    def __init__(self, dataFrame):
        self.df = dataFrame
        self.dfSelected = dataFrame
        self.dfd4 = None
        self.dfd3 = None

    def applySelection(self, selection):
        print("Apply selection")
        self.dfSelected = self.df.loc[ selection ]
        
        self.dfd4 = self.dfSelected.loc[ self.dfSelected.d4==True ]
        self.dfd3 = self.dfSelected.loc[ self.dfSelected.d3==True ]

    def printCounts(self):
        print("d4 weights",self.dfd4.shape[0],np.sum(getattr(self.dfd4,weightName)))
        print("d3 weights",self.dfd3.shape[0])#,np.sum(getattr(self.dfd3,weightName)))


#print("Blind 4 tag SR")
#df = df.loc[ (df.SR==False) | (df.d4==False) ]

dfo = dataFrameOrganizer(df)



#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SB==True) )
#dfo.applySelection( (dfo.df.passHLT==True) | (dfo.df.passHLT==False) )
dfo.printCounts()
#print(getattr(dfo.dfd4,weightName))
#print(getattr(dfo.dfd4,"m4j"))

#weightData = getattr(dfo.dfd4,weightName)
#print(weightData)
#print(weightData.size)
#print(type(weightData))
##for index in range(weightData.size):
##    print(index, weightData.loc[index])
#print("Start")
#print(weightData[:10])
#print(weightData.at[0])
#print("end")

#pd.set_option('display.max_rows', None)

print("Begin")
print(type(dfo.dfd4))
print(dfo.dfd4[[weightName,"dRjjClose",args.FvTName,"canJet0_pt"]][:])

print("End")
print("FvT")
#print(dfo.df.FvT_Nominal)





#print("All")
#print(dfo.dfd4["m4j"])
#print("ONE")
#print(dfo.dfd4["m4j"].at[4])
#print(dfo.dfd4["m4j"].loc[4])
##print(getattr(dfo.dfd4,"m4j").loc(4))


#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.CR==True) )
dfo.printCounts()



#dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) & (dfo.df.passXWt==True) )
dfo.applySelection( (dfo.df.passHLT==True) & (dfo.df.SR==True) )
dfo.printCounts()

