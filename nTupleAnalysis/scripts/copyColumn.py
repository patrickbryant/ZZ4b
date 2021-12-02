import time, os, sys
from pathlib import Path
import multiprocessing
from glob import glob
#from copy import copy
import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#sys.path.insert(0, 'ZZ4b/nTupleAnalysis/scripts/') #https://github.com/patrickbryant/PlotTools
#import matplotlibHelpers as pltHelper
import gc

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--inputFile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('--target', default=None, help='Which weights to use for JCM.')
parser.add_argument('--destination', default=None, help='Which weights to use for JCM.')
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





fileReaders = multiprocessing.Pool(10)

# Read .h5 files
dataFiles = glob(args.inputFile)

if args.destination and args.target:
    print("Copy",args.target,"output to",args.destination)
    for i, fileName in enumerate(dataFiles):
        print("Processing",fileName)
        df = pd.read_hdf(fileName, key='df')

        df[args.destination] = df[args.target] 

        df.to_hdf(fileName, key='df', format='table', mode='w')

        del df
        gc.collect()            
        print("File %2d/%d updated file  %s"%(i+1,len(dataFiles),fileName))



