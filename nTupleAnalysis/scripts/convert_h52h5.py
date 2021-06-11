import pandas as pd

import numpy as np
import os, sys
from glob import glob
from copy import copy
from array import array
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis

import multiprocessing
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--inFile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5', type=str, help='Input h5 File.')
parser.add_argument('-o', '--outputName', default='', type=str, help='Output root File dir.')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
parser.add_argument(      '--varList', default=None, help="comma separated list of variables")
args = parser.parse_args()

inPaths = args.inFile.split()
inFiles = []
for path in inPaths:
    if "root://" in path:
        inFiles.append(path)
    else:
        inFiles += glob(path)
print(inFiles)


for f in inFiles:
    h5FileIn = f
    h5FileOut = h5FileIn.replace(".h5","_"+args.outputName+".h5")
    print("Reading",h5FileIn)
    storeIn = pd.HDFStore(h5FileIn, 'r')

    nrows = int(storeIn.get_storer('df').nrows)
    ncols = int(storeIn.get_storer('df').ncols)
    print("Input: nrows", nrows, "ncols",ncols)
    df = storeIn.select('df', start=0, stop=1)

    chunksize = 1e4

    print("Writting",h5FileOut)
    storeOut = pd.HDFStore(h5FileOut ,mode='w')

    data = {}
    #varList = "DvT3_DataVsTT_3b_pt3"
    varsToCopy = args.varList.split(",") + ["dRjjClose"]
    print("Variableh output:")
    for v in varsToCopy:
        print("\t",v)

    dfOut = None

    for chunk in range(int(nrows//chunksize) + 1):
        start, stop= int(chunk*chunksize), int((chunk+1)*chunksize)
        df = storeIn.select('df', start=start, stop=stop)

        if dfOut is None:   dfOut = df[varsToCopy]
        else:               dfOut = dfOut.append(df[varsToCopy])


    
    storeOut.append('df', dfOut, format='table', data_columns=None, index=False)
    storeOut.close()

    #
    #  Test
    #
    storeTest = pd.HDFStore(h5FileOut, 'r')
    nrows = int(storeTest.get_storer('df').nrows)
    ncols = int(storeTest.get_storer('df').ncols)
    print("Output: nrows", nrows, "ncols",ncols)

    print("converted:",f)
    
print("done")
