from pathlib import Path
import multiprocessing
import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--inputFile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-o', '--outputFile', default='', type=str, help='Prefix to output files.')
parser.add_argument('--weightName', default="3bMix4b_rWbW2", help='Which weights to use for JCM.')
args = parser.parse_args()


def getFrame(fileName):
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])
    print("Reading",fileName)
    thisFrame = pd.read_hdf(fileName, key='df')
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    return thisFrame



dfInput = getFrame(args.inputFile)

#print(dfInput)
#print(type(dfInput))
#print(dfInput.head())

drop_vars = []
for s in ["0","1","2","3","4"]:
    for postfix in ["pd4","pd3","pt4","pt3","pm4","pm3","p4","p3","pd","pt","q_1234","q_1324","q_1423"]:
        drop_vars.append("FvT_"+args.weightName+"_v"+s+"_"+postfix)

for v in drop_vars:
    if  v in dfInput.keys():
        print("removing",v)
        dfInput = dfInput.drop(v, axis=1)


for k in dfInput.keys():
    print(k)

store = pd.HDFStore(args.outputFile,mode='w')
store.append('df', dfInput, format='table', data_columns=None, index=False)
store.close()
