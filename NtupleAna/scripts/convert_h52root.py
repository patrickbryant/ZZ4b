import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import glob, os, sys
from copy import copy
from array import array

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--infile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5', type=str, help='Input h5 file.')
parser.add_argument('-o', '--outfile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root', type=str, help='Output root file dir.')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

# Read .h5 file
df = pd.read_hdf(args.infile, key='df')

f = ROOT.TFile(args.outfile, "UPDATE")
tree = f.Get("Events;1")

nTagClassifier = None
if 'nTagClassifier' in df:
    print "Add nTagClassifier"
    nTagClassifier = array('f', [0])
    tree.Branch('nTagClassifier', nTagClassifier, 'nTagClassifier/F')

ZHvsBackgroundClassifier = None
if 'ZHvsBackgroundClassifier' in df:
    print "Add ZHvsBackgroundClassifier"
    ZHvsBackgroundClassifier = array('f', [0])
    tree.Branch('ZHvsBackgroundClassifier', ZHvsBackgroundClassifier, 'ZHvsBackgroundClassifier/F')

for i, row in df.iterrows():
    tree.GetEntry(i)
    if nTagClassifier != None: nTagClassifier[0] = row['nTagClassifier']
    if ZHvsBackgroundClassifier != None: ZHvsBackgroundClassifier[0] = row['ZHvsBackgroundClassifier']
    tree.Fill()

    if(i+1)%10000 == 0 or i+1 == df.shape[0]:
        sys.stdout.write("\rEvent "+str(i+1)+" of "+str(df.shape[0])+" | "+str(int((i+1)*100.0/df.shape[0]))+"% ")
        sys.stdout.flush()

tree.SetEntries(df.shape[0])

print

f.Write(tree.GetName(), ROOT.gROOT.kOverwrite)
f.Close()
