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
df = pd.read_hdf(args.infile, key="df")

print df.iloc[0]

f = ROOT.TFile(args.outfile, "UPDATE")
tree = f.Get("Events;1")


ZHvsBackgroundClassifierStatus = False
if "ZHvsBackgroundClassifier" in df:
    ZHvsBackgroundClassifierStatus = True
    if "nil" in str(tree.FindBranch("ZHvsBackgroundClassifier")):
        print "Add ZHvsBackgroundClassifier"
    else:
        print "Update ZHvsBackgroundClassifier"
        #tree.SetBranchStatus("ZHvsBackgroundClassifier",0)
        b = tree.GetBranch("ZHvsBackgroundClassifier")
        tree.GetListOfBranches().Remove(b)
        l = tree.GetLeaf("ZHvsBackgroundClassifier")
        tree.GetListOfLeaves().Remove(l)


nTagClassifierStatus = False
if "nTagClassifier" in df:
    nTagClassifierStatus = True
    if "nil" in str(tree.FindBranch("nTagClassifier")):
        print "Add nTagClassifier"
    else:
        print "Update nTagClassifier"
        #tree.SetBranchStatus("nTagClassifier", 0)
        b = tree.GetBranch("nTagClassifier")
        tree.GetListOfBranches().Remove(b)
        l = tree.GetLeaf("nTagClassifier")
        tree.GetListOfLeaves().Remove(l)


f.Write(tree.GetName(), ROOT.gROOT.kOverwrite)
f.Close()
print tree

if nTagClassifierStatus == False and ZHvsBackgroundClassifierStatus == False: 
    print "Nothing to add or update..."
    exit()


f = ROOT.TFile(args.outfile, "UPDATE")
f.ls()
newTree = f.Get("Events;1")
print newTree
#newTree.Show(0)


nTagClassifier = np.float32([0])
ZHvsBackgroundClassifier = np.float32([0])
if           nTagClassifierStatus: newTree.Branch(          "nTagClassifier",          nTagClassifier,            "nTagClassifier/F")
if ZHvsBackgroundClassifierStatus: newTree.Branch("ZHvsBackgroundClassifier", ZHvsBackgroundClassifier, "ZHvsBackgroundClassifier/F")

n=0
for i, row in df.iterrows():
    n+=1
    newTree.GetEntry(i)
    if           nTagClassifierStatus:           nTagClassifier[0] = row[          "nTagClassifier"]
    if ZHvsBackgroundClassifierStatus: ZHvsBackgroundClassifier[0] = row["ZHvsBackgroundClassifier"]
    newTree.Fill()

    if(i+1)%10000 == 0 or (i+1) == df.shape[0]:
        sys.stdout.write("\rEvent "+str(i+1)+" of "+str(df.shape[0])+" | "+str(int((i+1)*100.0/df.shape[0]))+"% ")
        sys.stdout.flush()
        #break

newTree.SetEntries(df.shape[0])
#newTree.SetEntries(n)
#newTree.Show(0)
print

f.Write(newTree.GetName(), ROOT.gROOT.kOverwrite)
f.Close()
