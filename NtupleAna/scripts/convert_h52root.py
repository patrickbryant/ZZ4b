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


ZHvsBackgroundClassifier = np.float32([0])
ZHvsBackgroundClassifierStatus = False
if "ZHvsBackgroundClassifier" in df:
    ZHvsBackgroundClassifierStatus = True
    if "nil" in str(tree.FindBranch("ZHvsBackgroundClassifier")):
        print "Add ZHvsBackgroundClassifier"
    else:
        print "Update ZHvsBackgroundClassifier"
        b = tree.GetBranch("ZHvsBackgroundClassifier")
        tree.GetListOfBranches().Remove(b)
        tree.Write()
        tree = f.Get("Events;1")
    # if "nil" in str(tree.FindBranch("ZHvsBackgroundClassifier")):
    #     print "Add ZHvsBackgroundClassifier"
    #     tree.Branch("ZHvsBackgroundClassifier", ZHvsBackgroundClassifier, "ZHvsBackgroundClassifier/F")
    # else:
    #     print "Update ZHvsBackgroundClassifier", ROOT.AddressOf(ZHvsBackgroundClassifier, 'float32')
    #     #tree.SetBranchAddress("ZHvsBackgroundClassifier", ROOT.AddressOf(ZHvsBackgroundClassifier, "float32"))
    #     tree.SetBranchAddress("ZHvsBackgroundClassifier", ZHvsBackgroundClassifier)


nTagClassifier = np.float32([0])
nTagClassifierStatus = False
if "nTagClassifier" in df:
    nTagClassifierStatus = True
    if "nil" in str(tree.FindBranch("nTagClassifier")):
        print "Add nTagClassifier"
    else:
        print "Update nTagClassifier"
        b = tree.GetBranch("nTagClassifier")
        tree.GetListOfBranches().Remove(b)
        tree.Write()
        tree = f.Get("Events;1")
    # if "nil" in str(tree.FindBranch("nTagClassifier")):
    #     print "Add nTagClassifier"
    #     tree.Branch("nTagClassifier", nTagClassifier, "nTagClassifier/F")
    # else:
    #     print "Update nTagClassifier",ROOT.AddressOf(nTagClassifier, 'float32')
    #     tree.SetBranchAddress("nTagClassifier", nTagClassifier)


if nTagClassifierStatus == False and ZHvsBackgroundClassifierStatus == False: 
    print "Nothing to add or update..."
    exit()


if           nTagClassifierStatus: tree.Branch(          "nTagClassifier",          nTagClassifier,            "nTagClassifier/F")
if ZHvsBackgroundClassifierStatus: tree.Branch("ZHvsBackgroundClassifier", ZHvsBackgroundClassifier, "ZHvsBackgroundClassifier/F")


for i, row in df.iterrows():
    tree.GetEntry(i)
    if           nTagClassifierStatus:           nTagClassifier[0] = row[          "nTagClassifier"]
    if ZHvsBackgroundClassifierStatus: ZHvsBackgroundClassifier[0] = row["ZHvsBackgroundClassifier"]
    tree.Fill()

    if(i+1)%10000 == 0 or i+1 == df.shape[0]:
        sys.stdout.write("\rEvent "+str(i+1)+" of "+str(df.shape[0])+" | "+str(int((i+1)*100.0/df.shape[0]))+"% ")
        sys.stdout.flush()

tree.SetEntries(df.shape[0])
tree.Show(0)
print

f.Write(tree.GetName(), ROOT.gROOT.kOverwrite)
f.Close()
