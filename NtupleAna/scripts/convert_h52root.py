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

def addOrUpdate(branch):
    if branch in df:
        if "nil" in str(tree.FindBranch(branch)):
            print "Add", branch
        else:
            print "Update", branch
            tree.SetBranchStatus(branch, 0)
            # b = tree.GetBranch("ZHvsBackgroundClassifier")
            # tree.GetListOfBranches().Remove(b)
            # l = tree.GetLeaf("ZHvsBackgroundClassifier")
            # tree.GetListOfLeaves().Remove(l)
        return True
    return False
        
class variable:
    def __init__(self, name):
        self.name = name
        self.convert = addOrUpdate(name)
        self.array = np.float32([0])

variables = [variable("ZHvsBackgroundClassifier"),
             variable("nTagClassifier")]

overwrite=False
for variable in variables:
    if variable.convert: overwrite=True
if not overwrite:
    print "Nothing to add or update..."
    exit()

print "Clone tree"
newTree = tree.CloneTree()
print "Overwrite tree"
f.Write(tree.GetName(), ROOT.gROOT.kOverwrite)
print "Close",args.outfile
f.Close()

print "Reopen",args.outfile
f = ROOT.TFile(args.outfile, "UPDATE")
f.ls()
newTree = f.Get("Events;1")
print newTree
#newTree.Show(0)


convertVariables=[]
for variable in variables:
    if variable.convert:
        convertVariables.append(variable)
        newTree.Branch(variable.name, variable.array, variable.name+"/F")
variables=convertVariables

n=0
for i, row in df.iterrows():
    n+=1
    newTree.GetEntry(i)
    for variable in variables:
        if variable.convert:
            variable.array[0] = row[variable.name]
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
