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
cloneTree=False
def addOrUpdate(branch):
    add, update = False, False
    if branch in df:
        if "nil" in str(tree.FindBranch(branch)):
            add=True
            print "Add", branch
        else:
            print "Update", branch
            update=True
            tree.SetBranchStatus(branch, 0)
    return add, update
        
class variable:
    def __init__(self, name, dtype=np.float32):
        self.name = name
        self.add, self.update = addOrUpdate(name)
        self.convert = self.add or self.update
        self.array = np.array([0], dtype=dtype)

variables = [variable("ZHvsBackgroundClassifier"),
             variable("nTagClassifier")]

convertVariables=[]
for variable in variables:
    if variable.convert: convertVariables.append(variable)
    if variable.update: cloneTree = True
variables=convertVariables
if len(variables)==0:
    print "Nothing to add or update..."
    exit()

if cloneTree:
    print "Clone tree"
    newTree = tree.CloneTree()
    print "Overwrite tree"
    f.Write(tree.GetName(), ROOT.gROOT.kOverwrite)
    print "Close",args.outfile
    f.Close()
    print "Reopen",args.outfile
    f = ROOT.TFile(args.outfile, "UPDATE")
    f.ls()
    tree = f.Get("Events;1")
    print tree

for variable in variables:
    tree.Branch(variable.name, variable.array, variable.name+"/F")

n=0
for i, row in df.iterrows():
    n+=1
    tree.GetEntry(i)
    for variable in variables:
        if variable.convert:
            variable.array[0] = row[variable.name]
    tree.Fill()

    if(i+1)%10000 == 0 or (i+1) == df.shape[0]:
        sys.stdout.write("\rEvent "+str(i+1)+" of "+str(df.shape[0])+" | "+str(int((i+1)*100.0/df.shape[0]))+"% ")
        sys.stdout.flush()
        #break

tree.SetEntries(df.shape[0])
#tree.SetEntries(n)
tree.Show(0)
print

f.Write(tree.GetName(), ROOT.gROOT.kOverwrite)
f.Close()
