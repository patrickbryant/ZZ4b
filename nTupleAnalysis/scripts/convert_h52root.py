import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import os, sys
from glob import glob
from copy import copy
from array import array
import multiprocessing
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--inFile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5', type=str, help='Input h5 File.')
#parser.add_argument('-o', '--outFile', default='', type=str, help='Output root File dir.')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

inPaths = args.inFile.split()
inFiles = []
for path in inPaths:
    inFiles += glob(path)
print inFiles

def convert(inFile):
    print inFile
    # Read .h5 File
    #df = pd.read_hdf(inFile, key="df", chunksize=)
    store = pd.HDFStore(inFile, 'r')
    nrows = int(store.get_storer('df').nrows)
    chunksize = 1e4
    df = store.select('df', start=0, stop=1)

    #print df.iloc[0]
    
    outFile = inFile.replace(".h5",".root")
    f_old = ROOT.TFile(outFile, "READ")
    tree = f_old.Get("Events;1")
    runs = f_old.Get("Runs")
    lumi = f_old.Get("LuminosityBlocks")

    tempFile = outFile.replace(".root","_temp.root")
    f_new = ROOT.TFile(tempFile,"RECREATE")
    print "Clone tree"
    newTree = tree.CloneTree(0)
    newRuns = runs.CloneTree()
    newLumi = lumi.CloneTree()

    cloneTree=False
    def addOrUpdate(branch):
        add, update = False, False
        if branch in df:
            #print tree.FindBranch(branch)
            if "nil" in str(tree.FindBranch(branch)):
                add=True
                print "Add", branch
            else:
                print "Update", branch
                update=True
                #tree.SetBranchStatus(branch, 0)
        return add, update
        
    class variable:
        def __init__(self, name, dtype=np.float32):
            self.name = name
            self.add, self.update = addOrUpdate(name)
            self.convert = self.add or self.update
            self.array = np.array([0], dtype=dtype)

    variables = [variable("FvT"),
                 variable("FvT_pd4"),
                 variable("FvT_pd3"),
                 variable("FvT_pt4"),
                 variable("FvT_pt3"),
                 variable("FvT_pm4"),
                 variable("FvT_pm3"),
                 variable("FvT_pd"),
                 variable("FvT_pt"),
                 variable("FvT_p3"),
                 variable("FvT_p4"),
                 variable("SvB_ps"),
                 variable("SvB_pzz"),
                 variable("SvB_pzh"),
                 variable("SvB_ptt"),
                 ]

    convertVariables=[]
    for variable in variables:
        if variable.convert: convertVariables.append(variable)
        if variable.update: cloneTree = True
    variables=convertVariables
    if len(variables)==0:
        print "Nothing to add or update..."
        return

    print "tree.GetEntries() before",tree.GetEntries()
    #newTree.SetName("temp")
    # if cloneTree:
    #     print "Clone tree"
    #     newTree = tree.CloneTree(0)
    #     # print "Overwrite tree"
    #     # f.Write(tree.GetName(), ROOT.gROOT.kOverwrite)
    #     # print "Close",outFile
    #     # f.Close()
    #     # print "Reopen",outFile
    #     # f = ROOT.TFile(outFile, "UPDATE")
    #     # f.ls()
    #     # tree = f.Get("Events;1")
    #     print newTree

    for variable in variables:
        if variable.add:
            newTree.Branch(variable.name, variable.array, variable.name+"/F")
        if variable.update:
            newTree.SetBranchAddress(variable.name, variable.array)#, variable.name+"/F")

    n=0
    #for i, row in df.iterrows():
    for chunk in range(int(nrows//chunksize) + 1):
        start, stop= int(chunk*chunksize), int((chunk+1)*chunksize)
        df = store.select('df', start=start, stop=stop)
        for i, row in df.iterrows():
            #e = int(i + chunk*chunksize)
            tree.GetEntry(n)
            for variable in variables:
                if variable.convert:
                    variable.array[0] = row[variable.name]
            newTree.Fill()
            n+=1

            if(n)%10000 == 0 or (n) == nrows:
                sys.stdout.write("\rEvent "+str(n)+" of "+str(nrows)+" | "+str(int((n)*100.0/nrows))+"% "+outFile)
                sys.stdout.flush()

    #f.Delete("Events;1")
    #tree.Delete()
    #newTree.SetName("Events")
    print outFile,"store.close()"
    store.close()
    #tree.SetEntries(nrows)
    print "newTree.GetEntries() after",newTree.GetEntries()
    #print outFile,"tree.SetEntries(%d)"%n
    #tree.SetEntries(n)
    #tree.Show(0)
    print 

    print outFile,".Write(newTree.GetName(), ROOT.gROOT.kOverwrite)"
    #f.Write(newTree.GetName(), ROOT.gROOT.kOverwrite)
    #f.Append(newTree)
    f_new.Write()
    f_new.ls()
    print outFile,".Close()"
    f_new.Close()
    f_old.Close()
    cmd = "mv "+tempFile+" "+outFile
    print cmd
    os.system(cmd)
    print "done:",inFile,"->",outFile



workers = multiprocessing.Pool(12)
done=0
for output in workers.imap_unordered(convert,inFiles):
    if output != None:
        print output
    else: 
        done+=1
        print "finished converting",done,"of",len(inFiles),"files"
#for f in inFiles: convert(f)
for f in inFiles: print "converted:",f
print "done"
