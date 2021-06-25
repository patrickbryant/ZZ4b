import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import os, sys, time
from glob import glob
from copy import copy
from array import array
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
import multiprocessing
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument( '--inFileH5', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5', type=str, help='Input h5 File.')
parser.add_argument( '--inFileROOT', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5', type=str, help='Input h5 File.')
parser.add_argument(      '--outFile', default='', type=str, help='Output root File dir.')
parser.add_argument(      '--outName', default='', type=str, help='Output root File dir.')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
parser.add_argument(      '--varList', default=None, help="comma separated list of variables")
parser.add_argument(      '--treeName', default="FvT", help="comma separated list of jcmNames")
parser.add_argument(      '--weightPrefix', default="", help="prefix for the weights in the tree")
args = parser.parse_args()

inPaths = args.inFileH5.split()
inFilesH5 = []
for path in inPaths:
    if "root://" in path:
        inFilesH5.append(path)
    else:
        inFilesH5 += glob(path)
print inFilesH5


inPaths = args.inFileROOT.split()
inFilesROOT = []
for path in inPaths:
    if "root://" in path:
        inFilesROOT.append(path)
    else:
        inFilesROOT += glob(path)
print inFilesROOT



def convert(inFileH5, inFileROOT):
    print 'convert(',inFileH5,inFileROOT,")"
    h5File = inFileH5
    removeLocalH5File = False

    xrdcpOutFile = ("root://" in args.outFile)

    outFile = args.outFile if args.outFile else inFileH5.replace(".h5",args.outName+".root")
    outFile = outFile.split('/')
    outDir, outFile  = '/'.join(outFile[:-1])+'/', outFile[-1]

    if "root://" in inFileH5: # first need to copy h5 file locally
        picoAODh5 = inFileH5.split('/')[-1]
        cmd = 'xrdcp -f '+inFileH5+' '+picoAODh5
        print cmd
        os.system(cmd)
        h5File = picoAODh5
        removeLocalH5File = True



    # Read .h5 File
    if args.debug: print "make pd.HDFStore"
    store = pd.HDFStore(h5File, 'r')
    if args.debug: print store
    nrows = int(store.get_storer('df').nrows)
    chunksize = 1e4
    df = store.select('df', start=0, stop=1)

    print 'ROOT.TFile.Open('+inFileROOT+', "READ")'
    f_ref = ROOT.TFile.Open(inFileROOT, "READ")
    tree = f_ref.Get("Events")
    runs = f_ref.Get("Runs")
    lumi = f_ref.Get("LuminosityBlocks")


    class variable:
        def __init__(self, name, dtype=np.float32, default=0):
            self.name = name
            self.add = True
            self.convert = self.add
            self.array = np.array([0], dtype=dtype)
            self.default = np.array([default], dtype=dtype)

        def set_default(self):
            if self.convert:
                self.array[0] = self.default[0]


    variables = [variable("dRjjClose")]

    if args.varList:
        varList = args.varList.split(",")
        for varName in varList:
            print "Adding wieghts for ",varName
            variables += [
                variable(varName),
                ]

    convertVariables=[]
    for variable in variables:
        if variable.convert: convertVariables.append(variable)

    variables=convertVariables
    if len(variables)==0:
        print "Nothing to add or update..."
        return

    print "tree.GetEntries() reference",tree.GetEntries()

    #
    # 
    print "Make temp file",outFile
    f_new = ROOT.TFile(outFile, "RECREATE")
    newTree = ROOT.TTree("Events",args.treeName+" weights " )

    for variable in variables:
        if variable.add:
            if variable.name == 'dRjjClose':
                newTree.Branch("weight_"+variable.name, variable.array, "weightFile_"+variable.name+"/F")
            else:
                newTree.Branch(args.weightPrefix+variable.name, variable.array, args.weightPrefix+variable.name+"/F")

    #
    #  Event Loop
    #

    n=0
    nTree = tree.GetEntries()
    chunk = 0
    startTime = time.time()
    for n in range(nTree):
        tree.GetEntry(n)
        in_h5 = tree.passHLT and (tree.SB or tree.CR or tree.SR)
        if in_h5: # this event is the next row in the h5 file, grab it
            try:
                i, row = next(df)
            except: # ran out of events in this chunk of the h5, get the next chunk
                start, stop = int(chunk*chunksize), int((chunk+1)*chunksize)
                chunk += 1
                df = store.select('df', start=start, stop=stop).iterrows()
                i, row = next(df)

            for variable in variables:
                if variable.convert:
                    variable.array[0] = row[variable.name]

        else: # not in h5, use defaults
            for variable in variables:
                variable.set_default()

        newTree.Fill()                
        if n and ((n)%10000 == 0 or (n) == nrows):
            elapsed = time.time()-startTime
            rate = n/elapsed
            secondsRemaining = (nTree-n)/rate
            h, m, s = int(secondsRemaining/3600), int(secondsRemaining/60)%60, int(secondsRemaining)%60
            sys.stdout.write("\rProcessed %10d of %10d (%4.0f/s) | %3.0f%% (done in %02d:%02d:%02d) | %s"%(n,nTree,rate, n*100.0/nTree, h,m,s, outDir+outFile))
            sys.stdout.flush()

    print inFileH5,"store.close()"
    store.close()

    print "number of Events in H5 File",nrows
    print "newTree.GetEntries() after",newTree.GetEntries()
    print 


    f_new.Write()
    f_new.ls()
    print outFile,".Close()"
    f_new.Close()

    if removeLocalH5File:
        cmd = "rm "+h5File
        print cmd
        os.system(cmd)

    if xrdcpOutFile:
        cmd = 'xrdcp -f %s %s%s'%(outFile, outDir, outFile)
        print cmd
        os.system(cmd)        
        cmd = 'rm '+outFile
        print cmd
        os.system(cmd)

    print "done:",inFileH5,"->",outFile


for i in range(len(inFilesH5)):
    convert(inFilesH5[i],inFilesROOT[i])
    print "converted:",inFilesH5[i]

print "done"
