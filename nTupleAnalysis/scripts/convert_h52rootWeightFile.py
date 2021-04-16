import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
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
parser.add_argument(      '--outFile', default='', type=str, help='Output root File dir.')
parser.add_argument(      '--outName', default='', type=str, help='Output root File dir.')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
parser.add_argument(      '--fvtNameList', default=None, help="comma separated list of jcmNames")
parser.add_argument(      '--classifierName', default="FvT", help="comma separated list of jcmNames")
args = parser.parse_args()

inPaths = args.inFile.split()
inFiles = []
for path in inPaths:
    if "root://" in path:
        inFiles.append(path)
    else:
        inFiles += glob(path)
print inFiles



def convert(inFile):
    print inFile
    h5File = inFile
    removeLocalH5File = False
    outFile = args.outFile if args.outFile else inFile.replace(".h5",args.outName+".root")

    if "root://" in inFile: # first need to copy h5 file locally
        picoAODh5 = inFile.split('/')[-1]
        cmd = 'xrdcp -f '+inFile+' '+picoAODh5
        print cmd
        os.system(cmd)
        h5File = picoAODh5
        removeLocalH5File = True

        outFile = picoAODh5.replace(".h5","_weights_"+args.outName+".root")

    # Read .h5 File
    #df = pd.read_hdf(inFile, key="df", chunksize=)
    if args.debug: print "make pd.HDFStore"
    store = pd.HDFStore(h5File, 'r')
    if args.debug: print store
    nrows = int(store.get_storer('df').nrows)
    chunksize = 1e4
    df = store.select('df', start=0, stop=1)

    class variable:
        def __init__(self, name, dtype=np.float32):
            self.name = name
            self.add = True
            self.convert = self.add
            self.array = np.array([0], dtype=dtype)

    variables = [variable("m4j")]

    if args.fvtNameList:
        fvtNameList = args.fvtNameList.split(",")
        for fvtName in fvtNameList:
            print "Adding "+args.classifierName+" wieghts for ",fvtName
            variables += [
                variable(args.classifierName+fvtName),
                ]

    convertVariables=[]
    for variable in variables:
        if variable.convert: convertVariables.append(variable)

    variables=convertVariables
    if len(variables)==0:
        print "Nothing to add or update..."
        return

    #
    # 
    print "Make file",outFile
    f_new = ROOT.TFile(outFile, "RECREATE")
    newTree = ROOT.TTree("Events",args.classifierName+" weights " )

    for variable in variables:
        if variable.add:
            newTree.Branch("weight_"+variable.name, variable.array, "weight_"+variable.name+"/F")

    #
    #  Event Loop
    #

    n=0
    #for i, row in df.iterrows():
    for chunk in range(int(nrows//chunksize) + 1):
        start, stop= int(chunk*chunksize), int((chunk+1)*chunksize)
        df = store.select('df', start=start, stop=stop)
        for i, row in df.iterrows():
            #e = int(i + chunk*chunksize)
            for variable in variables:
                if variable.convert:
                    variable.array[0] = row[variable.name]
            newTree.Fill()
            n+=1

            if(n)%10000 == 0 or (n) == nrows:
                sys.stdout.write("\rEvent %10d of %10d | %3.0f%% %s"%(n,nrows, (100.0*n)/nrows, outFile))
                sys.stdout.flush()
                

    print outFile,"store.close()"
    store.close()

    print "number of Events in H5 File",n
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

    print "done:",inFile,"->",outFile


workers = multiprocessing.Pool(min(len(inFiles),3))
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
