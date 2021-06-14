import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import os, sys, time
from glob import glob
from copy import copy
from array import array
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import parseXRD
import multiprocessing
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--inFile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5', type=str, help='Input h5 File.')
parser.add_argument('-o', '--outFile', default='', type=str, help='Output root File dir.')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
parser.add_argument(      '--fvtNameList', default=None, help="comma separated list of jcmNames")
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
    print 'convert(%s)'%inFile
    h5File = inFile
    removeLocalH5File = False
    outFile = args.outFile if args.outFile else inFile.replace(".h5",".root")
    outFile = outFile.split('/')
    outDir, outFile  = '/'.join(outFile[:-1])+'/', outFile[-1]
    #tempDir = outDir

    url, path = None, None
    if 'root://' in outDir:
        url, path = parseXRD(outDir)

    xrdcpOutFile = False
    if "root://" in inFile: # first need to copy h5 file locally
        picoAODh5 = inFile.split('/')[-1]
        cmd = 'xrdcp -f '+inFile+' '+picoAODh5
        print cmd
        os.system(cmd)
        h5File = picoAODh5
        removeLocalH5File = True

        # # Also need to copy the root file locally
        # tempDir = ''
        # xrdcpOutFile = True
        # cmd = 'xrdcp -f '+outDir+outFile+' '+outFile
        # print cmd
        # os.system(cmd)

    # Read .h5 File
    #df = pd.read_hdf(inFile, key="df", chunksize=)
    if args.debug: print "make pd.HDFStore"
    store = pd.HDFStore(h5File, 'r')
    if args.debug: print store
    nrows = int(store.get_storer('df').nrows)
    chunksize = 1e4
    df = store.select('df', start=0, stop=1)

    #print df.iloc[0]
    
    print 'ROOT.TFile.Open("%s%s", "READ")'%(outDir,outFile)
    f_old = ROOT.TFile.Open(outDir+outFile, "READ")
    if args.debug: print f_old
    tree = f_old.Get("Events")
    runs = f_old.Get("Runs")
    lumi = f_old.Get("LuminosityBlocks")

    #tree.Show(0)

    cloneTree=False
    def addOrUpdate(branch):
        add, update = False, False
        #print "addOrUpdate(%s)"%branch
        if branch in df:
            #print "tree.FindBranch(%s)"%branch
            try: 
                tree.FindBranch(branch).GetAddress()
                print "Update", branch
                update=True
                #tree.SetBranchStatus(branch, 0)
            except ReferenceError:
                add=True
                print "Add", branch
        else:
            print "not in df",branch
        return add, update
        
    class variable:
        def __init__(self, name, dtype=np.float32, default=0):
            self.name = name
            self.add, self.update = addOrUpdate(name)
            self.convert = self.add or self.update
            self.array = np.array([0], dtype=dtype)
            self.default = np.array([default], dtype=dtype)

        def set_default(self):
            if self.convert:
                self.array[0] = self.default[0]

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
                 variable("FvT_q_1234"),
                 variable("FvT_q_1324"),
                 variable("FvT_q_1423"),
                 variable("SvB_ps"),
                 variable("SvB_pzz"),
                 variable("SvB_pzh"),
                 variable("SvB_ptt"),
                 variable("SvB_q_1234"),
                 variable("SvB_q_1324"),
                 variable("SvB_q_1423"),
                 variable("SvB_MA_ps"),
                 variable("SvB_MA_pzz"),
                 variable("SvB_MA_pzh"),
                 variable("SvB_MA_ptt"),
                 variable("SvB_MA_q_1234"),
                 variable("SvB_MA_q_1324"),
                 variable("SvB_MA_q_1423"),
                 ]

    if args.fvtNameList:
        fvtNameList = args.fvtNameList.split(",")
        for fvtName in fvtNameList:
            print "Adding FVT wieghts for ",fvtName
            variables += [
                variable("FvT"+fvtName),
                #variable("FvT"+fvtName+"_pd4"),
                #variable("FvT"+fvtName+"_pd3"),
                #variable("FvT"+fvtName+"_pt4"),
                #variable("FvT"+fvtName+"_pt3"),
                #variable("FvT"+fvtName+"_pm4"),
                #variable("FvT"+fvtName+"_pm3"),
                #variable("FvT"+fvtName+"_pd"),
                #variable("FvT"+fvtName+"_pt"),
                #variable("FvT"+fvtName+"_p3"),
                #variable("FvT"+fvtName+"_p4"),
                #variable("FvT"+fvtName+"_q_1234"),
                #variable("FvT"+fvtName+"_q_1324"),
                #variable("FvT"+fvtName+"_q_1423"),
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

    tempFile = outFile.replace(".root","_temp.root")
    print "Make temp file",outDir+tempFile
    f_new = ROOT.TFile.Open(outDir+tempFile, "RECREATE")
    print "Clone tree"
    newTree = tree.CloneTree(0)
    print "Clone runs"
    newRuns = runs.CloneTree()
    print "Clone lumi"
    newLumi = lumi.CloneTree()

    for variable in variables:
        if variable.add:
            newTree.Branch(variable.name, variable.array, variable.name+"/F")
        if variable.update:
            newTree.SetBranchAddress(variable.name, variable.array)#, variable.name+"/F")

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
                

    print outDir+outFile,"store.close()"
    store.close()

    if removeLocalH5File:
        cmd = "rm "+h5File
        print cmd
        os.system(cmd)

    print "newTree.GetEntries() after",newTree.GetEntries()
    print 

    print outDir+tempFile,".Write()"
    f_new.Write()
    print outDir+tempFile,".Close()"
    f_new.Close()
    print outDir+outFile,".Close()"
    f_old.Close()

    if url:
        cmd = 'xrdfs %s mv %s%s %s%s'%(url, path,tempFile, path,outFile)
        print cmd
        os.system(cmd)
    else:
        cmd = 'mv %s%s %s%s'%(outDir,tempFile, outDir,outFile)        
        print cmd
        os.system(cmd)

    # if xrdcpOutFile:
    #     cmd = 'xrdcp -f %s %s%s'%(outFile, outDir, outFile)
    #     print cmd
    #     os.system(cmd)        
    #     cmd = 'rm '+outFile
    #     print cmd
    #     os.system(cmd)

    print "done:",inFile,"->",outDir+outFile


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
