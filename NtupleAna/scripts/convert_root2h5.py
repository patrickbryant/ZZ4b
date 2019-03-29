import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import glob, os, sys
from copy import copy

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--infile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root', type=str, help='Input root file.')
parser.add_argument('-o', '--outfile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5', type=str, help='Output pq file dir.')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

treeStr = args.infile 
tree = ROOT.TChain("Events")
tree.Add(treeStr)

# Initialize TTree
tree.SetBranchStatus("*",0)
tree.SetBranchStatus("ZHSB",1)
tree.SetBranchStatus("passDEtaBB",1)
tree.SetBranchStatus("weight",1)
tree.SetBranchStatus("fourTag",1)
tree.SetBranchStatus("dRjjClose",1)
tree.SetBranchStatus("dRjjOther",1)
tree.SetBranchStatus("aveAbsEta",1)
tree.SetBranchStatus("canJet1_pt",1)
tree.SetBranchStatus("canJet3_pt",1)
tree.Show(0)

nEvts = tree.GetEntries()
assert nEvts > 0
print(" >> Input file:",treeStr)
print(" >> nEvts:",nEvts)
outStr = args.outfile
print(" >> Output file:",outStr)


##### Start Conversion #####

# Event range to process
iEvtStart = 0
iEvtEnd   = 1000
iEvtEnd   = nEvts 
assert iEvtEnd <= nEvts
print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")

nWritten = 0
data = {'weight': [],
        'ZHSB': [],
        'passDEtaBB': [],
        'fourTag': [],
        'canJet1_pt': [],
        'canJet3_pt': [],
        'dRjjClose': [],
        'dRjjOther': [],
        'aveAbsEta': [],
        } 
#df = pd.DataFrame({''})
sw = ROOT.TStopwatch()
sw.Start()
for iEvt in list(range(iEvtStart,iEvtEnd)):

    # Initialize event
    tree.GetEntry(iEvt)
    if (iEvt+1) % 1000 == 0 or iEvt+1 == iEvtEnd:
        sys.stdout.write("\rProcessed "+str(iEvt+1)+" of "+str(nEvts)+" | "+str(int((iEvt+1)*100.0/nEvts))+"% ")
        sys.stdout.flush()

    data['weight']    .append(copy(tree.weight))
    data['ZHSB']      .append(copy(tree.ZHSB))
    data['passDEtaBB'].append(copy(tree.passDEtaBB))
    data['fourTag']   .append(copy(tree.fourTag))
    data['canJet1_pt'].append(copy(tree.canJet1_pt))
    data['canJet3_pt'].append(copy(tree.canJet3_pt))
    data['dRjjClose'] .append(copy(tree.dRjjClose))
    data['dRjjOther'] .append(copy(tree.dRjjOther))
    data['aveAbsEta'] .append(copy(tree.aveAbsEta))

    nWritten += 1

print

data['weight']     = np.array(data['weight'],     np.float32)
data['ZHSB']       = np.array(data['ZHSB'],       np.bool_)
data['passDEtaBB'] = np.array(data['passDEtaBB'], np.bool_)
data['fourTag']    = np.array(data['fourTag'],    np.bool_)
data['canJet1_pt'] = np.array(data['canJet1_pt'], np.float32)
data['canJet3_pt'] = np.array(data['canJet3_pt'], np.float32)
data['dRjjClose']  = np.array(data['dRjjClose'],  np.float32)
data['dRjjOther']  = np.array(data['dRjjOther'],  np.float32)
data['aveAbsEta']  = np.array(data['aveAbsEta'],  np.float32)

df=pd.DataFrame(data)
print "df.dtypes"
print df.dtypes
print "df.shape", df.shape

df.to_hdf(args.outfile, key='df', format='table', mode='w')

sw.Stop()
print() 
print(" >> nWritten:",nWritten)
print(" >> Real time:",sw.RealTime()/60.,"minutes")
print(" >> CPU time: ",sw.CpuTime() /60.,"minutes")
print(" >> ======================================")
