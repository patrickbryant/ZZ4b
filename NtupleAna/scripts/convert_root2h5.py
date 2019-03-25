#import pyarrow.parquet as pq
#import pyarrow as pa # pip install pyarrow==0.7.1
import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import glob, os, sys

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--infile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root', type=str, help='Input root file.')
parser.add_argument('-o', '--outfile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5', type=str, help='Output pq file dir.')
args = parser.parse_args()

treeStr = args.infile 
tree = ROOT.TChain("Events")
tree.Add(treeStr)

# Initialize TTree
tree.SetBranchStatus("*",0)
tree.SetBranchStatus("ZHSB",1)
tree.SetBranchStatus("passDEtaBB",1)
tree.SetBranchStatus("weight",1)
tree.SetBranchStatus("threeTag",1)
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

    # Event Selection
    if not (tree.ZHSB and tree.passDEtaBB and (tree.threeTag or tree.fourTag)): continue

    weight = tree.weight
    threeTag = tree.threeTag
    fourTag = tree.fourTag
    canJet1_pt = tree.canJet1_pt
    canJet3_pt = tree.canJet3_pt
    dRjjClose = tree.dRjjClose
    dRjjOther = tree.dRjjOther
    aveAbsEta = tree.aveAbsEta
    
    data['weight'].append(weight)
    # data['threeTag'].append(threeTag
    data['fourTag'].append(fourTag)
    data['canJet1_pt'].append(canJet1_pt)
    data['canJet3_pt'].append(canJet3_pt)
    data['dRjjClose'].append(dRjjClose)
    data['dRjjOther'].append(dRjjOther)
    data['aveAbsEta'].append(aveAbsEta)
    #data['w'] = weight
    #data['X'] = [canJet1_pt, canJet3_pt, dRjjClose, dRjjOther, aveAbsEta]
    #data['y'] = fourTag

    #pqdata = [pa.array([d]) if np.isscalar(d) or type(d) == list else pa.array([d.tolist()]) for d in data.values()]
    #table = pa.Table.from_arrays(pqdata, data.keys())
    #if nWritten == 0:
    #    print "Initialize ParquetWriter at iEvt =",iEvt
    #    print table.schema
    #    writer = pq.ParquetWriter(outStr, table.schema, compression='snappy')

    #writer.write_table(table)
    nWritten += 1

print

df=pd.DataFrame(data)
print "df.dtypes"
print df.dtypes
print "df.shape", df.shape

df.to_hdf(args.outfile, key='df', format='table', mode='w')

#writer.close()

sw.Stop()
print() 
print(" >> nWritten:",nWritten)
print(" >> Real time:",sw.RealTime()/60.,"minutes")
print(" >> CPU time: ",sw.CpuTime() /60.,"minutes")
print(" >> ======================================")

#pqIn = pq.ParquetFile(outStr)
#print(pqIn.metadata)
#print(pqIn.schema)
#X = pqIn.read_row_group(0, columns=['weight','dRjjClose','dRjjOther','aveAbsEta']).to_pydict()
#print(X)
