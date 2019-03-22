import pyarrow.parquet as pq
import pyarrow as pa # pip install pyarrow==0.7.1
import ROOT
import numpy as np
import glob, os, sys

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--infile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root', type=str, help='Input root file.')
parser.add_argument('-o', '--outfile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.parquet', type=str, help='Output pq file dir.')
args = parser.parse_args()

treeStr = args.infile 
tree = ROOT.TChain("Events")
tree.Add(treeStr)
tree.SetBranchStatus("*",0)
tree.SetBranchStatus("ZHSB",1)
tree.SetBranchStatus("weight",1)
tree.SetBranchStatus("dRjjClose",1)
tree.SetBranchStatus("dRjjOther",1)
tree.SetBranchStatus("aveAbsEta",1)
#tree.SetBranchStatus("Jet_pt",1)
#tree.SetBranchStatus("Jet_bRegCorr",1)
tree.Show(0)
nEvts = tree.GetEntries()
assert nEvts > 0
print " >> Input file:",treeStr
print " >> nEvts:",nEvts
outStr = args.outfile
print " >> Output file:",outStr

##### EVENT SELECTION START #####

# Event range to process
iEvtStart = 0
iEvtEnd   = 100
iEvtEnd   = nEvts 
assert iEvtEnd <= nEvts
print " >> Processing entries: [",iEvtStart,"->",iEvtEnd,")"

nWritten = 0
data = {} # Arrays to be written to parquet should be saved to data dict
sw = ROOT.TStopwatch()
sw.Start()
for iEvt in range(iEvtStart,iEvtEnd):

    # Initialize event
    tree.GetEntry(iEvt)
    if iEvt+1 % 1000 == 0 or iEvt+1 == iEvtEnd:
        sys.stdout.write("\rProcessed "+str(iEvt+1)+" of "+str(nEvts)+" | "+str(int((iEvt+1)*100.0/nEvts))+"% ")
        sys.stdout.flush()

    weight = tree.weight
    dRjjClose = tree.dRjjClose
    dRjjOther = tree.dRjjOther
    aveAbsEta = tree.aveAbsEta

    if not tree.ZHSB: continue
    
    data['weight'] = weight
    data['dRjjClose'] = dRjjClose
    data['dRjjOther'] = dRjjOther
    data['aveAbsEta'] = aveAbsEta

    pqdata = [pa.array([d]) if np.isscalar(d) or type(d) == list else pa.array([d.tolist()]) for d in data.values()]
    table = pa.Table.from_arrays(pqdata, data.keys())
    if nWritten == 0:
        writer = pq.ParquetWriter(outStr, table.schema, compression='snappy')

    nWritten += 1

    writer.write_table(table)

writer.close()

sw.Stop()
print 
print " >> nWritten:",nWritten
print " >> Real time:",sw.RealTime()/60.,"minutes"
print " >> CPU time: ",sw.CpuTime() /60.,"minutes"
print " >> ======================================"

pqIn = pq.ParquetFile(outStr)
print(pqIn.metadata)
print(pqIn.schema)
X = pqIn.read_row_group(0, columns=['weight']).to_pydict()
print(X)
