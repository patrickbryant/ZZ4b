# py ZZ4b/nTupleAnalysis/scripts/dumpBranches.py root://cmsxrootd-site.fnal.gov//store/mc/RunIISummer20UL18NanoAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/50000/B0AE8B0B-D19A-3548-AF2A-C275409CCB11.root  *L1* 

import ROOT 
import sys

print "Dumping from ",sys.argv[1]
inFile = ROOT.TFile.Open(sys.argv[1])
inFile.ls()

eventTree = inFile.Get("Events")
if len(sys.argv) > 2:
    eventTree.Print(sys.argv[2])
else:
    eventTree.Print()
