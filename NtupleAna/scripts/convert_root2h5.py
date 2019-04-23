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
tree.SetBranchStatus("ZHSB",1); tree.SetBranchStatus("ZHCR",1), tree.SetBranchStatus("ZHSR",1)
tree.SetBranchStatus("passDEtaBB",1)
tree.SetBranchStatus("passHLT",1)
tree.SetBranchStatus("weight",1)
tree.SetBranchStatus("pseudoTagWeight",1)
tree.SetBranchStatus("fourTag",1)
tree.SetBranchStatus("dRjjClose",1)
tree.SetBranchStatus("dRjjOther",1)
tree.SetBranchStatus("aveAbsEta",1)
tree.SetBranchStatus("canJet0_pt",1); tree.SetBranchStatus("canJet1_pt",1); tree.SetBranchStatus("canJet2_pt",1); tree.SetBranchStatus("canJet3_pt",1)
tree.SetBranchStatus("canJet0_eta",1); tree.SetBranchStatus("canJet1_eta",1); tree.SetBranchStatus("canJet2_eta",1); tree.SetBranchStatus("canJet3_eta",1)
tree.SetBranchStatus("canJet0_phi",1); tree.SetBranchStatus("canJet1_phi",1); tree.SetBranchStatus("canJet2_phi",1); tree.SetBranchStatus("canJet3_phi",1)
tree.SetBranchStatus("canJet0_e",1); tree.SetBranchStatus("canJet1_e",1); tree.SetBranchStatus("canJet2_e",1); tree.SetBranchStatus("canJet3_e",1)
tree.SetBranchStatus("nSelJets",1)
tree.SetBranchStatus("m4j",1)
tree.SetBranchStatus("xWt0",1)
tree.SetBranchStatus("xWt1",1)
nTagClassifierStatus = False if "nil" in str(tree.FindBranch("nTagClassifier")) else True
if nTagClassifierStatus: tree.SetBranchStatus("nTagClassifier",1)
print nTagClassifierStatus
ZHvsBackgroundClassifierStatus = False if "nil" in str(tree.FindBranch("ZHvsBackgroundClassifier")) else True
if ZHvsBackgroundClassifierStatus: tree.SetBranchStatus("ZHvsBackgroundClassifier",1)
tree.Show(0)

nEvts = tree.GetEntries()
assert nEvts > 0
print " >> Input file:",treeStr
print " >> nEvts:",nEvts
outStr = args.outfile
print " >> Output file:",outStr

rotate=True

##### Start Conversion #####

# Event range to process
iEvtStart = 0
iEvtEnd   = 1000
iEvtEnd   = nEvts 
assert iEvtEnd <= nEvts
print " >> Processing entries: [",iEvtStart,"->",iEvtEnd,")"

nWritten = 0
data = {'weight': [],
        'pseudoTagWeight': [],
        'ZHSB': [], 'ZHCR': [], 'ZHSR': [],
        'passDEtaBB': [],
        'passHLT': [],
        'fourTag': [],
        'canJet0_pt' : [], 'canJet1_pt' : [], 'canJet2_pt' : [], 'canJet3_pt' : [],
        'canJet0_eta': [], 'canJet1_eta': [], 'canJet2_eta': [], 'canJet3_eta': [],
        'canJet0_phi': [], 'canJet1_phi': [], 'canJet2_phi': [], 'canJet3_phi': [],
        'canJet0_e'  : [], 'canJet1_e'  : [], 'canJet2_e'  : [], 'canJet3_e'  : [],
        'd01': [], 'd23': [], 'd02': [], 'd13': [], 'd03': [], 'd12': [], 
        'nSelJets': [],
        'm4j': [],
        'xWt0': [],
        'xWt1': [],
        'dRjjClose': [],
        'dRjjOther': [],
        'aveAbsEta': [],
        } 
if nTagClassifierStatus: data['nTagClassifier'] = []
if ZHvsBackgroundClassifierStatus: data['ZHvsBackgroundClassifier'] = []
#df = pd.DataFrame({''})
sw = ROOT.TStopwatch()
sw.Start()
for iEvt in list(range(iEvtStart,iEvtEnd)):

    # Initialize event
    tree.GetEntry(iEvt)
    if (iEvt+1) % 1000 == 0 or iEvt+1 == iEvtEnd:
        sys.stdout.write("\rProcessed "+str(iEvt+1)+" of "+str(nEvts)+" | "+str(int((iEvt+1)*100.0/nEvts))+"% ")
        sys.stdout.flush()


    data['canJet0_pt'].append(copy(tree.canJet0_pt)); data['canJet1_pt'].append(copy(tree.canJet1_pt)); data['canJet2_pt'].append(copy(tree.canJet2_pt)); data['canJet3_pt'].append(copy(tree.canJet3_pt))
    data['canJet0_eta'].append(copy(tree.canJet0_eta)); data['canJet1_eta'].append(copy(tree.canJet1_eta)); data['canJet2_eta'].append(copy(tree.canJet2_eta)); data['canJet3_eta'].append(copy(tree.canJet3_eta))
    data['canJet0_phi'].append(copy(tree.canJet0_phi)); data['canJet1_phi'].append(copy(tree.canJet1_phi)); data['canJet2_phi'].append(copy(tree.canJet2_phi)); data['canJet3_phi'].append(copy(tree.canJet3_phi))
    data['canJet0_e'].append(copy(tree.canJet0_e)); data['canJet1_e'].append(copy(tree.canJet1_e)); data['canJet2_e'].append(copy(tree.canJet2_e)); data['canJet3_e'].append(copy(tree.canJet3_e))

    jets = [ROOT.TLorentzVector(),ROOT.TLorentzVector(),ROOT.TLorentzVector(),ROOT.TLorentzVector()]
    jets[0].SetPtEtaPhiE(tree.canJet0_pt, tree.canJet0_eta, tree.canJet0_phi, tree.canJet0_e)
    jets[1].SetPtEtaPhiE(tree.canJet1_pt, tree.canJet1_eta, tree.canJet1_phi, tree.canJet1_e)
    jets[2].SetPtEtaPhiE(tree.canJet2_pt, tree.canJet2_eta, tree.canJet2_phi, tree.canJet2_e)
    jets[3].SetPtEtaPhiE(tree.canJet3_pt, tree.canJet3_eta, tree.canJet3_phi, tree.canJet3_e)
    d01 = (jets[0]+jets[1]).M()
    d23 = (jets[2]+jets[3]).M()
    d02 = (jets[0]+jets[2]).M()
    d13 = (jets[1]+jets[3]).M()
    d03 = (jets[0]+jets[3]).M()
    d12 = (jets[1]+jets[2]).M()
    data['d01'].append(d01)
    data['d23'].append(d23)
    data['d02'].append(d02)
    data['d13'].append(d13)
    data['d03'].append(d03)
    data['d12'].append(d12)
    
    
    data['m4j'].append(copy(tree.m4j))
    data['xWt0'].append(copy(tree.xWt0))
    data['xWt1'].append(copy(tree.xWt1))
    data['weight']    .append(copy(tree.weight))
    data['pseudoTagWeight']    .append(copy(tree.pseudoTagWeight))
    data['passHLT'].append(copy(tree.passHLT))
    data['ZHSB'].append(copy(tree.ZHSB)); data['ZHCR'].append(copy(tree.ZHCR)); data['ZHSR'].append(copy(tree.ZHSR))
    data['passDEtaBB'].append(copy(tree.passDEtaBB))
    data['fourTag']   .append(copy(tree.fourTag))
    data['nSelJets'].append(copy(tree.nSelJets))
    data['dRjjClose'] .append(copy(tree.dRjjClose))
    data['dRjjOther'] .append(copy(tree.dRjjOther))
    data['aveAbsEta'] .append(copy(tree.aveAbsEta))
    if nTagClassifierStatus: data['nTagClassifier'].append(copy(tree.nTagClassifier))
    if ZHvsBackgroundClassifierStatus: data['ZHvsBackgroundClassifier'].append(copy(tree.ZHvsBackgroundClassifier))

    nWritten += 1

print

data['m4j'] = np.array(data['m4j'], np.float32)
data['xWt0'] = np.array(data['xWt0'], np.float32)
data['xWt1'] = np.array(data['xWt1'], np.float32)
data['weight']     = np.array(data['weight'],     np.float32)
data['pseudoTagWeight']     = np.array(data['pseudoTagWeight'],     np.float32)
data['ZHSB'] = np.array(data['ZHSB'], np.bool_); data['ZHCR'] = np.array(data['ZHCR'], np.bool_); data['ZHSR'] = np.array(data['ZHSR'], np.bool_)
data['passHLT'] = np.array(data['passHLT'], np.bool_)
data['passDEtaBB'] = np.array(data['passDEtaBB'], np.bool_)
data['fourTag']    = np.array(data['fourTag'],    np.bool_)
data['canJet0_pt'] = np.array(data['canJet0_pt'], np.float32); data['canJet1_pt'] = np.array(data['canJet1_pt'], np.float32); data['canJet2_pt'] = np.array(data['canJet2_pt'], np.float32); data['canJet3_pt'] = np.array(data['canJet3_pt'], np.float32)
data['canJet0_eta'] = np.array(data['canJet0_eta'], np.float32); data['canJet1_eta'] = np.array(data['canJet1_eta'], np.float32); data['canJet2_eta'] = np.array(data['canJet2_eta'], np.float32); data['canJet3_eta'] = np.array(data['canJet3_eta'], np.float32)
data['canJet0_phi'] = np.array(data['canJet0_phi'], np.float32); data['canJet1_phi'] = np.array(data['canJet1_phi'], np.float32); data['canJet2_phi'] = np.array(data['canJet2_phi'], np.float32); data['canJet3_phi'] = np.array(data['canJet3_phi'], np.float32)
data['canJet0_e'] = np.array(data['canJet0_e'], np.float32); data['canJet1_e'] = np.array(data['canJet1_e'], np.float32); data['canJet2_e'] = np.array(data['canJet2_e'], np.float32); data['canJet3_e'] = np.array(data['canJet3_e'], np.float32)
data['d01'] = np.array(data['d01'], np.float32)
data['d23'] = np.array(data['d23'], np.float32)
data['d02'] = np.array(data['d02'], np.float32)
data['d13'] = np.array(data['d13'], np.float32)
data['d03'] = np.array(data['d03'], np.float32)
data['d12'] = np.array(data['d12'], np.float32)
data['nSelJets']   = np.array(data['nSelJets'],   np.uint32)
data['dRjjClose']  = np.array(data['dRjjClose'],  np.float32)
data['dRjjOther']  = np.array(data['dRjjOther'],  np.float32)
data['aveAbsEta']  = np.array(data['aveAbsEta'],  np.float32)
if nTagClassifierStatus: data['nTagClassifier'] = np.array(data['nTagClassifier'], np.float32)
if ZHvsBackgroundClassifierStatus: data['ZHvsBackgroundClassifier'] = np.array(data['ZHvsBackgroundClassifier'], np.float32)

df=pd.DataFrame(data)
print "df.dtypes"
print df.dtypes
print "df.shape", df.shape

df.to_hdf(args.outfile, key='df', format='table', mode='w')

sw.Stop()
print
print " >> nWritten:",nWritten
print " >> Real time:",sw.RealTime()/60.,"minutes"
print " >> CPU time: ",sw.CpuTime() /60.,"minutes"
print " >> ======================================"
