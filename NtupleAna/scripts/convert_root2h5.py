import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import os, sys
from glob import glob
from copy import copy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root', type=str, help='Input root file.')
parser.add_argument('-o', '--outfile', default='', type=str, help='Output pq file dir. Default is input file name with .root->.h5')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

inStr=args.infile
inFiles = glob(inStr)
print inFiles


for inFile in inFiles:

    tree = ROOT.TChain("Events")
    tree.Add(inFile)

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
    tree.SetBranchStatus("aveAbsEtaOth",1)
    tree.SetBranchStatus("canJet0_pt",1); tree.SetBranchStatus("canJet1_pt",1); tree.SetBranchStatus("canJet2_pt",1); tree.SetBranchStatus("canJet3_pt",1)
    tree.SetBranchStatus("canJet0_eta",1); tree.SetBranchStatus("canJet1_eta",1); tree.SetBranchStatus("canJet2_eta",1); tree.SetBranchStatus("canJet3_eta",1)
    tree.SetBranchStatus("canJet0_phi",1); tree.SetBranchStatus("canJet1_phi",1); tree.SetBranchStatus("canJet2_phi",1); tree.SetBranchStatus("canJet3_phi",1)
    tree.SetBranchStatus("canJet0_e",1); tree.SetBranchStatus("canJet1_e",1); tree.SetBranchStatus("canJet2_e",1); tree.SetBranchStatus("canJet3_e",1)
    tree.SetBranchStatus("nSelJets",1)
    tree.SetBranchStatus("nPSTJets",1)
    tree.SetBranchStatus("st",1)
    tree.SetBranchStatus("m4j",1)
    tree.SetBranchStatus("xWt0",1)
    tree.SetBranchStatus("xWt1",1)
    FvTStatus = False if "nil" in str(tree.FindBranch("FvT")) else True
    if FvTStatus: tree.SetBranchStatus("FvT",1)
    print FvTStatus
    ZHvBStatus = False if "nil" in str(tree.FindBranch("ZHvB")) else True
    if ZHvBStatus: tree.SetBranchStatus("ZHvB",1)
    tree.Show(0)

    nEvts = tree.GetEntries()
    assert nEvts > 0
    print " >> Input file:",inFile
    print " >> nEvts:",nEvts
    outfile = args.outfile if args.outfile else inFile.replace(".root",".h5")
    print " >> Output file:",outfile

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
            'canJet0_m'  : [], 'canJet1_m'  : [], 'canJet2_m'  : [], 'canJet3_m'  : [],
            'm01': [], 'm23': [], 'm02': [], 'm13': [], 'm03': [], 'm12': [], 
            'pt01': [], 'pt23': [], 'pt02': [], 'pt13': [], 'pt03': [], 'pt12': [], 
            'dR01': [], 'dR23': [], 'dR02': [], 'dR13': [], 'dR03': [], 'dR12': [], 
            'nSelJets': [],
            'nPSTJets': [],
            'st': [],
            'm4j': [],
            'xWt0': [],
            'xWt1': [],
            'dR0123': [], 'dR0213': [], 'dR0312': [],
            'dRjjClose': [],
            'dRjjOther': [],
            'aveAbsEta': [],
            'aveAbsEtaOth': [],
            } 
    if FvTStatus: data['FvT'] = []
    if ZHvBStatus: data['ZHvB'] = []

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
        data['canJet0_m'].append(copy(jets[0].M())); data['canJet1_m'].append(copy(jets[1].M())); data['canJet2_m'].append(copy(jets[2].M())); data['canJet3_m'].append(copy(jets[3].M()))
        m01 = (jets[0]+jets[1]).M()
        m23 = (jets[2]+jets[3]).M()
        m02 = (jets[0]+jets[2]).M()
        m13 = (jets[1]+jets[3]).M()
        m03 = (jets[0]+jets[3]).M()
        m12 = (jets[1]+jets[2]).M()
        data['m01'].append(m01)
        data['m23'].append(m23)
        data['m02'].append(m02)
        data['m13'].append(m13)
        data['m03'].append(m03)
        data['m12'].append(m12)

        pt01 = (jets[0]+jets[1]).Pt()
        pt23 = (jets[2]+jets[3]).Pt()
        pt02 = (jets[0]+jets[2]).Pt()
        pt13 = (jets[1]+jets[3]).Pt()
        pt03 = (jets[0]+jets[3]).Pt()
        pt12 = (jets[1]+jets[2]).Pt()
        data['pt01'].append(pt01)
        data['pt23'].append(pt23)
        data['pt02'].append(pt02)
        data['pt13'].append(pt13)
        data['pt03'].append(pt03)
        data['pt12'].append(pt12)

        dR01 = jets[0].DeltaR(jets[1])
        dR23 = jets[2].DeltaR(jets[3])
        dR02 = jets[0].DeltaR(jets[2])
        dR13 = jets[1].DeltaR(jets[3])
        dR03 = jets[0].DeltaR(jets[3])
        dR12 = jets[1].DeltaR(jets[2])
        data['dR01'].append(dR01)
        data['dR23'].append(dR23)
        data['dR02'].append(dR02)
        data['dR13'].append(dR13)
        data['dR03'].append(dR03)
        data['dR12'].append(dR12)
    
        dR0123 = (jets[0]+jets[1]).DeltaR(jets[2]+jets[3])
        dR0213 = (jets[0]+jets[2]).DeltaR(jets[1]+jets[3])
        dR0312 = (jets[0]+jets[3]).DeltaR(jets[1]+jets[2])
        data['dR0123'].append(dR0123)
        data['dR0213'].append(dR0213)
        data['dR0312'].append(dR0312)
    
        data['st'].append(copy(tree.st))
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
        data['nPSTJets'].append(copy(tree.nPSTJets))
        data['dRjjClose'] .append(copy(tree.dRjjClose))
        data['dRjjOther'] .append(copy(tree.dRjjOther))
        data['aveAbsEta'] .append(copy(tree.aveAbsEta))
        data['aveAbsEtaOth'] .append(copy(tree.aveAbsEtaOth))
        if FvTStatus: data['FvT'].append(copy(tree.FvT))
        if ZHvBStatus: data['ZHvB'].append(copy(tree.ZHvB))

        nWritten += 1

    print

    data['st'] = np.array(data['st'], np.float32)
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
    data['canJet0_m'] = np.array(data['canJet0_m'], np.float32); data['canJet1_m'] = np.array(data['canJet1_m'], np.float32); data['canJet2_m'] = np.array(data['canJet2_m'], np.float32); data['canJet3_m'] = np.array(data['canJet3_m'], np.float32)
    data['m01'] = np.array(data['m01'], np.float32)
    data['m23'] = np.array(data['m23'], np.float32)
    data['m02'] = np.array(data['m02'], np.float32)
    data['m13'] = np.array(data['m13'], np.float32)
    data['m03'] = np.array(data['m03'], np.float32)
    data['m12'] = np.array(data['m12'], np.float32)
    data['pt01'] = np.array(data['pt01'], np.float32)
    data['pt23'] = np.array(data['pt23'], np.float32)
    data['pt02'] = np.array(data['pt02'], np.float32)
    data['pt13'] = np.array(data['pt13'], np.float32)
    data['pt03'] = np.array(data['pt03'], np.float32)
    data['pt12'] = np.array(data['pt12'], np.float32)
    data['dR01'] = np.array(data['dR01'], np.float32)
    data['dR23'] = np.array(data['dR23'], np.float32)
    data['dR02'] = np.array(data['dR02'], np.float32)
    data['dR13'] = np.array(data['dR13'], np.float32)
    data['dR03'] = np.array(data['dR03'], np.float32)
    data['dR12'] = np.array(data['dR12'], np.float32)
    data['dR0123'] = np.array(data['dR0123'], np.float32)
    data['dR0213'] = np.array(data['dR0213'], np.float32)
    data['dR0312'] = np.array(data['dR0312'], np.float32)
    data['nSelJets']   = np.array(data['nSelJets'],   np.uint32)
    data['nPSTJets']   = np.array(data['nPSTJets'],   np.uint32)
    data['dRjjClose']  = np.array(data['dRjjClose'],  np.float32)
    data['dRjjOther']  = np.array(data['dRjjOther'],  np.float32)
    data['aveAbsEta']  = np.array(data['aveAbsEta'],  np.float32)
    data['aveAbsEtaOth']  = np.array(data['aveAbsEtaOth'],  np.float32)
    if FvTStatus: data['FvT'] = np.array(data['FvT'], np.float32)
    if ZHvBStatus: data['ZHvB'] = np.array(data['ZHvB'], np.float32)

    df=pd.DataFrame(data)
    print "df.dtypes"
    print df.dtypes
    print "df.shape", df.shape

    df.to_hdf(outfile, key='df', format='table', mode='w')

    sw.Stop()
    print
    print " >> nWritten:",nWritten
    print " >> Real time:",sw.RealTime()/60.,"minutes"
    print " >> CPU time: ",sw.CpuTime() /60.,"minutes"
    print " >> ======================================"
