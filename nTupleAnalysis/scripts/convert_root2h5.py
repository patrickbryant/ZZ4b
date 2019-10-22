import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import os, sys
from glob import glob
from copy import copy
import multiprocessing
mZ, mH = 91.0, 125.0
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inFile', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root', type=str, help='Input root file.')
parser.add_argument('-o', '--outfile', default='', type=str, help='Output pq file dir. Default is input file name with .root->.h5')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

inPaths = args.inFile.split()
inFiles = []
for path in inPaths:
    inFiles += glob(path)
print inFiles

# inStr=args.infile
# inFiles = glob(inStr)
# print inFiles

class variable:
    def __init__(self, name, dtype=np.float32):
        self.name = name
        self.status = False
        self.dtype = dtype

    def setStatus(self,tree):
        self.status = 0 if 'nil' in str(tree.FindBranch(self.name)) else 1
        print self.name, self.status
        if self.status: tree.SetBranchStatus(self.name,self.status)

variables = [variable("ZHvB"),
             variable("ZZvB"),
             variable("FvT"),
             variable("FvT_pd4"),
             variable("FvT_pd3"),
             variable("FvT_pt4"),
             variable("FvT_pt3"),
             variable("FvT_pm4"),
             variable("FvT_pm3"),
             variable("HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", dtype=np.bool_),
             variable("nIsoMuons",dtype=np.uint32)
             ]

def convert(inFile):

    tree = ROOT.TChain("Events")
    tree.Add(inFile)

    # Initialize TTree
    tree.SetBranchStatus("*",0)
    tree.SetBranchStatus("ZHSB",1); tree.SetBranchStatus("ZHCR",1), tree.SetBranchStatus("ZHSR",1)
    tree.SetBranchStatus("ZZSB",1); tree.SetBranchStatus("ZZCR",1), tree.SetBranchStatus("ZZSR",1)
    tree.SetBranchStatus("SB",1); tree.SetBranchStatus("CR",1), tree.SetBranchStatus("SR",1)
    tree.SetBranchStatus("passDEtaBB",1)
    tree.SetBranchStatus("passHLT",1)
    tree.SetBranchStatus("weight",1)
    tree.SetBranchStatus("nPVsGood",1)
    tree.SetBranchStatus("pseudoTagWeight",1)
    mcPseudoTagWeight_Status = False if 'nil' in str(tree.FindBranch("mcPseudoTagWeight")) else True
    if mcPseudoTagWeight_Status: tree.SetBranchStatus("mcPseudoTagWeight",1)
    tree.SetBranchStatus("fourTag",1)
    tree.SetBranchStatus("dRjjClose",1)
    tree.SetBranchStatus("dRjjOther",1)
    tree.SetBranchStatus("aveAbsEta",1)
    tree.SetBranchStatus("aveAbsEtaOth",1)
    tree.SetBranchStatus("canJet0_pt",1); tree.SetBranchStatus("canJet1_pt",1); tree.SetBranchStatus("canJet2_pt",1); tree.SetBranchStatus("canJet3_pt",1)
    tree.SetBranchStatus("canJet0_eta",1); tree.SetBranchStatus("canJet1_eta",1); tree.SetBranchStatus("canJet2_eta",1); tree.SetBranchStatus("canJet3_eta",1)
    tree.SetBranchStatus("canJet0_phi",1); tree.SetBranchStatus("canJet1_phi",1); tree.SetBranchStatus("canJet2_phi",1); tree.SetBranchStatus("canJet3_phi",1)
    canJet_m_Status = False if "nil" in str(tree.FindBranch("canJet0_m")) else True
    if canJet_m_Status:
        tree.SetBranchStatus("canJet0_m",1); tree.SetBranchStatus("canJet1_m",1); tree.SetBranchStatus("canJet2_m",1); tree.SetBranchStatus("canJet3_m",1)
    else:
        tree.SetBranchStatus("canJet0_e",1); tree.SetBranchStatus("canJet1_e",1); tree.SetBranchStatus("canJet2_e",1); tree.SetBranchStatus("canJet3_e",1)
    tree.SetBranchStatus("nSelJets",1)
    tree.SetBranchStatus("nPSTJets",1)
    tree.SetBranchStatus("st",1)
    tree.SetBranchStatus("stNotCan",1)
    tree.SetBranchStatus("m4j",1)
    tree.SetBranchStatus("xWt0",1)
    tree.SetBranchStatus("xWt1",1)
    xWt_Status = False if 'nil' in str(tree.FindBranch("xWt")) else True
    if xWt_Status: tree.SetBranchStatus("xWt" ,1)
    tree.SetBranchStatus("nAllNotCanJets",1)
    tree.SetBranchStatus("notCanJet_pt",1)
    tree.SetBranchStatus("notCanJet_eta",1)
    tree.SetBranchStatus("notCanJet_phi",1)
    tree.SetBranchStatus("notCanJet_m",1)

    for var in variables:
        var.setStatus(tree)

    #tree.Show(0)

    nEvts = tree.GetEntries()
    assert nEvts > 0
    print " >> Input file:",inFile
    print " >> nEvts:",nEvts
    outfile = args.outfile if args.outfile else inFile.replace(".root",".h5")
    print " >> Output file:",outfile
    store = pd.HDFStore(outfile,mode='w')
    #store.close()

    ##### Start Conversion #####

    # Event range to process
    iEvtStart = 0
    iEvtEnd   = 10000
    iEvtEnd   = nEvts 
    chunkSize = 10000
    assert iEvtEnd <= nEvts
    print " >> Processing entries: [",iEvtStart,"->",iEvtEnd,")"

    sw = ROOT.TStopwatch()
    sw.Start()

    nWritten = 0
    for iEvtStart in range(0,iEvtEnd,chunkSize):
        data = {'weight': [],
                'nPVsGood': [],
                'pseudoTagWeight': [],
                'mcPseudoTagWeight': [],
                'ZHSB': [], 'ZHCR': [], 'ZHSR': [],
                'ZZSB': [], 'ZZCR': [], 'ZZSR': [],
                'SB': [], 'CR': [], 'SR': [],
                'passDEtaBB': [],
                'passHLT': [],
                'fourTag': [],
                'canJet0_pt' : [], 'canJet1_pt' : [], 'canJet2_pt' : [], 'canJet3_pt' : [],
                'canJet0_eta': [], 'canJet1_eta': [], 'canJet2_eta': [], 'canJet3_eta': [],
                'canJet0_phi': [], 'canJet1_phi': [], 'canJet2_phi': [], 'canJet3_phi': [],
                'canJet0_m'  : [], 'canJet1_m'  : [], 'canJet2_m'  : [], 'canJet3_m'  : [],
                #'canJet0_e'  : [], 'canJet1_e'  : [], 'canJet2_e'  : [], 'canJet3_e'  : [],
                'm01': [], 'm23': [], 'm02': [], 'm13': [], 'm03': [], 'm12': [], 
                'm123': [], 'm023': [], 'm013': [], 'm012': [],
                'pt01': [], 'pt23': [], 'pt02': [], 'pt13': [], 'pt03': [], 'pt12': [], 
                'dR01': [], 'dR23': [], 'dR02': [], 'dR13': [], 'dR03': [], 'dR12': [], 
                'nSelJets': [],
                'nPSTJets': [],
                'st': [],
                'stNotCan': [],
                's4j': [],
                'm4j': [],
                'xWt0': [],
                'xWt1': [],
                'xWt' : [],
                'dR0123': [], 'dR0213': [], 'dR0312': [],
                'mZH0123': [], 'mZH0213': [], 'mZH0312': [],
                'mZZ0123': [], 'mZZ0213': [], 'mZZ0312': [],
                'dRjjClose': [],
                'dRjjOther': [],
                'aveAbsEta': [],
                'aveAbsEtaOth': [],
                } 
        nOthJetsMax = 12
        for i in range(nOthJetsMax):
            data['notCanJet'+str(i)+'_pt'] = []
            data['notCanJet'+str(i)+'_eta'] = []
            data['notCanJet'+str(i)+'_phi'] = []
            data['notCanJet'+str(i)+'_m'] = []
            data['notCanJet'+str(i)+'_isSelJet'] = []

        for var in variables:
            if var.status: data[var.name] = []

        for iEvt in range(iEvtStart, min(iEvtStart+chunkSize, iEvtEnd)):

            # Initialize event
            tree.GetEntry(iEvt)
            if (iEvt+1) % 1000 == 0 or iEvt+1 == iEvtEnd:
                sys.stdout.write("\rProcessed "+str(iEvt+1)+" of "+str(nEvts)+" | "+str(int((iEvt+1)*100.0/nEvts))+"% ")
                sys.stdout.flush()


            jets = [ROOT.TLorentzVector(),ROOT.TLorentzVector(),ROOT.TLorentzVector(),ROOT.TLorentzVector()]

            data['canJet0_pt'].append(copy(tree.canJet0_pt)); data['canJet1_pt'].append(copy(tree.canJet1_pt)); data['canJet2_pt'].append(copy(tree.canJet2_pt)); data['canJet3_pt'].append(copy(tree.canJet3_pt))
            data['canJet0_eta'].append(copy(tree.canJet0_eta)); data['canJet1_eta'].append(copy(tree.canJet1_eta)); data['canJet2_eta'].append(copy(tree.canJet2_eta)); data['canJet3_eta'].append(copy(tree.canJet3_eta))
            data['canJet0_phi'].append(copy(tree.canJet0_phi)); data['canJet1_phi'].append(copy(tree.canJet1_phi)); data['canJet2_phi'].append(copy(tree.canJet2_phi)); data['canJet3_phi'].append(copy(tree.canJet3_phi))
            if canJet_m_Status:
                data['canJet0_m'].append(copy(tree.canJet0_m)); data['canJet1_m'].append(copy(tree.canJet1_m)); data['canJet2_m'].append(copy(tree.canJet2_m)); data['canJet3_m'].append(copy(tree.canJet3_m))
                jets[0].SetPtEtaPhiM(tree.canJet0_pt, tree.canJet0_eta, tree.canJet0_phi, tree.canJet0_m)
                jets[1].SetPtEtaPhiM(tree.canJet1_pt, tree.canJet1_eta, tree.canJet1_phi, tree.canJet1_m)
                jets[2].SetPtEtaPhiM(tree.canJet2_pt, tree.canJet2_eta, tree.canJet2_phi, tree.canJet2_m)
                jets[3].SetPtEtaPhiM(tree.canJet3_pt, tree.canJet3_eta, tree.canJet3_phi, tree.canJet3_m)

            else:
                # data['canJet0_e'].append(copy(tree.canJet0_e)); data['canJet1_e'].append(copy(tree.canJet1_e)); data['canJet2_e'].append(copy(tree.canJet2_e)); data['canJet3_e'].append(copy(tree.canJet3_e))
                jets[0].SetPtEtaPhiE(tree.canJet0_pt, tree.canJet0_eta, tree.canJet0_phi, tree.canJet0_e)
                jets[1].SetPtEtaPhiE(tree.canJet1_pt, tree.canJet1_eta, tree.canJet1_phi, tree.canJet1_e)
                jets[2].SetPtEtaPhiE(tree.canJet2_pt, tree.canJet2_eta, tree.canJet2_phi, tree.canJet2_e)
                jets[3].SetPtEtaPhiE(tree.canJet3_pt, tree.canJet3_eta, tree.canJet3_phi, tree.canJet3_e)
                data['canJet0_m'].append(jets[0].M()); data['canJet1_m'].append(jets[1].M()); data['canJet2_m'].append(jets[2].M()); data['canJet3_m'].append(jets[3].M())



            d01, d23 = jets[0]+jets[1], jets[2]+jets[3]
            d02, d13 = jets[0]+jets[2], jets[1]+jets[3]
            d03, d12 = jets[0]+jets[3], jets[1]+jets[2]

            m01, m23 = d01.M(), d23.M()
            m02, m13 = d02.M(), d13.M()
            m03, m12 = d03.M(), d12.M()
            data['m01'].append(m01)
            data['m23'].append(m23)
            data['m02'].append(m02)
            data['m13'].append(m13)
            data['m03'].append(m03)
            data['m12'].append(m12)

            m123 = (jets[1]+jets[2]+jets[3]).M() #missing 0
            m023 = (jets[0]+jets[2]+jets[3]).M() #missing 1
            m013 = (jets[0]+jets[1]+jets[3]).M() #missing 2
            m012 = (jets[0]+jets[1]+jets[2]).M() #missing 3
            data['m123'].append(m123)
            data['m023'].append(m023)
            data['m013'].append(m013)
            data['m012'].append(m012)

            pt01, pt23 = d01.Pt(), d23.Pt()
            pt02, pt13 = d02.Pt(), d13.Pt()
            pt03, pt12 = d03.Pt(), d12.Pt()
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

            dR0123 = d01.DeltaR(d23)
            dR0213 = d02.DeltaR(d13)
            dR0312 = d03.DeltaR(d12)
            data['dR0123'].append(dR0123)
            data['dR0213'].append(dR0213)
            data['dR0312'].append(dR0312)

            ds0123 = [d01, d23] if m01 > m23 else [d23, d01]
            ds0213 = [d02, d13] if m02 > m13 else [d13, d02]
            ds0312 = [d03, d12] if m03 > m12 else [d12, d03]
            mZH0123 = (ds0123[0]*(mH/ds0123[0].M()) + ds0123[1]*(mZ/ds0123[1].M())).M()
            mZH0213 = (ds0213[0]*(mH/ds0213[0].M()) + ds0213[1]*(mZ/ds0213[1].M())).M()
            mZH0312 = (ds0312[0]*(mH/ds0312[0].M()) + ds0312[1]*(mZ/ds0312[1].M())).M()
            data['mZH0123'].append(mZH0123)
            data['mZH0213'].append(mZH0213)
            data['mZH0312'].append(mZH0312)

            mZZ0123 = (ds0123[0]*(mZ/ds0123[0].M()) + ds0123[1]*(mZ/ds0123[1].M())).M()
            mZZ0213 = (ds0213[0]*(mZ/ds0213[0].M()) + ds0213[1]*(mZ/ds0213[1].M())).M()
            mZZ0312 = (ds0312[0]*(mZ/ds0312[0].M()) + ds0312[1]*(mZ/ds0312[1].M())).M()
            data['mZZ0123'].append(mZZ0123)
            data['mZZ0213'].append(mZZ0213)
            data['mZZ0312'].append(mZZ0312)

            data['st'].append(copy(tree.st))
            data['stNotCan'].append(copy(tree.stNotCan))
            data['s4j'].append(tree.canJet0_pt + tree.canJet1_pt + tree.canJet2_pt + tree.canJet3_pt)
            data['m4j'].append(copy(tree.m4j))
            data['xWt0'].append(copy(tree.xWt0))
            data['xWt1'].append(copy(tree.xWt1))
            if xWt_Status: data['xWt'].append(copy(tree.xWt))
            else: data['xWt'].append(0)
            data['weight']    .append(copy(tree.weight))
            data['nPVsGood']    .append(copy(tree.nPVsGood))
            data['pseudoTagWeight']    .append(copy(tree.pseudoTagWeight))
            if mcPseudoTagWeight_Status: data['mcPseudoTagWeight']    .append(copy(tree.mcPseudoTagWeight))
            else: data['mcPseudoTagWeight']    .append(1)
            # if not tree.fourTag:
            #     if abs(tree.mcPseudoTagWeight - tree.weight*((1-tree.FvT)/tree.FvT))>1e-5: print inFile, "ERROR:",tree.fourTag,tree.mcPseudoTagWeight,"!=",tree.weight*(1-tree.FvT)/tree.FvT
            # else:
            #     if abs(tree.mcPseudoTagWeight - tree.weight)>1e-5: print inFile, "ERROR:",tree.fourTag,tree.mcPseudoTagWeight,"!=",tree.weight
            data['passHLT'].append(copy(tree.passHLT))
            data['ZHSB'].append(copy(tree.ZHSB)); data['ZHCR'].append(copy(tree.ZHCR)); data['ZHSR'].append(copy(tree.ZHSR))
            data['ZZSB'].append(copy(tree.ZZSB)); data['ZZCR'].append(copy(tree.ZZCR)); data['ZZSR'].append(copy(tree.ZZSR))
            data['SB'].append(copy(tree.SB)); data['CR'].append(copy(tree.CR)); data['SR'].append(copy(tree.SR))
            data['passDEtaBB'].append(copy(tree.passDEtaBB))
            data['fourTag']   .append(copy(tree.fourTag))
            data['nSelJets'].append(copy(tree.nSelJets))
            data['nPSTJets'].append(copy(tree.nPSTJets))
            data['dRjjClose'] .append(copy(tree.dRjjClose))
            data['dRjjOther'] .append(copy(tree.dRjjOther))
            data['aveAbsEta'] .append(copy(tree.aveAbsEta))
            data['aveAbsEtaOth'] .append(copy(tree.aveAbsEtaOth))

            for i in range(nOthJetsMax):
                if i < tree.nAllNotCanJets:
                    data['notCanJet'+str(i)+'_pt'].append(copy(tree.notCanJet_pt[i]))
                    data['notCanJet'+str(i)+'_eta'].append(copy(tree.notCanJet_eta[i]))
                    data['notCanJet'+str(i)+'_phi'].append(copy(tree.notCanJet_phi[i]))
                    data['notCanJet'+str(i)+'_m'].append(copy(tree.notCanJet_m[i]))
                    data['notCanJet'+str(i)+'_isSelJet'].append(1 if tree.notCanJet_pt[i]>40 and abs(tree.notCanJet_eta[i])<2.4 else 0)
                    if abs(tree.notCanJet_eta[i])>2.4 and tree.notCanJet_pt[i]<40: print("ERROR: This notCanJet should have failed forward pileup veto",tree.notCanJet_eta[i],tree.notCanJet_pt[i])
                else:
                    data['notCanJet'+str(i)+'_pt'].append(0)
                    data['notCanJet'+str(i)+'_eta'].append(0)
                    data['notCanJet'+str(i)+'_phi'].append(0)
                    data['notCanJet'+str(i)+'_m'].append(0)
                    data['notCanJet'+str(i)+'_isSelJet'].append(-1)

            for var in variables:
                if var.status: data[var.name].append(copy(getattr(tree, var.name)))

            nWritten += 1

        #print

        data['st'] = np.array(data['st'], np.float32)
        data['stNotCan'] = np.array(data['stNotCan'], np.float32)
        data['s4j'] = np.array(data['s4j'], np.float32)
        data['m4j'] = np.array(data['m4j'], np.float32)
        data['xWt0'] = np.array(data['xWt0'], np.float32)
        data['xWt1'] = np.array(data['xWt1'], np.float32)
        data['xWt']  = np.array(data['xWt'], np.float32)
        data['weight']     = np.array(data['weight'],     np.float32)
        data['nPVsGood']     = np.array(data['nPVsGood'],     np.float32)
        data['pseudoTagWeight']     = np.array(data['pseudoTagWeight'],     np.float32)
        data['mcPseudoTagWeight']     = np.array(data['mcPseudoTagWeight'],     np.float32)
        data['ZHSB'] = np.array(data['ZHSB'], np.bool_); data['ZHCR'] = np.array(data['ZHCR'], np.bool_); data['ZHSR'] = np.array(data['ZHSR'], np.bool_)
        data['ZZSB'] = np.array(data['ZZSB'], np.bool_); data['ZZCR'] = np.array(data['ZZCR'], np.bool_); data['ZZSR'] = np.array(data['ZZSR'], np.bool_)
        data['SB'] = np.array(data['SB'], np.bool_); data['CR'] = np.array(data['CR'], np.bool_); data['SR'] = np.array(data['SR'], np.bool_)
        data['passHLT'] = np.array(data['passHLT'], np.bool_)
        data['passDEtaBB'] = np.array(data['passDEtaBB'], np.bool_)
        data['fourTag']    = np.array(data['fourTag'],    np.bool_)
        data['canJet0_pt'] = np.array(data['canJet0_pt'], np.float32); data['canJet1_pt'] = np.array(data['canJet1_pt'], np.float32); data['canJet2_pt'] = np.array(data['canJet2_pt'], np.float32); data['canJet3_pt'] = np.array(data['canJet3_pt'], np.float32)
        data['canJet0_eta'] = np.array(data['canJet0_eta'], np.float32); data['canJet1_eta'] = np.array(data['canJet1_eta'], np.float32); data['canJet2_eta'] = np.array(data['canJet2_eta'], np.float32); data['canJet3_eta'] = np.array(data['canJet3_eta'], np.float32)
        data['canJet0_phi'] = np.array(data['canJet0_phi'], np.float32); data['canJet1_phi'] = np.array(data['canJet1_phi'], np.float32); data['canJet2_phi'] = np.array(data['canJet2_phi'], np.float32); data['canJet3_phi'] = np.array(data['canJet3_phi'], np.float32)
        data['canJet0_m'] = np.array(data['canJet0_m'], np.float32); data['canJet1_m'] = np.array(data['canJet1_m'], np.float32); data['canJet2_m'] = np.array(data['canJet2_m'], np.float32); data['canJet3_m'] = np.array(data['canJet3_m'], np.float32)
        data['m01'] = np.array(data['m01'], np.float32)
        data['m23'] = np.array(data['m23'], np.float32)
        data['m02'] = np.array(data['m02'], np.float32)
        data['m13'] = np.array(data['m13'], np.float32)
        data['m03'] = np.array(data['m03'], np.float32)
        data['m12'] = np.array(data['m12'], np.float32)
        data['m123'] = np.array(data['m123'], np.float32)
        data['m023'] = np.array(data['m023'], np.float32)
        data['m013'] = np.array(data['m013'], np.float32)
        data['m012'] = np.array(data['m012'], np.float32)
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
        data['mZH0123'] = np.array(data['mZH0123'], np.float32)
        data['mZH0213'] = np.array(data['mZH0213'], np.float32)
        data['mZH0312'] = np.array(data['mZH0312'], np.float32)
        data['mZZ0123'] = np.array(data['mZZ0123'], np.float32)
        data['mZZ0213'] = np.array(data['mZZ0213'], np.float32)
        data['mZZ0312'] = np.array(data['mZZ0312'], np.float32)
        data['nSelJets']   = np.array(data['nSelJets'],   np.uint32)
        data['nPSTJets']   = np.array(data['nPSTJets'],   np.uint32)
        data['dRjjClose']  = np.array(data['dRjjClose'],  np.float32)
        data['dRjjOther']  = np.array(data['dRjjOther'],  np.float32)
        data['aveAbsEta']  = np.array(data['aveAbsEta'],  np.float32)
        data['aveAbsEtaOth']  = np.array(data['aveAbsEtaOth'],  np.float32)

        for i in range(nOthJetsMax):
            data['notCanJet'+str(i)+'_pt'] = np.array(data['notCanJet'+str(i)+'_pt'], np.float32)
            data['notCanJet'+str(i)+'_eta'] = np.array(data['notCanJet'+str(i)+'_eta'], np.float32)
            data['notCanJet'+str(i)+'_phi'] = np.array(data['notCanJet'+str(i)+'_phi'], np.float32)
            data['notCanJet'+str(i)+'_m'] = np.array(data['notCanJet'+str(i)+'_m'], np.float32)
            data['notCanJet'+str(i)+'_isSelJet'] = np.array(data['notCanJet'+str(i)+'_isSelJet'], np.float32)

        for var in variables:
            if var.status: data[var.name] = np.array(data[var.name], var.dtype)

        #print "writing chunk"
        #store = pd.HDFStore(outfile,mode='a')
        df=pd.DataFrame(data)
        #print "df.shape",df.shape
        store.append('df', df, format='table', data_columns=None, index=False)
        #print "wrote chunk",store['df'].shape
        #print "store['df'].shape",store['df'].shape
        #store.close()
        del df
        del data
        #print "df.dtypes"
        #print df.dtypes
        #print "df.shape", df.shape

    #df.to_hdf(outfile, key='df', format='table', mode='w', data_columns=True)
    store.close()
    sw.Stop()
    print
    print " >> nWritten:",nWritten
    print " >> Real time:",sw.RealTime()/60.,"minutes"
    print " >> CPU time: ",sw.CpuTime() /60.,"minutes"
    print " >> ======================================"



workers = multiprocessing.Pool(6)
for output in workers.imap_unordered(convert,inFiles):
    print output
