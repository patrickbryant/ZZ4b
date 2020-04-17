import ROOT
from array import array

class toyTree:
    def __init__(self, name, debug = False):
        self.name = name
        self.debug = debug
        print("Create Toy Tree:",name+"_toyTree.root")
        self.f = ROOT.TFile(name+"_toyTree.root","RECREATE")
        self.t = ROOT.TTree("Tree",name)

        #Jet 4-vectors
        maxJets = 20
        self.n   = array('i',         [0])
        self.pt  = array('f', maxJets*[0])
        self.eta = array('f', maxJets*[0])
        self.phi = array('f', maxJets*[0])
        self.e   = array('f', maxJets*[0])

        self.t.Branch('nJets',     self.n,   'nJets/I')
        self.t.Branch('jetPt',     self.pt,  'jetPt[nJets]/F')
        self.t.Branch('jetEta',    self.eta, 'jetEta[nJets]/F')
        self.t.Branch('jetPhi',    self.phi, 'jetPhi[nJets]/F')
        self.t.Branch('jetEnergy', self.e,   'jetEnergy[nJets]/F')

        #High Level Variables
        self.dRjjClose = array('f', [0])
        self.dRjjOther = array('f', [0])
        self.aveAbsEta = array('f', [0])
        self.t.Branch('dRjjClose', self.dRjjClose, 'dRjjClose/F')
        self.t.Branch('dRjjOther', self.dRjjOther, 'dRjjOther/F')
        self.t.Branch('aveAbsEta', self.aveAbsEta, 'aveAbsEta/F')
        self.m4j = array('f', [0])
        self.mHH = array('f', [0])
        self.t.Branch('m4j', self.m4j, 'm4j/F')
        self.t.Branch('mHH', self.mHH, 'mHH/F')

        #Region
        self.SB = array('i', [0])
        self.CR = array('i', [0])
        self.SR = array('i', [0])

        self.t.Branch('SB', self.SB, 'SB/I')
        self.t.Branch('CR', self.CR, 'CR/I')
        self.t.Branch('SR', self.SR, 'SR/I')

        #Weight
        self.weight = array('f', [1])
        self.t.Branch('weight', self.weight, 'weight/F')

    def Fill(self, event):
        self.n[0] = len(event.recoJets)
        for i in list(range(self.n[0])):
            self.pt [i] = event.recoJets[i].pt
            self.eta[i] = event.recoJets[i].eta
            self.phi[i] = event.recoJets[i].phi
            self.e  [i] = event.recoJets[i].e

        self.dRjjClose[0] = event.leadGC.dR
        self.dRjjOther[0] = event.sublGC.dR
        self.aveAbsEta[0] = event.aveAbsEta

        self.m4j[0] = event.m4j
        self.mHH[0] = event.views[0].mHH

        self.SB[0] = event.views[0].HHSB
        self.CR[0] = event.views[0].HHCR
        self.SR[0] = event.views[0].HHSR

        self.weight[0] = 1.0
        
        self.t.Fill()

    def Write(self):
        print(self.name+"_toyTree.root:",self.t.GetEntries()," entries")
        if self.debug: self.t.Show(0)
        self.f.Write()
        self.f.Close()
