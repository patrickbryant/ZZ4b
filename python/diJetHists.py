from hists import *
from jetHists import *

class diJetHists:
    def __init__(self, outFile, directory, name):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.m   = makeTH1F(self.thisDir, name+"_m",   directory+"/"+name+"_m;   "+name+" diJet mass [GeV];   Entries", 150,  0,  300)
        self.dR  = makeTH1F(self.thisDir, name+"_dR",  directory+"/"+name+"_dR;  "+name+" diJet #DeltaR_{jj}; Entries",  80,  0,    4)
        self.pt  = makeTH1F(self.thisDir, name+"_pt",  directory+"/"+name+"_pt;  "+name+" diJet p_{T} [GeV];  Entries",  80,  0,  400)
        self.st  = makeTH1F(self.thisDir, name+"_st",  directory+"/"+name+"_st;  "+name+" diJet s_{T} [GeV];  Entries",  80, 50,  450)
        self.eta = makeTH1F(self.thisDir, name+"_eta", directory+"/"+name+"_eta; "+name+" diJet #eta;         Entries",  60, -3,    3)
        self.phi = makeTH1F(self.thisDir, name+"_phi", directory+"/"+name+"_phi; "+name+" diJet #phi;         Entries",  64, -3.2,  3.2)

        self.lead = jetHists(outFile, directory, name+"_lead")
        self.subl = jetHists(outFile, directory, name+"_subl")

    def Fill(self, diJet, weight=1):
        self.m  .Fill(diJet.m,   weight)
        self.dR .Fill(diJet.dR,  weight)
        self.pt .Fill(diJet.pt,  weight)
        self.st .Fill(diJet.st,  weight)
        self.eta.Fill(diJet.eta, weight)
        self.phi.Fill(diJet.phi, weight)

        self.lead.Fill(diJet.lead, weight)
        self.subl.Fill(diJet.subl, weight)

    def Write(self, outFile=None):
        self.thisDir.cd()
        self.m  .Write()
        self.dR .Write()
        self.pt .Write()
        self.st .Write()
        self.eta.Write()
        self.phi.Write()

        self.lead.Write()
        self.subl.Write()
