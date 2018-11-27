from hists import *

class jetHists:
    def __init__(self, outFile, directory, name):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.m   = makeTH1F(self.thisDir, name+"_m",   directory+"/"+name+"_m;   "+name+" mass [GeV];   Entries",  50,  0,  100)
        self.pt  = makeTH1F(self.thisDir, name+"_pt",  directory+"/"+name+"_pt;  "+name+" p_{T} [GeV];  Entries",  80,  0,  400)
        self.eta = makeTH1F(self.thisDir, name+"_eta", directory+"/"+name+"_eta; "+name+" #eta;         Entries",  60, -3,    3)
        self.phi = makeTH1F(self.thisDir, name+"_phi", directory+"/"+name+"_phi; "+name+" #phi;         Entries",  64, -3.2,  3.2)

    def Fill(self, diJet, weight=1):
        self.m  .Fill(diJet.m,   weight)
        self.pt .Fill(diJet.pt,  weight)
        self.eta.Fill(diJet.eta, weight)
        self.phi.Fill(diJet.phi, weight)

    def Write(self, outFile=None):
        self.thisDir.cd()
        self.m  .Write()
        self.pt .Write()
        self.eta.Write()
        self.phi.Write()
