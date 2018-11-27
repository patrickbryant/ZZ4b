from hists import *

class particleHists:
    def __init__(self, outFile, directory, name):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.m   = makeTH1F(self.thisDir, name+"_m",   directory+"/"+name+"_m;   "+name+" mass [GeV];   Entries", 200,   0,  200)
        self.pt  = makeTH1F(self.thisDir, name+"_pt",  directory+"/"+name+"_pt;  "+name+" p_{T} [GeV];  Entries",  80,   0,  400)
        self.eta = makeTH1F(self.thisDir, name+"_eta", directory+"/"+name+"_eta; "+name+" #eta;         Entries",  60,  -3,    3)
        self.phi = makeTH1F(self.thisDir, name+"_phi", directory+"/"+name+"_phi; "+name+" #phi;         Entries",  64,  -3.2,  3.2)

        self.PID = makeTH1F(self.thisDir, name+"_PID", directory+"/"+name+"_PID; "+name+" PID;          Entries",  52, -26,  26)
        self.SF  = makeTH1F(self.thisDir, name+"_SF",  directory+"/"+name+"_SF;  "+name+" SF;           Entries",  20,   0,   2)

    def Fill(self, particle, weight=1):
        self.m  .Fill(particle.m,   weight)
        self.pt .Fill(particle.pt,  weight)
        self.eta.Fill(particle.eta, weight)
        self.phi.Fill(particle.phi, weight)

        self.PID.Fill(particle.PID, weight)
        self.SF.Fill(particle.SF,   weight)

    def Write(self, outFile=None):
        self.thisDir.cd()
        self.m  .Write()
        self.pt .Write()
        self.eta.Write()
        self.phi.Write()

        self.PID.Write()
        self.SF .Write()
