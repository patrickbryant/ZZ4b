from hists import *
from particleHists import *

class truthHists:
    def __init__(self, outFile, directory):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.nbs    = makeTH1F(self.thisDir, "nbs", directory+"/nbs; Truth number of b's; Entries",    7, -0.5,  6.5)
        self.mbs    = makeTH1F(self.thisDir, "mbs", directory+"/mbs; Truth m_{b's} [GeV]; Entries",  220,  100, 1200)
        self.bosons = particleHists(outFile, directory, "bosons")
        self.bs     = particleHists(outFile, directory, "bs")

    def Fill(self, event, weight=1, view=None):
        self.nbs.Fill(len(event.bs), weight)
        self.mbs.Fill(event.mbs, weight)
        for boson in event.Zs + event.Hs:
            self.bosons.Fill(boson, weight)
        for b in event.bs:
            self.bs.Fill(b, weight)
            
    def Write(self, outFile=None):
        self.thisDir.cd()
        self.nbs.Write()
        self.mbs.Write()
        self.bosons.Write()
        self.bs.Write()
