from hists import *

class cutflowHists:
    def __init__(self, outFile, directory):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.unitWeight = makeTH1F(self.thisDir, "unitWeight", directory+"_unitWeight; ; Entries", 1, 1, 2)
        self.unitWeight.SetCanExtend(1)
        self.unitWeight.GetXaxis().FindBin("all")

        self.weighted = makeTH1F(self.thisDir, "weighted", directory+"_weighted; ; Entries", 1, 1, 2)
        self.weighted.SetCanExtend(1)
        self.weighted.GetXaxis().FindBin("all")

    def Fill(self, cut, weight=1):
        self.unitWeight.Fill(cut, 1)
        self.weighted  .Fill(cut, weight)
        
    def Write(self, outFile=None):
        self.thisDir.cd()
        self.unitWeight.Write()
        self.weighted  .Write()
