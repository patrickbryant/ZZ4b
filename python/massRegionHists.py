from hists import *
from eventViewHists import *

class massRegionHists:
    def __init__(self, outFile, directory):
        #self.thisDir = outFile.GetDirectory(directory)
        #if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
        #    self.thisDir = outFile.mkdir(directory)
        outFile.mkdir(directory+"/inclusive")
        self.inclusive = eventViewHists(outFile, directory+"/inclusive")
        outFile.mkdir(directory+"/ZZ")
        self.ZZ        = eventViewHists(outFile, directory+"/ZZ", True)

    def Fill(self, view, weight=1, event=None):
        self.inclusive.Fill(view, weight)
        if view.ZZ: self.ZZ.Fill(view, weight, event)
        
    def Write(self, outFile=None):
        #self.thisDir.cd()
        self.inclusive.Write()
        self.ZZ.Write()
