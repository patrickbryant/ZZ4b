from hists import *
from massRegionHists import *

class eventHists:
    def __init__(self, outFile, directory):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.m4j = makeTH1F(self.thisDir, "m4j", directory+"_m4j; m_{4j} [GeV]; Entries", 220, 100, 1200)

        outFile.mkdir(directory+"/allViews")
        self.allViews = massRegionHists(outFile, directory+"/allViews")
        outFile.mkdir(directory+"/mainView")
        self.mainView = massRegionHists(outFile, directory+"/mainView")

    def Fill(self, event, weight=1, view=None):
        self.m4j.Fill(event.m4j, weight)
        
        if type(view) == list:
            for v in view:
                self.allViews.Fill(v, weight)
            self.mainView.Fill(view[0], weight)
        elif view:
            for v in event.views:
                self.allViews.Fill(v, weight)
            self.mainView.Fill(view, weight)
        else:
            for v in event.views:
                self.allViews.Fill(v, weight)
            self.mainView.Fill(event.views[0], weight)
            
    def Write(self, outFile=None):
        self.thisDir.cd()
        self.m4j.Write()

        self.allViews.Write()
        self.mainView.Write()
