from hists import *
from massRegionHists import *
from truthHists import *
from particleHists import *

class eventHists:
    def __init__(self, outFile, directory, truth=False):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        #plot quantities that are defined at the event level
        self.m4j = makeTH1F(self.thisDir, "m4j", directory+"_m4j; m_{4j} [GeV]; Entries", 220, 100, 1200)
        self.xWt = makeTH1F(self.thisDir, "xWt", directory+"_xWt; x_{Wt}; Entries", 50, 0 , 5)

        self.m4j_vs_nViews = makeTH2F(self.thisDir, "m4j_vs_nViews",
                                      directory+"_m4j_vs_nViews; m_{4j} [GeV]; # of event views; Entries",
                                      110,100,1200, 3,0.5,3.5  )

        #plot quantities that are only defined once diJets are constructed, ie "Event Views"
        outFile.mkdir(directory+"/allViews")
        self.allViews = massRegionHists(outFile, directory+"/allViews")
        outFile.mkdir(directory+"/mainView")
        self.mainView = massRegionHists(outFile, directory+"/mainView")

        #plot truth quantities from the simulation, ie information that cannot be directly observed in real data -- quantities based on the actual particle type rather than the infered particle type from detector measurements. 
        self.truth = None
        if truth:
            outFile.mkdir(directory+"/truth")
            self.truth = truthHists(outFile, directory+"/truth")

    def Fill(self, event, weight=1, view=None):
        self.m4j.Fill(event.m4j, weight)
        self.xWt.Fill(event.xWt, weight)

        self.m4j_vs_nViews.Fill(event.m4j, len(event.views), weight)

        if type(view) == list:
            for v in view:
                self.allViews.Fill(v, weight, event)
            self.mainView.Fill(view[0], weight, event)
        elif view:
            for v in event.views:
                self.allViews.Fill(v, weight, event)
            self.mainView.Fill(view, weight, event)
        else:
            for v in event.views:
                self.allViews.Fill(v, weight, event)
            self.mainView.Fill(event.views[0], weight, event)

        if self.truth:
            self.truth.Fill(event, weight)
            
    def Write(self, outFile=None):
        self.thisDir.cd()
        self.m4j.Write()
        self.xWt.Write()

        self.m4j_vs_nViews.Write()

        self.allViews.Write()
        self.mainView.Write()

        if self.truth:
            self.truth.Write()
