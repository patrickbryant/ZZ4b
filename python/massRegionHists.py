from hists import *
from eventViewHists import *

class massRegionHists:
    def __init__(self, outFile, directory):
        #once diJets are constructed, one can ask what regions of the 2D plane defined by (dijet1 mass, dijet2 mass) a given pair of dijets lands in.
        #example: make plots of dijet masses for events where both dijets are consistent with the Z boson mass.

        #plot all dijet pairings
        outFile.mkdir(directory+"/inclusive")
        self.inclusive = eventViewHists(outFile, directory+"/inclusive")

        #Signal Region (SR)
        #only plot dijet pairings that are consistent with BB->(jj)(jj) where B is Z or H
        outFile.mkdir(directory+"/ZZSR")
        self.ZZSR        = eventViewHists(outFile, directory+"/ZZSR", True)
        outFile.mkdir(directory+"/ZHSR")
        self.ZHSR        = eventViewHists(outFile, directory+"/ZHSR", True)
        outFile.mkdir(directory+"/HHSR")
        self.HHSR        = eventViewHists(outFile, directory+"/HHSR", True)

        #Control Region (CR)
        outFile.mkdir(directory+"/ZZCR")
        self.ZZCR        = eventViewHists(outFile, directory+"/ZZCR", True)
        outFile.mkdir(directory+"/ZHCR")
        self.ZHCR        = eventViewHists(outFile, directory+"/ZHCR", True)
        outFile.mkdir(directory+"/HHCR")
        self.HHCR        = eventViewHists(outFile, directory+"/HHCR", True)

        #Sideband (SB)
        outFile.mkdir(directory+"/ZZSB")
        self.ZZSB        = eventViewHists(outFile, directory+"/ZZSB", True)
        outFile.mkdir(directory+"/ZHSB")
        self.ZHSB        = eventViewHists(outFile, directory+"/ZHSB", True)
        outFile.mkdir(directory+"/HHSB")
        self.HHSB        = eventViewHists(outFile, directory+"/HHSB", True)
        
    def Fill(self, view, weight=1, event=None):
        self.inclusive.Fill(view, weight)
        if view.ZZSR: self.ZZSR.Fill(view, weight, event)
        if view.ZHSR: self.ZHSR.Fill(view, weight, event)
        if view.HHSR: self.HHSR.Fill(view, weight, event)

        if view.ZZCR: self.ZZCR.Fill(view, weight, event)
        if view.ZHCR: self.ZHCR.Fill(view, weight, event)
        if view.HHCR: self.HHCR.Fill(view, weight, event)

        if view.ZZSB: self.ZZSB.Fill(view, weight, event)
        if view.ZHSB: self.ZHSB.Fill(view, weight, event)
        if view.HHSB: self.HHSB.Fill(view, weight, event)

    def Write(self, outFile=None):
        #self.thisDir.cd()
        self.inclusive.Write()
        self.ZZSR.Write()
        self.ZHSR.Write()
        self.HHSR.Write()

        self.ZZCR.Write()
        self.ZHCR.Write()
        self.HHCR.Write()

        self.ZZSB.Write()
        self.ZHSB.Write()
        self.HHSB.Write()
