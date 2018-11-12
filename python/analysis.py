#/usr/local/Cellar/python/3.7.1/bin/python3
#have to use python 3 with ROOT 6. ROOT 6 is required for ExRootAnalysis which is the madgraph package for converting .lhe files to .root files.
import ROOT

#Load the classes for LHE objects in ROOT
EXROOTANALYSIS_PATH='/Applications/MG5_aMC_v2_6_2/ExRootAnalysis/libExRootAnalysis.so'
ROOT.gSystem.Load(EXROOTANALYSIS_PATH)

from particle import *
from diJet import *
from eventView import *
from eventViewHists import *
from eventData import *

class analysis:
    def __init__(self, tree, outFileName, debug=False):
        self.debug = debug
        self.tree  = tree
        self.tree.GetEntry(0)
        if self.debug:
            self.tree.Show()

        self.outFileName = outFileName
        self.outFile = ROOT.TFile(self.outFileName,"RECREATE")

        #hists
        self.allViews = eventViewHists(self.outFile, "allViews")
        self.minDhhSR = eventViewHists(self.outFile, "SR")

        #event
        self.thisEvent = eventData(self.tree, self.debug)

        #preselection
        self.minPt  = 40
        self.maxEta = 2.5

        #high level selection
        self.maxXZZ = 1.6
            
    #Event Loop
    def eventLoop(self,events=None):
        #events is a list of events to process
        nEvents=self.tree.GetEntries()
        if not events: events = range(nEvents)

        print( "Processing",nEvents,"Events" )
        i=0
        for e in events:
            self.processEvent(e)
            i+=1
            if (i+1)%1000 == 0: print( "Processed",str(i+1).rjust(len(str(nEvents))),"of",str(nEvents),"Events" )
            
    def processEvent(self, entry):
        #initialize event and do truth level stuff before moving to reco (actual data analysis) stuff
        self.thisEvent.update(entry)

        #basic cuts
        passJetMultiplicity = len(self.thisEvent.bs) >= 4
        if not passJetMultiplicity:
            if self.debug: print( "Fail Jet Multiplicity" )
            return

        nPassJetPt = 0
        for b in self.thisEvent.bs:
            if b.pt > self.minPt:
                nPassJetPt += 1
        passJetPt = nPassJetPt >= 4
        if not passJetPt:
            if self.debug: print( "Fail Jet Pt" )
            return

        nPassJetEta = 0
        for b in self.thisEvent.bs:
            if b.eta < self.maxEta:
                nPassJetEta += 1
        passJetEta = nPassJetEta >= 4
        if not passJetEta:
            if self.debug: print( "Fail Jet Eta" )
            return

        #if event passes basic cuts start doing higher level constructions
        self.thisEvent.buildViews(self.thisEvent.bs)

        for view in self.thisEvent.views:
            self.allViews.Fill(view)

        passXZZ = self.thisEvent.views[0].xZZ < self.maxXZZ
        if not passXZZ:
            if self.debug: print( "Fail xZZ =",self.thisEvent.views[0].xZZ )
            return

        self.minDhhSR.Fill(self.thisEvent.views[0])
        
                
    def Write(self):
        self.allViews.Write()
        self.minDhhSR.Write()

        self.outFile.Close()



