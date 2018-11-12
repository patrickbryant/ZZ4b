#/usr/local/Cellar/python/3.7.1/bin/python3
#have to use python 3 with ROOT 6. ROOT 6 is required for ExRootAnalysis which is the madgraph package for converting .lhe files to .root files.
import ROOT

#Load the classes for LHE objects in ROOT
EXROOTANALYSIS_PATH='/Applications/MG5_aMC_v2_6_2/ExRootAnalysis/libExRootAnalysis.so'
ROOT.gSystem.Load(EXROOTANALYSIS_PATH)

PID_Z = 23
PID_b = 5

from particle import *
from ZCandidate import *
from eventView import *
from eventViewHists import *

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

        #preselection
        self.minPt  = 40
        self.maxEta = 2.5

        #high level selection
        self.maxXZZ = 1.6
            
        #event level objects
        self.particles = []

    #Event Loop
    def eventLoop(self,events=None):
        #events is a list of events to process
        nEvents=self.tree.GetEntries()
        if not events: events = range(nEvents)

        print( "Processing",nEvents,"Events" )
        i=0
        for e in events:
            self.tree.GetEntry(e)
            self.processEvent()
            i+=1
            if (i+1)%1000 == 0: print( "Processed",str(i+1).rjust(len(str(nEvents))),"of",str(nEvents),"Events" )
            
    def getParticles(self):
        self.particles = []
        for p in range(self.tree.Particle_size):
            self.particles.append(particle(self.tree, p))

    def processEvent(self, event=None):
        if event: self.tree.GetEntry(event)
        if self.debug: self.tree.Show()

        self.getParticles()

        Zs = []
        bs = []

        for p in self.particles:
            if self.debug: p.dump()
            if p.PID == PID_Z:
                Zs.append(p)
            if abs(p.PID) == PID_b:
                bs.append(p)
                if self.particles[p.mom].PID == PID_Z:
                    self.particles[p.mom].daughters.append(p)

        passJetMultiplicity = len(bs) >= 4
        if not passJetMultiplicity:
            if self.debug: print( "Fail Jet Multiplicity" )
            return

        nPassJetPt = 0
        for b in bs:
            if b.pt > self.minPt:
                nPassJetPt += 1
        passJetPt = nPassJetPt >= 4
        if not passJetPt:
            if self.debug: print( "Fail Jet Pt" )
            return

        nPassJetEta = 0
        for b in bs:
            if b.eta < self.maxEta:
                nPassJetEta += 1
        passJetEta = nPassJetEta >= 4
        if not passJetEta:
            if self.debug: print( "Fail Jet Eta" )
            return
        
        combinations = [[[0,1],[2,3]],
                        [[0,2],[1,3]],
                        [[0,3],[1,2]]]

        views = []

        for combination in combinations:
            dijet1 = ZCandidate( bs[combination[0][0]], bs[combination[0][1]] )
            dijet2 = ZCandidate( bs[combination[1][0]], bs[combination[1][1]] )
            views.append(eventView(dijet1, dijet2))

        #sort by dhh. View with minimum dhh is views[0] after sort.
        views.sort(key=lambda view: view.dhh)
        if self.debug:
            views[0].dump()

        for view in views:
            self.allViews.Fill(view)

        passXZZ = views[0].xZZ < self.maxXZZ
        if not passXZZ:
            if self.debug: print( "Fail xZZ =",views[0].xZZ )
            return

        self.minDhhSR.Fill(views[0])
        
                
    def Write(self):
        self.allViews.Write()
        self.minDhhSR.Write()

        self.outFile.Close()



