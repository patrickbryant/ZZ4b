from particle import *
from diJet import *
from eventView import *
from copy import copy
from ROOT import TLorentzVector
PID_Z = 23
PID_H = 25
PID_b = 5

combinations = [[[0,1],[2,3]],
                [[0,2],[1,3]],
                [[0,3],[1,2]]]

def smearJets(jets):
    smearedJets = []
    for jet in jets: smearedJets.append( particle(None, None, jet) )
    return smearedJets

class eventData:
    def __init__(self, tree, debug=False):
        self.debug = debug
        self.tree = tree

        self.particles = []
        self.weight = 1
        self.number = None
        self.bJets = []
        self.views = []

        self.Zs = []
        self.Hs = []
        self.bs = []
        self.mbs = -99

        self.recoJets = []
        self.m4j = None

        self.xWt = 1e6
        self.passTopVeto = False

        self.SR = False

    def reset(self):
        self.particles = []
        self.weight = 1
        self.number = None
        self.bJets = []
        self.views = []

        self.Zs = []
        self.Hs = []
        self.bs = []
        self.mbs = -99

        self.recoJets = []
        self.m4j = None

        self.xWt = 1e6
        self.passTopVeto = False

        self.SR = False

    def getParticles(self):
        self.particles = []
        for p in range(self.tree.Particle_size):
            self.particles.append(particle(self.tree, p))

    def getTruth(self):
        self.Zs = []
        self.Hs = []
        self.bs = []
        for p in self.particles:
            if self.debug: p.dump()
            if p.PID == PID_Z:
                self.Zs.append(p)
            if p.PID == PID_H:
                self.Hs.append(p)
            if abs(p.PID) == PID_b:
                self.bs.append(p)
                if self.particles[p.mom].PID == PID_Z:
                    self.particles[p.mom].daughters.append(p)
                if self.particles[p.mom].PID == PID_H:
                    self.particles[p.mom].daughters.append(p)

        if self.bs:
            pbs = copy(self.bs[0].p)
            for b in self.bs[1:]: pbs += b.p
            self.mbs = pbs.M()

    def applyTagSF(self,jets):
        for jet in jets: self.weight *= jet.SF

    def update(self,entry):
        self.reset()
        self.tree.GetEntry(entry)
        if self.debug: self.tree.Show()
        self.weight = self.tree.Event[0].Weight
        self.number = self.tree.Event[0].Number

        self.getParticles()
        self.getTruth()

        self.recoJets = smearJets(self.bs)
        for jet in self.recoJets:#for now assume flat 70% b-tag efficiency
            jet.SF = 0.7
            if self.debug: print("recoJet | "+jet.getDump())

        if len(self.recoJets) == 4:
            self.m4j = (self.recoJets[0].p + self.recoJets[1].p + self.recoJets[2].p + self.recoJets[3].p).M()

    def buildViews(self,jets):
        # consider all possible diJet pairings of the four selected b-jets
        self.views = []

        for combination in combinations:
            diJet1 = diJet( jets[combination[0][0]], jets[combination[0][1]] )
            diJet2 = diJet( jets[combination[1][0]], jets[combination[1][1]] )
            view = eventView(diJet1, diJet2)
            view.nViews = len(combinations)
            self.views.append(view)

        #sort by dBB. View with minimum dBB is views[0] after sort.
        self.views.sort(key=lambda view: view.dBB)
        if self.debug:
            self.views[0].dump()

    def buildTops(self,bJets,otherJets):
        allJets = bJets + otherJets
        self.xWt = 1e6
        for b in bJets:
            for j1 in allJets:
                if j1 == b: continue
                for j2 in allJets:
                    if j2 == b:  continue
                    if j2 == j1: continue
                    mW  = (j1.p + j2.p).M()
                    mt  = (b.p + j1.p + j2.p).M()
                    xWt = ( ((mt - 172.5)/(0.1*mt))**2 + ((mW - 80.4)/(0.1*mW))**2 )**0.5
                    if xWt < self.xWt: self.xWt = xWt

        self.passTopVeto = self.xWt > 1.5


    def applyMDRs(self):
        #m4j Dependent Requirements for views
        passingViews = []
        for view in self.views:
            if view.passMDRs:
                passingViews.append(view)

        self.views = passingViews

        for view in self.views:
            view.nViews = len(self.views)
        
