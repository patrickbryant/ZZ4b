#/usr/local/Cellar/python/3.7.1/bin/python3
#have to use python 3 with ROOT 6. ROOT 6 is required for ExRootAnalysis which is the madgraph package for converting .lhe files to .root files.
import ROOT

#Load the classes for LHE objects in ROOT
EXROOTANALYSIS_PATH='/Applications/MG5_aMC_v2_6_2/ExRootAnalysis/libExRootAnalysis.so'
ROOT.gSystem.Load(EXROOTANALYSIS_PATH)

PID_Z = 23
PID_b = 5

import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--inFileName',           dest="inFileName",         default="", help="")
parser.add_option('-o', '--outFileName',          dest="outFileName",        default="", help="")
parser.add_option('-n', '--nEvents',              dest="nEvents",            default=None, help="Number of events to process")
parser.add_option('-d', '--debug',                dest="debug",    action="store_true", default=False, help="Debug")
o, a = parser.parse_args()

class particle:
    def __init__(self, tree, i):
        self.PID       = tree.Particle[i].PID
        self.mom       = tree.Particle[i].Mother1
        self.daughters = []
        self.m         = tree.Particle[i].M
        self.p         = ROOT.TLorentzVector(tree.Particle[i].Px,
                                             tree.Particle[i].Py,
                                             tree.Particle[i].Pz,
                                             tree.Particle[i].E)
        self.pt  = self.p.Pt()
        self.eta = self.p.Eta()
        self.phi = self.p.Phi()

    def getDump(self):
        return "PID "+str(self.PID).rjust(3)+" | mom "+str(self.mom).rjust(3)+" | mass "+str(self.m).ljust(12)+" | pt "+str(self.pt).ljust(20)+" | eta "+str(self.eta).ljust(20)+" | phi "+str(self.phi).ljust(20) 
        
    def dump(self):
        print(self.getDump())


class ZCandidate:
    def __init__(self, particle1, particle2):
        self.particle1 = particle1
        self.particle2 = particle2

        self.p   = particle1.p + particle2.p
        self.dR  = particle1.p.DeltaR(particle2.p)
        self.m   = self.p.M()
        self.pt  = self.p.Pt()
        self.pt  = self.p.Pt()
        self.eta = self.p.Eta()
        self.phi = self.p.Phi()
        self.st  = self.particle1.pt + self.particle2.pt

        self.lead = self.particle1 if self.particle1.pt > self.particle2.pt else self.particle2
        self.subl = self.particle2 if self.particle1.pt > self.particle2.pt else self.particle1

    def dump(self):
        print( "Z Candidate | mass",str(self.m).ljust(12),"| dR",str(self.dR).ljust(20),"| pt",str(self.pt).ljust(20),"| eta",str(self.eta).ljust(20),"| phi",str(self.phi).ljust(20) )
        print( "  lead "+self.lead.getDump() )
        print( "  subl "+self.subl.getDump() )


def getDhh(m1,m2):
    return abs(m1-m2)

def getXZZ(m1,m2):
    return ( ((m1-91)/(0.1*m1))**2 + ((m2-91)/(0.1*m2))**2 )**0.5


class eventView:
    def __init__(self, dijet1, dijet2):
        self.dijet1 = dijet1
        self.dijet2 = dijet2

        self.lead   = dijet1 if dijet1.pt > dijet2.pt else dijet2
        self.subl   = dijet2 if dijet1.pt > dijet2.pt else dijet1

        self.leadSt = dijet1 if dijet1.st > dijet2.st else dijet2
        self.sublSt = dijet2 if dijet1.st > dijet2.st else dijet1

        self.dhh = getDhh(self.leadSt.m, self.sublSt.m)
        self.xZZ = getXZZ(self.leadSt.m, self.sublSt.m)

    def dump(self):
        print("\nEvent View")
        print("dhh",self.dhh,"xZZ",self.xZZ)
        self.dijet1.dump()
        self.dijet2.dump()

        
def makeTH1F(directory,name,title,bins,low,high):
    h = ROOT.TH1F(name,title,bins,low,high)
    h.SetDirectory(directory)
    return h

def makeTH2F(directory,name,title,xBins,xLow,xHigh,yBins,yLow,yHigh):
    h = ROOT.TH2F(name,title,xBins,xLow,xHigh,yBins,yLow,yHigh)
    h.SetDirectory(directory)
    return h

class eventViewHists:
    def __init__(self, outFile, directory):
        self.thisDir = outFile.mkdir(directory)

        self.leadSt_m = makeTH1F(self.thisDir, "leadSt_m", directory+"_leadSt_m;    leading(S_{T}) mass [GeV]; Entries", 250, 0, 250)
        self.sublSt_m = makeTH1F(self.thisDir, "sublSt_m", directory+"_sublSt_m; subleading(S_{T}) mass [GeV]; Entries", 250, 0, 250)
        self.leadSt_m_vs_sublSt_m = makeTH2F(self.thisDir, "leadSt_m_vs_sublSt_m",
                                             directory+"_leadSt_m_vs_sublSt_m; leading(S_{T}) mass [GeV]; subleading(S_{T}) mass [GeV];Entries",
                                             50,0,250, 50,0,250)
        self.xZZ = makeTH1F(self.thisDir, "xZZ", directory+"_xZZ; xZZ; Entries", 50, 0, 5)

    def Fill(self, view):
        self.leadSt_m.Fill(view.leadSt.m)
        self.sublSt_m.Fill(view.sublSt.m)
        self.leadSt_m_vs_sublSt_m.Fill(view.leadSt.m, view.sublSt.m)
        self.xZZ.Fill(view.xZZ)

    def Write(self, outFile=None):
        self.thisDir.cd()
        self.leadSt_m.Write()
        self.sublSt_m.Write()
        self.leadSt_m_vs_sublSt_m.Write()
        self.xZZ.Write()        

        
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


f=ROOT.TFile(o.inFileName)
tree=f.Get("LHEF")


a = analysis(tree, o.outFileName, o.debug)
if o.nEvents: a.eventLoop(range(int(o.nEvents)))
else:         a.eventLoop()
f.Close()
a.Write()

