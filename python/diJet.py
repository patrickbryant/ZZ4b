class diJet:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

        self.p   = p1.p + p2.p
        self.dR  = p1.p.DeltaR(p2.p)
        self.m   = self.p.M()
        self.pt  = self.p.Pt()
        self.pt  = self.p.Pt()
        self.eta = self.p.Eta()
        self.phi = self.p.Phi()
        self.st  = self.p1.pt + self.p2.pt

        self.lead = self.p1 if self.p1.pt > self.p2.pt else self.p2
        self.subl = self.p2 if self.p1.pt > self.p2.pt else self.p1

    def dump(self):
        print( "diJet | mass",str(self.m).ljust(12),"| dR",str(self.dR).ljust(20),"| pt",str(self.pt).ljust(20),"| eta",str(self.eta).ljust(20),"| phi",str(self.phi).ljust(20) )
        print( "  lead "+self.lead.getDump() )
        print( "  subl "+self.subl.getDump() )


