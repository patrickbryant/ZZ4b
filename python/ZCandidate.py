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


