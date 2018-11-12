from ROOT import TLorentzVector

class particle:
    def __init__(self, tree, i):
        self.PID       = tree.Particle[i].PID
        self.mom       = tree.Particle[i].Mother1
        self.daughters = []
        self.m         = tree.Particle[i].M
        self.p         = TLorentzVector(tree.Particle[i].Px,
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


