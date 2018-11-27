from ROOT import TLorentzVector
from copy import copy
import numpy as np

a=0.5
c=0.02
def hCalResolution(E):
    return ( (a/E**0.5)**2 + c**2 )**0.5

class particle:
    def __init__(self, tree=None, i=None, original=None):
        if tree: #Build particle instance from TTree
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
            self.SF  = 1
            self.smear = None
            self.res   = None
        elif original: #Make a resolution smeared version of the original
            self.PID       = copy(original.PID)
            self.mom       = copy(original.mom)
            self.daughters = copy(original.daughters)
            self.res   = hCalResolution(original.p.E())
            self.smear = np.random.normal(1, self.res)
            if self.smear < 0: self.smear = 0
            self.p         = TLorentzVector(original.p.Px() * self.smear,
                                            original.p.Py() * self.smear,
                                            original.p.Pz() * self.smear,
                                            original.p.E()  * self.smear)
            self.pt  = self.p.Pt()
            self.eta = self.p.Eta()
            self.phi = self.p.Phi()
            self.m   = self.p.M()
            self.SF  = 1

    def getDump(self):
        out = "PID "+str(self.PID).rjust(3)+" | mom "+str(self.mom).rjust(3)+" | mass "+str(self.m).ljust(12)+" | pt "+str(self.pt).ljust(20)+" | eta "+str(self.eta).ljust(20)+" | phi "+str(self.phi).ljust(20)
        if self.res: out += " | res "+str(self.res).ljust(20)+" | smear "+str(self.smear).ljust(20)
        return out
        
    def dump(self):
        print(self.getDump())
