from ROOT import TLorentzVector
from copy import copy
import numpy as np

#b-jet performance taken from https://indico.cern.ch/event/727605/contributions/2995766/attachments/1645135/2629007/JMET_NChernyavskaya_07_04_2018.pdf slide 20. For some reason they define \sigma as (75%-25%)/2 or 25% instead of 34%
# estimate c by resolution at 450 (sigma~0.04), scaled by 34/25 => 0.04*34/25=0.0544
# then get a by resolution at 50 (sigma~0.11) so 0.11*34/25=((a/50^0.5)^2 + 0.0544^2)^0.5 => a = 0.98
a=0.98
c=0.0544
def jetEnergyResolution(E):
    return ( (a/E**0.5)**2 + c**2 )**0.5
def jetMassResolution(E):
    return E/10.
class particle:
    def __init__(self, tree=None, i=None, original=None):
        if tree: #Build particle instance from TTree
            self.PID       = tree.Particle[i].PID
            self.mom       = tree.Particle[i].Mother1
            self.daughters = []
            self.m         = tree.Particle[i].M
            self.e         = tree.Particle[i].E
            self.p         = TLorentzVector()
            self.p.SetPxPyPzE(tree.Particle[i].Px,
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

            self.res   = jetEnergyResolution(original.p.E())
            self.smearE = -1
            while self.smearE < 0: self.smearE = np.random.normal(0.985, self.res)
            
            # self.offsetM = -1
            # while self.offsetM+original.p.M() < 5: self.offsetM = np.random.normal(5, 5)
            # self.smearM = -1
            # while self.smearM < 0.5: self.smearM = np.random.normal(1, 1)

            self.p         = TLorentzVector()
            self.p.SetPtEtaPhiM( original.p.Pt() * self.smearE,
                                 original.p.Eta(),
                                 original.p.Phi(),
                                 0)
            # self.p.SetPtEtaPhiM( original.p.Pt() * self.smearE,
            #                      original.p.Eta(),
            #                      original.p.Phi(),
            #                      (original.p.M()+self.offsetM) * self.smearM)
            self.pt  = self.p.Pt()
            self.eta = self.p.Eta()
            self.phi = self.p.Phi()
            self.m   = self.p.M()
            self.e   = self.p.E()
            self.SF  = 1

    def getDump(self):
        out = "PID "+str(self.PID).rjust(3)+" | mom "+str(self.mom).rjust(3)+" | mass "+str(self.m).ljust(12)+" | pt "+str(self.pt).ljust(20)+" | eta "+str(self.eta).ljust(20)+" | phi "+str(self.phi).ljust(20)
        if self.res: out += " | res "+str(self.res).ljust(20)+" | smear "+str(self.smear).ljust(20)
        return out
        
    def dump(self):
        print(self.getDump())
