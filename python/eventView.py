import selection as sel
import constants as con

def getDBB(m1,m2):
    return abs(m1-m2)

def getXZZ(m1,m2):
    return ( ((m1-sel.leadZZ)/(0.1*m1))**2 + ((m2-sel.sublZZ)/(0.1*m2))**2 )**0.5        

def getXZH(m1,m2):
    return ( ((m1-sel.leadZH)/(0.1*m1))**2 + ((m2-sel.sublZH)/(0.1*m2))**2 )**0.5        

def getXHH(m1,m2):
    return ( ((m1-sel.leadHH)/(0.1*m1))**2 + ((m2-sel.sublHH)/(0.1*m2))**2 )**0.5        

class eventView:
    def __init__(self, diJet1, diJet2):
        self.diJet1 = diJet1
        self.diJet2 = diJet2

        #diJets sorted by pt
        self.lead   = diJet1 if diJet1.pt > diJet2.pt else diJet2
        self.subl   = diJet2 if diJet1.pt > diJet2.pt else diJet1

        #diJets sorted by scalar sum of jet pt (st)
        self.leadSt = diJet1 if diJet1.st > diJet2.st else diJet2
        self.sublSt = diJet2 if diJet1.st > diJet2.st else diJet1

        #diJets sorted by mass
        self.leadM  = diJet1 if diJet1.m  > diJet2.m  else diJet2
        self.sublM  = diJet2 if diJet1.m  > diJet2.m  else diJet1

        
        #mass plane variables
        self.dBB = getDBB(self.leadSt.m, self.sublSt.m) #Distance from being equal mass boson candidates
        self.xZZ = getXZZ(self.leadSt.m, self.sublSt.m) #0 for perfect consistency with ZZ->4b
        self.xZH = getXZH(self.leadM .m, self.sublM .m) #0 for perfect consistency with ZH->4b
        self.xHH = getXHH(self.leadSt.m, self.sublSt.m) #0 for perfect consistency with HH->4b

        
        #four body mass
        self.m4j = (self.diJet1.p  + self.diJet2.p ).M()
        #diBoson masses under different hypotheses
        self.mZZ = (self.diJet1.pZ + self.diJet2.pZ).M()
        self.mZH = (self.sublM .pZ + self.leadM .pH).M()
        self.mHH = (self.diJet1.pH + self.diJet2.pH).M()


        #booleans for whether or not this event view lies in the given mass region
        #Signal Regions
        self.ZZSR =  self.xZZ < sel.xZZSR
        self.ZHSR =  self.xZH < sel.xZHSR
        self.HHSR =  self.xHH < sel.xHHSR
        #Control Regions
        self.rZZCR = ( (self.leadSt.m - sel.leadZZ*sel.sZZCR)**2 + (self.sublSt.m - sel.sublZZ*sel.sZZCR)**2 )**0.5
        self.rZHCR = ( (self.leadSt.m - sel.leadZH*sel.sZHCR)**2 + (self.sublSt.m - sel.sublZH*sel.sZHCR)**2 )**0.5
        self.rHHCR = ( (self.leadSt.m - sel.leadHH*sel.sHHCR)**2 + (self.sublSt.m - sel.sublHH*sel.sHHCR)**2 )**0.5
        self.ZZCR = (self.rZZCR < sel.rZZCR) and not self.ZZSR
        self.ZHCR = (self.rZHCR < sel.rZHCR) and not self.ZHSR
        self.HHCR = (self.rHHCR < sel.rHHCR) and not self.HHSR
        #Sidebands
        self.rZZSB = ( (self.leadSt.m - sel.leadZZ*sel.sZZSB)**2 + (self.sublSt.m - sel.sublZZ*sel.sZZSB)**2 )**0.5
        self.rZHSB = ( (self.leadSt.m - sel.leadZH*sel.sZHSB)**2 + (self.sublSt.m - sel.sublZH*sel.sZHSB)**2 )**0.5
        self.rHHSB = ( (self.leadSt.m - sel.leadHH*sel.sHHSB)**2 + (self.sublSt.m - sel.sublHH*sel.sHHSB)**2 )**0.5
        self.ZZSB = (self.rZZCR < sel.rZZSB) and not self.ZZSR and not self.ZZCR
        self.ZHSB = (self.rZHCR < sel.rZHCR) and not self.ZHSR and not self.ZHCR
        self.HHSB = (self.rHHCR < sel.rHHCR) and not self.HHSR and not self.HHCR

        
        #booleans for event view requirements. These were optimized for the ATLAS HH search with 2015+2016 data. See page 124 of https://cds.cern.ch/record/2644551?ln=en
        self.passLeadStMDR = (360/self.m4j - 0.5 < self.leadSt.dR) and (self.leadSt.dR < 653/self.m4j + 0.475) if self.m4j < 1250 else (self.leadSt.dR < 1)
        self.passSublStMDR = (235/self.m4j       < self.sublSt.dR) and (self.sublSt.dR < 875/self.m4j + 0.350) if self.m4j < 1250 else (self.sublSt.dR < 1)
        self.passMDRs = self.passLeadStMDR and self.passSublStMDR

        #booleans for event cuts on selected view. These were optimized for the ATLAS HH search with 2015+2016 data. See page 130 of https://cds.cern.ch/record/2644551?ln=en
        self.passLeadMDC = self.lead.pt > self.m4j*0.51 - 103
        self.passSublMDC = self.subl.pt > self.m4j*0.33 -  73
        self.passMDCs = self.passLeadMDC and self.passSublMDC

        self.dEta = self.leadSt.eta - self.sublSt.eta
        self.passHCdEta = abs(self.dEta) < 1.5

        #keep track of how many views satisfy the event view requirements
        self.nViews = None
        
    def dump(self):
        print("\nEvent View")
        print("dbb",self.dBB,"xZZ",self.xZZ,"mZZ",self.mZZ)
        self.diJet1.dump()
        self.diJet2.dump()

        
