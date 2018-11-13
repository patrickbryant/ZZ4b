import selection as sel

def getDhh(m1,m2):
    return abs(m1-m2)

def getXZZ(m1,m2):
    return ( ((m1-91)/(0.1*m1))**2 + ((m2-91)/(0.1*m2))**2 )**0.5        

mZ = 91.188

class eventView:
    def __init__(self, diJet1, diJet2):
        self.diJet1 = diJet1
        self.diJet2 = diJet2

        self.lead   = diJet1 if diJet1.pt > diJet2.pt else diJet2
        self.subl   = diJet2 if diJet1.pt > diJet2.pt else diJet1

        self.leadSt = diJet1 if diJet1.st > diJet2.st else diJet2
        self.sublSt = diJet2 if diJet1.st > diJet2.st else diJet1

        self.dhh = getDhh(self.leadSt.m, self.sublSt.m)
        self.xZZ = getXZZ(self.leadSt.m, self.sublSt.m)

        self.m4j = (self.diJet1.p + self.diJet2.p).M()
        self.mZZ = (self.diJet1.p*(mZ/self.diJet1.m) + self.diJet2.p*(mZ/self.diJet2.m)).M()

        self.ZZ = self.xZZ < sel.maxXZZ

        self.passLeadStMDR = (360/self.m4j - 0.5 < self.leadSt.dR) and (self.leadSt.dR < 653/self.m4j + 0.475) if self.m4j < 1250 else (self.leadSt.dR < 1)
        self.passSublStMDR = (235/self.m4j       < self.sublSt.dR) and (self.leadSt.dR < 875/self.m4j + 0.350) if self.m4j < 1250 else (self.sublSt.dR < 1)
        self.passMDRs = self.passLeadStMDR and self.passSublStMDR

        self.passLeadMDC = self.lead.pt > self.m4j*0.51 - 103
        self.passSublMDC = self.subl.pt > self.m4j*0.33 -  73
        self.passMDCs = self.passLeadMDC and self.passSublMDC
        
    def dump(self):
        print("\nEvent View")
        print("dhh",self.dhh,"xZZ",self.xZZ,"mZZ",self.mZZ)
        self.diJet1.dump()
        self.diJet2.dump()

        
