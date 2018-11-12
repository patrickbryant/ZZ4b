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

        
