from particle import *
from diJet import *
from eventView import *
PID_Z = 23
PID_b = 5

combinations = [[[0,1],[2,3]],
                [[0,2],[1,3]],
                [[0,3],[1,2]]]

class eventData:
    def __init__(self, tree, debug=False):
        self.debug = debug
        self.tree = tree

        self.particles = []
        self.weight = None
        self.number = None
        self.bJets = []
        self.views = []

        self.Zs = []
        self.bs = []

    def reset(self):
        self.particles = []
        self.weight = None
        self.number = None
        self.bJets = []
        self.views = []

        self.Zs = []
        self.bs = []

    def getParticles(self):
        self.particles = []
        for p in range(self.tree.Particle_size):
            self.particles.append(particle(self.tree, p))        

    def getZs(self):
        self.Zs = []
        self.bs = []
        for p in self.particles:
            if self.debug: p.dump()
            if p.PID == PID_Z:
                self.Zs.append(p)
            if abs(p.PID) == PID_b:
                self.bs.append(p)
                if self.particles[p.mom].PID == PID_Z:
                    self.particles[p.mom].daughters.append(p)

    def update(self,entry):
        self.reset()
        self.tree.GetEntry(entry)
        if self.debug: self.tree.Show()

        self.getParticles()
        self.getZs()

    def buildViews(self,jets):
        # consider all possible dijet pairings of the three b-jets
        self.views = []

        for combination in combinations:
            dijet1 = diJet( jets[combination[0][0]], jets[combination[0][1]] )
            dijet2 = diJet( jets[combination[1][0]], jets[combination[1][1]] )
            self.views.append(eventView(dijet1, dijet2))

        #sort by dhh. View with minimum dhh is views[0] after sort.
        self.views.sort(key=lambda view: view.dhh)
        if self.debug:
            self.views[0].dump()


        
