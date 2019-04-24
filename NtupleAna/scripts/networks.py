import numpy as np
np.random.seed(0)#always pick the same training sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Lin_View(nn.Module):
    def __init__(self):
        super(Lin_View, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)


class basicDNN(nn.Module):
    def __init__(self, inputFeatures, layers, nodes, pDropout):
        super(basicDNN, self).__init__()
        self.name = 'FC%dx%d_pdrop%.2f'%(layers, nodes, pDropout)
        fc=[]
        fc.append(nn.Linear(inputFeatures, nodes))
        fc.append(nn.ReLU())
        #fc.append(nn.Dropout(p=pDropout))
        for l in range(layers):
            fc.append(nn.Linear(nodes, nodes))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(p=pDropout))
            #if l < layers-1: fc.append(nn.Dropout(p=pDropout))
        fc.append(nn.Linear(nodes, 1))
        self.net = nn.Sequential(*fc)
        
    def forward(self, x):
        return self.net(x)


class basicCNN(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout):
        super(basicCNN, self).__init__()
        self.name = 'basicCNN_%d_%d_%d_%d_pdrop%.2f'%(dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  ##kernel=3
        self.conv1 = nn.Sequential(*[nn.Conv1d(     jetFeatures,        dijetFeatures, 2, stride=2), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv1d(   dijetFeatures,      quadjetFeatures, 2, stride=2), nn.ReLU()])
        self.conv3 = nn.Sequential(*[nn.Conv1d( quadjetFeatures, combinatoricFeatures, 3, stride=1), nn.ReLU()])

        self.line1 = nn.Sequential(*[nn.Linear(combinatoricFeatures, nodes), nn.ReLU()])
        self.line2 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
        self.line3 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
        self.line4 =                 nn.Linear(nodes, 1)

    def forward(self, x, a):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        
        x = self.line1(x)
        x = self.line2(x)
        x = self.line3(x)
        x = self.line4(x)
        return x


class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures, doReLU=False):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|            
        if doReLU:
            self.conv = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 3, stride=3), nn.ReLU()])
        else:
            self.conv = nn.Conv1d(self.nd, self.nd, 3, stride=3)

    def forward(self, x, d):
        n = x.shape[0]
        d = torch.cat( (x[:,:, 0: 2], d[:,:,0].view(n, self.nd, 1),
                        x[:,:, 2: 4], d[:,:,1].view(n, self.nd, 1),
                        x[:,:, 4: 6], d[:,:,2].view(n, self.nd, 1),
                        x[:,:, 6: 8], d[:,:,3].view(n, self.nd, 1),
                        x[:,:, 8:10], d[:,:,4].view(n, self.nd, 1),
                        x[:,:,10:12], d[:,:,5].view(n, self.nd, 1)), 2 )
        return self.conv(d)

class dijetResNetBlock(nn.Module):
    def __init__(self, dijetFeatures, nLayers=3):
        super(dijetResNetBlock, self).__init__()
        self.nd = dijetFeatures
        #self.reinforce1 = dijetReinforceLayer(self.nd, doReLU=True)
        #self.reinforce2 = dijetReinforceLayer(self.nd, doReLU=True)
        self.reinforce3 = dijetReinforceLayer(self.nd, doReLU=True)
        self.reinforce4 = dijetReinforceLayer(self.nd, doReLU=False)
        self.ResReLU = nn.ReLU()

    def forward(self, x, d):
        d0 = d.clone()
        #d = self.reinforce1(x, d)
        #d = self.reinforce2(x, d)
        d = self.reinforce3(x, d)
        d = self.reinforce4(x, d)
        return self.ResReLU( d + d0 )


class quadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures, doReLU=False):
        super(quadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        if doReLU:
            self.conv = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 3, stride=3), nn.ReLU()])
        else:
            self.conv = nn.Conv1d(self.nq, self.nq, 3, stride=3)

    def forward(self, x, q):
        n = x.shape[0]
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4:6], q[:,:,2].view(n,self.nq,1)), 2)
        return self.conv(q)

class jetQuadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures, doReLU=False):
        super(jetQuadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures
        # |1|2|3|4|1,2|3,4|1,2,3,4|1|3|2|4|1,3|2,4|1,3,2,4|1|4|2|3|1,4|2,3|1,4,2,3|  ##stride=7 kernel=7 preserve original jet info in quadjet feature forming
        #                 |1,2,3,4|               |1,3,2,4|               |1,4,2,3|  
        if doReLU:
            self.conv = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 7, stride=7), nn.ReLU()])
        else:
            self.conv = nn.Conv1d(self.nq, self.nq, 7, stride=7)

    def forward(self, j, d, q):
        n = j.shape[0]
        q = torch.cat( (j[:,:, 0: 4], d[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        j[:,:, 4: 8], d[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        j[:,:, 8:12], d[:,:, 4:6], q[:,:,2].view(n,self.nq,1)), 2)
        return self.conv(q)

class quadjetResNetBlock(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetResNetBlock, self).__init__()
        self.nq = quadjetFeatures
        #self.reinforce1 = quadjetReinforceLayer(self.nq, doReLU=True)
        #self.reinforce2 = quadjetReinforceLayer(self.nq, doReLU=True)
        self.reinforce3 = quadjetReinforceLayer(self.nq, doReLU=True)
        self.reinforce4 = quadjetReinforceLayer(self.nq, doReLU=False)
        self.ReLU = nn.ReLU()

    def forward(self, x, q):
        q0 = q.clone()
        #q = self.reinforce1(x, q)
        #q = self.reinforce2(x, q)
        q = self.reinforce3(x, q)
        q = self.reinforce4(x, q)
        return self.ReLU( q + q0 )

class quadjetPresResNetBlock(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetPresResNetBlock, self).__init__()
        self.nq = quadjetFeatures
        #self.reinforce1 = jetQuadjetReinforceLayer(self.nq, doReLU=True)
        #self.reinforce2 = jetQuadjetReinforceLayer(self.nq, doReLU=True)
        self.reinforce3 = jetQuadjetReinforceLayer(self.nq, doReLU=True)
        self.reinforce4 = jetQuadjetReinforceLayer(self.nq, doReLU=False)
        self.ReLU = nn.ReLU()

    def forward(self, j, d, q):
        q0 = q.clone()
        #q = self.reinforce1(j, d, q)
        #q = self.reinforce2(j, d, q)
        q = self.reinforce3(j, d, q)
        q = self.reinforce4(j, d, q)
        return self.ReLU( q + q0 )


class ResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures):
        super(ResNet, self).__init__()
        self.name = 'ResNet_%d_%d_%d'%(dijetFeatures, quadjetFeatures, combinatoricFeatures)
        self.nd = dijetFeatures
        self.nq = quadjetFeatures
        self.nAq = 1
        self.nAv = 3
        #self.useAncillary = False

        self.toDijetFeatureSpace = nn.Conv1d(jetFeatures, self.nd, 1)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 2, stride=2), nn.ReLU()])

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock = dijetResNetBlock(self.nd)

        self.toQuadjetFeatureSpace = nn.Conv1d(self.nd, self.nq, 1)
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = nn.Sequential(*[nn.Conv1d(self.nq+self.nAq, self.nq+self.nAq, 2, stride=2), nn.ReLU()])

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nq+self.nAq)

        self.viewSelector   = nn.Conv1d(self.nq+self.nAq+self.nAv, combinatoricFeatures, 3, stride=1)
        #self.ancillary      = nn.Sequential(*[nn.Linear(nAncillary+combinatoricFeatures, nAncillary+combinatoricFeatures), nn.ReLU()])
        #self.ancillary      = nn.Linear(combinatoricFeatures + nAncillary, combinatoricFeatures + nAncillary)
        #self.ancillary      = nn.Linear(nAncillary, nAncillary)
        #self.out = nn.Sequential(*[nn.ReLU(), nn.Linear(combinatoricFeatures, 1)])
        self.out = nn.Sequential(*[nn.ReLU(), nn.Linear(combinatoricFeatures, 1)])

    def forward(self, x, a):
        n = x.shape[0]

        x = self.toDijetFeatureSpace(x)
        d = self.dijetBuilder(x)
        d = self.dijetResNetBlock(x,d)

        x = self.toQuadjetFeatureSpace(d)
        dijetMasses = a[:,0:6].view(n,1,6)
        x = torch.cat( (x, dijetMasses), 1 ) # manually add dijet mass to dijet feature space
        q = self.quadjetBuilder(x)
        q = self.quadjetResNetBlock(x,q)

        ancillaryView = a[:,6:6+self.nAv].view(n,self.nAv,1) # |ancillaryView|
        ancillaryView = torch.cat( (ancillaryView, ancillaryView, ancillaryView), 2) # |ancillaryView|ancillaryView|ancillaryView|
        q = torch.cat( (q, ancillaryView), 1) # manually add ancillaryView features to combinatoric feature space
        x = self.viewSelector(q)
        x = x.view(x.shape[0], -1)
        
        #x = torch.cat( (x, a), 1)
        #x = self.ancillary(x)
        #if self.useAncillary:
        #    x = torch.cat( (x, a[:,6:7]), 1)
        #else:
        #    x = torch.cat( (x, torch.zeros((n,1), dtype=torch.float32, device=device)), 1)
        #x = self.ancillary(x)

        return self.out(x)


class PresResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillary):
        super(PresResNet, self).__init__()
        self.name = 'PresResNet_%d_%d_%d'%(dijetFeatures, quadjetFeatures, combinatoricFeatures)
        self.nd = dijetFeatures
        self.nq = quadjetFeatures
        self.useAncillary = False

        self.toDijetFeatureSpace = nn.Conv1d(jetFeatures, self.nd, 1)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 2, stride=2), nn.ReLU()])

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock = dijetResNetBlock(self.nd)

        self.jetToQuadjetFeatureSpace = nn.Conv1d(jetFeatures, self.nq, 1)
        self.dijetToQuadjetFeatureSpace = nn.Conv1d(self.nd, self.nq, 1)
        self.quadjetBuilder = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 2, stride=2), nn.ReLU()])

        # |1|2|3|4|1,2|3,4|1,2,3,4|1|3|2|4|1,3|2,4|1,3,2,4|1|4|2|3|1,4|2,3|1,4,2,3|  ##stride=7 kernel=7 preserve original jet info in quadjet feature forming
        #                 |1,2,3,4|               |1,3,2,4|               |1,4,2,3|  
        self.quadjetPresResNetBlock1 = quadjetPresResNetBlock(self.nq)
        self.quadjetPresResNetBlock2 = quadjetPresResNetBlock(self.nq)
        #self.quadjetPresResNetBlock3 = quadjetPresResNetBlock(self.nq)
        #self.quadjetPresResNetBlock4 = quadjetPresResNetBlock(self.nq)

        self.viewSelector   = nn.Conv1d(self.nq, combinatoricFeatures, 3, stride=1)
        self.out = nn.Sequential(*[nn.ReLU(), nn.Linear(combinatoricFeatures+nAncillary, 1)])

    def forward(self, x, a):
        n = x.shape[0]

        j = self.toDijetFeatureSpace(x)
        d = self.dijetBuilder(j)
        d = self.dijetResNetBlock(j,d)

        j = self.jetToQuadjetFeatureSpace(x)
        d = self.dijetToQuadjetFeatureSpace(d)
        q = self.quadjetBuilder(d)
        q = self.quadjetPresResNetBlock1(j,d,q)
        q = self.quadjetPresResNetBlock2(j,d,q)
        #q = self.quadjetPresResNetBlock3(j,d,q)
        #q = self.quadjetPresResNetBlock4(j,d,q)

        x = self.viewSelector(q)
        x = x.view(x.shape[0], -1)
        
        if self.useAncillary:
            x = torch.cat( (x, a), 1)
        else:
            x = torch.cat( (x, torch.zeros(a.shape, dtype=torch.float32, device=device)), 1)

        return self.out(x)


class deepResNet(nn.Module):
    def __init__(self, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillary):
        super(deepResNet, self).__init__()
        self.name = 'deepResNet_%d_%d_%d'%(dijetFeatures, quadjetFeatures, combinatoricFeatures)
        self.nd = dijetFeatures
        self.nq = quadjetFeatures
        self.useAncillary = False

        #self.toDijetFeatureSpace = nn.Sequential(*[nn.Conv1d(4, self.nd, 1), nn.ReLU()])
        self.toDijetFeatureSpace = nn.Conv1d(jetFeatures, self.nd, 1)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 2, stride=2), nn.ReLU()])

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock1 = dijetResNetBlock(self.nd)
        self.dijetResNetBlock2 = dijetResNetBlock(self.nd)
        #self.dijetResNetBlock3 = dijetResNetBlock(self.nd)
        #self.dijetResNetBlock4 = dijetResNetBlock(self.nd)
        #self.dijetResNetBlock5 = dijetResNetBlock(self.nd)

        self.toQuadjetFeatureSpace = nn.Conv1d(self.nd, self.nq, 1)
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 2, stride=2), nn.ReLU()])

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock1 = quadjetResNetBlock(self.nq)
        self.quadjetResNetBlock2 = quadjetResNetBlock(self.nq)
        #self.quadjetResNetBlock3 = quadjetResNetBlock(self.nq)
        #self.quadjetResNetBlock4 = quadjetResNetBlock(self.nq)
        #self.quadjetResNetBlock5 = quadjetResNetBlock(self.nq)

        self.viewSelector = nn.Conv1d(self.nq, combinatoricFeatures, 3, stride=1)

        #self.ancillary      = nn.Linear(combinatoricFeatures + nAncillary, combinatoricFeatures + nAncillary)
        self.out = nn.Sequential(*[nn.ReLU(), nn.Linear(combinatoricFeatures+nAncillary, 1)])
        #self.out = nn.Sequential(*[nn.ReLU(), nn.Linear(combinatoricFeatures, 1)])

    def forward(self, x, a):
        n = x.shape[0]

        x = self.toDijetFeatureSpace(x)
        d = self.dijetBuilder(x)
        d = self.dijetResNetBlock1(x,d)
        d = self.dijetResNetBlock2(x,d)
        #d = self.dijetResNetBlock3(x,d)
        #d = self.dijetResNetBlock4(x,d)
        #d = self.dijetResNetBlock5(x,d)

        x = self.toQuadjetFeatureSpace(d)
        q = self.quadjetBuilder(x)
        q = self.quadjetResNetBlock1(x,q)
        q = self.quadjetResNetBlock2(x,q)
        #q = self.quadjetResNetBlock3(x,q)
        #q = self.quadjetResNetBlock4(x,q)
        #q = self.quadjetResNetBlock5(x,q)

        x = self.viewSelector(q)
        x = x.view(x.shape[0], -1)

        if self.useAncillary:
            #a = self.ancillary(a)
            x = torch.cat( (x, a), 1)
        else:
            x = torch.cat( (x, torch.zeros(a.shape, dtype=torch.float32, device=device)), 1)
        
        return self.out(x)

