import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Lin_View(nn.Module):
    def __init__(self):
        super(Lin_View, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

def SiLU(x): #SiLU https://arxiv.org/pdf/1702.03118.pdf   Swish https://arxiv.org/pdf/1710.05941.pdf
    return x * torch.sigmoid(x)

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
        
    def forward(self, x, p, a):
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
    def __init__(self, dijetFeatures):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|            
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
        self.reinforce1 = dijetReinforceLayer(self.nd)
        self.reinforce2 = dijetReinforceLayer(self.nd)
        # self.reinforce3 = dijetReinforceLayer(self.nd)
        # self.reinforce4 = dijetReinforceLayer(self.nd)
        # self.reinforce5 = dijetReinforceLayer(self.nd)
        # self.reinforce6 = dijetReinforceLayer(self.nd)

    def forward(self, x, d):
        d0 = d.clone()
        d = SiLU(d)
        d = self.reinforce1(x, d)
        d = d+d0
        d = SiLU(d)
        d = self.reinforce2(x, d)
        #d2 = d.clone()
        d = d+d0
        d = SiLU(d)
        # d = self.reinforce3(x, d)
        # #d3 = d.clone()
        # d = d+d0
        # d = self.ReLU3(d)
        # d = self.reinforce4(x, d)
        # #d4 = d.clone()
        # d = d+d3
        # d = self.ReLU4(d)
        # d = self.reinforce5(x, d)
        # d = d+d3
        # d = self.ReLU5(d)
        # d = self.reinforce6(x, d)
        # d = d+d3
        # d = self.ReLU6(d)
        return d


class quadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.conv = nn.Conv1d(self.nq, self.nq, 3, stride=3)

    def forward(self, x, q):
        n = x.shape[0]
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4:6], q[:,:,2].view(n,self.nq,1)), 2)
        return self.conv(q)


class quadjetResNetBlock(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetResNetBlock, self).__init__()
        self.nq = quadjetFeatures
        self.reinforce1 = quadjetReinforceLayer(self.nq)
        self.reinforce2 = quadjetReinforceLayer(self.nq)
        # self.reinforce3 = quadjetReinforceLayer(self.nq)
        # self.reinforce4 = quadjetReinforceLayer(self.nq)
        # self.reinforce5 = quadjetReinforceLayer(self.nq)
        # self.reinforce6 = quadjetReinforceLayer(self.nq)

    def forward(self, x, q):
        q0 = q.clone()
        q = SiLU(q)
        q = self.reinforce1(x, q)
        q = q+q0
        q = SiLU(q)
        q = self.reinforce2(x, q)
        #q2 = q.clone()
        q = q+q0
        q = SiLU(q)
        # q = self.reinforce3(x, q)
        # #q3 = q.clone()
        # q = q+q0
        # q = self.ReLU3(q)
        # q = self.reinforce4(x, q)
        # #q4 = q.clone()
        # q = q+q0
        # q = self.ReLU4(q)
        # q = self.reinforce5(x, q)
        # q = q+q0
        # q = self.ReLU5(q)
        # q = self.reinforce6(x, q)
        # q = q+q0
        # q = self.ReLU6(q)
        return q


class jetRNN(nn.Module):
    def __init__(self, jetFeatures, hiddenFeatures):
        super(jetRNN, self).__init__()
        self.nj = jetFeatures
        self.nh = hiddenFeatures
        self.lstm = nn.LSTM(self.nj, self.nh, num_layers=1, batch_first=True)

    def forward(self, js, ls):#j[event][jet][mu] l[event][nj]
        batch_size, seq_len, feature_len = js.size()
        js = torch.nn.utils.rnn.pack_padded_sequence(js, ls, batch_first=True)
        hs, _ = self.lstm(js)
        
        hs, ls = torch.nn.utils.rnn.pad_packed_sequence(hs, batch_first=False)
        hs = hs.view(batch_size*seq_len, self.nh)
        ls = [(l-1)*batch_size + i for i,l in enumerate(ls)]
        ls = torch.tensor(ls, dtype=torch.int64).to("cuda")
        hs = hs.index_select(0,ls)
        hs = hs.view(batch_size,self.nh)
        return hs

class ResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillaryFeatures):
        super(ResNet, self).__init__()
        self.name = 'ResNet_%d_%d_%d'%(dijetFeatures, quadjetFeatures, combinatoricFeatures)
        self.nj = jetFeatures
        self.nd, self.nAd = dijetFeatures, 2
        self.nq, self.nAq = quadjetFeatures, 2 #+nAncillaryFeatures
        self.nc, self.nAv = combinatoricFeatures, nAncillaryFeatures

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        #self.otherJetRNN = jetRNN(self.nj,3)

        self.toDijetFeatureSpace = nn.Conv1d(self.nj, self.nd, 1)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = nn.Conv1d(self.nd, self.nd-self.nAd, 2, stride=2)
        # ancillary dijet features get appended to output of dijetBuilder

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock = dijetResNetBlock(self.nd)

        self.toQuadjetFeatureSpace = nn.Conv1d(self.nd, self.nq, 1)
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = nn.Conv1d(self.nq, self.nq-self.nAq, 2, stride=2)
        # ancillary quadjet features get appended to output of quadjetBuilder

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nq)
        # ancillary view features get appended to output of quadjetResNetBlock

        self.toViewFeatureSpace = nn.Conv1d(self.nq+self.nAv, self.nc, 1)
        self.viewConv1 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewConv2 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewSelector = nn.Conv1d(self.nc, 1, 3, stride=1)
        #self.viewSelector = nn.Sequential(*[nn.MaxPool1d(3), nn.Conv1d(self.nc, 1, 1)])

    def rotate(self, p, R): # p[event, mu, jet], mu=2 is phi
        pR = p.clone()
        pR[:,2,:] = (pR[:,2,:] + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
        return pR

    def flipPhi(self, p): # p[event, mu, jet], mu=2 is phi
        pF = p.clone()
        pF[:,2,:] = -1*pF[:,2,:]
        return pF

    def flipEta(self, p): # p[event, mu, jet], mu=1 is eta
        pF = p.clone()
        pF[:,1,:] = -1*pF[:,1,:]
        return pF

    def invPart(self,p,da,qa):
        n = p.shape[0]
        p = self.toDijetFeatureSpace(p)
        d = self.dijetBuilder(p)
        d = torch.cat( (d, da), 1 ) # manually add dijet mass and dRjj to dijet feature space
        d = self.dijetResNetBlock(p,d)
        
        d = self.toQuadjetFeatureSpace(d)
        q = self.quadjetBuilder(d)
        #nj = va[:,0:1,:]
        q = torch.cat( (q, qa), 1) # manually add features to quadjet feature space
        return self.quadjetResNetBlock(d,q) 

    def forward(self, x, p, da, qa, va):#, js, ls):
        n = p.shape[0]

        #js = torch.transpose(p,1,2)
        #ls = torch.ones(n,dtype=torch.long)*12
        #hs = self.otherJetRNN(js,ls)

        da = torch.cat( (da[:,0:6].view(n,1,6), da[:,6:12].view(n,1,6)), 1) #format dijet masses and dRjjs 
        qa = torch.cat( (qa[:,0:3].view(n,1,3), qa[:,3: 6].view(n,1,3)), 1) #format delta R between boson candidates and mZH's for quadjet feature space
        va = va[:,:self.nAv].view(n,self.nAv,1) # |va|
        va = torch.cat( (va, va, va), 2) # |va|va|va|

        qs = []
        ps = []
        randomR = np.random.uniform(0,2.0/self.nR, self.nR) if self.training else np.zeros(self.nR)
        for i in range(self.nR):
            ps.append(self.rotate(p, self.R[i]+randomR[i]))
            qs.append(self.invPart(ps[-1], da, qa))
            if self.doFlip:
                ps.append(self.flipPhi(ps[-1])) #result has flipped phi only
                qs.append(self.invPart(ps[-1], da, qa))
                ps.append(self.flipEta(ps[-1])) #result has flipped phi and eta
                qs.append(self.invPart(ps[-1], da, qa))
                ps.append(self.flipEta(ps[-3])) #result has flipped eta only
                qs.append(self.invPart(ps[-1], da, qa))
                # ps.append(self.flipEta(ps[-2])) #result has flipped eta only
                # qs.append(self.invPart(ps[-1], da, qa))

        q = sum(qs)/self.nRF

        v = torch.cat( (q, va), 1) # manually add features to event view feature space
        v = self.toViewFeatureSpace(v)

        v0 = v.clone()
        v = SiLU(v)
        v = self.viewConv1(v)
        v = v+v0
        v = SiLU(v)
        v = self.viewConv2(v)
        v = v+v0
        v = SiLU(v)

        v = self.viewSelector(v)
        v = v.view(n, -1)
        return v
