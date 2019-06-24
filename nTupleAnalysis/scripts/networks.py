import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Lin_View(nn.Module):
    def __init__(self):
        super(Lin_View, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

def ReLU(x):
    return F.relu(x)

def SiLU(x): #SiLU https://arxiv.org/pdf/1702.03118.pdf   Swish https://arxiv.org/pdf/1710.05941.pdf
    return x * torch.sigmoid(x)

def NonLU(x): # Non-Linear Unit
    return ReLU(x)
    #return SiLU(x)

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


class jetLSTM(nn.Module):
    def __init__(self, jetFeatures, hiddenFeatures):
        super(jetLSTM, self).__init__()
        self.nj = jetFeatures
        self.nh = hiddenFeatures
        self.lstm = nn.LSTM(self.nj, self.nh, num_layers=2, batch_first=True)
        #self.lstm0 = nn.LSTM(self.nj, self.nh, num_layers=1, batch_first=True)
        #self.lstm1 = nn.LSTM(self.nh, self.nh, num_layers=1, batch_first=True)
        

    def forward(self, js):#j[event][jet][mu] l[event][nj]
        ls = (js[:,1,:]!=0).sum(dim=1) # count how many jets in each batch have pt > 0. pt==0 for padded entries
        idx = ls + torch.tensor(ls==0, dtype=torch.long).to("cuda") - 1 # add 1 to ls when there are no other jets to return the 
        js = torch.transpose(js,1,2) # switch jet and mu indices because LSTM expects jet index before jet component index

        batch_size, seq_len, feature_len = js.size()

        hs, _ = self.lstm(js)

        # hs, _ = self.lstm0(js)
        # hs0 = hs.clone()
        # #hs = NonLU(hs) #lstm already has a tanh nonlinearity at the output
        # hs, _ = self.lstm1(hs)
        # hs = hs0+hs

        hs = hs.contiguous().view(batch_size*seq_len, self.nh)
        #idxs = [i*seq_len + l-1 for i,l in enumerate(ls)]
        ran = torch.arange(0,n).to("cuda")
        idx = ran*seq_len + idx
        #idxs = torch.tensor(idxs, dtype=torch.int64).to("cuda")
        h = hs.index_select(0,idxs)
        h = h.view(batch_size,self.nh,1)
        return h



class MultiHeadAttention(nn.Module): # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec https://arxiv.org/pdf/1706.03762.pdf
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = None
        if dropout: self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def attention(self, q, k, v, mask=None):
    
        scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            #print("mask\n",mask[0])
            #print(mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9)
            mask = mask.transpose(-2,-1)
            scores = scores.masked_fill(mask == 0, -1e9)

        #print("scores\n",scores[0])
        #print(scores.shape)
        scores = F.softmax(scores, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0)
            mask = mask.transpose(-2,-1)
            scores = scores.masked_fill(mask == 0, 0)
        #print("scores softmax\n",scores[0])
        #print("v\n",v[0])
        if self.dropout:
            scores = self.dropout(scores)
        
        output = torch.matmul(scores, v)
        #print("output",output[0])
        #input()
        return output

    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        #print(q.shape, q[0])
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * (d_model//h==d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention 
        scores = self.attention(q, k, v, mask)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
        #print(output[0])
    
        return output


class Norm(nn.Module): #https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#1b3f
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class multijetAttention(nn.Module):
    def __init__(self, jetFeatures, embedFeatures):
        super(multijetAttention, self).__init__()
        self.nj = jetFeatures
        self.ne = embedFeatures

        self.embed = nn.Conv1d(self.nj, self.ne, 1)
        self.conv1 = nn.Conv1d(self.ne, self.ne, 1)
        self.conv2 = nn.Conv1d(self.ne, self.ne, 1)
        #self.norm1 = Norm(self.ne)
        self.attention = MultiHeadAttention(4, self.ne, None)
        
        internalNodes=16
        #self.norm2 = Norm(self.ne)
        self.linear1 = nn.Linear(self.ne, internalNodes)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(internalNodes, self.ne)
        
        self.maxPool = nn.MaxPool1d(7)

    def forward(self, js):
        batch_size, seq_len, _ = js.size()

        mask = (js[:,4,:]!=-1).unsqueeze(1)
        mask = torch.transpose(mask,1,2)

        #ls = (js[:,4,:]!=-1).sum(dim=1) # count how many jets in each batch have isSelJet != -1. isSelJet==-1 for padded entries
        #idx = ls + torch.tensor(ls==0, dtype=torch.long).to("cuda") - 1 # add 1 to ls when there are no other jets

        js = self.embed(js)
        js0 = js.clone()
        js = self.conv1(js)
        js = NonLU(js+js0)
        js = self.conv2(js)
        js = NonLU(js+js0)
        js = torch.transpose(js,1,2) # switch jet and mu indices because attention model expects sequence item index before item component index

        #js = self.norm1(js)
        #print("idx",idx[0])
        #print("input js\n",js[0])
        #js = js + self.attention(js, js, js, mask) #dropout on output of attention?
        js = self.attention(js, js, js, mask)
        #print("attention js\n",js[0])
        #js = self.norm1(js + self.attention(js, js, js, mask))
        
        # print("ls\n",ls[0])
        # ls = ls + torch.tensor(ls==0, dtype=torch.long).to("cuda")
        # print("js\n",js[0])
        # j = js.sum(1)/ls.unsqueeze(1).float()
        # print("j\n",j[0])
        # input()

        # js = js.view(batch_size*seq_len, self.ne)
        # ran = torch.arange(0,batch_size).to("cuda")
        # idx = ran*seq_len + idx
        # j = js.index_select(0,idx)
        # j = j.view(batch_size, self.ne)
        #print("j\n",j[0])
        #input()

        #j = self.norm2(j)
        js0 = js.clone()
        js = self.linear1(js)
        js = NonLU(js)
        js = self.dropout(js)
        js = self.linear2(js)
        js = js + js0
        #js = self.norm2(js+js0)

        # #print("js\n",js[0])
        # #print("mask\n",mask[0])
        # js = js.masked_fill(mask == 0, -1e9)#make sure padding never passes the maxpool
        # #print("masked js\n",js[0])
        # js = js.transpose(1,2)
        # j = self.maxPool(js) # maximize the features over the valid jets
        # j = j.masked_fill(j==-1e9, -1) # if negative billion makes it through because there are no valid jets, replace with -1
        # j = j.view(batch_size,self.ne,1)        

        #print("j\n",j[0])
        #input()

        j = js[:,0,:]


        return j


class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures, bottleneck=False, useOthJets=False):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        self.ks = 4 if useOthJets else 3
        self.nx = 2#self.ks-1 
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|            

        # if we are using other jet info:
        # |1|2|o|1,2|3|4|o|3,4|1|3|o|1,3|2|4|o|2,4|1|4|o|1,4|2|3|o|2,3|  ##stride=4 kernel=4 reinforce dijet features with output of other jet LSTM
        #       |1,2|     |3,4|     |1,3|     |2,4|     |1,4|     |2,3| 
        self.conv = nn.Conv1d(bottleneck if bottleneck else self.nd, bottleneck if bottleneck else self.nd, self.ks, stride=self.ks)

        self.compress = None
        self.expand   = None
        if bottleneck: 
            self.compress = nn.Conv1d(self.nd, bottleneck, 1)
            self.expand   = nn.Conv1d(bottleneck, self.nd, 1)

    def forward(self, x, d, o=None):
        n = x.shape[0]
        if o is not None:
            # d = torch.cat( (x[:,:, self.nx*0: self.nx*1], o, d[:,:,0].view(n, self.nd, 1),
            #                 x[:,:, self.nx*1: self.nx*2], o, d[:,:,1].view(n, self.nd, 1),
            #                 x[:,:, self.nx*2: self.nx*3], o, d[:,:,2].view(n, self.nd, 1),
            #                 x[:,:, self.nx*3: self.nx*4], o, d[:,:,3].view(n, self.nd, 1),
            #                 x[:,:, self.nx*4: self.nx*5], o, d[:,:,4].view(n, self.nd, 1),
            #                 x[:,:, self.nx*5: self.nx*6], o, d[:,:,5].view(n, self.nd, 1)), 2 )
            d = torch.cat( (o, x[:,:, self.nx*0: self.nx*1], d[:,:,0].view(n, self.nd, 1),
                            o, x[:,:, self.nx*1: self.nx*2], d[:,:,1].view(n, self.nd, 1),
                            o, x[:,:, self.nx*2: self.nx*3], d[:,:,2].view(n, self.nd, 1),
                            o, x[:,:, self.nx*3: self.nx*4], d[:,:,3].view(n, self.nd, 1),
                            o, x[:,:, self.nx*4: self.nx*5], d[:,:,4].view(n, self.nd, 1),
                            o, x[:,:, self.nx*5: self.nx*6], d[:,:,5].view(n, self.nd, 1)), 2 )
        else:
            d = torch.cat( (x[:,:, self.nx*0: self.nx*1], d[:,:,0].view(n, self.nd, 1),
                            x[:,:, self.nx*1: self.nx*2], d[:,:,1].view(n, self.nd, 1),
                            x[:,:, self.nx*2: self.nx*3], d[:,:,2].view(n, self.nd, 1),
                            x[:,:, self.nx*3: self.nx*4], d[:,:,3].view(n, self.nd, 1),
                            x[:,:, self.nx*4: self.nx*5], d[:,:,4].view(n, self.nd, 1),
                            x[:,:, self.nx*5: self.nx*6], d[:,:,5].view(n, self.nd, 1)), 2 )

        if self.compress: 
            d = self.compress(d)
            d = NonLU(d)

        d = self.conv(d)

        if self.expand:
            d = NonLU(d)
            d = self.expand(d)

        return d


class dijetResNetBlock(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, bottleneck=False, useOthJets=''):
        super(dijetResNetBlock, self).__init__()
        self.nj = jetFeatures
        self.nd = dijetFeatures

        self.convX0 = nn.Conv1d(self.nj, self.nd, 1)
        self.otherJets = None
        if useOthJets:
            if useOthJets == 'jetLSTM':
                self.otherJets = jetLSTM(self.nj+1,self.nd)
            if useOthJets == 'multijetAttention':
                self.otherJets = multijetAttention(self.nj+1, self.nd)
            #self.convO1 = nn.Conv1d(self.nd, self.nd, 1)
        self.reinforce1 = dijetReinforceLayer(self.nd, bottleneck, useOthJets)
        self.convX1 = nn.Conv1d(self.nd, self.nd, 1)
        self.reinforce2 = dijetReinforceLayer(self.nd, bottleneck, useOthJets)

    def forward(self, x, d, o=None):
        d0 = d.clone()

        x = self.convX0(x)
        x0 = x.clone()
        x = NonLU(x)

        if self.otherJets:
            n = x.shape[0]
            o = self.otherJets(o)
            o = o.view(n,self.nd,1)        
            # o0 = o.clone()
            # o = NonLU(o)
            # x  = torch.cat( (o , x [:,:,0:2], o , x [:,:,2:4], o , x [:,:,4:6], o , x [:,:,6:8], o , x [:,:,8:10], o , x [:,:,10:12]), 2)
            # x0 = torch.cat( (o0, x0[:,:,0:2], o0, x0[:,:,2:4], o0, x0[:,:,4:6], o0, x0[:,:,6:8], o0, x0[:,:,8:10], o0, x0[:,:,10:12]), 2)
            # x  = torch.cat( (x [:,:,0:2], o , x [:,:,2:4], o , x [:,:,4:6], o , x [:,:,6:8], o , x [:,:,8:10], o , x [:,:,10:12], o ), 2)
            # x0 = torch.cat( (x0[:,:,0:2], o0, x0[:,:,2:4], o0, x0[:,:,4:6], o0, x0[:,:,6:8], o0, x0[:,:,8:10], o0, x0[:,:,10:12], o0), 2)
            #x = torch.cat( (o, x[:,:,0:2], o, x[:,:,2:4], o, x[:,:,4:6], o, x[:,:,6:8], o, x[:,:,8:10], o, x[:,:,10:12]), 2)
            
        # x0 = x.clone()
        # x = NonLU(x)

        d = self.reinforce1(x, d, o)
        d = NonLU(d+d0)

        x = self.convX1(x)
        x = NonLU(x+x0)
        # if self.otherJets:
        #     o = self.convX1(o)
        #     o = NonLU(o+o0)
        d = self.reinforce2(x, d, o)
        d = d+d0

        return d


class quadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures, bottleneck=False):
        super(quadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.conv = nn.Conv1d(bottleneck if bottleneck else self.nq, bottleneck if bottleneck else self.nq, 3, stride=3)

        self.compress = None
        self.expand   = None
        if bottleneck: 
            self.compress = nn.Conv1d(self.nq, bottleneck, 1)
            self.expand   = nn.Conv1d(bottleneck, self.nq, 1)

    def forward(self, x, q):
        n = x.shape[0]
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4:6], q[:,:,2].view(n,self.nq,1)), 2)

        if self.compress: 
            q = self.compress(q)
            q = NonLU(q)

        q = self.conv(q)

        if self.expand:
            q = NonLU(q)
            q = self.expand(q)

        return q


class quadjetResNetBlock(nn.Module):
    def __init__(self, dijetFeatures, quadjetFeatures, bottleneck=False):
        super(quadjetResNetBlock, self).__init__()
        self.nd = dijetFeatures
        self.nq = quadjetFeatures

        self.convX0 = nn.Conv1d(self.nd, self.nq, 1)
        self.reinforce1 = quadjetReinforceLayer(self.nq, bottleneck)
        self.convX1 = nn.Conv1d(self.nq, self.nq, 1)
        self.reinforce2 = quadjetReinforceLayer(self.nq, bottleneck)

    def forward(self, x, q):
        q0 = q.clone()

        x = self.convX0(x) 
        x0 = x.clone()
        x = NonLU(x)
        q = self.reinforce1(x, q)
        q = NonLU(q+q0)

        x = self.convX1(x)
        x = NonLU(x+x0)
        q = self.reinforce2(x, q)
        q = q+q0

        return q


class eventReinforceLayer(nn.Module):
    def __init__(self, eventFeatures):
        super(eventReinforceLayer, self).__init__()
        self.ne = eventFeatures
        # |1,2,3,4|1,3,2,4|1,4,2,3|e|
        #                         |e|
        self.conv = nn.Conv1d(self.ne, self.ne, 4)

    def forward(self, x, e):
        n = x.shape[0]
        e = torch.cat( (x, e), 2)
        return self.conv(e)


class eventResNetBlock(nn.Module):
    def __init__(self, quadjetFeatures, eventFeatures):
        super(eventResNetBlock, self).__init__()
        self.nq = quadjetFeatures
        self.ne = eventFeatures

        self.convX0 = nn.Conv1d(self.nq, self.ne, 1)
        self.reinforce1 = eventReinforceLayer(self.ne)
        self.convX1 = nn.Conv1d(self.ne, self.ne, 1)
        self.reinforce2 = eventReinforceLayer(self.ne)

    def forward(self, x, e):
        e0 = e.clone()

        x = self.convX0(x) 
        x0 = x.clone()
        x = NonLU(x)
        e = self.reinforce1(x, e)
        e = NonLU(e+e0)

        x = self.convX1(x)
        x = NonLU(x+x0)
        e = self.reinforce2(x, e)
        e = NonLU(e+e0)

        return e
        

class ResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillaryFeatures, useOthJets=''):
        super(ResNet, self).__init__()
        self.nj = jetFeatures
        self.nd, self.nAd = dijetFeatures, 2 #total dijet features, engineered dijet features
        self.nq, self.nAq = quadjetFeatures, 2 #total quadjet features, engineered quadjet features
        self.nAv = nAncillaryFeatures
        #self.nAv = 0
        self.nc = self.nq+self.nAv
        #self.nc = combinatoricFeatures
        dijetBottleneck   = None
        quadjetBottleneck = None#6
        self.name = 'ResNet'+('+'+useOthJets if useOthJets else '')+'_%d_%d_%d'%(dijetFeatures, quadjetFeatures, self.nc)

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = nn.Conv1d(self.nj, self.nd-self.nAd, 2, stride=2)
        # ancillary dijet features get appended to output of dijetBuilder

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock = dijetResNetBlock(self.nj, self.nd, dijetBottleneck, useOthJets)

        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = nn.Conv1d(self.nd, self.nq-self.nAq, 2, stride=2)
        # ancillary quadjet features get appended to output of quadjetBuilder

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nd, self.nq, quadjetBottleneck)
        # ancillary view features get appended to output of quadjetResNetBlock

        #self.viewConv1 = nn.Conv1d(self.nc, self.nc, 1)
        #self.viewConv2 = nn.Conv1d(self.nc, self.nc, 1)
        #self.viewSelector = nn.Conv1d(self.nc, 1, 3)
        #self.viewSelector = nn.Sequential(*[nn.MaxPool1d(3), nn.Conv1d(self.nc, 1, 1)])

        #self.combiner = nn.MaxPool1d(3)
        self.combiner = nn.Conv1d(self.nq, self.nq, 3)
        self.viewConv1 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewConv2 = nn.Conv1d(self.nc, self.nc, 1)
        self.out = nn.Conv1d(self.nc, 1, 1)

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

    def invPart(self,p,o,da,qa):
        n = p.shape[0]

        d = self.dijetBuilder(p)
        d = NonLU(d)
        d = torch.cat( (d, da), 1 ) # manually add dijet mass and dRjj to dijet feature space
        d = self.dijetResNetBlock(p,d,o)
        
        q = self.quadjetBuilder(d)
        q = NonLU(q)
        q = torch.cat( (q, qa), 1) # manually add features to quadjet feature space
        q = self.quadjetResNetBlock(d,q) 
        return q

    def forward(self, x, p, o, da, qa, va):#, js, ls):
        n = p.shape[0]

        da = torch.cat( (da[:,0:6].view(n,1,6), da[:,6:12].view(n,1,6)), 1) #format dijet masses and dRjjs 
        qa = torch.cat( (qa[:,0:3].view(n,1,3), qa[:,3: 6].view(n,1,3)), 1) #format delta R between boson candidates and mZH's for quadjet feature space
        va = va[:,:self.nAv].view(n,self.nAv,1) # |va|
        #va = torch.cat( (va, va, va), 2) # |va|va|va|

        ps, os, qs = [], [], []
        randomR = np.random.uniform(0,2.0/self.nR, self.nR) if self.training else np.zeros(self.nR)
        for i in range(self.nR):
            ps.append(self.rotate(p, self.R[i]+randomR[i]))
            os.append(self.rotate(o, self.R[i]+randomR[i]))
            qs.append(self.invPart(ps[-1], os[-1], da, qa))
            if self.doFlip:
                #flip phi of original
                ps.append(self.flipPhi(ps[-1]))
                os.append(self.flipPhi(os[-1]))
                qs.append(self.invPart(ps[-1], os[-1], da, qa))

                #flip phi and eta of original
                ps.append(self.flipEta(ps[-1]))
                os.append(self.flipEta(os[-1]))
                qs.append(self.invPart(ps[-1], os[-1], da, qa))

                #flip eta of original
                ps.append(self.flipEta(ps[-3]))
                os.append(self.flipEta(os[-3]))
                qs.append(self.invPart(ps[-1], os[-1], da, qa))

        q = sum(qs)/self.nRF # average over rotations and flips

        q = NonLU(q)
        #v = torch.cat( (q, va), 1) # manually add features to event view feature space
        #v = q
        
        # v0 = v.clone()
        # v = self.viewConv1(v)
        # v = NonLU(v+v0)
        # v = self.viewConv2(v)
        # v = NonLU(v+v0)

        # v = self.viewSelector(v)
        # v = v.view(n, -1)
        # return v

        v = self.combiner(q)
        v = torch.cat( (v, va), 1) # manually add features to event view feature space
        v0 = v.clone()
        v = self.viewConv1(v)
        v = NonLU(v+v0)
        v = self.viewConv2(v)
        v = NonLU(v+v0)
        v = self.out(v)
        v = v.view(n, -1)
        return v


