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
        

    def forward(self, js):#j[event][jet][mu] l[event][nj]
        ls = (js[:,1,:]!=0).sum(dim=1) # count how many jets in each batch have pt > 0. pt==0 for padded entries
        idx = ls + torch.tensor(ls==0, dtype=torch.long).to("cuda") - 1 # add 1 to ls when there are no other jets to return the 
        js = torch.transpose(js,1,2) # switch jet and mu indices because LSTM expects jet index before jet component index

        batch_size, seq_len, feature_len = js.size()

        hs, _ = self.lstm(js)

        hs = hs.contiguous().view(batch_size*seq_len, self.nh)
        ran = torch.arange(0,n).to("cuda")
        idx = ran*seq_len + idx
        h = hs.index_select(0,idxs)
        h = h.view(batch_size,self.nh,1)
        return h



class MultiHeadAttention(nn.Module): # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec https://arxiv.org/pdf/1706.03762.pdf
    def __init__(self, heads, d_model, dropout = 0.1, selfAttention=False):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.selfAttention = selfAttention
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = None
        if dropout: self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def attention(self, q, k, v, mask=None, debug=False):
    
        scores = torch.matmul(q, k.transpose(-2, -1)) /  np.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            if self.selfAttention:
                scores = scores.masked_fill(mask == 0, -1e9)
            mask = mask.transpose(-2,-1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0)
            if self.selfAttention:
                mask = mask.transpose(-2,-1)
                scores = scores.masked_fill(mask == 0, 0)
        if debug:
            print("scores softmax\n",scores[0])
            print("v\n",v[0])

        if self.dropout:
            scores = self.dropout(scores)
        
        output = torch.matmul(scores, v)
        if debug:
            print("output\n",output[0])
            input()
        return output

    def forward(self, q, k, v, mask=None, qLinear=0, debug=False):
        
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * (d_model//h==d_k)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention 
        scores = self.attention(q, k, v, mask, debug)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output


class Norm(nn.Module): #https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#1b3f
    def __init__(self, d_model, eps = 1e-6, dim=-1):
        super().__init__()
        self.dim = dim
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        x=x.transpose(self.dim,-1)
        x = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        x=x.transpose(self.dim,-1)
        return x


class encoder(nn.Module):
    def __init__(self, inFeatures, hiddenFeatures, outFeatures, dropout):
        super(encoder, self).__init__()
        self.ni = inFeatures
        self.nh = hiddenFeatures
        self.no = outFeatures
        self.d = dropout

        self.input = nn.Linear(self.ni, self.nh)
        self.dropout = nn.Dropout(self.d)
        self.output = nn.Linear(self.nh, self.no)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = NonLU(x)
        x = self.output(x)
        return x

class transformer(nn.Module): # Attention is All You Need https://arxiv.org/pdf/1706.03762.pdf
    def __init__(self, features):
        self.nf = features
        self.encoderSelfAttention = MultiHeadAttention(1, self.nf, selfAttention=True)
        self.normEncoderSelfAttention = Norm(self.nf)
        self.encoderFF = encoder(self.nf, self.nf*2)#*4 in the paper
        self.normEncoderFF = Norm(self.nf)

        self.decoderSelfAttention = MultiHeadAttention(1, self.nf, selfAttention=True)
        self.normDecoderSelfAttention = Norm(self.nf)
        self.decoderAttention = MultiHeadAttention(1, self.nf)
        self.normDecoderAttention = Norm(self.nf)
        self.decoderFF = encoder(self.nf, self.nf*2)#*4 in the paper
        self.normDecoderFF = Norm(self.nf)

    def forward(self, inputs, mask, outputs):
        #encoder block
        inputs = self.normEncoderSelfAttention( inputs + self.encoderSelfAttention(inputs, inputs, inputs, mask=mask) )
        inputs = self.normEncoderFF( inputs + self.encoderFF(inputs) )
        
        #decoder block
        outputs = self.normDecoderSelfAttention( outputs + self.decoderSelfAttention(outputs, outputs, outputs) )
        outputs = self.normDecoderAttention( outputs + self.decoderAttention(outputs, inputs, inputs, mask=mask) )
        outputs = self.normDecoderFF( outputs + self.decoderFF(outputs) )
        
        return outputs
        

class multijetAttention(nn.Module):
    def __init__(self, jetFeatures, embedFeatures, nh=1):
        super(multijetAttention, self).__init__()
        self.nj = jetFeatures
        self.ne = embedFeatures
        self.nh = nh

        #self.selfAttention1 = MultiHeadAttention(self.nh, self.ne, dropout=None, selfAttention=True)
        self.attention1 = MultiHeadAttention(self.nh, self.ne, dropout=None)
        #self.selfAttention2 = MultiHeadAttention(self.nh, self.ne, dropout=None, selfAttention=True)
        self.attention2 = MultiHeadAttention(self.nh, self.ne, dropout=None)
        
    def forward(self, q, kv, mask, qLinear=0, debug=False):
        batch_size, _, seq_len = kv.size()

        mask = mask.unsqueeze(1)
        mask = mask.transpose(1,2)

        q = q.transpose(1,2)
        kv= kv.transpose(1,2) # switch jet and mu indices because attention model expects sequence item index before item component index [batch,pixel,feature]
        if debug:
            print("q\n", q[0])        
            print("kv\n", kv[0])
            print("mask\n",mask[0])

        q = q + self.attention1(q,  kv, kv, mask, debug=debug)
        q = q + self.attention2(q,  kv, kv, mask, debug=debug)

        q = q.transpose(1,2) #switch back to [event, feature, jet] matrix for convolutions
        
        return q


class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures, bottleneck=False, useOthJets=False, swapJets=False):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        self.ks = 4 if useOthJets else 3
        self.nx = 2#self.ks-1 
        self.swapJets = swapJets
        self.reinforce = not self.swapJets
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

    def forward(self, x, d):#, o=None):
        n = x.shape[0]
        # if o is not None:
        #     d = torch.cat( (x[:,:, self.nx*0: self.nx*1], d[:,:,0].view(n, self.nd, 1), o[:,:,0].view(n, self.nd, 1),
        #                     x[:,:, self.nx*1: self.nx*2], d[:,:,1].view(n, self.nd, 1), o[:,:,1].view(n, self.nd, 1),
        #                     x[:,:, self.nx*2: self.nx*3], d[:,:,2].view(n, self.nd, 1), o[:,:,2].view(n, self.nd, 1),
        #                     x[:,:, self.nx*3: self.nx*4], d[:,:,3].view(n, self.nd, 1), o[:,:,3].view(n, self.nd, 1),
        #                     x[:,:, self.nx*4: self.nx*5], d[:,:,4].view(n, self.nd, 1), o[:,:,4].view(n, self.nd, 1),
        #                     x[:,:, self.nx*5: self.nx*6], d[:,:,5].view(n, self.nd, 1), o[:,:,5].view(n, self.nd, 1)), 2 )
        # else:
        if self.reinforce:
            d = torch.cat( (x[:,:, self.nx*0: self.nx*1], d[:,:,0].view(n, self.nd, 1),
                            x[:,:, self.nx*1: self.nx*2], d[:,:,1].view(n, self.nd, 1),
                            x[:,:, self.nx*2: self.nx*3], d[:,:,2].view(n, self.nd, 1),
                            x[:,:, self.nx*3: self.nx*4], d[:,:,3].view(n, self.nd, 1),
                            x[:,:, self.nx*4: self.nx*5], d[:,:,4].view(n, self.nd, 1),
                            x[:,:, self.nx*5: self.nx*6], d[:,:,5].view(n, self.nd, 1)), 2 )

        if self.swapJets:
            d = torch.cat( (x[:,:, self.nx*1: self.nx*2], d[:,:,0].view(n, self.nd, 1),
                            x[:,:, self.nx*0: self.nx*1], d[:,:,1].view(n, self.nd, 1),
                            x[:,:, self.nx*3: self.nx*4], d[:,:,2].view(n, self.nd, 1),
                            x[:,:, self.nx*2: self.nx*3], d[:,:,3].view(n, self.nd, 1),
                            x[:,:, self.nx*5: self.nx*6], d[:,:,4].view(n, self.nd, 1),
                            x[:,:, self.nx*4: self.nx*5], d[:,:,5].view(n, self.nd, 1)), 2 )

        if self.compress: 
            d = self.compress(d)
            d = NonLU(d)

        d = self.conv(d)

        if self.expand:
            d = NonLU(d)
            d = self.expand(d)

        return d


class dijetResNetBlock(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, swapJets=False, bottleneck=False, useOthJets='', nOtherJetFeatures=4, device='cuda'):
        super(dijetResNetBlock, self).__init__()
        self.nj = jetFeatures
        self.nd = dijetFeatures
        self.device = device
        self.update0 = False

        self.reinforce1 = dijetReinforceLayer(self.nd, swapJets=swapJets) #bottleneck)#, useOthJets)
        self.convJ1 = nn.Conv1d(self.nd, self.nd, 1)
        self.reinforce2 = dijetReinforceLayer(self.nd, swapJets=swapJets) #bottleneck)#, useOthJets)

        self.multijetAttention = None
        if useOthJets:
            self.inConvO0 = nn.Conv1d(      5, self.nd, 1)
            self.inConvO1 = nn.Conv1d(self.nd, self.nd, 1)
            self.inConvO2 = nn.Conv1d(self.nd, self.nd, 1)

            nhOptions = []
            for i in range(1,self.nd+1):
                if (self.nd%i)==0: nhOptions.append(i)
            print("possible values of multiHeadAttention nh:",nhOptions)
            self.multijetAttention = multijetAttention(5, self.nd, nh=nhOptions[1])

    def forward(self, j, d, j0=None, d0=None, o=None, mask=None, debug=False):
        if d0 is None:
            d0 = d.clone()

        d = self.reinforce1(j, d)
        j = self.convJ1(j)
        d = d+d0
        j = j+j0
        d = NonLU(d)
        j = NonLU(j)

        d = self.reinforce2(j, d)
        d = d+d0
        d = NonLU(d)

        if self.multijetAttention:
            n, features, jets = o.shape
            o = self.inConvO0(o)
            o = NonLU(o)
            o0= o.clone()
            o = self.inConvO1(o)
            o = NonLU(o+o0)
            o = self.inConvO2(o)
            o = NonLU(o+o0)
            d = self.multijetAttention(d, o, mask, debug=debug)

        return j, d, o


class quadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures, bottleneck=False, useOthJets=False):
        super(quadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures
        self.ks = 4 if useOthJets else 3

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.conv = nn.Conv1d(bottleneck if bottleneck else self.nq, bottleneck if bottleneck else self.nq, self.ks, stride=self.ks)

        self.compress = None
        self.expand   = None
        if bottleneck: 
            self.compress = nn.Conv1d(self.nq, bottleneck, 1)
            self.expand   = nn.Conv1d(bottleneck, self.nq, 1)

    def forward(self, x, q):#, o):
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
    def __init__(self, dijetFeatures, quadjetFeatures, bottleneck=False, useOthJets=False, device='cuda'):
        super(quadjetResNetBlock, self).__init__()
        self.nd = dijetFeatures
        self.nq = quadjetFeatures
        self.device = device
        self.update0 = False

        self.reinforce1 = quadjetReinforceLayer(self.nq, bottleneck)#, useOthJets)
        self.convD1 = nn.Conv1d(self.nq, self.nq, 1)
        self.reinforce2 = quadjetReinforceLayer(self.nq, bottleneck)#, useOthJets)

        self.multijetAttention = None
        if useOthJets:
            self.inConvO0 = nn.Conv1d(5, self.nq, 1)
            self.inConvO1 = nn.Conv1d(self.nq, self.nq, 1)
            self.inConvO2 = nn.Conv1d(self.nq, self.nq, 1)
            self.multijetAttention = multijetAttention(5, self.nq, nh=2)

    def forward(self, d, q, d0=None, q0=None, o=None, mask=None, debug=False):
        if q0 is None:
            q0 = q.clone()

        q = self.reinforce1(d, q)
        d = self.convD1(d)
        q = q+q0
        d = d+d0
        q = NonLU(q)
        d = NonLU(d)

        q = self.reinforce2(d, q)
        q = q+q0
        q = NonLU(q)
            
        if self.multijetAttention:
            n, features, jets = o.shape
            o0 = torch.cat( (o.clone(), torch.zeros(n,self.nd-features,jets).to(self.device)), 1)
            o = self.inConvO0(o)
            o = NonLU(o+o0)
            o = self.inConvO1(o)
            o = NonLU(o+o0)
            o = self.inConvO2(o)
            o = NonLU(o+o0)

            q = self.multijetAttention(q, o, mask, debug=debug)

        return d, q


class ResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillaryFeatures, useOthJets='', device='cuda'):
        super(ResNet, self).__init__()
        self.debug = False
        self.nj = jetFeatures
        self.nd, self.nAd = dijetFeatures, 2 #total dijet features, engineered dijet features
        self.nq, self.nAq = quadjetFeatures, 2 #total quadjet features, engineered quadjet features
        self.nAv = nAncillaryFeatures
        self.nAv = 2
        self.nc = combinatoricFeatures
        self.device = device
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
        self.convJ0 = nn.Conv1d(self.nj, self.nd, 1)
        # ancillary dijet features get appended to output of dijetBuilder

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock  = dijetResNetBlock(self.nj, self.nd, device=self.device, useOthJets=useOthJets)
        # self.convJ1 = nn.Conv1d(self.nj, self.nd, 1)
        # self.dijetResNetBlock2 = dijetResNetBlock(self.nj, self.nd, device=self.device, swapJets=True, useOthJets=useOthJets)

        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = nn.Conv1d(self.nd, self.nq-self.nAq, 2, stride=2)
        self.convD0 = nn.Conv1d(self.nd, self.nq, 1)
        # ancillary quadjet features get appended to output of quadjetBuilder

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nd, self.nq, device=self.device)

        self.combiner = nn.Conv1d(self.nq, self.nc-self.nAv, 3)
        #self.combiner2 = nn.Conv1d(self.nq, self.nc, 3)
        self.viewConv1 = nn.Conv1d(self.nc, self.nc, 1)
        self.viewConv2 = nn.Conv1d(self.nc, self.nc, 1)
        self.out = nn.Conv1d(self.nc, 1, 1)

    def rotate(self, j, R): # j[event, mu, jet], mu=2 is phi
        jR = j.clone()
        jR[:,2,:] = (jR[:,2,:] + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
        return jR

    def flipPhi(self, j): # j[event, mu, jet], mu=2 is phi
        jF = j.clone()
        jF[:,2,:] = -1*jF[:,2,:]
        return jF

    def flipEta(self, j): # j[event, mu, jet], mu=1 is eta
        jF = j.clone()
        jF[:,1,:] = -1*jF[:,1,:]
        return jF

    def invPart(self, j, o, mask, da, qa, va):
        n = j.shape[0]
        jRaw = j.clone()
            
        d = self.dijetBuilder(j)
        d = NonLU(d)
        d = torch.cat( (d, da), 1 ) # manually add dijet mass and dRjj to dijet feature space
        d0 = d.clone()

        j = self.convJ0(jRaw)
        j = NonLU(j)
        j0= j.clone()

        j, d, _ = self.dijetResNetBlock( j, d, j0=j0, d0=d0, o=o, mask=mask, debug=self.debug)
        # d0 = d.clone() #update d0
        # j = self.convJ1(jRaw) #restart jet convolutions from input 4-vectors
        # j = NonLU(j)
        # j0= j.clone()
        # _, d, _ = self.dijetResNetBlock2(j, d, j0=j0, d0=d0, o=o, mask=mask, debug=self.debug)

        q = self.quadjetBuilder(d)
        q = NonLU(q)
        q = torch.cat( (q, qa), 1) # manually add features to quadjet feature space
        q0 = q.clone()

        if self.nd < self.nq:
            d0 = torch.cat( (d.clone(), torch.zeros(n,self.nq-self.nd,6).to(self.device)), 1)
        elif self.nd == self.nq:
            d0 = d.clone()

        if self.nd <= self.nq:
            d = NonLU(self.convD0(d)+d0)
        else:
            d = self.convD0(d)
            d0= d.clone()
            d = NonLU(d)

        d, q = self.quadjetResNetBlock(d, q, d0=d0, q0=q0, o=o, mask=mask, debug=self.debug) 

        return q

    def forward(self, x, j, o, da, qa, va):
        n = j.shape[0]

        da = da.view(n,self.nAd,6)
        qa = qa.view(n,self.nAq,3)
        if self.nAv:
            #va = va[:,1:].view(n,self.nAv,1) # |va|
            va = va.view(n,self.nAv,1) # |va|

        mask = o[:,4,:]!=-1

        if self.training: #random permutation
            # c = torch.randperm(3)
            # j = j.view(n,-1,3,4)[:,:,c,:].view(n,-1,12)
            # qa = qa[:,:,c]
            # c = torch.cat( (0+torch.randperm(2), 2+torch.randperm(2), 4+torch.randperm(2)), 0)
            # j = j.view(n,-1,6,2)[:,:,c,:].view(n,-1,12)
            # da = da[:,:,c]
            nPermutationChunks=5
            cn=n//nPermutationChunks
            r =n%cn
            for i in range(nPermutationChunks):
                l = i*cn
                u = (i+1)*cn + (r if i+1==nPermutationChunks else 0)

                c = torch.randperm(3)
                j [l:u] = j [l:u].view(u-l,-1,3,4)[:,:,c,:].view(u-l,-1,12)
                qa[l:u] = qa[l:u,:,c]

                c = torch.cat( (0+torch.randperm(2), 2+torch.randperm(2), 4+torch.randperm(2)), 0)
                j [l:u] = j [l:u].view(u-l,-1,6,2)[:,:,c,:].view(u-l,-1,12)
                da[l:u] = da[l:u,:,c]


        js, os, qs = [], [], []
        randomR = np.random.uniform(0,2.0/self.nR, self.nR) if self.training else np.zeros(self.nR)
        for i in range(self.nR):
            js.append(self.rotate(j, self.R[i]+randomR[i]))
            os.append(self.rotate(o, self.R[i]+randomR[i]))
            qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))
            if self.doFlip:
                #flip phi of original
                js.append(self.flipPhi(js[-1]))
                os.append(self.flipPhi(os[-1]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

                #flip phi and eta of original
                js.append(self.flipEta(js[-1]))
                os.append(self.flipEta(os[-1]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

                #flip eta of original
                js.append(self.flipEta(js[-3]))
                os.append(self.flipEta(os[-3]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, va))

        q = sum(qs)/self.nRF # average over rotations and flips

        v = self.combiner(q)
        if self.nAv:
            v = torch.cat( (v, va), 1) # manually add features to event view feature space
        v0= v.clone()
        #v = NonLU(v)
        #v = self.combiner2(q)
        #v = v+v0
        #v = NonLU(v)

        v = self.viewConv1(v)
        v = v+v0
        v = NonLU(v)
        v = self.viewConv2(v)
        v = v+v0
        v = NonLU(v)
        v = self.out(v)
        v = v.view(n, -1)
        return v

