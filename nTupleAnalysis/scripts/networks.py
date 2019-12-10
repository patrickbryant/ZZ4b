import collections
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

def NonLU(x, training=False): # Non-Linear Unit
    #return ReLU(x)
    #return F.rrelu(x, training=training)
    return SiLU(x)

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


class stats:
    def __init__(self):
        self.grad = collections.OrderedDict()
        self.mean = collections.OrderedDict()
        self. std = collections.OrderedDict()
        self.summary = ''

    def update(self,attr,grad):
        try:
            self.grad[attr] = torch.cat( (self.grad[attr], grad), dim=0)
        except (KeyError, TypeError):
            self.grad[attr] = grad.clone()

    def compute(self):
        self.summary = ''
        self.grad['combined'] = None
        for attr, grad in self.grad.items():
            try:
                self.grad['combined'] = torch.cat( (self.grad['combined'], grad), dim=1)
            except TypeError:
                self.grad['combined'] = grad.clone()

            self.mean[attr] = grad.mean(dim=0).norm()
            self. std[attr] = grad.std()
            #self.summary += attr+': <%1.1E> +/- %1.1E r=%1.1E'%(self.mean[attr],self.std[attr],self.mean[attr]/self.std[attr])
        self.summary = 'grad: <%1.1E> +/- %1.1E SNR=%1.1f'%(self.mean['combined'],self.std['combined'],(self.mean['combined']/self.std['combined']).log10())

    def dump(self):
        for attr, grad in self.grad.items():
            print(attr, grad.shape, grad.mean(dim=0).norm(2), grad.std())

    def reset(self):
        for attr in self.grad:
            self.grad[attr] = None
        

def make_hook(gradStats,module,attr):
    def hook(grad):
        gradStats.update(attr, grad/getattr(module,attr).norm(2))
    return hook


class conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, name=None, index=None, doGradStats=False):
        super(conv1d, self).__init__()
        self.module = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.name = name
        self.index = index
        self.gradStats = None
        self.k = 1.0/(in_channels * kernel_size)
        if doGradStats:
            self.gradStats = stats()
            self.module.weight.register_hook( make_hook(self.gradStats, self.module, 'weight') )
            #self.module.bias  .register_hook( make_hook(self.gradStats, self.module, 'bias'  ) )
    def randomize(self):
        nn.init.uniform_(self.module.weight, -(self.k**0.5), self.k**0.5)
        nn.init.uniform_(self.module.bias,   -(self.k**0.5), self.k**0.5)
    def forward(self,x):
        return self.module(x)


class linear(nn.Module):
    def __init__(self, in_channels, out_channels, name=None, index=None, doGradStats=False):
        super(linear, self).__init__()
        self.module = nn.Linear(in_channels, out_channels)
        self.name = name
        self.index = index
        self.gradStats = None
        self.k = 1.0/in_channels
        if doGradStats:
            self.gradStats = stats()
            self.module.weight.register_hook( make_hook(self.gradStats, self.module, 'weight') )
            #self.module.bias  .register_hook( make_hook(self.gradStats, self.module, 'bias'  ) )
    def randomize(self):
        nn.init.uniform_(self.module.weight, -(self.k**0.5), self.k**0.5)
        nn.init.uniform_(self.module.bias,   -(self.k**0.5), self.k**0.5)
    def forward(self,x):
        return self.module(x)


class layerOrganizer:
    def __init__(self):
        self.layers = collections.OrderedDict()
        self.nTrainableParameters = 0

    def addLayer(self, newLayer, inputLayers=None, startIndex=1):
        if inputLayers:
            inputIndicies = [layer.index for layer in inputLayers]
            newLayer.index = max(inputIndicies) + 1
        else:
            newLayer.index = startIndex

        try:
            self.layers[newLayer.index].append(newLayer)
        except (KeyError, AttributeError):
            self.layers[newLayer.index] = [newLayer]
            

    def countTrainableParameters(self):
        self.nTrainableParameters = 0
        for index in self.layers:
            for layer in self.layers[index]:
                for param in layer.parameters():
                    self.nTrainableParameters += param.numel() if param.requires_grad else 0

    def setLayerRequiresGrad(self, index, requires_grad=True):
        self.countTrainableParameters()
        print("Change trainable parameters from",self.nTrainableParameters, end=' ')
        try: # treat index as list of indices
            for i in index:
                for layer in self.layers[i]:
                    for param in layer.parameters():
                        param.requires_grad=requires_grad
        except TypeError: # index is just an int
            for layer in self.layers[index]:
                for param in layer.parameters():
                    param.requires_grad=requires_grad
        self.countTrainableParameters()
        print("to",self.nTrainableParameters)

    def initLayer(self, index):
        try: # treat index as list of indices
            print("Rerandomize layer indicies",index)
            for i in index:
                for layer in self.layers[i]:
                    layer.randomize()
        except TypeError: # index is just an int
            print("Rerandomize layer index",index)
            for layer in self.layers[index]:
                layer.randomize()

    def computeStats(self):
        for index in self.layers:
            for layer in self.layers[index]:
                layer.gradStats.compute()
    def resetStats(self):
        for index in self.layers:
            for layer in self.layers[index]:
                layer.gradStats.reset()

    def print(self):
        for index in self.layers:
            print("----- Layer %2d -----"%(index))
            for layer in self.layers[index]:
                print('|',layer.name.ljust(40), end='')
            print('')
            for layer in self.layers[index]:
                if layer.gradStats:
                    print('|',layer.gradStats.summary.ljust(40), end='')
                else:
                    print('|',' '*40, end='')
            print('')
            # for layer in self.layers[index]:
            #     print('|',str(layer.module).ljust(45), end=' ')
            # print('|')


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
    def __init__(self, heads, d_model, selfAttention=False, layers=None, inputLayers=None):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.selfAttention = selfAttention
        
        self.q_linear = linear(d_model, d_model, name='attention query linear') # nn.Linear(d_model, d_model)
        self.v_linear = linear(d_model, d_model, name='attention value linear') # nn.Linear(d_model, d_model)
        self.k_linear = linear(d_model, d_model, name='attention key   linear') # nn.Linear(d_model, d_model)
        self.o_linear = linear(d_model, d_model, name='attention out   linear') # nn.Linear(d_model, d_model)

        if layers:
            layers.addLayer(self.q_linear, inputLayers)
            layers.addLayer(self.v_linear, inputLayers)
            layers.addLayer(self.k_linear, inputLayers)
            layers.addLayer(self.o_linear, [self.q_linear, self.v_linear, self.k_linear])
    
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
        output = self.o_linear(concat)
    
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
    def __init__(self, inFeatures, hiddenFeatures, outFeatures, dropout, transpose=False):
        super(encoder, self).__init__()
        self.ni = inFeatures
        self.nh = hiddenFeatures
        self.no = outFeatures
        self.d = dropout
        self.transpose = transpose

        self.input = nn.Linear(self.ni, self.nh)
        self.dropout = nn.Dropout(self.d)
        self.output = nn.Linear(self.nh, self.no)

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1,2)
        x = self.input(x)
        x = self.dropout(x)
        x = NonLU(x, self.training)
        x = self.output(x)
        if self.transpose:
            x = x.transpose(1,2)
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
    def __init__(self, jetFeatures, embedFeatures, nh=1, layers=None, inputLayers=None):
        super(multijetAttention, self).__init__()
        self.nj = jetFeatures
        self.ne = embedFeatures
        self.nh = nh

        self.attention = MultiHeadAttention(self.nh, self.ne, layers=layers, inputLayers=inputLayers)
        
    def forward(self, q, kv, mask, q0=None, qLinear=0, debug=False):
        batch_size, _, seq_len = kv.size()

        mask = mask.unsqueeze(1)
        mask = mask.transpose(1,2)

        q = q.transpose(1,2)
        kv= kv.transpose(1,2) # switch jet and mu indices because attention model expects sequence item index before item component index [batch,pixel,feature]

        if debug:
            print("q\n", q[0])        
            print("kv\n", kv[0])
            print("mask\n",mask[0])

        if type(q0)==type(None):
            q0= q.clone()

        q = q0 + self.attention(q,  kv, kv, mask, debug=debug)
        q = NonLU(q, self.training)
        q = q0 + self.attention(q,  kv, kv, mask, debug=debug)
        q = NonLU(q, self.training)

        q = q.transpose(1,2) #switch back to [event, feature, jet] matrix for convolutions
        
        return q


class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|            
        self.conv = conv1d(self.nd, self.nd, 3, stride=3, name='dijet reinforce convolution') # nn.Conv1d(self.nd, self.nd, 3, stride=3)

    def forward(self, x, d):
        d = torch.cat( (x[:,:, 0: 2], d[:,:,0:1],
                        x[:,:, 2: 4], d[:,:,1:2],
                        x[:,:, 4: 6], d[:,:,2:3],
                        x[:,:, 6: 8], d[:,:,3:4],
                        x[:,:, 8:10], d[:,:,4:5],
                        x[:,:,10:12], d[:,:,5:6]), 2 )
        d = self.conv(d)
        return d


class dijetResNetBlock(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, useOthJets='', nOtherJetFeatures=4, device='cuda', layers=None, inputLayers=None):
        super(dijetResNetBlock, self).__init__()
        self.nj = jetFeatures
        self.nd = dijetFeatures
        self.device = device
        self.update0 = False

        self.reinforce1 = dijetReinforceLayer(self.nd)
        self.convJ1 = conv1d(self.nd, self.nd, 1, name='jet convolution 1') # nn.Conv1d(self.nd, self.nd, 1)
        self.reinforce2 = dijetReinforceLayer(self.nd)

        layers.addLayer(self.reinforce1.conv, inputLayers)
        layers.addLayer(self.convJ1, [inputLayers[0]])
        layers.addLayer(self.reinforce2.conv, [self.convJ1, self.reinforce1.conv])

        self.multijetAttention = None
        if useOthJets:
            self.inConvO0 = conv1d(      5, self.nd, 1, name='other jet convolution 0') # nn.Conv1d(      5, self.nd, 1)
            self.inConvO1 = conv1d(self.nd, self.nd, 1, name='other jet convolution 1') # nn.Conv1d(self.nd, self.nd, 1)
            self.inConvO2 = conv1d(self.nd, self.nd, 1, name='other jet convolution 2') # nn.Conv1d(self.nd, self.nd, 1)

            layers.addLayer(self.inConvO0)
            layers.addLayer(self.inConvO1, [self.inConvO0])
            layers.addLayer(self.inConvO2, [self.inConvO1])

            nhOptions = []
            for i in range(1,self.nd+1):
                if (self.nd%i)==0: nhOptions.append(i)
            print("possible values of multiHeadAttention nh:",nhOptions,"using",nhOptions[1])
            self.multijetAttention = multijetAttention(5, self.nd, nh=nhOptions[1], layers=layers, inputLayers=[self.inConvO2, self.reinforce2.conv])

    def forward(self, j, d, j0=None, d0=None, o=None, mask=None, debug=False):
        if d0 is None:
            d0 = d.clone()

        d = self.reinforce1(j, d)
        j = self.convJ1(j)
        d = d+d0
        j = j+j0
        d = NonLU(d, self.training)
        j = NonLU(j, self.training)

        d = self.reinforce2(j, d)
        d = d+d0
        d = NonLU(d, self.training)

        if self.multijetAttention:
            n, features, jets = o.shape
            o = self.inConvO0(o)
            o = NonLU(o, self.training)
            o0= o.clone()
            o = self.inConvO1(o)
            o = NonLU(o+o0, self.training)
            o = self.inConvO2(o)
            o = NonLU(o+o0, self.training)
            # o = self.encoderO(o)
            d = self.multijetAttention(d, o, mask, q0=None, debug=debug)


        return j, d, o


class quadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.conv = conv1d(self.nq, self.nq, 3, stride=3, name='quadjet reinforce convolution') # nn.Conv1d(self.nq, self.nq, 3, stride=3)

    def forward(self, d, q):#, o):
        d_sym     =  (d[:,:,(0,2,4)] + d[:,:,(1,3,5)])/2
        d_antisym = ((d[:,:,(0,2,4)] - d[:,:,(1,3,5)])/2).abs()
        q = torch.cat( (d_sym[:,:, 0:1], d_antisym[:,:, 0:1], q[:,:, 0:1],
                        d_sym[:,:, 1:2], d_antisym[:,:, 1:2], q[:,:, 1:2],
                        d_sym[:,:, 2:3], d_antisym[:,:, 2:3], q[:,:, 2:3]), 2)
        q = self.conv(q)
        return q


class quadjetResNetBlock(nn.Module):
    def __init__(self, dijetFeatures, quadjetFeatures, useOthJets=False, device='cuda', layers=None, inputLayers=None):
        super(quadjetResNetBlock, self).__init__()
        self.nd = dijetFeatures
        self.nq = quadjetFeatures
        self.device = device
        self.update0 = False

        self.reinforce1 = quadjetReinforceLayer(self.nq)
        self.convD1 = conv1d(self.nq, self.nq, 1, name='dijet convolution 1') # nn.Conv1d(self.nq, self.nq, 1)
        self.reinforce2 = quadjetReinforceLayer(self.nq)

        layers.addLayer(self.reinforce1.conv, inputLayers)
        layers.addLayer(self.convD1, [inputLayers[0]])
        layers.addLayer(self.reinforce2.conv, [self.convD1, self.reinforce1.conv])

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
        # d_sym     = self.convD1_sym(    d_sym)
        # d_antisym = self.convD1_antisym(d_antisym)
        q = q+q0
        d = d+d0
        # d_sym     = d_sym    +d0_sym
        # d_antisym = d_antisym+d0_antisym
        q = NonLU(q, self.training)
        d = NonLU(d, self.training)
        # d_sym     = NonLU(d_sym,     self.training)
        # d_antisym = NonLU(d_antisym, self.training)

        q = self.reinforce2(d, q)
        q = q+q0
        q = NonLU(q, self.training)
            
        if self.multijetAttention:
            n, features, jets = o.shape
            o0 = torch.cat( (o.clone(), torch.zeros(n,self.nd-features,jets).to(self.device)), 1)
            o = self.inConvO0(o)
            o = NonLU(o+o0, self.training)
            o = self.inConvO1(o)
            o = NonLU(o+o0, self.training)
            o = self.inConvO2(o)
            o = NonLU(o+o0, self.training)

            q = self.multijetAttention(q, o, mask, debug=debug)

        return d, q


class ResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nAncillaryFeatures, useOthJets='', device='cuda', nClasses=1):
        super(ResNet, self).__init__()
        self.debug = False
        self.nj = jetFeatures
        self.nd, self.nAd = dijetFeatures, 2 #total dijet features, engineered dijet features
        self.nq, self.nAq = quadjetFeatures, 2 #total quadjet features, engineered quadjet features
        self.nAe = nAncillaryFeatures
        #self.nAe = 2
        self.ne = combinatoricFeatures
        self.device = device
        dijetBottleneck   = None
        self.name = 'ResNet'+('+'+useOthJets if useOthJets else '')+'_%d_%d_%d'%(dijetFeatures, quadjetFeatures, self.ne)
        self.nClasses = nClasses

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        self.layers = layerOrganizer()

        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = conv1d(self.nj, self.nd, 2, stride=2, name='dijet builder') # nn.Conv1d(self.nj, self.nd, 2, stride=2)
        self.dijetAncillaryEmbedder = conv1d(self.nAd, self.nd, 1, name='dijet ancillary feature embedder') # nn.Conv1d(self.nAd, self.nd, 1)
        # self.dijetAncillaryCombiner = conv1d(self.nd, self.nd, 2, stride=2, name='dijet ancillary feature combiner') # nn.Conv1d(self.nAq, self.nq, 1)
        self.convJ0 = conv1d(self.nj, self.nd, 1, name='jet convolution 0') # nn.Conv1d(self.nj, self.nd, 1)
        # ancillary dijet features get appended to output of dijetBuilder

        self.layers.addLayer(self.convJ0)
        self.layers.addLayer(self.dijetBuilder)
        self.layers.addLayer(self.dijetAncillaryEmbedder, startIndex=self.dijetBuilder.index)
        # self.layers.addLayer(self.dijetAncillaryCombiner, startIndex=self.dijetAncillaryEmbedder.index)

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock = dijetResNetBlock(self.nj, self.nd, device=self.device, useOthJets=useOthJets, layers=self.layers, inputLayers=[self.convJ0, self.dijetBuilder, self.dijetAncillaryEmbedder])

        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = conv1d(self.nd, self.nq, 2, stride=2, name='quadjet builder') # nn.Conv1d(self.nd, self.nq, 2, stride=2)
        self.quadjetAncillaryEmbedder = conv1d(self.nAq, self.nq, 1, name='quadjet ancillary feature embedder') # nn.Conv1d(self.nAq, self.nq, 1)
        self.convD0 = conv1d(self.nd, self.nq, 1, name='dijet convolution 0') # nn.Conv1d(self.nd, self.nq, 1)
        # ancillary quadjet features get appended to output of quadjetBuilder

        dijetResNetBlockOutputLayer = self.dijetResNetBlock.multijetAttention.attention.o_linear if useOthJets else self.dijetResNetBlock.reinforce2.conv
        self.layers.addLayer(self.convD0, [dijetResNetBlockOutputLayer])
        self.layers.addLayer(self.quadjetBuilder, [dijetResNetBlockOutputLayer])
        self.layers.addLayer(self.quadjetAncillaryEmbedder, startIndex=self.quadjetBuilder.index)

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nd, self.nq, device=self.device, layers=self.layers, inputLayers=[self.convD0, self.quadjetBuilder, self.quadjetAncillaryEmbedder])

        self.selectQ = conv1d(self.nq, 1, 1, name='quadjet selector') # nn.Conv1d(self.nq, 1, 1)
        #self.selectQ_neg = conv1d(self.nq, 1, 1, name='quadjet negative selector') # nn.Conv1d(self.nq, 1, 1)

        self.layers.addLayer(self.selectQ, [self.quadjetResNetBlock.reinforce2.conv])
        #self.layers.addLayer(self.selectQ_neg, [self.quadjetResNetBlock.reinforce2.conv])

        self.eventAncillaryEmbedder = conv1d(self.nAe, self.ne, 1, name='Event ancillary feature embedder') # nn.Conv1d(self.nAv, self.ne, 1)
        self.eventConv1 = conv1d(self.ne, self.ne, 1, name='event convolution 1') # nn.Conv1d(self.ne, self.ne, 1)
        self.eventConv2 = conv1d(self.ne, self.ne, 1, name='event convolution 2') # nn.Conv1d(self.ne, self.ne, 1)
        self.eventConv3 = conv1d(self.ne, self.ne, 1, name='event convolution 3') # nn.Conv1d(self.ne, self.ne, 1)
        self.eventConv4 = conv1d(self.ne, self.ne, 1, name='event convolution 4') # nn.Conv1d(self.ne, self.ne, 1)
        self.out = conv1d(self.ne, self.nClasses, 1, name='out')   

        self.layers.addLayer(self.eventAncillaryEmbedder, startIndex=self.selectQ.index)
        self.layers.addLayer(self.eventConv1, [self.eventAncillaryEmbedder, self.quadjetResNetBlock.reinforce2.conv, self.selectQ])
        self.layers.addLayer(self.eventConv2, [self.eventConv1])
        self.layers.addLayer(self.eventConv3, [self.eventConv2])
        self.layers.addLayer(self.eventConv4, [self.eventConv3])
        self.layers.addLayer(self.out,       [self.eventConv4])

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
        d = NonLU(d, self.training)
        da = self.dijetAncillaryEmbedder(da)
        #da = NonLU(da, self.training)
        d += da
        d0 = d.clone()

        j = self.convJ0(jRaw)
        j = NonLU(j, self.training)
        j0= j.clone()

        j, d, _ = self.dijetResNetBlock( j, d, j0=j0, d0=d0, o=o, mask=mask, debug=self.debug)

        d_sym     =  (d[:,:,(0,2,4)] + d[:,:,(1,3,5)])/2
        d_antisym = ((d[:,:,(0,2,4)] - d[:,:,(1,3,5)])/2).abs()
        d_symantisym = torch.cat( (d_sym[:,:, 0:1], d_antisym[:,:, 0:1],
                                   d_sym[:,:, 1:2], d_antisym[:,:, 1:2],
                                   d_sym[:,:, 2:3], d_antisym[:,:, 2:3]), 2)
        q = self.quadjetBuilder(d_symantisym)
        q = NonLU(q, self.training)
        qa = self.quadjetAncillaryEmbedder(qa)        
        #qa = NonLU(qa, self.training)
        q += qa
        q0 = q.clone()
        d0 = d.clone()
        d = NonLU(self.convD0(d)+d0, self.training)

        d, q = self.quadjetResNetBlock(d, q, d0=d0, q0=q0, o=o, mask=mask, debug=self.debug) 

        q_score = self.selectQ(q)
        q_score = F.softmax(q_score,dim=2)
        #q_score_neg = F.softmax(self.selectQ_neg(q),dim=2)
        q = torch.matmul(q,q_score.transpose(1,2)) #- torch.matmul(q,q_score_neg.transpose(1,2))

        return q

    def forward(self, x, j, o, da, qa, ea):
        n = j.shape[0]

        da = da.view(n,self.nAd,6)
        qa = qa.view(n,self.nAq,3)
        if self.nAe:
            ea = ea.view(n,self.nAe,1)

        #oj = torch.cat( (j[:,:,0:4], torch.ones(n,1,4).to(self.device)), 1) 
        #o = torch.cat( (oj,o), 2)
        mask = o[:,4,:]!=-1

        js, os, qs = [], [], []
        randomR = np.random.uniform(0,2.0/self.nR, self.nR) if self.training else np.zeros(self.nR)
        for i in range(self.nR):
            js.append(self.rotate(j, self.R[i]+randomR[i]))
            os.append(self.rotate(o, self.R[i]+randomR[i]))
            qs.append(self.invPart(js[-1], os[-1], mask, da, qa, ea))
            if self.doFlip:
                #flip phi of original
                js.append(self.flipPhi(js[-1]))
                os.append(self.flipPhi(os[-1]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, ea))

                #flip phi and eta of original
                js.append(self.flipEta(js[-1]))
                os.append(self.flipEta(os[-1]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, ea))

                #flip eta of original
                js.append(self.flipEta(js[-3]))
                os.append(self.flipEta(os[-3]))
                qs.append(self.invPart(js[-1], os[-1], mask, da, qa, ea))

        e = sum(qs)/self.nRF # average over rotations and flips
        if self.nAe:
            ea = self.eventAncillaryEmbedder(ea)
            #ea = NonLU(ea, self.training)
            e += ea
        e0= e.clone()

        e = self.eventConv1(e)
        e = e+e0
        e = NonLU(e, self.training)

        e = self.eventConv2(e)
        e = e+e0
        e = NonLU(e, self.training)

        e = self.eventConv3(e)
        e = e+e0
        e = NonLU(e, self.training)

        e = self.eventConv4(e)
        e = e+e0
        e = NonLU(e, self.training)

        e = self.out(e)
        e = e.view(n, self.nClasses)
        return e

