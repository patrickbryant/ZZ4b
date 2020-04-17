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
    #return F.leaky_relu(x, negative_slope=0.1)
    return SiLU(x)
    #return F.elu(x)

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer, denom = 1, 1
    for i in range(n, n-r, -1): numer *= i
    for i in range(1, r+1,  1): denom *= i
    return numer//denom #double slash means integer division or "floor" division
    

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


class GhostBatchNorm1d(nn.Module): #https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition. I've replaced the running mean and std rules with Adam-like updates.
    def __init__(self, features, ghost_batch_size, eta=0.9, bias=True):
        super(GhostBatchNorm1d, self).__init__()
        self.features = features
        self.register_buffer('gbs', torch.tensor(ghost_batch_size, dtype=torch.long))
        self.register_buffer('bessel_correction', torch.tensor(ghost_batch_size/(ghost_batch_size-1.0), dtype=torch.float))
        self.gamma = nn.Parameter(torch .ones(self.features))
        self.bias  = nn.Parameter(torch.zeros(self.features))
        self.bias.requires_grad = bias

        self.register_buffer('eta', torch.tensor(eta, dtype=torch.float))
        self.register_buffer('m', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('s', torch .ones((1,self.features,1), dtype=torch.float))
        self.register_buffer('m_biased', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('s_biased', torch.zeros((1,self.features,1), dtype=torch.float))

        # use Adam style updates for running mean and standard deviation https://arxiv.org/pdf/1412.6980.pdf
        self.register_buffer('t', torch.tensor(0, dtype=torch.float))
        self.register_buffer('alpha', torch.tensor(0.001, dtype=torch.float))
        self.register_buffer('beta1', torch.tensor(0.9,   dtype=torch.float))
        self.register_buffer('beta2', torch.tensor(0.999, dtype=torch.float))
        self.register_buffer('eps', torch.tensor(1e-5, dtype=torch.float))
        self.register_buffer('m_biased_first_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('s_biased_first_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('m_biased_second_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('s_biased_second_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('m_first_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('s_first_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('m_second_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('s_second_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        

    def forward(self, x, debug=False):
        if self.training:
            batch_size = x.shape[0]
            locations = x.shape[2]
            ngb = batch_size // self.gbs

            #
            # Apply batch normalization with Ghost Batch statistics
            #
            x = x.transpose(1,2).contiguous().view(ngb, self.gbs*locations, self.features, 1)

            gbm = x.mean(dim=1, keepdim=True) if self.bias.requires_grad else 0
            gbv = x. var(dim=1, keepdim=True)
            gbs = (gbv + self.eps).sqrt()
            
            x = x - gbm
            x = x/gbs
            x = x.view(batch_size, locations, self.features)
            x = self.gamma * x
            x = x + self.bias
            x = x.transpose(1,2)


            #
            # Keep track of running mean and standard deviation. 
            #

            # Use mean over ghost batches for running mean and std
            # ngb = torch.tensor(ngb, dtype=torch.float).to('cuda')
            bm = gbm.detach().mean(dim=0) if self.bias.requires_grad else 0
            bs = gbs.detach().mean(dim=0) #/ self.bessel_correction
            self.m = self.eta*self.m + (1-self.eta)*bm
            self.s = self.eta*self.s + (1-self.eta)*bs
            # self.m_biased = self.eta*self.m_biased + (1-self.eta)*bm
            # self.s_biased = self.eta*self.s_biased + (1-self.eta)*bs
            # self.m = self.m_biased / (1-self.eta**self.t)
            # self.s = self.s_biased / (1-self.eta**self.t)

            # # increment time step for use in bias correction
            # self.t = self.t+1 

            # # get gradients
            # m_grad = bm - self.m 
            # s_grad = bs - self.s

            # # update biased first moment estimate
            # self.m_biased_first_moment  = self.beta1 * self.m_biased_first_moment   +  (1-self.beta1) * m_grad
            # self.s_biased_first_moment  = self.beta1 * self.s_biased_first_moment   +  (1-self.beta1) * s_grad

            # # update biased second moment estimate
            # self.m_biased_second_moment = self.beta2 * self.m_biased_second_moment  +  (1-self.beta2) * m_grad**2
            # self.s_biased_second_moment = self.beta2 * self.s_biased_second_moment  +  (1-self.beta2) * s_grad**2

            # # correct bias
            # self.m_first_moment  = self.m_biased_first_moment  / (1-self.beta1**self.t)
            # self.s_first_moment  = self.s_biased_first_moment  / (1-self.beta1**self.t)
            # self.m_second_moment = self.m_biased_second_moment / (1-self.beta2**self.t)
            # self.s_second_moment = self.s_biased_second_moment / (1-self.beta2**self.t)
            
            # # update running mean and standard deviation
            # self.m = self.m + self.alpha * self.m_first_moment / (self.m_second_moment+self.eps).sqrt()
            # self.s = self.s + self.alpha * self.s_first_moment / (self.s_second_moment+self.eps).sqrt()
            
                
            return x
        else:
            #inference stage, use running mean and standard deviation
            x = (self.gamma * ((x - self.m)/self.s).transpose(1,2) + self.bias).transpose(1,2)
            return x


class conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, name=None, index=None, doGradStats=False, hiddenIn=False, hiddenOut=False, batchNorm=False, batchNormMomentum=0.9):
        super(conv1d, self).__init__()
        self.bias = bias and not batchNorm #if doing batch norm, bias is in BN layer, not convolution
        self.module = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=self.bias, groups=groups)
        if batchNorm:
            self.batchNorm = GhostBatchNorm1d(out_channels, 32, batchNormMomentum, bias=bias) #nn.BatchNorm1d(out_channels)
        else:
            self.batchNorm = False

        self.hiddenIn=hiddenIn
        if self.hiddenIn:
            self.moduleHiddenIn = nn.Conv1d(in_channels,in_channels,1)
        self.hiddenOut=hiddenOut
        if self.hiddenOut:
            self.moduleHiddenOut = nn.Conv1d(out_channels,out_channels,1)
        self.name = name
        self.index = index
        self.gradStats = None
        self.k = 1.0/(in_channels * kernel_size)
        if doGradStats:
            self.gradStats = stats()
            self.module.weight.register_hook( make_hook(self.gradStats, self.module, 'weight') )
            #self.module.bias  .register_hook( make_hook(self.gradStats, self.module, 'bias'  ) )
    def randomize(self):
        if self.hiddenIn:
            nn.init.uniform_(self.moduleHiddenIn.weight, -(self.k**0.5), self.k**0.5)
            nn.init.uniform_(self.moduleHiddenIn.bias,   -(self.k**0.5), self.k**0.5)            
        if self.hiddenOut:
            nn.outit.uniform_(self.moduleHiddenOut.weight, -(self.k**0.5), self.k**0.5)
            nn.outit.uniform_(self.moduleHiddenOut.bias,   -(self.k**0.5), self.k**0.5)            
        nn.init.uniform_(self.module.weight, -(self.k**0.5), self.k**0.5)
        if self.bias:
            nn.init.uniform_(self.module.bias,   -(self.k**0.5), self.k**0.5)
    def forward(self,x,debug=False):
        if self.hiddenIn:
            x = NonLU(self.moduleHiddenIn(x), self.moduleHiddenIn.training)
        if self.hiddenOut:
            x = NonLU(self.module(x), self.module.training)
            return self.moduleHiddenOut(x)
        x = self.module(x)
        if self.batchNorm:
            x = self.batchNorm(x, debug=debug)
        return x


class linear(nn.Module):
    def __init__(self, in_channels, out_channels, name=None, index=None, doGradStats=False, bias=True):
        super(linear, self).__init__()
        self.module = nn.Linear(in_channels, out_channels, bias=bias)
        self.bias=bias
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
        if self.bias:
            nn.init.uniform_(self.module.bias,   -(self.k**0.5), self.k**0.5)
    def forward(self,x):
        return self.module(x)


class layerOrganizer:
    def __init__(self):
        self.layers = collections.OrderedDict()
        self.nTrainableParameters = 0

    def addLayer(self, newLayer, inputLayers=None, startIndex=1):
        if inputLayers:
            inputIndicies = inputLayers # [layer.index for layer in inputLayers]
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
    def __init__(self, 
                 dim_query=8,    dim_key=8,    dim_value=8, dim_attention=8, heads=2, dim_valueAttention=None,
                 groups_query=1, groups_key=1, groups_value=1, dim_out=8, outBias=False,
                 selfAttention=False, layers=None, inputLayers=None,
                 bothAttention=False,
                 iterations=1):
        super().__init__()
        
        self.h = heads #len(heads) #heads
        self.da = dim_attention #sum(heads)#dim_attention
        self.dq = dim_query
        self.dk = dim_key
        self.dv = dim_value
        self.dh = self.da // self.h #max(heads) #self.da // self.h
        self.dva= dim_valueAttention if dim_valueAttention else dim_attention
        self.dvh= self.dva// self.h
        self.do = dim_out
        self.iter = iterations
        self.sqrt_dh = np.sqrt(self.dh)
        self.selfAttention = selfAttention
        self.bothAttention = bothAttention

        self.q_linear = conv1d(self.dq, self.da,  1, groups=groups_query, name='attention query linear', batchNorm=False) # nn.Linear(da, da)
        self.k_linear = conv1d(self.dk, self.da,  1, groups=groups_key,   name='attention key   linear') # nn.Linear(da, da)
        self.v_linear = conv1d(self.dv, self.dva, 1, groups=groups_value, name='attention value linear') # nn.Linear(da, da)
        if self.bothAttention:
            self.sq_linear = conv1d(self.dk, self.da, 1, groups=groups_key,   name='self attention query linear') # nn.Linear(da, da)
            self.so_linear = conv1d(self.dva, self.dk, 1,   name='self attention out linear') # nn.Linear(da, da)
        #self.register_parameter(name='score_w', param=nn.Parameter(torch.ones(1)))
        #self.register_parameter(name='score_b', param=nn.Parameter(torch.zeros(1)))
        self.o_linear = conv1d(self.dva, self.do, 1, stride=1, name='attention out   linear', bias=outBias, batchNorm=True) # nn.Linear(da, da)


        if layers:
            layers.addLayer(self.q_linear, inputLayers)
            layers.addLayer(self.k_linear, inputLayers)
            layers.addLayer(self.v_linear, inputLayers)
            layers.addLayer(self.o_linear, [self.q_linear.index, self.v_linear.index, self.k_linear.index])
    
    def attention(self, q, k, v, mask, debug=False):
    
        q_k_overlap = torch.matmul(q, k.transpose(2, 3)) /  self.sqrt_dh
        if mask is not None:
            if self.selfAttention:
                q_k_overlap = q_k_overlap.masked_fill(mask, -1e9)
            mask = mask.transpose(2,3)
            q_k_overlap = q_k_overlap.masked_fill(mask, -1e9)

        v_probability = F.softmax(q_k_overlap, dim=-1) # compute joint probability distribution for which values best correspond to the query
        #v_score   = self.score_w*torch.log(F.relu( q_k_overlap+self.score_b)+1)
        #v_score      -= self.score_b*torch.log(F.relu(-q_k_overlap)+1)
        #v_score       = torch.sigmoid(q_k_overlap)
        v_weights     = v_probability #* v_score
        if mask is not None:
            v_weights = v_weights.masked_fill(mask, 0)
            if self.selfAttention:
                mask = mask.transpose(2,3)
                v_weights = v_weights.masked_fill(mask, 0)
        if debug:
            print("q\n",q[0])
            print("k\n",k[0])
            print("mask\n",mask[0])
            print("v_probability\n",v_probability[0])
            #print("score_w,score_b",self.score_w.data,self.score_b.data)
            #print("v_score\n",v_score[0])
            print("v_weights\n",v_weights[0])
            print("v\n",v[0])

        output = torch.matmul(v_weights, v)
        if debug:
            print("output\n",output[0])
            input()
        return output

    def forward(self, q, k, v, q0=None, mask=None, qLinear=0, debug=False, selfAttention=False):
        
        bs = q.size(0)
        sq = None
        k = self.k_linear(k)
        v = self.v_linear(v)

        #check if all items are going to be masked
        sl = mask.shape[1]
        vqk_mask = mask.sum(dim=1)==sl
        vqk_mask = vqk_mask.view(bs, 1, 1)

        # #hack to make unequal head dimensions 3 and 6, add three zero padded features before splitting into two heads of dim 6
        # q = F.pad(input=q, value=0, pad=(0,0,1,0,0,0))
        # k = F.pad(input=k, value=0, pad=(0,0,1,0,0,0))
        # v = F.pad(input=v, value=0, pad=(0,0,1,0,0,0))

        #split into heads
        k = k.view(bs, self.h, self.dh,  -1)
        v = v.view(bs, self.h, self.dvh, -1)
        
        # transpose to get dimensions bs * h * sl * (da//h==dh)
        k = k.transpose(2,3)
        v = v.transpose(2,3)
        mask = mask.view(bs, 1, -1, 1)

        #now do q transformations iter number of times
        for i in range(1,self.iter+1):
            #q0 = q.clone()
            if selfAttention:
                q = self.sq_linear(q)
            else:
                q = self. q_linear(q)

            q = q.view(bs, self.h, self.dh, -1)
            q = q.transpose(2,3)

            # calculate attention 
            vqk = self.attention(q, k, v, mask, debug) # outputs a linear combination of values (v) given the overlap of the queries (q) with the keys (k)

            # concatenate heads and put through final linear layer
            vqk = vqk.transpose(2,3).contiguous().view(bs, self.dva, -1)

            if debug:
                print("vqk\n",vqk[0])
                input()
            if selfAttention:
                vqk = self.so_linear(vqk)
            else:
                vqk = self. o_linear(vqk)

            # if all the input items are masked, we don't want the bias term of the output layer to have any impact
            vqk = vqk.masked_fill(vqk_mask, 0)

            q = q0 + vqk

            if i==self.iter:
                q0 = q.clone()
            if not selfAttention:
                q = NonLU(q, self.training)

            #if selfAttention: break
                
        return q, q0


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
    def __init__(self, jetFeatures, embedFeatures, attentionFeatures, nh=1, layers=None, inputLayers=None):
        super(multijetAttention, self).__init__()
        self.nj = jetFeatures
        self.ne = embedFeatures
        self.na = attentionFeatures
        self.nh = nh
        #self.jetEmbed = conv1d(5, 5, 1, name='other jet embed')
        #self.jetConv1 = conv1d(5, 5, 1, name='other jet convolution 1')
        #self.jetConv2 = conv1d(5, 5, 1, name='other jet convolution 2')
        self.attention = MultiHeadAttention(   dim_query=self.ne, dim_key=5,    dim_value=5, dim_attention=8, heads=2, dim_valueAttention=6, dim_out=self.ne,
                                            groups_query=1,    groups_key=1, groups_value=1, 
                                            selfAttention=False, outBias=False, layers=layers, inputLayers=inputLayers,
                                            bothAttention=False,
                                            iterations=2)
        self.outputLayer = self.attention.o_linear.index
        
    def forward(self, q, kv, mask, q0=None, qLinear=0, debug=False):
        if debug:
            print("q\n",   q[0])        
            print("kv\n",  kv[0])
            print("mask\n",mask[0])

        # kv0 = kv.clone()

        # kv = self.jetEmbed(kv)
        # kv = NonLU(kv, self.training)
        # kv = kv0 + kv

        # kv = self.jetConv1(kv)
        # kv = NonLU(kv, self.training)        
        # kv = kv0 + kv

        # kv = self.jetConv2(kv)
        # kv = NonLU(kv, self.training)        
        # kv = kv0 + kv

        #kv = NonLU(kv, self.training)
        #kv, _ = self.attention(kv, kv, kv, q0=kv0, mask=mask, debug=debug, selfAttention=True)
        #kv, _ = self.attention(kv, kv, kv, q0=kv0, mask=mask, debug=debug, selfAttention=True)
        q, q0 = self.attention( q, kv, kv, q0= q0, mask=mask, debug=debug, selfAttention=False)
        # q = q0 + q
        # q = NonLU(q, self.training)

        # vqk = self.attention(q, kv, kv, mask=mask, debug=debug, selfAttention=False)
        # q = q0 + vqk
        # q0 = q.clone()
        # q = NonLU(q, self.training)

        return q, q0


class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures, batchNorm=False):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|            
        self.conv = conv1d(self.nd, self.nd, 3, stride=3, name='dijet reinforce convolution', batchNorm=batchNorm) # nn.Conv1d(self.nd, self.nd, 3, stride=3)

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

        self.reinforce1 = dijetReinforceLayer(self.nd, batchNorm=True)
        self.convJ = conv1d(self.nd, self.nd, 1, name='jet convolution', batchNorm=False) # nn.Conv1d(self.nd, self.nd, 1)
        self.reinforce2 = dijetReinforceLayer(self.nd, batchNorm=True)

        layers.addLayer(self.reinforce1.conv, inputLayers)
        layers.addLayer(self.convJ, [inputLayers[0]])
        layers.addLayer(self.reinforce2.conv, [self.convJ.index, self.reinforce1.conv.index])

        self.outputLayer = self.reinforce2.conv.index

        self.multijetAttention = None
        if useOthJets:
            self.na = 6
            nhOptions = []
            for i in range(1,self.na+1):
                if (self.na%i)==0: nhOptions.append(i)
            print("possible values of multiHeadAttention nh:",nhOptions,"using",nhOptions[1])
            self.multijetAttention = multijetAttention(self.nj ,self.nd, self.na, nh=nhOptions[1], layers=layers)#, inputLayers=[self.jetEmbed])
            self.outputLayer = self.multijetAttention.outputLayer

    def forward(self, j, d, da=None, j0=None, d0=None, o=None, mask=None, debug=False):

        d = self.reinforce1(j, d)
        j = self.convJ(j)
        d = d+d0
        j = j+j0
        d = NonLU(d, self.training)
        j = NonLU(j, self.training)

        d = self.reinforce2(j, d)
        d = d+d0
        d0 = d.clone()
        d = NonLU(d, self.training)

        if self.multijetAttention:
            d, d0 = self.multijetAttention(d, o, mask, q0=d0, debug=debug)

        return d, d0


class quadjetReinforceLayer(nn.Module):
    def __init__(self, quadjetFeatures, batchNorm=False):
        super(quadjetReinforceLayer, self).__init__()
        self.nq = quadjetFeatures

        # make fixed convolution to compute average of dijet pixel pairs (symmetric bilinear)
        self.sym = nn.Conv1d(self.nq, self.nq, 2, stride=2, bias=False, groups=self.nq)
        self.sym.weight.data.fill_(0.5)
        self.sym.weight.requires_grad = False

        # make fixed convolution to compute difference of dijet pixel pairs (antisymmetric bilinear)
        self.antisym = nn.Conv1d(self.nq, self.nq, 2, stride=2, bias=False, groups=self.nq)
        self.antisym.weight.data.fill_(0.5)
        self.antisym.weight.data[:,:,1] *= -1
        self.antisym.weight.requires_grad = False

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.conv = conv1d(self.nq, self.nq, 3, stride=3, name='quadjet reinforce convolution', batchNorm=batchNorm)

    def forward(self, d, q):#, o):
        d_sym     = self.    sym(d)       # (d[:,:,(0,2,4)] + d[:,:,(1,3,5)])/2
        d_antisym = self.antisym(d).abs() #((d[:,:,(0,2,4)] - d[:,:,(1,3,5)])/2).abs()
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

        self.reinforce1 = quadjetReinforceLayer(self.nq, batchNorm=True)
        self.convD = conv1d(self.nq, self.nq, 1, name='dijet convolution', batchNorm=False) # nn.Conv1d(self.nq, self.nq, 1)
        self.reinforce2 = quadjetReinforceLayer(self.nq, batchNorm=True)

        layers.addLayer(self.reinforce1.conv, inputLayers)
        layers.addLayer(self.convD, [inputLayers[0]])
        layers.addLayer(self.reinforce2.conv, [self.convD.index, self.reinforce1.conv.index])

    def forward(self, d, q, qa=None, d0=None, q0=None, o=None, mask=None, debug=False):

        q = self.reinforce1(d, q)
        d = self.convD(d)
        q = q+q0
        d = d+d0
        q = NonLU(q, self.training)
        d = NonLU(d, self.training)

        q = self.reinforce2(d, q)
        q = q+q0
        q0 = q.clone()
        q = NonLU(q, self.training)

        return q, q0


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
        self.store = None
        self.storeData = {}

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        self.layers = layerOrganizer()


        # embed inputs to dijetResNetBlock in target feature space
        self.jetEmbed = conv1d(self.nj, self.nd, 1, name='jet embed', batchNorm=False)
        self.dijetAncillaryEmbed = conv1d(self.nAd, self.nd, 1, name='dijet ancillary feature embed', batchNorm=False) # nn.Conv1d(self.nAd, self.nd, 1)

        self.layers.addLayer(self.jetEmbed)
        self.layers.addLayer(self.dijetAncillaryEmbed)


        # Stride=3 Kernel=3 reinforce dijet features, in parallel update jet features for next reinforce layer
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock = dijetResNetBlock(self.nj, self.nd, device=self.device, useOthJets=useOthJets, layers=self.layers, inputLayers=[self.jetEmbed.index, self.dijetAncillaryEmbed.index])
        

        # embed inputs to quadjetResNetBlock in target feature space
        self.dijetEmbed = conv1d(self.nd, self.nq, 1, name='dijet embed', batchNorm=False)
        self.quadjetAncillaryEmbed = conv1d(self.nAq, self.nq, 1, name='quadjet ancillary feature embed', batchNorm=False) # nn.Conv1d(self.nAq, self.nq, 1)

        self.layers.addLayer(self.dijetEmbed, [self.dijetResNetBlock.reinforce2.conv.index])
        self.layers.addLayer(self.quadjetAncillaryEmbed, startIndex=self.dijetEmbed.index)

        # Stride=3 Kernel=3 reinforce quadjet features, in parallel update dijet features for next reinforce layer
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4|2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nd, self.nq, device=self.device, layers=self.layers, inputLayers=[self.dijetEmbed.index, self.quadjetAncillaryEmbed.index])


        # Layers for event-level ancillary feature processing
        self.eventAncillaryEmbed1 = conv1d(self.nAe, self.ne, 1, name='Event ancillary feature embed1', batchNorm=False) # nn.Conv1d(self.nAv, self.ne, 1)
        self.eventAncillaryEmbed2 = conv1d(self.nAe, self.ne, 1, name='Event ancillary feature embed2', batchNorm=False) # nn.Conv1d(self.nAv, self.ne, 1)
        self.eventConv1 = conv1d(self.ne, self.ne, 1, name='event convolution 1', batchNorm=True) # nn.Conv1d(self.ne, self.ne, 1)
        self.eventConv2 = conv1d(self.ne, self.ne, 1, name='event convolution 2', batchNorm=False) # nn.Conv1d(self.ne, self.ne, 1)

        # Calculalte score for each quadjet, add them together with corresponding weight, and go to final output layer
        self.select_q = conv1d(self.ne, 1, 1, name='quadjet selector', batchNorm=False) # nn.Conv1d(self.nq, 1, 1)
        self.out      = conv1d(self.ne, self.nClasses, 1, name='out', batchNorm=True)#, batchNormMomentum=0.99)   
        #self.out      = conv1d(self.nClasses, self.nClasses, self.ne, groups=self.nClasses, name='out', batchNorm=True)#, batchNormMomentum=0.99)   

        self.layers.addLayer(self.eventAncillaryEmbed1, startIndex=self.quadjetResNetBlock.reinforce2.conv.index)
        self.layers.addLayer(self.eventConv1, [self.quadjetResNetBlock.reinforce2.conv.index])
        self.layers.addLayer(self.eventAncillaryEmbed2, startIndex=self.eventConv1.index)
        self.layers.addLayer(self.eventConv2, [self.eventConv1.index])
        self.layers.addLayer(self.select_q, [self.eventConv2.index])
        self.layers.addLayer(self.out,      [self.eventConv2.index, self.select_q.index])


    def rotate(self, jR, R): # j[event, mu, jet], mu=2 is phi
        #jR = j.clone() 
        jR[:,2,:] = (jR[:,2,:] + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
        return jR

    def flipPhi(self, jF): # j[event, mu, jet], mu=2 is phi
        #jF = j.clone() 
        jF[:,2,:] = -1*jF[:,2,:]
        return jF

    def flipEta(self, jF): # j[event, mu, jet], mu=1 is eta
        #jF = j.clone() 
        jF[:,1,:] = -1*jF[:,1,:]
        return jF


    def invPart(self, j, o, mask, da, qa):
        n = j.shape[0]

        # Now embed the jet 4-vectors into the target feature space
        j = self.jetEmbed(j)
        j0 = j.clone()
        j = NonLU(j, self.training)

        # Now embed the dijet ancillary features in the target feature space
        d = self.dijetAncillaryEmbed(da)
        d0 = d.clone()
        d = NonLU(d, self.training)

        d, d0 = self.dijetResNetBlock(j, d, da=da, j0=j0, d0=d0, o=o, mask=mask, debug=self.debug)

        if self.store:
            self.storeData['dijets'] = d[0].detach().to('cpu').numpy()

        d = self.dijetEmbed(d)
        d = d+d0
        d = NonLU(d, self.training)

        # Now embed the quadjet ancillary features in the target feature space
        q = self.quadjetAncillaryEmbed(qa)        
        q0 = q.clone()
        q = NonLU(q, self.training)

        q, q0 = self.quadjetResNetBlock(d, q, qa=qa, d0=d0, q0=q0, o=o, mask=mask, debug=self.debug) 

        if self.store:
            self.storeData['quadjets'] = q[0].detach().to('cpu').numpy()

        return q, q0


    def forward(self, x, j, o, da, qa, ea):
        n = j.shape[0]

        da = da.view(n,self.nAd,6)
        qa = qa.view(n,self.nAq,3)
        mask = o[:,4,:]==-1

        if self.store:
            self.storeData[  'canJets'] = j[0].to('cpu').numpy()
            self.storeData['otherJets'] = o[0].to('cpu').numpy()

        # Copy inputs nRF times to compute each of the symmetry transformations 
        j = j.repeat(self.nRF, 1, 1)
        o = o.repeat(self.nRF, 1, 1)
        da = da.repeat(self.nRF, 1, 1)
        qa = qa.repeat(self.nRF, 1, 1)
        mask = mask.repeat(self.nRF, 1)

        # apply each of the symmetry transformations over which we will average
        randomR = np.random.uniform(0,2.0/self.nR, self.nR) if self.training else np.zeros(self.nR)
        for r in range(self.nR):
            i = r*(4 if self.doFlip else 1)
            l, u = i*n, (i+1)*n
            j[l:u] = self.rotate(j[l:u], self.R[r]+randomR[r])
            o[l:u] = self.rotate(o[l:u], self.R[r]+randomR[r])

            if self.doFlip:
                #flip phi
                l, u = (i+1)*n, (i+2)*n
                j[l:u] = self.rotate(j[l:u], self.R[r]+randomR[r])
                o[l:u] = self.rotate(o[l:u], self.R[r]+randomR[r])
                j[l:u] = self.flipPhi(j[l:u])
                o[l:u] = self.flipPhi(o[l:u])

                #flip eta
                l, u = (i+2)*n, (i+3)*n
                j[l:u] = self.rotate(j[l:u], self.R[r]+randomR[r])
                o[l:u] = self.rotate(o[l:u], self.R[r]+randomR[r])
                j[l:u] = self.flipEta(j[l:u])
                o[l:u] = self.flipEta(o[l:u])

                #flip phi and eta
                l, u = (i+3)*n, (i+4)*n
                j[l:u] = self.rotate(j[l:u], self.R[r]+randomR[r])
                o[l:u] = self.rotate(o[l:u], self.R[r]+randomR[r])
                j[l:u] = self.flipPhi(j[l:u])
                o[l:u] = self.flipPhi(o[l:u])
                j[l:u] = self.flipEta(j[l:u])
                o[l:u] = self.flipEta(o[l:u])

        # compute the quadjet pixels and average them over the symmetry transformations
        q, q0 = self.invPart(j, o, mask, da, qa)
        q, q0 = q.view(self.nRF, n, self.nq, 3), q0.view(self.nRF, n, self.nq, 3)
        q, q0 = q.mean(dim=0), q0.mean(dim=0)

        # Everything from here on out has no dependence on eta/phi flips and minimal dependence on phi rotations
        if self.store:
            self.storeData['quadjets_sym'] = q[0].detach().to('cpu').numpy()

        ea = ea.view(n,self.nAe,1)

        # project the event-level ancillary features into the target feature space and shift the quadjet image and then apply the nonlinearity
        q = self.eventConv1(q) + self.eventAncillaryEmbed1(ea)
        q = q0 + q
        q = NonLU(q, self.training)

        # project the event-level ancillary features into the target feature space and shift the quadjet image and then apply the nonlinearity
        q = self.eventConv2(q) + self.eventAncillaryEmbed2(ea)
        q = q0 + q
        q = NonLU(q, self.training)

        if self.store:
            self.storeData['quadjets_sym_eventAncillary'] = q[0].detach().to('cpu').numpy()

        #compute a score for each event view (quadjet) 
        q_score = self.select_q(q)
        #convert the score to a 'probability' with softmax. This way the classifier is learning which view is most relevant to the classification task at hand.
        q_score = F.softmax(q_score, dim=2)
        #add together the quadjets with their corresponding probability weight
        e = torch.matmul(q, q_score.transpose(1,2))

        if self.store:
            self.storeData['q_score'] = q_score[0].detach().to('cpu').numpy()
        if self.store:
            self.storeData['event'] = e[0].detach().to('cpu').numpy()

        #project the final event-level pixel into the class score space
        c = self.out(e)
        c = c.view(n, self.nClasses)

        if self.store:
            classProb = F.softmax(c, dim=1)
            self.storeData['classProb'] = classProb[0].detach().to('cpu').numpy()

        return c, q_score.view(n, 3)


    def writeStore(self):
        np.save(self.store, self.storeData)

