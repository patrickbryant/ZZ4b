import collections
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Lin_View(nn.Module):
    def __init__(self):
        super(Lin_View, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

#pytorch 0.4 does not have inverse hyperbolic trig functions
def asinh(x):
    xsign = x.sign()
    xunsigned = x * xsign
    loggand = xunsigned+(xunsigned.pow(2)+1).sqrt() # numerically unstable if you do x+(x.pow(2)+1).sqrt() because is very close to zero when x is very negative
    return torch.log(loggand)*xsign # if x is zero then asinh(x) is also zero so don't need zero protection on xsign

def acosh(x):
    return torch.log(x+(x**2-1).sqrt())

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

# some basic four-vector operations
def PxPyPzE(v): # need this to be able to add four-vectors
    pt  = v[:,0:1,:]
    eta = v[:,1:2,:]
    phi = v[:,2:3,:]
    m   = v[:,3:4,:]
    
    Px, Py, Pz = pt*phi.cos(), pt*phi.sin(), pt*eta.sinh()
    E = (pt**2 + Pz**2 + m**2).sqrt()

    return torch.cat( (Px,Py,Pz,E), 1 )

def PtEtaPhiM(v):
    px = v[:,0:1]
    py = v[:,1:2]
    pz = v[:,2:3]
    e  = v[:,3:4]

    Pt  = (px**2+py**2).sqrt()
    ysign = py.sign()
    ysign = ysign + (ysign==0.0).float() # if py==0, px==Pt and acos(1)=pi/2 so we need zero protection on py.sign()
    Phi = (px/(Pt+0.00001)).acos() * ysign
    Eta = asinh(pz/(Pt+0.00001))
    M   = (e**2 - px**2 - py**2 - pz**2).sqrt()

    # if torch.isinf(Eta).any():
    #     index = torch.isinf(Eta).any(dim=-1).view(-1)
    #     print()
    #     print("PtEtaPhiM")
    #     print("pz",pz[index])
    #     print("Pt",Pt[index])

    return torch.cat( (Pt, Eta, Phi, M) , 1 )    

def calcDeltaR(v1, v2): #expects eta, phi representation
    dPhi12 = (v1[:,2:3]-v2[:,2:3])%math.tau
    dPhi21 = (v2[:,2:3]-v1[:,2:3])%math.tau
    dPhi = torch.min(dPhi12,dPhi21)
    dR = ((v1[:,1:2]-v2[:,1:2])**2 + dPhi**2).sqrt()
    return dR

def addFourVectors(v1, v2, v1PxPyPzE=None, v2PxPyPzE=None): # output added four-vectors in pt,eta,phi,m coordinates and opening angle between constituents
    #vX[batch index, (pt,eta,phi,m), object index]
    dR  = calcDeltaR(v1, v2)

    if v1PxPyPzE is None:
        v1PxPyPzE = PxPyPzE(v1)
    if v2PxPyPzE is None:
        v2PxPyPzE = PxPyPzE(v2)

    v12PxPyPzE = v1PxPyPzE + v2PxPyPzE
    v12        = PtEtaPhiM(v12PxPyPzE)

    return v12, v12PxPyPzE, dR

def diObjectMass(v1PxPyPzE, v2PxPyPzE):
    v12PxPyPzE = v1PxPyPzE + v2PxPyPzE
    M = (v12PxPyPzE[:,3:4]**2 - v12PxPyPzE[:,0:1]**2 - v12PxPyPzE[:,1:2]**2 - v12PxPyPzE[:,2:3]**2).sqrt()
    return M

def matrixMdR(v1, v2, v1PxPyPzE=None, v2PxPyPzE=None): #output matrix M.shape = (batch size, 2, n v1 objects, m v2 objects)
    if v1PxPyPzE is None:
        v1PxPyPzE = PxPyPzE(v1)
    if v2PxPyPzE is None:
        v2PxPyPzE = PxPyPzE(v2)

    b = v1.shape[0]
    n, m = v1.shape[2], v2.shape[2]

    # use PxPyPzE representation to compute M
    v1PxPyPzE = v1PxPyPzE  .view(b, -1, n, 1)
    v1PxPyPzE = v1PxPyPzE.repeat(1,  1, 1, m)
    v2PxPyPzE = v2PxPyPzE  .view(b, -1, 1, m)
    v2PxPyPzE = v2PxPyPzE.repeat(1,  1, n, 1)
    
    M = diObjectMass(v1PxPyPzE, v2PxPyPzE)

    # use PtEtaPhiM representation to compute dR
    v1 = v1.view(b, -1, n, 1)
    v2 = v2.view(b, -1, 1, m)
    v1 = v1.repeat(1, 1, 1, m)
    v2 = v2.repeat(1, 1, n, 1)

    dR = calcDeltaR(v1, v2)

    return torch.cat( (M, dR), 1 )
    


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
    

# class basicDNN(nn.Module):
#     def __init__(self, inputFeatures, layers, nodes, pDropout):
#         super(basicDNN, self).__init__()
#         self.name = 'FC%dx%d_pdrop%.2f'%(layers, nodes, pDropout)
#         fc=[]
#         fc.append(nn.Linear(inputFeatures, nodes))
#         fc.append(nn.ReLU())
#         #fc.append(nn.Dropout(p=pDropout))
#         for l in range(layers):
#             fc.append(nn.Linear(nodes, nodes))
#             fc.append(nn.ReLU())
#             fc.append(nn.Dropout(p=pDropout))
#             #if l < layers-1: fc.append(nn.Dropout(p=pDropout))
#         fc.append(nn.Linear(nodes, 1))
#         self.net = nn.Sequential(*fc)
        
#     def forward(self, x, p, a):
#         return self.net(x)

def checkMemory():
    t_mem = torch.cuda.get_device_properties(0).total_memory
    c_mem = torch.cuda.memory_cached(0)
    a_mem = torch.cuda.memory_allocated(0)
    f_mem = c_mem-a_mem  # free inside cache in units of bytes
    memoryString = 'CUDA cached, allocated: %3.0f%%, %3.0f%% '%(100*c_mem/t_mem, 100*a_mem/t_mem)
    print(memoryString)


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


class scaler(nn.Module):
    def __init__(self, features, shape=None):
        super(scaler, self).__init__()
        self.features = features
        if shape is None:
            self.register_buffer('m', torch.zeros((1,self.features,1), dtype=torch.float))
            self.register_buffer('s', torch .ones((1,self.features,1), dtype=torch.float))
        else:
            self.register_buffer('m', torch.zeros(shape, dtype=torch.float))
            self.register_buffer('s', torch .ones(shape, dtype=torch.float))
        
    def forward(self, x, mask=None, debug=False):
            x = x - self.m
            x = x / self.s
            return x


class GhostBatchNorm1d(nn.Module): #https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition. 
    def __init__(self, features, ghost_batch_size=32, number_of_ghost_batches=64, nAveraging=1, eta=0.9, bias=True):
        super(GhostBatchNorm1d, self).__init__()
        self.features = features
        self.register_buffer('gbs', torch.tensor(ghost_batch_size, dtype=torch.long))
        #if number_of_ghost_batches is not None:
        self.register_buffer('ngb', torch.tensor(number_of_ghost_batches*nAveraging, dtype=torch.long))
        #else:
        #    self.ngb = None
        self.register_buffer('bessel_correction', torch.tensor(ghost_batch_size/(ghost_batch_size-1.0), dtype=torch.float))
        self.gamma = nn.Parameter(torch .ones(self.features))
        self.bias  = nn.Parameter(torch.zeros(self.features))
        self.bias.requires_grad = bias

        self.register_buffer('eps', torch.tensor(1e-5, dtype=torch.float))
        self.register_buffer('eta', torch.tensor(eta, dtype=torch.float))
        self.register_buffer('m', torch.zeros((1,self.features,1), dtype=torch.float))
        self.register_buffer('s', torch .ones((1,self.features,1), dtype=torch.float))
        # self.register_buffer('m_biased', torch.zeros((1,self.features,1), dtype=torch.float))
        # self.register_buffer('s_biased', torch.zeros((1,self.features,1), dtype=torch.float))

        # # use Adam style updates for running mean and standard deviation https://arxiv.org/pdf/1412.6980.pdf
        # self.register_buffer('t', torch.tensor(0, dtype=torch.float))
        # self.register_buffer('alpha', torch.tensor(0.001, dtype=torch.float))
        # self.register_buffer('beta1', torch.tensor(0.9,   dtype=torch.float))
        # self.register_buffer('beta2', torch.tensor(0.999, dtype=torch.float))
        # self.register_buffer('m_biased_first_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        # self.register_buffer('s_biased_first_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        # self.register_buffer('m_biased_second_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        # self.register_buffer('s_biased_second_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        # self.register_buffer('m_first_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        # self.register_buffer('s_first_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        # self.register_buffer('m_second_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        # self.register_buffer('s_second_moment', torch.zeros((1,self.features,1), dtype=torch.float))
        

    def forward(self, x, mask=None, debug=False):
        if self.training:
            batch_size = x.shape[0]
            pixels = x.shape[2]
            #if self.ngb is not None: # if number of ghost batches is specified, compute corresponding ghost batch size
            self.gbs = batch_size // self.ngb
            #else: # ghost batch size is specified, compute corresponding number of ghost batches
            #self.ngb = batch_size // self.gbs

            #
            # Apply batch normalization with Ghost Batch statistics
            #
            x = x.transpose(1,2).contiguous().view(self.ngb, self.gbs*pixels, self.features, 1)

            
            if mask is None:
                gbm = x.mean(dim=1, keepdim=True) if self.bias.requires_grad else 0
                gbv = x. var(dim=1, keepdim=True)
                gbs = (gbv + self.eps).sqrt()

                # Use mean over ghost batches for running mean and std
                bm = gbm.detach().mean(dim=0) if self.bias.requires_grad else 0
                bs = gbs.detach().mean(dim=0) #/ self.bessel_correction
            else:
                # Compute masked mean and std for each ghost batch
                mask = mask.view(self.ngb, self.gbs*pixels, 1, 1)
                nUnmasked = (mask==0).sum(dim=1,keepdim=True).float()
                denomMean = nUnmasked +   (nUnmasked==0).float() # prevent divide by zero
                denomVar  = nUnmasked + 2*(nUnmasked==0).float() + (nUnmasked==1).float() - 1 # prevent divide by zero with bessel correction
                x   = x.masked_fill(mask, 0)
                xs  = x.sum(dim=1, keepdim=True)
                gbm = xs  / denomMean if self.bias.requires_grad else 0
                x2  = x**2
                x2s = x2.sum(dim=1, keepdim=True)
                x2m = x2s / denomMean
                gbv = x2m - gbm**2
                gbv = gbv * nUnmasked / denomVar
                gbs = (gbv + self.eps).sqrt()

                # Compute masked mean and std over the whole batch
                nUnmasked = nUnmasked.detach()
                xs  = xs .detach()
                x2s = x2s.detach()
                nUnmasked = nUnmasked.sum(dim=0)
                denomMean = nUnmasked +   (nUnmasked==0).float() # prevent divide by zero
                denomVar  = nUnmasked + 2*(nUnmasked==0).float() + (nUnmasked==1).float() - 1 # prevent divide by zero with bessel correction
                bm  = xs .sum(dim=0) / denomMean if self.bias.requires_grad else 0
                x2s = x2s.sum(dim=0)
                x2m = x2s / denomMean
                bv = x2m - bm**2
                bv = bv * nUnmasked / denomVar
                bs = (bv + self.eps).sqrt()

                

            x = x - gbm
            x = x/gbs
            x = x.view(batch_size, pixels, self.features)
            x = self.gamma * x
            x = x + self.bias
            x = x.transpose(1,2) # back to standard indexing for convolutions: [batch, feature, pixel]


            #
            # Keep track of running mean and standard deviation. 
            #

            # Simplest possible method
            self.m = self.eta*self.m + (1-self.eta)*bm
            self.s = self.eta*self.s + (1-self.eta)*bs

            # # Simplest method + bias correction
            # self.m_biased = self.eta*self.m_biased + (1-self.eta)*bm
            # self.s_biased = self.eta*self.s_biased + (1-self.eta)*bs
            # # increment time step for use in bias correction
            # self.t = self.t+1 
            # self.m = self.m_biased / (1-self.eta**self.t)
            # self.s = self.s_biased / (1-self.eta**self.t)

            # # Adam inspired method
            # # get 'gradients'
            # m_grad = bm - self.m 
            # s_grad = bs - self.s

            # # update biased first moment estimate
            # self.m_biased_first_moment  = self.beta1 * self.m_biased_first_moment   +  (1-self.beta1) * m_grad
            # self.s_biased_first_moment  = self.beta1 * self.s_biased_first_moment   +  (1-self.beta1) * s_grad

            # # update biased second moment estimate
            # self.m_biased_second_moment = self.beta2 * self.m_biased_second_moment  +  (1-self.beta2) * m_grad**2
            # self.s_biased_second_moment = self.beta2 * self.s_biased_second_moment  +  (1-self.beta2) * s_grad**2

            # # increment time step for use in bias correction
            # self.t = self.t+1 

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
            x = x - self.m
            x = x / self.s
            x = x.transpose(1,2)
            x = x * self.gamma
            x = x + self.bias
            x = x.transpose(1,2)            
            #x = (self.gamma * ((x - self.m)/self.s).transpose(1,2) + self.bias).transpose(1,2)
            return x


class conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bias=True, groups=1, name=None, index=None, doGradStats=False, hiddenIn=False, hiddenOut=False, batchNorm=False, batchNormMomentum=0.9, nAveraging=1):
        super(conv1d, self).__init__()
        self.bias = bias and not batchNorm #if doing batch norm, bias is in BN layer, not convolution
        self.module = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=self.bias, groups=groups)
        if batchNorm:
            self.batchNorm = GhostBatchNorm1d(out_channels, nAveraging=nAveraging, eta=batchNormMomentum, bias=bias) #nn.BatchNorm1d(out_channels)
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
    def forward(self,x, mask=None, debug=False):
        if self.hiddenIn:
            x = NonLU(self.moduleHiddenIn(x), self.moduleHiddenIn.training)
        if self.hiddenOut:
            x = NonLU(self.module(x), self.module.training)
            return self.moduleHiddenOut(x)
        x = self.module(x)
        if self.batchNorm:
            x = self.batchNorm(x, mask=mask, debug=debug)
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

def vectorPrint(vector, formatString="%6.1f", end='\n'):
    vectorString = ", ".join([formatString%element for element in vector])
    print(vectorString, end=end)

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

    def print(self, batchNorm=False):
        for index in self.layers:
            print("----- Layer %2d -----"%(index))
            for layer in self.layers[index]:
                print('|',layer.name.ljust(40), end='')
            print('')
            if batchNorm:
                for layer in self.layers[index]:
                    if layer.batchNorm:
                        print(" mean",end='')
                        vectorPrint(layer.batchNorm.m[0].transpose(0,1)[0])
                        print("  std", end='')
                        vectorPrint(layer.batchNorm.s[0].transpose(0,1)[0])
                        print("gamma", end='')
                        vectorPrint(layer.batchNorm.gamma.data)
                        print(" bias", end='')
                        vectorPrint(layer.batchNorm.bias.data)
                    else:
                        print('|',' '*40, end='')
                print('')
            # for layer in self.layers[index]:
            #     if layer.gradStats:
            #         print('|',layer.gradStats.summary.ljust(40), end='')
            #     else:
            #         print('|',' '*40, end='')
            # print('')
            # for layer in self.layers[index]:
            #     print('|',str(layer.module).ljust(45), end=' ')
            # print('|')


# class basicCNN(nn.Module):
#     def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout):
#         super(basicCNN, self).__init__()
#         self.name = 'basicCNN_%d_%d_%d_%d_pdrop%.2f'%(dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout)
#         # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
#         # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
#         # |1,2,3,4|1,2,3,4|1,2,3,4|  ##kernel=3
#         self.conv1 = nn.Sequential(*[nn.Conv1d(     jetFeatures,        dijetFeatures, 2, stride=2), nn.ReLU()])
#         self.conv2 = nn.Sequential(*[nn.Conv1d(   dijetFeatures,      quadjetFeatures, 2, stride=2), nn.ReLU()])
#         self.conv3 = nn.Sequential(*[nn.Conv1d( quadjetFeatures, combinatoricFeatures, 3, stride=1), nn.ReLU()])

#         self.line1 = nn.Sequential(*[nn.Linear(combinatoricFeatures, nodes), nn.ReLU()])
#         self.line2 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
#         self.line3 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
#         self.line4 =                 nn.Linear(nodes, 1)

#     def forward(self, x, a):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.view(x.shape[0], -1)
        
#         x = self.line1(x)
#         x = self.line2(x)
#         x = self.line3(x)
#         x = self.line4(x)
#         return x


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
        #self.sqrt_dh = np.sqrt(self.dh)
        self.selfAttention = selfAttention
        self.bothAttention = bothAttention

        #self.q_linear = conv1d(self.dq, self.da,  1, groups=groups_query, name='attention query linear')#, batchNorm=True, nAveraging=4)
        self.k_linear = conv1d(self.dk, self.da,  1, groups=groups_key,   name='attention key   linear')#, batchNorm=True, nAveraging=4)
        self.v_linear = conv1d(self.dv, self.dva, 1, groups=groups_value, name='attention value linear')#, batchNorm=True, nAveraging=4)
        self.q_k_overlap_GBN = GhostBatchNorm1d(self.h, nAveraging=4)
        # self.q_k_overlap_scale = nn.Parameter(torch .ones(1))
        # self.q_k_overlap_bias  = nn.Parameter(torch.zeros(1))

        #self.overlap_pairFeature_conv = nn.Conv3d(3, 1, 1, stride=1)
        self.pairFeature_conv = nn.Conv2d(2, self.dva, 1, stride=1)

        if self.bothAttention:
            self.sq_linear = conv1d(self.dk, self.da, 1, groups=groups_key,   name='self attention query linear')
            self.so_linear = conv1d(self.dva, self.dk, 1,   name='self attention out linear')
        #self.o_linear = conv1d(self.dva, self.do, 1, stride=1, name='attention out   linear', bias=outBias, batchNorm=True, nAveraging=4)

        self.negativeInfinity = torch.tensor(-1e9, dtype=torch.float).to('cuda')

        if layers:
            #layers.addLayer(self.q_linear, inputLayers)
            layers.addLayer(self.k_linear, inputLayers)
            layers.addLayer(self.v_linear, inputLayers)
            #layers.addLayer(self.o_linear, [self.q_linear.index, self.v_linear.index, self.k_linear.index])
    
    def attention(self, q, k, v, mask, doMdR=None, debug=False):
        bs, qsl, sl = q.shape[0], q.shape[2], k.shape[-1]

        q_k_overlap = torch.matmul(q, k) #/  self.sqrt_dh
        # q_k_overlap = q_k_overlap * self.q_k_overlap_scale
        # q_k_overlap = q_k_overlap + self.q_k_overlap_bias
        
        #q_k_overlap.shape = [bs, h, qsl, sl]
        # if mask[0].sum(dim=1)<9 and self.training:
        #     print('before\n',q_k_overlap[0])
        #     print(self.q_k_overlap_GBN.m, self.q_k_overlap_GBN.s)
        #     print(self.q_k_overlap_GBN.bias, self.q_k_overlap_GBN.gamma)

        # if doMdR is not None:
        #     q_k_overlap = q_k_overlap.view(bs, 1, self.h, qsl, sl)
        #     q_k_MdR = doMdR.view(bs, 2, 1, qsl, sl)
        #     q_k_MdR = q_k_MdR.repeat(1, 1, self.h, 1, 1)
        #     q_k_overlap = torch.cat( (q_k_overlap, q_k_MdR), 1)
        #     q_k_overlap = self.overlap_pairFeature_conv(q_k_overlap).view(bs, self.h, qsl, sl)

        # masked ghost batch normalizatino of q_k_overlap
        q_k_overlap = q_k_overlap.transpose(1,2)
        q_k_overlap = q_k_overlap.contiguous().view(bs*qsl, self.h, sl)
        q_k_overlap = self.q_k_overlap_GBN(q_k_overlap, mask.repeat(1,1,qsl,1).view(bs*qsl,1,sl))
        q_k_overlap = q_k_overlap.transpose(1,2)
        q_k_overlap = q_k_overlap.contiguous().view(bs, self.h, qsl, sl)


        if mask is not None:
            if self.selfAttention:
                q_k_overlap = q_k_overlap.masked_fill(mask, self.negativeInfinity)
            mask = mask.transpose(2,3)
            q_k_overlap = q_k_overlap.masked_fill(mask, self.negativeInfinity)

        v_probability = F.softmax(q_k_overlap, dim=-1) # compute joint probability distribution for which values best correspond to the query

        # if mask[0].sum(dim=2)<9 and self.training:
        #     print('after\n',q_k_overlap[0])
        #     print(v_probability[0])

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
            print("v_weights\n",v_weights[0])
            print("v\n",v[0])

        output = torch.matmul(v_weights, v)

        if doMdR is not None:
            # doMdR is (bs, 2,  6, 12)
            doMdR = self.pairFeature_conv(doMdR)
            # doMdR is (bs, 8,  6, 12)
            doMdR = doMdR.view(bs, self.h, self.dvh, qsl, sl)
            # doMdR is (bs, 1, 8,  6, 12)

            # v_weights is (bs, 1,  6, 12)
            v_weights = v_weights.view(bs, self.h, 1, qsl, sl)
            # v_weights is (bs, 1, 1,  6, 12)
            v_weights = v_weights.repeat(1, 1, self.dvh, 1, 1)
            # v_weights is (bs, 1, 8,  6, 12)

            doMdR = v_weights * doMdR # weight each set of jet,dijet features
            doMdR = doMdR.sum(dim=4) # sum over jets
            #  doMdR is (bs, 1, 8, 6)
            doMdR = doMdR.transpose(2,3)
            # output is (bs, 1, 6, 8)
            output = output + doMdR

        if debug:
            print("output\n",output[0])
            input()
        return output

    def forward(self, q, k, v, q0=None, mask=None, doMdR=None, qLinear=0, debug=False, selfAttention=False):
        
        bs = q.shape[0]
        qsl = q.shape[2]
        sq = None
        k = self.k_linear(k, mask)
        v = self.v_linear(v, mask)

        #check if all items are going to be masked
        sl = mask.shape[1]
        vqk_mask = mask.sum(dim=1)==sl
        vqk_mask = vqk_mask.view(bs, 1, 1).repeat(1,1,qsl)

        # #hack to make unequal head dimensions 3 and 6, add three zero padded features before splitting into two heads of dim 6
        # q = F.pad(input=q, value=0, pad=(0,0,1,0,0,0))
        # k = F.pad(input=k, value=0, pad=(0,0,1,0,0,0))
        # v = F.pad(input=v, value=0, pad=(0,0,1,0,0,0))

        #split into heads
        k = k.view(bs, self.h, self.dh,  sl)
        v = v.view(bs, self.h, self.dvh, sl)
        
        # transpose to get dimensions bs * h * sl * (da//h==dh)
        #k = k.transpose(2,3)
        v = v.transpose(2,3)
        mask = mask.view(bs, 1, sl, 1)

        #now do q transformations iter number of times
        for i in range(1,self.iter+1):
            # if selfAttention:
            #     q = self.sq_linear(q)
            # else:
            #     q = self. q_linear(q, vqk_mask)

            q = q.view(bs, self.h, self.dh, qsl)
            q = q.transpose(2,3)

            # calculate attention 
            vqk = self.attention(q, k, v, mask, doMdR=doMdR, debug=debug) # outputs a linear combination of values (v) given the overlap of the queries (q) with the keys (k)
            # output is (bs, 1, 6, 8)

            # concatenate heads and put through final linear layer
            vqk = vqk.transpose(2,3).contiguous().view(bs, self.dva, qsl)

            # vqk0 = vqk.clone()
            # vqk = NonLU(vqk, self.training)
            # vqk = self.o_linear(vqk, vqk_mask)
            # vqk = vqk+vqk0

            # if debug:
            #     print("vqk\n",vqk[0])
            #     input()
            # if selfAttention:
            #     vqk = self.so_linear(vqk)
            # else:
            #     vqk = self. o_linear(vqk, vqk_mask)

            # if all the input items are masked, we don't want the bias term of the output layer to have any impact
            vqk = vqk.masked_fill(vqk_mask, 0)

            q = q0 + vqk

            if i==self.iter:
                q0 = q.clone()
            if not selfAttention:
                q = NonLU(q, self.training)
                
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
        # self.jetEmbed = conv1d(5, 8, 1, name='other jet embed', batchNorm=False)
        # self.jetConv1 = conv1d(8, 8, 1, name='other jet convolution 1', batchNorm=True, nAveraging=4)
        # self.jetConv2 = conv1d(5, 5, 1, name='other jet convolution 2', batchNorm=False)

        # layers.addLayer(self.jetEmbed)
        # layers.addLayer(self.jetConv1, [self.jetEmbed.index])
        # inputLayers.append(self.jetConv1.index)

        self.attention = MultiHeadAttention(   dim_query=self.ne, dim_key=5,    dim_value=5, dim_attention=self.ne, heads=1, dim_valueAttention=self.ne, dim_out=self.ne,
                                            groups_query=1,    groups_key=1, groups_value=1, 
                                            selfAttention=False, outBias=False, layers=layers, inputLayers=inputLayers,
                                            bothAttention=False,
                                            iterations=2)
        self.outputLayer = self.attention.v_linear.index
        #self.outputLayer = self.attention.o_linear.index

        
    def forward(self, q, kv, mask, q0=None, doMdR=None, qLinear=0, debug=False):
        if debug:
            print("q\n",   q[0])        
            print("kv\n",  kv[0])
            print("mask\n",mask[0])

        # kv = self.jetEmbed(kv, mask)
        # kv0 = kv.clone()
        # kv = NonLU(kv, self.training)

        # kv = self.jetConv1(kv, mask)
        # kv = kv+kv0
        # kv = NonLU(kv, self.training)        

        # kv = self.jetConv2(kv, mask)
        # kv = kv+kv0
        # kv = NonLU(kv, self.training)        

        q, q0 = self.attention(q, kv, kv, q0=q0, mask=mask, doMdR=doMdR, debug=debug, selfAttention=False)

        return q, q0


class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures, batchNorm=False):
        super(dijetReinforceLayer, self).__init__()
        self.nd = dijetFeatures
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|            
        self.conv = conv1d(self.nd, self.nd, 3, stride=3, name='dijet reinforce convolution', batchNorm=batchNorm, nAveraging=4)

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
        self.convJ = conv1d(self.nd, self.nd, 1, name='jet convolution', batchNorm=True, nAveraging=4)
        self.reinforce2 = dijetReinforceLayer(self.nd, batchNorm=False)

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
            self.multijetAttention = multijetAttention(self.nj ,self.nd, self.na, nh=nhOptions[1], layers=layers, inputLayers=[self.reinforce2.conv.index])
            self.outputLayer = self.multijetAttention.outputLayer

    def forward(self, j, d, j0=None, d0=None, o=None, mask=None, doMdR=None, debug=False):

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
            d0 = d.clone()
            d, d0 = self.multijetAttention(d, o, mask, q0=d0, doMdR=doMdR, debug=debug)

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
        self.conv = conv1d(self.nq, self.nq, 3, stride=3, name='quadjet reinforce convolution', batchNorm=batchNorm, nAveraging=4)

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
        self.convD = conv1d(self.nq, self.nq, 1, name='dijet convolution', batchNorm=True, nAveraging=4)
        self.reinforce2 = quadjetReinforceLayer(self.nq, batchNorm=False)

        layers.addLayer(self.reinforce1.conv, inputLayers)
        layers.addLayer(self.convD, [inputLayers[0]])
        layers.addLayer(self.reinforce2.conv, [self.convD.index, self.reinforce1.conv.index])

    def forward(self, d, q, d0=None, q0=None, o=None, mask=None, debug=False):

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


    


class InputGBN(nn.Module):
    def __init__(self, jetFeatures, ancillaryFeatures, ancillaryDijetFeatures, ancillaryQuadjetFeatures, useOthJets='', device='cuda'):
        super(InputGBN, self).__init__()

        self.debug = False
        self.ancillaryFeatures = ancillaryFeatures
        self.nj = jetFeatures
        self.nA  = len(ancillaryFeatures)
        #self.nAd = ancillaryDijetFeatures #engineered dijet features
        #self.nAq = ancillaryQuadjetFeatures #engineered quadjet features
        self.device = device
        self.useOthJets = bool(useOthJets)

        self.ancillaryGBN = GhostBatchNorm1d(self.nA)

        self.canJetScaler = scaler(self.nj)
        if self.useOthJets:
            self.othJetScaler = scaler(self.nj+1)
            self.doMdRScaler = scaler(2, shape=(1,2,1,1))
            self.doMdRScaler.m[0,0,0,0] = 100.
            self.doMdRScaler.m[0,1,0,0] = np.pi
            self.doMdRScaler.s[0,0,0,0] = 50.
            self.doMdRScaler.s[0,1,0,0] = np.pi/2
        self.dijetScaler = scaler(4)
        self.quadjetScaler = scaler(4)

        # embed inputs to dijetResNetBlock in target feature space
        self.jetPtGBN = GhostBatchNorm1d(1)#only apply to pt
        self.jetEtaGBN = GhostBatchNorm1d(1, bias=False)#learn scale for eta, but keep bias at zero for eta flip symmetry to make sense
        self.jetMassGBN = GhostBatchNorm1d(1)#only apply to mass
        if self.useOthJets:
            self.othJetPtGBN = GhostBatchNorm1d(1)
            self.othJetEtaGBN = GhostBatchNorm1d(1, bias=False)
            self.othJetMassGBN = GhostBatchNorm1d(1)
            
        self.dijetPtGBN = GhostBatchNorm1d(1)#only apply to pt
        self.dijetEtaGBN = GhostBatchNorm1d(1, bias=False)#learn scale for eta, but keep bias at zero for eta flip symmetry to make sense
        self.dijetMassGBN = GhostBatchNorm1d(1)#only apply to mass
        self.dijetdRjjGBN = GhostBatchNorm1d(1)#only apply to dRjj
        #self.dijetGBN = GhostBatchNorm1d(self.nAd)

        self.quadjetPtGBN = GhostBatchNorm1d(1)#only apply to pt
        self.quadjetEtaGBN = GhostBatchNorm1d(1, bias=False)#learn scale for eta, but keep bias at zero for eta flip symmetry to make sense
        self.quadjetMassGBN = GhostBatchNorm1d(1)#only apply to mass
        self.quadjetdRddGBN = GhostBatchNorm1d(1)#only apply to dRdd

    def print(self):
        print("Jet GBN:")
        print("          pt,    eta,      m")
        print(" mean ",end='')
        vectorPrint([self.jetPtGBN.m[0][0], self.jetEtaGBN.m[0][0], self.jetMassGBN.m[0][0]])
        print("  std ", end='')
        vectorPrint([self.jetPtGBN.s[0][0], self.jetEtaGBN.s[0][0], self.jetMassGBN.s[0][0]])
        print("gamma ", end='')
        vectorPrint([self.jetPtGBN.gamma.data[0], self.jetEtaGBN.gamma.data[0], self.jetMassGBN.gamma.data[0]])
        print(" bias ", end='')
        vectorPrint([self.jetPtGBN.bias.data[0], self.jetEtaGBN.bias.data[0], self.jetMassGBN.bias.data[0]])

        if self.useOthJets:
            print("\nothJet GBN:")
            print("          pt,    eta,      m")
            print(" mean ",end='')
            vectorPrint([self.othJetPtGBN.m[0][0], self.othJetEtaGBN.m[0][0], self.othJetMassGBN.m[0][0]])
            print("  std ", end='')
            vectorPrint([self.othJetPtGBN.s[0][0], self.othJetEtaGBN.s[0][0], self.othJetMassGBN.s[0][0]])
            print("gamma ", end='')
            vectorPrint([self.othJetPtGBN.gamma.data[0], self.othJetEtaGBN.gamma.data[0], self.othJetMassGBN.gamma.data[0]])
            print(" bias ", end='')
            vectorPrint([self.othJetPtGBN.bias.data[0], self.othJetEtaGBN.bias.data[0], self.othJetMassGBN.bias.data[0]])

        print("\nDijet GBN:")
        print("           m,     dR")
        print(" mean ",end='')
        vectorPrint(self.dijetGBN.m[0].transpose(0,1)[0])
        print("  std ", end='')
        vectorPrint(self.dijetGBN.s[0].transpose(0,1)[0])
        print("gamma ", end='')
        vectorPrint(self.dijetGBN.gamma.data)
        print(" bias ", end='')
        vectorPrint(self.dijetGBN.bias.data)

        print("\nAncillary GBN:")
        print("               "+" ".join(self.ancillaryFeatures))
        print(" mean ",end='')
        vectorPrint(self.ancillaryGBN.m[0].transpose(0,1)[0])
        print("  std ", end='')
        vectorPrint(self.ancillaryGBN.s[0].transpose(0,1)[0])
        print("gamma ", end='')
        vectorPrint(self.ancillaryGBN.gamma.data)
        print(" bias ", end='')
        vectorPrint(self.ancillaryGBN.bias.data)

    def forward(self, j, o, a):
        n = j.shape[0]
        j = j.view(n,self.nj,12)
        a = a.view(n,self.nA,1)
        a = self.ancillaryGBN( a )

        d, dPxPyPzE, dRjj = addFourVectors(j[:,:,(0,2,4,6,8,10)], 
                                           j[:,:,(1,3,5,7,9,11)])
        #d = torch.cat( (d4v, dRjj) , 1 )

        q, qPxPyPzE, dRdd = addFourVectors(d[:,:,(0,2,4)],
                                           d[:,:,(1,3,5)], 
                                           v1PxPyPzE = dPxPyPzE[:,:,(0,2,4)],
                                           v2PxPyPzE = dPxPyPzE[:,:,(1,3,5)])
        #q = torch.cat( (q4v, dRdd), 1 )


        # Scale inputs
        j = self.canJetScaler(j)
        d = self.dijetScaler(d)
        q = self.quadjetScaler(q)

        # Learn optimal scale for these inputs
        jPt, jEta, jPhi, jMass = j[:,0:1,:], j[:,1:2,:], j[:,2:3,:], j[:,3:4,:]
        jPt, jEta,       jMass = self.jetPtGBN( jPt ), self.jetEtaGBN( jEta ), self.jetMassGBN( jMass )
        j = torch.cat( (jPt, jEta, jPhi, jMass), dim=1 )

        dPt, dEta, dPhi, dMass       = d[:,0:1,:], d[:,1:2,:], d[:,2:3,:], d[:,3:4,:]
        dPt, dEta,       dMass, dRjj = self.dijetPtGBN( dPt ), self.dijetEtaGBN( dEta ), self.dijetMassGBN( dMass ), self.dijetdRjjGBN( dRjj )
        d = torch.cat( (dPt, dEta, dPhi, dMass, dRjj), dim=1 )

        qPt, qEta, qPhi, qMass       = q[:,0:1,:], q[:,1:2,:], q[:,2:3,:], q[:,3:4,:]
        qPt, qEta,       qMass, dRdd = self.quadjetPtGBN( qPt ), self.quadjetEtaGBN( qEta ), self.quadjetMassGBN( qMass ), self.quadjetdRddGBN( dRdd )
        q = torch.cat( (qPt, qEta, qPhi, qMass, dRdd), dim=1 )

        #d = self.dijetGBN(d)
        #q = self.quadjetGBN(q)

        # # attach ancillary features at desired level
        # a = a.repeat(1, 1, 3) # use in quadjet image
        # q = torch.cat( (q, a), 1 )


        # do the same data prep for the other jets if we are using them
        mask, doMdR = None, None
        if self.useOthJets:
            o = o.view(n,5,12)

            # compute matrix of trijet masses and opening angles between dijets and other jets
            doMdR = matrixMdR(d, o, v1PxPyPzE = dPxPyPzE)
            doMdR = self.doMdRScaler(doMdR)

            o = self.othJetScaler(o)

            mask = o[:,4,:]==-1

            oPt, oEta, oPhi, oMass, oIsSelJet = o[:,0:1,:], o[:,1:2,:], o[:,2:3,:], o[:,3:4,:], o[:,4:5,:]
            oPt, oEta,       oMass            = self.othJetPtGBN( oPt, mask ), self.othJetEtaGBN( oEta, mask ), self.othJetMassGBN( oMass, mask )
            o = torch.cat( (oPt, oEta, oPhi, oMass, oIsSelJet), dim=1 )


        return j, o, a, mask, d ,q, doMdR


class ResNet(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, ancillaryFeatures, useOthJets='', device='cuda', nClasses=1):
        super(ResNet, self).__init__()
        self.debug = False
        self.nj = jetFeatures
        self.nA = len(ancillaryFeatures)
        self.nd, self.nAd =   dijetFeatures, 0#+self.nA #total dijet features, engineered dijet features
        self.nq, self.nAq =   dijetFeatures, 0+self.nA #total quadjet features, engineered quadjet features
        #self.nAe = nAncillaryFeatures
        self.ne = dijetFeatures #combinatoricFeatures
        self.device = device
        dijetBottleneck   = None
        self.name = 'ResNet'+('+'+useOthJets if useOthJets else '')+'_%d'%(dijetFeatures)
        self.useOthJets = bool(useOthJets)
        self.nClasses = nClasses
        self.store = None
        self.storeData = {}
        self.onnx = False

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        self.layers = layerOrganizer()

        # this module handles input shifting scaling and learns the optimal scale and shift for the appropriate inputs
        self.inputGBN = InputGBN(self.nj, ancillaryFeatures, self.nAd, self.nAq, useOthJets=useOthJets, device=device)
            
        self.jetEmbed = conv1d(self.nj, self.nd, 1, name='jet embed', batchNorm=False)
        self.dijetEmbed1 = conv1d(5+self.nAd, self.nd, 1, name='dijet embed', batchNorm=False)

        self.layers.addLayer(self.jetEmbed)
        self.layers.addLayer(self.dijetEmbed1)


        # Stride=3 Kernel=3 reinforce dijet features, in parallel update jet features for next reinforce layer
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock = dijetResNetBlock(self.nj, self.nd, device=self.device, useOthJets=useOthJets, layers=self.layers, inputLayers=[self.jetEmbed.index, self.dijetEmbed1.index])
        

        # embed inputs to quadjetResNetBlock in target feature space
        self.dijetEmbed2 = conv1d(self.nd, self.nq, 1, name='dijet embed', batchNorm=False)
        self.quadjetEmbed = conv1d(5+self.nAq, self.nq, 1, name='quadjet embed', batchNorm=False)

        self.layers.addLayer(self.dijetEmbed2, [self.dijetResNetBlock.outputLayer])
        self.layers.addLayer(self.quadjetEmbed, startIndex=self.dijetEmbed2.index)

        # Stride=3 Kernel=3 reinforce quadjet features, in parallel update dijet features for next reinforce layer
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4|2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock = quadjetResNetBlock(self.nd, self.nq, device=self.device, layers=self.layers, inputLayers=[self.dijetEmbed2.index, self.quadjetEmbed.index])

        # Event level convolutions, eta/phi dependence has been averaged out
        self.eventConv1 = conv1d(self.ne, self.ne, 1, name='event convolution 1', batchNorm=True) 
        self.eventConv2 = conv1d(self.ne, self.ne, 1, name='event convolution 2', batchNorm=False)

        # Calculate score for each quadjet, add them together with corresponding weight, and go to final output layer
        self.select_q = conv1d(self.ne, 1, 1, name='quadjet selector', batchNorm=True) 
        self.out      = conv1d(self.ne, self.nClasses, 1, name='out', batchNorm=True)


        self.layers.addLayer(self.eventConv1, [self.quadjetResNetBlock.reinforce2.conv.index])
        self.layers.addLayer(self.eventConv2, [self.eventConv1.index])
        self.layers.addLayer(self.select_q, [self.eventConv2.index])
        self.layers.addLayer(self.out,      [self.eventConv2.index, self.select_q.index])

        self.negativePhiFour = torch.tensor([1,1,-1,1], dtype=torch.float).to('cuda').view(1,4,1)
        self.negativeEtaFour = torch.tensor([1,-1,1,1], dtype=torch.float).to('cuda').view(1,4,1)
        self.negativePhiFive = torch.tensor([1,1,-1,1,1], dtype=torch.float).to('cuda').view(1,5,1)
        self.negativeEtaFive = torch.tensor([1,-1,1,1,1], dtype=torch.float).to('cuda').view(1,5,1)
        

    def rotate(self, j, R): # j[event, mu, jet], mu=2 is phi
        jPhi = j[:,2:3,:]
        jPhi = (jPhi + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
        j = torch.cat( (j[:,:2],jPhi,j[:,3:]), dim=1)
        return j

    def flipPhi(self, j, four=True): # j[event, mu, jet], mu=2 is phi
        if four:
            j = j * self.negativePhiFour
        else:
            j = j * self.negativePhiFive
        return j

    def flipEta(self, j, four=True): # j[event, mu, jet], mu=1 is eta
        if four:
            j = j * self.negativeEtaFour
        else:
            j = j * self.negativeEtaFive
        return j

    def makeSymmetriesVector(self, v, randomR, four=True):
        if self.training:
            v = self.rotate(v, randomR)

        n = v.shape[0]
        features = 4 if four else 5

        v = v.repeat(4, 1, 1)
        v = v.view(4, n, features, -1)
        vR, vRP, vRE, vRPE = v[0], v[1], v[2], v[3]
        
        vRP  = self.flipPhi(vRP,  four)
        vRE  = self.flipEta(vRE,  four)
        vRPE = self.flipPhi(vRPE, four)
        vRPE = self.flipEta(vRPE, four)
        v = torch.cat( (vR, vRP, vRE, vRPE), dim=0)

        return v
            
            
    def makeSymmetries(self, j, o, a, mask, d, q, doMdR):
        # Copy inputs 4 times to compute each of the symmetry transformations 
        # Randomly rotate the event in phi during training
        n = j.shape[0]
        randomR = 2*torch.rand(n,1,1, device='cuda') if self.training else None
        j = self.makeSymmetriesVector(j, randomR, four=True)
        d = self.makeSymmetriesVector(d, randomR, four=False)
        q = self.makeSymmetriesVector(q, randomR, four=False)

        # attach ancillary features at desired level
        a = a.repeat(4, 1, 3) # use in quadjet image
        q = torch.cat( (q, a), 1 )

        if self.useOthJets:
            o = self.makeSymmetriesVector(o, randomR, four=False)
            mask  = mask .repeat(4, 1)
            doMdR = doMdR.repeat(4, 1, 1, 1)

        return j, o, mask, d, q, doMdR


    def EtaPhiInvariantPart(self, j, o, mask, d, q, doMdR):
        #n = j.shape[0]

        #
        # Build up dijet pixels with jet pixels and dijet ancillary features
        #

        # Embed the jet 4-vectors and dijet ancillary features into the target feature space
        j = self.jetEmbed(j)
        d = self.dijetEmbed1(d) 
        j0 = j.clone()
        d0 = d.clone()
        j = NonLU(j, self.training)
        d = NonLU(d, self.training)

        d, d0 = self.dijetResNetBlock(j, d, j0=j0, d0=d0, o=o, mask=mask, doMdR=doMdR, debug=self.debug)

        if self.store:
            self.storeData['dijets'] = d[0].detach().to('cpu').numpy()

        #
        # Build up quadjet pixels with dijet pixels and dijet ancillary features
        #
            
        # Embed the dijet pixels and quadjet ancillary features into the target feature space
        d = self.dijetEmbed2(d)
        q = self.quadjetEmbed(q)        
        d = d+d0 # d0 from dijetResNetBlock since the number of dijet and quadjet features are the same
        q0 = q.clone()
        d = NonLU(d, self.training)
        q = NonLU(q, self.training)

        q, q0 = self.quadjetResNetBlock(d, q, d0=d0, q0=q0, o=o, mask=mask, debug=self.debug) 

        if self.store:
            self.storeData['quadjets'] = q[0].detach().to('cpu').numpy()

        return q, q0


    def forward(self, j, o, a):
        j, o, a, mask, d, q, doMdR = self.inputGBN(j, o, a) # format inputs to array of objects and apply scalers and GBNs
        n = j.shape[0]

        if self.store:
            self.storeData[  'canJets'] = j[0].detach().to('cpu').numpy()
            self.storeData['otherJets'] = o[0].detach().to('cpu').numpy()

        # #can do these here because they have no eta/phi information
        # d = self.dijetEmbed1(d) 
        # q = self.quadjetEmbed(q)        

        # Copy inputs nRF times and apply a different symmetry transformation to each copy
        j, o, mask, d, q, doMdR = self.makeSymmetries(j, o, a, mask, d, q, doMdR)

        # compute the quadjet pixels and average them over the symmetry transformations
        q, q0 = self.EtaPhiInvariantPart(j, o, mask, d, q, doMdR)
        q, q0 = q.view(4, n, self.nq, 3), q0.view(4, n, self.nq, 3)
        q, q0 = q.mean(dim=0), q0.mean(dim=0)

        # Everything from here on out has no dependence on eta/phi flips and minimal dependence on phi rotations
        if self.store:
            self.storeData['quadjets_sym'] = q[0].detach().to('cpu').numpy()

        q = self.eventConv1(q)
        q = q+q0
        q = NonLU(q, self.training)

        q = self.eventConv2(q)
        q = q+q0
        q = NonLU(q, self.training)

        if self.store:
            self.storeData['quadjets_sym_eventConv'] = q[0].detach().to('cpu').numpy()

        #compute a score for each event view (quadjet) 
        q_score = self.select_q(q)
        #convert the score to a 'probability' with softmax. This way the classifier is learning which view is most relevant to the classification task at hand.
        q_score = F.softmax(q_score, dim=-1)
        #add together the quadjets with their corresponding probability weight
        e = torch.matmul(q, q_score.transpose(1,2))
        q_score = q_score.view(n,3)

        if self.store:
            self.storeData['q_score'] = q_score[0].detach().to('cpu').numpy()
            self.storeData['event'] = e[0].detach().to('cpu').numpy()

        #project the final event-level pixel into the class score space
        c_score = self.out(e)
        c_score = c_score.view(n, self.nClasses)

        if self.store or self.onnx:
            c_score = F.softmax(c_score, dim=1)
        if self.store:
            self.storeData['c_score'] = c_score[0].detach().to('cpu').numpy()

        return c_score, q_score


    def writeStore(self):
        np.save(self.store, self.storeData)


class ResNetEnsemble(nn.Module):
    def __init__(self, ResNets):
        super(ResNetEnsemble, self).__init__()
        self.net0 = ResNets[0]
        self.net1 = ResNets[1]
        self.net2 = ResNets[2]

        self.training = False

        self.net0.onnx = False
        self.net1.onnx = False
        self.net2.onnx = False
        
    def forward(self, j, o, d, q):
        c_score0, q_score0 = self.net0(j, o, d, q)
        c_score1, q_score1 = self.net1(j, o, d, q)
        c_score2, q_score2 = self.net2(j, o, d, q)

        q_score = torch.stack([q_score0, q_score1, q_score2])
        q_score = q_score.mean(dim=0)

        c_score = torch.stack([c_score0, c_score1, c_score2])
        c_score = c_score.mean(dim=0)
        c_score = F.softmax(c_score, dim=-1)

        return c_score, q_score

    @torch.no_grad()
    def exportONNX(self, modelONNX):
        # Create a random input for the network. The onnx export will use this to trace out all the operations done by the model.
        # We can later check that the model output is the same with onnx and pytorch evaluation.
        # test_input = (torch.ones(1, 4, 12, requires_grad=True).to('cuda'),
        #               torch.ones(1, 5, 12, requires_grad=True).to('cuda'),
        #               torch.ones(1, self.net.nAd, 6, requires_grad=True).to('cuda'),
        #               torch.ones(1, self.net.nAq, 3, requires_grad=True).to('cuda'),
        #               )
        J = torch.tensor([182.747, 141.376, 109.942, 50.8254, 182.747, 109.942, 141.376, 50.8254, 182.747, 50.8254, 141.376, 109.942, 
                          0.772827, 1.2832, 1.44385, 2.06543, 0.772827, 1.44385, 1.2832, 2.06543, 0.772827, 2.06543, 1.2832, 1.44385, 
                          2.99951, -0.797241, 0.561157, -2.83203, 2.99951, 0.561157, -0.797241, -2.83203, 2.99951, -2.83203, -0.797241, 0.561157, 
                          14.3246, 10.5783, 13.1129, 7.70751, 14.3246, 13.1129, 10.5783, 7.70751, 14.3246, 7.70751, 10.5783, 13.1129],
                         requires_grad=False).to('cuda').view(1,48)
        O = torch.tensor([22.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                          0.0322418, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          -0.00404358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                          4.01562, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                          0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         requires_grad=False).to('cuda').view(1,60)
        D = torch.tensor([316.5, 157.081, 284.569, 160.506, 142.039, 159.722, 
                          2.53827, 2.95609, 2.529, 2.17997, 1.36923, 1.36786],
                         requires_grad=False).to('cuda').view(1,12)
        Q = torch.tensor([3.18101, 2.74553, 2.99015, 
                          525.526, 525.526, 525.526, 
                          4.51741, 4.51741, 4.51741, 
                          0.554433, 0.554433, 0.554433, 
                          4, 4, 4, 
                          2016, 2016, 2016],
                         requires_grad=False).to('cuda').view(1,18)
        # Export the model
        self.eval()
        torch_out = self(J, O, D, Q)
        print("test output:",torch_out)
        print("Export ONNX:",modelONNX)
        torch.onnx.export(self,                                            # model being run
                          (J, O, D, Q),                                    # model input (or a tuple for multiple inputs)
                          modelONNX,                                       # where to save the model (can be a file or file-like object)
                          export_params=True,                              # store the trained parameter weights inside the model file
                          #opset_version= 7,                               # the ONNX version to export the model to
                          #do_constant_folding=True,                       # whether to execute constant folding for optimization
                          input_names  = ['J','O','D','Q'],                # the model's input names
                          output_names = ['c_score', 'q_score'],           # the model's output names
                          #dynamic_axes={ 'input' : {0 : 'batch_size'},    # variable lenght axes
                          #              'output' : {0 : 'batch_size'}}
                          verbose = False
                          )

        # import onnx
        # onnx_model = onnx.load(self.modelONNX)
        # # Check that the IR is well formed
        # onnx.checker.check_model(onnx_model)






# 
# Simple networks for comparison with nominal
# 

class BasicCNN(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, useOthJets='', device='cuda', nClasses=1):
        super(BasicCNN, self).__init__()
        self.debug = False
        self.nj = jetFeatures
        self.nd, self.nAd = dijetFeatures, 2 #total dijet features, engineered dijet features
        self.nq, self.nAq = quadjetFeatures, 6 #total quadjet features, engineered quadjet features
        #self.nAe = nAncillaryFeatures
        self.ne = combinatoricFeatures
        self.device = device
        dijetBottleneck   = None
        self.name = 'BasicCNN'+('+'+useOthJets if useOthJets else '')+'_%d_%d_%d'%(dijetFeatures, quadjetFeatures, self.ne)
        self.useOthJets = bool(useOthJets)
        self.nClasses = nClasses
        self.store = None
        self.storeData = {}
        self.onnx = False

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        self.layers = layerOrganizer()


        # make fixed convolution to compute average of dijet pixel pairs (symmetric bilinear)
        self.sym = nn.Conv1d(self.nq, self.nq, 2, stride=2, bias=False, groups=self.nq)
        self.sym.weight.data.fill_(0.5)
        self.sym.weight.requires_grad = False

        # make fixed convolution to compute difference of dijet pixel pairs (antisymmetric bilinear)
        self.antisym = nn.Conv1d(self.nq, self.nq, 2, stride=2, bias=False, groups=self.nq)
        self.antisym.weight.data.fill_(0.5)
        self.antisym.weight.data[:,:,1] *= -1
        self.antisym.weight.requires_grad = False


        # this module handles input shifting scaling and learns the optimal scale and shift for the appropriate inputs
        self.inputGBN = InputGBN(self.nj, self.nAd, self.nAq, useOthJets=useOthJets, device=device)
            
        #self.jetEmbed     = conv1d(self.nj, self.nd, 1, name='jet embed', batchNorm=False)
        self.dijetEmbed   = conv1d(self.nAd, self.nd, 1, name='dijet embed', batchNorm=False)
        self.quadjetEmbed = conv1d(self.nAq, self.nq, 1, name='quadjet embed', batchNorm=False)

        #self.layers.addLayer(self.jetEmbed)
        self.layers.addLayer(self.dijetEmbed)

        self.jetsToDijets     = conv1d(self.nj, self.nd, 2, stride=2, name='jets to dijets',     batchNorm=True, nAveraging=4)
        self.layers.addLayer(self.jetsToDijets)#, [self.jetEmbed.index])

        self.quadjetInput = self.jetsToDijets.index
        if useOthJets:
            self.multijetAttention = multijetAttention(None, self.nd, None, nh=1, layers=self.layers, inputLayers=[self.jetsToDijets.index])
            self.quadjetInput = self.multijetAttention.outputLayer

        self.dijetsToQuadjets = conv1d(self.nd, self.nd, 2, stride=2, name='dijets to quadjets', batchNorm=True, nAveraging=4)
        # self.quadjetsToEvent  = conv1d(self.nd, self.nd, 3, stride=1, name='quadjets to event',  batchNorm=True, nAveraging=4)

        self.layers.addLayer(self.dijetsToQuadjets, [self.jetsToDijets.index])
        # self.layers.addLayer(self.quadjetsToEvent, [self.dijetsToQuadjets.index])
        self.layers.addLayer(self.quadjetEmbed, startIndex=self.dijetsToQuadjets.index)

        # Calculate score for each quadjet, add them together with corresponding weight, and go to final output layer
        self.select_q = conv1d(self.ne, 1, 1, name='quadjet selector', batchNorm=True) 
        self.out      = conv1d(self.ne, self.nClasses, 1, name='out', batchNorm=True)

        self.layers.addLayer(self.select_q, [self.dijetsToQuadjets.index])
        self.layers.addLayer(self.out,        [self.select_q.index])

        # # Event level convolutions, eta/phi dependence has been averaged out
        # self.eventConv1 = conv1d(self.ne, self.ne, 1, name='event convolution 1', batchNorm=True) 
        # self.eventConv2 = conv1d(self.ne, self.ne, 1, name='event convolution 2', batchNorm=False)

        # # Calculate score for each quadjet, add them together with corresponding weight, and go to final output layer
        # self.select_q = conv1d(self.ne, 1, 1, name='quadjet selector', batchNorm=True) 
        # self.out      = conv1d(self.ne, self.nClasses, 1, name='out', batchNorm=True)

        # self.layers.addLayer(self.select_q, [self.eventConv2.index])
        # self.layers.addLayer(self.eventConv1, [self.quadjetsToEvent.index])
        # self.layers.addLayer(self.eventConv2, [self.eventConv1.index])
        # self.layers.addLayer(self.out,        [self.eventConv2.index])

        self.negativePhiCanJets = torch.tensor([1,1,-1,1], dtype=torch.float).to('cuda').view(1,4,1)
        self.negativeEtaCanJets = torch.tensor([1,-1,1,1], dtype=torch.float).to('cuda').view(1,4,1)
        self.negativePhiOthJets = torch.tensor([1,1,-1,1,1], dtype=torch.float).to('cuda').view(1,5,1)
        self.negativeEtaOthJets = torch.tensor([1,-1,1,1,1], dtype=torch.float).to('cuda').view(1,5,1)
        

    def rotate(self, j, R): # j[event, mu, jet], mu=2 is phi
        jPhi = j[:,2:3,:]
        jPhi = (jPhi + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
        j = torch.cat( (j[:,:2],jPhi,j[:,3:]), dim=1)
        return j

    def flipPhi(self, j, canJets=True): # j[event, mu, jet], mu=2 is phi
        if canJets:
            j = j * self.negativePhiCanJets
        else:
            j = j * self.negativePhiOthJets
        return j

    def flipEta(self, j, canJets=True): # j[event, mu, jet], mu=1 is eta
        if canJets:
            j = j * self.negativeEtaCanJets
        else:
            j = j * self.negativeEtaOthJets
        return j


    def makeSymmetries(self, j, o, mask, d, q):
        n = j.shape[0]
        # Copy inputs nRF times to compute each of the symmetry transformations 
        j = j.repeat(self.nRF, 1, 1)
        d = d.repeat(self.nRF, 1, 1)
        q = q.repeat(self.nRF, 1, 1)
        if self.useOthJets:
            o    = o   .repeat(self.nRF, 1, 1)
            mask = mask.repeat(self.nRF, 1)

        # Randomly rotate the event in phi during training
        if self.training:
            randomR = 2*torch.rand(n,1,1, device='cuda')

        # apply each of the symmetry transformations over which we will average after learning eta/phi dependent features
        j = j.view(4, n, self.nj, 12)
        jR, jRP, jRE, jRPE = j[0], j[1], j[2], j[3]
        if self.training: 
            jR = self.rotate(jR, randomR)
        if self.useOthJets:
            o = o.view(4, n, 5, 12)
            oR, oRP, oRE, oRPE = o[0], o[1], o[2], o[3]
            if self.training: 
                oR = self.rotate(oR, randomR)

        #flip phi
        if self.training: 
            jRP = self.rotate( jRP, randomR)
        jRP = self.flipPhi(jRP)
        if self.useOthJets:
            if self.training: 
                oRP = self.rotate( oRP, randomR)
            oRP = self.flipPhi(oRP, canJets=False)

        #flip eta
        if self.training:
            jRE = self.rotate( jRE, randomR)
        jRE = self.flipEta(jRE)
        if self.useOthJets:
            if self.training: 
                oRE = self.rotate( oRE, randomR)
            oRE = self.flipEta(oRE, canJets=False)

        #flip phi and eta
        if self.training: 
            jRPE = self.rotate( jRPE, randomR)
        jRPE = self.flipPhi(jRPE)
        jRPE = self.flipEta(jRPE)
        if self.useOthJets:
            if self.training: 
                oRPE = self.rotate( oRPE, randomR)
            oRPE = self.flipPhi(oRPE, canJets=False)
            oRPE = self.flipEta(oRPE, canJets=False)


        j = torch.cat( (jR, jRP, jRE, jRPE), dim=0)
        if self.useOthJets:
            o = torch.cat( (oR, oRP, oRE, oRPE), dim=0)

        return j, o, mask, d, q


    def EtaPhiInvariantPart(self, j, o, mask, d, q):
        n = j.shape[0]

        # Make dijets from jets, add them to embedded dijet ancillary features and NonLU the result
        d = d + self.jetsToDijets(j)

        if self.useOthJets:
            d0 = d.clone()
            d = NonLU(d, self.training)
            d, _ = self.multijetAttention(d, o, mask, q0=d0)
        else:
            d = NonLU(d, self.training)

        d_sym     = self.    sym(d)       # (d[:,:,(0,2,4)] + d[:,:,(1,3,5)])/2
        d_antisym = self.antisym(d).abs() #((d[:,:,(0,2,4)] - d[:,:,(1,3,5)])/2).abs()
        d = torch.cat( (d_sym[:,:, 0:1], d_antisym[:,:, 0:1],
                        d_sym[:,:, 1:2], d_antisym[:,:, 1:2],
                        d_sym[:,:, 2:3], d_antisym[:,:, 2:3]), 2)

        # Make quadjets from dijets, add them to embedded quadjet ancillary features and NonLU the result
        q = NonLU(q + self.dijetsToQuadjets(d), self.training)

        return q#, e0


    def forward(self, j, o, d, q):
        j, o, mask, d, q = self.inputGBN(j, o, d, q) # format inputs to array of objects and apply scalers and GBNs
        n = j.shape[0]

        #can do these here because they have no eta/phi information
        d = self.dijetEmbed(d) 
        q = self.quadjetEmbed(q)        

        # Copy inputs nRF times and apply a different symmetry transformation to each copy
        j, o, mask, d, q = self.makeSymmetries(j, o, mask, d, q)

        # compute the quadjet pixels and average them over the symmetry transformations
        q = self.EtaPhiInvariantPart(j, o, mask, d, q)
        q = q.view(self.nRF, n, self.nq, 3)
        q = q.mean(dim=0)

        #compute a score for each event view (quadjet) 
        q_score = self.select_q(q)
        #convert the score to a 'probability' with softmax. This way the classifier is learning which view is most relevant to the classification task at hand.
        q_score = F.softmax(q_score, dim=-1)
        #add together the quadjets with their corresponding probability weight
        e = torch.matmul(q, q_score.transpose(1,2))
        q_score = q_score.view(n,3)

        #project the final event-level pixel into the class score space
        c_score = self.out(e)
        c_score = c_score.view(n, self.nClasses)

        return c_score, q_score


class BasicDNN(nn.Module):
    def __init__(self, jetFeatures, dijetFeatures, quadjetFeatures, combinatoricFeatures, useOthJets='', device='cuda', nClasses=1):
        super(BasicDNN, self).__init__()
        self.debug = False
        self.nj = jetFeatures
        self.nd, self.nAd = dijetFeatures, 2 #total dijet features, engineered dijet features
        self.nq, self.nAq = quadjetFeatures, 6 #total quadjet features, engineered quadjet features
        #self.nAe = nAncillaryFeatures
        self.ne = combinatoricFeatures
        self.device = device
        dijetBottleneck   = None
        self.name = 'BasicDNN_128_128_128_pdrop0.4'
        self.useOthJets = bool(useOthJets)
        self.nClasses = nClasses
        self.store = None
        self.storeData = {}
        self.onnx = False

        self.doFlip = True
        self.nR     = 1
        self.R      = [2.0/self.nR * i for i in range(self.nR)]
        self.nRF    = self.nR * (4 if self.doFlip else 1)

        self.layers = layerOrganizer()


        # this module handles input shifting scaling and learns the optimal scale and shift for the appropriate inputs
        self.inputGBN = InputGBN(self.nj, self.nAd, self.nAq, useOthJets=useOthJets, device=device)

        self.conv1 = conv1d(16+12+8, 128, 1, name='conv1 36->128',batchNorm=False, nAveraging=4)
        self.layers.addLayer(self.conv1)
        self.drop1 = nn.Dropout(p=0.4)

        self.conv2 = conv1d(128, 128, name='conv2 128->128', batchNorm=True, nAveraging=4)
        self.layers.addLayer(self.conv2, [self.conv1.index])
        self.drop2 = nn.Dropout(p=0.4)

        self.conv3 = conv1d(128, 128, name='conv3 128->128', batchNorm=False, nAveraging=4)
        self.layers.addLayer(self.conv3, [self.conv2.index])            
        self.drop3 = nn.Dropout(p=0.4)

        self.out      = conv1d(128, self.nClasses, 1, name='out', batchNorm=True)
        self.layers.addLayer(self.out,        [self.conv3.index])


        self.negativePhiCanJets = torch.tensor([1,1,-1,1], dtype=torch.float).to('cuda').view(1,4,1)
        self.negativeEtaCanJets = torch.tensor([1,-1,1,1], dtype=torch.float).to('cuda').view(1,4,1)
        self.negativePhiOthJets = torch.tensor([1,1,-1,1,1], dtype=torch.float).to('cuda').view(1,5,1)
        self.negativeEtaOthJets = torch.tensor([1,-1,1,1,1], dtype=torch.float).to('cuda').view(1,5,1)
        

    def rotate(self, j, R): # j[event, mu, jet], mu=2 is phi
        jPhi = j[:,2:3,:]
        jPhi = (jPhi + 1 + R)%2 - 1 # add 1 to change phi coordinates from [-1,1] to [0,2], add the rotation R modulo 2 and change back to [-1,1] coordinates
        j = torch.cat( (j[:,:2],jPhi,j[:,3:]), dim=1)
        return j

    def flipPhi(self, j, canJets=True): # j[event, mu, jet], mu=2 is phi
        if canJets:
            j = j * self.negativePhiCanJets
        else:
            j = j * self.negativePhiOthJets
        return j

    def flipEta(self, j, canJets=True): # j[event, mu, jet], mu=1 is eta
        if canJets:
            j = j * self.negativeEtaCanJets
        else:
            j = j * self.negativeEtaOthJets
        return j


    def makeSymmetries(self, j, o, mask, d, q):
        n = j.shape[0]
        # Copy inputs nRF times to compute each of the symmetry transformations 
        j = j.repeat(self.nRF, 1, 1)
        d = d.repeat(self.nRF, 1, 1)
        q = q.repeat(self.nRF, 1, 1)
        if self.useOthJets:
            o    = o   .repeat(self.nRF, 1, 1)
            mask = mask.repeat(self.nRF, 1)

        # Randomly rotate the event in phi during training
        if self.training:
            randomR = 2*torch.rand(n,1,1, device='cuda')

        # apply each of the symmetry transformations over which we will average after learning eta/phi dependent features
        j = j.view(4, n, self.nj, -1)
        jR, jRP, jRE, jRPE = j[0], j[1], j[2], j[3]
        if self.training: 
            jR = self.rotate(jR, randomR)
        if self.useOthJets:
            o = o.view(4, n, 5, -1)
            oR, oRP, oRE, oRPE = o[0], o[1], o[2], o[3]
            if self.training: 
                oR = self.rotate(oR, randomR)

        #flip phi
        if self.training: 
            jRP = self.rotate( jRP, randomR)
        jRP = self.flipPhi(jRP)
        if self.useOthJets:
            if self.training: 
                oRP = self.rotate( oRP, randomR)
            oRP = self.flipPhi(oRP, canJets=False)

        #flip eta
        if self.training:
            jRE = self.rotate( jRE, randomR)
        jRE = self.flipEta(jRE)
        if self.useOthJets:
            if self.training: 
                oRE = self.rotate( oRE, randomR)
            oRE = self.flipEta(oRE, canJets=False)

        #flip phi and eta
        if self.training: 
            jRPE = self.rotate( jRPE, randomR)
        jRPE = self.flipPhi(jRPE)
        jRPE = self.flipEta(jRPE)
        if self.useOthJets:
            if self.training: 
                oRPE = self.rotate( oRPE, randomR)
            oRPE = self.flipPhi(oRPE, canJets=False)
            oRPE = self.flipEta(oRPE, canJets=False)


        j = torch.cat( (jR, jRP, jRE, jRPE), dim=0)
        if self.useOthJets:
            o = torch.cat( (oR, oRP, oRE, oRPE), dim=0)

        return j, o, mask, d, q


    def EtaPhiInvariantPart(self, x):
        x = NonLU(self.conv1(x), self.training)
        x = self.drop1(x)
        x = NonLU(self.conv2(x), self.training)
        x = self.drop2(x)
        x = NonLU(self.conv3(x), self.training)
        x = self.drop3(x)
        return x


    def forward(self, j, o, d, q):
        j, o, mask, d, q = self.inputGBN(j, o, d, q) # format inputs to array of objects and apply scalers and GBNs
        n = j.shape[0]

        # Remove duplicates of jets and quadjet ancillary info 
        j = j[:,:,0:4]
        d = d.contiguous().view(n, 12, 1)
        q = torch.cat( (q[:,0,:].view(n,3,1), q[:,1:,0].view(n,self.nAq-1,1)) , dim=1)

        # Copy inputs nRF times and apply a different symmetry transformation to each copy
        j, o, mask, d, q = self.makeSymmetries(j, o, mask, d, q)

        # put all features into a single pixel and process as fully connected feed forward network
        j = j.contiguous().view(n*self.nRF, 16, 1)
        x = torch.cat( (j, d, q), dim=1)

        # compute the quadjet pixels and average them over the symmetry transformations
        x = self.EtaPhiInvariantPart(x)
        x = x.view(self.nRF, n, 128, 1)
        x = x.mean(dim=0)

        #project the final event-level pixel into the class score space
        c_score = self.out(x)
        c_score = c_score.view(n, self.nClasses)

        return c_score, None
