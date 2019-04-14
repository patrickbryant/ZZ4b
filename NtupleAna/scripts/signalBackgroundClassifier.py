import time, os, sys
import numpy as np
import pandas as pd
np.random.seed(0)#always pick the same training sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from sklearn.metrics import roc_curve, auc # pip/conda install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlibHelpers as pltHelper
class Lin_View(nn.Module):
    def __init__(self):
        super(Lin_View, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0], -1)

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-b', '--background', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD1.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-s', '--signal',     default='/uscms/home/bryantp/nobackup/ZZ4b/bothZH4b2018/picoAOD0.h5', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-l', '--lrInit', default=1e-3, type=float, help='Initial learning rate.')
parser.add_argument('-p', '--pDropout', default=0.2, type=float, help='p(drop) for dropout.')
parser.add_argument('-c', '--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")

args = parser.parse_args()


class basicCNN(nn.Module):
    def __init__(self, dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout):
        super(basicCNN, self).__init__()
        self.name = 'basicCNN_%d_%d_%d_%d_pdrop%.2f'%(dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  ##kernel=3
        self.conv1 = nn.Sequential(*[nn.Conv1d(               4,        dijetFeatures, 2, stride=2), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv1d(   dijetFeatures,      quadjetFeatures, 2, stride=2), nn.ReLU()])
        self.conv3 = nn.Sequential(*[nn.Conv1d( quadjetFeatures, combinatoricFeatures, 3, stride=1), nn.ReLU()])

        self.line1 = nn.Sequential(*[nn.Linear(combinatoricFeatures, nodes), nn.ReLU()])
        self.line2 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
        self.line3 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
        self.line4 =                 nn.Linear(nodes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        
        x = self.line1(x)
        x = self.line2(x)
        x = self.line3(x)
        x = self.line4(x)
        return x


# class dijetCNN(nn.Module):
#     def __init__(self, dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout):
#         super(dijetCNN, self).__init__()
#         self.name = 'dijetCNN_%d_%d_%d_%d_pdrop%.2f'%(dijetFeatures, quadjetFeatures, combinatoricFeatures, nodes, pDropout)
#         self.nd = dijetFeatures
#         self.nq = quadjetFeatures

#         #self.toDijetFeatureSpace = nn.Sequential(*[nn.Conv1d(4, self.nd, 1), nn.ReLU()])
#         self.toDijetFeatureSpace = nn.Conv1d(4, self.nd, 1)
#         # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
#         # |1,2|3,4|1,3|2,4|1,4|2,3|  
#         self.dijetBuilder = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 2, stride=2), nn.ReLU()])

#         # |1|1,2|2|3|3,4|4|1|1,3|3|2|2,4|4|1|1,4|4|2|2,3|3|  ##stride=3 kernel=3 reinforce dijet features
#         #   |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
#         self.dijetReinforce1 = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 3, stride=3), nn.ReLU()])
#         self.dijetReinforce2 = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 3, stride=3), nn.ReLU()])

#         self.toQuadjetFeatureSpace = nn.Conv1d(self.nd, self.nq, 1)
#         # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
#         # |1,2,3,4|1,2,3,4|1,2,3,4|  
#         self.quadjetBuilder = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 2, stride=2), nn.ReLU()])

#         # |1,2|1,2,3,4|3,4|1,2|1,2,3,4|3,4|1,2|1,2,3,4|3,4|  
#         #     |1,2,3,4|       |1,2,3,4|       |1,2,3,4|  
#         self.quadjetReinforce1 = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 3, stride=3), nn.ReLU()])
#         self.quadjetReinforce2 = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 3, stride=3), nn.ReLU()])

#         self.viewSelector   = nn.Sequential(*[nn.Conv1d(self.nq, combinatoricFeatures, 3, stride=1), nn.ReLU()])

#         self.line1 = nn.Sequential(*[nn.Linear(combinatoricFeatures, nodes), nn.ReLU(), nn.Dropout(p=pDropout)])
#         self.line2 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
#         self.line3 = nn.Sequential(*[nn.Linear(nodes, nodes),                nn.ReLU(), nn.Dropout(p=pDropout)])
#         self.line4 =                 nn.Linear(nodes, 1)

#     def forward(self, x):
#         n = x.shape[0]
#         x = self.toDijetFeatureSpace(x)
#         d = self.dijetBuilder(x)
#         d = torch.cat( (x[:,:, 0].view(n,self.nd,1), d[:,:,0].view(n,self.nd,1), x[:,:, 1].view(n,self.nd,1),
#                         x[:,:, 2].view(n,self.nd,1), d[:,:,1].view(n,self.nd,1), x[:,:, 3].view(n,self.nd,1),
#                         x[:,:, 4].view(n,self.nd,1), d[:,:,2].view(n,self.nd,1), x[:,:, 5].view(n,self.nd,1),
#                         x[:,:, 6].view(n,self.nd,1), d[:,:,3].view(n,self.nd,1), x[:,:, 7].view(n,self.nd,1),
#                         x[:,:, 8].view(n,self.nd,1), d[:,:,4].view(n,self.nd,1), x[:,:, 9].view(n,self.nd,1),
#                         x[:,:,10].view(n,self.nd,1), d[:,:,5].view(n,self.nd,1), x[:,:,11].view(n,self.nd,1)), 2)
#         d = self.dijetReinforce1(d)
#         d = torch.cat( (x[:,:, 0].view(n,self.nd,1), d[:,:,0].view(n,self.nd,1), x[:,:, 1].view(n,self.nd,1),
#                         x[:,:, 2].view(n,self.nd,1), d[:,:,1].view(n,self.nd,1), x[:,:, 3].view(n,self.nd,1),
#                         x[:,:, 4].view(n,self.nd,1), d[:,:,2].view(n,self.nd,1), x[:,:, 5].view(n,self.nd,1),
#                         x[:,:, 6].view(n,self.nd,1), d[:,:,3].view(n,self.nd,1), x[:,:, 7].view(n,self.nd,1),
#                         x[:,:, 8].view(n,self.nd,1), d[:,:,4].view(n,self.nd,1), x[:,:, 9].view(n,self.nd,1),
#                         x[:,:,10].view(n,self.nd,1), d[:,:,5].view(n,self.nd,1), x[:,:,11].view(n,self.nd,1)), 2)
#         d = self.dijetReinforce2(d)
#         x = self.toQuadjetFeatureSpace(d)
#         q = self.quadjetBuilder(x)
#         q = torch.cat( (x[:,:, 0].view(n,self.nq,1), q[:,:,0].view(n,self.nq,1), x[:,:, 1].view(n,self.nq,1),
#                         x[:,:, 2].view(n,self.nq,1), q[:,:,1].view(n,self.nq,1), x[:,:, 3].view(n,self.nq,1),
#                         x[:,:, 4].view(n,self.nq,1), q[:,:,2].view(n,self.nq,1), x[:,:, 5].view(n,self.nq,1)), 2)
#         q = self.quadjetReinforce1(q)
#         q = torch.cat( (x[:,:, 0].view(n,self.nq,1), q[:,:,0].view(n,self.nq,1), x[:,:, 1].view(n,self.nq,1),
#                         x[:,:, 2].view(n,self.nq,1), q[:,:,1].view(n,self.nq,1), x[:,:, 3].view(n,self.nq,1),
#                         x[:,:, 4].view(n,self.nq,1), q[:,:,2].view(n,self.nq,1), x[:,:, 5].view(n,self.nq,1)), 2)
#         q = self.quadjetReinforce2(q)
#         x = q
#         x = self.viewSelector(x)
#         x = x.view(x.shape[0], -1)
        
#         x = self.line1(x)
#         x = self.line2(x)
#         x = self.line3(x)
#         x = self.line4(x)
#         return x

class ResNet(nn.Module):
    def __init__(self, dijetFeatures, quadjetFeatures, combinatoricFeatures):
        super(ResNet, self).__init__()
        self.name = 'ResNet_%d_%d_%d'%(dijetFeatures, quadjetFeatures, combinatoricFeatures)
        self.nd = dijetFeatures
        self.nq = quadjetFeatures

        #self.toDijetFeatureSpace = nn.Sequential(*[nn.Conv1d(4, self.nd, 1), nn.ReLU()])
        self.toDijetFeatureSpace = nn.Conv1d(4, self.nd, 1)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 2, stride=2), nn.ReLU()])

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetReinforce1 = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 3, stride=3), nn.ReLU()])
        self.dijetReinforce2 = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 3, stride=3), nn.ReLU()])
        self.dijetReinforce3 = nn.Conv1d(self.nd, self.nd, 3, stride=3)
        self.dijetReLU = nn.ReLU()

        self.toQuadjetFeatureSpace = nn.Conv1d(self.nd, self.nq, 1)
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,3,2,4|1,4,2,3|  
        self.quadjetBuilder = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 2, stride=2), nn.ReLU()])

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetReinforce1 = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 3, stride=3), nn.ReLU()])
        self.quadjetReinforce2 = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 3, stride=3), nn.ReLU()])
        self.quadjetReinforce3 = nn.Conv1d(self.nq, self.nq, 3, stride=3)
        self.quadjetReLU = nn.ReLU()

        self.viewSelector   = nn.Sequential(*[nn.Conv1d(self.nq, combinatoricFeatures, 3, stride=1), nn.ReLU()])
        self.out = nn.Linear(combinatoricFeatures, 1)

    def forward(self, x):
        n = x.shape[0]
        x = self.toDijetFeatureSpace(x)
        d = self.dijetBuilder(x)
        d1 = d
        d = torch.cat( (x[:,:, 0: 2], d[:,:,0].view(n,self.nd,1),
                        x[:,:, 2: 4], d[:,:,1].view(n,self.nd,1),
                        x[:,:, 4: 6], d[:,:,2].view(n,self.nd,1),
                        x[:,:, 6: 8], d[:,:,3].view(n,self.nd,1),
                        x[:,:, 8:10], d[:,:,4].view(n,self.nd,1),
                        x[:,:,10:  ], d[:,:,5].view(n,self.nd,1)), 2)
        d = self.dijetReinforce1(d)
        d = torch.cat( (x[:,:, 0: 2], d[:,:,0].view(n,self.nd,1),
                        x[:,:, 2: 4], d[:,:,1].view(n,self.nd,1),
                        x[:,:, 4: 6], d[:,:,2].view(n,self.nd,1),
                        x[:,:, 6: 8], d[:,:,3].view(n,self.nd,1),
                        x[:,:, 8:10], d[:,:,4].view(n,self.nd,1),
                        x[:,:,10:  ], d[:,:,5].view(n,self.nd,1)), 2)
        d = self.dijetReinforce2(d)
        d = torch.cat( (x[:,:, 0: 2], d[:,:,0].view(n,self.nd,1),
                        x[:,:, 2: 4], d[:,:,1].view(n,self.nd,1),
                        x[:,:, 4: 6], d[:,:,2].view(n,self.nd,1),
                        x[:,:, 6: 8], d[:,:,3].view(n,self.nd,1),
                        x[:,:, 8:10], d[:,:,4].view(n,self.nd,1),
                        x[:,:,10:  ], d[:,:,5].view(n,self.nd,1)), 2)
        d = self.dijetReinforce3(d)
        x = self.dijetReLU( d + d1 )

        x = self.toQuadjetFeatureSpace(d)
        q = self.quadjetBuilder(x)
        q1 = q
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4: ], q[:,:,2].view(n,self.nq,1)), 2)
        q = self.quadjetReinforce1(q)
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4: ], q[:,:,2].view(n,self.nq,1)), 2)
        q = self.quadjetReinforce2(q)
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4: ], q[:,:,2].view(n,self.nq,1)), 2)
        q = self.quadjetReinforce3(q)
        x = self.quadjetReLU( q + q1 )

        x = self.viewSelector(x)
        x = x.view(x.shape[0], -1)
        
        x = self.out(x)
        return x


class dijetResNetBlock(nn.Module):
    def __init__(self, dijetFeatures):
        super(dijetResNetBlock, self).__init__()
        self.nd = dijetFeatures
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetReinforce1 = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 3, stride=3), nn.ReLU()])
        self.dijetReinforce2 = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 3, stride=3), nn.ReLU()])
        self.dijetReinforce3 = nn.Conv1d(self.nd, self.nd, 3, stride=3)
        self.dijetReLU = nn.ReLU()

    def forward(self, x, d):
        n = x.shape[0]
        d1 = d
        d = torch.cat( (x[:,:, 0: 2], d[:,:,0].view(n,self.nd,1),
                        x[:,:, 2: 4], d[:,:,1].view(n,self.nd,1),
                        x[:,:, 4: 6], d[:,:,2].view(n,self.nd,1),
                        x[:,:, 6: 8], d[:,:,3].view(n,self.nd,1),
                        x[:,:, 8:10], d[:,:,4].view(n,self.nd,1),
                        x[:,:,10:  ], d[:,:,5].view(n,self.nd,1)), 2)
        d = self.dijetReinforce1(d)
        d = torch.cat( (x[:,:, 0: 2], d[:,:,0].view(n,self.nd,1),
                        x[:,:, 2: 4], d[:,:,1].view(n,self.nd,1),
                        x[:,:, 4: 6], d[:,:,2].view(n,self.nd,1),
                        x[:,:, 6: 8], d[:,:,3].view(n,self.nd,1),
                        x[:,:, 8:10], d[:,:,4].view(n,self.nd,1),
                        x[:,:,10:  ], d[:,:,5].view(n,self.nd,1)), 2)
        d = self.dijetReinforce2(d)
        d = torch.cat( (x[:,:, 0: 2], d[:,:,0].view(n,self.nd,1),
                        x[:,:, 2: 4], d[:,:,1].view(n,self.nd,1),
                        x[:,:, 4: 6], d[:,:,2].view(n,self.nd,1),
                        x[:,:, 6: 8], d[:,:,3].view(n,self.nd,1),
                        x[:,:, 8:10], d[:,:,4].view(n,self.nd,1),
                        x[:,:,10:  ], d[:,:,5].view(n,self.nd,1)), 2)
        d = self.dijetReinforce3(d)
        d = self.dijetReLU( d + d1 )
        return d


class quadjetResNetBlock(nn.Module):
    def __init__(self, quadjetFeatures):
        super(quadjetResNetBlock, self).__init__()
        self.nq = quadjetFeatures
        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetReinforce1 = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 3, stride=3), nn.ReLU()])
        self.quadjetReinforce2 = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 3, stride=3), nn.ReLU()])
        self.quadjetReinforce3 = nn.Conv1d(self.nq, self.nq, 3, stride=3)
        self.quadjetReLU = nn.ReLU()

    def forward(self, x, q):
        n = x.shape[0]
        q1 = q
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4: ], q[:,:,2].view(n,self.nq,1)), 2)
        q = self.quadjetReinforce1(q)
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4: ], q[:,:,2].view(n,self.nq,1)), 2)
        q = self.quadjetReinforce2(q)
        q = torch.cat( (x[:,:, 0:2], q[:,:,0].view(n,self.nq,1),
                        x[:,:, 2:4], q[:,:,1].view(n,self.nq,1),
                        x[:,:, 4: ], q[:,:,2].view(n,self.nq,1)), 2)
        q = self.quadjetReinforce3(q)
        q = self.quadjetReLU( q + q1 )
        return q


class deepResNet(nn.Module):
    def __init__(self, dijetFeatures, quadjetFeatures, combinatoricFeatures):
        super(deepResNet, self).__init__()
        self.name = 'deepResNet_%d_%d_%d'%(dijetFeatures, quadjetFeatures, combinatoricFeatures)
        self.nd = dijetFeatures
        self.nq = quadjetFeatures

        #self.toDijetFeatureSpace = nn.Sequential(*[nn.Conv1d(4, self.nd, 1), nn.ReLU()])
        self.toDijetFeatureSpace = nn.Conv1d(4, self.nd, 1)
        # |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        # |1,2|3,4|1,3|2,4|1,4|2,3|  
        self.dijetBuilder = nn.Sequential(*[nn.Conv1d(self.nd, self.nd, 2, stride=2), nn.ReLU()])

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|    
        self.dijetResNetBlock1 = dijetResNetBlock(self.nd)
        self.dijetResNetBlock2 = dijetResNetBlock(self.nd)
        self.dijetResNetBlock3 = dijetResNetBlock(self.nd)

        self.toQuadjetFeatureSpace = nn.Conv1d(self.nd, self.nq, 1)
        # |1,2|3,4|1,3|2,4|1,4|2,3|  ##stride=2 kernel=2 gives all possible dijet->quadjet constructions
        # |1,2,3,4|1,2,3,4|1,2,3,4|  
        self.quadjetBuilder = nn.Sequential(*[nn.Conv1d(self.nq, self.nq, 2, stride=2), nn.ReLU()])

        # |1,2|3,4|1,2,3,4|1,3|2,4|1,3,2,4|1,4,2,3|1,4,2,3|
        #         |1,2,3,4|       |1,3,2,4|       |1,4,2,3|  
        self.quadjetResNetBlock1 = quadjetResNetBlock(self.nq)
        self.quadjetResNetBlock2 = quadjetResNetBlock(self.nq)
        self.quadjetResNetBlock3 = quadjetResNetBlock(self.nq)

        self.viewSelector   = nn.Sequential(*[nn.Conv1d(self.nq, combinatoricFeatures, 3, stride=1), nn.ReLU()])
        self.out = nn.Linear(combinatoricFeatures, 1)

    def forward(self, x):
        n = x.shape[0]

        x = self.toDijetFeatureSpace(x)
        d = self.dijetBuilder(x)
        d = self.dijetResNetBlock1(x,d)
        d = self.dijetResNetBlock2(x,d)
        x = self.dijetResNetBlock3(x,d)

        x = self.toQuadjetFeatureSpace(d)
        q = self.quadjetBuilder(x)
        q = self.quadjetResNetBlock1(x,q)
        q = self.quadjetResNetBlock2(x,q)
        x = self.quadjetResNetBlock3(x,q)

        x = self.viewSelector(x)
        x = x.view(x.shape[0], -1)
        
        x = self.out(x)
        return x


class modelParameters:
    def __init__(self, fileName=''):
        # self.xVariables=['canJet0_pt', 'canJet1_pt', 'canJet2_pt', 'canJet3_pt',
        #                  'canJet0_eta', 'canJet1_eta', 'canJet2_eta', 'canJet3_eta',
        #                  'canJet0_phi', 'canJet1_phi', 'canJet2_phi', 'canJet3_phi',
        #                  'canJet0_e', 'canJet1_e', 'canJet2_e', 'canJet3_e',
        #                  ]
        #             |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        self.layer1Pix = "012302130312"
        self.layer1Col = ['_pt', '_eta', '_phi', '_e']
        self.xVariables=[['canJet'+i+'_pt', 'canJet'+i+'_eta', 'canJet'+i+'_phi', 'canJet'+i+'_e'] for i in self.layer1Pix] #index[pixel][color]
        #self.xVariables=[['canJet'+jet+mu for jet in self.layer1Pix] for mu in self.layer1Col] #index[color][pixel]
        #self.xVariables[color][pixel]
        if fileName:
            self.dijetFeatures        = int(fileName.split('_')[2])
            self.quadjetFeatures      = int(fileName.split('_')[3])
            self.combinatoricFeatures = int(fileName.split('_')[4])
            self.nodes                = None#int(fileName.split('_')[5])
            self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_lr')]) if '_pdrop' in fileName else None
            self.lrInit        = float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =   int(fileName[fileName.find('e_epoch')+7 : fileName.find('_auc')])
            self.roc_auc_best  = float(fileName[fileName.find(   '_auc')+4 : fileName.find('.pkl')])
            self.scalers = torch.load(fileName)['scalers']

        else:
            self.dijetFeatures = 12
            self.quadjetFeatures = 20
            self.combinatoricFeatures = 40
            self.nodes = 128
            self.pDropout      = args.pDropout
            self.lrInit        = args.lrInit
            self.startingEpoch = 0
            self.roc_auc_best  = 0.8 #SvsB_onlyCNN_12_24_48_128_pdrop0.20_lr0.001_epochs50_stdscale_epoch33_auc0.8286 without batchnorm # 0.7737 dijetCNN_12_24_48_128_pdrop0.20_lr0.001_epochs50_stdscale_epoch30_auc0.7737
            self.scalers = {}

        #self.name = 'SvsB_FC%dx%d_pdrop%.2f_lr%s_epochs%d_stdscale'%(self.nodes, self.layers, self.pDropout, str(self.lrInit), args.epochs+self.startingEpoch)

        #self.net = basicCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        #self.net = dijetCNN(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout).to(device)
        #self.name = 'SvsB_dijetCNN_%d_%d_%d_%d_pdrop%.2f_lr%s_epochs%d_stdscale'%(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.nodes, self.pDropout, str(self.lrInit), args.epochs+self.startingEpoch)
        #self.net = ResNet(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures).to(device)
        self.net = deepResNet(self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures).to(device)
        self.name = 'SvsB_'+self.net.name+'_lr%s_epochs%d_stdscale'%(str(self.lrInit), args.epochs+self.startingEpoch)

        self.dump()
        print(self.net)


        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state
    
    def dump(self):
        print(self.name)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('roc_auc_best:',self.roc_auc_best)


# Run on gpu if available
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
print('torch.cuda.is_available()',torch.cuda.is_available())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device",device,torch.cuda.device_count(),torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU:",device)

model = modelParameters(args.model)

n_queue = 10
train_batch_size = 32 #36
eval_batch_size = 2048
foundNewBest = False
print_step = 100
train_fraction = 0.5

if args.model and args.update:
    for fileName in [args.background, args.signal]:
        print("Add classifier output to",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')

        n = df.shape[0]
        print("n",n)

        X = [np.float32(df[jet]) for jet in model.xVariables]
        X = torch.FloatTensor([np.float32([[X[jet][event][mu] for jet in range(len(model.xVariables))] for mu in range(4)]) for event in range(n)])
        y = np.zeros(n, dtype=np.uint8).reshape(-1,1)
        print('X.shape', X.shape)

        for jet in range(X.shape[2]):
            X[:,:,jet] = torch.FloatTensor(model.scalers[0].transform(X[:,:,jet]))

        # Set up data loaders
        dset   = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        loader = DataLoader(dataset=dset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        print('Batches:', len(loader))

        model.net.eval()
        y_pred = []
        for i, (X, y) in enumerate(loader):
            X = X.to(device)
            logits = model.net(X)#.view(-1,1)
            binary_pred = logits.ge(0.).byte()
            prob_pred = torch.sigmoid(logits)
            y_pred.append(prob_pred.tolist())
            if (i+1) % print_step == 0:
                sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(loader)))
                sys.stdout.flush()

        y_pred = np.float32(np.concatenate(y_pred).reshape(1,df.shape[0])[0])
        print(y_pred)
        df['ZHvsBackgroundClassifier'] = pd.Series(y_pred, index=df.index)
        print("df.dtypes")
        print(df.dtypes)
        print("df.shape", df.shape)
        df.to_hdf(fileName, key='df', format='table', mode='w')

        del dset
        del loader

    exit()

# Read .h5 file
dfB = pd.read_hdf(args.background, key='df')
dfS = pd.read_hdf(args.signal,     key='df')

#select events in desired region for training/validation/test
dfB = dfB.loc[ (dfB['fourTag']==False) & ((dfB['ZHSB']==True)|(dfB['ZHCR']==True)|(dfB['ZHSR']==True)) & (dfB['passDEtaBB']==True) ]
dfS = dfS.loc[ (dfS['fourTag']==True ) & ((dfS['ZHSB']==True)|(dfS['ZHCR']==True)|(dfS['ZHSR']==True)) & (dfS['passDEtaBB']==True) ]

nS      = dfS.shape[0]
nB      = dfB.shape[0]
print("nS",nS)
print("nB",nB)

# compute relative weighting for S and B
sum_wS = np.sum(np.float32(dfS['weight']))
sum_wB = np.sum(np.float32(dfB['weight']))
print("sum_wS",sum_wS)
print("sum_wB",sum_wB)

sum_wStoS = np.sum(np.float32(dfS.loc[ dfS['ZHSR']==True ]['weight']))
sum_wBtoB = np.sum(np.float32(dfB.loc[ dfB['ZHSR']==False]['weight']))
print("sum_wStoS",sum_wStoS)
print("sum_wBtoB",sum_wBtoB)
rate_StoS = sum_wStoS/sum_wS
rate_BtoB = sum_wBtoB/sum_wB
print("Cut Based WP:",rate_StoS,"Signal Eff.", rate_BtoB,"1-Background Eff.")

#
# Split into training and validation sets
#
nTrainS = int(nS*train_fraction)
nTrainB = int(nB*train_fraction)
nValS   = nS-nTrainS
nValB   = nB-nTrainB

#random ordering to mix up which data is used for training or validation
idxS    = np.random.permutation(nS)
idxB    = np.random.permutation(nB)

#define dataframes for trainging and validation
dfS['weight'] = dfS['weight']*sum_wB/sum_wS
dfS_train = dfS.iloc[idxS[:nTrainS]]
dfS_val   = dfS.iloc[idxS[nTrainS:]]
dfB_train = dfB.iloc[idxB[:nTrainB]]
dfB_val   = dfB.iloc[idxB[nTrainB:]]

df_train = pd.concat([dfB_train, dfS_train], sort=False)
nTrain   = df_train.shape[0]
df_val   = pd.concat([dfB_val,   dfS_val  ], sort=False)
nVal     = df_val  .shape[0]

#Convert to list np array
X_train=[np.float32(df_train[jet]) for jet in model.xVariables]
X_val  =[np.float32(df_val  [jet]) for jet in model.xVariables]
#make 3D tensor with correct axes [event][color][pixel] = [event][mu (4-vector component)][jet]
X_train=torch.FloatTensor([np.float32([[X_train[jet][event][mu] for jet in range(len(model.xVariables))] for mu in range(4)]) for event in range(nTrain)])
X_val  =torch.FloatTensor([np.float32([[X_val  [jet][event][mu] for jet in range(len(model.xVariables))] for mu in range(4)]) for event in range(nVal  )])

y_train=torch.FloatTensor(  np.concatenate( (np.zeros(nTrainB, dtype=np.uint8).reshape(-1,1), 
                                             np.ones( nTrainS, dtype=np.uint8).reshape(-1,1)) )  )
y_val  =torch.FloatTensor(  np.concatenate( (np.zeros(nValB,   dtype=np.uint8).reshape(-1,1), 
                                             np.ones( nValS,   dtype=np.uint8).reshape(-1,1)) )  )

w_train=torch.FloatTensor( np.float32(df_train['weight']).reshape(-1,1) )
w_val  =torch.FloatTensor( np.float32(df_val  ['weight']).reshape(-1,1) )

print('X_train.shape, y_train.shape, w_train.shape:', X_train.shape, y_train.shape, w_train.shape)
print('X_val  .shape, y_val  .shape, w_val  .shape:', X_val  .shape, y_val  .shape, w_val  .shape)

# Standardize inputs

if not args.model:
    # model.scalers[0] = StandardScaler(with_mean=False)
    # model.scalers[0].fit(X_train[:,:,1].index_select(1,torch.LongTensor([0,3]))) ##only fit the scalar to one jet spectra. Don't want each pt ordered jet scale to be different

    model.scalers[0] = StandardScaler(with_mean=False)
    model.scalers[0].fit(X_train[:,:,1])
    model.scalers[0].scale_[1] = 2.5   # eta max
    model.scalers[0].scale_[2] = np.pi # pi
    model.scalers[0].scale_[3] = model.scalers[0].scale_[0]
    print("scale_",model.scalers[0].scale_)

for jet in range(X_train.shape[2]):
    X_train[:,:,jet] = torch.FloatTensor(model.scalers[0].transform(X_train[:,:,jet]))
    X_val  [:,:,jet] = torch.FloatTensor(model.scalers[0].transform(X_val  [:,:,jet]))


# Set up data loaders
dset_train   = TensorDataset(X_train, y_train, w_train)
dset_val     = TensorDataset(X_val,   y_val,   w_val)
train_loader = DataLoader(dataset=dset_train, batch_size=train_batch_size, shuffle=True,  num_workers=n_queue, pin_memory=True)
eval_train_loader = DataLoader(dataset=dset_train, batch_size=eval_batch_size, shuffle=False,  num_workers=n_queue, pin_memory=True)
val_loader   = DataLoader(dataset=dset_val,   batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
print('len(train_loader), len(val_loader):', len(train_loader), len(val_loader))
print('N trainable params:',sum(p.numel() for p in model.net.parameters() if p.requires_grad))

optimizer = optim.Adam(model.net.parameters(), lr=model.lrInit)

#Function to perform training epoch
def train(s):
    #print('-------------------------------------------------------------')
    model.net.train()
    now = time.time()
    y_pred, y_true, w_ordered = [], [], []
    for i, (X, y, w) in enumerate(train_loader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model.net(X)#.view(-1,1)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w) # binary classification
        #loss = F.binary_cross_entropy_with_logits(logits, y) # binary classification
        #loss = F.mse_loss(logits, y) # regression
        #break
        loss.backward()
        optimizer.step()
        #break
        prob_pred = torch.sigmoid(logits)
        y_pred.append(prob_pred.tolist())
        y_true.append(y.tolist())
        w_ordered.append(w.tolist())
        if (i+1) % print_step == 0:
            binary_pred = logits.ge(0.).byte()
            accuracy = binary_pred.eq(y.byte()).float().mean().item()
            sys.stdout.write('\rTraining %3.0f%%     '%(float(i+1)*100/len(train_loader)))
            sys.stdout.flush()

    now = time.time() - now

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    w_ordered = np.concatenate(w_ordered)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    bar=int((roc_auc-0.5)*200) if roc_auc > 0.5 else 0
    print('\r'+s+' ROC AUC: %0.4f   (Training Set)'%(roc_auc),("-"*bar)+"|")
    return y_pred, y_true, w_ordered, fpr, tpr, roc_auc


def evaluate(loader):
    now = time.time()
    model.net.eval()
    loss, accuracy = [], []
    y_pred, y_true, w_ordered = [], [], []
    for i, (X, y, w) in enumerate(loader):
        X, y, w = X.to(device), y.to(device), w.to(device)
        logits = model.net(X)#.view(-1,1)
        binary_pred = logits.ge(0.).byte()
        prob_pred = torch.sigmoid(logits)
        batch_loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction='none') # binary classification
        #batch_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none') # binary classification
        accuracy.append(binary_pred.eq(y.byte()).float().tolist())
        loss.append(batch_loss.tolist())
        y_pred.append(prob_pred.tolist())
        y_true.append(y.tolist())
        w_ordered.append(w.tolist())
        if (i+1) % print_step == 0:
            sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(loader)))
            sys.stdout.flush()

    now = time.time() - now

    accuracy = np.concatenate(accuracy)
    loss = np.concatenate(loss)
    y_pred = np.transpose(np.concatenate(y_pred))[0]
    y_true = np.transpose(np.concatenate(y_true))[0]
    w_ordered = np.transpose(np.concatenate(w_ordered))[0]

    return y_pred, y_true, w_ordered, accuracy, loss


#function to check performance on validation set
def validate(s):
    y_pred, y_true, w_ordered, accuracy, loss = evaluate(val_loader)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    bar=int((roc_auc-0.5)*200) if roc_auc > 0.5 else 0
    print('\r'+s+' ROC AUC: %0.4f (Validation Set)'%(roc_auc),("#"*bar)+"|", end = " ")
    return y_pred, y_true, w_ordered, fpr, tpr, roc_auc


#Simple ROC Curve plot function
def plotROC(fpr, tpr, name): #fpr = false positive rate, tpr = true positive rate
    roc_auc = auc(fpr, tpr)
    f = plt.figure()
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    plt.plot([0,1], [1,0], color='0.8', linestyle='--')

    plt.xlabel('Rate( Signal to Signal )')
    plt.ylabel('Rate( Background to Background )')

    plt.plot(tpr, 1-fpr)
    plt.text(0.72, 0.98, "ROC AUC = %0.4f"%(roc_auc))
    plt.scatter(rate_StoS, rate_BtoB, marker='o', c='r')
    plt.text(rate_StoS+0.03, rate_BtoB+0.02, "Cut Based WP")
    plt.text(rate_StoS+0.03, rate_BtoB-0.03, "(%0.2f, %0.2f)"%(rate_StoS, rate_BtoB))
    f.savefig(name)
    plt.close(f)


def plotNet(y_pred, y_true, w, name):
    fig = pltHelper.plot([y_pred[y_true==1], y_pred[y_true==0]], 
                         [b/20.0 for b in range(21)],
                         "NN Output", "Events / Bin", 
                         weights=[w[y_true==1],w[y_true==0]],
                         samples=['Signal','Background'],
                         ratio=True,
                         ratioRange=[0,5])
    fig.savefig(name)
    plt.close(fig)
    

#model initial state
y_pred_val, y_true_val, w_ordered_val, fpr, tpr, roc_auc = validate(">> Epoch %3d/%d <<<<<<<<"%(model.startingEpoch, args.epochs+model.startingEpoch))
print()
if args.model:
    print(y_pred_val)
    plotROC(fpr, tpr, args.model.replace('.pkl', '_ROC_val.pdf'))
    plotNet(y_pred_val, y_true_val, w_ordered_val, args.model.replace('.pkl','_NetOutput_val.pdf'))

# Training loop
for epoch in range(model.startingEpoch+1, model.startingEpoch+args.epochs+1):
    epochString = '>> Epoch %3d/%d <<<<<<<<'%(epoch, args.epochs+model.startingEpoch)

    # Run training
    y_pred_train, y_true_train, w_ordered_train, fpr_train, tpr_train, roc_auc_train =    train(epochString)

    # Run Validation
    y_pred_val,   y_true_val,   w_ordered_val,   fpr_val,   tpr_val,   roc_auc_val   = validate(epochString)

    roc_auc = roc_auc_val
    if roc_auc > model.roc_auc_best:
        foundNewBest = True
        model.roc_auc_best = roc_auc
    
        filename = 'ZZ4b/NtupleAna/pytorchModels/%s_epoch%d_auc%.4f.pkl'%(model.name, epoch, model.roc_auc_best)
        print("*", filename)
        y_pred_train, y_true_train, w_ordered_train, _, _ = evaluate(eval_train_loader)
        fpr_train, tpr_train, _ = roc_curve(y_true_train, y_pred_train)
        plotROC(fpr_train, tpr_train, filename.replace('.pkl', '_ROC_train.pdf'))
        plotROC(fpr_val,   tpr_val,   filename.replace('.pkl', '_ROC_val.pdf'))
        plotNet(y_pred_train, y_true_train, w_ordered_train, filename.replace('.pkl','_NetOutput_train.pdf'))
        plotNet(y_pred_val,   y_true_val,   w_ordered_val,   filename.replace('.pkl','_NetOutput_val.pdf'))
        
        model_dict = {'model': model.net.state_dict(), 'optim': optimizer.state_dict(), 'scalers': model.scalers}
        torch.save(model_dict, filename)
    else:
        print()

print()
print(">> DONE <<<<<<<<")
if foundNewBest: print("Best ROC AUC =", model.roc_auc_best)
