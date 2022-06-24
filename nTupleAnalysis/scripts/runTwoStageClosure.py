from __future__ import print_function
import ROOT
ROOT.gROOT.SetBatch(True)
#ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2")
import sys
import operator
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import collections
import PlotTools
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
from array import array
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
#import mpl_toolkits
#from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
CMURED = '#d34031'
#https://xkcd.com/color/rgb/
COLORS=['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:cherry', 'xkcd:bright red',
        'xkcd:pine', 'xkcd:magenta', 'xkcd:cerulean', 'xkcd:eggplant', 'xkcd:coral', 'xkcd:blue purple',
        'xkcd:tea', 'xkcd:burple', 'xkcd:deep aqua', 'xkcd:orange pink', 'xkcd:terracota']

year = 'RunII'
lumi = 132.6
classifier = 'SvB'
rebin = {'zz': 4, 'zh': 5, 'hh': 10}
closure_fit_x_min = 0#0.01
#rebin = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
maxBasisEnsemble = 5
maxBasisClosure  = 5
channels = ['zz',
            'zh',
            'hh',
            ]

# BEs = [                                                '1',
#                                                    '2*x-1',
#                                             '6*x^2 -6*x+1',
#                                    '20*x^3 -30*x^2+12*x-1',
#                           '70*x^4 -140*x^3 +90*x^2-20*x+1',
#                 '252*x^5 -630*x^4 +560*x^3-210*x^2+30*x-1',
#        '924*x^6-2772*x^5+3150*x^4-1680*x^3+420*x^2-42*x+1',
#                                           '3432*x^7-  12012*x^6+ 16632*x^5- 11550*x^4+ 4200*x^3- 756*x^2+ 56*x-1',
#                              '12870*x^8-  51480*x^7+  84084*x^6- 72072*x^5+ 34650*x^4- 9240*x^3+1260*x^2- 72*x+1',
#                  '48620*x^9- 218790*x^8+ 411840*x^7- 420420*x^6+252252*x^5- 90090*x^4+18480*x^3-1980*x^2+ 90*x-1',
#     '184756*x^10-923780*x^9+1969110*x^8-2333760*x^7+1681680*x^6-756756*x^5+210210*x^4-34320*x^3+2970*x^2-110*x+1',
#        ]
BEs = ['1',           # 0
       'sin(1*pi*x)', # 1
       'cos(1*pi*x)', # 2
       'sin(2*pi*x)', # 3
       'cos(2*pi*x)', # 4
       'sin(3*pi*x)', # 5
       'cos(3*pi*x)', # 6
       'sin(4*pi*x)', # 7
       'cos(4*pi*x)', # 8
       'sin(5*pi*x)', # 9
       'cos(5*pi*x)', #10
]

BE = []
for i, s in enumerate(BEs): 
    BE.append( ROOT.TF1('BE%d'%i, s, 0, 1) )

USER = getUSER()
CMSSW = getCMSSW()
basePath = '/uscms/home/%s/nobackup/%s/src'%(USER, CMSSW)

#mixName = '3bMix4b_rWbW2'
mixName = '3bDvTMix4bDvT'
ttAverage = False

doSpuriousSignal = True
dataAverage = True
nMixes = 15
region = 'SR'
#region = 'SRNoHH'
#region = 'notSR'
#hists_closure_MixedToUnmixed_3bMix4b_rWbW2_b0p60p3_SRNoHH_e25_os012.root
#hists_closure_MixedToUnmixed_3bMix4b_rWbW2_b0p60p3_SRNoHH.root
#closureFileName = 'ZZ4b/nTupleAnalysis/combine/hists_closure_MixedToUnmixed_'+mixName+'_b0p60p3_'+region+'.root'
#closureFileName = 'ZZ4b/nTupleAnalysis/combine/hists_closure_'+mixName+'_b0p60p3_'+region+'.root'
closureFileName = 'ZZ4b/nTupleAnalysis/combine/hists_closure_'+mixName+'_'+region+'_weights_newSBDef.root'
if 'MA' in classifier:
    closureFileName = closureFileName.replace('weights_', 'weights_MA_')

print(closureFileName)
f=ROOT.TFile(closureFileName, 'UPDATE')
mixes = [mixName+'_v%d'%i for i in range(nMixes)]

probThreshold = 0.05 #0.045500263896 #0.682689492137 # 1sigma 

regionName = {'SB': 'Sideband',
              'CR': 'Control Region',
              'SR': 'Signal Region',
              'notSR': 'Sideband',
              'SRNoHH': 'Signal Region (Veto HH)',
          }

        
def addYears(directory, processes=['ttbar','multijet','data_obs']):
    hists = []
    for process in processes:        
        try:
            f.Get('%s/%s'%(directory,process)).IsZombie()
            # already exists, don't need to make it
        except ReferenceError:
            # make sum of years
            hists.append( f.Get(directory+'2016/'+process) )
            hists[-1].Add(f.Get(directory+'2017/'+process))
            hists[-1].Add(f.Get(directory+'2018/'+process))
            try:
                f.Get(directory).IsZombie()
            except ReferenceError:
                f.mkdir(directory)
            f.cd(directory)
            hists[-1].Write()

def addMixes(directory):
    hists = []
    for process in ['ttbar','multijet','data_obs']:
        try:
            f.Get('%s/%s'%(directory,process)).IsZombie()
        except ReferenceError:
            hists.append( f.Get(mixes[0]+'/'+directory+'/'+process) )

            if ttAverage and process=='ttbar': # skip averaging if ttAverage and process=='ttbar'
                pass
            else:
                for mix in mixes[1:]:
                    hists[-1].Add( f.Get(mix+'/'+directory+'/'+process) )
                hists[-1].Scale(1.0/nMixes)

            if process=='multijet' or process=='ttbar':
                for bin in range(1,hists[-1].GetSize()-1):
                    hists[-1].SetBinError(bin, nMixes**0.5 * hists[-1].GetBinError(bin))

            try:
                f.Get(directory).IsZombie()
            except ReferenceError:
                f.mkdir(directory)

            f.cd(directory)

            hists[-1].Write()

for channel in channels:
    for mix in mixes:
       addYears(mix+'/'+channel)
    try:
        addYears(mixName+'_vAll_oneFit/'+channel, processes=['multijet'])
    except:
        print('Do not have '+mixName+'_vAll_oneFit/'+channel)

for channel in channels:
   addMixes(channel)
   for year in ['2016', '2017', '2018']:
       addMixes(channel+year)

# Get Signal templates for spurious signal fits
zzFile = ROOT.TFile('/uscms/home/%s/nobackup/ZZ4b/ZZ4bRunII/hists.root'%(USER), 'READ')
zhFile = ROOT.TFile('/uscms/home/%s/nobackup/ZZ4b/bothZH4bRunII/hists.root'%(USER), 'READ')
hhFile = ROOT.TFile('/uscms/home/%s/nobackup/ZZ4b/HH4bRunII/hists.root'%(USER), 'READ')
for ch in channels:
    var = '%s_ps_%s'%(classifier, ch)
    histPath = 'passPreSel/fourTag/mainView/%s/%s'%(region,var)
    signal =   zzFile.Get(histPath)
    signal.Add(zhFile.Get(histPath))
    signal.Add(hhFile.Get(histPath))
    signal.SetName('signal')
    f.cd(ch)
    signal.Write()
zzFile.Close()
zhFile.Close()
hhFile.Close()

f.Close()
f=ROOT.TFile(closureFileName, 'UPDATE')


def pearsonr(x,y,n=None):
    r, p_raw = scipy.stats.pearsonr(x,y)
    if n is None: 
        return (r, p_raw)
    # if n <= 2: # pearson r cdf is not well defined for n<=2
    #     return (r, 1.)
    # corrected p-value using different number of degrees of freedom than just the number of samples (array length)
    dist = scipy.stats.beta(n/2. - 1, n/2. - 1, loc=-1, scale=2)
    p_cor = 2*dist.cdf(-abs(r))
    # print('p_raw = %2.0f%%'%(p_raw*100))
    # print('p_cor = %2.0f%%'%(p_cor*100)) # p_cor should always be larger than p_raw
    return (r, p_cor)

def fTest(chi2_1, chi2_2, ndf_1, ndf_2):
    print('chi2_1, chi2_2, ndf_1, ndf_2 = %f, %f, %d, %d'%(chi2_1, chi2_2, ndf_1, ndf_2))
    d1 = (ndf_1-ndf_2)
    d2 = ndf_2
    print('d1, d2 = %d, %d'%(d1, d2))
    N = (chi2_1-chi2_2)/d1
    D = chi2_2/d2
    print('N, D = %f, %f'%(N, D))
    fStat = N/D
    fProb = scipy.stats.f.cdf(fStat, d1, d2)
    expectedFStat = scipy.stats.distributions.f.isf(0.05, d1, d2)
    print('    f(%i,%i) = %f (expected at 95%%: %f)'%(d1,d2,fStat,expectedFStat))
    print('f.cdf(%i,%i) = %3.0f%%'%(d1,d2,100*fProb))
    print()
    return fProb




class multijetEnsemble:
    def __init__(self, channel):
        self.channel = channel
        self.rebin = rebin[channel]
        mkpath('closureFits/%s/%s/rebin%i/%s/%s'%(mixName, classifier, self.rebin, region, self.channel))

        self.data_minus_ttbar = f.Get('%s/ttbar'%self.channel)
        self.data_minus_ttbar.SetName('%s_average_%s'%('data_minus_ttbar', self.channel))
        self.data_minus_ttbar.Scale(-1)
        self.data_minus_ttbar.Add(f.Get('%s/data_obs'%self.channel))
        self.data_minus_ttbar.Rebin(self.rebin)
        self.average = f.Get('%s/multijet'%self.channel)
        self.average.SetName('%s_average_%s'%(self.average.GetName(), self.channel))
        self.models  = [f.Get('%s/%s/multijet'%(mix, self.channel)) for mix in mixes]
        for m, model in enumerate(self.models): model.SetName('%s_%s_%s'%(model.GetName(), mixes[m], self.channel))
        self.nBins   = self.average.GetSize()-2 # size includes under/overflow bins

        try:
            self.allMixFvT = f.Get('%s/%s/multijet'%(mixName+'_vAll_oneFit', self.channel))
            self.allMixFvT.SetName('%s_allMixFvT_%s'%(self.allMixFvT.GetName(), self.channel))
        except:
            self.allMixFvT = None

        self.signal = f.Get('%s/signal'%self.channel)
        self.signal.Rebin(self.rebin)

        f.cd(self.channel)

        self.average_rebin = self.average.Clone()
        self.average_rebin.SetName('%s_rebin'%self.average.GetName())
        self.average_rebin.Rebin(self.rebin)

        self.models_rebin = [model.Clone() for model in self.models]
        for model in self.models_rebin: model.SetName('%s_rebin'%model.GetName())
        for model in self.models_rebin: model.Rebin(self.rebin)
        self.nBins_rebin = self.average_rebin.GetSize()-2

        if self.allMixFvT is not None:
            self.allMixFvT_rebin = self.allMixFvT.Clone()
            self.allMixFvT_rebin.SetName('%s_rebin'%self.allMixFvT.GetName())
            self.allMixFvT_rebin.Rebin(self.rebin)
        else:
            self.allMixFvT_rebin = None


        f.cd(self.channel)
        self.nBins_ensemble = self.nBins_rebin * nMixes
        self.bin_width = 1./self.nBins_rebin
        self.fit_bin_min = int(1 + closure_fit_x_min//self.bin_width)
        self.nBins_fit = self.nBins_rebin - int(closure_fit_x_min//self.bin_width)
        self.multijet_ensemble_average = ROOT.TH1F('multijet_ensemble_average', '', self.nBins_ensemble, 0.5, 0.5+self.nBins_ensemble)
        self.multijet_ensemble         = ROOT.TH1F('multijet_ensemble'        , '', self.nBins_ensemble, 0.5, 0.5+self.nBins_ensemble)
        if self.allMixFvT is not None:
            self.allMixFvT_ensemble    = ROOT.TH1F('allMixFvT_ensemble'       , '', self.nBins_ensemble, 0.5, 0.5+self.nBins_ensemble)
        else:
            self.allMixFvT_ensemble    = None
        self.data_minus_ttbar_ensemble         = ROOT.TH1F('data_minus_ttbar_ensemble'        , '', self.nBins_ensemble, 0.5, 0.5+self.nBins_ensemble)

        for m in range(nMixes):
            for b in range(self.nBins_rebin):
                local_bin    = 1 + b
                ensemble_bin = 1 + b + m*self.nBins_rebin
                #error = (self.models_rebin[m].GetBinError(local_bin)**2 + (self.average_rebin.GetBinError(local_bin)/nMixes)**2 + (2/nMixes)**2)**0.5
                error = (self.models_rebin[m].GetBinError(local_bin)**2 + (2/nMixes)**2)**0.5
                self.multijet_ensemble_average.SetBinContent(ensemble_bin, self.average_rebin.GetBinContent(local_bin))
                self.multijet_ensemble_average.SetBinError  (ensemble_bin, error)
                
                self.multijet_ensemble        .SetBinContent(ensemble_bin, self.models_rebin[m].GetBinContent(local_bin))
                #self.multijet_ensemble        .SetBinError  (ensemble_bin, self.models_rebin[m].GetBinError  (local_bin))
                self.multijet_ensemble        .SetBinError  (ensemble_bin, 0.0)

                if self.allMixFvT_ensemble is not None:
                    self.allMixFvT_ensemble.SetBinContent(ensemble_bin, self.allMixFvT_rebin.GetBinContent(local_bin))
                    #self.allMixFvT_ensemble.SetBinError  (ensemble_bin, self.allMixFvT_rebin.GetBinError(local_bin))
                    self.allMixFvT_ensemble.SetBinError  (ensemble_bin, 0.0)
                self.data_minus_ttbar_ensemble.SetBinContent(ensemble_bin, self.data_minus_ttbar.GetBinContent(local_bin))
                self.data_minus_ttbar_ensemble.SetBinError  (ensemble_bin, self.data_minus_ttbar.GetBinError  (local_bin))

        f.cd(self.channel)
        self.multijet_ensemble_average.Write()
        self.multijet_ensemble        .Write()
        if self.allMixFvT_ensemble is not None:
            self.allMixFvT_ensemble   .Write()
        self.data_minus_ttbar_ensemble.Write()


        self.bases = range(0, maxBasisEnsemble+1, 1)
        # Make kernel for basis orthogonalization
        h = np.array([self.average_rebin.GetBinContent(bin) for bin in range(1,self.nBins_rebin+1)])
        h_err = np.array([self.multijet_ensemble_average.GetBinError(bin) for bin in range(1,self.nBins_rebin+1)])
        # h = np.array([self.average_rebin.GetBinError(bin)+2 for bin in range(1,self.nBins_rebin+1)])
        self.h = h
        # Make matrix of initial basis
        B = np.array([[b.Integral(self.average_rebin.GetBinLowEdge(bin), self.average_rebin.GetXaxis().GetBinUpEdge(bin))/self.average_rebin.GetBinWidth(bin) for bin in range(1,self.nBins_rebin+1)] for b in BE])
        S = np.array([[self.signal.GetBinContent(bin) for bin in range(1,self.nBins_rebin+1)]])
        S = S/h
        S = S/S.max()
        S = S.repeat(len(BE), axis=0)
        self.basis_element = B
        self.basis_signal  = S
        for basis in self.bases[1:]:
            self.plotBasis('initial', basis)
        # Subtract off cross correlation from higher order basis elements
        for i in range(1,len(B)):
            c = (B[i-1]*h**1.0*B[i-1]).sum() 
            B[i:] = B[i:] - (B[i-1]*h**1.0*B[i:]).sum(axis=1, keepdims=True)*B[i-1]/c # make each b_i orthogonal to those before it
        for i in range(0,len(B)):
            B[i] = B[i] * np.sign(B[i,-1]) # set all b_i's to be positive for the last bin
            c = (B[i]*h**1.0*B[i]).sum()
            S[i:] = S[i:] - (B[i]*h**1.0*S[i:]).sum(axis=1, keepdims=True)*B[i]/c # make each s_i orthogonal to the b_j where j<=i
        for basis in self.bases[1:]:
            self.plotBasis('diagonalized', basis)
        # scale dynamic range of each element to 1
        for i in range(1,len(B)):
            d = B[i].max() - B[i].min()
            B[i] = B[i]/d
        for i in range(len(S)):
            S[i] = S[i]/S[i,-1] * self.signal.GetBinContent(self.nBins_rebin)/h[-1]
        for basis in self.bases[1:]:
            self.plotBasis('normalized', basis)



        self.fit_result = {}
        self.eigenVars = {}
        self.multijet_TF1, self.multijet_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.pulls = {}
        self.pearsonr = {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.cUp, self.cDown = {}, {}
        # self.fProb = {}
        self.basis = None
        self.exit_message = ['--- None (%s) --- Multijet Ensemble'%self.channel.upper()]
        min_r = 1.0
        for basis in self.bases:
            self.makeFitFunction(basis)
            self.fit(basis)
            self.plotFitResults(basis)
            for i in range(1,basis):
                self.plotFitResults(basis, projection=(i,i+1))
            self.plotPulls(basis)

            #if abs(self.pearsonr[basis]['total'][0]) < min_r:
            if self.basis is None and abs(self.pearsonr[basis]['total'][1]) > probThreshold:
                min_r = abs(self.pearsonr[basis]['total'][0])
                self.basis = basis # store first basis to satisfy min threshold. Will be used in closure fits
                self.exit_message = []
                self.exit_message.append('-'*50)
                self.exit_message.append('%s channel'%self.channel.upper())
                self.exit_message.append('Satisfied adjacent bin de-correlation p-value for multijet ensemble variance at basis %d:'%self.basis)
                self.exit_message.append('>> p-value, r-value = %2.0f%%, %0.2f '%(100*self.pearsonr[self.basis]['total'][1], self.pearsonr[self.basis]['total'][0]))
                self.exit_message.append('-'*50)
        if self.basis is None:
            self.basis = self.bases[ np.argmin([abs(self.pearsonr[basis]['total'][0]) for basis in self.bases]) ]
            self.exit_message = []
            self.exit_message.append('-'*50)
            self.exit_message.append('%s channel'%self.channel.upper())
            self.exit_message.append('Minimized adjacent bin correlation abs(r) for multijet ensemble variance at basis %d:'%self.basis)
            self.exit_message.append('>> p-value, r-value = %2.0f%%, %0.2f '%(100*self.pearsonr[self.basis]['total'][1], self.pearsonr[self.basis]['total'][0]))
            self.exit_message.append('-'*50)

        self.plotPearson()

    def print_exit_message(self):
        for line in self.exit_message: print(line)

    def makeFitFunction(self, basis):

        def background_UserFunction(xArray, pars):
            ensemble_bin = int(xArray[0])
            m = (ensemble_bin-1)//self.nBins_rebin
            local_bin = 1 + ((ensemble_bin-1)%self.nBins_rebin)
            model = self.models[m]

            l, u = self.average_rebin.GetBinLowEdge(local_bin), self.average_rebin.GetXaxis().GetBinUpEdge(local_bin)

            p = 1.0                
            for BE_idx in range(basis+1):
                par_idx = m*(basis+1)+BE_idx
                p += pars[par_idx] * self.basis_element[BE_idx][local_bin-1]

            return p*self.multijet_ensemble.GetBinContent(ensemble_bin)

        f.cd(self.channel)
        self.multijet_TF1[basis] = ROOT.TF1 ('multijet_ensemble_TF1_basis%d'%basis, background_UserFunction, 0.5, 0.5+self.nBins_ensemble, nMixes*(basis+1))
        self.multijet_TH1[basis] = ROOT.TH1F('multijet_ensemble_TH1_basis%d'%basis, '', self.nBins_ensemble, 0.5, 0.5+self.nBins_ensemble)

        for m in range(nMixes):
            for o in range(basis+1):
                self.multijet_TF1[basis].SetParName  (m*(basis+1)+o, 'v%d c_%d'%(m, o))
                self.multijet_TF1[basis].SetParameter(m*(basis+1)+o, 0.0)



    def getEigenvariations(self, basis=None, debug=False):
        if basis is None: basis = self.basis
        n = basis+1
        
        if n == 1:
            self.eigenVars[basis] = [np.array([[self.multijet_TF1[basis].GetParError(m*n)]]) for m in range(nMixes)]
            return

        cov = [ROOT.TMatrixD(n,n) for m in range(nMixes)]
        cor = [ROOT.TMatrixD(n,n) for m in range(nMixes)]

        for m in range(nMixes):
            for i in range(n):
                for j in range(n): # full fit is block diagonal in nMixes blocks since there is no correlation between fit parameters of different multijet models
                    cov[m][i][j] = self.fit_result[basis].CovMatrix  (m*n+i, m*n+j)
                    cor[m][i][j] = self.fit_result[basis].Correlation(m*n+i, m*n+j)

        if debug:
            for m in range(nMixes):
                print('Covariance Matrix:',m)
                cov[m].Print()
                print('Correlation Matrix:',m)
                cor[m].Print()
        
        eigenVal = [ROOT.TVectorD(n) for m in range(nMixes)]
        eigenVec = [cov[m].EigenVectors(eigenVal[m]) for m in range(nMixes)]
        
        for m in range(nMixes):
            # define relative sign of eigen-basis such that the first coordinate is always positive
            for j in range(n):
                if eigenVec[m][0][j] >= 0: continue
                for i in range(n):
                    eigenVec[m][i][j] *= -1

            if debug:
                print('Eigenvectors (columns)',m)
                eigenVec[m].Print()
                print('Eigenvalues',m)
                eigenVal[m].Print()

        self.eigenVars[basis] = [np.zeros((n,n), dtype=np.float) for m in range(nMixes)]
        for m in range(nMixes):
            for i in range(n):
                for j in range(n):
                    self.eigenVars[basis][m][i,j] = eigenVec[m][i][j] * eigenVal[m][j]**0.5

        if debug:
            for m in range(nMixes):
                print('Eigenvariations',m)
                for j in range(n):
                    print(j, self.eigenVars[basis][m][:,j])


    def getParameterDistribution(self, basis):
        n = basis+1
        parMean    = np.array([0 for i in range(n)], dtype=np.float)
        parMeanErr = np.array([0 for i in range(n)], dtype=np.float)
        parMean2   = np.array([0 for i in range(n)], dtype=np.float)
        for m in range(nMixes):
            parMean    += self.fit_parameters[basis][m]    / nMixes
            parMean2   += self.fit_parameters[basis][m]**2 / nMixes
            parMeanErr += self.fit_parameters_error[basis][m] / nMixes
        var = parMean2 - parMean**2
        parStd  = var**0.5 
        parStd *= nMixes / (nMixes-1) # bessel's correction https://en.wikipedia.org/wiki/Bessel's_correction
        print('Parameter Mean:',parMean)
        print('Parameter  Std:',parStd)

        for i in range(n):
            #cUp   =  ( (abs(parMean[i])+parStd[i])**2 + parMeanErr[i]**2 )**0.5 # * n**0.5 # add scaling term so that 1 sigma corresponds to quadrature sum over i of (abs(parMean[i])+parStd[i])
            cUp   =  abs(parMean[i])+parStd[i]
            cDown = -cUp
            try:
                self.cUp  [basis].append( cUp )
                self.cDown[basis].append( cDown )
            except KeyError:
                self.cUp  [basis] = [cUp  ]
                self.cDown[basis] = [cDown]


    def fit(self, basis):
        self.fit_result[basis] = self.multijet_ensemble_average.Fit(self.multijet_TF1[basis], 'N0SQ')
        self.getEigenvariations(basis)
        self.pvalue[basis], self.chi2[basis], self.ndf[basis] = self.multijet_TF1[basis].GetProb(), self.multijet_TF1[basis].GetChisquare(), self.multijet_TF1[basis].GetNDF()
        print('Fit multijet ensemble %s at basis %d'%(self.channel, basis))
        print('chi2/ndf = %3.2f/%3d = %2.2f'%(self.chi2[basis], self.ndf[basis], self.chi2[basis]/self.ndf[basis]))
        print(' p-value = %0.2f'%self.pvalue[basis])

        self.ymax[basis] = self.multijet_TF1[basis].GetMaximum(1,self.nBins_ensemble)
        self.fit_parameters[basis], self.fit_parameters_error[basis] = [], []
        n = basis+1
        for m in range(nMixes):
            self.fit_parameters      [basis].append( np.array([self.multijet_TF1[basis].GetParameter(m*n+o) for o in range(basis+1)]) )
            self.fit_parameters_error[basis].append( np.array([self.multijet_TF1[basis].GetParError (m*n+o) for o in range(basis+1)]) )
        self.getParameterDistribution(basis)

        for bin in range(1,self.nBins_ensemble+1):
            self.multijet_TH1[basis].SetBinContent(bin, self.multijet_TF1[basis].Eval(bin))
            #self.multijet_TH1[basis].SetBinError  (bin, self.multijet_ensemble.GetBinError(bin))
            self.multijet_TH1[basis].SetBinError  (bin, 0.0)

        pulls = []
        bins = range(self.fit_bin_min,self.nBins_ensemble+1)
        for bin in bins:
            error = self.multijet_ensemble_average.GetBinError(bin)
            pull = (self.multijet_TF1[basis].Eval(bin) - self.multijet_ensemble_average.GetBinContent(bin))/error if error>0 else 0
            pulls.append(pull)
        self.pulls[basis] = np.array(pulls)

        # check bin to bin correlations using pearson R test
        xs = np.array([self.pulls[basis][m*self.nBins_fit  : (m+1)*self.nBins_fit-1] for m in range(nMixes)])
        ys = np.array([self.pulls[basis][m*self.nBins_fit+1: (m+1)*self.nBins_fit  ] for m in range(nMixes)])
        # x1s = np.array([self.pulls[basis][m*self.nBins_fit  : (m+1)*self.nBins_fit-1] for m in range(nMixes)])
        # y1s = np.array([self.pulls[basis][m*self.nBins_fit+1: (m+1)*self.nBins_fit  ] for m in range(nMixes)])
        # x2s = np.array([self.pulls[basis][m*self.nBins_fit  : (m+1)*self.nBins_fit-2] for m in range(nMixes)])
        # y2s = np.array([self.pulls[basis][m*self.nBins_fit+2: (m+1)*self.nBins_fit  ] for m in range(nMixes)])
        # x3s = np.array([self.pulls[basis][m*self.nBins_fit  : (m+1)*self.nBins_fit-3] for m in range(nMixes)])
        # y3s = np.array([self.pulls[basis][m*self.nBins_fit+3: (m+1)*self.nBins_fit  ] for m in range(nMixes)])
        # xs, ys = np.concatenate((x1s,x2s,x3s), axis=1), np.concatenate((y1s,y2s,y3s), axis=1)
        x, y = xs.flatten(), ys.flatten()
        r, p = pearsonr(x,y, n=len(x)-nMixes*(basis+1))
        self.pearsonr[basis] = {'total': (r, p),
                                #'mixes': [pearsonr(xs[m],ys[m], n=self.nBins_fit-1 - basis-1) for m in range(nMixes)]}
                                'mixes': [pearsonr(xs[m],ys[m],n=len(xs[m])-basis-1) for m in range(nMixes)]}
        print('-------------------------')
        print('>> r, prob = %0.2f, %0.2f'%self.pearsonr[basis]['total'])
        print('-------------------------')
        # n = x.shape[0] - nMixes*(basis+1)
        # dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
        # p_manual = 2*dist.cdf(-abs(r))
        # print('manual R p-value: n, p = %d, %f'%(n,p_manual))
        # raw_input()
        f.cd(self.channel)
        self.multijet_TH1[basis].Write()


    def plotBasis(self, name, basis):
        fig, (ax) = plt.subplots(nrows=1)
        x = [self.average_rebin.GetBinCenter(bin) for bin in range(1, self.nBins_rebin+1)]
        xlim = [0,1]
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_title('%s Multiplicitive Basis (%s)'%(name[0].upper()+name[1:], self.channel.upper()))

        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        for i, y in enumerate(self.basis_element[:basis+1]):
            ax.plot(x, y, label='b$_{%i}$'%i, linewidth=1)

        # if name == 'normalized':
        #     ax.plot(x, self.basis_signal[basis]*10, label=r'Spurious Signal ($\times 10$)', linewidth=1)
        # else:
        #     ax.plot(x, self.basis_signal[basis],    label=r'Spurious Signal',               linewidth=1)

        ax.set_xlabel('P(Signal)')
        ax.set_ylabel('Multijet Scale')

        ax.legend(fontsize='small', loc='best')

        if type(self.rebin) is list:
            figname = 'closureFits/%s/%s/variable_rebin/%s/%s/%s_basis%i.pdf'%(mixName, classifier, region, self.channel, name, basis)
        else:
            figname = 'closureFits/%s/%s/rebin%i/%s/%s/%s_basis%i.pdf'%(mixName, classifier, self.rebin, region, self.channel, name, basis)
        print('fig.savefig( '+figname+' )')
        plt.tight_layout()
        fig.savefig( figname )
        plt.close(fig)

        fig, (ax) = plt.subplots(nrows=1)
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_xticks(np.arange(0,1.1,0.1))

        ax.set_title('%s Additive Basis (%s)'%(name[0].upper()+name[1:], self.channel.upper()))

        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        for i, y in enumerate(self.basis_element[:basis+1]):
            ax.plot(x, y*self.h, label='b$_{%i}$'%i, linewidth=1)

        # if name == 'normalized':
        #     ax.plot(x, self.basis_signal[basis]*self.h*100, label=r'Spurious Signal ($\times 100$)', linewidth=1)
        # else:
        #     ax.plot(x, self.basis_signal[basis]*self.h,     label=r'Spurious Signal',                linewidth=1)

        ax.set_xlabel('P(Signal)')
        ax.set_ylabel('Events')

        ax.legend(fontsize='small', loc='best')

        if type(self.rebin) is list:
            figname = 'closureFits/%s/%s/variable_rebin/%s/%s/%s_additive_basis%i.pdf'%(mixName, classifier, region, self.channel, name, basis)
        else:
            figname = 'closureFits/%s/%s/rebin%i/%s/%s/%s_additive_basis%i.pdf'%(mixName, classifier, self.rebin, region, self.channel, name, basis)
        print('fig.savefig( '+figname+' )')
        plt.tight_layout()
        fig.savefig( figname )
        plt.close(fig)
        

    def plotPearson(self):
        fig, (ax) = plt.subplots(nrows=1)
        #ax.set_ylim(0.001,1)
        #plt.yscale('log')
        x = np.array(sorted(self.pearsonr.keys()))+1
        ax.set_ylim(-1,1)
        ax.set_xticks(x)
        xlim = [x[0]-0.5, x[-1]+0.5]
        ax.set_xlim(xlim[0],xlim[1])
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)

        r = np.array([self.pearsonr[o]['total'][0] for o in x-1])
        p = np.array([self.pearsonr[o]['total'][1] for o in x-1])
        ax.set_title('Multijet Model Variance Fits (%s)'%self.channel.upper())
        ax.plot(x, r, label='Combined', color='k', linewidth=2)
        ax.plot(x, p, label='p-value',  color='r', linewidth=2)

        if self.basis is not None:
            ax.plot([self.basis+1, self.basis+1], [-1,1], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
            ax.scatter(self.basis+1, p[x==(self.basis+1)], color='k', marker='*', s=100, zorder=10)

        for m in range(nMixes):
            r = [self.pearsonr[o]['mixes'][m][0] for o in x-1]
            #p = [self.pearsonr[o]['mixes'][m][1] for o in x]
            label = 'v$_{%d}$'%m
            ax.plot(x, r, color=COLORS[m], linewidth=1, alpha=0.5, label=label)#underscore tells pyplot to not show this in the legend
            # ax.plot(x, r, color=colors[m], linewidth=1, alpha=0.3, linestyle='dotted', label='_'+label)#underscore tells pyplot to not show this in the legend
            # ax.plot(x, p, color=colors[m], linewidth=1, alpha=0.3, label=label)
        
        ax.plot(xlim, [probThreshold,probThreshold], color='r', alpha=0.5, linestyle='--', linewidth=0.5)

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Adjacent Bin Pearson Correlation (r) and p-value')
        plt.legend(fontsize='small', loc='best')

        if type(self.rebin) is list:
            name = 'closureFits/%s/%s/variable_rebin/%s/%s/0_variance_pearsonr_multijet_variance.pdf'%(mixName, classifier, region, self.channel)
        else:
            name = 'closureFits/%s/%s/rebin%i/%s/%s/0_variance_pearsonr_multijet_variance.pdf'%(mixName, classifier, self.rebin, region, self.channel)
        print('fig.savefig( '+name+' )')
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)


    def plotFitResults(self, basis, projection=(0,1)):
        n = basis+1
        if n>1:
            dims = tuple(list(projection) + [d for d in range(n) if d not in projection])
        else:
            dims = (0,1)

        #plot fit parameters
        x,y,s,c = [],[],[],[]
        for m in range(nMixes):
            x.append( 100*self.fit_parameters[basis][m][dims[0]] )
            if n==1:
                y.append( 0 )
            if n>1:
                y.append( 100*self.fit_parameters[basis][m][dims[1]] )
            if n>2:
                c.append( 100*self.fit_parameters[basis][m][dims[2]] )
            if n>3:
                s.append( 100*self.fit_parameters[basis][m][dims[3]] )

        x = np.array(x)
        y = np.array(y)
    
        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }
        if n>2:
            kwargs['c'] = c
            kwargs['cmap'] = 'BuPu'
        if n>3:
            s = np.array(s)
            smin = s.min()
            smax = s.max()
            srange = smax-smin
            s = s-s.min() #shift so that min is at zero
            s = s/s.max() #scale so that max is 1
            s = (s+5.0/25)*25 #shift and scale so that min is 5.0 and max is 25+5.0
            kwargs['s'] = s

        fig, (ax) = plt.subplots(nrows=1, figsize=(7,6)) if n>2 else plt.subplots(nrows=1, figsize=(6,6))
        ax.set_aspect(1)
        ax.set_title('Multijet Model Variance Fits (%s)'%self.channel.upper())
        ax.set_xlabel('c$_'+str(dims[0])+'$ (\%)')
        ax.set_ylabel('c$_'+str(dims[1])+'$ (\%)')

        xlim, ylim = [-8,8], [-8,8]
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0,0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = np.arange(-6, 8, 2)
        yticks = np.arange(-6, 8, 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if n>1:
            # draw 1\sigma ellipse 
            ellipse = Ellipse((0,0), 
                              width =100*(self.cUp[basis][dims[0]]-self.cDown[basis][dims[0]]),
                              height=100*(self.cUp[basis][dims[1]]-self.cDown[basis][dims[1]]),
                              facecolor = 'none',
                              edgecolor = 'b', # CMURED,
                              linestyle = '-',
                              linewidth = 0.75,
                              zorder=1,
            )
            ax.add_patch(ellipse)

        bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0, pad=0)
        if n>2:
            # draw range bars for other priors
            for i, d in enumerate(dims[2:]):
                thisx = xlim[-1] - 0.5*(n-2) + 0.5*i
                up, down = self.cUp[basis][d], self.cDown[basis][d]
                ax.quiver(thisx, 0, 0, 100*up,   color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                ax.quiver(thisx, 0, 0, 100*down, color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

                ax.annotate('c$_{%d}$'%(d), [thisx, 100*down-0.5], ha='center', va='center', bbox=bbox)

        maxr=np.zeros((2, len(x)), dtype=np.float)
        minr=np.zeros((2, len(x)), dtype=np.float)
        if n>1:
            #generate a ton of random points on a hypersphere in dim=n so surface is dim=n-1.
            points  = np.random.randn(n, min(100**(n-1),10**7)) # random points in a hypercube
            points /= np.linalg.norm(points, axis=0) # normalize them to the hypersphere surface

            # for each model, find the point which maximizes the change in c_0**2 + c_1**2
            for m in range(nMixes):
                plane = np.matmul( self.eigenVars[basis][m][dims[:2],:], points )
                r2 = plane[0]**2
                if n>1:
                    r2 += plane[1]**2

                maxr[:,m] = plane[:,r2==r2.max()].T[0]

                #construct orthogonal unit vector to maxr
                minrvec = np.copy(maxr[::-1,m])
                minrvec[0] *= -1
                minrvec /= np.linalg.norm(minrvec)

                #find maxr along minrvec to get minr
                dr2 = np.matmul( minrvec, plane )**2
                #minr[:,m] = plane[:,dr2==dr2.max()].T[0]#this guy is the ~right length but might be slightly off orthogonal
                minr[:,m] = minrvec * dr2.max()**0.5#this guy is the ~right length and is orthogonal by construction
        else:
            for m in range(nMixes):
                maxr[0,m] = self.eigenVars[basis][m][dims[0]]
        
        minr *= 100
        maxr *= 100

        # print(maxr)
        # print(minr)
        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        
        plt.scatter(x, y, **kwargs)
        plt.tight_layout()

        for m in range(nMixes):
            x_offset, y_offset = (maxr[0,m]+minr[0,m])/2, (maxr[1,m]+minr[1,m])/2
            ax.annotate('v$_{%d}$'%m, (x[m]+x_offset, y[m]+y_offset), bbox=bbox)

        if n>2:
            plt.colorbar(label='c$_'+str(dims[2])+'$ (\%)')#, cax=cax) 
            plt.subplots_adjust(right=1)

        if n>3:
            l1 = plt.scatter([],[], s=(0.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
            l2 = plt.scatter([],[], s=(1.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
            l3 = plt.scatter([],[], s=(2.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
            l4 = plt.scatter([],[], s=(3.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')

            handles = [l1,
                       l2,
                       l3,
                       l4]
            labels = ['%0.2f'%smin,
                      '%0.2f'%(smin+srange*1.0/3),
                      '%0.2f'%(smin+srange*2.0/3),
                      '%0.2f'%smax]

            leg = plt.legend(handles, labels, 
                             ncol=1, 
                             fontsize='medium',
                             loc='best',
                             title='c$_'+str(dims[3])+'$ (\%)', 
                             scatterpoints = 1)
        
        projection = '_'.join([str(d) for d in projection])
        if type(self.rebin) is list:
            name = 'closureFits/%s/%s/variable_rebin/%s/%s/0_variance_parameters_basis%d_projection_%s.pdf'%(mixName, classifier, region, self.channel, basis, projection)
        else:
            name = 'closureFits/%s/%s/rebin%i/%s/%s/0_variance_parameters_basis%d_projection_%s.pdf'%(mixName, classifier, self.rebin, region, self.channel, basis, projection)
        print('fig.savefig( '+name+' )')
        try:
            fig.savefig( name )
            plt.close(fig)
        except IndexError:
            print('Weird index error...')


    def plotPulls(self, basis):
        n = basis+1

        xs = np.array([self.pulls[basis][m*self.nBins_fit  :(m+1)*self.nBins_fit-1] for m in range(nMixes)])
        ys = np.array([self.pulls[basis][m*self.nBins_fit+1:(m+1)*self.nBins_fit  ] for m in range(nMixes)])
        # x1s = np.array([self.pulls[basis][m*self.nBins_fit  :(m+1)*self.nBins_fit-1] for m in range(nMixes)])
        # y1s = np.array([self.pulls[basis][m*self.nBins_fit+1:(m+1)*self.nBins_fit  ] for m in range(nMixes)])
        # x2s = np.array([self.pulls[basis][m*self.nBins_fit  :(m+1)*self.nBins_fit-2] for m in range(nMixes)])
        # y2s = np.array([self.pulls[basis][m*self.nBins_fit+2:(m+1)*self.nBins_fit  ] for m in range(nMixes)])
        # x3s = np.array([self.pulls[basis][m*self.nBins_fit  :(m+1)*self.nBins_fit-3] for m in range(nMixes)])
        # y3s = np.array([self.pulls[basis][m*self.nBins_fit+3:(m+1)*self.nBins_fit  ] for m in range(nMixes)])
        # xs, ys = np.concatenate((x1s,x2s,x3s), axis=1), np.concatenate((y1s,y2s,y3s), axis=1)

        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }

        fig, (ax) = plt.subplots(nrows=1, figsize=(6,6))
        ax.set_aspect(1)
        ax.set_title('Adjacent Bin Pulls (%s, %d parameters)'%(self.channel.upper(), basis+1))
        ax.set_xlabel('Bin$_{i}$, Pull')
        ax.set_ylabel('Bin$_{i+1}$ Pull')
        # ax.set_xlabel('Bin$_{2i}$, Pull')
        # ax.set_ylabel('Bin$_{2i+1}$ Pull')

        #xlim, ylim = list(ax.get_xlim()), list(ax.get_ylim())
        #lim_max = max(int(max([abs(lim) for lim in xlim+ylim])), 1)
        lim_max = 1.5*max(abs(xs).max(), abs(ys).max())
        xlim, ylim = [-lim_max, lim_max], [-lim_max, lim_max]
        #xlim, ylim = [-5,5], [-5,5]
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0,0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # xticks = np.arange(-int(lim_max)+1, int(lim_max), 1)
        # yticks = np.arange(-int(lim_max)+1, int(lim_max), 1)
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)

        for m in range(nMixes):
            #r, p = scipy.stats.pearsonr(xs[m], ys[m])
            (r, p) = self.pearsonr[basis]['mixes'][m]
            kwargs['label'] = 'v$_{%d}$, r=%0.2f (%2.0f%s)'%(m, r, p*100, '\%')
            kwargs['c'] = COLORS[m]
            plt.scatter(xs[m], ys[m], **kwargs)
        plt.tight_layout()

        #x, y = xs.flatten(), ys.flatten()
        #r, p = scipy.stats.pearsonr(x,y)
        (r, p) = self.pearsonr[basis]['total']
    
        plt.legend(fontsize='small', loc='upper left', ncol=2, title='Overall r=%0.2f (%2.0f%s)'%(r,p*100,'\%'))
        
        if type(self.rebin) is list:
            name = 'closureFits/%s/%s/variable_rebin/%s/%s/0_variance_pull_correlation_basis%d.pdf'%(mixName, classifier, region, self.channel, basis)
        else:
            name = 'closureFits/%s/%s/rebin%i/%s/%s/0_variance_pull_correlation_basis%d.pdf'%(mixName, classifier, self.rebin, region, self.channel, basis)            
        print('fig.savefig( '+name+' )')
        fig.savefig( name )
        plt.close(fig)


    def plotFit(self, basis):
        samples=collections.OrderedDict()
        samples[closureFileName] = collections.OrderedDict()
        # samples[closureFileName]['%s/data_minus_ttbar_ensemble'%self.channel] = {
        #     'label' : '#LTMixed Data#GT - #lower[0.10]{t#bar{t}}',
        #     'legend': 1,
        #     'ratioDrawOptions' : 'P ex0',
        #     'isData' : True,
        #     'ratio' : 'numer A',
        #     'color' : 'ROOT.kBlack'}
        samples[closureFileName]['%s/multijet_ensemble_average'%self.channel] = {
            'label' : '#LTMultijet Model#GT',
            'legend': 2,
            'isData' : True,
            'ratio' : 'denom A',
            'color' : 'ROOT.kBlack'}
        samples[closureFileName]['%s/multijet_ensemble'%self.channel] = {
            'label' : 'Multijet Models',
            'legend': 3,
            'stack' : 1,
            'ratio' : 'numer A',
            'color' : 'ROOT.kYellow'}
        samples[closureFileName]['%s/multijet_ensemble_TH1_basis%d'%(self.channel, basis)] = {
            'label' : 'Fit (%d parameter%s)'%(basis+1, 's' if basis else ''),
            'legend': 4,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlue'}
        # samples[closureFileName]['%s/allMixFvT_ensemble'%self.channel] = {
        #     'label' : 'FvT Fit to all Mixes',
        #     'legend': 5,
        #     'ratio' : 'numer A',
        #     'color' : 'ROOT.kGray+2'}

        xTitle = 'P(Signal) #(Bin) + #(Bins)#(Mix) #cbar P(%s) is largest'%(self.channel.upper())
            
        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.nBins_rebin*m+0.5,  0,self.nBins_rebin*m+0.5,self.ymax[0]*1.1] for m in range(1,nMixes+1)],
                      'ratioErrors': False,
                      'ratio'     : 'significance',#True,
                      'rMin'      : -3,#0.9,
                      'rMax'      :  3,#1.1,
                      'rTitle'    : 'Pulls',#'Data / Bkgd.',
                      # 'ratioErrors': True,
                      # 'ratio'      : True,
                      # 'rMin'       : 0.9,
                      # 'rMax'       : 1.1,
                      # 'rTitle'     : 'Model / Average',
                      'xTitle'    : xTitle,
                      'yTitle'    : 'Events',
                      'yMax'      : self.ymax[0]*1.6,#*ymaxScale, # make room to show fit parameters
                      'xleg'      : [0.13, 0.13+0.4],
                      'legendSubText' : [#'#bf{Fit:}',
                                         #'#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2[basis],self.ndf[basis],self.chi2[basis]/self.ndf[basis]),
                                         #'p-value = %2.0f%%'%(self.pvalue[basis]*100),
                                         '#bf{Adjacent Bin Pull Correlation:}',
                                         'r = %1.2f'%(self.pearsonr[basis]['total'][0]),
                                         'p-value = %2.0f%%'%(self.pearsonr[basis]['total'][1]*100),
                                         ],
                      'lstLocation' : 'right',
                      'rPadFraction': 0.5,
                      'outputName': '0_variance_multijet_ensemble_basis%d'%(basis)}

        parameters['ratioLines'] = [[self.nBins_rebin*m+0.5, parameters['rMin'], self.nBins_rebin*m+0.5, parameters['rMax']] for m in range(1,nMixes+1)]

        if type(self.rebin) is list:
            parameters['outputDir'] = 'closureFits/%s/%s/variable_rebin/%s/%s/'%(mixName, classifier, region, self.channel)
        else:
            parameters['outputDir'] = 'closureFits/%s/%s/rebin%i/%s/%s/'%(mixName, classifier, self.rebin, region, self.channel)

        print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
        PlotTools.plot(samples, parameters, debug=False)




class closure:
    def __init__(self, channel, multijet):
        self.channel = channel
        self.rebin = rebin[channel]
        self.multijet = multijet
        self.ttbar = f.Get('%s/ttbar'%self.channel)
        self.ttbar.SetName('%s_average_%s'%(self.ttbar.GetName(), self.channel))
        self.data_obs = f.Get('%s/data_obs'%self.channel)
        self.data_obs.SetName('%s_average_%s'%(self.data_obs.GetName(), self.channel))
        self.nBins = self.data_obs.GetSize()-2 # GetSize includes under/overflow bins

        self.doSpuriousSignal = False
        self.spuriousSignal = {}
        self.spuriousSignalError = {}
        self.closure_ss_zero_TH1 = {}
        self.closure_ss_TH1 = {}
        self.signal_orthogonal_TH1 = {}
        # self.signal = f.Get('%s/signal'%self.channel)
        # self.signal.Rebin(rebin)

        f.cd(self.channel)

        self.ttbar_rebin = self.ttbar.Clone()
        self.ttbar_rebin.SetName('%s_rebin'%self.ttbar.GetName())
        self.ttbar_rebin.Rebin(self.rebin)
        self.data_obs_rebin = self.data_obs.Clone()
        self.data_obs_rebin.SetName('%s_rebin'%self.data_obs.GetName())
        self.data_obs_rebin.Rebin(self.rebin)
        self.nBins_rebin = self.data_obs_rebin.GetSize()-2

        self.bin_width = 1./self.nBins_rebin
        self.fit_x_min = 0.5 + closure_fit_x_min/self.bin_width

        self.basis_element = self.multijet.basis_element

        f.cd(self.channel)
        #self.bases = range(self.multijet.basis, maxBasisClosure+1, 2)
        self.bases = range(-1, maxBasisClosure+1)
        max_basis = max(self.bases[-1], self.multijet.basis)
        self.nBins_closure = self.nBins_rebin + max_basis+1 # add bins for multijet shape priors
        self.multijet_closure = ROOT.TH1F('multijet_closure', '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.ttbar_closure    = ROOT.TH1F('ttbar_closure',    '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.data_obs_closure = ROOT.TH1F('data_obs_closure', '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.signal_closure   = ROOT.TH1F('signal_closure',   '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)

        for bin in range(1, self.nBins_rebin+1):
            self.multijet_closure.SetBinContent(bin, self.multijet.average_rebin.GetBinContent(bin))
            self.ttbar_closure   .SetBinContent(bin, self.ttbar_rebin           .GetBinContent(bin))
            self.signal_closure  .SetBinContent(bin, self.multijet.signal       .GetBinContent(bin))
            self.data_obs_closure.SetBinContent(bin, self.data_obs_rebin        .GetBinContent(bin))

            self.multijet_closure.SetBinError  (bin, 0.0)
            self.ttbar_closure   .SetBinError  (bin, 0.0)
            # self.signal_closure  .SetBinError  (bin, 0.0)
            # self.multijet_closure.SetBinError  (bin, self.multijet.average_rebin.GetBinError(bin))
            # self.ttbar_closure   .SetBinError  (bin, self.ttbar_rebin           .GetBinError(bin))
            self.signal_closure  .SetBinError  (bin, self.multijet.signal.GetBinError(bin))
            error = (self.data_obs_rebin.GetBinError(bin)**2 + self.ttbar_rebin.GetBinError(bin)**2 + self.multijet.average_rebin.GetBinError(bin)**2 + (2.0/nMixes)**2)**0.5 # adding 2 in quadrature improves gaussian approx of poisson errors
            self.data_obs_closure.SetBinError  (bin, error)

        for bin in range(self.nBins_rebin+1, self.nBins_closure+1):
            self.data_obs_closure.SetBinError  (bin, 1.0)

        f.cd(self.channel)
        self.multijet_closure.Write()
        self.ttbar_closure   .Write()
        self.data_obs_closure.Write()
        self.signal_closure  .Write()

        self.fit_result = {}
        self.fit_result_ss = {}
        self.eigenVars = {}
        self.eigenVars_ss = {}
        self.closure_TF1, self.closure_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.pvalue_ss, self.chi2_ss, self.ndf_ss = {}, {}, {}
        self.chi2_ss_zero, self.ndf_ss_zero = {}, {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.fit_parameters_ss, self.fit_parameters_error_ss = {}, {}
        self.cUp, self.cDown = {}, {}
        self.fProb = {-1: np.nan}
        self.fProb_ss = {}
        self.basis = None
        self.exit_message = ['--- NONE (%s) ---'%self.channel.upper()]

        for basis in self.bases:
            self.makeFitFunction(basis)
            self.fit(basis)
            self.fitSpuriousSignal(basis)
            self.writeClosureResults(basis)
            self.plotFitResults(basis)
            max_basis = max(basis, self.multijet.basis)
            for j in range(1,max_basis):
                self.plotFitResults(basis, projection=(j, j+1))
            # self.plotFitResults(basis, doSpuriousSignal=True)
            # for i in range(1,max_basis):
            #     self.plotFitResults(basis, projection=(i, i+1), doSpuriousSignal=True)
            # for i in range(0,max_basis+1):
            #     self.plotFitResults(basis, projection=(max_basis+1, i), doSpuriousSignal=True)

        for i, basis in enumerate(self.bases[:-1]):
            next_basis = self.bases[i+1]
            print('fit f-test basis',next_basis)
            #self.fProb[next_basis] = 0.5
            self.fProb[next_basis] = fTest(self.chi2[basis], self.chi2[next_basis], self.ndf[basis], self.ndf[next_basis])

            if self.basis is None and (self.pvalue[basis] > probThreshold) and (self.fProb[next_basis]<0.95):
                self.exit_message = []
                print(self.pvalue)
                print(self.fProb)
                self.basis = basis # store first basis to satisfy min threshold. Will be used in closure fits
                self.exit_message.append('-'*50)
                self.exit_message.append('%s channel'%self.channel.upper())
                self.exit_message.append('Satisfied goodness of fit and f-test')
                self.exit_message.append('>> %d, %d basis elements (variance, bias)'%(self.multijet.basis, self.basis))
                self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% with %d basis elements (p-value above threshold and f-test prefers this fit over previous)'%(100*self.pvalue[basis], 100*self.fProb[basis], basis))
                self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% with %d basis elements (f-test does not prefer this over previous fit at greater than 95%%)'%(100*self.pvalue[next_basis], 100*self.fProb[next_basis], next_basis))
                if self.fProb_ss[basis] < 0.95:
                    self.exit_message.append('>> SS f-test = %2.0f%%. Do not need to include spurious signal systematic :)'%(100*self.fProb_ss[basis]))
                else:
                    self.exit_message.append('>> SS f-test = %2.0f%%! STRONG EVIDENCE FOR SPURIOUS SIGNAL SYSTEMATIC'%(100*self.fProb_ss[basis]))
                self.exit_message.append('-'*50)

        self.plotPValues()
        if self.basis is None:
            self.basis = self.bases[-1]


    def makeFitFunction(self, basis):

        max_basis = max(basis, self.multijet.basis)
        def background_UserFunction(xArray, pars):
            bin = int(xArray[0])

            if bin > self.nBins_rebin:
                BE_idx = bin-self.nBins_rebin-1
                if self.doSpuriousSignal:

                    if BE_idx > max_basis: # do nothing with extra bins
                        return 0.0

                    BE_coefficient = pars[BE_idx]                
                    if BE_coefficient > 0:
                        return -BE_coefficient/abs(self.cUp  [basis][BE_idx])
                    else:
                        return -BE_coefficient/abs(self.cDown[basis][BE_idx])

                BE_idx += basis+1 # only apply priors to higher order terms
                if BE_idx > self.multijet.basis:
                    return 0.0
                
                # use variance priors
                BE_coefficient = pars[BE_idx]                
                if BE_coefficient>0:
                    return -BE_coefficient/abs(self.multijet.cUp  [self.multijet.basis][BE_idx])
                else:
                    return -BE_coefficient/abs(self.multijet.cDown[self.multijet.basis][BE_idx])                    


            # in distribution: evaluate basis elements times multijet
            p = 1.0
            n = max_basis+1
            for BE_idx in range(n):
                p += pars[BE_idx] * self.basis_element[BE_idx][bin-1]

            mj = self.multijet.average_rebin.GetBinContent(bin)
            background = p*mj + self.ttbar_rebin.GetBinContent(bin)
            spuriousSignal = pars[n] * self.multijet.signal.GetBinContent(bin)
            #spuriousSignal = pars[n] * mj * self.multijet.basis_signal[max_basis][bin-1]
  
            return background + spuriousSignal

        f.cd(self.channel)
        n = max_basis + 1
        self.closure_TF1[basis] = ROOT.TF1 ('closure_TF1_basis%d'%basis, background_UserFunction, 0.5, 0.5+self.nBins_closure, n+1)# +1 for spurious signal
        self.closure_TH1[basis] = ROOT.TH1F('closure_TH1_basis%d'%basis,  '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.closure_ss_zero_TH1[basis]   = ROOT.TH1F('closure_ss_zero_TH1_basis%d'%basis,   '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.closure_ss_TH1[basis]        = ROOT.TH1F('closure_ss_TH1_basis%d'%basis,        '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.signal_orthogonal_TH1[basis] = ROOT.TH1F('signal_orthogonal_TH1_basis%d'%basis, '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)

        #for o in range(max(basis, self.multijet.basis)+1):
        for b in range(n):
            self.closure_TF1[basis].SetParName  (b, 'c_%d'%b)
            self.closure_TF1[basis].SetParameter(b, 0.0)
        self.closure_TF1[basis].SetParName  (n, 'spurious signal')
        self.closure_TF1[basis].FixParameter(n, 0)

    def getEigenvariations(self, basis, doSpuriousSignal=False, debug=False):
        n = max(self.multijet.basis, basis)+1
        if doSpuriousSignal: n+=1

        if n == 1:
            self.eigenVars[basis] = np.array([self.closure_TF1[basis].GetParError(0)])
            return

        cov = ROOT.TMatrixD(n,n)
        cor = ROOT.TMatrixD(n,n)
        fit_result = self.fit_result[basis] if not doSpuriousSignal else self.fit_result_ss[basis]
        for i in range(n):
            for j in range(n):
                cov[i][j] = fit_result.CovMatrix  (i, j)
                cor[i][j] = fit_result.Correlation(i, j)

        if debug:
            print('Covariance Matrix:')
            cov.Print()
            print('Correlation Matrix:')
            cor.Print()
        
        eigenVal = ROOT.TVectorD(n)
        eigenVec = cov.EigenVectors(eigenVal)
        
        # define relative sign of eigen-basis such that the first coordinate is always positive
        for j in range(n):
            if eigenVec[0][j] >= 0: continue
            for i in range(n):
                eigenVec[i][j] *= -1

        if debug:
            print('Eigenvectors (columns)')
            eigenVec.Print()
            print('Eigenvalues')
            eigenVal.Print()

        eigenVars = np.zeros((n,n), dtype=np.float)
        for i in range(n):
            for j in range(n):
                eigenVars[i,j] = eigenVec[i][j] * eigenVal[j]**0.5
        if not doSpuriousSignal:
            self.eigenVars[basis] = eigenVars
        else:
            self.eigenVars_ss[basis] = eigenVars

        if debug:
            print('Eigenvariations')
            for j in range(n):
                print(j, self.eigenVars[basis][:,j])


    def getParameterDistribution(self, basis):

        n = max(self.multijet.basis, basis) + 1
        self.cUp[basis], self.cDown[basis] = {}, {}
        for i in range(n):
            if i <= self.multijet.basis: # use variance and bias
                cUp = (self.multijet.cUp[self.multijet.basis][i]**2 + self.fit_parameters[basis][i]**2 + self.fit_parameters_error[basis][i]**2)**0.5
            else: # only have bias
                cUp = (self.fit_parameters[basis][i]**2 + self.fit_parameters_error[basis][i]**2)**0.5
            cDown = -cUp
            self.cUp  [basis][i] = cUp
            self.cDown[basis][i] = cDown


    def fit(self, basis):
        n = max(self.multijet.basis, basis)+1
        nConstrained = max(self.multijet.basis-basis, 0)
        nUnconstrained = n - nConstrained
        fit_x_max = self.nBins_rebin+0.5+nConstrained
        self.fit_result[basis] = self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0S','', self.fit_x_min, fit_x_max)
        self.getEigenvariations(basis)
        self.pvalue[basis], self.chi2[basis], self.ndf[basis] = self.closure_TF1[basis].GetProb(), self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        print('Fit closure %s with %d basis elements. x_range = (%f, %f)'%(self.channel, basis, self.fit_x_min, fit_x_max))
        print('chi2/ndf = %3.2f/%3d = %2.2f'%(self.chi2[basis], self.ndf[basis], self.chi2[basis]/self.ndf[basis]))
        print(' p-value = %0.2f'%self.pvalue[basis])
        print('nConstrained',nConstrained)
        print('nUnonstrained',nUnconstrained)
        print('expected ndfs = ',self.nBins_rebin-nUnconstrained)

        self.ymax[basis] = self.closure_TF1[basis].GetMaximum(1,self.nBins_closure)
        #self.ymax[basis] = max(self.closure_TF1[basis].GetMaximum(1,self.nBins_closure), 100*self.signal.GetMaximum())
        self.fit_parameters[basis], self.fit_parameters_error[basis] = [], []
        self.fit_parameters      [basis] = np.array([self.closure_TF1[basis].GetParameter(b) for b in range(n)])
        self.fit_parameters_error[basis] = np.array([self.closure_TF1[basis].GetParError (b) for b in range(n)])
        self.getParameterDistribution(basis)

        for bin in range(1,self.nBins_closure+1):
            self.closure_TH1[basis].SetBinContent(bin, self.closure_TF1[basis].Eval(bin))
            #self.closure_TH1[basis].SetBinError  (bin, self.data_obs_closure.GetBinError(bin))
            self.closure_TH1[basis].SetBinError  (bin, 0.0)
            
        f.cd(self.channel)
        self.closure_TH1[basis].Write()


    def fitSpuriousSignal(self, basis):
        self.doSpuriousSignal = True
        max_basis = max(self.multijet.basis, basis)
        n = max_basis+1
        self.closure_TF1[basis].FixParameter(n, 0)
        self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0', '', self.fit_x_min, self.nBins_rebin+0.5+n)
        self.chi2_ss_zero[basis], self.ndf_ss_zero[basis] = self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        for bin in range(1,self.nBins_closure+1):
            self.closure_ss_zero_TH1[basis].SetBinContent(bin, self.closure_TF1[basis].Eval(bin))
            self.closure_ss_zero_TH1[basis].SetBinError  (bin, 0.0)
        f.cd(self.channel)
        self.closure_ss_zero_TH1[basis].Write()

        self.closure_TF1[basis].SetParameter(n, 0)
        self.closure_TF1[basis].SetParLimits(n, -10, 10)
        self.fit_result_ss[basis] = self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0S', '', self.fit_x_min, self.nBins_rebin+0.5+n)
        self.getEigenvariations(basis, doSpuriousSignal=True)
        self.spuriousSignal[basis]      = self.closure_TF1[basis].GetParameter(n)
        self.spuriousSignalError[basis] = self.closure_TF1[basis].GetParError (n)

        self.pvalue_ss[basis], self.chi2_ss[basis], self.ndf_ss[basis] = self.closure_TF1[basis].GetProb(), self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        print('Fit spurious signal %s with %d basis elements'%(self.channel, basis))
        print('chi2/ndf = %3.2f/%3d = %2.2f'%(self.chi2_ss[basis], self.ndf_ss[basis], self.chi2_ss[basis]/self.ndf_ss[basis]))
        print(' p-value = %0.2f'%self.pvalue_ss[basis])

        print('SS f-test basis',basis)
        self.fProb_ss[basis] = fTest(self.chi2_ss_zero[basis], self.chi2_ss[basis], self.ndf_ss_zero[basis], self.ndf_ss[basis])

        self.fit_parameters_ss      [basis] = np.array([self.closure_TF1[basis].GetParameter(b) for b in range(n+1)])
        self.fit_parameters_error_ss[basis] = np.array([self.closure_TF1[basis].GetParError (b) for b in range(n+1)])

        for bin in range(1,self.nBins_closure+1):
            self.closure_ss_TH1[basis].SetBinContent(bin, self.closure_TF1[basis].Eval(bin))
            self.signal_orthogonal_TH1[basis].SetBinContent(bin, self.multijet.basis_signal[max_basis][bin-1]*self.multijet.average_rebin.GetBinContent(bin) if bin<=self.nBins_rebin else 0.0)
            self.closure_ss_TH1[basis].SetBinError  (bin, 0.0)
            self.signal_orthogonal_TH1[basis].SetBinError(bin, 0.0)
        f.cd(self.channel)
        self.closure_ss_TH1[basis].Write()
        self.signal_orthogonal_TH1[basis].Write()
 
        self.closure_TF1[basis].FixParameter(n, 0)
        self.doSpuriousSignal = False
        print('spurious signal = %2.2f +/- %f'%(self.spuriousSignal[basis], self.spuriousSignalError[basis]))


    def writeClosureResults(self,basis=None):
        if basis is None: basis = self.basis
        max_basis = max(self.multijet.basis, basis)
        nBEs = max_basis+1
        closureResults = 'ZZ4b/nTupleAnalysis/combine/closureResults_%s_%s_basis%d.txt'%(classifier,self.channel,basis)
        #closureResultsRoot = ROOT.TFile(closureResults.replace('.txt', '.root'), 'RECREATE')
        closureResultsFile = open(closureResults, 'w')
        print('Write Closure Results File: \n>> %s'%(closureResults))
        for i in range(nBEs):
            cUp   = self.cUp  [basis][i]
            cDown = self.cDown[basis][i]
            BE_string  = ', '.join('%7.4f'%BE_i for BE_i in self.basis_element[i])
            systUp     = '1+(%9.6f)*np.array([%s])'%(cUp,   BE_string)
            systDown   = '1+(%9.6f)*np.array([%s])'%(cDown, BE_string)
            systUp     = 'multijet_basis%i_%sUp   %s'%(i, channel, systUp)
            systDown   = 'multijet_basis%i_%sDown %s'%(i, channel, systDown)
            print(systUp)
            print(systDown)
            closureResultsFile.write(systUp+'\n')
            closureResultsFile.write(systDown+'\n')

        if self.fProb_ss[basis] >= 0.95:
            ssUp   = self.spuriousSignal[basis]+self.spuriousSignalError[basis]
            ssDown = self.spuriousSignal[basis]-self.spuriousSignalError[basis]
            # if self.spuriousSignalError[basis] < abs(self.spuriousSignal[basis]):
            #     print('WARNING: Spurious Signal for channel %s is inconsistent with zero: (%f, %f)'%(self.channel, ssDown, ssUp), end='')
            #     ssUp   = max([abs(ssUp), abs(ssDown)])
            #     ssDown = -ssUp
            #     print(' -> (%f, %f)'%(ssDown, ssUp))
            SS_string  = ', '.join('%7.4f'%SS_i for SS_i in self.multijet.basis_signal[max_basis]*10)
            systUp     = '1+(%9.6f)*np.array([%s])'%(ssUp/10,   SS_string)
            systDown   = '1+(%9.6f)*np.array([%s])'%(ssDown/10, SS_string)
            systUp     = 'multijet_spurious_signal_%sUp   %s'%(channel, systUp)
            systDown   = 'multijet_spurious_signal_%sDown %s'%(channel, systDown)
            print(systUp)
            print(systDown)
            closureResultsFile.write(systUp+'\n')
            closureResultsFile.write(systDown+'\n')        
        closureResultsFile.close()


    def plotFitResults(self, basis, projection=(0,1), doSpuriousSignal=False):
        max_basis = max(self.multijet.basis, basis)
        n = max_basis+1
        d_ss = n
        if doSpuriousSignal: n+=1
        if n>1:
            dims = tuple(list(projection) + [d for d in range(n) if d not in projection])
        else:
            dims = (0,1)

        labels = ['c$_'+str(d)+'$' for d in dims]
        if doSpuriousSignal: labels[dims.index(n-1)] = r'$\zeta$'

        #plot fit parameters
        x,y,s,c = [],[],[],[]
        parameters = self.fit_parameters[basis] if not doSpuriousSignal else self.fit_parameters_ss[basis]
        x.append( parameters[dims[0]] * (1 if dims[0]>basis else 100) )
        if n==1:
            y.append( 0 )
        if n>1:
            y.append( parameters[dims[1]] * (1 if dims[1]==d_ss else 100) )
        if n>2:
            c.append( parameters[dims[2]] * (1 if dims[2]==d_ss else 100) )
        if n>3:
            s.append( parameters[dims[3]] * (1 if dims[3]==d_ss else 100) )

        x = np.array(x)
        y = np.array(y)
    
        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }

        fig, (ax) = plt.subplots(nrows=1, figsize=(6,6))
        ax.set_aspect(1)
        if not doSpuriousSignal:
            ax.set_title('Multijet Model Bias Fit (%s)'%self.channel.upper())
        else:
            ax.set_title('Multijet Model Spurious Signal Fit (%s)'%self.channel.upper())
        ax.set_xlabel(labels[0]+('' if dims[0]==d_ss else ' (\%)'))
        ax.set_ylabel(labels[1]+('' if dims[1]==d_ss else ' (\%)'))

        xlim, ylim = [-10,10], [-10,10]
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0,0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = np.arange(xlim[0]+2, xlim[1], 2)
        yticks = np.arange(ylim[0]+2, ylim[1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if n>1 and not doSpuriousSignal:
            # draw 1\sigma ellipses
            try:
                width = self.multijet.cUp[self.multijet.basis][dims[0]]-self.multijet.cDown[self.multijet.basis][dims[0]]
            except IndexError:
                width = 0.0 # no variance for this basis element
            try:
                height = self.multijet.cUp[self.multijet.basis][dims[1]]-self.multijet.cDown[self.multijet.basis][dims[1]]
            except IndexError:
                height = 0.0
            ellipse_self_consistency = Ellipse((0,0), 
                                               width =100*width,
                                               height=100*height,
                                               facecolor = 'none',
                                               edgecolor = 'b', #CMURED,
                                               linestyle = '-',
                                               linewidth = 0.75,
                                               zorder=1,
            )
            ax.add_patch(ellipse_self_consistency)

            ellipse_closure = Ellipse((0,0), 
                                      width =100*(self.cUp[basis][dims[0]]-self.cDown[basis][dims[0]]),
                                      height=100*(self.cUp[basis][dims[1]]-self.cDown[basis][dims[1]]),
                                      facecolor = 'none',
                                      edgecolor = 'r', #CMURED,
                                      linestyle = '-',
                                      linewidth = 0.75,
                                      zorder=1,
            )
            ax.add_patch(ellipse_closure)

        bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0)
        if n>2 and not doSpuriousSignal:
            # draw range bars for other priors
            for i, d in enumerate(dims[2:]):
                up, down = self.cUp[basis][d], self.cDown[basis][d]
                thisx = xlim[-1] - 0.5*(n-2) + 0.5*i
                ax.quiver(thisx, 0, 0, 100*up,   color='r', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                ax.quiver(thisx, 0, 0, 100*down, color='r', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

                ax.annotate(labels[i+2], [thisx, 100*down-0.5], ha='center', va='center', bbox=bbox)

            for i, d in enumerate(dims[2:]):
                try:
                    up, down = self.multijet.cUp[self.multijet.basis][d], self.multijet.cDown[self.multijet.basis][d]
                    thisx = xlim[-1] - 0.5*(n-2) + 0.5*i
                    ax.quiver(thisx, 0, 0, 100*up,   color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                    ax.quiver(thisx, 0, 0, 100*down, color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                except IndexError:
                    pass # there is no variance term for this basis

        maxr=np.zeros((2, len(x)), dtype=np.float)
        minr=np.zeros((2, len(x)), dtype=np.float)
        if n>1:
            #generate a ton of random points on a hypersphere in dim=n so surface is dim=n-1.
            points  = np.random.randn(n, min(100**(n-1),10**7)) # random points in a hypercube
            points /= np.linalg.norm(points, axis=0) # normalize them to the hypersphere surface

            # find the point which maximizes the change in c_0**2 + c_1**2
            for i in range(len(x)):
                eigenVars = self.eigenVars[basis] if not doSpuriousSignal else self.eigenVars_ss[basis]
                plane = np.matmul( eigenVars[dims[0:2],:], points )
                r2 = plane[0]**2
                if n>1:
                    r2 += plane[1]**2

                maxr[:,i] = plane[:,r2==r2.max()].T[0]

                #construct orthogonal unit vector to maxr
                minrvec = np.copy(maxr[::-1,i])
                minrvec[0] *= -1
                minrvec /= np.linalg.norm(minrvec)

                #find maxr along minrvec to get minr
                dr2 = np.matmul( minrvec, plane )**2
                #minr[:,i] = plane[:,dr2==dr2.max()].T[0]#this guy is the ~right length but might be slightly off orthogonal
                minr[:,i] = minrvec * dr2.max()**0.5#this guy is the ~right length and is orthogonal by construction
        else:
            for i in range(len(x)):
                maxr[0,i] = self.eigenVars[basis][dims[0]]

        if dims[0]!=d_ss:
            minr[0] *= 100
            maxr[0] *= 100
        if dims[1]!=d_ss:
            minr[1] *= 100
            maxr[1] *= 100

        # print(maxr)
        # print(minr)
        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        
        plt.scatter(x, y, **kwargs)

        if n>2:
            for i in range(len(x)):
                label = '\n'.join(['%s = %2.1f%s'%(labels[dims.index(d)], parameters[d]*(1 if d==d_ss else 100), '' if d==d_ss else '\%') for d in dims[2:]])
                # xy = np.array([x[i],y[i]])
                # xy = [xy+minr[:,i]+maxr[:,i],
                #       xy+minr[:,i]-maxr[:,i],
                #       xy-minr[:,i]+maxr[:,i],
                #       xy-minr[:,i]-maxr[:,i]]
                # xy = max(xy, key=lambda p: p[0]**2+p[1]**2)
                # if xy[0]>0:
                #     horizontalalignment = 'left'
                # else:
                #     horizontalalignment = 'right'
                # if xy[1]>0:
                #     verticalalignment = 'bottom'
                # else:
                #     verticalalignment = 'top'
                xy = [xlim[-1]-3, ylim[-1]-1]
                ax.annotate(label, xy,# label,
                            ha='left', va='top',
                            bbox=bbox)

        projection = '_'.join([str(d) for d in projection])
        if not doSpuriousSignal:
            if type(self.rebin) is list:
                name = 'closureFits/%s/%s/variable_rebin/%s/%s/1_bias_parameters_basis%d_projection_%s.pdf'%(mixName, classifier, region, self.channel, basis, projection)
            else:
                name = 'closureFits/%s/%s/rebin%i/%s/%s/1_bias_parameters_basis%d_projection_%s.pdf'%(mixName, classifier, self.rebin, region, self.channel, basis, projection)            
        else:
            if type(self.rebin) is list:
                name = 'closureFits/%s/%s/variable_rebin/%s/%s/2_spurious_signal_parameters_basis%d_projection_%s.pdf'%(mixName, classifier, region, self.channel, basis, projection)
            else:
                name = 'closureFits/%s/%s/rebin%i/%s/%s/2_spurious_signal_parameters_basis%d_projection_%s.pdf'%(mixName, classifier, self.rebin, region, self.channel, basis, projection)            
        print('fig.savefig( '+name+' )')
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)


    def plotPValues(self):
        fig, (ax) = plt.subplots(nrows=1)
        x = np.array(sorted(self.pvalue.keys()))+1
        ax.set_ylim(0,1)
        xlim = [x[0]-0.5, x[-1]+.5]
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(x)
        #plt.yscale('log')

        y = [self.pvalue[i-1] for i in x]
        ax.set_title('Multijet Model Bias Fit (%s)'%self.channel.upper())
        ax.plot(xlim, [probThreshold,probThreshold], color='b', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot(xlim, [0.95, 0.95],                  color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot(x, y, label='p-value', color='b', linewidth=2)
        if self.basis is not None:
            ax.plot([self.basis+1, self.basis+1], [0,1], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
            ax.scatter(self.basis+1, self.pvalue[self.basis], color='k', marker='*', s=100, zorder=10)
        
        x = np.array(sorted(self.fProb.keys()))+1
        y = [self.fProb[i-1] for i in x]
        marker = '' if len(x)>1 else 'o'
        ax.plot(x, y, label='f-test', color='k', linewidth=2, marker=marker)

        x = np.array(sorted(self.fProb_ss.keys()))+1
        y = [self.fProb_ss[i-1] for i in x]
        marker = '' if len(x)>1 else 'o'
        ax.plot(x, y, label='f-test Spurious Signal', color='k', linestyle='--', linewidth=2, marker=marker)

        ax.set_xlabel('Unconstrained Parameters')
        ax.set_ylabel('Fit p-value')
        ax.legend(loc='upper left', fontsize='small')

        if type(self.rebin) is list:
            name = 'closureFits/%s/%s/variable_rebin/%s/%s/1_bias_pvalues.pdf'%(mixName, classifier, region, self.channel)
        else:
            name = 'closureFits/%s/%s/rebin%i/%s/%s/1_bias_pvalues.pdf'%(mixName, classifier, self.rebin, region, self.channel)
        print('fig.savefig( '+name+' )')
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)



    def plotMix(self, mix):
        samples=collections.OrderedDict()
        samples[closureFileName] = collections.OrderedDict()
        if type(mix)==int:
            samples[closureFileName]['%s/%s/data_obs'%(mixes[mix], self.channel)] = {
                'label' : 'Mixed Data Set %d, %.1f/fb'%(mix, lumi),
                'legend': 1,
                'isData' : True,
                #'drawOptions': 'P ex0',
                'ratio' : 'numer A',
                'color' : 'ROOT.kBlack'}
        else:
            samples[closureFileName]['%s/data_obs'%(self.channel)] = {
                'label' : '#LTMixed Data#GT %.1f/fb'%(lumi),
                'legend': 1,
                'isData' : True,
                #'drawOptions': 'P ex0',
                'ratio' : 'numer A',
                'color' : 'ROOT.kBlack'}
            # samples[closureFileName]['%s/ratio_c0_up'%(self.channel)] = {
            #     'pad': 'rPad',
            #     'drawOptions': 'HIST',
            #     'color' : 'ROOT.kYellow'}
        if type(mix)==int:
            samples[closureFileName]['%s/%s/multijet'%(mixes[mix], self.channel)] = {
                'label' : 'Multijet Model %d'%mix,
                'legend': 2,
                'stack' : 3,
                'ratio' : 'denom A',
                'color' : 'ROOT.kYellow'}
        else:
            samples[closureFileName]['%s/multijet'%self.channel] = {
                'label' : '#LTMultijet#GT',
                'legend': 2,
                'stack' : 3,
                'ratio' : 'denom A',
                'color' : 'ROOT.kYellow'}
        samples[closureFileName]['%s/ttbar'%self.channel] = {
            'label' : '#lower[0.10]{t#bar{t}}',
            'legend': 3,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : 'ROOT.kAzure-9'}
        samples[closureFileName]['%s/signal'%self.channel] = {
            'label' : 'ZZ+ZH+HH(#times100)',
            'legend': 4,
            'weight': 100,
            'color' : 'ROOT.kViolet'}

        xTitle = classifier+' P(Signal) #cbar P('+self.channel.upper()+') is largest'
            
        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'ratioErrors': True,
                      'ratio'     : True,
                      'rMin'      : 0.9,
                      'rMax'      : 1.1,
                      'rebin'     : self.rebin,
                      'rTitle'    : 'Data / Bkgd.',
                      'xTitle'    : xTitle,
                      'yTitle'    : 'Events',
                      'yMax'      : 1.4*(self.ymax[0]),#*ymaxScale, # make room to show fit parameters
                      #'xleg'      : [0.13, 0.13+0.5] if 'SR' in region else ,
                      # 'legendSubText' : ['#bf{Fit:}',
                      #                    '#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2[basis],self.ndf[basis],self.chi2[basis]/self.ndf[basis]),
                      #                    'p-value = %2.0f%%'%(self.pvalue[basis]*100),
                      #                    ],
                      'lstLocation' : 'right',
                      'outputName': 'mix_%s'%(str(mix))}

        if type(self.rebin) is list:
            parameters['outputDir'] = 'closureFits/%s/%s/variable_rebin/%s/%s/'%(mixName, classifier, region, self.channel)
        else:
            parameters['outputDir'] = 'closureFits/%s/%s/rebin%i/%s/%s/'%(mixName, classifier, self.rebin, region, self.channel)

        print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
        PlotTools.plot(samples, parameters, debug=False)


    def plotFit(self, basis, plotSpuriousSignal=False):
        samples=collections.OrderedDict()
        samples[closureFileName] = collections.OrderedDict()
        samples[closureFileName]['%s/data_obs_closure'%self.channel] = {
            'label' : '#LTMixed Data#GT %.1f/fb'%lumi,
            'legend': 1,
            'isData' : True,
            #'ratioDrawOptions': 'P ex0',
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}
        samples[closureFileName]['%s/multijet_closure'%self.channel] = {
            'label' : '#LTMultijet#GT',
            'legend': 2,
            'stack' : 3,
            'ratio' : 'denom A',
            'color' : 'ROOT.kYellow'}
        samples[closureFileName]['%s/ttbar_closure'%self.channel] = {
            'label' : '#lower[0.10]{t#bar{t}}',
            'legend': 3,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : 'ROOT.kAzure-9'}
        if not plotSpuriousSignal:
            samples[closureFileName]['%s/closure_TH1_basis%d'%(self.channel, basis)] = {
                'label' : 'Fit (%d unconstrained parameter%s)'%(basis+1, 's' if basis else ''),
                'legend': 4,
                'ratio': 'denom A', 
                'color' : 'ROOT.kRed'}
        if plotSpuriousSignal:
            samples[closureFileName]['%s/closure_ss_zero_TH1_basis%d'%(self.channel, basis)] = {
                'label' : 'Fit #zeta=0',
                'legend': 5,
                'ratio': 'denom A', 
                'color' : 'ROOT.kGreen+3'}
            samples[closureFileName]['%s/closure_ss_TH1_basis%d'%(self.channel, basis)] = {
                'label' : 'Fit #zeta=%1.1f#pm%1.1f'%(self.spuriousSignal[basis], self.spuriousSignalError[basis]),
                'legend': 6,
                'ratio': 'denom A', 
                'color' : 'ROOT.kViolet'}
            samples[closureFileName]['%s/signal_closure'%self.channel] = {
                'label' : 'ZZ+ZH+HH(#times100)',
                'legend': 7,
                'weight': 100,
                'color' : 'ROOT.kViolet+7'}
            # samples[closureFileName]['%s/signal_orthogonal_TH1_basis%d'%(self.channel, basis)] = {
            #     'label' : 'Orthogonalized Signal(#times100)',
            #     'legend': 8,
            #     'weight': 100, 
            #     'color' : 'ROOT.kViolet-6'}

        xTitle = classifier+' P(Signal) Bin #cbar P('+self.channel.upper()+') is largest'
            
        ymaxScale = 1.4 #+ max(0, (basis-2)/4.0)
        if doSpuriousSignal:
            ymaxScale = 1.7 #+ max(0, (basis-2)/4.0)

        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.fit_x_min,        0,self.fit_x_min,      self.ymax[0]/2],
                                       [self.nBins_rebin+0.5,  0,self.nBins_rebin+0.5,self.ymax[0]/2]],
                      'ratioErrors': False,
                      'ratio'     : 'significance',#True,
                      'rMin'      : -5,#0.9,
                      'rMax'      : 5,#1.1,
                      'rTitle'    : 'Pulls',#'Data / Bkgd.',
                      # 'ratioErrors': True,
                      # 'ratio'      : True,
                      # 'rMin'       : 0.9,
                      # 'rMax'       : 1.1,
                      # 'rTitle'     : 'Model / Average',
                      'xTitle'    : xTitle,
                      'yTitle'    : 'Events',
                      'yMax'      : self.ymax[0]*ymaxScale,#*ymaxScale, # make room to show fit parameters
                      'xleg'      : [0.13, 0.13+0.42],
                      'lstLocation' : 'right',
                      'outputName': '%s_basis%d'%('2_spurious_signal' if plotSpuriousSignal else '1_bias', basis)}

        n = max(self.multijet.basis, basis)+1
        if plotSpuriousSignal:
            parameters['legendSubText'] = ['#bf{Spurious Signal Fit:}',
                                           '#chi^{2}/DoF = %2.1f/%d = %1.2f (#zeta=0)'%(self.chi2_ss_zero[basis],self.ndf_ss_zero[basis],self.chi2_ss_zero[basis]/self.ndf_ss_zero[basis]),
                                           '#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2_ss[basis],self.ndf_ss[basis],self.chi2_ss[basis]/self.ndf_ss[basis]),
                                           'p-value = %2.0f%% (f-test = %2.0f%%)'%(self.pvalue_ss[basis]*100, self.fProb_ss[basis]*100)]
            for i in range(n):
                parameters['legendSubText'] += ['#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma'%(i, self.fit_parameters_ss[basis][i]*100, abs(self.fit_parameters_ss[basis][i])/self.fit_parameters_error_ss[basis][i])]
        else:
            parameters['legendSubText'] = ['#bf{Fit:}',
                                         '#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2[basis],self.ndf[basis],self.chi2[basis]/self.ndf[basis]),
                                         'p-value = %2.0f%%'%(self.pvalue[basis]*100)]
            for i in range(n):
                parameters['legendSubText'] += ['#color[%d]{#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma}'%(4 if i>basis else 2, i, self.fit_parameters[basis][i]*100, abs(self.fit_parameters[basis][i])/self.fit_parameters_error[basis][i])]

        parameters['ratioLines'] = [[self.fit_x_min,       parameters['rMin'], self.fit_x_min,       parameters['rMax']],
                                    [self.nBins_rebin+0.5, parameters['rMin'], self.nBins_rebin+0.5, parameters['rMax']]]
        #parameters['xMax'] = self.nBins_rebin+self.multijet.basis+1.5 if not plotSpuriousSignal else self.nBins_rebin+basis+1.5
        if plotSpuriousSignal:
            parameters['xMax'] = self.nBins_rebin+0.5+max(self.multijet.basis, basis)+1
        else:
            parameters['xMax'] = self.nBins_rebin+0.5+max(self.multijet.basis-basis, 0)

        if type(self.rebin) is list:
            parameters['outputDir'] = 'closureFits/%s/%s/variable_rebin/%s/%s/'%(mixName, classifier, region, self.channel)
        else:
            parameters['outputDir'] = 'closureFits/%s/%s/rebin%i/%s/%s/'%(mixName, classifier, self.rebin, region, self.channel)

        print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
        PlotTools.plot(samples, parameters, debug=False)

    def print_exit_message(self):
        for line in self.exit_message: print(line)





# make multijet ensembles and perform fits
multijetEnsembles = {}
for channel in channels:
    if type(rebin[channel]) is list:
        mkpath('%s/closureFits/%s/%s/variable_rebin/%s/%s'%(basePath, mixName, classifier, region, channel))
    else:
        mkpath('%s/closureFits/%s/%s/rebin%i/%s/%s'%(basePath, mixName, classifier, rebin[channel], region, channel))
    multijetEnsembles[channel] = multijetEnsemble(channel)

# run closure fits using average multijet model 
closures = {}
for channel in channels:
    closures[channel] = closure(channel, multijetEnsembles[channel])

# close input file and make plots 
f.Close()
for channel in channels:
    for basis in multijetEnsembles[channel].bases:
        multijetEnsembles[channel].plotFit(basis)
    for basis in closures[channel].bases:
        closures[channel].plotFit(basis)
        closures[channel].plotFit(basis, plotSpuriousSignal=True)
    for m in range(nMixes):
        closures[channel].plotMix(m)
    closures[channel].plotMix('ave')
for channel in channels:
    multijetEnsembles[channel].print_exit_message()
    closures[channel].print_exit_message()
    
