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

year = "RunII"
lumi = 132.6
rebin = 10
#rebin = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
maxOrderEnsemble = 4
maxOrderClosure  = 4
channels = ["zz",
            "zh",
            ]

lps = [                                                "1",
                                                   "2*x-1",
                                            "6*x^2 -6*x+1",
                                   "20*x^3 -30*x^2+12*x-1",
                          "70*x^4 -140*x^3 +90*x^2-20*x+1",
                "252*x^5 -630*x^4 +560*x^3-210*x^2+30*x-1",
       "924*x^6-2772*x^5+3150*x^4-1680*x^3+420*x^2-42*x+1"
       ]
lp = []
for i, s in enumerate(lps): 
    lp.append( ROOT.TF1("lp%i"%i, s, 0, 1) )

USER = getUSER()
CMSSW = getCMSSW()
basePath = '/uscms/home/%s/nobackup/%s/src'%(USER, CMSSW)

#mixName = "3bMix4b_rWbW2"
mixName = "3bDvTMix4bDvT"
ttAverage = False

doSpuriousSignal = True
dataAverage = True
nMixes = 10
# region = "SR"
region = 'SRNoHH'
#region = 'notSR'
#hists_closure_MixedToUnmixed_3bMix4b_rWbW2_b0p60p3_SRNoHH_e25_os012.root
#hists_closure_MixedToUnmixed_3bMix4b_rWbW2_b0p60p3_SRNoHH.root
#closureFileName = "ZZ4b/nTupleAnalysis/combine/hists_closure_MixedToUnmixed_"+mixName+"_b0p60p3_"+region+".root"
#closureFileName = "ZZ4b/nTupleAnalysis/combine/hists_closure_"+mixName+"_b0p60p3_"+region+".root"
closureFileName = "ZZ4b/nTupleAnalysis/combine/hists_closure_"+mixName+"_"+region+"_weights_nf8.root"
print(closureFileName)
f=ROOT.TFile(closureFileName, "UPDATE")
mixes = [mixName+"_v%d"%i for i in range(nMixes)]

probThreshold = 0.05 #0.045500263896 #0.682689492137 # 1sigma 

regionName = {"SB": "Sideband",
              "CR": "Control Region",
              "SR": "Signal Region",
              "notSR": "Sideband",
              "SRNoHH": "Signal Region (Veto HH)",
          }

        
def addYears(directory, processes=["ttbar","multijet","data_obs"]):
    hists = []
    for process in processes:        
        try:
            f.Get('%s/%s'%(directory,process)).IsZombie()
            # already exists, don't need to make it
        except ReferenceError:
            # make sum of years
            hists.append( f.Get(directory+"2016/"+process) )
            hists[-1].Add(f.Get(directory+"2017/"+process))
            hists[-1].Add(f.Get(directory+"2018/"+process))
            try:
                f.Get(directory).IsZombie()
            except ReferenceError:
                f.mkdir(directory)
            f.cd(directory)
            hists[-1].Write()

def addMixes(directory):
    hists = []
    for process in ["ttbar","multijet","data_obs"]:
        try:
            f.Get('%s/%s'%(directory,process)).IsZombie()
        except ReferenceError:
            hists.append( f.Get(mixes[0]+"/"+directory+"/"+process) )

            if ttAverage and process=='ttbar': # skip averaging if ttAverage and process=='ttbar'
                pass
            else:
                for mix in mixes[1:]:
                    hists[-1].Add( f.Get(mix+"/"+directory+"/"+process) )
                hists[-1].Scale(1.0/nMixes)

            if process=='multijet':
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

# Get Signal templates for spurious signal fits
zzFile = ROOT.TFile('/uscms/home/%s/nobackup/ZZ4b/ZZ4bRunII/hists.root'%(USER), 'READ')
zhFile = ROOT.TFile('/uscms/home/%s/nobackup/ZZ4b/bothZH4bRunII/hists.root'%(USER), 'READ')
for ch in channels:
    var = 'SvB_ps_%s'%ch
    histPath = 'passMDRs/fourTag/mainView/%s/%s'%(region,var)
    signal =   zzFile.Get(histPath)
    signal.Add(zhFile.Get(histPath))
    signal.SetName('signal')
    f.cd(ch)
    signal.Write()
zzFile.Close()
zhFile.Close()

f.Close()
f=ROOT.TFile(closureFileName, "UPDATE")


def pearsonr(x,y,n=None):
    r, p_raw = scipy.stats.pearsonr(x,y)
    if n is None: 
        return (r, p_raw)
    # if n <= 2: # pearson r cdf is not well defined for n<=2
    #     return (r, 1.)
    # corrected p-value using different number of degrees of freedom than just the number of samples (array length)
    dist = scipy.stats.beta(n/2. - 1, n/2. - 1, loc=-1, scale=2)
    p_cor = 2*dist.cdf(-abs(r))
    return (r, p_cor)

def fTest(chi2_1, chi2_2, ndf_1, ndf_2):
    d1 = (ndf_1-ndf_2)
    d2 = ndf_2
    N = (chi2_1-chi2_2)/d1
    D = chi2_2/d2
    fStat = N/D
    fProb = scipy.stats.f.cdf(fStat, d1, d2)
    expectedFStat = scipy.stats.distributions.f.isf(0.05, d1, d2)
    print("d1, d2 = %d, %d"%(d1, d2))
    print("N, D = %f, %f"%(N, D))
    print("    f(%i,%i) = %f (expected at 95%%: %f)"%(d1,d2,fStat,expectedFStat))
    print("f.cdf(%i,%i) = %3.0f%%"%(d1,d2,100*fProb))
    print()
    return fProb




class multijetEnsemble:
    def __init__(self, channel):
        self.channel = channel
        self.data_minus_ttbar = f.Get('%s/ttbar'%self.channel)
        self.data_minus_ttbar.SetName('%s_average_%s'%('data_minus_ttbar', self.channel))
        self.data_minus_ttbar.Scale(-1)
        self.data_minus_ttbar.Add(f.Get('%s/data_obs'%self.channel))
        self.data_minus_ttbar.Rebin(rebin)
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

        f.cd(self.channel)

        self.average_rebin = self.average.Clone()
        self.average_rebin.SetName('%s_rebin'%self.average.GetName())
        self.average_rebin.Rebin(rebin)

        self.models_rebin = [model.Clone() for model in self.models]
        for model in self.models_rebin: model.SetName('%s_rebin'%model.GetName())
        for model in self.models_rebin: model.Rebin(rebin)
        self.nBins_rebin = self.average_rebin.GetSize()-2

        if self.allMixFvT is not None:
            self.allMixFvT_rebin = self.allMixFvT.Clone()
            self.allMixFvT_rebin.SetName('%s_rebin'%self.allMixFvT.GetName())
            self.allMixFvT_rebin.Rebin(rebin)
        else:
            self.allMixFvT_rebin = None

        self.lp_integral = np.array([[l.Integral(self.average_rebin.GetBinLowEdge(bin), self.average_rebin.GetXaxis().GetBinUpEdge(bin))/self.average_rebin.GetBinWidth(bin) for bin in range(1,self.nBins_rebin+1)] for l in lp])

        f.cd(self.channel)
        self.nBins_ensemble = self.nBins_rebin * nMixes
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
                error = (self.models_rebin[m].GetBinError(local_bin)**2 + (self.average_rebin.GetBinError(local_bin)/nMixes)**2)**0.5
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

        self.fit_result = {}
        self.eigenVars = {}
        self.multijet_TF1, self.multijet_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.pulls = {}
        self.pearsonr = {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.cUp, self.cDown = {}, {}
        self.fProb = {}
        self.order = None
        for order in range(maxOrderEnsemble+1): 
            self.makeFitFunction(order)
            self.fit(order)
            self.plotFitResults(order)
            self.plotPulls(order)
            try:
                self.fProb[order] = fTest(self.chi2[order-1], self.chi2[order], self.ndf[order-1], self.ndf[order])
            except KeyError:
                pass
                
            #if self.order is None and self.pvalue[order] > probThreshold:
            if self.order is None and self.pearsonr[order]['total'][1] > probThreshold:
                self.order = order # store first order to satisfy min threshold. Will be used in closure fits
        #self.plotPValues()
        self.plotPearson()


    def makeFitFunction(self, order):

        def background_UserFunction(xArray, pars):
            ensemble_bin = int(xArray[0])
            m = (ensemble_bin-1)//self.nBins_rebin
            local_bin = 1 + ((ensemble_bin-1)%self.nBins_rebin)
            model = self.models[m]

            l, u = self.average_rebin.GetBinLowEdge(local_bin), self.average_rebin.GetXaxis().GetBinUpEdge(local_bin)

            p = 1.0                
            for lp_idx in range(order+1):
                par_idx = m*(order+1)+lp_idx
                p += pars[par_idx] * self.lp_integral[lp_idx][local_bin-1]

            return p*self.multijet_ensemble.GetBinContent(ensemble_bin)

        f.cd(self.channel)
        self.multijet_TF1[order] = ROOT.TF1 ('multijet_ensemble_TF1_order%d'%order, background_UserFunction, 0.5, 0.5+self.nBins_ensemble, nMixes*(order+1))
        self.multijet_TH1[order] = ROOT.TH1F('multijet_ensemble_TH1_order%d'%order, '', self.nBins_ensemble, 0.5, 0.5+self.nBins_ensemble)

        for m in range(nMixes):
            for o in range(order+1):
                self.multijet_TF1[order].SetParName  (m*(order+1)+o, 'v%d c_%d'%(m, o))
                self.multijet_TF1[order].SetParameter(m*(order+1)+o, 0.0)



    def getEigenvariations(self, order=None, debug=False):
        if order is None: order = self.order
        n = order+1

        if n == 1:
            self.eigenVars[order] = [np.array([[self.multijet_TF1[order].GetParError(m*n)]]) for m in range(nMixes)]
            return

        cov = [ROOT.TMatrixD(n,n) for m in range(nMixes)]
        cor = [ROOT.TMatrixD(n,n) for m in range(nMixes)]

        for m in range(nMixes):
            for i in range(n):
                for j in range(n): # full fit is block diagonal in nMixes blocks since there is no correlation between fit parameters of different multijet models
                    cov[m][i][j] = self.fit_result[order].CovMatrix  (m*n+i, m*n+j)
                    cor[m][i][j] = self.fit_result[order].Correlation(m*n+i, m*n+j)

        if debug:
            for m in range(nMixes):
                print("Covariance Matrix:",m)
                cov[m].Print()
                print("Correlation Matrix:",m)
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
                print("Eigenvectors (columns)",m)
                eigenVec[m].Print()
                print("Eigenvalues",m)
                eigenVal[m].Print()

        self.eigenVars[order] = [np.zeros((n,n), dtype=np.float) for m in range(nMixes)]
        for m in range(nMixes):
            for i in range(n):
                for j in range(n):
                    self.eigenVars[order][m][i,j] = eigenVec[m][i][j] * eigenVal[m][j]**0.5

        if debug:
            for m in range(nMixes):
                print("Eigenvariations",m)
                for j in range(n):
                    print(j, self.eigenVars[order][m][:,j])


    def getParameterDistribution(self, order):
        n = order+1
        parMean  = np.array([0 for i in range(n)], dtype=np.float)
        parMean2 = np.array([0 for i in range(n)], dtype=np.float)
        for m in range(nMixes):
            parMean  += self.fit_parameters[order][m]    / nMixes
            parMean2 += self.fit_parameters[order][m]**2 / nMixes
        var = parMean2 - parMean**2
        parStd = var**0.5
        print("Parameter Mean:",parMean)
        print("Parameter  Std:",parStd)

        for i in range(n):
            cUp   =  (abs(parMean[i])+parStd[i]) * n**0.5 # add scaling term so that 1 sigma corresponds to quadrature sum over i of (abs(parMean[i])+parStd[i])
            cDown = -cUp
            try:
                self.cUp  [order].append( cUp )
                self.cDown[order].append( cDown )
            except KeyError:
                self.cUp  [order] = [cUp  ]
                self.cDown[order] = [cDown]


    def fit(self, order):
        self.fit_result[order] = self.multijet_ensemble_average.Fit(self.multijet_TF1[order], 'N0S')
        self.getEigenvariations(order)
        self.pvalue[order], self.chi2[order], self.ndf[order] = self.multijet_TF1[order].GetProb(), self.multijet_TF1[order].GetChisquare(), self.multijet_TF1[order].GetNDF()
        print('Fit multijet ensemble %s at order %d'%(self.channel, order))
        print('chi2/ndf = %3.2f/%3d = %2.2f'%(self.chi2[order], self.ndf[order], self.chi2[order]/self.ndf[order]))
        print(' p-value = %0.2f'%self.pvalue[order])

        self.ymax[order] = self.multijet_TF1[order].GetMaximum(1,self.nBins_ensemble)
        self.fit_parameters[order], self.fit_parameters_error[order] = [], []
        n = order+1
        for m in range(nMixes):
            self.fit_parameters      [order].append( np.array([self.multijet_TF1[order].GetParameter(m*n+o) for o in range(order+1)]) )
            self.fit_parameters_error[order].append( np.array([self.multijet_TF1[order].GetParError (m*n+o) for o in range(order+1)]) )
        self.getParameterDistribution(order)

        for bin in range(1,self.nBins_ensemble+1):
            self.multijet_TH1[order].SetBinContent(bin, self.multijet_TF1[order].Eval(bin))
            #self.multijet_TH1[order].SetBinError  (bin, self.multijet_ensemble.GetBinError(bin))
            self.multijet_TH1[order].SetBinError  (bin, 0.0)

        self.pulls[order] = np.array([(self.multijet_TF1[order].Eval(bin) - self.multijet_ensemble_average.GetBinContent(bin))/self.multijet_ensemble_average.GetBinError(bin) for bin in range(1,self.nBins_ensemble+1)])

        # check bin to bin correlations using pearson R test
        xs = np.array([self.pulls[order][m*self.nBins_rebin  : (m+1)*self.nBins_rebin-1] for m in range(nMixes)])
        ys = np.array([self.pulls[order][m*self.nBins_rebin+1: (m+1)*self.nBins_rebin  ] for m in range(nMixes)])
        # xs = np.array([self.pulls[order][m*self.nBins_rebin  :(m+1)*self.nBins_rebin-1:2] for m in range(nMixes)])
        # ys = np.array([self.pulls[order][m*self.nBins_rebin+1:(m+1)*self.nBins_rebin  :2] for m in range(nMixes)])
        x, y = xs.flatten(), ys.flatten()
        r, p = pearsonr(x,y)#, n=nMixes*(self.nBins_rebin/2 - order-1))
        self.pearsonr[order] = {'total': (r, p),
                                #'mixes': [pearsonr(xs[m],ys[m], n=self.nBins_rebin-1 - order-1) for m in range(nMixes)]}
                                'mixes': [pearsonr(xs[m],ys[m]) for m in range(nMixes)]}
        print(' R, prob = %0.2f, %0.2f'%self.pearsonr[order]['total'])
        # n = x.shape[0] - nMixes*(order+1)
        # dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
        # p_manual = 2*dist.cdf(-abs(r))
        # print('manual R p-value: n, p = %d, %f'%(n,p_manual))
        # raw_input()
        f.cd(self.channel)
        self.multijet_TH1[order].Write()
        

    def plotPValues(self):
        fig, (ax) = plt.subplots(nrows=1)
        ax.set_ylim(0.001,1)
        ax.set_xticks(range(maxOrderEnsemble+1))
        plt.yscale('log')

        x = sorted(self.pvalue.keys())
        y = [self.pvalue[o] for o in x]
        ax.set_title('Multijet Self-Consistency Fit (%s)'%self.channel.upper())
        ax.plot(x, y, label='Multijet Model Self-Consistency', color='b', linewidth=2)
        
        ax.plot([0,maxOrderEnsemble], [probThreshold,probThreshold], color='k', alpha=0.5, linestyle='--', linewidth=1)

        ax.set_xlabel('Polynomial Order')
        ax.set_ylabel('Fit p-value')
        #ax.legend(loc='best', fontsize='small')

        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/pvalues_multijet_self_consistency.pdf'%(mixName, region, self.channel)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/pvalues_multijet_self_consistency.pdf'%(mixName, rebin, region, self.channel)
        print("fig.savefig( "+name+" )")
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)


    def plotPearson(self):
        fig, (ax) = plt.subplots(nrows=1)
        #ax.set_ylim(0.001,1)
        #plt.yscale('log')
        ax.set_ylim(0,1)
        ax.set_xticks(range(maxOrderEnsemble+1))

        x = sorted(self.pearsonr.keys())
        r = [self.pearsonr[o]['total'][0] for o in x]
        p = [self.pearsonr[o]['total'][1] for o in x]
        ax.set_title('Multijet Self-Consistency Fit (%s)'%self.channel.upper())
        ax.plot(x, r, label='R-value', color='black', linewidth=2, linestyle='--')
        ax.plot(x, p, label='p-value', color='black', linewidth=2)

        colors=['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:cherry', 'xkcd:bright red',
                'xkcd:pine', 'xkcd:magenta', 'xkcd:cerulean', 'xkcd:eggplant', 'xkcd:coral', 'xkcd:blue purple']
        for m in range(nMixes):
            r = [self.pearsonr[o]['mixes'][m][0] for o in x]
            p = [self.pearsonr[o]['mixes'][m][1] for o in x]
            label = 'v$_%d$'%m
            ax.plot(x, r, color=colors[m], linewidth=1, alpha=0.5, linestyle='--', label='_'+label)#underscore tells pyplot to not show this in the legend
            ax.plot(x, p, color=colors[m], linewidth=1, alpha=0.5, label=label)
        
        ax.plot([0,maxOrderEnsemble], [probThreshold,probThreshold], color='k', alpha=0.5, linestyle='--', linewidth=1)

        ax.set_xlabel('Polynomial Order')
        ax.set_ylabel('Adjacent Bin Pearson R-test')
        plt.legend(fontsize='small', loc='best')

        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/stage0_pearsonr_multijet_self_consistency.pdf'%(mixName, region, self.channel)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/stage0_pearsonr_multijet_self_consistency.pdf'%(mixName, rebin, region, self.channel)
        print("fig.savefig( "+name+" )")
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)


    def plotFitResults(self, order):
        n = order+1

        #plot fit parameters
        x,y,s,c = [],[],[],[]
        for m in range(nMixes):
            x.append( 100*self.fit_parameters[order][m][0] )
            if n==1:
                y.append( 0 )
            if n>1:
                y.append( 100*self.fit_parameters[order][m][1] )
            if n>2:
                c.append( 100*self.fit_parameters[order][m][2] )
            if n>3:
                s.append( 100*self.fit_parameters[order][m][3] )

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
        ax.set_title('Multijet Self-Consistency Fit Parameters (%s)'%self.channel.upper())
        ax.set_xlabel('c$_0$ (\%)')
        ax.set_ylabel('c$_1$ (\%)')

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
                              width =100*(self.cUp[order][0]-self.cDown[order][0]),
                              height=100*(self.cUp[order][1]-self.cDown[order][1]),
                              facecolor = 'none',
                              edgecolor = 'b', # CMURED,
                              linestyle = '-',
                              linewidth = 0.5,
                              zorder=1,
            )
            #transf = transforms.Affine2D().scale(self.cUp[order][0]-self.cDown[order][0], self.cUp[order][1]-self.cDown[order][1])
            #ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse)

        maxr=np.zeros((2, len(x)), dtype=np.float)
        minr=np.zeros((2, len(x)), dtype=np.float)
        if n>1:
            #generate a ton of random points on a hypersphere in dim=n so surface is dim=n-1.
            points  = np.random.uniform(0,1,  (n, min(100**(n-1),10**7) )  ) # random points in a hypercube
            points /= np.linalg.norm(points, axis=0) # normalize them to the hypersphere surface

            # for each model, find the point which maximizes the change in c_0**2 + c_1**2
            for m in range(nMixes):
                plane = np.matmul( self.eigenVars[order][m][0:min(n,2),:], points )
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
                maxr[0,m] = self.eigenVars[order][m][0]
        
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
            x_offset, y_offset = maxr[0,m]+minr[0,m], maxr[1,m]+minr[1,m]
            bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0, pad=0)
            ax.annotate('v$_%d$'%m, (x[m]+x_offset, y[m]+y_offset), bbox=bbox)

        if n>2:
            plt.colorbar(label='c$_2$ (\%)')#, cax=cax) 
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
            labels = ["%0.2f"%smin,
                      "%0.2f"%(smin+srange*1.0/3),
                      "%0.2f"%(smin+srange*2.0/3),
                      "%0.2f"%smax]

            leg = plt.legend(handles, labels, 
                             ncol=1, 
                             fontsize='medium',
                             loc='best',
                             title='c$_3$ (\%)', 
                             scatterpoints = 1)
        
        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/stage0_parameters_order%d_multijet_self_consistency.pdf'%(mixName, region, self.channel, order)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/stage0_parameters_order%d_multijet_self_consistency.pdf'%(mixName, rebin, region, self.channel, order)            
        print("fig.savefig( "+name+" )")
        fig.savefig( name )
        plt.close(fig)


    def plotPulls(self, order):
        n = order+1

        xs = np.array([self.pulls[order][m*self.nBins_rebin  :(m+1)*self.nBins_rebin-1:2] for m in range(nMixes)])
        ys = np.array([self.pulls[order][m*self.nBins_rebin+1:(m+1)*self.nBins_rebin  :2] for m in range(nMixes)])
        # xs = np.array([self.pulls[order][m*self.nBins_rebin  : (m+1)*self.nBins_rebin-1] for m in range(nMixes)])
        # ys = np.array([self.pulls[order][m*self.nBins_rebin+1: (m+1)*self.nBins_rebin  ] for m in range(nMixes)])

        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }

        fig, (ax) = plt.subplots(nrows=1, figsize=(6,6))
        ax.set_aspect(1)
        ax.set_title('Adjacent Bin Pulls (%s, order %d)'%(self.channel.upper(), order))
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

        colors=['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:cherry', 'xkcd:bright red',
                'xkcd:pine', 'xkcd:magenta', 'xkcd:cerulean', 'xkcd:eggplant', 'xkcd:coral', 'xkcd:blue purple']
        for m in range(nMixes):
            #r, p = scipy.stats.pearsonr(xs[m], ys[m])
            (r, p) = self.pearsonr[order]['mixes'][m]
            kwargs['label'] = 'v$_%d$, R=%0.2f (%2.0f%s)'%(m, r, p*100, '\%')
            kwargs['c'] = colors[m]
            plt.scatter(xs[m], ys[m], **kwargs)
        plt.tight_layout()

        #x, y = xs.flatten(), ys.flatten()
        #r, p = scipy.stats.pearsonr(x,y)
        (r, p) = self.pearsonr[order]['total']
    
        plt.legend(fontsize='small', loc='upper left', ncol=2, title='Overall R=%0.2f (%2.0f%s)'%(r,p*100,'\%'))
        
        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/stage0_pull_correlation_order%d_multijet_self_consistency.pdf'%(mixName, region, self.channel, order)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/stage0_pull_correlation_order%d_multijet_self_consistency.pdf'%(mixName, rebin, region, self.channel, order)            
        print("fig.savefig( "+name+" )")
        fig.savefig( name )
        plt.close(fig)


    def plotFit(self, order):
        samples=collections.OrderedDict()
        samples[closureFileName] = collections.OrderedDict()
        samples[closureFileName]['%s/data_minus_ttbar_ensemble'%self.channel] = {
            'label' : '#LTMixed Data#GT - #lower[0.10]{t#bar{t}}',
            'legend': 1,
            'ratioDrawOptions' : 'P ex0',
            'isData' : True,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}
        samples[closureFileName]['%s/multijet_ensemble_average'%self.channel] = {
            'label' : '#LTMultijet Model#GT',
            'legend': 2,
            #'isData' : True,
            'ratio' : 'denom A',
            'color' : 'ROOT.kBlack'}
        samples[closureFileName]['%s/multijet_ensemble'%self.channel] = {
            'label' : 'Multijet Models',
            'legend': 3,
            'stack' : 1,
            'ratio' : 'numer A',
            'color' : 'ROOT.kYellow'}
        samples[closureFileName]['%s/multijet_ensemble_TH1_order%d'%(self.channel, order)] = {
            'label' : 'Fit (order %d)'%order,
            'legend': 4,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlue'}
        samples[closureFileName]['%s/allMixFvT_ensemble'%self.channel] = {
            'label' : 'FvT Fit to all Mixes',
            'legend': 5,
            'ratio' : 'numer A',
            'color' : 'ROOT.kGray+2'}

        if 'zz' in self.channel:
            xTitle = 'P_{ZZ or ZH} #%s + #%s#%s #cbar P%s > P%s'%(PlotTools.subscript('Bin'), PlotTools.subscript('Mix'), PlotTools.subscript('Bins'), PlotTools.subscript('ZZ'), PlotTools.subscript('ZH'))
        if 'zh' in self.channel:
            xTitle = 'P_{ZZ or ZH} #%s + #%s#%s #cbar P%s #geq P%s'%(PlotTools.subscript('Bin'), PlotTools.subscript('Mix'), PlotTools.subscript('Bins'), PlotTools.subscript('ZH'), PlotTools.subscript('ZZ'))
            #xTitle = 'P_{ZZ or ZH} #_{Bin} + #_{Mix}#_{Bins} (P_{ZH} #geq P_{ZZ})'
            
        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.nBins_rebin*m+0.5,  0,self.nBins_rebin*m+0.5,self.ymax[0]*1.1] for m in range(1,nMixes+1)],
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
                      'yMax'      : self.ymax[0]*1.7,#*ymaxScale, # make room to show fit parameters
                      'xleg'      : [0.13, 0.13+0.4],
                      'legendSubText' : ['#bf{Fit:}',
                                         '#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2[order],self.ndf[order],self.chi2[order]/self.ndf[order]),
                                         'p-value = %2.0f%%'%(self.pvalue[order]*100),
                                         '#bf{Adjacent Bin Pull Correlation:}',
                                         'R, p-value = %1.2f, %2.0f%%'%(self.pearsonr[order]['total'][0], self.pearsonr[order]['total'][1]*100),
                                         ],
                      'lstLocation' : 'right',
                      'rPadFraction': 0.5,
                      'outputName': 'stage0_multijet_ensemble_order%d'%(order)}

        parameters['ratioLines'] = [[self.nBins_rebin*m+0.5, parameters['rMin'], self.nBins_rebin*m+0.5, parameters['rMax']] for m in range(1,nMixes+1)]

        if type(rebin) is list:
            parameters['outputDir'] = 'closureFits/%s/variable_rebin/%s/%s/'%(mixName, region, self.channel)
        else:
            parameters['outputDir'] = 'closureFits/%s/rebin%i/%s/%s/'%(mixName, rebin, region, self.channel)

        print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
        PlotTools.plot(samples, parameters, debug=False)




class closure:
    def __init__(self, channel, multijet):
        self.channel = channel
        self.multijet = multijet
        self.ttbar = f.Get('%s/ttbar'%self.channel)
        self.ttbar.SetName('%s_average_%s'%(self.ttbar.GetName(), self.channel))
        self.data_obs = f.Get('%s/data_obs'%self.channel)
        self.data_obs.SetName('%s_average_%s'%(self.data_obs.GetName(), self.channel))
        self.nBins = self.data_obs.GetSize()-2 # GetSize includes under/overflow bins

        self.doSpuriousSignal = False
        self.spuriousSignal = {}
        self.spuriousSignalError = {}
        self.closure_spuriousSignal_TH1 = {}
        self.signal = f.Get('%s/signal'%self.channel)
        self.signal.Rebin(rebin)

        f.cd(self.channel)

        self.ttbar_rebin = self.ttbar.Clone()
        self.ttbar_rebin.SetName('%s_rebin'%self.ttbar.GetName())
        self.ttbar_rebin.Rebin(rebin)
        self.data_obs_rebin = self.data_obs.Clone()
        self.data_obs_rebin.SetName('%s_rebin'%self.data_obs.GetName())
        self.data_obs_rebin.Rebin(rebin)
        self.nBins_rebin = self.data_obs_rebin.GetSize()-2

        self.lp_integral = np.array([[l.Integral(self.data_obs_rebin.GetBinLowEdge(bin), self.data_obs_rebin.GetXaxis().GetBinUpEdge(bin))/self.data_obs_rebin.GetBinWidth(bin) for bin in range(1,self.nBins_rebin+1)] for l in lp])

        f.cd(self.channel)
        self.nBins_closure = self.nBins_rebin + maxOrderClosure + 1 # add bins for multijet self-consistency function constraints and 
        self.multijet_closure = ROOT.TH1F('multijet_closure', '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.ttbar_closure    = ROOT.TH1F('ttbar_closure',    '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.data_obs_closure = ROOT.TH1F('data_obs_closure', '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.signal_closure   = ROOT.TH1F('signal_closure',   '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)

        for bin in range(1, self.nBins_rebin+1):
            self.multijet_closure.SetBinContent(bin, self.multijet.average_rebin.GetBinContent(bin))
            self.ttbar_closure   .SetBinContent(bin, self.ttbar_rebin           .GetBinContent(bin))
            self.signal_closure  .SetBinContent(bin, self.signal                .GetBinContent(bin))
            self.data_obs_closure.SetBinContent(bin, self.data_obs_rebin        .GetBinContent(bin))

            self.multijet_closure.SetBinError  (bin, 0.0)
            self.ttbar_closure   .SetBinError  (bin, 0.0)
            # self.signal_closure  .SetBinError  (bin, 0.0)
            # self.multijet_closure.SetBinError  (bin, self.multijet.average_rebin.GetBinError(bin))
            # self.ttbar_closure   .SetBinError  (bin, self.ttbar_rebin           .GetBinError(bin))
            self.signal_closure  .SetBinError  (bin, self.signal                .GetBinError(bin))
            error = (self.data_obs_rebin.GetBinError(bin)**2 + self.ttbar_rebin.GetBinError(bin)**2 + self.multijet.average_rebin.GetBinError(bin)**2)**0.5
            self.data_obs_closure.SetBinError  (bin, error)

        for bin in range(self.nBins_rebin+1, self.nBins_closure+1):
            self.data_obs_closure.SetBinError  (bin, 1.0)

        f.cd(self.channel)
        self.multijet_closure.Write()
        self.ttbar_closure   .Write()
        self.data_obs_closure.Write()
        self.signal_closure  .Write()

        self.fit_result = {}
        self.eigenVars = {}
        self.closure_TF1, self.closure_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.pvalue_ss, self.chi2_ss, self.ndf_ss = {}, {}, {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.fit_parameters_ss, self.fit_parameters_error_ss = {}, {}
        self.cUp, self.cDown = {}, {}
        self.fProb = {self.multijet.order: np.nan}
        self.order = None
        self.exit_message = []
        for order in range(self.multijet.order, maxOrderClosure+1): 
            self.makeFitFunction(order)
            self.fit(order)
            # try:
            if order>self.multijet.order:
                self.fProb[order] = fTest(self.chi2[order-1], self.chi2[order], self.ndf[order-1], self.ndf[order])
                if self.order is None and (self.pvalue[order-1] > probThreshold) and (self.fProb[order]<0.95):
                    print(self.pvalue)
                    print(self.fProb)
                    self.order = order-1 # store first order to satisfy min threshold. Will be used in closure fits
                    self.exit_message.append('-'*50)
                    self.exit_message.append('%s channel'%self.channel.upper())
                    self.exit_message.append('Satisfied goodness of fit and f-test at order %d:'%self.order)
                    self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% at order %d (p-value above threshold and f-test prefers this over previous order)'%(100*self.pvalue[self.order], 100*self.fProb[self.order], self.order))
                    self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% at order %d (f-test does not prefer this over previous order at greater than 95%%)'%(100*self.pvalue[order], 100*self.fProb[order], order))
                    self.exit_message.append('-'*50)
                # else:
                #     thisOrder = order-1
                #     self.exit_message.append('#'*50)
                #     self.exit_message.append('%s channel'%self.channel.upper())
                #     self.exit_message.append('Failed goodness of fit or f-test at order %d:'%thisOrder)
                #     self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% at order %d'%(100*self.pvalue[thisOrder], 100*self.fProb[thisOrder], thisOrder))
                #     self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% at order %d'%(100*self.pvalue[order], 100*self.fProb[order], order))
                #     self.exit_message.append('#'*50)
            # except KeyError:
            #     pass
            self.fitSpuriousSignal(order)
            self.writeClosureResults(order)
            self.plotFitResults(order)
        self.plotPValues()


    def makeFitFunction(self, order):

        def background_UserFunction(xArray, pars):
            bin = int(xArray[0])

            if bin > self.nBins_rebin:
                lp_idx = bin-self.nBins_rebin-1

                if lp_idx > order: # do nothing with extra bins
                    return 0.0

                lp_coefficient = pars[lp_idx]
                if self.doSpuriousSignal:
                    if lp_coefficient>0:
                        return -lp_coefficient/abs(self.cUp  [order][lp_idx])
                    else:
                        return -lp_coefficient/abs(self.cDown[order][lp_idx])                    

                # if lp_idx > self.multijet.order: # do nothing with extra bins
                #     return 0.0

                # if lp_coefficient>0:
                #     return -lp_coefficient/abs(self.multijet.cUp  [self.multijet.order][lp_idx])
                # else:
                #     return -lp_coefficient/abs(self.multijet.cDown[self.multijet.order][lp_idx])
                if lp_coefficient>0:
                    return -lp_coefficient/abs(self.multijet.cUp  [order][lp_idx])
                else:
                    return -lp_coefficient/abs(self.multijet.cDown[order][lp_idx])

            p = 1.0
            for lp_idx in range(max(order, self.multijet.order)+1):
                lpi = self.lp_integral[lp_idx][bin-1]
                p += pars[lp_idx] * lpi

            background = p*self.multijet.average_rebin.GetBinContent(bin) + self.ttbar_rebin.GetBinContent(bin)
            spuriousSignal = pars[order+1] * self.signal.GetBinContent(bin)
  
            return background + spuriousSignal

        f.cd(self.channel)
        self.closure_TF1[order] = ROOT.TF1 ('closure_TF1_order%d'%order, background_UserFunction, 0.5, 0.5+self.nBins_closure, max(order, self.multijet.order)+2)#+1 for order, +1 for spurious signal
        self.closure_TH1[order] = ROOT.TH1F('closure_TH1_order%d'%order,  '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)
        self.closure_spuriousSignal_TH1[order] = ROOT.TH1F('closure_spuriousSignal_TH1_order%d'%order,  '', self.nBins_closure, 0.5, 0.5+self.nBins_closure)

        for o in range(max(order, self.multijet.order)+1):
            if o <= self.multijet.order:
                self.closure_TF1[order].SetParName  (o, 'self-consistency c_%d'%o)
            else:
                self.closure_TF1[order].SetParName  (o, 'closure c_%d'%o)
            self.closure_TF1[order].SetParameter(o, 0.0)
        self.closure_TF1[order].SetParName  (order+1, 'spurious signal')
        self.closure_TF1[order].FixParameter(order+1, 0)

    def getEigenvariations(self, order, debug=False):
        n = order+1

        if n == 1:
            self.eigenVars[order] = np.array([self.closure_TF1[order].GetParError(0)])
            return

        cov = ROOT.TMatrixD(n,n)
        cor = ROOT.TMatrixD(n,n)

        for i in range(n):
            for j in range(n):
                cov[i][j] = self.fit_result[order].CovMatrix  (i, j)
                cor[i][j] = self.fit_result[order].Correlation(i, j)

        if debug:
            print("Covariance Matrix:")
            cov.Print()
            print("Correlation Matrix:")
            cor.Print()
        
        eigenVal = ROOT.TVectorD(n)
        eigenVec = cov.EigenVectors(eigenVal)
        
        # define relative sign of eigen-basis such that the first coordinate is always positive
        for j in range(n):
            if eigenVec[0][j] >= 0: continue
            for i in range(n):
                eigenVec[i][j] *= -1

        if debug:
            print("Eigenvectors (columns)")
            eigenVec.Print()
            print("Eigenvalues")
            eigenVal.Print()

        self.eigenVars[order] = np.zeros((n,n), dtype=np.float)
        for i in range(n):
            for j in range(n):
                self.eigenVars[order][i,j] = eigenVec[i][j] * eigenVal[j]**0.5

        if debug:
            print("Eigenvariations")
            for j in range(n):
                print(j, self.eigenVars[order][:,j])


    def getParameterDistribution(self, order):
        # new_dimensions = order - self.multijet.order
        # for o in range(max(order, self.multijet.order)+1):
        #     if o <= self.multijet.order:
        #         cUp = (self.multijet.cUp[self.multijet.order][o]**2 + self.fit_parameters[order][o]**2)**0.5
        #     else:
        #         cUp = new_dimensions**0.5 * abs(self.fit_parameters[order][o])
        for o in range(order+1):
            cUp = (self.multijet.cUp[order][o]**2 + self.fit_parameters[order][o]**2)**0.5
            cDown = -cUp
            try:
                self.cUp  [order].append( cUp )
                self.cDown[order].append( cDown )
            except KeyError:
                self.cUp  [order] = [cUp  ]
                self.cDown[order] = [cDown]


    def fit(self, order):
        self.fit_result[order] = self.data_obs_closure.Fit(self.closure_TF1[order], 'N0S')
        self.getEigenvariations(order)
        self.pvalue[order], self.chi2[order], self.ndf[order] = self.closure_TF1[order].GetProb(), self.closure_TF1[order].GetChisquare(), self.closure_TF1[order].GetNDF()
        print('Fit closure %s at order %d'%(self.channel, order))
        print('chi2/ndf = %3.2f/%3d = %2.2f'%(self.chi2[order], self.ndf[order], self.chi2[order]/self.ndf[order]))
        print(' p-value = %0.2f'%self.pvalue[order])

        self.ymax[order] = max(self.closure_TF1[order].GetMaximum(1,self.nBins_closure), 100*self.signal.GetMaximum())
        self.fit_parameters[order], self.fit_parameters_error[order] = [], []
        n = order+1
        self.fit_parameters      [order] = np.array([self.closure_TF1[order].GetParameter(o) for o in range(n)])
        self.fit_parameters_error[order] = np.array([self.closure_TF1[order].GetParError (o) for o in range(n)])
        self.getParameterDistribution(order)

        for bin in range(1,self.nBins_closure+1):
            self.closure_TH1[order].SetBinContent(bin, self.closure_TF1[order].Eval(bin))
            #self.closure_TH1[order].SetBinError  (bin, self.data_obs_closure.GetBinError(bin))
            self.closure_TH1[order].SetBinError  (bin, 0.0)
            
        f.cd(self.channel)
        self.closure_TH1[order].Write()


    def fitSpuriousSignal(self, order):
        self.doSpuriousSignal = True
        self.closure_TF1[order].SetParameter(order+1, 0)
        self.closure_TF1[order].SetParLimits(order+1, -100, 100)
        self.data_obs_closure.Fit(self.closure_TF1[order], 'N0')
        self.spuriousSignal[order]      = self.closure_TF1[order].GetParameter(order+1)
        self.spuriousSignalError[order] = self.closure_TF1[order].GetParError (order+1)

        self.pvalue_ss[order], self.chi2_ss[order], self.ndf_ss[order] = self.closure_TF1[order].GetProb(), self.closure_TF1[order].GetChisquare(), self.closure_TF1[order].GetNDF()
        print('Fit spurious signal %s at order %d'%(self.channel, order))
        print('chi2/ndf = %3.2f/%3d = %2.2f'%(self.chi2_ss[order], self.ndf_ss[order], self.chi2_ss[order]/self.ndf_ss[order]))
        print(' p-value = %0.2f'%self.pvalue_ss[order])

        n = order+1
        self.fit_parameters_ss      [order] = np.array([self.closure_TF1[order].GetParameter(o) for o in range(n)])
        self.fit_parameters_error_ss[order] = np.array([self.closure_TF1[order].GetParError (o) for o in range(n)])

        for bin in range(1,self.nBins_closure+1):
            self.closure_spuriousSignal_TH1[order].SetBinContent(bin, self.closure_TF1[order].Eval(bin))
            #self.closure_spuriousSignal_TH1[order].SetBinError  (bin, self.data_obs_closure.GetBinError(bin))
            self.closure_spuriousSignal_TH1[order].SetBinError  (bin, 0.0)
        f.cd(self.channel)
        self.closure_spuriousSignal_TH1[order].Write()

        self.closure_TF1[order].FixParameter(order+1, 0)
        self.doSpuriousSignal = False
        print('spurious signal = %2.2f +/- %f'%(self.spuriousSignal[order], self.spuriousSignalError[order]))


    def writeClosureResults(self,order=None):
        if order is None: order = self.order
        nLPs = order+1
        closureResults = "ZZ4b/nTupleAnalysis/combine/closureResults_%s_order%d.txt"%(self.channel,order)
        closureResultsFile = open(closureResults, 'w')
        print('Write Closure Results File: \n>> %s'%(closureResults))
        for i in range(nLPs):
            cUp   = self.cUp  [order][i]
            cDown = self.cDown[order][i]
            systUp    = "1+"
            systDown  = "1+"
            systUp   += "(%f)*(%s)"%(cUp,   lps[i].replace(' ',''))
            systDown += "(%f)*(%s)"%(cDown, lps[i].replace(' ',''))
            systUp    = "multijet_LP%i_%sUp   %s"%(i, channel, systUp)
            systDown  = "multijet_LP%i_%sDown %s"%(i, channel, systDown)
            print(systUp)
            print(systDown)
            closureResultsFile.write(systUp+'\n')
            closureResultsFile.write(systDown+'\n')

        ssUp   = self.spuriousSignal[order]
        ssDown = -ssUp
        systUp   = 'multijet_spurious_signal_%sUp   %f'%(channel, ssUp)
        systDown = 'multijet_spurious_signal_%sDown %f'%(channel, ssDown)
        print(systUp)
        print(systDown)
        closureResultsFile.write(systUp+'\n')
        closureResultsFile.write(systDown+'\n')        
        closureResultsFile.close()


    def plotFitResults(self, order):
        n = order+1

        #plot fit parameters
        x,y,s,c = [],[],[],[]
        x.append( 100*self.fit_parameters[order][0] )
        if n==1:
            y.append( 0 )
        if n>1:
            y.append( 100*self.fit_parameters[order][1] )
        if n>2:
            c.append( 100*self.fit_parameters[order][2] )
        if n>3:
            s.append( 100*self.fit_parameters[order][3] )

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
        ax.set_title('Closure Fit Parameters (%s)'%self.channel.upper())
        ax.set_xlabel('c$_0$ (\%)')
        ax.set_ylabel('c$_1$ (\%)')

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
            # draw 1\sigma ellipses
            ellipse_self_consistency = Ellipse((0,0), 
                                               # width =100*(self.multijet.cUp[self.multijet.order][0]-self.multijet.cDown[self.multijet.order][0]),
                                               # height=100*(self.multijet.cUp[self.multijet.order][1]-self.multijet.cDown[self.multijet.order][1]),
                                               width =100*(self.multijet.cUp[order][0]-self.multijet.cDown[order][0]),
                                               height=100*(self.multijet.cUp[order][1]-self.multijet.cDown[order][1]),
                                               facecolor = 'none',
                                               edgecolor = 'b', #CMURED,
                                               linestyle = '-',
                                               linewidth = 0.5,
                                               zorder=1,
            )
            #transf = transforms.Affine2D().scale(self.cUp[order][0]-self.cDown[order][0], self.cUp[order][1]-self.cDown[order][1])
            #ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse_self_consistency)

            ellipse_closure = Ellipse((0,0), 
                                      width =100*(self.cUp[order][0]-self.cDown[order][0]),
                                      height=100*(self.cUp[order][1]-self.cDown[order][1]),
                                      facecolor = 'none',
                                      edgecolor = 'r', #CMURED,
                                      linestyle = '-',
                                      linewidth = 0.5,
                                      zorder=1,
            )
            #transf = transforms.Affine2D().scale(self.cUp[order][0]-self.cDown[order][0], self.cUp[order][1]-self.cDown[order][1])
            #ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse_closure)

        maxr=np.zeros((2, len(x)), dtype=np.float)
        minr=np.zeros((2, len(x)), dtype=np.float)
        if n>1:
            #generate a ton of random points on a hypersphere in dim=n so surface is dim=n-1.
            points  = np.random.uniform(0,1,  (n, min(100**(n-1),10**7) )  ) # random points in a hypercube
            points /= np.linalg.norm(points, axis=0) # normalize them to the hypersphere surface

            # find the point which maximizes the change in c_0**2 + c_1**2
            for i in range(len(x)):
                plane = np.matmul( self.eigenVars[order][0:min(n,2),:], points )
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
                maxr[0,i] = self.eigenVars[order][0]
        
        minr *= 100
        maxr *= 100

        # print(maxr)
        # print(minr)
        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        
        plt.scatter(x, y, **kwargs)

        if n>2:
            for i in range(len(x)):
                label = '\n'.join(['c$_%d$ = %2.1f'%(o, 100*self.fit_parameters[order][o])+'\%' for o in range(2,n)])
                xy = np.array([x[i],y[i]])
                xy = [xy+minr[:,i]+maxr[:,i],
                      xy+minr[:,i]-maxr[:,i],
                      xy-minr[:,i]+maxr[:,i],
                      xy-minr[:,i]-maxr[:,i]]
                xy = max(xy, key=lambda p: p[0]**2+p[1]**2)
                if xy[0]>0:
                    horizontalalignment = 'left'
                else:
                    horizontalalignment = 'right'
                if xy[1]>0:
                    verticalalignment = 'bottom'
                else:
                    verticalalignment = 'top'
                bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0)
                ax.annotate(label, xy,# label,
                            ha=horizontalalignment, va=verticalalignment,
                            bbox=bbox)

        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/stage1_parameters_order%d_closure.pdf'%(mixName, region, self.channel, order)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/stage1_parameters_order%d_closure.pdf'%(mixName, rebin, region, self.channel, order)            
        print("fig.savefig( "+name+" )")
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)


    def plotPValues(self):
        fig, (ax) = plt.subplots(nrows=1)
        ax.set_ylim(0,1)
        ax.set_xticks(range(maxOrderClosure+1))
        #plt.yscale('log')

        x = sorted(self.pvalue.keys())
        y = [self.pvalue[o] for o in x]
        ax.set_title('Closure Fit (%s)'%self.channel.upper())
        ax.plot([self.multijet.order,maxOrderClosure], [probThreshold,probThreshold], color='r', alpha=0.5, linestyle='--', linewidth=1)
        ax.plot([self.multijet.order,maxOrderClosure], [0.95,0.95], color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.plot(x, y, label='p-value', color='r', linewidth=2)
        if self.order is not None:
            ax.scatter(self.order, self.pvalue[self.order], color='k', marker='*', zorder=10)
        
        x = sorted(self.fProb.keys())
        x = x[1:]
        y = [self.fProb[o] for o in x]
        marker = '' if len(x)>1 else 'o'
        ax.plot(x, y, label='f-test', color='k', linewidth=2, marker=marker)

        ax.set_xlabel('Polynomial Order')
        ax.set_ylabel('Fit p-value')
        ax.legend(loc='upper left', fontsize='small')

        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/stage1_pvalues_closure.pdf'%(mixName, region, self.channel)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/stage1_pvalues_closure.pdf'%(mixName, rebin, region, self.channel)
        print("fig.savefig( "+name+" )")
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
            'label' : 'ZZ+ZH(#times100)',
            'legend': 4,
            'weight': 100,
            'color' : 'ROOT.kViolet'}

        if 'zz' in self.channel:
            xTitle = 'SvB P_{ZZ}+P_{ZH} : P_{ZZ} > P_{ZH}'
        if 'zh' in self.channel:
            xTitle = 'SvB P_{ZZ}+P_{ZH} : P_{ZH} #geq P_{ZH}'
            
        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'ratioErrors': True,
                      'ratio'     : True,
                      'rMin'      : 0.9,
                      'rMax'      : 1.1,
                      'rebin'     : rebin,
                      'rTitle'    : 'Data / Bkgd.',
                      'xTitle'    : xTitle,
                      'yTitle'    : 'Events',
                      'yMax'      : 1.4*(self.ymax[self.order] if self.order is not None else self.ymax[maxOrderClosure]),#*ymaxScale, # make room to show fit parameters
                      #'xleg'      : [0.13, 0.13+0.5] if 'SR' in region else ,
                      # 'legendSubText' : ['#bf{Fit:}',
                      #                    '#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2[order],self.ndf[order],self.chi2[order]/self.ndf[order]),
                      #                    'p-value = %2.0f%%'%(self.pvalue[order]*100),
                      #                    ],
                      'lstLocation' : 'right',
                      'outputName': 'mix_%s'%(str(mix))}

        if type(rebin) is list:
            parameters['outputDir'] = 'closureFits/%s/variable_rebin/%s/%s/'%(mixName, region, self.channel)
        else:
            parameters['outputDir'] = 'closureFits/%s/rebin%i/%s/%s/'%(mixName, rebin, region, self.channel)

        print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
        PlotTools.plot(samples, parameters, debug=False)


    def plotFit(self, order, plotSpuriousSignal=False):
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
        samples[closureFileName]['%s/closure_TH1_order%d'%(self.channel, order)] = {
            'label' : 'Fit (order %d)'%order,
            'legend': 4,
            'ratio': 'denom A', 
            'color' : 'ROOT.kRed'}
        if plotSpuriousSignal:
            samples[closureFileName]['%s/closure_spuriousSignal_TH1_order%d'%(self.channel, order)] = {
                'label' : 'Spurious Signal Fit #mu=%1.1f#pm%1.1f'%(self.spuriousSignal[order], self.spuriousSignalError[order]),
                'legend': 5,
                'ratio': 'denom A', 
                'color' : 'ROOT.kBlue'}
            samples[closureFileName]['%s/signal_closure'%self.channel] = {
                'label' : 'ZZ+ZH(#times100)',
                'legend': 6,
                'weight': 100,
                'color' : 'ROOT.kViolet'}

        if 'zz' in self.channel:
            xTitle = 'SvB P_{ZZ}+P_{ZH} Bin : P_{ZZ} > P_{ZH}'
        if 'zh' in self.channel:
            xTitle = 'SvB P_{ZZ}+P_{ZH} Bin : P_{ZH} #geq P_{ZH}'
            
        ymaxScale = 1.7 + max(0, (order-2)/4.0) if 'zz'==self.channel else 1.7 + max(0, (order-2)/4.0)

        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.nBins_rebin+0.5,  0,self.nBins_rebin+0.5,self.ymax[order]*1.0]],
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
                      'yMax'      : self.ymax[order]*ymaxScale,#*ymaxScale, # make room to show fit parameters
                      'xleg'      : [0.13, 0.13+0.45],
                      'lstLocation' : 'right',
                      'outputName': 'stage%d_closure_order%d'%(2 if plotSpuriousSignal else 1, order)}

        if plotSpuriousSignal:
            parameters['legendSubText'] = ['#bf{Spurious Signal Fit:}',
                                         '#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2_ss[order],self.ndf_ss[order],self.chi2_ss[order]/self.ndf_ss[order]),
                                         'p-value = %2.0f%%'%(self.pvalue_ss[order]*100)]
            for i in range(order+1):
                parameters["legendSubText"] += ["#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma"%(i, self.fit_parameters_ss[order][i]*100, abs(self.fit_parameters_ss[order][i])/self.fit_parameters_error_ss[order][i])]
        else:
            parameters['legendSubText'] = ['#bf{Fit:}',
                                         '#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2[order],self.ndf[order],self.chi2[order]/self.ndf[order]),
                                         'p-value = %2.0f%%'%(self.pvalue[order]*100)]
            for i in range(order+1):
                parameters["legendSubText"] += ["#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma"%(i, self.fit_parameters[order][i]*100, abs(self.fit_parameters[order][i])/self.fit_parameters_error[order][i])]

        parameters['ratioLines'] = [[self.nBins_rebin+0.5, parameters['rMin'], self.nBins_rebin+0.5, parameters['rMax']]]
        #parameters['xMax'] = self.nBins_rebin+self.multijet.order+1.5 if not plotSpuriousSignal else self.nBins_rebin+order+1.5
        parameters['xMax'] = self.nBins_rebin+order+1.5

        if type(rebin) is list:
            parameters['outputDir'] = 'closureFits/%s/variable_rebin/%s/%s/'%(mixName, region, self.channel)
        else:
            parameters['outputDir'] = 'closureFits/%s/rebin%i/%s/%s/'%(mixName, rebin, region, self.channel)

        print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
        PlotTools.plot(samples, parameters, debug=False)

    def print_exit_message(self):
        for line in self.exit_message: print(line)





# make multijet ensembles and perform self-consistency fits
multijetEnsembles = {}
for channel in channels:
    if type(rebin) is list:
        mkpath("%s/closureFits/%s/variable_rebin/%s/%s"%(basePath, mixName, region, channel))
    else:
        mkpath("%s/closureFits/%s/rebin%i/%s/%s"%(basePath, mixName, rebin, region, channel))
    multijetEnsembles[channel] = multijetEnsemble(channel)

# run closure fits using average multijet model and constrained self-consistency function
closures = {}
for channel in channels:
    closures[channel] = closure(channel, multijetEnsembles[channel])

# close input file and make plots of multijet self-consistency fits
f.Close()
for channel in channels:
    for order in range(maxOrderEnsemble+1):
        multijetEnsembles[channel].plotFit(order)
    for order in range(multijetEnsembles[channel].order, maxOrderClosure+1):
        closures[channel].plotFit(order)
        closures[channel].plotFit(order, plotSpuriousSignal=True)
    for m in range(nMixes):
        closures[channel].plotMix(m)
    closures[channel].plotMix('ave')
    closures[channel].print_exit_message()
    
