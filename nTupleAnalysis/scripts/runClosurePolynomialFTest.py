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
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

CMURED = '#d34031'

year = "RunII"
lumi = 132.6
rebin = 5
#rebin = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
maxOrder = 5

USER = getUSER()
CMSSW = getCMSSW()
basePath = '/uscms/home/%s/nobackup/%s/src'%(USER, CMSSW)

mixName = "3bMix4b_rWbW2"
ttAverage = False

# mixName = "3bMix4b_4bTT"
# ttAverage = False

# mixName = "3bMix4b_4bTT_rWbW2"
# ttAverage = False

doSpuriousSignal = True
dataAverage = True
nMixes = 10
region = "SR"
region = 'SRNoHH'
#closureFileName = "ZZ4b/nTupleAnalysis/combine/hists_closure.root"
#closureFileName = "ZZ4b/nTupleAnalysis/combine/hists_closure_3bMix4b_rWbW2_b0p60p3_SR.root"
closureFileName = "ZZ4b/nTupleAnalysis/combine/hists_closure_MixedToUnmixed_"+mixName+"_b0p60p3_"+region+".root"
closureFileName = "ZZ4b/nTupleAnalysis/combine/hists_closure_"+mixName+"_b0p60p3_"+region+".root"

regionName = {"SB": "Sideband",
              "CR": "Control Region",
              "SR": "Signal Region",
              "SRNoHH": "Signal Region (Veto HH)",
          }


channels = ["zz",
            "zh",
            ]

# Get Signal templates for spurious signal fits
zzFile = ROOT.TFile('/uscms/home/%s/nobackup/ZZ4b/ZZ4bRunII/hists.root'%(USER), 'READ')
zhFile = ROOT.TFile('/uscms/home/%s/nobackup/ZZ4b/bothZH4bRunII/hists.root'%(USER), 'READ')
signals = {}
for ch in channels:
    var = 'SvB_ps_%s'%ch
    histPath = 'passMDRs/fourTag/mainView/%s/%s'%('ZZZHSR',var)
    signals[ch]  =  zzFile.Get(histPath)
    signals[ch].Add(zhFile.Get(histPath))
    signals[ch].SetName('total signal in channel %s'%ch)
    if type(rebin)==list:
        signals[ch], _ = PlotTools.do_variable_rebinning(signals[ch], rebin, scaleByBinWidth=False)
    else:
        signals[ch].Rebin(rebin)
    signals[ch].SetDirectory(0)
zzFile.Close()
zhFile.Close()

f=ROOT.TFile(closureFileName, "UPDATE")
mixes = [mixName+"_v%d"%i for i in range(nMixes)]

probThreshold = 0.682689492137 # 1sigma 
        
def addYears(directory):
    hists = []
    for process in ["ttbar","multijet","data_obs"]:
        try:
            hists.append( f.Get(directory+"2016/"+process) )
            #hists[-1].Sumw2()
            hists[-1].Add(f.Get(directory+"2017/"+process))
            hists[-1].Add(f.Get(directory+"2018/"+process))
            try:
                f.Get(directory).IsZombie()
            except ReferenceError:
                f.mkdir(directory)
            f.cd(directory)
            hists[-1].Write()
        except AttributeError:
            f.ls()
            exit()

def addMixes(directory):
    hists = []
    for process in ["ttbar","multijet","data_obs"]:
        hists.append( f.Get(mixes[0]+"/"+directory+"/"+process) )
        #hists[-1].Sumw2()
        #hists[-1].SetName("data_obs")
        for mix in mixes[1:]:
            hists[-1].Add( f.Get(mix+"/"+directory+"/"+process) )

        try:
            f.Get(directory).IsZombie()
        except ReferenceError:
            f.mkdir(directory)

        f.cd(directory)
        nMixes = len(mixes)
        hists[-1].Scale(1.0/nMixes)

        # nBins = hists[-1].GetSize()-2
        # for bin in range(1,nBins+1):
        #     error = hists[-1].GetBinError(bin)
        #     hists[-1].SetBinError(bin, error * nMixes**0.5)

        hists[-1].Write()

for mix in mixes:
   for channel in channels:
       addYears(mix+'/'+channel)

for channel in channels:
   addMixes(channel)

f.Close()
f=ROOT.TFile(closureFileName, "UPDATE")

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


def fTest(chi2_1, chi2_2, ndf_1, ndf_2):
    d1 = (ndf_1-ndf_2)
    d2 = ndf_2
    N = (chi2_1-chi2_2)/d1
    D = chi2_2/d2
    fStat = N/D
    fProb = scipy.stats.f.cdf(fStat, d1, d2)
    expectedFStat = scipy.stats.distributions.f.isf(0.05, d1, d2)
    print "d1, d2 = %d, %d"%(d1, d2)
    print "N, D = %f, %f"%(N, D)
    print "    f(%i,%i) = %f (expected at 95%%: %f)"%(d1,d2,fStat,expectedFStat)
    print "f.cdf(%i,%i) = %3.0f%%"%(d1,d2,100*fProb)
    print 
    return fProb


class mixedData:
    def __init__(self, directory, name):
        self.directory = directory
        self.name = name
        #self.hist = ROOT.TH1F( f.Get( self.directory+"/data_obs" ) )
        self.hist = f.Get( self.directory+"/data_obs" ).Clone()
        self.hist.SetName(str(np.random.uniform()))
        if type(rebin)==list:
            self.hist, _ = PlotTools.do_variable_rebinning(self.hist, rebin, scaleByBinWidth=False)
        else:
            self.hist.Rebin(rebin)
        self.chi2 = None
        
    def SetBinError(self, bin, ttbar_error, multijet_error):
        data_error = self.hist.GetBinError(bin)
        total_error = (data_error**2 + ttbar_error**2 + multijet_error**2)**0.5
        self.hist.SetBinError(bin, total_error)

    def fit(self, model):
        self.hist.Fit(model, "N0QS")
        self.chi2 = model.GetChisquare()
        self.ndf  = model.GetNDF()


class model:
    def __init__(self, directory, order=None, data_obs_name=None, ttbar_name=None, mix=None, channel=''):
        self.directory= directory
        self.channel = channel
        f.cd(self.directory)
        #f.ls()
        self.name     = directory.replace("/","_").replace("_"+channel,'')
        self.mix = mix
        self.mixNumber = int(mix[mix.find('_v')+2:])
        self.ttbar_name = ttbar_name if ttbar_name else directory+"/ttbar"
        self.ttbar    = f.Get( self.ttbar_name ).Clone() #f.Get(directory+"/ttbar")
        self.ttbar.SetName(str(np.random.uniform()))
        self.multijet = f.Get(directory+"/multijet")
        self.data_obs_name = data_obs_name if data_obs_name else directory+"/data_obs"
        self.data_obs = f.Get( self.data_obs_name ).Clone()
        self.data_obs.SetName(str(np.random.uniform()))
        self.data_obs_raw = self.data_obs.Clone()
        self.data_obs_raw.SetName(str(np.random.uniform()))
        self.nBins = self.data_obs.GetSize()-2 #underflow and overflow

        self.mixes=[mixedData(directory.replace(mix, otherMix), otherMix) for otherMix in mixes]

        self.background_TH1s = {} #ROOT.TH1F("background_TH1", "", self.nBins, 0, 1)
        self.background_TH1s['None'] = ROOT.TH1F("background_TH1", "", self.nBins, 0, 1)
        for b in range(1,self.nBins+1):
            self.background_TH1s['None'].SetBinContent(b, self.ttbar.GetBinContent(b)+self.multijet.GetBinContent(b))
            self.background_TH1s['None'].SetBinError(b, (self.ttbar.GetBinError(b)**2+self.multijet.GetBinError(b)**2)**0.5)
        f.cd(self.directory)
        self.background_TH1s['None'].Write()

        self.rebin = rebin
        if type(self.rebin)==list:
            self.ttbar,    _ = PlotTools.do_variable_rebinning(self.ttbar,    self.rebin, scaleByBinWidth=False)
            self.multijet, _ = PlotTools.do_variable_rebinning(self.multijet, self.rebin, scaleByBinWidth=False)
            self.data_obs, _ = PlotTools.do_variable_rebinning(self.data_obs, self.rebin, scaleByBinWidth=False)
        else:
            self.ttbar   .Rebin(self.rebin)
            self.multijet.Rebin(self.rebin)
            self.data_obs.Rebin(self.rebin)

        self.nBins_rebin = self.data_obs.GetSize()-2

        # So that fit includes stat error from background templates, combine all stat error in quadrature
        for bin in range(1,self.nBins_rebin+1):
            data_error = self.data_obs_raw.GetBinError(bin)
            # data_content = self.data_obs.GetBinContent(bin)
            # print '%2d, %1.4f%%'%(bin,100*data_error**2/data_content)
            ttbar_error = self.ttbar.GetBinError(bin)
            multijetErrorScale = (self.data_obs.GetBinContent(bin)-self.ttbar.GetBinContent(bin)) / self.multijet.GetBinContent(bin)
            multijet_error = self.multijet.GetBinError(bin) * multijetErrorScale
            total_error = (data_error**2 + ttbar_error**2 + multijet_error**2)**0.5
            # multijet_content = self.multijet.GetBinContent(bin)
            # ttbar_content = self.ttbar.GetBinContent(bin)
            # print '%2d | mj %2.1f%% | tt %2.1f%% | total %2.1f%%'%(bin, 100*multijet_error/multijet_content, 100*ttbar_error/ttbar_content, 100*(multijet_error**2+ttbar_error**2)**0.5/(multijet_content+ttbar_content))
            self.data_obs.SetBinError(bin, total_error)

            for otherMix in self.mixes: otherMix.SetBinError(bin, ttbar_error, multijet_error)

        print self.data_obs.Integral(), self.ttbar.Integral()+self.multijet.Integral()

        self.polynomials = {}
        self.background_TF1s = {}
        self.chi2s = {}
        self.ndfs = {}
        self.chi2PerNdfs = {}
        self.probs = {}
        self.fProbs = {}
        self.parValues = {}
        self.parErrors = {}
        self.ymaxs = {}
        self.eigenVars = {}
        self.eigenPars = {}
        self.residuals = {}
        #self.order = 2

        if order is not None:
            self.order = order
            self.makeBackgrondTF1(order)


    def makeBackgrondTF1(self, order):
        #self.polynomials[order] = ROOT.TF1("pol"+str(order), "pol"+str(order), 0, 1)

        def background_UserFunction(xArray, pars):
            x = xArray[0]
            #self.polynomials[order].SetParameters(par)
            bin = self.ttbar.FindBin(x)
            l, u = self.ttbar.GetBinLowEdge(bin), self.ttbar.GetXaxis().GetBinUpEdge(bin)
            w = self.ttbar.GetBinWidth(bin)

            p = 1
            for i, par in enumerate(pars):
                p += par*lp[i].Integral(l,u)/w #(u-l)

            #p = self.polynomials[order].Integral(l,u)/w
            return self.ttbar.GetBinContent(bin) + p*self.multijet.GetBinContent(bin)

        self.background_TF1s[order] = ROOT.TF1 ("background_TF1_order%d"%order, background_UserFunction, 0, 1, order+1)
        self.background_TH1s[order] = ROOT.TH1F("background_TH1_order%d"%order, "", self.nBins, 0, 1)


    def getEigenvariations(self, order=None, debug=False):
        if order is None: order = self.order
        n = order+1
        if n == 1:
            self.eigenVars[order] = np.array([[self.background_TF1s[order].GetParError(0)]])
            return

        cov = ROOT.TMatrixD(n,n)
        cor = ROOT.TMatrixD(n,n)
        for i in range(n):
            for j in range(n):
                cov[i][j] = self.fitResult.CovMatrix(i,j)
                cor[i][j] = self.fitResult.Correlation(i,j)

        if debug:
            print "Covariance Matrix:"
            cov.Print()
            print "Correlation Matrix:"
            cor.Print()

        # eigenMatrix = ROOT.TMatrixDEigen(cov)
        # eigenVal = eigenMatrix.GetEigenValues()
        # eigenVec = eigenMatrix.GetEigenVectors()
        
        eigenVal = ROOT.TVectorD(n)
        eigenVec = cov.EigenVectors(eigenVal)
        
        # define relative sign of eigen-basis such that the first coordinate is always positive
        for j in range(n):
            if eigenVec[0][j] >= 0: continue
            for i in range(n):
                eigenVec[i][j] *= -1

        if debug:
            print "Eigenvectors (columns)"
            eigenVec.Print()
            print "Eigenvalues"
            eigenVal.Print()

        errorVec = np.zeros((n,n), dtype=np.float)
        #errorVec = [np.array(range(n), dtype=np.float) for i in range(n)]
        for i in range(n):
            for j in range(n):
                errorVec[i,j] = eigenVec[i][j] * eigenVal[j]**0.5

        self.eigenVars[order] = errorVec

        if debug:
            print "Eigenvariations"
            for j in range(n):
                print j, errorVec[:,j]

        #get best fit parameters in eigen-basis
        eigenPar = ROOT.TMatrixD(n,1)
        eigenPar.Mult(eigenVec, ROOT.TMatrixD(n,1,self.background_TF1s[order].GetParameters()) )
        if debug:
            print "Best fit parameters in eigen-basis"
            eigenPar.Print()
        self.eigenPars[order] = np.array(range(n), dtype=np.float)
        for i in range(n):
            self.eigenPars[order][i] = eigenPar[i][0]


    def storeFitResult(self, order=None):
        if order is None: order = self.order
        self.chi2s[order] = self.background_TF1s[order].GetChisquare()
        self.ndfs[order] = self.background_TF1s[order].GetNDF()
        self.chi2PerNdfs[order] = self.chi2s[order]/self.ndfs[order]
        self.probs[order] = self.background_TF1s[order].GetProb()
        nParam = self.background_TF1s[order].GetNumberFreeParameters()
        self.parValues[order] = np.array( [self.background_TF1s[order].GetParameter(i) for i in range(nParam)] )
        self.parErrors[order] = np.array( [self.background_TF1s[order].GetParError (i) for i in range(nParam)] )
        self.ymaxs[order] = self.background_TF1s[order].GetMaximum(0,1)
        self.write(order)


    def dumpFitResult(self, order=None):
        if order is None: order = self.order
        print "    chi^2 =",self.chi2s[order]
        print "      ndf =",self.ndfs[order]
        print "chi^2/ndf =",self.chi2PerNdfs[order]
        print "  p-value =",self.probs[order]

    
    def getResiduals(self, order=None):
        if order is None: order = self.order
        residuals = []
        for bin in range(1,self.nBins_rebin+1):
            x = self.data_obs.GetBinCenter(bin)
            try:
                res = (self.data_obs.GetBinContent(bin) - self.background_TF1s[order].Eval(x)) / self.data_obs.GetBinError(bin)
            except:
                res = 0
            residuals.append(res)
        self.residuals[order] = np.array(residuals)


    def fit(self, order=None):
        if order is None: order = self.order
        if order not in self.background_TF1s: self.makeBackgrondTF1(order)            
        self.fitResult = self.data_obs.Fit(self.background_TF1s[order], "N0QS")
        status = int(self.fitResult)
        #self.getResiduals(order)
        self.getEigenvariations(order)
        self.storeFitResult(order)
        self.dumpFitResult(order)


    def fitAllMixes(self):
        fig, (ax) = plt.subplots(nrows=1)

        ax.set_title("FvT trained with %s (%s channel)"%(self.name.replace("_"+self.channel,"").replace("_","\_"), self.channel))
        ax.set_xlabel('$\chi^2$')
        ax.set_ylabel('Arb. Units')

        chi2s = []
        markers = ['o', 'v', '^', '<', '>', 's', '*']
        colors = ['b','g','c','m']
        for i, otherMix in enumerate(self.mixes): 
            otherMix.fit(self.background_TF1s[self.order])
            chi2s.append(otherMix.chi2)
            
            kwargs = {'lw': 1,
                      'marker': markers[i%len(markers)],
                      'edgecolors': 'k',
                      'color': colors[i%len(colors)],
                      'label': otherMix.name.replace("_","\_"),
                      }

            plt.scatter(chi2s[-1], 0.0, **kwargs)

        chi2s = np.array(chi2s)
        print "chi2s.mean():",chi2s.mean(), "chi2s.std():",chi2s.std()


        x = np.linspace(scipy.stats.chi2.ppf(0.001, self.mixes[0].ndf), scipy.stats.chi2.ppf(0.999, self.mixes[0].ndf), 100)
        ax.plot(x, scipy.stats.chi2.pdf(x, self.mixes[0].ndf), 'r-', lw=2, alpha=1, label='$\chi^2$ PDF (NDF = %d)'%self.mixes[0].ndf)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim([-0.005, ylim[1]])
        
        ax.legend(loc='upper right', fontsize='small', scatterpoints=1)

        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/chi2s_%s_order%i.pdf'%(mixName, region, self.channel, self.name, self.order)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/chi2s_%s_order%i.pdf'%(mixName, rebin, region, self.channel, self.name, self.order)
        print "fig.savefig( "+name+" )"
        fig.savefig( name )
        plt.close(fig)


    def fTest(self, order2, order1): #order2 > order1
        for order in [order1, order2]:
            if order in self.chi2s: continue
            self.makeBackgrondTF1(order)
            self.fit(order)

        d1 = self.ndfs[order1] - self.ndfs[order2]
        d2 = self.ndfs[order2]
        
        N = (self.chi2s[order1] - self.chi2s[order2])/d1
        D = self.chi2s[order2]/d2

        self.fStat = N/D
        self.fProbs[order2] = scipy.stats.f.cdf(self.fStat, order2-order1, self.ndfs[order2])

        expectedFStat = scipy.stats.distributions.f.isf(0.05, d1, d2)
        print "Single Model F-Test:"
        print "d1, d2 = %d, %d"%(d1,d2)
        print "N, D = %f, %f"%(N, D)
        print "    f(%i,%i) = %f (expected at 95%%: %f)"%(order2,order1,self.fStat,expectedFStat)
        print "f.cdf(%i,%i) = %2.0f%%"%(order2,order1,100*self.fProbs[order2])
        print
        return self.fProbs[order2]


    def runFTest(self):
        order = 0
        while self.fTest(order+1, order) > 0.95:
            order += 1
        print "F-Test prefers order %i over %i at only %2.0f%%"%(order+1, order, 100*self.fProbs[order+1])
        self.setOrder(order)


    def setOrder(self, order):
        self.order = order
        self.background_TF1 = self.background_TF1s[order]
        self.chi2 = self.chi2s[order]
        self.ndf = self.ndfs[order]
        self.chi2PerNdf = self.chi2PerNdfs[order]
        self.prob = self.probs[order]
        self.parValue = self.parValues[order]#np.array([self.parValues[order][i] for i in range(order+1)], dtype=np.float) #self.parValues[order]
        self.parError = self.parErrors[order]
        self.ymax = self.ymaxs[order]


    # def toy(self):
    #     self.data_toy = ROOT.TH1F("data_toy", "", self.nBins_rebin, 0, 1)
    #     for bin in range(1,self.nBins_rebin+1):
    #         ttbar_content, ttbar_error = self.ttbar.GetBinContent(bin), self.ttbar.GetBinError(bin)
    #         multijet_content, multijet_error = self.multijet.GetBinContent(bin), self.multijet.GetBinError(bin)
    #         total_content = ttbar_content + multijet_content
    #         total_error = (total_content + ttbar_error**2 + multijet_error**2)**0.5
    #         toy_content = np.random.normal(total_content, total_error)
    #         toy_error = (toy_content + ttbar_error**2 + multijet_error**2)**0.5
    #         self.data_toy.SetBinContent(bin, toy_content)
    #         self.data_toy.SetBinError(bin, toy_error)
    #     print "Fit Toy:"
    #     self.data_toy.Fit(self.background_TF1, "L N0")
    #     self.dumpFitResult()


    def write(self, order=None):
        if order is None: order = self.order
        f.cd(self.directory)

        for bin_rebin in range(1,self.nBins_rebin+1):
            x = self.data_obs.GetBinCenter(bin_rebin)
            c_tf1 = self.background_TF1s[order].Eval(x)
            c_none = 0
            for i in range(1,rebin+1):
                bin = (bin_rebin-1)*rebin+i
                c_none += self.background_TH1s['None'].GetBinContent(bin)
            norm = c_tf1/c_none
            for i in range(1,rebin+1):
                bin = (bin_rebin-1)*rebin+i
                self.background_TH1s[order].SetBinContent(bin, norm*self.background_TH1s['None'].GetBinContent(bin))
                self.background_TH1s[order].SetBinError  (bin, norm*self.background_TH1s['None'].GetBinError  (bin))
            
        self.background_TH1s[order].Write()


    def plotFit(self, order=None):
        if order is None: order = self.order
        samples=collections.OrderedDict()
        samples[closureFileName] = collections.OrderedDict()
        samples[closureFileName][self.data_obs_name] = {
            "label" : ("Ave. Mix %.1f/fb")%(lumi) if dataAverage else ("Mix v%d %.1f/fb")%(self.mixNumber, lumi),
            "legend": 1,
            "isData" : True,
            "ratio" : "numer A",
            "color" : "ROOT.kBlack"}
        samples[closureFileName][self.directory+"/multijet"] = {
            "label" : "Multijet Model %s"%(self.name[-2:]),
            "legend": 2,
            "stack" : 3,
            "ratio" : "denom A",
            "color" : "ROOT.kYellow"}
        samples[closureFileName][self.ttbar_name] = {
            "label" : "t#bar{t}",
            "legend": 3,
            "stack" : 2,
            "ratio" : "denom A",
            "color" : "ROOT.kAzure-9"}
        samples[closureFileName][self.directory+"/background_TH1_order%d"%order] = {
            "label" : "Fit (order %d)"%order,
            "legend": 4,
            "ratio": "denom A", 
            "color" : "ROOT.kRed"}

        if "zz" in self.directory:
            xTitle = "SvB Regressed P(ZZ)+P(ZH), P(ZZ) > P(ZH)"
        if "zh" in self.directory:
            xTitle = "SvB Regressed P(ZZ)+P(ZH), P(ZH) #geq P(ZZ)"
            
        if region=='SB':
            ymaxScale = 1.6
        if region=='CR':
            ymaxScale = 1.6
        if region=='SR':
            ymaxScale = (1.0 + self.order/6.0) if self.channel == 'zz' else max(1.6,1.2 + self.order/6.0)
        if region=='SRNoHH':
            ymaxScale = 1.6

        parameters = {"titleLeft"   : "#bf{CMS} Internal",
                      "titleCenter" : regionName[region],
                      "titleRight"  : "Pass #DeltaR(j,j)",
                      "maxDigits"   : 4,
                      "ratio"     : True,
                      "rMin"      : 0.9,
                      "rMax"      : 1.1,
                      "rebin"     : self.rebin,
                      "rTitle"    : "Data / Bkgd.",
                      "xTitle"    : xTitle,
                      "yTitle"    : "Events",
                      "yMax"      : self.ymaxs[order]*ymaxScale, # make room to show fit parameters
                      "xleg"      : [0.13, 0.13+0.33],
                      "legendSubText" : ["#bf{Fit:} #chi^{2}/DoF, p-value = %0.2f, %2.0f%%"%(self.chi2s[order]/self.ndfs[order], self.probs[order]*100),
                                         ],
                      "lstLocation" : "right",
                      "outputName": 'fit_order%d%s'%(order, self.name.replace(mixName,""))}
        if type(rebin) is list:
            parameters["outputDir"] = 'closureFits/%s/variable_rebin/%s/%s/'%(mixName, region, self.channel)
        else:
            parameters["outputDir"] = 'closureFits/%s/rebin%i/%s/%s/'%(mixName, rebin, region, self.channel)
            
        for i in range(order+1):
            percentError = 100*self.parErrors[order][i]/abs(self.parValues[order][i])
            if percentError > 100:
                errorString = ">100"
            else:
                errorString = "%4.0f"%percentError
            #parameters["legendSubText"] += ["#font[82]{c_{%i} =%6.3f #pm%s%%}"%(i, self.parValue[i], errorString)]
            parameters["legendSubText"] += ["#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma"%(i, self.parValues[order][i]*100, abs(self.parValues[order][i])/self.parErrors[order][i])]

        # parameters["legendSubText"] += ["",
        #                                 "#chi^{2}/DoF = %0.2f"%(self.chi2/self.ndf),
        #                                 "p-value = %2.0f%%"%(self.prob*100)]

        print "make ",parameters["outputDir"]+parameters["outputName"]+".pdf"
        PlotTools.plot(samples, parameters, debug=False)


class ensemble:
    def __init__(self, channel, models):
        self.includeSpuriousSignal = False
        self.spuriousSignal = {}
        self.signal = signals[channel]
        self.channel = channel
        self.models = models
        self.n = len(models)
        self.chi2s = {}
        self.ndfs = {}
        self.probs = {-1:float('nan')}
        self.correlatedfProbs = {0:float('nan')}
        self.uncorrelatedfProbs = {0:float('nan')}
        self.fProbs = {0:float('nan')}
        self.order = 5

        try:
            f.Get(self.channel).IsZombie()
        except ReferenceError:
            f.mkdir(self.channel)


        self.background_TF1s = {}
        #self.chi2s = {}
        #self.ndfs = {}
        self.chi2PerNdfs = {}
        #self.probs = {}
        self.parValues = {}
        self.parValue = None
        self.parErrors = {}
        self.ymaxs = {}
        self.uncorrelatedChi2s = {}
        self.correlatedChi2s = {}
        self.uncorrelatedNdfs = {}
        self.correlatedNdfs = {}
        self.uncorrelatedProbs = {}
        self.correlatedProbs = {}
        self.chi2, self.ndf, self.prob = None, None, None
        self.cUp, self.cDown = {}, {}

        self.localBins = models[0].nBins_rebin
        #self.nBins = (self.n+3) * self.localBins # use last set of models[0].nBins as error terms for per bin NPs
        self.nBins  = self.localBins * self.n # will fit all mix models simultaneously
        self.nBins += self.localBins          # use set of models[0].nBins as error terms for correlated stat error between mixes
        self.nBins +=   (maxOrder+1) * self.n # add extra bins for LP constraints in spurious signal fits
        self.data_obs = {}
        for order in range(maxOrder+1): # allow for fit using each order to have different uncorrelated uncertainties
            self.data_obs[order] = ROOT.TH1F('data_obs_order%d_ensemble'%order,'', self.nBins, 0.5, 0.5+self.nBins)
        self.ttbar    = ROOT.TH1F(   'ttbar_ensemble','', self.nBins, 0.5, 0.5+self.nBins)
        self.multijet = ROOT.TH1F('multijet_ensemble','', self.nBins, 0.5, 0.5+self.nBins)
        self.background_TF1 = None
        self.background_TH1s = {}#ROOT.TH1F("background_ensemble_TH1", "", self.nBins, 0.5, 0.5+self.nBins)
        self.background_spuriousSignal_TH1s = {}

        self.uM = 1.0 # scale of uncorrelated multijet stat uncertainty, only source comes from FvT weights, ie 4b stats in SB propagated through FvT training
        self.cM = 1.0 # scale of   correlated multijet stat uncertainty (correlated between models, comes from 3b data stats and 3b ttbar implicitly subtracted by FvT weight)
        self.cT = 1.0

        self.errorInflation = 1.0
        self.uM *= self.errorInflation
        self.cM *= self.errorInflation
        self.cT *= self.errorInflation

        self.model_ave = {}
        self.model_ave['None'] = ROOT.TH1F(self.models[0].background_TH1s['None'])
        self.model_ave['None'].SetName('background_ave_TH1')
        self.model_ave['None'].Scale(0)
        for model in self.models:        
            self.model_ave['None'].Add(model.background_TH1s['None'])
        self.model_ave['None'].Scale(1.0/self.n)
        f.cd(self.channel)
        self.model_ave['None'].Write()

        self.bin_centers = [self.models[0].data_obs.GetBinCenter(bin) for bin in range(1,self.localBins+1)]
        self.model_std = {}
        self.model_mean= {}
        for order in range(maxOrder+1):
            for model in self.models:
                model.fit(order)

            #                      underflow           std( array of TF1 Evals at a given bin center for each model )                for each bin center
            self.model_std[order] = [0.0] + [np.array([model.background_TF1s[order].Eval(x) for model in self.models]).std() for x in self.bin_centers]

            self.model_ave[order] = ROOT.TH1F(self.models[0].background_TH1s[order])
            self.model_ave[order].SetName('background_ave_TH1_order%d'%order)
            self.model_ave[order].Scale(0)
            for model in self.models:        
                self.model_ave[order].Add(model.background_TH1s[order])
            self.model_ave[order].Scale(1.0/self.n)
            f.cd(self.channel)
            self.model_ave[order].Write()

        # background_binErrors = [0]
        # for b in range(1,self.localBins+1):
        #     background_binErrors.append( (self.models[0].ttbar.GetBinError(b)**2+self.models[0].multijet.GetBinError(b)**2)**0.5 )
        #     contents = []
        #     x = self.models[0].data_obs.GetBinCenter(b)
        #     for model in self.models:
        #         contents.append(model.background_TF1s[1].Eval(x))
        #     contents = np.array(contents)
        #     self.model_std.append(contents.std()) # standard error of models in this bin should serve as a proxy for the uncorrelated multijet error (the correlated components are shared between models)
        #     self.model_mean.append(contents.mean())
        # self.model_std, self.model_mean = np.array(self.model_std), np.array(self.model_mean)
        # background_binErrors = np.array(background_binErrors)
        # print self.channel, self.model_std
        # print self.channel, background_binErrors
        # print self.channel, self.model_std/background_binErrors

        for m, model in enumerate(self.models):
            for b in range(1,model.nBins_rebin+1):
                multijetErrorScale = (model.data_obs.GetBinContent(b) - model.ttbar.GetBinContent(b)) / model.multijet.GetBinContent(b)
                for order in range(maxOrder+1):
                    self.data_obs[order].SetBinContent(m*model.nBins_rebin+b, model.data_obs.GetBinContent(b))
                    self.data_obs[order].SetBinError  (m*model.nBins_rebin+b,        self.model_std[order][b] * self.uM * multijetErrorScale) # put all uncorrelated stat error here
                self.multijet.SetBinContent(m*model.nBins_rebin+b, model.multijet.GetBinContent(b))
                self.multijet.SetBinError  (m*model.nBins_rebin+b, model.multijet.GetBinError  (b) * self.cM) # correlated stat error for multijet
                self.ttbar   .SetBinContent(m*model.nBins_rebin+b, model.ttbar   .GetBinContent(b))
                self.ttbar   .SetBinError  (m*model.nBins_rebin+b, model.ttbar   .GetBinError  (b) * self.cT) # correlated stat error for ttbar
        #for b in range(1, 3*self.localBins+1):
        f.cd(self.channel)
        for order in range(maxOrder+1):
            for b in range(self.localBins*self.n+1, self.nBins+1):
                self.data_obs[order].SetBinContent(b, 0.0)
                self.data_obs[order].SetBinError  (b, 1.0)
            self.data_obs[order].Write()
        self.ttbar.Write()
        self.multijet.Write()


    def plotBackgroundModels(self, order=None):
        if order is None: order = self.order
        samples=collections.OrderedDict()
        samples[closureFileName] = collections.OrderedDict()
        samples[closureFileName][self.models[0].data_obs_name] = {
            "label" : "Ave. Mix %.1f/fb"%lumi,
            "legend": 1,
            "isData" : True,
            #"ratio" : "numer A",
            #"drawOptions": 'PE ex0',
            "color" : "ROOT.kBlack"}
        if order is 'None':
            background = '%s/background_ave_TH1'%(self.channel)
            backgroundLabel = 'Ave. Model'
        else:
            background = '%s/background_ave_TH1_order%d'%(self.channel, order)
            backgroundLabel = "Ave. Fit (order %d)"%order
        samples[closureFileName][background] = {
            "label" : backgroundLabel,
            "legend": 2,
            "ratio" : "denom A",
            "lineColor" : "ROOT.kRed",
            "lineAlpha" : 1,
            "lineWidth" : 1,
            #"color" : "ROOT.kBlack",
            "drawOptions": 'HIST C',
        }
        for m, model in enumerate(self.models):
            if order is 'None':
                background = model.directory+"/background_TH1"
            else:
                background = model.directory+"/background_TH1_order%d"%order
            samples[closureFileName][background] = {
                #"label" : "Fit (order %d)"%order,
                #"legend": m+1,
                "ratio": "numer A", 
                "lineColor" : "ROOT.kRed",
                "lineAlpha" : 0.3,
                "lineWidth" : 1,
                "drawOptions": 'HIST C',
            }
            if not m: 
                samples[closureFileName][background]['label'] = 'Models' if order is 'None' else 'Fits'
                samples[closureFileName][background]['legend'] = 3

        # samples[closureFileName][self.models[0].ttbar_name] = {
        #     "label" : "t#bar{t}",
        #     "legend": 3,
        #     "stack" : 1,
        #     #"ratio" : "denom A",
        #     "color" : "ROOT.kAzure-9"}

        if "zz" in self.channel:
            xTitle = "SvB Regressed P(ZZ)+P(ZH), P(ZZ) > P(ZH)"
        if "zh" in self.channel:
            xTitle = "SvB Regressed P(ZZ)+P(ZH), P(ZH) #geq P(ZZ)"
            
        # if region=='SB':
        #     ymaxScale = 1.6
        # if region=='CR':
        #     ymaxScale = 1.6
        # if region=='SR':
        #     ymaxScale = (1.0 + self.order/6.0) if self.channel == 'zz' else max(1.6,1.2 + self.order/6.0)
        # if region=='SRNoHH':
        #     ymaxScale = 1.6

        parameters = {"titleLeft"   : "#bf{CMS} Internal",
                      "titleCenter" : regionName[region],
                      "titleRight"  : "Pass #DeltaR(j,j)",
                      "maxDigits"   : 4,
                      "ratio"     : True,
                      "rMin"      : 0.9,
                      "rMax"      : 1.1,
                      "rebin"     : rebin,
                      "rTitle"    : "Fit / Ave. Fit",
                      "xTitle"    : xTitle,
                      "yTitle"    : "Events",
                      #"yMax"      : self.ymaxs[order]*ymaxScale, # make room to show fit parameters
                      "xleg"      : [0.13, 0.13+0.33],
                      "outputName": 'models_order%s'%(str(order))}
        if type(rebin) is list:
            parameters["outputDir"] = 'closureFits/%s/variable_rebin/%s/%s/'%(mixName, region, self.channel)
        else:
            parameters["outputDir"] = 'closureFits/%s/rebin%i/%s/%s/'%(mixName, rebin, region, self.channel)
            
        print "make ",parameters["outputDir"]+parameters["outputName"]+".pdf"
        PlotTools.plot(samples, parameters, debug=False)


    def makeBackgrondTF1(self, order):

        def background_UserFunction(xArray, pars):
            ensembleBin = int(xArray[0])
            m = (ensembleBin-1)//self.localBins
            if m > self.n: # in extra bins used for LP constraints in spurious signal fits
                if not self.includeSpuriousSignal: return 0.0
                lpBin = ensembleBin-1-(self.n+1)*self.localBins

                if lpBin > self.n*(order+1)-1: return 0.0 # this bin is unused at this order

                # bins are arranged such that lp_0 of every model is listed together, then lp_1 of every model, etc
                m = lpBin%self.n
                lp_idx = lpBin//self.n
                par_idx = m*(order+1)+lp_idx
                lp_coefficient = pars[par_idx]
                if lp_coefficient > 0: # return lp_coefficient scaled by its up   uncertainty
                    return -lp_coefficient/abs(self.cUp  [order][lp_idx])
                else:                  # return lp_coefficient scaled by its down uncertainty
                    return -lp_coefficient/abs(self.cDown[order][lp_idx])

            localBin = ((ensembleBin-1)%self.localBins)+1
            pullIndex    = self.n*(order+1) + localBin-1
            localBinPull = pars[pullIndex]

            if m == self.n:
                return -localBinPull
            else:
                model = self.models[m]

                l, u = model.multijet.GetBinLowEdge(localBin), model.multijet.GetXaxis().GetBinUpEdge(localBin)
                w = model.multijet.GetBinWidth(localBin)

                p = 1.0                
                for lp_idx in range(order+1):
                    par_idx = m*(order+1)+lp_idx
                    p += pars[par_idx]*lp[lp_idx].Integral(l,u)/w

                #correlatedDataError     =    (self.data_obs.GetBinContent(ensembleBin)/self.n)**0.5 * self.errorInflation
                correlatedDataError     =    model.data_obs_raw.GetBinError(   localBin) * self.errorInflation
                correlatedttbarError    =     self.ttbar       .GetBinError(ensembleBin) 
                correlatedMultijetError = p * self.multijet    .GetBinError(ensembleBin) 
                correlatedPull = localBinPull * (correlatedDataError**2 + correlatedttbarError**2 + correlatedMultijetError**2)**0.5

                background = self.ttbar.GetBinContent(ensembleBin) + p*self.multijet.GetBinContent(ensembleBin) + correlatedPull
                spuriousSignal = 0
                if self.includeSpuriousSignal:
                    spuriousSignalIndex = m + self.n*(order+1)+self.localBins # one parameter per model after the LP coefficients and localBinPulls
                    
                    spuriousSignal = pars[spuriousSignalIndex] * self.signal.GetBinContent(localBin)
                
                return background + spuriousSignal

        #self.background_TF1s[order] = ROOT.TF1("background_ensemble_TF1_pol"+str(order), background_UserFunction, 0.5, 0.5+self.nBins, self.n*(order+1)+3*self.localBins)
        self.background_TF1s[order] = ROOT.TF1("background_ensemble_TF1_pol"+str(order), background_UserFunction, 0.5, 0.5+self.nBins, self.n*(order+1)+self.localBins+self.n)

        for m in range(self.n):
            for o in range(order+1):
                #print m*(order+1)+o, 'v%d c_%d'%(m, o)
                self.background_TF1s[order].SetParName  (m*(order+1)+o, 'v%d c_%d'%(m, o))
                self.background_TF1s[order].SetParameter(m*(order+1)+o, 0.0)
        #for b in range(3*self.localBins):
        for b in range(self.localBins):
            if b//self.localBins==0: item='data'
            if b//self.localBins==1: item='multijet'
            if b//self.localBins==2: item='ttbar'
            self.background_TF1s[order].SetParName  (self.n*(order+1)+b, 'pull %s bin %d'%(item,(b%self.localBins)+1))
            self.background_TF1s[order].SetParameter(self.n*(order+1)+b, 0.0)
        for m in range(self.n):
            self.background_TF1s[order].SetParName  (self.n*(order+1)+self.localBins+m, 'v%d spurious signal'%m)
            self.background_TF1s[order].FixParameter(self.n*(order+1)+self.localBins+m, 0)
            # self.background_TF1s[order].SetParameter(self.n*(order+1)+self.localBins+m, 0.0)
            # self.background_TF1s[order].SetParLimits(self.n*(order+1)+self.localBins+m, 1.0, 1.0)


    def storeFitResult(self, order=None):
        if order is None: order = self.order

        residuals = []
        for b in range(1, self.nBins+1):
            res = (self.data_obs[order].GetBinContent(b) - self.background_TF1s[order].Eval(b)) / self.data_obs[order].GetBinError(b) 
            residuals.append(res)
        residuals = np.array(residuals)
        print "residuals.mean()",residuals.mean()
        print "residuals.std() ",residuals.std()

        uncorrelatedResiduals = np.array( [residuals[m*self.localBins:(m+1)*self.localBins] for m in range(self.n)] )
        print "uncorrelatedResiduals.mean(axis=0)"
        print uncorrelatedResiduals.mean(axis=0)
        print "uncorrelatedResiduals .std(axis=0)"
        print uncorrelatedResiduals .std(axis=0)
        print "uncorrelatedResiduals.mean(axis=0) / uncorrelatedResiduals.std(axis=0)"
        print uncorrelatedResiduals.mean(axis=0) / uncorrelatedResiduals.std(axis=0)

        correlatedChi2 = 0.0
        for b in range(self.localBins*self.n+1, self.nBins+1):
            correlatedChi2 += ((self.background_TF1s[order].Eval(b)-self.data_obs[order].GetBinContent(b))/self.data_obs[order].GetBinError(b))**2
        print "correlatedChi2",correlatedChi2
        self.correlatedChi2s[order] = correlatedChi2
        self.correlatedNdfs[order] = self.localBins-order-1
        self.correlatedProbs[order] = scipy.stats.distributions.chi2.sf(self.correlatedChi2s[order], self.correlatedNdfs[order])

        self.uncorrelatedChi2s[order] = self.background_TF1s[order].GetChisquare() - self.correlatedChi2s[order]
        self.uncorrelatedNdfs[order] = (self.localBins-order-1)*self.n - self.localBins
        self.uncorrelatedProbs[order] = scipy.stats.distributions.chi2.sf(self.uncorrelatedChi2s[order], self.uncorrelatedNdfs[order])

        self.chi2s[order] = self.background_TF1s[order].GetChisquare() #self.correlatedChi2s[order] + self.uncorrelatedChi2s[order] #self.background_TF1s[order].GetChisquare()/self.n# + self.correlatedChi2s[order]
        self.ndfs[order] = self.n*(self.localBins-order-1) - self.localBins
        #self.background_TF1s[order].GetNDF() - self.localBins #self.correlatedNdfs[order] + self.uncorrelatedNdfs[order] #self.background_TF1s[order].GetNDF()/self.n #- 3*self.localBins
        self.chi2PerNdfs[order] = self.chi2s[order]/self.ndfs[order]

        #self.probs[order] = self.correlatedProbs[order] * self.uncorrelatedProbs[order]
        self.probs[order] = scipy.stats.distributions.chi2.sf(self.chi2s[order], self.ndfs[order])
        #self.probs[order] = self.background_TF1s[order].GetProb()

        nParam = self.background_TF1s[order].GetNumberFreeParameters()
        self.parValues[order] = np.array( [self.background_TF1s[order].GetParameter(i) for i in range(nParam)] )
        self.parErrors[order] = np.array( [self.background_TF1s[order].GetParError (i) for i in range(nParam)] )
        self.ymaxs[order] = self.background_TF1s[order].GetMaximum(1,self.nBins)

        self.background_TH1s[order] = ROOT.TH1F("background_ensemble_TH1_order%d"%order, "", self.nBins, 0.5, 0.5+self.nBins)

        for bin in range(1,self.nBins+1):
            c = self.background_TF1s[order].Eval(bin)
            self.background_TH1s[order].SetBinContent(bin, c)
            self.background_TH1s[order].SetBinError(bin, self.data_obs[order].GetBinError(bin))#0.001*c**0.5)

        f.cd(self.channel)
        self.background_TH1s[order].Write()
        print "exit storeFitResult"


    def dumpFitResult(self, order=None):
        if order is None: order = self.order
        print "    chi^2 =",self.chi2s[order]
        print "      ndf =",self.ndfs[order]
        print "chi^2/ndf =",self.chi2PerNdfs[order]
        print "  p-value =",self.probs[order]
        print "corChi^2/chi^2 = ",self.correlatedChi2s[order]/self.chi2s[order]


    def fit(self, order):
        for m in self.models:
            if order not in m.chi2s:
                print "\n>>> ", order, m.name
                m.makeBackgrondTF1(order)
                m.fit(order)

        if order not in self.background_TF1s:
            self.makeBackgrondTF1(order)
            self.data_obs[order].Fit(self.background_TF1s[order], "N0")
            self.storeFitResult(order)
            self.dumpFitResult(order)
            self.getParameterDistribution(order)
            if doSpuriousSignal:
                self.fitSpuriousSignal(order)

    def fitSpuriousSignal(self, order=None):
        if order is None: order = self.order
        self.includeSpuriousSignal = True
        for m in range(self.n):
            parNumber = self.background_TF1s[order].GetParNumber('v%d spurious signal'%m)
            self.background_TF1s[order].SetParameter(parNumber, 0.0)
            self.background_TF1s[order].SetParLimits(parNumber, -100.0, 100.0)
        self.data_obs[order].Fit(self.background_TF1s[order], "N0")
        
        self.background_spuriousSignal_TH1s[order] = ROOT.TH1F("background_spuriousSignal_ensemble_TH1_order%d"%order, "", self.nBins, 0.5, 0.5+self.nBins)
        for bin in range(1,self.nBins+1):
            c = self.background_TF1s[order].Eval(bin)
            self.background_spuriousSignal_TH1s[order].SetBinContent(bin, c)
            self.background_spuriousSignal_TH1s[order].SetBinError(bin, self.data_obs[order].GetBinError(bin))#0.001*c**0.5)
        f.cd(self.channel)
        self.background_spuriousSignal_TH1s[order].Write()

        self.includeSpuriousSignal = False
        self.spuriousSignal[order] = []
        for m in range(self.n):
            parNumber = self.background_TF1s[order].GetParNumber('v%d spurious signal'%m)
            self.spuriousSignal[order].append( self.background_TF1s[order].GetParameter(parNumber) )
            self.background_TF1s[order].FixParameter(parNumber, 0.0)
        self.spuriousSignal[order] = np.array( self.spuriousSignal[order] )
        print 'spuriousSignal',self.spuriousSignal[order]
        print '          mean',self.spuriousSignal[order].mean()
        print '           std',self.spuriousSignal[order].std()
        print "exit fitSpuriousSignal"
        
        
    # def write(self):
    #     try:
    #         f.Get(self.channel).IsZombie()
    #     except ReferenceError:
    #         f.mkdir(self.channel)

    #     f.cd(self.channel)
    #     self.ttbar.Write()
    #     self.multijet.Write()
 
    def combinedFTest(self, order2, order1):

        for order in [order1, order2]:
            # self.chi2s[order], self.ndfs[order] = 0, 0
            self.fit(order)
        for m in self.models: m.fTest(order2, order1)

        # for order in [order1, order2]:
        #     if self.ndfs[order]: continue
        #     for m in self.models:
        #         #expectedchi2 = scipy.stats.distributions.chi2.isf(0.95, m.ndfs[order])
        #         self.chi2s[order] += m.chi2s[order] #- expectedchi2 * (self.n-1.0)/self.n # only want to count the correlated stat component once
        #         #self.ndfs[order] += m.ndfs[order]
        #     self.ndfs[order] = self.models[0].ndfs[order]*(self.n)
        #     expectedchi2 = scipy.stats.distributions.chi2.isf(0.95, self.models[0].ndfs[order]) # only want to count the correlated stat component once
        #     print "Expected chi2/ndf at 95%% for order %i = %f"%(order, expectedchi2/self.models[0].ndfs[order])
        #     print "Average observed chi2/ndf = ",self.chi2s[order]/self.ndfs[order]
        #     self.chi2s[order] -= expectedchi2 * (self.n-1)
            
        #     self.probs[order] = scipy.stats.distributions.chi2.sf(self.chi2s[order], self.ndfs[order])

        print "\n"+"#"*10
        for order in [order1, order2]:
            print " Order %i:"%order
            print "  chi2/ndf = ", self.chi2s[order]/self.ndfs[order]
            print "  cor prob = %3.1f%%"%(100*self.correlatedProbs[order])
            print "uncor prob = %3.1f%%"%(100*self.uncorrelatedProbs[order])
            print "      prob = %3.1f%%"%(100*self.probs[order])

        # d1 = (self.ndfs[order1]-self.ndfs[order2])
        # d2 = self.ndfs[order2]
        # N = (self.chi2s[order1]-self.chi2s[order2])/d1
        # D = self.chi2s[order2]/d2
        # fStat = N/D # / self.n
        # fProb = scipy.stats.f.cdf(fStat, d1, d2)
        # expectedFStat = scipy.stats.distributions.f.isf(0.05, d1, d2)
        print "-"*10
        # print "d1, d2 = %d, %d"%(d1, d2)
        # print "N, D = %f, %f"%(N, D)
        # print "    f(%i,%i) = %f (expected at 95%%: %f)"%(d1,d2,fStat,expectedFStat)
        # print "f.cdf(%i,%i) = %2.0f%%"%(d1,d2,100*fProb)
        print "Correlated F-Test"
        self.correlatedfProbs[order2] = fTest(self.correlatedChi2s[order1], self.correlatedChi2s[order2], self.correlatedNdfs[order1], self.correlatedNdfs[order2])
        print "Uncorrelated F-Test"
        self.uncorrelatedfProbs[order2] = fTest(self.uncorrelatedChi2s[order1], self.uncorrelatedChi2s[order2], self.uncorrelatedNdfs[order1], self.uncorrelatedNdfs[order2])
        print "Full F-Test"
        #self.fProbs[order2] = fTest(self.chi2s[order1], self.chi2s[order2], self.ndfs[order1], self.ndfs[order2])
        self.fProbs[order2] = self.correlatedfProbs[order2] * self.uncorrelatedfProbs[order2]
        print "              = %3.0f%%"%(100*self.fProbs[order2])
        print "#"*10
        if self.fProbs[order2] < 0.95: self.fTestSatisfied = True
        self.fTestSatisfied = True

        if not self.done and (self.probs[order1] > probThreshold) and self.fTestSatisfied:
            print "Done"
            self.done = True
            self.order = order1
            self.prob = self.probs[order1]
            self.fProb = self.fProbs[order2]

    def setOrder(self, order=None):
        if order is None: 
            order = self.order
        else:
            self.order = order

        self.ymax = max([m.ymaxs[order] for m in self.models])

        for m in self.models:
            m.setOrder(order)
            m.ymax = self.ymax
            
        self.background_TF1 = self.background_TF1s[order]
        self.parValue = self.parValues[order] #np.array([self.parValues[order][i] for i in range(self.n*(self.order+1)+self.localBins)], dtype=np.float) #self.parValues[order]
        self.chi2 = self.chi2s[self.order]
        self.ndf = self.ndfs[self.order]
        self.prob = self.probs[self.order]

    def runCombinedFTest(self):
        # for order in range(6):
        #     for m in self.models:
        #         if order not in m.chi2s:
        #             print "\n>>> ", order, m.name
        #             m.makeBackgrondTF1(order)
        #             m.fit(order)

        self.fTestSatisfied = False
        self.done = False
        order = -1
        while order < maxOrder-1:# and (self.fProb > 0.95 or self.prob < 0.05):
            order += 1
            self.combinedFTest(order+1, order)
            #self.fitSpuriousSignal(order)

        if not self.done:
            if self.probs[maxOrder] > probThreshold and self.fTestSatisfied:
                self.done = True
                self.order = maxOrder
                self.prob = self.probs[maxOrder]
                self.fProb = -0.99

        # for order in range(maxOrder+1):
        #     self.setOrder(order)
        #     self.fitAllMixes()

        if self.done:
            print
            print "Combined F-Test prefers order %i (%3.0f%%) over %i (%3.0f%%) at      %3.0f%% confidence"      %(self.order  , 100*self.probs[self.order  ], self.order-1, 100*self.probs[self.order-1], 100*self.fProbs[self.order])
            try:
                print "       while it prefers order %i (%3.0f%%) over %i (%3.0f%%) at only %3.0f%% so keep order %i"%(self.order+1, 100*self.probs[self.order+1], self.order  , 100*self.probs[self.order  ], 100*self.fProb, self.order)
            except:
                pass
        else:
            print
            print "Failed to satisfy F-Test and/or goodness of fit."
            print "pvalues:",self.probs
            for o in range(maxOrder+1):
                if self.probs[o]>probThreshold and not self.done:
                    self.order = o
                    self.done = True
            if not self.done:
                self.order = max(self.probs.iteritems(), key=operator.itemgetter(1))[0]
    
        self.setOrder()
        # for m in self.models:
        #     m.write()
        #self.write()
        self.writeClosureResults()
        #self.getParameterDistribution()

    def fitAllMixes(self):
        for m in self.models: 
            m.fitAllMixes()
            
        fig, (ax) = plt.subplots(nrows=1)

        ax.set_title("Fit to average of mixes (%s channel)"%self.channel)
        ax.set_xlabel('$\chi^2$')
        ax.set_ylabel('Arb. Units')

        chi2s = []
        markers = ['o', 'v', '^', '<', '>', 's', '*']
        colors = ['b','g','c','m']
        for i, m in enumerate(self.models): 
            chi2s.append(m.chi2)
            
            kwargs = {'lw': 1,
                      'marker': markers[i%len(markers)],
                      'edgecolors': 'k',
                      'color': colors[i%len(colors)],
                      'label': m.name.replace("_","\_"),
                      }

            plt.scatter(chi2s[-1], 0.0, **kwargs)

        chi2s = np.array(chi2s)
        print "chi2s.mean():",chi2s.mean(), "chi2s.std():",chi2s.std()


        x = np.linspace(scipy.stats.chi2.ppf(0.001, self.models[0].ndf), scipy.stats.chi2.ppf(0.999, self.models[0].ndf), 100)
        ax.plot(x, scipy.stats.chi2.pdf(x, self.models[0].ndf), 'r-', lw=2, alpha=1, label='$\chi^2$ PDF (NDF = %d)'%self.models[0].ndf)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim([-0.005, ylim[1]])
        
        ax.legend(loc='upper right', fontsize='small', scatterpoints=1)

        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/chi2s_order%i.pdf'%(mixName, region, self.channel, self.order)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/chi2s_order%i.pdf'%(mixName, rebin, region, self.channel, self.order)

        print "fig.savefig( "+name+" )"
        fig.savefig( name )
        plt.close(fig)

    def getParameterDistribution(self, order=None):
        if order is None: order = self.order
        nModels = len(self.models)
        nLPs = order+1
        parMean  = np.array([0 for i in range(nLPs)], dtype=np.float)
        parMean2 = np.array([0 for i in range(nLPs)], dtype=np.float)
        # for m in self.models:
        #     parMean  += m.parValue    / nModels
        #     parMean2 += m.parValue**2 / nModels
        for m in range(nModels):
            # print(self.parValue)
            # print(m,m*n,(m+1)*n,self.parValue[m*n:(m+1)*n])
            parMean  += self.parValues[order][m*nLPs:(m+1)*nLPs]    / nModels
            parMean2 += self.parValues[order][m*nLPs:(m+1)*nLPs]**2 / nModels
        var = parMean2 - parMean**2
        parStd = var**0.5
        print "Parameter Mean:",parMean
        print "Parameter  Std:",parStd

        for i in range(nLPs):
            cUp   =  (abs(parMean[i])+parStd[i]) * (order+1)**0.5 # add scaling term so that 1 sigma corresponds to quadrature sum over i of (abs(parMean[i])+parStd[i])
            cDown = -cUp
            try:
                self.cUp  [order].append( cUp )
                self.cDown[order].append( cDown )
            except KeyError:
                self.cUp  [order] = [cUp  ]
                self.cDown[order] = [cDown]


    def writeClosureResults(self,order=None):
        if order is None: order = self.order
        nLPs = order+1
        closureResults = "ZZ4b/nTupleAnalysis/combine/closureResults_%s_order%d.txt"%(self.channel,order)
        closureResultsFile = open(closureResults, 'w')
        for i in range(nLPs):
            cUp   = self.cUp  [order][i]
            cDown = self.cDown[order][i]
            systUp    = "1+"
            systDown  = "1+"
            systUp   += "(%f)*(%s)"%(cUp,   lps[i].replace(' ',''))
            systDown += "(%f)*(%s)"%(cDown, lps[i].replace(' ',''))
            systUp    = "multijet_LP%i_%sUp   %s"%(i, channel, systUp)
            systDown  = "multijet_LP%i_%sDown %s"%(i, channel, systDown)
            print systUp
            print systDown
            closureResultsFile.write(systUp+'\n')
            closureResultsFile.write(systDown+'\n')
        if doSpuriousSignal:
            ssUp   = abs(self.spuriousSignal[order].mean())+self.spuriousSignal[order].std()
            ssDown = -ssUp
            systUp   = 'multijet_spurious_signal_%sUp   %f'%(channel, ssUp)
            systDown = 'multijet_spurious_signal_%sDown %f'%(channel, ssDown)
            print systUp
            print systDown
            closureResultsFile.write(systUp+'\n')
            closureResultsFile.write(systDown+'\n')        
        closureResultsFile.close()

    def plotFitResultsByOrder(self):
        fig, (ax) = plt.subplots(nrows=1)
        ax.set_ylim(0.001,1)
        ax.set_xticks(range(len(self.probs)))
        plt.yscale('log')

        # x = sorted(self.fProbs.keys())
        # y = [self.fProbs[o] for o in x]
        # ax.plot(x, y, label='', color='r', alpha=0.5, linestyle='--', linewidth=2)

        x = sorted(self.probs.keys())
        y = [self.probs[o] for o in x]
        ax.plot(x, y, label='Combined', color='r', linewidth=2)
        
        colors=['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:cherry', 'xkcd:bright red',
                'xkcd:pine', 'xkcd:magenta', 'xkcd:cerulean', 'xkcd:eggplant', 'xkcd:coral', 'xkcd:blue purple']
        for i, m in enumerate(self.models):
            x = sorted(m.probs.keys())
            y = [m.probs[o] for o in x]
            ax.plot(x, y, label=m.name.replace('_','\_'), alpha=0.5, color=colors[i])
            # x = x[1:]
            # y = [m.fProbs[o] for o in x]
            # ax.plot(x, y, alpha=0.5, linestyle='--', color=colors[i])
            
        maxOrder = x[-1]
        ax.plot([0,maxOrder], [probThreshold,probThreshold], color='k', alpha=0.5, linestyle='--', linewidth=1)
        #ax.plot([0,maxOrder], [0.95,0.95], color='k', alpha=0.5, linestyle='--', linewidth=1)

        ax.set_xlabel('Polynomial Order')
        ax.set_ylabel('Fit p-value')
        ax.legend(loc='lower right', fontsize='small')

        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/pvalues.pdf'%(mixName, region, self.channel)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/pvalues.pdf'%(mixName, rebin, region, self.channel)
        print "fig.savefig( "+name+" )"
        fig.savefig( name )
        plt.close(fig)

    def plotFitResults(self, order=None):
        if order is None: order = self.order
        n = order+1

        x,y,s,c = [],[],[],[]
        x = np.concatenate( [m.eigenVars[order][0] for m in self.models] )
        if n==1:
            y = np.array([[0] for m in self.models])
        if n>1: 
            y = np.concatenate( [m.eigenVars[order][1] for m in self.models] )
        if n>2:
            c = np.concatenate( [m.eigenVars[order][2] for m in self.models] )
        if n>3:
            s = np.concatenate( [m.eigenVars[order][3] for m in self.models] )

        kwargs = {'lw': 1,
                  'marker': 'o',
                  'edgecolors': 'k',
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
            s = (s+10.0/30)*30 #shift and scale so that min is 10.0 and max is 30+10.0
            kwargs['s'] = s

        fig, (ax) = plt.subplots(nrows=1)
        ax.set_title('Fit Eigen-Variations: '+self.channel)
        ax.set_xlabel('c$_0$')
        ax.set_ylabel('c$_1$')
        
        plt.scatter(x, y, **kwargs)
        if 'c' in kwargs:
            cbar = plt.colorbar(label='c$_2$',
                                #use_gridspec=False, location="top"
                                ) 

        if 's' in kwargs:
            #lt = plt.scatter([],[], s=0,    lw=0, edgecolors='none',  facecolors='none')
            l1 = plt.scatter([],[], s=(0.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
            l2 = plt.scatter([],[], s=(1.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
            l3 = plt.scatter([],[], s=(2.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
            l4 = plt.scatter([],[], s=(3.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')

            handles = [#lt,
                       l1,
                       l2,
                       l3,
                       l4]
            labels = [#"p$_3$",
                      "%0.2f"%smin,
                      "%0.2f"%(smin+srange*1.0/3),
                      "%0.2f"%(smin+srange*2.0/3),
                      "%0.2f"%smax,
                     ]

            leg = plt.legend(handles, labels, 
                             ncol=1, 
                             fontsize='medium',
                             #frameon=False, #fancybox=False, edgecolor='black',
                             #markerfirst=False,
                             #bbox_to_anchor=(1.22, 1), loc='upper left',
                             loc='best',
                             #handlelength=0.8, #handletextpad=1, #borderpad = 0.5,
                             title='c$_3$', 
                             #title_fontsize='small',
                             #columnspacing=1.8,
                             #labelspacing=1.5,
                             scatterpoints = 1, 
                             #scatteryoffsets=[0.5],
                             )

        
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.plot([0,0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/eigenvariations.pdf'%(mixName, region, self.channel)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/eigenvariations.pdf'%(mixName, rebin, region, self.channel)
        print "fig.savefig( "+name+" )"
        fig.savefig( name )
        plt.close(fig)


        #plot fit parameters
        x,y,s,c = [],[],[],[]
        for m in self.models:
            x.append( m.parValues[order][0] )
            if n==1:
                y.append( 0 )
            if n>1:
                y.append( m.parValues[order][1] )
            if n>2:
                c.append( m.parValues[order][2] )
            if n>3:
                s.append( m.parValues[order][3] )

        x = np.array(x)
        y = np.array(y)
    
        kwargs = {'lw': 1,
                  'marker': 'o',
                  'edgecolors': 'k',
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
            s = (s+10.0/30)*30 #shift and scale so that min is 10.0 and max is 30+10.0
            kwargs['s'] = s

        fig, (ax) = plt.subplots(nrows=1)
        ax.set_aspect(1)
        ax.set_title('Legendre Polynomial Coefficients: '+self.channel)
        ax.set_xlabel('c$_0$')
        ax.set_ylabel('c$_1$')

        maxr=np.zeros((2, len(x)), dtype=np.float)
        minr=np.zeros((2, len(x)), dtype=np.float)
        if n>1:
            #generate a ton of random points on a hypersphere in dim=n so surface is dim=n-1.
            points  = np.random.uniform(0,1,  (n, min(100**(n-1),10**7) )  ) # random points in a hypercube
            points /= np.linalg.norm(points, axis=0) # normalize them to the hypersphere surface

            # for each model, find the point which maximizes the change in c_0**2 + c_1**2
            for i, m in enumerate(self.models):
                eigenVars = m.eigenVars[order]
                plane = np.matmul( eigenVars[0:min(n,2),:], points )
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
            for i, m in enumerate(self.models):
                maxr[0,i] = m.eigenVars[order][0]

        print maxr
        print minr
        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002)

        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002)

        
        plt.scatter(x, y, **kwargs)


        if 'c' in kwargs:
            cbar = plt.colorbar(label='c$_2$',
                                #use_gridspec=False, location="top"
                                ) 

        if 's' in kwargs:
            #lt = plt.scatter([],[], s=0,    lw=0, edgecolors='none',  facecolors='none')
            l1 = plt.scatter([],[], s=(0.0/3+10.0/30)*30,  lw=1, edgecolors='black', facecolors='none')
            l2 = plt.scatter([],[], s=(1.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
            l3 = plt.scatter([],[], s=(2.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
            l4 = plt.scatter([],[], s=(3.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')

            handles = [#lt,
                       l1,
                       l2,
                       l3,
                       l4]
            labels = [#"p$_3$",
                      "%0.2f"%smin,
                      "%0.2f"%(smin+srange*1.0/3),
                      "%0.2f"%(smin+srange*2.0/3),
                      "%0.2f"%smax,
                     ]

            leg = plt.legend(handles, labels, 
                             ncol=1, 
                             fontsize='medium',
                             #frameon=False, #fancybox=False, edgecolor='black',
                             #markerfirst=False,
                             #bbox_to_anchor=(1.22, 1), loc='upper left',
                             loc='best',
                             #handlelength=0.8, #handletextpad=1, #borderpad = 0.5,
                             title='c$_3$', 
                             #title_fontsize='small',
                             #columnspacing=1.8,
                             #labelspacing=1.5,
                             scatterpoints = 1, 
                             #scatteryoffsets=[0.5],
                             )

        
        #xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = [-0.1,0.1], [-0.1,0.1]
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.plot([0,0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if type(rebin) is list:
            name = 'closureFits/%s/variable_rebin/%s/%s/fitParameters.pdf'%(mixName, region, self.channel)
        else:
            name = 'closureFits/%s/rebin%i/%s/%s/fitParameters.pdf'%(mixName, rebin, region, self.channel)            
        print "fig.savefig( "+name+" )"
        fig.savefig( name )
        plt.close(fig)


        # x,y,s,c = [],[],[],[]
        # for m in self.models:
        #     x.append( m.eigenPars[order][0] )
        #     if n>1:
        #         y.append( m.eigenPars[order][1] )
        #     if n>2:
        #         c.append( m.eigenPars[order][2] )
        #     if n>3:
        #         s.append( m.eigenPars[order][3] )

        # kwargs = {'lw': 1,
        #           'marker': 'o',
        #           'edgecolors': 'k',
        #           }
        # if n>2:
        #     kwargs['c'] = c
        #     kwargs['cmap'] = 'BuPu'
        # if n>3:
        #     s = np.array(s)
        #     smin = s.min()
        #     smax = s.max()
        #     srange = smax-smin
        #     s = s-s.min() #shift so that min is at zero
        #     s = s/s.max() #scale so that max is 1
        #     s = (s+10.0/30)*30 #shift and scale so that min is 10.0 and max is 30+10.0
        #     kwargs['s'] = s

        # fig, (ax) = plt.subplots(nrows=1)
        # ax.set_title('Fit Parameters (Eigen-Basis): '+channel)
        # ax.set_xlabel('e$_0$')
        # ax.set_ylabel('e$_1$')
        
        # plt.scatter(x, y, **kwargs)
        # if 'c' in kwargs:
        #     cbar = plt.colorbar(label='e$_2$',
        #                         ticks=np.linspace(0.25, -1, 1, endpoint=True)
        #                         #use_gridspec=False, location="top"
        #                         ) 

        # if 's' in kwargs:
        #     #lt = plt.scatter([],[], s=0,    lw=0, edgecolors='none',  facecolors='none')
        #     l1 = plt.scatter([],[], s=(0.0/3+10.0/30)*30,  lw=1, edgecolors='black', facecolors='none')
        #     l2 = plt.scatter([],[], s=(1.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
        #     l3 = plt.scatter([],[], s=(2.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')
        #     l4 = plt.scatter([],[], s=(3.0/3+10.0/30)*30, lw=1, edgecolors='black', facecolors='none')

        #     handles = [#lt,
        #                l1,
        #                l2,
        #                l3,
        #                l4]
        #     labels = [#"p$_3$",
        #               "%0.2f"%smin,
        #               "%0.2f"%(smin+srange*1.0/3),
        #               "%0.2f"%(smin+srange*2.0/3),
        #               "%0.2f"%smax,
        #              ]

        #     leg = plt.legend(handles, labels, 
        #                      ncol=1, 
        #                      fontsize='medium',
        #                      #frameon=False, #fancybox=False, edgecolor='black',
        #                      #markerfirst=False,
        #                      #bbox_to_anchor=(1.22, 1), loc='upper left',
        #                      loc='best',
        #                      #handlelength=0.8, #handletextpad=1, #borderpad = 0.5,
        #                      title='e$_3$', 
        #                      #title_fontsize='small',
        #                      #columnspacing=1.8,
        #                      #labelspacing=1.5,
        #                      scatterpoints = 1, 
        #                      #scatteryoffsets=[0.5],
        #                      )

        
        # xlim, ylim = ax.get_xlim(), ax.get_ylim()
        # xlim, ylim = [-1,1], [-1,1]
        # ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
        # ax.plot([0,0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        
        # name = 'closureFits/%s/fitParameters_eigenbasis_'+channel+'.pdf'
        # print "fig.savefig( "+name+" )"
        # fig.savefig( name )

    def plotFit(self, order=None, plotSpuriousSignal=False):
        if order is None: order = self.order
        samples=collections.OrderedDict()
        samples[closureFileName] = collections.OrderedDict()
        samples[closureFileName][self.channel+"/data_obs_order%d_ensemble"%order] = {
            "label" : ("Mixed Data %.1f/fb")%(lumi),
            "legend": 1,
            "isData" : True,
            "ratio" : "numer A",
            "color" : "ROOT.kBlack"}
        samples[closureFileName][self.channel+"/multijet_ensemble"] = {
            "label" : "Multijet Models",
            "legend": 2,
            "stack" : 3,
            "ratio" : "denom A",
            "color" : "ROOT.kYellow"}
        samples[closureFileName][self.channel+"/ttbar_ensemble"] = {
            "label" : "t#bar{t}",
            "legend": 3,
            "stack" : 2,
            "ratio" : "denom A",
            "color" : "ROOT.kAzure-9"}
        samples[closureFileName][self.channel+"/background_ensemble_TH1_order%d"%order] = {
            "label" : "Fit (order %d)"%order,
            "legend": 4,
            "ratio": "denom A", 
            "color" : "ROOT.kRed"}
        if doSpuriousSignal and plotSpuriousSignal:
            samples[closureFileName][self.channel+"/background_spuriousSignal_ensemble_TH1_order%d"%order] = {
                "label" : "Fit Spurious Signal #mu=%1.1f#pm%1.1f"%(self.spuriousSignal[order].mean(), self.spuriousSignal[order].std()),
                "legend": 5,
                "ratio": "denom A", 
                "color" : "ROOT.kBlue"}

        if "zz" in self.channel:
            xTitle = "SvB P(ZZ)+P(ZH) Bin, P(ZZ) > P(ZH)"
        if "zh" in self.channel:
            xTitle = "SvB P(ZZ)+P(ZH) Bin, P(ZH) #geq P(ZZ)"
            
        # if region=='SB':
        #     ymaxScale = 1.6
        # if region=='CR':
        #     ymaxScale = 1.6
        # if region=='SR':
        #     ymaxScale = (1.0 + self.order/6.0) if self.channel == 'zz' else max(1.6,1.2 + self.order/6.0)

        xLPPulls = (self.n+1)*self.localBins+0.5
        parameters = {"titleLeft"   : "#bf{CMS} Internal",
                      "titleCenter" : regionName[region],
                      "titleRight"  : "Pass #DeltaR(j,j)",
                      "maxDigits"   : 4,
                      "drawLines"   : [[self.localBins*m+0.5, 0,self.localBins*m+0.5,self.ymax*1.1] for m in range(1,self.n+1)],# + [[xLPPulls+o*self.n,  0, xLPPulls+o*self.n, self.ymax*1.1] for o in range(1,order+2)],
                      "ratioLines"  : [[self.localBins*m+0.5,-5,self.localBins*m+0.5,        5    ] for m in range(1,self.n+2)] + [[xLPPulls+o*self.n, -5, xLPPulls+o*self.n, 5]             for o in range(1,order+2)],
                      "ratioErrors": False,
                      "ratio"     : "significance",#True,
                      "rMin"      : -5,#0.9,
                      "rMax"      : 5,#1.1,
                      #"rebin"     : 1,
                      "rTitle"    : "Residuals",#"Data / Bkgd.",
                      "xTitle"    : xTitle,
                      "yTitle"    : "Events",
                      "yMax"      : self.ymax*1.7,#*ymaxScale, # make room to show fit parameters
                      #"xMax"      : self.localBins*(self.n+1)+0.5,
                      "xleg"      : [0.13, 0.13+0.33],
                      "legendSubText" : ["#bf{Fit:}",
                                         "#chi^{2}/DoF = %2.1f/%d = %1.2f"%(self.chi2s[order],self.ndfs[order],self.chi2s[order]/self.ndfs[order]),
                                         "p-value = %2.0f%%"%(self.probs[order]*100),
                                         ],
                      "lstLocation" : "right",
                      "outputName": 'fit_ensemble%s_order%d'%('_spuriousSignal' if plotSpuriousSignal else '', order)}

        if type(rebin) is list:
            parameters["outputDir"] = 'closureFits/%s/variable_rebin/%s/%s/'%(mixName, region, self.channel)
        else:
            parameters["outputDir"] = 'closureFits/%s/rebin%i/%s/%s/'%(mixName, rebin, region, self.channel)

        if not doSpuriousSignal or not plotSpuriousSignal:
            parameters['xMax'] = self.localBins*(self.n+1)+0.5
        # for i in range(self.order+1):
        #     percentError = 100*self.parError[i]/abs(self.parValue[i])
        #     if percentError > 100:
        #         errorString = ">100"
        #     else:
        #         errorString = "%4.0f"%percentError
        #     #parameters["legendSubText"] += ["#font[82]{c_{%i} =%6.3f #pm%s%%}"%(i, self.parValue[i], errorString)]
        #     parameters["legendSubText"] += ["#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma"%(i, self.parValue[i]*100, abs(self.parValue[i])/self.parError[i])]

        # parameters["legendSubText"] += ["",
        #                                 "#chi^{2}/DoF = %0.2f"%(self.chi2/self.ndf),
        #                                 "p-value = %2.0f%%"%(self.prob*100)]

        print "make ",parameters["outputDir"]+parameters["outputName"]+".pdf"
        PlotTools.plot(samples, parameters, debug=False)


models = {}
ensembles = {}

for channel in channels:    
    models[channel] = []
    if type(rebin) is list:
        mkpath("%s/closureFits/%s/variable_rebin/%s/%s"%(basePath, mixName, region, channel))
    else:
        mkpath("%s/closureFits/%s/rebin%i/%s/%s"%(basePath, mixName, rebin, region, channel))
    for i, mix in enumerate(mixes):
        if dataAverage:
            if ttAverage: models[channel].append( model(mix+"/"+channel, data_obs_name=channel+"/data_obs", ttbar_name=channel+"/ttbar", mix=mix, channel=channel) )
            else:         models[channel].append( model(mix+"/"+channel, data_obs_name=channel+"/data_obs", mix=mix, channel=channel) )
        else:
            if ttAverage: models[channel].append( model(mix+"/"+channel, ttbar_name=channel+"/ttbar", mix=mix, channel=channel) )
            else:         models[channel].append( model(mix+"/"+channel, mix=mix, channel=channel) )

# for channel in channels:    
#     for i, mix in enumerate(mixes):
#         for order in range(maxOrder): 
#             models[channel][i].fit(order)
# f.Close()
# for channel in channels:    
#     for i, mix in enumerate(mixes):
#         for order in range(maxOrder): 
#             models[channel][i].plotFit(order)
# exit()

for channel in channels:    
    print "-"*20
    print "Combined F-Test:", channel
    ensembles[channel] = ensemble(channel, models[channel])
    ensembles[channel].runCombinedFTest()
    print "-"*20
    ensembles[channel].plotFitResultsByOrder()
    ensembles[channel].plotFitResults()
    #ensembles[channel].fitAllMixes()

for channel in channels:
    for order in range(maxOrder+1):
        ensembles[channel].writeClosureResults(order)

f.Close()

for channel in channels:
    ensembles[channel].plotBackgroundModels('None')
    for order in range(3):
        ensembles[channel].plotBackgroundModels(order)
    for order in range(maxOrder+1):
        ensembles[channel].plotFit(order, False)
        ensembles[channel].plotFit(order, True)
    for i, mix in enumerate(mixes):
        models[channel][i].plotFit()


