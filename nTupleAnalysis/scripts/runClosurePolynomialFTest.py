import ROOT
ROOT.gROOT.SetBatch(True)
#ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2")
import sys
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import collections
import PlotTools
from array import array
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

year = "RunII"
lumi = 132.6
rebin = 4

closureFileName = "ZZ4b/nTupleAnalysis/combine/hists_closure.root"

f=ROOT.TFile(closureFileName, "UPDATE")

mixes = ["3bMix4b_rWbW2_v0",
         "3bMix4b_rWbW2_v1",
         "3bMix4b_rWbW2_v2",
         "3bMix4b_rWbW2_v3",
         "3bMix4b_rWbW2_v4",
         "3bMix4b_rWbW2_v5",
         "3bMix4b_rWbW2_v6",
         ]
channels = ["zz",
            "zh",
            ]

def addYears(directory):
    hists = []
    for process in ["ttbar","multijet","data_obs"]:
        hists.append( f.Get(directory+"2016/"+process) )
        hists[-1].SetName(process)
        hists[-1].Add(f.Get(directory+"2017/"+process))
        hists[-1].Add(f.Get(directory+"2018/"+process))
        f.mkdir(directory)
        f.cd(directory)
        hists[-1].Write()

for mix in mixes:
    for channel in channels:
        addYears(mix+'/'+channel)

f.Close()
f=ROOT.TFile(closureFileName, "UPDATE")

# lpx = ROOT.TF1("lpx", "2*(x-0.5)", 0, 1)
# lp0 = ROOT.TF1("lp0", "1", -1, 1)
# lp1 = ROOT.TF1("lp1", "x", -1, 1)
# lp2 = ROOT.TF1("lp2", "0.5  *(        3*x^2-1)", -1, 1)
# lp3 = ROOT.TF1("lp3", "0.5  *(        5*x^3-3*x)", -1, 1)
# lp4 = ROOT.TF1("lp4", "0.125*(35*x^4-30*x^2+3)", -1, 1)
# lp5 = ROOT.TF1("lp5", "0.125*(63*x^5-70*x^3+15*x)", -1, 1)
lps = [                                     "1",
                                        "2*x-1",
                                 "6*x^2- 6*x+1",
                        "20*x^3- 30*x^2+12*x-1",
                "70*x^4-140*x^3+ 90*x^2-20*x+1",
       "252*x^5-630*x^4+560*x^3-210*x^2+30*x-1",
       ]
lp = []
for i, s in enumerate(lps): 
    lp.append( ROOT.TF1("lp%i"%i, s, 0, 1) )
# lp0 = ROOT.TF1("lp0",                                      "1", 0, 1)
# lp1 = ROOT.TF1("lp1",                                  "2*x-1", 0, 1)
# lp2 = ROOT.TF1("lp2",                           "6*x^2- 6*x+1", 0, 1)
# lp3 = ROOT.TF1("lp3",                  "20*x^3- 30*x^2+12*x-1", 0, 1)
# lp4 = ROOT.TF1("lp4",          "70*x^4-140*x^3+ 90*x^2-20*x+1", 0, 1)
# lp5 = ROOT.TF1("lp5", "252*x^5-630*x^4+560*x^3-210*x^2+30*x-1", 0, 1)
# lp = [lp0, lp1, lp2, lp3, lp4, lp5]

class model:
    def __init__(self, directory, order=None):
        self.directory= directory
        f.cd(self.directory)
        #f.ls()
        self.name     = directory.replace("/","_")
        self.ttbar    = f.Get(directory+"/ttbar")
        self.multijet = f.Get(directory+"/multijet")
        self.data_obs = f.Get(directory+"/data_obs")
        self.nBins = self.data_obs.GetSize()-2 #underflow and overflow

        self.background_TH1 = ROOT.TH1F("background_TH1", "", self.nBins, 0, 1)
        
        self.rebin = rebin
        self.ttbar   .Rebin(self.rebin)
        self.multijet.Rebin(self.rebin)
        self.data_obs.Rebin(self.rebin)

        self.nBins_rebin = self.data_obs.GetSize()-2

        # So that fit includes stat error from background templates, combine all stat error in quadrature
        for bin in range(1,self.nBins_rebin+1):
            data_error = self.data_obs.GetBinError(bin)
            ttbar_error = self.ttbar.GetBinError(bin)
            multijet_error = 1.0*self.multijet.GetBinError(bin)
            total_error = (data_error**2 + ttbar_error**2 + multijet_error**2)**0.5
            self.data_obs.SetBinError(bin, total_error)


        self.polynomials = {}
        self.background_TF1s = {}
        self.chi2s = {}
        self.ndfs = {}
        self.chi2PerNdfs = {}
        self.probs = {}
        self.parValues = {}
        self.parErrors = {}
        self.ymaxs = {}
        self.eigenVars = {}
        self.eigenPars = {}
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

        self.background_TF1s[order] = ROOT.TF1 ("background_TF1_pol"+str(order), background_UserFunction, 0, 1, order+1)


    def getEigenvariations(self, order=None, debug=False):
        if order is None: order = self.order

        n = order+1
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

        errorVec = [np.array(range(n), dtype=np.float) for i in range(n)]
        for i in range(n):
            for j in range(n):
                errorVec[j][i] = eigenVec[i][j] * eigenVal[j]**0.5

        self.eigenVars[order] = errorVec

        if debug:
            print "Eigenvariations"
            for i in range(n):
                print i, errorVec[i]

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
        self.parValues[order] = self.background_TF1s[order].GetParameters()
        self.parErrors[order] = self.background_TF1s[order].GetParErrors()
        self.ymaxs[order] = self.background_TF1s[order].GetMaximum(0,1)


    def dumpFitResult(self, order=None):
        if order is None: order = self.order
        print "    chi^2 =",self.chi2s[order]
        print "      ndf =",self.ndfs[order]
        print "chi^2/ndf =",self.chi2PerNdfs[order]
        print "  p-value =",self.probs[order]


    def fit(self, order=None):
        if order is None: order = self.order
        if order not in self.background_TF1s: self.makeBackgrondTF1(order)            
        self.fitResult = self.data_obs.Fit(self.background_TF1s[order], "L N0QS")
        status = int(self.fitResult)
        self.getEigenvariations(order)
        self.storeFitResult(order)
        self.dumpFitResult(order)


    def fTest(self, order2, order1): #order2 > order1
        for order in [order1, order2]:
            if order in self.chi2s: continue
            self.makeBackgrondTF1(order)
            self.fit(order)

        self.fStat = ( (self.chi2s[order1] - self.chi2s[order2])/(order2 - order1) ) / (self.chi2PerNdfs[order2])
        self.fProb = scipy.stats.f.cdf(self.fStat, order2-order1, self.ndfs[order2])
        print "    f(%i,%i) = %f"%(order2,order1,self.fStat)
        print "f.cdf(%i,%i) = %f"%(order2,order1,self.fProb)
        return self.fProb


    def runFTest(self):
        order = 0
        while self.fTest(order+1, order) > 0.95:
            order += 1
        print "F-Test prefers order %i over %i at only %2.0f%%"%(order+1, order, 100*self.fProb)
        self.setOrder(order)


    def setOrder(self, order):
        self.order = order
        self.background_TF1 = self.background_TF1s[order]
        self.chi2 = self.chi2s[order]
        self.ndf = self.ndfs[order]
        self.chi2PerNdf = self.chi2PerNdfs[order]
        self.prob = self.probs[order]
        self.parValue = np.array([self.parValues[order][i] for i in range(order+1)], dtype=np.float) #self.parValues[order]
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


    def write(self):
        f.cd(self.directory)

        for bin in range(1,self.nBins+1):
            x = self.background_TH1.GetBinCenter(bin)
            self.background_TH1.SetBinContent(bin, self.background_TF1.Eval(x)/self.rebin)

        self.background_TH1.Write()


    def plotFit(self):
        samples=collections.OrderedDict()
        samples[closureFileName] = collections.OrderedDict()
        samples[closureFileName][self.directory+"/data_obs"] = {
            "label" : ("Mixed Data %.1f/fb")%(lumi),
            "legend": 1,
            "isData" : True,
            "ratio" : "numer A",
            "color" : "ROOT.kBlack"}
        samples[closureFileName][self.directory+"/multijet"] = {
            "label" : "Multijet Model",
            "legend": 2,
            "stack" : 3,
            "ratio" : "denom A",
            "color" : "ROOT.kYellow"}
        samples[closureFileName][self.directory+"/ttbar"] = {
            "label" : "t#bar{t}",
            "legend": 3,
            "stack" : 2,
            "ratio" : "denom A",
            "color" : "ROOT.kAzure-9"}
        samples[closureFileName][self.directory+"/background_TH1"] = {
            "label" : "Fit",
            "legend": 4,
            "ratio": "denom A", 
            "color" : "ROOT.kRed"}

        if "zz" in self.directory:
            xTitle = "SvB Regressed P(ZZ)+P(ZH), P(ZZ) > P(ZH)"
        if "zh" in self.directory:
            xTitle = "SvB Regressed P(ZZ)+P(ZH), P(ZH) #geq P(ZZ)"
        parameters = {"titleLeft"   : "#bf{CMS} Internal",
                      "titleCenter" : "Signal Region",
                      "titleRight"  : "Pass #DeltaR(j,j)",
                      "maxDigits"   : 4,
                      "ratio"     : True,
                      "rMin"      : 0.9,
                      "rMax"      : 1.1,
                      "rebin"     : self.rebin,
                      "rTitle"    : "Data / Bkgd.",
                      "xTitle"    : xTitle,
                      "yTitle"    : "Events",
                      "yMax"      : self.ymax*2.0,
                      "xleg"      : [0.15, 0.15+0.33],
                      "legendSubText" : ["#bf{Fit Result:}",
                                         ],
                      "lstLocation" : "right",
                      "outputDir" : "",
                      "outputName": self.name}
        for i in range(self.order+1):
            parameters["legendSubText"] += ["#font[12]{p}_{%i} = %0.3f #pm %0.1f%%"%(i, self.parValue[i], 100*self.parError[i]/self.parValue[i])]

        parameters["legendSubText"] += ["",
                                        "#chi^{2}/DoF = %0.2f"%(self.chi2/self.ndf),
                                        "p-value = %2.0f%%"%(self.prob*100)]

        PlotTools.plot(samples, parameters, debug=False)


class ensemble:
    def __init__(self, channel, models):
        self.channel = channel
        self.models = models
        self.n = len(models)
        self.chi2s = {}
        self.ndfs = {}
        self.probs = {}
        self.fProbs = {}
        self.order = 1

    def combinedFTest(self, order2, order1):
        for order in [order1, order2]:
            self.chi2s[order], self.ndfs[order] = 0, 0
            for m in self.models:
                if order not in m.chi2s:
                    print "\n>>> ", order, m.name
                    m.makeBackgrondTF1(order)
                    m.fit(order)

                self.chi2s[order] += m.chi2s[order]
                self.ndfs[order] += m.ndfs[order]

            self.probs[order] = scipy.stats.distributions.chi2.sf(self.chi2s[order], self.ndfs[order])

        print "\n"+"#"*10
        for order in [order1, order2]:
            print " Order %i:"%order
            print "  chi2/ndf = ", self.chi2s[order]/self.ndfs[order]
            print "      prob = %2.1f%%"%(100*self.probs[order])

        fStat = ( (self.chi2s[order1] - self.chi2s[order2])/(order2 - order1)/self.n ) / (self.chi2s[order2]/self.ndfs[order2])
        self.fProbs[order1] = scipy.stats.f.cdf(fStat, (order2-order1)*self.n, self.ndfs[order2])
        print "-"*10
        print "    f(%i,%i) = %f"%(order2,order1,fStat)
        print "f.cdf(%i,%i) = %2.0f%%"%(order2,order1,100*self.fProbs[order1])
        print "#"*10

        if self.fProbs[order1] < 0.95: self.fTestSatisfied = True

        if not self.done and self.probs[order1] > 0.05 and self.fTestSatisfied:
            self.done = True
            self.order = order1
            self.prob = self.probs[order1]
            self.fProb = self.fProbs[order1]


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
        while order < 4:# and (self.fProb > 0.95 or self.prob < 0.05):
            order += 1
            self.combinedFTest(order+1, order)

        if self.done:
            print "\nCombined F-Test prefers order %i over %i at %2.0f%%"%(self.order+1, self.order, 100*self.fProb)
        else:
            print "Failed to satisfy F-Test and/or goodness of fit."
    
        for m in self.models:
            m.setOrder(self.order)
            m.write()
        self.getParameterDistribution()

    def getParameterDistribution(self):
        nModels = len(self.models)
        n = self.order+1
        self.parMean  = np.array([0 for i in range(n)], dtype=np.float)
        self.parMean2 = np.array([0 for i in range(n)], dtype=np.float)
        for m in self.models:
            self.parMean  += m.parValue    / nModels
            self.parMean2 += m.parValue**2 / nModels
        var = self.parMean2 - self.parMean**2
        self.parStd = var**0.5
        print "Parameter Mean:",self.parMean
        print "Parameter  Std:",self.parStd
        #cov = np.zeros((n,n))
        #for i in range(n): cov[i][i] = var[i]
        #gaussian = scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=var)

        closureResults = "ZZ4b/nTupleAnalysis/combine/closureResults_%s.txt"%self.channel
        closureResultsFile = open(closureResults, 'w')
        for i in range(n):
            systUp   = "1+" #if i else ""
            systDown = "1+" #if i else ""
            cUp   = max(self.parMean[i], self.parMean[i] + self.parStd[i],  self.parStd[i])
            cDown = min(self.parMean[i], self.parMean[i] - self.parStd[i], -self.parStd[i])
            systUp   += "(%f)*(%s)"%(cUp,   lps[i].replace(' ',''))
            systDown += "(%f)*(%s)"%(cDown, lps[i].replace(' ',''))
            systUp   = "multijet_LP%i_%sUp   %s"%(i, channel, systUp)
            systDown = "multijet_LP%i_%sDown %s"%(i, channel, systDown)
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

        x = sorted(self.fProbs.keys())
        y = [self.fProbs[o] for o in x]
        ax.plot(x, y, label='F-Test', color='black', alpha=0.8, linewidth=2)

        x = sorted(self.probs.keys())
        y = [self.probs[o] for o in x]
        ax.plot(x, y, label='Combined', color='r', linewidth=2)
        
        for m in self.models:
            x = sorted(m.probs.keys())
            y = [m.probs[o] for o in x]
            ax.plot(x, y, label=m.name.replace('_','\_'), alpha=0.5)
            
        maxOrder = x[-1]
        ax.plot([0,maxOrder], [0.05,0.05], color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.plot([0,maxOrder], [0.95,0.95], color='k', alpha=0.5, linestyle='--', linewidth=1)

        ax.set_xlabel('Polynomial Order')
        ax.set_ylabel('Fit p-value')
        ax.legend(loc='lower right', fontsize='small')

        name = "pvalues_"+channel+"_rebin"+str(rebin)+".pdf"
        print "fig.savefig( "+name+" )"
        fig.savefig( name )

    def plotFitResults(self, order=None):
        if order is None: order = self.order
        n = order+1

        x,y,s,c = [],[],[],[]
        for m in self.models:
            eigenVars = m.eigenVars[order]
            for eigenVar in eigenVars:
                x.append( eigenVar[0] )
                if n>1:
                    y.append( eigenVar[1] )
                if n>2:
                    c.append( eigenVar[2] )
                if n>3:
                    s.append( eigenVar[3] )

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
        ax.set_title('Fit Eigen-Variations: '+channel)
        ax.set_xlabel('p$_0$')
        ax.set_ylabel('p$_1$')
        
        plt.scatter(x, y, **kwargs)
        if 'c' in kwargs:
            cbar = plt.colorbar(label='p$_2$',
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
                             title='p$_3$', 
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
        
        name = 'eigenvariations_'+channel+'.pdf'
        print "fig.savefig( "+name+" )"
        fig.savefig( name )


        #plot fit parameters
        x,y,s,c = [],[],[],[]
        for m in self.models:
            x.append( m.parValues[order][0] )
            if n>1:
                y.append( m.parValues[order][1] )
            if n>2:
                c.append( m.parValues[order][2] )
            if n>3:
                s.append( m.parValues[order][3] )


    
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
        ax.set_title('Legendre Polynomial Coefficients: '+channel)
        ax.set_xlabel('c$_0$')
        ax.set_ylabel('c$_1$')
        
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

        
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.plot([0,0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        name = 'fitParameters_'+channel+'.pdf'
        print "fig.savefig( "+name+" )"
        fig.savefig( name )


        x,y,s,c = [],[],[],[]
        for m in self.models:
            x.append( m.eigenPars[order][0] )
            if n>1:
                y.append( m.eigenPars[order][1] )
            if n>2:
                c.append( m.eigenPars[order][2] )
            if n>3:
                s.append( m.eigenPars[order][3] )

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
        ax.set_title('Fit Parameters (Eigen-Basis): '+channel)
        ax.set_xlabel('e$_0$')
        ax.set_ylabel('e$_1$')
        
        plt.scatter(x, y, **kwargs)
        if 'c' in kwargs:
            cbar = plt.colorbar(label='e$_2$',
                                ticks=np.linspace(0.25, -1, 1, endpoint=True)
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
                             title='e$_3$', 
                             #title_fontsize='small',
                             #columnspacing=1.8,
                             #labelspacing=1.5,
                             scatterpoints = 1, 
                             #scatteryoffsets=[0.5],
                             )

        
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xlim, ylim = [-1,1], [-1,1]
        ax.plot(xlim, [0,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.plot([0,0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        name = 'fitParameters_eigenbasis_'+channel+'.pdf'
        print "fig.savefig( "+name+" )"
        fig.savefig( name )



models = {}

for channel in channels:    
    models[channel] = []
    for i, mix in enumerate(mixes):
        models[channel].append( model(mix+"/"+channel) )
        #models[channel][i].fit(order=2)
        #print "Single F-Test:", channel, i
        #models[channel][i].runFTest()
        #models[channel][i].write()
    print "-"*20
    print "Combined F-Test:", channel
    allMixes = ensemble(channel, models[channel])
    allMixes.runCombinedFTest()
    print "-"*20
    allMixes.plotFitResultsByOrder()
    allMixes.plotFitResults()

f.Close()

for channel in channels:
    for i, mix in enumerate(mixes):
        models[channel][i].plotFit()


