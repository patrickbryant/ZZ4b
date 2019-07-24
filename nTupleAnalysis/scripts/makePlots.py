import sys
import collections
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import PlotTools
import optparse

parser = optparse.OptionParser()
parser.add_option('-d', '--debug',                dest="debug",         action="store_true", default=False, help="debug")
parser.add_option('-y', '--year',                 dest="year",          default="2018", help="Year specifies trigger (and lumiMask for data)")
parser.add_option('-l', '--lumi',                 dest="lumi",          default="1",    help="Luminosity for MC normalization: units [pb]")
parser.add_option('-o', '--outputBase',           dest="outputBase",    default="/uscms/home/bryantp/nobackup/ZZ4b/plots/", help="Base path for storing output histograms and picoAOD")
parser.add_option('-p', '--plotDir',              dest="plotDir",       default="plots/", help="Base path for storing output histograms and picoAOD")
parser.add_option('-j',            action="store_true", dest="useJetCombinatoricModel",       default=False, help="make plots after applying jetCombinatoricModel")
parser.add_option('-r',            action="store_true", dest="reweight",       default=False, help="make plots after reweighting by FvTWeight")
o, a = parser.parse_args()

#make sure outputBase ends with /
outputBase = o.outputBase + ("" if o.outputBase[-1] == "/" else "/")
outputPlot = outputBase+o.plotDir + ("" if o.plotDir[-1] == "/" else "/")
print "Plot output:",outputPlot

lumi = float(o.lumi)/1000

files = {"data"+o.year  : outputBase+"data"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root",
         "ZH4b"+o.year   : outputBase+"ZH4b"+o.year+"/hists.root",
         "ggZH4b"+o.year : outputBase+"ggZH4b"+o.year+"/hists.root",
         "bothZH4b"+o.year : outputBase+"bothZH4b"+o.year+"/hists.root",
         "ZZ4b"+o.year   : outputBase+"ZZ4b"+o.year+"/hists.root",
         "TTJets"+o.year : outputBase+"TTJets"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root",
         "TT"+o.year : outputBase+"TT"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root",
         "qcd"+o.year : outputBase+"qcd"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root",
         }


class nameTitle:
    def __init__(self, name, title):
        self.name  = name
        self.title = title

cuts = [#nameTitle("passPreSel", "Preselection"), 
        #nameTitle("passDijetMass", "Pass m_{jj} Cuts"), 
        nameTitle("passMDRs", "Pass MDR's"), 
        #nameTitle("passXWt", "xWt > 2"), 
        #nameTitle("passMDCs", "Pass MDC's"), 
        #nameTitle("passDEtaBB", "|#Delta#eta| < 1.5"),
        ]
views = ["allViews",
         "mainView",
         ]
regions = [nameTitle("inclusive", ""),
           #nameTitle("ZH", "ZH SB+CR+SR"),
           #nameTitle("ZH_SvB_high", "ZH SB+CR+SR SvB>0.5"), nameTitle("ZH_SvB_low", "ZH SB+CR+SR SvB<0.5"),
           #nameTitle("ZHSB", "ZH Sideband"), nameTitle("ZHCR", "ZH Control Region"), nameTitle("ZHSR", "ZH Signal Region"),
           #nameTitle("ZZ", "ZZ SB+CR+SR"),
           #nameTitle("ZZ_SvB_high", "ZZ SB+CR+SR SvB>0.5"), nameTitle("ZZ_SvB_low", "ZZ SB+CR+SR SvB<0.5"),
           #nameTitle("ZZSB", "ZZ Sideband"), nameTitle("ZZCR", "ZZ Control Region"), nameTitle("ZZSR", "ZZ Signal Region"),
           nameTitle("ZHSR", "ZH Signal Region"), nameTitle("ZZSR", "ZZ Signal Region"),
           nameTitle("SCSR", "SB+CR+SR"),
           nameTitle("SB", "Sideband"), nameTitle("CR", "Control Region"), nameTitle("SR", "Signal Region"),
           ]

plots=[]

class variable:
    def __init__(self, name, xTitle, yTitle = None, zTitle = None, rebin = None, divideByBinWidth = False, normalize = None, normalizeStack = None):
        self.name   = name
        self.xTitle = xTitle
        self.yTitle = yTitle
        self.zTitle = zTitle
        self.rebin  = rebin
        self.divideByBinWidth = divideByBinWidth
        self.normalize = normalize
        self.normalizeStack = normalizeStack

class standardPlot:
    def __init__(self, year, cut, view, region, var):
        self.samples=collections.OrderedDict()
        self.samples[files[  "data"+year]] = collections.OrderedDict()
        self.samples[files[   "qcd"+year]] = collections.OrderedDict()
        self.samples[files[    "TT"+year]] = collections.OrderedDict()
        self.samples[files[  "ZH4b"+year]] = collections.OrderedDict()
        self.samples[files["ggZH4b"+year]] = collections.OrderedDict()
        self.samples[files[  "ZZ4b"+year]] = collections.OrderedDict()
        self.samples[files[  "data"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label" : ("Data %.1f/fb, "+year)%(lumi),
                                                                                                   "legend": 1,
                                                                                                   "isData" : True,
                                                                                                   "ratio" : "numer A",
                                                                                                   "color" : "ROOT.kBlack"}
        self.samples[files["qcd"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {"label" : "Multijet Model",
                                                                                                    "legend": 2,
                                                                                                    "stack" : 3,
                                                                                                    "ratio" : "denom A",
                                                                                                    "color" : "ROOT.kYellow"}
        # self.samples[files["TT"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {"label" : "t#bar{t} (3-tag)",
        #                                                                                                "legend": 4,
        #                                                                                                #"stack" : 2,
        #                                                                                                #"ratio" : "denom A",
        #                                                                                                "color" : "ROOT.kAzure-4"}
        self.samples[files["TT"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label" : "t#bar{t} (4-tag)",
                                                                                                  "legend": 3,
                                                                                                  "stack" : 2,
                                                                                                  "ratio" : "denom A",
                                                                                                  "color" : "ROOT.kAzure-9"}
        # self.samples[files["TTJets"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {"label" : "t#bar{t}+jets (3-tag)",
        #                                                                                                "legend": 3,
        #                                                                                                #"stack" : 2,
        #                                                                                                #"ratio" : "denom A",
        #                                                                                                "color" : "ROOT.kAzure-4"}
        # self.samples[files["TTJets"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label" : "t#bar{t}+jets (4-tag)",
        #                                                                                               "legend": 4,
        #                                                                                               #"stack" : 2,
        #                                                                                               #"ratio" : "denom A",
        #                                                                                               "color" : "ROOT.kBlue"}
        # self.samples[files["ZH4b"+year    ]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {"label"    : "ZH#rightarrowb#bar{b}b#bar{b} (3-Tag)",
        #                                                                                             "legend"   : 3,
        #                                                                                             "stack"    : 4,
        #                                                                                             "color"    : "ROOT.kRed"}
        self.samples[files["ZH4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label"    : "ZH#rightarrowb#bar{b}b#bar{b} (#times100)",
                                                                                                        "legend"   : 5,
                                                                                                                     "weight" : 100,
                                                                                                        "color"    : "ROOT.kRed"}
        # self.samples[files["ggZH4b"+year  ]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {"label"    : "gg#rightarrowZH#rightarrowb#bar{b}b#bar{b} (3-Tag)",
        #                                                                                                  "legend"   : 4,
        #                                                                                                  "stack"    : 5,
        #                                                                                                  "color"    : "ROOT.kViolet"}
        self.samples[files["ggZH4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label"    : "gg#rightarrowZH#rightarrowb#bar{b}b#bar{b} (#times100)",
                                                                                                        "legend"   : 6,
                                                                                                                     "weight" : 100,
                                                                                                        "color"    : "ROOT.kViolet"}
        self.samples[files["ZZ4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label"    : "ZZ#rightarrowb#bar{b}b#bar{b} (#times100)",
                                                                                                    "legend"   : 7,
                                                                                                    "weight" : 100,
                                                                                                    "color"    : "ROOT.kGreen+3"}
        if var.name == "FvT": 
            del self.samples[files[  "ZH4b"+year]]
            del self.samples[files["ggZH4b"+year]]
            del self.samples[files[  "ZZ4b"+year]]
        # if var.name == "ZHvsB":
        #     del self.samples[files[  "ZH4b"+year]]
        #     del self.samples[files["ggZH4b"+year]]
        #     self.samples[files["bothZH4b"+year]] = collections.OrderedDict()
        #     self.samples[files["bothZH4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label"    : "Inclusive ZH#rightarrowb#bar{b}b#bar{b} (#times100)",
        #                                                                                                     "legend"   : 4,
        #                                                                                                     "weight" : 100,
        #                                                                                                     "color"    : "ROOT.kBlue"}

        self.parameters = {"titleLeft"   : "#bf{CMS} Internal",
                           "titleCenter" : region.title,
                           "titleRight"  : cut.title,
                           "ratio"     : True,
                           "rMin"      : 0.5 if not o.reweight else 0.5,
                           "rMax"      : 1.5 if not o.reweight else 1.5,
                           "rTitle"    : "Data / Bkgd.",
                           "xTitle"    : var.xTitle,
                           "yTitle"    : "Events" if not var.yTitle else var.yTitle,
                           "outputDir" : outputPlot+"data/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName": var.name}
        if var.divideByBinWidth: self.parameters["divideByBinWidth"] = True
        if var.rebin: self.parameters["rebin"] = var.rebin
        if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack

    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, o.debug or debug)


class massPlanePlot:
    def __init__(self, topDir, fileName, year, cut, tag, view, region, var):
        self.samples=collections.OrderedDict()
        self.samples[files[fileName.name]] = collections.OrderedDict()
        self.samples[files[fileName.name]][cut.name+"/"+tag+"/"+view+"/"+region.name+"/"+var.name] = {"drawOptions": "COLZ"}
                
        self.parameters = {"titleLeft"       : "#scale[0.7]{#bf{CMS} Internal}",
                           "titleCenter"      : "#scale[0.7]{"+region.title+"}",
                           "titleRight"      : "#scale[0.7]{"+cut.title+"}",
                           "subTitleRight"   : "#scale[0.7]{"+fileName.title+"}",
                           "yTitle"      : var.yTitle,
                           "xTitle"      : var.xTitle,
                           "zTitle"      : "Events / Bin",
                           "zTitleOffset": 1.4,
                           #"xMin"        : 0,
                           #"xMax"        : 250,
                           #"yMin"        : 0,
                           #"yMax"        : 250,
                           "zMin"        : 0,
                           "maxDigits"   : 3,
                           "xNdivisions" : 505,
                           "yNdivisions" : 505,
                           "rMargin"     : 0.15,
                           "rebin"       : 1, 
                           "canvasSize"  : [720,660],
                           "outputDir"   : outputPlot+topDir+"/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName"  : var.name+"_"+tag,
                           }
        #      ## HH Regions
        # "functions"   : [[" ((x-120*1.03)**2     + (y-110*1.03)**2)",     0,250,0,250,[30.0**2],"ROOT.kOrange+7",1],
        #                  [ "((x-120*1.05)**2     + (y-110*1.05)**2)",     0,250,0,250,[45.0**2],"ROOT.kYellow",  1],
        #                  ["(((x-120)/(0.1*x))**2 +((y-110)/(0.1*y))**2)", 0,250,0,250,[ 1.6**2],"ROOT.kRed",     7]],
        #      ## ZH Regions
        # "functions"   : [[" ((x-115*1.03)**2     + (y- 88*1.03)**2)",     0,250,0,250,[30.0**2],"ROOT.kOrange+7",1],
        #                  [ "((x-115*1.05)**2     + (y- 88*1.05)**2)",     0,250,0,250,[45.0**2],"ROOT.kYellow",  1],
        #                  ["(((x-115)/(0.1*x))**2 +((y- 88)/(0.1*y))**2)", 0,250,0,250,[ 1.6**2],"ROOT.kRed",     7]],
        #      ## ZZ Regions
        # "functions"   : [[" ((x- 90*1.03)**2     + (y-82.5*1.02)**2)",     0,250,0,250,[28.0**2],"ROOT.kOrange+7",1],
        #                  [ "((x- 90*1.05)**2     + (y-82.5*1.04)**2)",     0,250,0,250,[40.0**2],"ROOT.kYellow",  1],
        #                  ["(((x- 90)/(0.1*x))**2 +((y-82.5)/(0.1*y))**2)", 0,250,0,250,[ 1.6**2],"ROOT.kRed",     7]],
        self.parameters["functions"] = [["(((x-120.0)/(0.1*x))**2 +((y-115.0)/(0.1*y))**2)", 0,250,0,250,[ 1.6**2],"ROOT.kRed",     7],
                                        ["(((x-123.0)/(0.1*x))**2 +((y- 92.0)/(0.1*y))**2)", 0,250,0,250,[ 1.5**2],"ROOT.kRed",     7],
                                        ["(((x- 92.0)/(0.1*x))**2 +((y-123.0)/(0.1*y))**2)", 0,250,0,250,[ 1.5**2],"ROOT.kRed",     7],
                                        ["(((x- 91.0)/(0.1*x))**2 +((y- 87.2)/(0.1*y))**2)", 0,250,0,250,[ 1.6**2],"ROOT.kRed",     7]]

    def newSample(self, topDir, fileName, cut, tag, view, region, var):
        self.samples[files[fileName.name]] = collections.OrderedDict()
        self.samples[files[fileName.name]][cut.name+"/"+tag+"/"+view+"/"+region.name+"/"+var.name] = {"drawOptions": "COLZ"}
        self.parameters["titleCenter"]   = "#scale[0.7]{"+region.title+"}"
        self.parameters["titleRight"]    = "#scale[0.7]{"+cut.title+"}"
        self.parameters["subTitleRight"] = "#scale[0.7]{"+fileName.title+"}"
        self.parameters["outputDir"]     = outputPlot+topDir+"/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/"
        self.parameters["outputName"]    = var.name+"_"+tag

    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, o.debug or debug)


variables=[variable("nPVs", "Number of Primary Vertices"),
           variable("nPVsGood", "Number of 'Good' Primary Vertices"),
           variable("nSelJets", "Number of Selected Jets"),
           variable("nSelJets_lowSt", "Number of Selected Jets (s_{T,4j} < 320 GeV)"),
           variable("nSelJets_midSt", "Number of Selected Jets (320 < s_{T,4j} < 450 GeV)"),
           variable("nSelJets_highSt", "Number of Selected Jets (s_{T,4j} > 450 GeV)"),
           variable("nSelJetsUnweighted", "Number of Selected Jets (Unweighted)", normalizeStack="data"),
           variable("nPSTJets", "Number of Tagged + Pseudo-Tagged Jets"),
           variable("nPSTJets_lowSt", "Number of Tagged + Pseudo-Tagged Jets (s_{T,4j} < 320 GeV)"),
           variable("nPSTJets_midSt", "Number of Tagged + Pseudo-Tagged Jets (320 < s_{T,4j} < 450 GeV)"),
           variable("nPSTJets_highSt", "Number of Tagged + Pseudo-Tagged Jets (s_{T,4j} > 450 GeV)"),
           variable("nTagJets", "Number of Tagged Jets"),
           variable("nAllJets", "Number of Jets"),
           variable("st", "Scalar sum of all jet p_{T}'s [GeV]"),
           variable("stNotCan", "Scalar sum of all other jet p_{T}'s [GeV]"),
           #variable("FvT", "Four vs Three Tag Classifier Output", rebin=[i/100.0 for i in range(0,45,5)]+[i/100.0 for i in range(45,55)]+[i/100.0 for i in range(55,101,5)], yTitle = "Events / 0.01 Output"),
           variable("FvT", "Four vs Three Tag Classifier Output", rebin = 2),
           variable("ZHvB", "ZH vs Background Output", rebin=[float(i)/50 for i in range(51)], yTitle = "Events / 0.01 Output"),
           variable("xZH", "x_{ZH}"),
           variable("xZZ", "x_{ZZ}"),
           variable("xWt0", "Minimum x_{Wt} (#geq0 not candidate jets)", rebin=1),
           variable("xWt1", "Minimum x_{Wt} (#geq1 not candidate jets)", rebin=1),
           variable("xWt2", "Minimum x_{Wt} (#geq2 not candidate jets)", rebin=1),
           variable("xWt",  "x_{Wt}", rebin=1),
           variable("t/dRbW", "Top Candidate #DeltaR(b,W)"),
           variable("t/m", "Top Candidate Mass [GeV]"),
           variable("t/W/m", "W Boson Candidate Mass [GeV]"),
           variable("t/mbW", "Top Candidate m_{b,W} [GeV]"),
           variable("t/xWt", "Top Candidate x_{W,t}"),
           variable("t0/dRbW", "Top Candidate (#geq0 not candidate jets) #DeltaR(b,W)"),
           variable("t0/m", "Top Candidate (#geq0 not candidate jets) Mass [GeV]"),
           variable("t0/W/m", "W Boson Candidate (#geq0 not candidate jets) Mass [GeV]"),
           variable("t0/mbW", "Top Candidate (#geq0 not candidate jets) m_{b,W} [GeV]"),
           variable("t0/xWt", "Top Candidate (#geq0 not candidate jets) x_{W,t}"),
           variable("t1/dRbW", "Top Candidate (#geq1 not candidate jets) #DeltaR(b,W)"),
           variable("t1/m", "Top Candidate (#geq1 not candidate jets) Mass [GeV]"),
           variable("t1/W/m", "W Boson Candidate (#geq1 not candidate jets) Mass [GeV]"),
           variable("t1/mbW", "Top Candidate (#geq1 not candidate jets) m_{b,W} [GeV]"),
           variable("t1/xWt", "Top Candidate (#geq1 not candidate jets) x_{W,t}"),
           variable("t2/dRbW", "Top Candidate (#geq2 not candidate jets) #DeltaR(b,W)"),
           variable("t2/m", "Top Candidate (#geq2 not candidate jets) Mass [GeV]"),
           variable("t2/W/m", "W Boson Candidate (#geq2 not candidate jets) Mass [GeV]"),
           variable("t2/mbW", "Top Candidate (#geq2 not candidate jets) m_{b,W} [GeV]"),
           variable("t2/xWt", "Top Candidate (#geq2 not candidate jets) x_{W,t}"),
           variable("mZH", "m_{ZH} [GeV]", divideByBinWidth = True),
           variable("mZZ", "m_{ZZ} [GeV]", divideByBinWidth = True),
           variable("dBB", "D_{BB} [GeV]"),
           variable("dEtaBB", "#Delta#eta(B_{1}, B_{2})"),
           variable("dRBB", "#DeltaR(B_{1}, B_{2})"),
           variable("v4j/m_l", "m_{4j} [GeV]"),
           variable("v4j/pt_l", "p_{T,4j} [GeV]"),
           variable("v4j/pz_l", "p_{z,4j} [GeV]"),
           variable("s4j", "s_{T,4j} [GeV]"),
           variable("r4j", "p_{T,4j} / s_{T,4j}"),
           variable("m123", "m_{1,2,3} [GeV]"),
           variable("m023", "m_{0,2,3} [GeV]"),
           variable("m013", "m_{0,1,3} [GeV]"),
           variable("m012", "m_{0,1,2} [GeV]"),
           variable("canJets/pt_s", "Boson Candidate Jets p_{T} [GeV]"),
           variable("canJets/pt_m", "Boson Candidate Jets p_{T} [GeV]"),
           variable("canJets/pt_l", "Boson Candidate Jets p_{T} [GeV]"),
           variable("canJets/eta", "Boson Candidate Jets #eta"),
           variable("canJets/phi", "Boson Candidate Jets #phi"),
           variable("canJets/m",   "Boson Candidate Jets Mass [GeV]"),
           variable("canJets/e",   "Boson Candidate Jets Energy [GeV]"),
           variable("canJets/deepFlavB", "Boson Candidate Jets Deep Flavour B"),
           variable("canJets/deepB", "Boson Candidate Jets Deep CSV B"),
           variable("canJets/CSVv2", "Boson Candidate Jets CSVv2"),
           variable("othJets/pt_s", "Other Jets p_{T} [GeV]"),
           variable("othJets/pt_m", "Other Jets p_{T} [GeV]"),
           variable("othJets/pt_l", "Other Jets p_{T} [GeV]"),
           variable("othJets/eta", "Other Jets #eta"),
           variable("othJets/phi", "Other Jets #phi"),
           variable("othJets/m",   "Other Jets Mass [GeV]"),
           variable("othJets/e",   "Other Jets Energy [GeV]"),
           variable("othJets/deepFlavB", "Other Jets Deep Flavour B"),
           variable("othJets/deepB", "Other Jets Deep CSV B"),
           variable("othJets/CSVv2", "Other Jets CSVv2"),
           variable("allNotCanJets/pt_s", "All jets (p_{T}>20) excluding boson candidate jets p_{T} [GeV]"),
           variable("allNotCanJets/pt_m", "All jets (p_{T}>20) excluding boson candidate jets p_{T} [GeV]"),
           variable("allNotCanJets/pt_l", "All jets (p_{T}>20) excluding boson candidate jets p_{T} [GeV]"),
           variable("allNotCanJets/eta", "All jets (p_{T}>20) excluding boson candidate jets #eta"),
           variable("allNotCanJets/phi", "All jets (p_{T}>20) excluding boson candidate jets #phi"),
           variable("allNotCanJets/m",   "All jets (p_{T}>20) excluding boson candidate jets Mass [GeV]"),
           variable("allNotCanJets/e",   "All jets (p_{T}>20) excluding boson candidate jets Energy [GeV]"),
           variable("allNotCanJets/deepFlavB", "All jets (p_{T}>20) excluding boson candidate jets Deep Flavour B"),
           variable("allNotCanJets/deepB", "All jets (p_{T}>20) excluding boson candidate jets Deep CSV B"),
           variable("allNotCanJets/CSVv2", "All jets (p_{T}>20) excluding boson candidate jets CSVv2"),
           variable("aveAbsEta", "Boson Candidate Jets <|#eta|>"),
           variable("aveAbsEtaOth", "Other Jets <|#eta|>"),
           variable("canJet1/pt_s", "Boson Candidate Jet_{1} p_{T} [GeV]"),
           variable("canJet1/pt_m", "Boson Candidate Jet_{1} p_{T} [GeV]"),
           variable("canJet1/pt_l", "Boson Candidate Jet_{1} p_{T} [GeV]"),
           variable("canJet1/eta", "Boson Candidate Jet_{1} #eta"),
           variable("canJet1/phi", "Boson Candidate Jet_{1} #phi"),
           variable("canJet1/deepFlavB", "Boson Candidate Jet_{1} Deep Flavour B"),
           variable("canJet1/deepB", "Boson Candidate Jet_{1} Deep CSV B"),
           variable("canJet1/CSVv2", "Boson Candidate Jet_{1} CSVv2"),
           variable("canJet3/pt_s", "Boson Candidate Jet_{3} p_{T} [GeV]"),
           variable("canJet3/pt_m", "Boson Candidate Jet_{3} p_{T} [GeV]"),
           variable("canJet3/pt_l", "Boson Candidate Jet_{3} p_{T} [GeV]"),
           variable("canJet3/eta", "Boson Candidate Jet_{3} #eta"),
           variable("canJet3/phi", "Boson Candidate Jet_{3} #phi"),
           variable("canJet3/deepFlavB", "Boson Candidate Jet_{3} Deep Flavour B"),
           variable("canJet3/deepB", "Boson Candidate Jet_{3} Deep CSV B"),
           variable("canJet3/CSVv2", "Boson Candidate Jet_{3} CSVv2"),
           variable("leadSt/m",    "Leading S_{T} Dijet Mass [GeV]"),
           variable("leadSt/dR",   "Leading S_{T} Dijet #DeltaR(j,j)"),
           variable("leadSt/pt_m", "Leading S_{T} Dijet p_{T} [GeV]"),
           variable("leadSt/eta",  "Leading S_{T} Dijet #eta"),
           variable("leadSt/phi",  "Leading S_{T} Dijet #phi"),
           variable("sublSt/m",    "Subleading S_{T} Dijet Mass [GeV]"),
           variable("sublSt/dR",   "Subleading S_{T} Dijet #DeltaR(j,j)"),
           variable("sublSt/pt_m", "Subleading S_{T} Dijet p_{T} [GeV]"),
           variable("sublSt/eta",  "Subleading S_{T} Dijet #eta"),
           variable("sublSt/phi",  "Subleading S_{T} Dijet #phi"),
           variable("leadM/m",    "Leading Mass Dijet Mass [GeV]"),
           variable("leadM/dR",   "Leading Mass Dijet #DeltaR(j,j)"),
           variable("leadM/pt_m", "Leading Mass Dijet p_{T} [GeV]"),
           variable("leadM/eta",  "Leading Mass Dijet #eta"),
           variable("leadM/phi",  "Leading Mass Dijet #phi"),
           variable("sublM/m",    "Subleading Mass Dijet Mass [GeV]"),
           variable("sublM/dR",   "Subleading Mass Dijet #DeltaR(j,j)"),
           variable("sublM/pt_m", "Subleading Mass Dijet p_{T} [GeV]"),
           variable("sublM/eta",  "Subleading Mass Dijet #eta"),
           variable("sublM/phi",  "Subleading Mass Dijet #phi"),
           variable("lead/m",    "Leading P_{T} Dijet Mass [GeV]"),
           variable("lead/dR",   "Leading p_{T} Dijet #DeltaR(j,j)"),
           variable("lead/pt_m", "Leading p_{T} Dijet p_{T} [GeV]"),
           variable("lead/eta",  "Leading p_{T} Dijet #eta"),
           variable("lead/phi",  "Leading p_{T} Dijet #phi"),
           variable("subl/m",    "Subleading p_{T} Dijet Mass [GeV]"),
           variable("subl/dR",   "Subleading p_{T} Dijet #DeltaR(j,j)"),
           variable("subl/pt_m", "Subleading p_{T} Dijet p_{T} [GeV]"),
           variable("subl/eta",  "Subleading p_{T} Dijet #eta"),
           variable("subl/phi",  "Subleading p_{T} Dijet #phi"),
           variable("close/m",    "Minimum #DeltaR(j,j) Dijet Mass [GeV]"),
           variable("close/dR",   "Minimum #DeltaR(j,j) Dijet #DeltaR(j,j)"),
           variable("close/pt_m", "Minimum #DeltaR(j,j) Dijet p_{T} [GeV]"),
           variable("close/eta",  "Minimum #DeltaR(j,j) Dijet #eta"),
           variable("close/phi",  "Minimum #DeltaR(j,j) Dijet #phi"),
           variable("other/m",    "Complement of Minimum #DeltaR(j,j) Dijet Mass [GeV]"),
           variable("other/dR",   "Complement of Minimum #DeltaR(j,j) Dijet #DeltaR(j,j)"),
           variable("other/pt_m", "Complement of Minimum #DeltaR(j,j) Dijet p_{T} [GeV]"),
           variable("other/eta",  "Complement of Minimum #DeltaR(j,j) Dijet #eta"),
           variable("other/phi",  "Complement of Minimum #DeltaR(j,j) Dijet #phi"),
           ]

if True:
    for cut in cuts:
        for view in views:
            for region in regions:
                for var in variables:
                    plots.append(standardPlot(o.year, cut, view, region, var))

                if True:
                    massPlane = variable("leadSt_m_vs_sublSt_m", "Leading S_{T} Dijet Mass [GeV]", "Subleading S_{T} Dijet Mass [GeV]")
                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))


                    massPlane = variable("t/mW_vs_mt", "W Boson Candidate Mass [GeV]", "Top Quark Candidate Mass [GeV]")

                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))


                    massPlane = variable("t/xW_vs_xt", "x_{W}", "x_{t}")

                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))

                    massPlane = variable("t0/mW_vs_mt", "W Boson Candidate Mass [GeV]", "Top Quark Candidate Mass [GeV]")

                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))


                    massPlane = variable("t0/xW_vs_xt", "x_{W}", "x_{t}")

                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))

                    massPlane = variable("t1/mW_vs_mt", "W Boson Candidate Mass [GeV]", "Top Quark Candidate Mass [GeV]")

                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))

                    massPlane = variable("t1/xW_vs_xt", "x_{W}", "x_{t}")

                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))

                    massPlane = variable("t2/mW_vs_mt", "W Boson Candidate Mass [GeV]", "Top Quark Candidate Mass [GeV]")

                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))

                    massPlane = variable("t2/xW_vs_xt", "x_{W}", "x_{t}")

                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))

                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))

                if True:
                    massPlane = variable("leadSt_m_vs_sublSt_m", "Leading S_{T} Dijet Mass [GeV]", "Subleading S_{T} Dijet Mass [GeV]")
                    ZH4b = nameTitle("bothZH4b"+o.year, "ZH#rightarrowb#bar{b}b#bar{b}")
                    plots.append(massPlanePlot("bothZH4b", ZH4b, o.year, cut, "fourTag", view, region, massPlane))

                    ZZ4b = nameTitle(    "ZZ4b"+o.year, "ZZ#rightarrowb#bar{b}b#bar{b}")
                    plots.append(massPlanePlot(    "ZZ4b", ZZ4b, o.year, cut, "fourTag", view, region, massPlane))


                if True:
                    massPlane = variable("leadM_m_vs_sublM_m", "Leading Mass Dijet Mass [GeV]", "Subleading Mass Dijet Mass [GeV]")
                    data = nameTitle("data2018", ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(massPlanePlot("data", data, o.year, cut, "fourTag", view, region, massPlane))
                    
                    data = nameTitle("data2018", "Background")
                    plots.append(massPlanePlot("data", data, o.year, cut, "threeTag", view, region, massPlane))

                if True:
                    ZH4b = nameTitle("bothZH4b"+o.year, "ZH#rightarrowb#bar{b}b#bar{b}")
                    plots.append(massPlanePlot("bothZH4b", ZH4b, o.year, cut, "fourTag", view, region, massPlane))

                    ZZ4b = nameTitle(    "ZZ4b"+o.year, "ZZ#rightarrowb#bar{b}b#bar{b}")
                    plots.append(massPlanePlot(    "ZZ4b", ZZ4b, o.year, cut, "fourTag", view, region, massPlane))



class accxEffPlot:
    def __init__(self, topDir, fileName, year, region):
        self.samplesAbs=collections.OrderedDict()
        self.samplesAbs[files[fileName.name].replace("hists.root","accxEff.root")] = collections.OrderedDict()
        self.samplesAbs[files[fileName.name].replace("hists.root","accxEff.root")]["jetMultiplicity_over_all"] = {"label"      : "#geq4 Selected Jets",
                                                                                         "legend"     : 1,
                                                                                         "color"      : "ROOT.kViolet",
                                                                                         "drawOptions" : "HIST PC",
                                                                                         "marker"      : "20"}
        self.samplesAbs[files[fileName.name].replace("hists.root","accxEff.root")]["bTags_over_all"] = {"label"      : "#geq4 b-Tagged Jets",
                                                                               "legend"     : 2,
                                                                               "color"      : "ROOT.kBlue",
                                                                               "drawOptions" : "HIST PC",
                                                                               "marker"      : "20"}
        self.samplesAbs[files[fileName.name].replace("hists.root","accxEff.root")]["MDRs_over_all"] = {"label"      : "Boson #DeltaR(j,j)",
                                                                              "legend"     : 3,
                                                                              "color"      : "ROOT.kGreen",
                                                                              "drawOptions" : "HIST PC",
                                                                              "marker"      : "20"}
        self.samplesAbs[files[fileName.name].replace("hists.root","accxEff.root")]["MDCs_over_all"] = {"label"      : "Boson p_{T}",
                                                                              "legend"     : 4,
                                                                              "color"      : "ROOT.kOrange",
                                                                              "drawOptions" : "HIST PC",
                                                                              "marker"      : "20"}
        self.samplesAbs[files[fileName.name].replace("hists.root","accxEff.root")]["dEtaBB_over_all"] = {"label"      : "#Delta#eta(Z,H)",
                                                                                "legend"     : 5,
                                                                                "color"      : "ROOT.kRed",
                                                                                "drawOptions" : "HIST PC",
                                                                                "marker"      : "20"}
        self.samplesAbs[files[fileName.name].replace("hists.root","accxEff.root")]["dEtaBB_"+region.name+"_over_all"] = {"label"      : region.title,
                                                                                              "legend"     : 6,
                                                                                              "color"      : "ROOT.kRed+1",
                                                                                              "drawOptions" : "HIST PC",
                                                                                              "marker"      : "20"}
        self.samplesAbs[files[fileName.name].replace("hists.root","accxEff.root")]["dEtaBB_"+region.name+"_HLT_over_all"] = {"label"      : "HLT",
                                                                                                  "legend"     : 7,
                                                                                                  "color"      : "ROOT.kBlack",
                                                                                                  "drawOptions" : "HIST PC",
                                                                                                  "marker"      : "20"}

        self.parametersAbs = {"titleLeft"       : "#scale[0.7]{#bf{CMS} Internal}",
                           #"titleCenter"      : "#scale[0.7]{"+region.title+"}",
                           #"titleRight"      : "#scale[0.7]{"+cut.title+"}",
                           "titleRight"   : "#scale[0.7]{"+fileName.title+"}",
                           "yTitle"     : "Acceptance #times Efficiency",
                           "xTitle"     : "True m_{4b} [GeV]",
                           "xTitleOffset": 0.95,
                           "legendTextSize":0.03,
                           "legendColumns":2,
                           "canvasSize" : [600,500],
                           "rMargin"    : 0.05,
                           "lMargin"    : 0.12,
                           "yTitleOffset": 1.1,
                              "xMin" : 181,
                              "xMax" : 888,
                           "yMin"       : 0.001,
                           "yMax"       : 4,
                           "xleg"       : [0.15,0.71],
                           "yleg"       : [0.77,0.92],
                              "labelSize"  : 16,
                           "logY"       : True,
                           "outputDir"   : outputPlot+topDir+"/"+year+"/",
                           "outputName" : "absoluteAccxEff",
                           }

        self.samplesRel=collections.OrderedDict()
        self.samplesRel[files[fileName.name].replace("hists.root","accxEff.root")] = collections.OrderedDict()
        self.samplesRel[files[fileName.name].replace("hists.root","accxEff.root")]["jetMultiplicity_over_all"] = {"label"      : "#geq4 Jets",
                                                                                         "legend"     : 1,
                                                                                         "color"      : "ROOT.kViolet",
                                                                                         "drawOptions" : "HIST PC",
                                                                                         "marker"      : "20"}
        self.samplesRel[files[fileName.name].replace("hists.root","accxEff.root")]["bTags_over_jetMultiplicity"] = {"label"      : "#geq4 b-Tags / #geq4 Jets",
                                                                               "legend"     : 2,
                                                                               "color"      : "ROOT.kBlue",
                                                                               "drawOptions" : "HIST PC",
                                                                               "marker"      : "20"}
        self.samplesRel[files[fileName.name].replace("hists.root","accxEff.root")]["MDRs_over_bTags"] = {"label"      : "Boson #DeltaR(j,j) / #geq4 b-Tags",
                                                                              "legend"     : 3,
                                                                              "color"      : "ROOT.kGreen",
                                                                              "drawOptions" : "HIST PC",
                                                                              "marker"      : "20"}
        self.samplesRel[files[fileName.name].replace("hists.root","accxEff.root")]["MDCs_over_MDRs"] = {"label"      : "Boson p_{T} / #DeltaR(j,j)",
                                                                              "legend"     : 4,
                                                                              "color"      : "ROOT.kOrange",
                                                                              "drawOptions" : "HIST PC",
                                                                              "marker"      : "20"}
        self.samplesRel[files[fileName.name].replace("hists.root","accxEff.root")]["dEtaBB_over_MDCs"] = {"label"      : "#Delta#eta(Z,H) / Boson p_{T}",
                                                                                "legend"     : 5,
                                                                                "color"      : "ROOT.kRed",
                                                                                "drawOptions" : "HIST PC",
                                                                                "marker"      : "20"}
        self.samplesRel[files[fileName.name].replace("hists.root","accxEff.root")]["dEtaBB_"+region.name+"_over_dEtaBB"] = {"label"      : region.title+" / #Delta#eta(Z,H)",
                                                                                              "legend"     : 6,
                                                                                              "color"      : "ROOT.kRed+1",
                                                                                              "drawOptions" : "HIST PC",
                                                                                              "marker"      : "20"}
        self.samplesRel[files[fileName.name].replace("hists.root","accxEff.root")]["dEtaBB_"+region.name+"_HLT_over_dEtaBB_"+region.name] = {"label"      : "HLT / "+region.title,
                                                                                                  "legend"     : 7,
                                                                                                  "color"      : "ROOT.kBlack",
                                                                                                  "drawOptions" : "HIST PC",
                                                                                                  "marker"      : "20"}

        self.parametersRel = {"titleLeft"       : "#scale[0.7]{#bf{CMS} Internal}",
                           #"titleCenter"      : "#scale[0.7]{"+region.title+"}",
                           #"titleRight"      : "#scale[0.7]{"+cut.title+"}",
                           "titleRight"   : "#scale[0.7]{"+fileName.title+"}",
                           "yTitle"     : "Acceptance #times Efficiency",
                           "xTitle"     : "True m_{4b} [GeV]",
                           "xTitleOffset": 0.95,
                           "legendTextSize":0.03,
                           "legendColumns":2,
                           "canvasSize" : [600,500],
                           "rMargin"    : 0.05,
                           "lMargin"    : 0.12,
                           "yTitleOffset": 1.1,
                           "yMin"       : 0.0,
                           "yMax"       : 1.3,
                           "xleg"       : [0.15,0.71],
                           "yleg"       : [0.77,0.92],
                           "labelSize"  : 16,
                           "logY"       : False,
                           "drawLines"  : [[100,1,2000,1]],
                           "outputDir"   : outputPlot+topDir+"/"+year+"/",
                           "outputName" : "relativeAccxEff",
                           }
    
    def plot(self, debug = False):
        PlotTools.plot(self.samplesAbs, self.parametersAbs, o.debug or debug)
        PlotTools.plot(self.samplesRel, self.parametersRel, o.debug or debug)





if False:
    fileName = nameTitle("ggZH4b"+o.year, "gg#rightarrowZH#rightarrowb#bar{b}b#bar{b}")
    region = nameTitle("ZHSR", "X_{ZH} < 1.5")
    plots.append(accxEffPlot("ggZH4b", fileName, o.year, region))

    fileName = nameTitle("ZH4b"+o.year, "ZH#rightarrowb#bar{b}b#bar{b}")
    region = nameTitle("ZHSR", "X_{ZH} < 1.5")
    plots.append(accxEffPlot("ZH4b", fileName, o.year, region))


nPlots=len(plots)
for p in range(nPlots):
    sys.stdout.write("\rMade "+str(p+1)+" of "+str(nPlots)+" | "+str(int((p+1)*100.0/nPlots))+"% ")
    sys.stdout.flush()
    plots[p].plot()
print
