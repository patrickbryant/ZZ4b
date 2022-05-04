import sys, gc
import time
import collections
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import PlotTools
from ROOT import TFile
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

CMSSW = getCMSSW()
USER = getUSER()
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"
CONDOROUTPUTBASE = "/store/user/"+USER+"/condor/"
TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"

class nameTitle:
    def __init__(self, name, title):
        self.name  = name
        self.title = title

parser = optparse.OptionParser()
parser.add_option('-d', '--debug',                dest="debug",         action="store_true", default=False, help="debug")
parser.add_option('-y', '--year',                 dest="year",          default="2018", help="Year specifies trigger (and lumiMask for data)")
parser.add_option('-l', '--lumi',                 dest="lumi",          default="1",    help="Luminosity for MC normalization: units [pb]")
parser.add_option('-i', '--inputBase',            dest="inputBase",    default="None", help="Base path for where to get raw histograms")
parser.add_option('-o', '--outputBase',           dest="outputBase",    default="/uscms/home/"+USER+"/nobackup/ZZ4b/plots/", help="Base path for storing output plots")
parser.add_option('-p', '--plotDir',              dest="plotDir",       default="plots/", help="Base path for storing output plots")
parser.add_option('-j',            action="store_true", dest="useJetCombinatoricModel",       default=False, help="make plots after applying jetCombinatoricModel")
parser.add_option('-r',            action="store_true", dest="reweight",       default=False, help="make plots after reweighting by FvTWeight")
parser.add_option('-a',            action="store_true", dest="doAccxEff",      default=False, help="Make Acceptance X Efficiency plots")
parser.add_option('-m',            action="store_true", dest="doMain",      default=False, help="Make main plots")
parser.add_option('--data',        default=None, help="data file override")
parser.add_option('--data3b',      default=None, help="data3b file override")
parser.add_option('--TT',          default=None, help="TT file override")
parser.add_option('--ZZ',          default=None, help="ZZ file override")
parser.add_option('--ZH',          default=None, help="ZH file override")
parser.add_option('--ggZH',          default=None, help="ggZH file override")
parser.add_option('--bothZH',          default=None, help="bothZH file override")
parser.add_option('--ZZandZH',          default=None, help="ZZandZH file override")
parser.add_option('--qcd',         default=None, help="qcd file override")
parser.add_option('--noSignal',    action="store_true", help="dont plot signal")
parser.add_option('--doJECSyst',   action="store_true", dest="doJECSyst",      default=False, help="plot JEC variations")
parser.add_option('--histDetailLevel',  default="passPreSel,fourTag,SB,SR,SBSR,ttbar3b",      help="")
parser.add_option('--rMin',  default=0.9,      help="")
parser.add_option('--rMax',  default=1.1,      help="")
parser.add_option('--mixed',        default=None, help="mixed file override")
parser.add_option('--mixedSamples',        default=None, help="mixed file override")
parser.add_option('--mixedNames',        default=None, help="mixed file override")
parser.add_option('--mixedSamplesDen',        default=None, help="mixed file override")


o, a = parser.parse_args()
onlySignal2D = False
#make sure outputBase ends with /
outputBase = o.outputBase + ("" if o.outputBase[-1] == "/" else "/")
inputBase = outputBase
if o.inputBase != "None":
    inputBase = o.inputBase + ("" if o.inputBase[-1] == "/" else "/")
    
outputPlot = outputBase+o.plotDir + ("" if o.plotDir[-1] == "/" else "/")
print "Plot output:",outputPlot

lumiDict   = {"2016":  35.9e3,#35.8791
              "2017":  36.7e3,#36.7338
              "2018":  60.0e3,#59.9656
              #"17+18": 96.7e3,
              "RunII":132.6e3,
              }

lumi = float(o.lumi)/1000


# Jet Combinatoric Model
gitRepoBase= 'ZZ4b/nTupleAnalysis/weights/'
JCMRegion = "SB"
JCMVersion = "00-00-02"
def jetCombinatoricModel(year):
    return gitRepoBase+"data"+year+"/jetCombinatoricModel_"+JCMRegion+"_"+JCMVersion+".txt"

# jcm2016 = PlotTools.read_parameter_file(jetCombinatoricModel('2016'))
# jcm2017 = PlotTools.read_parameter_file(jetCombinatoricModel('2017'))
# jcm2018 = PlotTools.read_parameter_file(jetCombinatoricModel('2018'))

# JCMCut = 'passMDRs'
# mu_qcd = {'2016' : jcm2016['mu_qcd_'+JCMCut],#0.10446505802,#0.105369016424,
#           '2017' : jcm2017['mu_qcd_'+JCMCut],#0.171252317592,#0.172460552209,
#           '2018' : jcm2018['mu_qcd_'+JCMCut],#0.156187958018,#0.144728579751,
#           }
# # mu_qcd['17+18']  = mu_qcd['2017'] * lumiDict['2017']/(lumiDict['2017']+lumiDict['2018']) 
# # mu_qcd['17+18'] += mu_qcd['2018'] * lumiDict['2018']/(lumiDict['2017']+lumiDict['2018'])

# mu_qcd['RunII']  = mu_qcd['2016'] * lumiDict['2016']/(lumiDict['2016']+lumiDict['2017']+lumiDict['2018'])
# mu_qcd['RunII'] += mu_qcd['2017'] * lumiDict['2017']/(lumiDict['2016']+lumiDict['2017']+lumiDict['2018'])
# mu_qcd['RunII'] += mu_qcd['2018'] * lumiDict['2018']/(lumiDict['2016']+lumiDict['2017']+lumiDict['2018'])

jcm = PlotTools.read_parameter_file(jetCombinatoricModel('RunII'))
mu_qcd = jcm['mu_qcd_passPreSel']

files = {"data"+o.year  : inputBase+"data"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root",
         # "ZH4b"+o.year   : inputBase+"ZH4b"+o.year+"/hists.root",
         # "ggZH4b"+o.year : inputBase+"ggZH4b"+o.year+"/hists.root",
         "bothZH4b"+o.year : inputBase+"bothZH4b"+o.year+"/hists.root",
         "ZZZHHH4b"+o.year : inputBase+"ZZZHHH4b"+o.year+"/hists.root",
         "ZZ4b"+o.year   : inputBase+"ZZ4b"+o.year+"/hists.root",
         "HH4b"+o.year   : inputBase+"HH4b"+o.year+"/hists.root",
         #"TTJets"+o.year : inputBase+"TTJets"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root",
         "TT"+o.year : inputBase+"TT"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root",
         "qcd"+o.year : inputBase+"qcd"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root",
         }

if not o.reweight:
    files["qcd"+o.year] = inputBase+"qcd"+o.year+"/hists"+("_j" if o.useJetCombinatoricModel else "")+".root"


if o.doAccxEff:
    del files["data"+o.year]
    del files["TT"+o.year]
    del files['qcd'+o.year]
    for name, path in files.items():
        files[name] = path.replace('hists.root','accxEff.root')


#
#  Command Line overrides
#
if o.data is not None:
    print "Using data file",o.data
    files["data"+o.year] = o.data

if o.data3b is not None:
    print "Using data3b file",o.data3b
    files["data3b"+o.year] = o.data3b

if o.qcd is not None and 'fourTag' in o.histDetailLevel:
    print "Using qcd file",o.qcd
    files["qcd"+o.year] = o.qcd

if o.TT is not None:
    print "Using TT file",o.TT
    files["TT"+o.year] = o.TT

if o.ZZ is not None:
    print "Using ZZ file",o.ZZ
    files["ZZ4b"+o.year] = o.ZZ

if o.ZH is not None:
    print "Using ZH file",o.ZH
    files["ZH4b"+o.year] = o.ZH

if o.ggZH is not None:
    print "Using ggZH file",o.ggZH
    files["ggZH4b"+o.year] = o.ggZH

if o.bothZH is not None:
    print "Using ggZH file",o.bothZH
    files["bothZH4b"+o.year] = o.bothZH

if o.ZZandZH is not None:
    print "Using ZZandZH file",o.ZZandZH
    files["ZZZHHH4b"+o.year] = o.ZZandZH


mixedNames   = []



if o.mixedSamples is not None:
    
    mixedNames   = o.mixedNames.split(",")
    mixedSamples = o.mixedSamples.split(",")

    for mItr, mName in enumerate(mixedNames):
        print "Using mixed file",mixedSamples[mItr],"with name",mName
        files["mixed"+mName+o.year] = mixedSamples[mItr]

        mixedDenoms  = o.mixedSamplesDen.split(",")
        print "Using mixedDen file",mixedDenoms[mItr],"with name",mName
        files["mixedDenom"+mName+o.year] = mixedDenoms[mItr]



if o.noSignal:
    #del files["ZH4b"+o.year]
    #del files["ggZH4b"+o.year]
    del files["bothZH4b"+o.year]
    del files["ZZ4b"+o.year]
    #del files["ZZandZH4b"+o.year]
    del files["HH4b"+o.year]
    del files["ZZZHHH4b"+o.year]

# for sample in files:
#     files[sample] = TFile.Open(files[sample])

JECSysts = [nameTitle("_jerUp", "JER Up"), nameTitle("_jerDown", "JER Down"),
            nameTitle("_jesTotalUp", "JES Up"), nameTitle("_jesTotalDown", "JES Down")]


# DiJet Mass Plane Region Definitions
leadStBias = 1.02  # leading st dijet mass peak is shifted up by a few percent
sublStBias = 0.98  # sub-leading st dijet mass peak is shifted down by a few percent
mZ =  91.0 
mH = 125.0 
leadH = mH * leadStBias
sublH = mH * sublStBias
leadZ = mZ * leadStBias
sublZ = mZ * sublStBias

xMaxZZSR =  2.60 
rMaxZZCR = 28.00 
sZZCR =  1.01 
rMaxZZSB = 40.00 
sZZSB =  1.02 

xMaxZHSR =  1.90 
rMaxZHCR = 30.00 
sZHCR =  1.04 
rMaxZHSB = 45.00 
sZHSB =  1.06 

xMaxHHSR =  1.90 
rMaxHHCR = 30.00 
sHHCR =  1.04 
rMaxHHSB = 45.00 
sHHSB =  1.06 

SRs = [["(((x-"+str(leadH)+")/(0.1*x))**2 +((y-"+str(sublH)+")/(0.1*y))**2)", 0,250,0,250,[xMaxHHSR**2],"ROOT.kRed",     7],
       ["(((x-"+str(leadH)+")/(0.1*x))**2 +((y-"+str(sublZ)+")/(0.1*y))**2)", 0,250,0,250,[xMaxZHSR**2],"ROOT.kRed",     7],
       ["(((x-"+str(leadZ)+")/(0.1*x))**2 +((y-"+str(sublH)+")/(0.1*y))**2)", 0,250,0,250,[xMaxZHSR**2],"ROOT.kRed",     7],
       ["(((x-"+str(leadZ)+")/(0.1*x))**2 +((y-"+str(sublZ)+")/(0.1*y))**2)", 0,250,0,250,[xMaxZZSR**2],"ROOT.kRed",     7]]


cutDict = {
    "passPreSel"    : nameTitle("passPreSel", "Preselection"), 
    "passDijetMass" : nameTitle("passDijetMass", "Pass m(j,j) Cuts"), 
    "passMDRs"      : nameTitle("passMDRs", "Pass #DeltaR(j,j)"), 
    "passTTCR"      : nameTitle("passTTCR", "t#bar{t} CR"), 
    "passMuon"      : nameTitle("passMuon", "nIsoMed25Muons>0"), 
    "passSvB"       : nameTitle("passSvB", "Pass SvB"), 
    "passMjjOth"    : nameTitle("passMjjOth", "Pass #M(j,j)"), 
    "passXWt"       : nameTitle("passXWt", "rWbW > 3"), 
    "passMDCs"      : nameTitle("passMDCs", "Pass MDC's"), 
    "passDEtaBB"    : nameTitle("passDEtaBB", "|#Delta#eta| < 1.5"),
    }

cuts = []
for c in o.histDetailLevel.split(","):
    if c in cutDict:
        cuts.append(cutDict[c])
# if onlySignal2D:
#     cuts.append(cutDict['passPreSel'])

print "Plotting cuts"
for c in cuts:
    print "\t",c.name

#    cuts = [#nameTitle("passPreSel", "Preselection"), 
#        #nameTitle("passDijetMass", "Pass m(j,j) Cuts"), 
#        nameTitle("passMDRs", "Pass #DeltaR(j,j)"), 
#        #nameTitle("passMjjOth", "Pass #M(j,j)"), 
#        #nameTitle("passXWt", "rWbW > 3"), 
#        # nameTitle("passMDCs", "Pass MDC's"), 
#        # nameTitle("passDEtaBB", "|#Delta#eta| < 1.5"),
#        ]

views = [#"allViews",
         "mainView",
         ]
if onlySignal2D: views.append('allViews')

regionDict = {
    "inclusive" : nameTitle("inclusive", ""),
    'notSR' : nameTitle('notSR', 'Not SR'),
    'outSB' : nameTitle('outSB', 'Outer SB'),
    "ZH" : nameTitle("ZH", "ZH SB+CR+SR"),
    "ZH_SvB_high" : nameTitle("ZH_SvB_high", "ZH SB+CR+SR SvB>0.5"), 
    "ZH_SvB_low" : nameTitle("ZH_SvB_low", "ZH SB+CR+SR SvB<0.5"),
    "ZHSB" : nameTitle("ZHSB", "ZH Sideband"), 
    "ZHCR" : nameTitle("ZHCR", "ZH Control Region"), 
    "ZHSR" : nameTitle("ZHSR", "ZH Signal Region"),
    "ZZ" : nameTitle("ZZ", "ZZ SB+CR+SR"),
    "ZZ_SvB_high" : nameTitle("ZZ_SvB_high", "ZZ SB+CR+SR SvB>0.5"), 
    "ZZ_SvB_low" : nameTitle("ZZ_SvB_low", "ZZ SB+CR+SR SvB<0.5"),
    "ZZSB" : nameTitle("ZZSB", "ZZ Sideband"), 
    "ZZCR" : nameTitle("ZZCR", "ZZ Control Region"), 
    "ZZSR" : nameTitle("ZZSR", "ZZ Signal Region"),
    "SBSR" : nameTitle("SBSR", "SB and SR"),
    "SB" : nameTitle("SB", "Sideband"), 
    "CR" : nameTitle("CR", "Control Region"), 
    "SRNoHH" : nameTitle("SRNoHH", "Signal Region (HH Veto)"),
    "SR" : nameTitle("SR", "Signal Region"), 
    "HHSR" : nameTitle("HHSR", "HH Signal Region"),
    }

regions = []
for r in o.histDetailLevel.split(","):
    if r in regionDict:
        regions.append(regionDict[r])

print "Plotting regions"
for r in regions:
    print "\t",r.name

plots=[]


class variable:
    def __init__(self, name, xTitle, yTitle = None, zTitle = None, rebin = None, divideByBinWidth = False, normalize = None, normalizeStack = None, mu_qcd=1, logy=False):
        self.name   = name
        self.xTitle = xTitle
        self.yTitle = yTitle
        self.zTitle = zTitle
        self.rebin  = rebin
        self.logy  = logy
        self.divideByBinWidth = divideByBinWidth
        self.normalize = normalize
        self.normalizeStack = normalizeStack
        self.mu_qcd = mu_qcd

class standardPlot:
    def __init__(self, year, cut, view, region, var):
        self.samples=collections.OrderedDict()
        if len(mixedNames) == 1:
            self.samples[files[  "mixed"+mName+year]] = collections.OrderedDict()
            self.samples[files[  "mixedDenom"+mName+year]] = collections.OrderedDict()

        self.samples[files[    "data"+year]] = collections.OrderedDict()
        if o.reweight:
            multijet = "data"+year
            if "data3b"+year in files:
                multijet = "data3b"+year
                self.samples[files[     "data3b"+year]] = collections.OrderedDict()
        if not o.reweight:
            multijet = "qcd"+year
            self.samples[files[     "qcd"+year]] = collections.OrderedDict()
        self.samples[files[      "TT"+year]] = collections.OrderedDict()

        if "bothZH4b"+year in files:
            self.samples[files["bothZH4b"+year]] = collections.OrderedDict()
        if "ZZ4b"+year in files:
            self.samples[files[    "ZZ4b"+year]] = collections.OrderedDict()
        if "HH4b"+year in files:
            self.samples[files[    "HH4b"+year]] = collections.OrderedDict()

        if len(mixedNames) == 1:
            self.samples[files[  "mixed"+mName+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label" : "Mixed "+mName,
                "legend": 8,
                "isData" : False,
                "ratio" : "denom B",
                "color" : "ROOT.kRed"}

            self.samples[files[  "mixedDenom"+mName+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label" : "MixedDenom "+mName,
                "legend": 1,
                "isData" : True,
                "ratio" : "numer B",
                "color" : "ROOT.kRed"}

        self.samples[files[  "data"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
            "label" : ("Data %.1f/fb, "+year)%(lumi),
            "legend": 1,
            "isData" : True,
            "ratio" : "numer A",
            "color" : "ROOT.kBlack"}
        self.samples[files[multijet]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
            "label" : "Multijet Model",
            "weight": var.mu_qcd,
            "legend": 2,
            "stack" : 3,
            "ratio" : "denom A",
            "color" : "ROOT.kYellow"}
        self.samples[files["TT"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
            "label" : "t#bar{t}",
            "legend": 3,
            "stack" : 2,
            "ratio" : "denom A",
            "color" : "ROOT.kAzure-9"}


        # signalCut = cutDict['passMDRs']
        if "HH4b"+year in files:
            self.samples[files[    "HH4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label"    : "HH#rightarrowb#bar{b}b#bar{b} (#times100)",
                "legend"   : 5,
                "weight" : 100,
                "color"    : "ROOT.kBlue+1"}

        if "bothZH4b"+year in files:
            self.samples[files["bothZH4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label"    : "ZH#rightarrowb#bar{b}b#bar{b} (#times100)",
                "legend"   : 6,
                "weight" : 100,
                "color"    : "ROOT.kRed"}

        if "ZZ4b"+year in files:
            self.samples[files[    "ZZ4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label"    : "ZZ#rightarrowb#bar{b}b#bar{b} (#times100)",
                "legend"   : 7,
                "weight" : 100,
                "color"    : "ROOT.kGreen+3"}


        rMin = float(o.rMin)
        rMax = float(o.rMax)
        if not o.reweight or not o.year == "RunII":
            rMin = 0.5
            rMax = 1.5

        if cut.name in ["passMjjOth","passSvB"]:
            rMin = 0.5
            rMax = 1.5
        

        self.parameters = {"titleLeft"   : "#bf{CMS} Internal",
                           "titleCenter" : region.title,
                           "titleRight"  : cut.title,
                           "maxDigits"   : 4,
                           "ratio"     : True,
                           "rMin"      : rMin,
                           "rMax"      : rMax,
                           "rTitle"    : "Data / Bkgd.",
                           "xTitle"    : var.xTitle,
                           "yTitle"    : ("Events" if view != "allViews" else "Pairings") if not var.yTitle else var.yTitle,
                           "outputDir" : outputPlot+"data/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName": var.name}
        if var.divideByBinWidth: self.parameters["divideByBinWidth"] = True
        if var.rebin: self.parameters["rebin"] = var.rebin
        if var.logy: self.parameters["logY"] = True
        if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack
        #if 'SvB' in var.name and 'SR' in region.name: self.parameters['xleg'] = [0.6, 0.6+0.33]

    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, o.debug or debug)


class threeTagPlot:
    def __init__(self, year, cut, view, region, var):
        self.samples=collections.OrderedDict()
        self.samples[files[    "data"+year]] = collections.OrderedDict()
        self.samples[files[      "TT"+year]] = collections.OrderedDict()

        if "bothZH4b"+year in files:
            self.samples[files["bothZH4b"+year]] = collections.OrderedDict()
        if "ZZ4b"+year in files:
            self.samples[files[    "ZZ4b"+year]] = collections.OrderedDict()
        if "HH4b"+year in files:
            self.samples[files[    "HH4b"+year]] = collections.OrderedDict()

        self.samples[files[  "data"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
            "label" : "3b Data %.1f/fb, %s"%(lumi,year),
            "legend": 1,
            "isData" : True,
            "ratio" : "numer A",
            "color" : "ROOT.kBlack"}
        self.samples[files["TT"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
            "label" : "3b t#bar{t}",
            "legend": 2,
            "stack" : 1,
            "ratio" : "denom A",
            "color" : "ROOT.kAzure-9"}

        # signalCut = cutDict['passMDRs']
        if "HH4b"+year in files:
            self.samples[files[    "HH4b"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
                "label"    : "3b HH#rightarrowb#bar{b}b#bar{b} (#times100)",
                "legend"   : 5,
                "weight" : 100,
                "color"    : "ROOT.kBlue+1"}
        if "bothZH4b"+year in files:
            self.samples[files["bothZH4b"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
                "label"    : "3b ZH#rightarrowb#bar{b}b#bar{b} (#times100)",
                "legend"   : 6,
                "weight" : 100,
                "color"    : "ROOT.kRed"}
        if "ZZ4b"+year in files:
            self.samples[files[    "ZZ4b"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
                "label"    : "3b ZZ#rightarrowb#bar{b}b#bar{b} (#times100)",
                "legend"   : 7,
                "weight" : 100,
                "color"    : "ROOT.kGreen+3"}

        self.parameters = {"titleLeft"   : "#bf{CMS} Internal",
                           "titleCenter" : region.title,
                           "titleRight"  : cut.title,
                           "maxDigits"   : 4,
                           "ratio"     : True,
                           "rMin"      : 0,#0.5 if (not o.reweight and not o.year == "RunII") else float(o.rMin),
                           "rMax"      : 2,#1.5 if (not o.reweight and not o.year == "RunII") else float(o.rMax),
                           "rTitle"    : "Data / t#bar{t}",
                           "xTitle"    : var.xTitle,
                           "yTitle"    : ("Events" if view != "allViews" else "Views") if not var.yTitle else var.yTitle,
                           "outputDir" : outputPlot+"data3b/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName": var.name}
        if var.divideByBinWidth: self.parameters["divideByBinWidth"] = True
        if var.rebin: self.parameters["rebin"] = var.rebin
        if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack
        if 'SvB' in var.name and 'SR' in region.name: self.parameters['xleg'] = [0.3, 0.3+0.33]

    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, o.debug or debug)




class mcPlot:
    def __init__(self, year, cut, view, region, var):
        self.samples=collections.OrderedDict()
        self.samples[files[    "TT"+year]] = collections.OrderedDict()
        self.samples[files[    "HH4b"+year]] = collections.OrderedDict()
        self.samples[files["bothZH4b"+year]] = collections.OrderedDict()
        self.samples[files[    "ZZ4b"+year]] = collections.OrderedDict()

        self.samples[files["TT"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
            "label" : "t#bar{t} (3-tag #times0.25)",
            "legend": 1,
            "ratio" : "denom A",
            'weight': 0.25,
            "color" : "ROOT.kAzure-9"}
        self.samples[files["TT"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
            "label" : "t#bar{t} (4-tag)",
            "drawOptions" : "PE ex0",
            "legend": 2,
            "ratio" : "numer A",
            "color" : "ROOT.kAzure-9"}

        # signalCut = cutDict['passMDRs']
        self.samples[files["HH4b"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
            "label"    : "HH#rightarrowb#bar{b}b#bar{b} (3-tag #times1000)",
            "legend"   : 5,
            "ratio" : "denom B",
            "weight" : 1000,
            "color"    : "ROOT.kBlue+1"}
        self.samples[files["HH4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
            "label"    : "HH#rightarrowb#bar{b}b#bar{b} (4-tag #times100)",
            "drawOptions" : "PE ex0",
            "legend"   : 6,
            "ratio" : "numer B",
            "weight" : 100,
            "color"    : "ROOT.kBlue+1"}

        self.samples[files["bothZH4b"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
            "label"    : "ZH#rightarrowb#bar{b}b#bar{b} (3-tag #times1000)",
            "legend"   : 7,
            "ratio" : "denom C",
            "weight" : 1000,
            "color"    : "ROOT.kRed"}
        self.samples[files["bothZH4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
            "label"    : "ZH#rightarrowb#bar{b}b#bar{b} (4-tag #times100)",
            "drawOptions" : "PE ex0",
            "legend"   : 8,
            "ratio" : "numer C",
            "weight" : 100,
            "color"    : "ROOT.kRed"}
        
        self.samples[files["ZZ4b"+year]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {
            "label"    : "ZZ#rightarrowb#bar{b}b#bar{b} (3-tag #times1000)",
            "legend"   : 9,
            "ratio" : "denom D",
            "weight" : 1000,
            "color"    : "ROOT.kGreen+3"}
        self.samples[files["ZZ4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
            "label"    : "ZZ#rightarrowb#bar{b}b#bar{b} (4-tag #times100)",
            "drawOptions" : "PE ex0",
            "legend"   : 10,
            "ratio" : "numer D",
            "weight" : 100,
            "color"    : "ROOT.kGreen+3"}

        self.parameters = {"titleLeft"   : "#bf{CMS} Simulation Internal",
                           "titleCenter" : region.title,
                           "titleRight"  : cut.title,
                           "ratio"     : True,
                           "rMin"      : 0,
                           "rMax"      : 2,
                           "rTitle"    : "Four / Three",
                           "xTitle"    : var.xTitle,
                           "yTitle"    : ("Events" if view != "allViews" else "Views") if not var.yTitle else var.yTitle,
                           "outputDir" : outputPlot+"mc/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName": var.name}
        if var.divideByBinWidth: self.parameters["divideByBinWidth"] = True
        if var.rebin: self.parameters["rebin"] = var.rebin
        if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack

    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, o.debug or debug)


class JECPlot:
    def __init__(self, year, cut, view, region, var):
        self.samples=collections.OrderedDict()
        self.samples[files["bothZH4b"+year]] = collections.OrderedDict()
        self.samples[files[    "ZZ4b"+year]] = collections.OrderedDict()

        self.samples[files["bothZH4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
            "label"    : "ZH Nominal",
            #"drawOptions" : "HIST",
            "legend"   : 1,
            "ratio" : "denom A",
            "weight" : 1,
            "color"    : "ROOT.kRed"}
        
        self.samples[files["ZZ4b"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
            "label"    : "ZZ Nominal",
            #"drawOptions" : "HIST",
            "legend"   : 2,
            "ratio" : "denom B",
            "weight" : 1,
            "color"    : "ROOT.kGreen+3"}

        markers = ['2','3','4','5']
        for i, JECSyst in enumerate(JECSysts):
            bothZH4bFile = files["bothZH4b"+year].replace('.root', JECSyst.name+'.root')
            ZZ4bFile     = files[    "ZZ4b"+year].replace('.root', JECSyst.name+'.root')
            self.samples[bothZH4bFile] = collections.OrderedDict()
            self.samples[ZZ4bFile    ] = collections.OrderedDict()

            self.samples[bothZH4bFile][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label"    : "ZH"+JECSyst.title,
                "drawOptions" : "PE ex0",
                "legend"   : 2*i + 3,
                "ratio" : "numer A",
                "weight" : 1,
                "marker" : markers[i],
                "color"    : "ROOT.kRed"}

            self.samples[ZZ4bFile][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label"    : "ZZ"+JECSyst.title,
                "drawOptions" : "PE ex0",
                "legend"   : 2*i + 4,
                "ratio" : "numer B",
                "weight" : 1,
                "marker" : markers[i],
                "color"    : "ROOT.kGreen+3"}


        self.parameters = {"titleLeft"   : "#bf{CMS} Simulation Internal",
                           "titleCenter" : region.title,
                           "titleRight"  : cut.title,
                           "ratio"     : True,
                           "rMin"      : 0,
                           "rMax"      : 2,
                           "rTitle"    : "Syst. / Nom.",
                           "xTitle"    : var.xTitle,
                           "yTitle"    : ("Events" if view != "allViews" else "Views") if not var.yTitle else var.yTitle,
                           "outputDir" : outputPlot+"mc/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName": var.name+"_JECSysts"}
        if var.divideByBinWidth: self.parameters["divideByBinWidth"] = True
        if var.rebin: self.parameters["rebin"] = var.rebin
        if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack

    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, o.debug or debug)


sublMDRs = [["(235./x     - y)",100,1100,0,5,[0],"ROOT.kRed",1],
            ["(650./x+0.7 - y)",100,1100,0,5,[0],"ROOT.kRed",1]]
leadMDRs = [["(360./x-0.5 - y)",100,1100,0,5,[0],"ROOT.kRed",1],
            ["(650./x+0.5 - y)",100,1100,0,5,[0],"ROOT.kRed",1]]

class TH2Plot:
    def __init__(self, topDir, fileName, year, cut, tag, view, region, var, debug=False):
        self.debug = debug
        self.samples=collections.OrderedDict()

        if tag=="Background":
            if o.reweight:
                multijet = "data"+year
                if "data3b"+year in files:
                    multijet = "data3b"+year
            else:
                multijet = "qcd"+year
            self.samples[files[ multijet]] = collections.OrderedDict()
            self.samples[files["TT"+year]] = collections.OrderedDict()

            self.samples[files[multijet]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {"drawOptions": "COLZ",
                                                                                                      "stack" : 2}
            self.samples[files["TT"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"drawOptions": "COLZ",
                                                                                                      "stack" : 1}
        else:
            self.samples[files[fileName.name]] = collections.OrderedDict()
            self.samples[files[fileName.name]][cut.name+"/"+tag+"/"+view+"/"+region.name+"/"+var.name] = {"drawOptions": "COLZ"}
                
        self.parameters = {"titleLeft"       : "#scale[0.7]{#bf{CMS} Internal}",
                           "titleCenter"      : "#scale[0.7]{"+region.title+"}",
                           "titleRight"      : "#scale[0.7]{"+cut.title+"}",
                           "subTitleRight"   : "#scale[0.7]{"+fileName.title+"}",
                           "yTitle"      : var.yTitle,
                           "xTitle"      : var.xTitle,
                           "zTitle"      : ("Events / Bin" if view != "allViews" else "Pairings / Bin"),
                           "zTitleOffset": 1.4,
                           "zMin"        : 0,
                           "maxDigits"   : 4,
                           "xNdivisions" : 505,
                           "yNdivisions" : 505,
                           "rMargin"     : 0.15,
                           "rebin"       : 1, 
                           "canvasSize"  : [720,660],
                           "outputDir"   : outputPlot+topDir+"/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName"  : var.name+"_"+tag,
                           }
        if var.name == 'leadSt_m_vs_sublSt_m':
            self.parameters["functions"] = SRs
        if 'm4j_vs_leadSt_dR' in var.name:
            self.parameters['functions'] = leadMDRs
        if 'm4j_vs_sublSt_dR' in var.name:
            self.parameters['functions'] = sublMDRs
        if 'm4j_vs_nViews' in var.name:
            self.parameters['yMin'], self.parameters['yMax'] = 0.5, 3.5
            self.parameters["yNdivisions"] = 003

    def newSample(self, topDir, fileName, year, cut, tag, view, region, var):
        self.samples[files[fileName.name]] = collections.OrderedDict()
        self.samples[files[fileName.name]][cut.name+"/"+tag+"/"+view+"/"+region.name+"/"+var.name] = {"drawOptions": "COLZ"}
        self.parameters["titleCenter"]   = "#scale[0.7]{"+region.title+"}"
        self.parameters["titleRight"]    = "#scale[0.7]{"+cut.title+"}"
        self.parameters["subTitleRight"] = "#scale[0.7]{"+fileName.title+"}"
        self.parameters["outputDir"]     = outputPlot+topDir+"/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/"
        self.parameters["outputName"]    = var.name+"_"+tag

    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, o.debug or debug or self.debug)
        if self.debug: raw_input()


class mixedVsDataPlot:
    def __init__(self, year, cut, view, region, var):
        self.samples=collections.OrderedDict()

        colors = ["ROOT.kBlue","ROOT.kRed","ROOT.kBlack"]

        for mItr, mName in enumerate(mixedNames):
            self.samples[files[    "mixed"+mName+year]] = collections.OrderedDict()
            self.samples[files[    "mixedDenom"+mName+year]] = collections.OrderedDict()
        
            if len(mixedNames) == 1:
                self.samples[files[  "mixed"+mName+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                    "label" : "Mixed "+mName,
                    "legend": mItr+2,
                    "stack" : 1,
                    "isData" : False,
                    "ratio" : "numer "+mName,
                    "color" : "ROOT.kYellow"}

                self.samples[files[  "mixedDenom"+mName+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                    "label" : "MixedDenom "+mName,
                    "legend": 1,
                    "isData" : True,
                    "ratio" : "denom "+mName,
                    "color" : "ROOT.kBlue"}


            else:
                self.samples[files[  "mixed"+mName+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                    "label" : "Mixed "+mName,
                    "legend": mItr+2,
                    "isData" : False,
                    "ratio" : "numer "+mName,
                    "color" : colors[mItr]}
                

                self.samples[files[  "mixedDenom"+mName+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                    "label" : "MixedDenom "+mName,
                    "legend": 1,
                    "isData" : True,
                    "ratio" : "denom "+mName,
                    "color" : "ROOT.kBlack"}



        self.samples[files[    "data"+year]] = collections.OrderedDict()

        if len(mixedNames) == 1:
            self.samples[files["data"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label" : ("Nominal Data %.1f/fb, "+year)%(lumi),
                "legend": 1,
                "isData" : True,
                "color" : "ROOT.kBlack"}

        else:
            self.samples[files["data"+year]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {
                "label" : ("Nominal Data %.1f/fb, "+year)%(lumi),
                "legend": 1,
                "isData" : True,
                "color" : "ROOT.kBlack"}
    


        rMin = 0.8
        rMax = 1.2
        if not o.reweight or not o.year == "RunII":
            rMin = 0.5
            rMax = 1.5

        if cut.name in ["passMjjOth"]:
            rMin = 0.5
            rMax = 1.5
        

        self.parameters = {"titleLeft"   : "#bf{CMS} Internal",
                           "titleCenter" : region.title,
                           "titleRight"  : cut.title,
                           "maxDigits"   : 4,
                           "ratio"     : True,
                           "rMin"      : rMin,
                           "rMax"      : rMax,
                           "rTitle"    : "Mixed / Nominal.",
                           "xTitle"    : var.xTitle,
                           "yTitle"    : ("Events" if view != "allViews" else "Views") if not var.yTitle else var.yTitle,
                           "outputDir" : outputPlot+"data/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName": var.name}

        if var.divideByBinWidth: self.parameters["divideByBinWidth"] = True
        if var.rebin: self.parameters["rebin"] = var.rebin
        if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack
        if 'SvB' in var.name and 'SR' in region.name: self.parameters['xleg'] = [0.3, 0.3+0.33]


    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, o.debug or debug)




variables=[variable("nPVs", "Number of Primary Vertices"),
           variable("nPVsGood", "Number of 'Good' Primary Vertices"),
           variable("nSelJets", "Number of Selected Jets"),
           variable("nSelJets_lowSt", "Number of Selected Jets (s_{T,4j} < 320 GeV)"),
           variable("nSelJets_midSt", "Number of Selected Jets (320 < s_{T,4j} < 450 GeV)"),
           variable("nSelJets_highSt", "Number of Selected Jets (s_{T,4j} > 450 GeV)"),
           variable("nSelJetsUnweighted", "Number of Selected Jets (Unweighted)", mu_qcd=mu_qcd),
           variable("nPSTJets", "Number of Tagged + Pseudo-Tagged Jets"),
           variable("nPSTJets_lowSt", "Number of Tagged + Pseudo-Tagged Jets (s_{T,4j} < 320 GeV)"),
           variable("nPSTJets_midSt", "Number of Tagged + Pseudo-Tagged Jets (320 < s_{T,4j} < 450 GeV)"),
           variable("nPSTJets_highSt", "Number of Tagged + Pseudo-Tagged Jets (s_{T,4j} > 450 GeV)"),
           variable("nTagJets", "Number of Tagged Jets"),
           variable("nAllJets", "Number of Jets"),
           variable("nIsoMed25Muons", "Number of Isolated Medium Muons (p_{T} > 25 GeV)"),
           variable("nIsoMed40Muons", "Number of Isolated Medium Muons (p_{T} > 40 GeV)"),
           variable("st", "Scalar sum of all jet p_{T}'s [GeV]"),
           variable("stNotCan", "Scalar sum of all other jet p_{T}'s [GeV]"),
           #variable("FvT", "Four vs Three Tag Classifier Output", rebin=[i/100.0 for i in range(0,45,5)]+[i/100.0 for i in range(45,55)]+[i/100.0 for i in range(55,101,5)], yTitle = "Events / 0.01 Output"),
           variable("FvT", "FvT Classifier Reweight", rebin = 2),
           variable("FvT_pd4", "FvT Regressed P(Four-tag Data)", rebin = 2),
           variable("FvT_pd3", "FvT Regressed P(Three-tag Data)", rebin = 2),
           variable("FvT_pt4", "FvT Regressed P(Four-tag t#bar{t})", rebin = 2),
           variable("FvT_pt3", "FvT Regressed P(Three-tag t#bar{t})", rebin = 2),
           variable("FvT_pm4", "FvT Regressed P(Four-tag Multijet)", rebin = 2),
           variable("FvT_pm3", "FvT Regressed P(Three-tag Multijet)", rebin = 2),
           variable("FvT_pt",  "FvT Regressed P(t#bar{t})", rebin = 2),
           # variable("bdtScore",  "KlBdT Score", rebin = 2),
           # variable("SvB_MA_VHH_ps", "SvB_MA VHH Regressed", rebin = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70 ,0.78, 0.86, 0.93, 0.97, 0.99, 1.0]),
           # variable("SvB_MA_VHH_ps_sbdt", "SvB_MA VHH Regressed small bdt", rebin = 2),
           # variable("SvB_MA_VHH_ps_lbdt", "SvB_MA VHH Regressed large bdt", rebin = 2),
           variable("SvB_ps",  "SvB Regressed P(Signal)", rebin = 1),
           variable("SvB_pzz", "SvB Regressed P(ZZ)", rebin = 1),
           variable("SvB_pzh", "SvB Regressed P(ZH)", rebin = 1),
           variable("SvB_phh", "SvB Regressed P(HH)", rebin = 1),
           variable("SvB_ptt", "SvB Regressed P(t#bar{t})", rebin = [0.02*i for i in range(21)]),
           variable("SvB_ps_hh",  "SvB Regressed P(Signal) #cbar P(HH) is largest", rebin = 1),
           variable("SvB_ps_zh",  "SvB Regressed P(Signal) #cbar P(ZH) is largest", rebin = 1),
           variable("SvB_ps_zz",  "SvB Regressed P(Signal) #cbar P(ZZ) is largest", rebin = 1),
           variable("SvB_MA_ps",  "SvB_MA Regressed P(Signal)", rebin = 1),
           variable("SvB_MA_pzz", "SvB_MA Regressed P(ZZ)", rebin = 1),
           variable("SvB_MA_pzh", "SvB_MA Regressed P(ZH)", rebin = 1),
           variable("SvB_MA_phh", "SvB_MA Regressed P(HH)", rebin = 1),
           variable("SvB_MA_ptt", "SvB_MA Regressed P(t#bar{t})", rebin = [0.02*i for i in range(21)]),
           variable("SvB_MA_ps_hh",  "SvB_MA Regressed P(Signal) #cbar P(HH) is largest", rebin = 1),
           variable("SvB_MA_ps_zh",  "SvB_MA Regressed P(Signal) #cbar P(ZH) is largest", rebin = 1),
           variable("SvB_MA_ps_zz",  "SvB_MA Regressed P(Signal) #cbar P(ZZ) is largest", rebin = 1),
           # variable("SvB_ps_zh_0_75",  "SvB Regressed P(Signal) #cbar P(ZH) is largest, 0<p_{T,Z}<75", rebin = 5),
           # variable("SvB_ps_zh_75_150",  "SvB Regressed P(Signal) #cbar P(ZH) is largest, 75<p_{T,Z}<150", rebin = 2),
           # variable("SvB_ps_zh_150_250",  "SvB Regressed P(Signal) #cbar P(ZH) is largest, 150<p_{T,Z}<250", rebin = 2),
           # variable("SvB_ps_zh_250_400",  "SvB Regressed P(Signal) #cbar P(ZH) is largest, 250<p_{T,Z}<400", rebin = 2),
           # variable("SvB_ps_zh_400_inf",  "SvB Regressed P(Signal) #cbar P(ZH) is largest, 400<p_{T,Z}<inf", rebin = 5),
           # variable("SvB_ps_zz_0_75",  "SvB Regressed P(Signal) #cbar P(ZZ) is largest, 0<p_{T,Z}<75", rebin = 5),
           # variable("SvB_ps_zz_75_150",  "SvB Regressed P(Signal) #cbar P(ZZ) is largest, 75<p_{T,Z}<150", rebin = 2),
           # variable("SvB_ps_zz_150_250",  "SvB Regressed P(Signal) #cbar P(ZZ) is largest, 150<p_{T,Z}<250", rebin = 2),
           # variable("SvB_ps_zz_250_400",  "SvB Regressed P(Signal) #cbar P(ZZ) is largest, 250<p_{T,Z}<400", rebin = 2),
           # variable("SvB_ps_zz_400_inf",  "SvB Regressed P(Signal) #cbar P(ZZ) is largest, 400<p_{T,Z}<inf", rebin = 5),
           variable("FvT_q_score", "FvT q_score (selected pairing)", rebin = 2),
           variable("FvT_q_score_dR_min", "FvT q_score (min #DeltaR(j,j) pairing)", rebin = 2),
           variable("FvT_q_score_SvB_q_score_max", "FvT q_score (max SvB q_score pairing)", rebin = 2),
           variable("SvB_q_score", "SvB q_score (selected pairing)", rebin = 2),
           variable("SvB_q_score_FvT_q_score_max", "SvB q_score (max FvT q_score pairing)", rebin = 2),
           variable("SvB_MA_q_score", "SvB_MA q_score (selected pairing)", rebin = 2),
           variable("FvT_SvB_q_score_max_same", "FvT max q_score pairing == SvB max q_score pairing"),
           #variable("ZHvB", "ZH vs Background Output", rebin=5),
           #variable("ZZvB", "ZZ vs Background Output", rebin=5),
           variable("xZH", "x_{ZH}"),
           variable("xZZ", "x_{ZZ}"),
           variable("xHH", "x_{HH}"),
           variable("xWt0", "Minimum x_{Wt} (#geq0 not candidate jets)", rebin=1),
           variable("xWt1", "Minimum x_{Wt} (#geq1 not candidate jets)", rebin=1),
           # variable("xWt2", "Minimum x_{Wt} (#geq2 not candidate jets)", rebin=1),
           variable("xWt",  "x_{Wt}", rebin=1),
           variable("t/dRbW", "Top Candidate #DeltaR(b,W)"),
           variable("t/m", "Top Candidate Mass [GeV]"),
           variable("t/pt_m", "Top Candidate p_{T} [GeV]"),
           variable("t/W/m", "W Boson Candidate Mass [GeV]"),
           variable("t/W/dR", "W Boson Candidate #DeltaR(j,j)"),
           variable("t/b/pt_m", "Top Candidate b p_{T} [GeV]"),
           variable("t/mbW", "Top Candidate m_{b,W} [GeV]"),
           variable("t/xWt", "Top Candidate x_{W,t}"),
           variable("t/xWbW", "Top Candidate x_{W,bW}"),
           variable("t/rWbW", "Top Candidate r_{W,bW}"),
           variable("t0/dRbW", "Top Candidate (#geq0 not candidate jets) #DeltaR(b,W)"),
           variable("t0/m", "Top Candidate (#geq0 not candidate jets) Mass [GeV]"),
           variable("t0/W/m", "W Boson Candidate (#geq0 not candidate jets) Mass [GeV]"),
           variable("t0/W/dR", "W Boson Candidate (#geq0 not candidate jets) #DeltaR(j,j)"),
           variable("t0/mbW", "Top Candidate (#geq0 not candidate jets) m_{b,W} [GeV]"),
           variable("t0/xWt", "Top Candidate (#geq0 not candidate jets) x_{W,t}"),
           variable("t1/dRbW", "Top Candidate (#geq1 not candidate jets) #DeltaR(b,W)"),
           variable("t1/m", "Top Candidate (#geq1 not candidate jets) Mass [GeV]"),
           variable("t1/W/m", "W Boson Candidate (#geq1 not candidate jets) Mass [GeV]"),
           variable("t1/W/dR", "W Boson Candidate (#geq1 not candidate jets) #DeltaR(j,j)"),
           variable("t1/mbW", "Top Candidate (#geq1 not candidate jets) m_{b,W} [GeV]"),
           variable("t1/xWt", "Top Candidate (#geq1 not candidate jets) x_{W,t}"),
           # variable("t2/dRbW", "Top Candidate (#geq2 not candidate jets) #DeltaR(b,W)"),
           # variable("t2/m", "Top Candidate (#geq2 not candidate jets) Mass [GeV]"),
           # variable("t2/W/m", "W Boson Candidate (#geq2 not candidate jets) Mass [GeV]"),
           # variable("t2/mbW", "Top Candidate (#geq2 not candidate jets) m_{b,W} [GeV]"),
           # variable("t2/xWt", "Top Candidate (#geq2 not candidate jets) x_{W,t}"),
           variable("mZH", "m_{ZH} [GeV]", divideByBinWidth = True),
           variable("mZZ", "m_{ZZ} [GeV]", divideByBinWidth = True),
           variable("mHH", "m_{HH} [GeV]", divideByBinWidth = True),
           variable("dBB", "D_{BB} [GeV]"),
           variable("dEtaBB", "#Delta#eta(B_{1}, B_{2})"),
           variable("dRBB", "#DeltaR(B_{1}, B_{2})"),
           variable("v4j/m_l", "m_{4j} [GeV]"),
           variable("v4j/pt_l", "p_{T,4j} [GeV]"),
           variable("v4j/pz_l", "|p_{z,4j}| [GeV]", rebin=2),
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
           variable("canJets/m_s", "Boson Candidate Jets Mass [GeV]"),
           variable("canJets/e",   "Boson Candidate Jets Energy [GeV]"),
           variable("canJets/jetId",   "Boson Candidate Jets ID"),
           variable("canJets/deepFlavB", "Boson Candidate Jets Deep Flavour B"),
           variable("canJets/deepB", "Boson Candidate Jets Deep CSV B"),
           variable("canJets/CSVv2", "Boson Candidate Jets CSVv2"),
           variable("canJets/puId", "Boson Candidate Jets Pilup ID"),
           variable("othJets/pt_s", "Other Jets p_{T} [GeV]"),
           variable("othJets/pt_m", "Other Jets p_{T} [GeV]"),
           variable("othJets/pt_l", "Other Jets p_{T} [GeV]"),
           variable("othJets/eta", "Other Jets #eta"),
           variable("othJets/phi", "Other Jets #phi"),
           variable("othJets/m",   "Other Jets Mass [GeV]"),
           variable("othJets/m_s", "Other Jets Mass [GeV]"),
           variable("othJets/e",   "Other Jets Energy [GeV]"),
           variable("othJets/deepFlavB", "Other Jets Deep Flavour B"),
           variable("othJets/deepB", "Other Jets Deep CSV B"),
           variable("othJets/CSVv2", "Other Jets CSVv2"),
           variable("othJets/puId", "Other Jets Pilup ID"),
           variable("allNotCanJets/pt_s", "All jets (p_{T}>20) excluding boson candidate jets p_{T} [GeV]"),
           variable("allNotCanJets/pt_m", "All jets (p_{T}>20) excluding boson candidate jets p_{T} [GeV]"),
           variable("allNotCanJets/pt_l", "All jets (p_{T}>20) excluding boson candidate jets p_{T} [GeV]"),
           variable("allNotCanJets/eta", "All jets (p_{T}>20) excluding boson candidate jets #eta"),
           variable("allNotCanJets/phi", "All jets (p_{T}>20) excluding boson candidate jets #phi"),
           variable("allNotCanJets/m",   "All jets (p_{T}>20) excluding boson candidate jets Mass [GeV]"),
           variable("allNotCanJets/m_s", "All jets (p_{T}>20) excluding boson candidate jets Mass [GeV]"),
           variable("allNotCanJets/e",   "All jets (p_{T}>20) excluding boson candidate jets Energy [GeV]"),
           variable("allNotCanJets/deepFlavB", "All jets (p_{T}>20) excluding boson candidate jets Deep Flavour B"),
           variable("allNotCanJets/deepB", "All jets (p_{T}>20) excluding boson candidate jets Deep CSV B"),
           variable("allNotCanJets/CSVv2", "All jets (p_{T}>20) excluding boson candidate jets CSVv2"),
           variable("allNotCanJets/puId", "All jets (p_{T}>20) excluding boson candidate jets Pileup ID"),
           variable("allNotCanJets/jetId", "All jets (p_{T}>20) excluding boson candidate jets ID"),
           variable("aveAbsEta", "Boson Candidate Jets <|#eta|>"),
           variable("aveAbsEtaOth", "Other Jets <|#eta|>"),
           variable("mjjOther", "m_{V(jj)}  [GeV]",rebin=4),
           variable("ptjjOther", "p_{T}^{V(jj)}  [GeV]",rebin=4),

           variable("canJet0/pt_s", "Boson Candidate Jet_{0} p_{T} [GeV]"),
           variable("canJet0/pt_m", "Boson Candidate Jet_{0} p_{T} [GeV]"),
           variable("canJet0/pt_l", "Boson Candidate Jet_{0} p_{T} [GeV]"),
           variable("canJet0/eta", "Boson Candidate Jet_{0} #eta"),
           variable("canJet0/phi", "Boson Candidate Jet_{0} #phi"),
           variable("canJet0/deepFlavB", "Boson Candidate Jet_{0} Deep Flavour B"),
           variable("canJet0/deepB", "Boson Candidate Jet_{0} Deep CSV B"),
           variable("canJet0/CSVv2", "Boson Candidate Jet_{0} CSVv2"),

           variable("canJet1/pt_s", "Boson Candidate Jet_{1} p_{T} [GeV]"),
           variable("canJet1/pt_m", "Boson Candidate Jet_{1} p_{T} [GeV]"),
           variable("canJet1/pt_l", "Boson Candidate Jet_{1} p_{T} [GeV]"),
           variable("canJet1/eta", "Boson Candidate Jet_{1} #eta"),
           variable("canJet1/phi", "Boson Candidate Jet_{1} #phi"),
           variable("canJet1/deepFlavB", "Boson Candidate Jet_{1} Deep Flavour B"),
           variable("canJet1/deepB", "Boson Candidate Jet_{1} Deep CSV B"),
           variable("canJet1/CSVv2", "Boson Candidate Jet_{1} CSVv2"),

           variable("canJet2/pt_s", "Boson Candidate Jet_{2} p_{T} [GeV]"),
           variable("canJet2/pt_m", "Boson Candidate Jet_{2} p_{T} [GeV]"),
           variable("canJet2/pt_l", "Boson Candidate Jet_{2} p_{T} [GeV]"),
           variable("canJet2/eta", "Boson Candidate Jet_{2} #eta"),
           variable("canJet2/phi", "Boson Candidate Jet_{2} #phi"),
           variable("canJet2/deepFlavB", "Boson Candidate Jet_{2} Deep Flavour B"),
           variable("canJet2/deepB", "Boson Candidate Jet_{2} Deep CSV B"),
           variable("canJet2/CSVv2", "Boson Candidate Jet_{2} CSVv2"),

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
           #variable("leadSt/dPhi",   "Leading S_{T} Dijet #DeltaPhi(j,j)"),
           variable("leadSt/pt_m", "Leading S_{T} Dijet p_{T} [GeV]"),
           variable("leadSt/eta",  "Leading S_{T} Dijet #eta"),
           variable("leadSt/phi",  "Leading S_{T} Dijet #phi"),
           variable("sublSt/m",    "Subleading S_{T} Dijet Mass [GeV]"),
           variable("sublSt/dR",   "Subleading S_{T} Dijet #DeltaR(j,j)"),
           #variable("sublSt/dPhi",   "Subleading S_{T} Dijet #DeltaPhi(j,j)"),
           variable("sublSt/pt_m", "Subleading S_{T} Dijet p_{T} [GeV]"),
           variable("sublSt/eta",  "Subleading S_{T} Dijet #eta"),
           variable("sublSt/phi",  "Subleading S_{T} Dijet #phi"),
           variable("leadM/m",    "Leading Mass Dijet Mass [GeV]"),
           variable("leadM/dR",   "Leading Mass Dijet #DeltaR(j,j)"),
           #variable("leadM/dPhi",   "Leading Mass Dijet #DeltaPhi(j,j)"),
           variable("leadM/pt_m", "Leading Mass Dijet p_{T} [GeV]"),
           variable("leadM/eta",  "Leading Mass Dijet #eta"),
           variable("leadM/phi",  "Leading Mass Dijet #phi"),
           variable("sublM/m",    "Subleading Mass Dijet Mass [GeV]"),
           variable("sublM/dR",   "Subleading Mass Dijet #DeltaR(j,j)"),
           #variable("sublM/dPhi",   "Subleading Mass Dijet #DeltaPhi(j,j)"),
           variable("sublM/pt_m", "Subleading Mass Dijet p_{T} [GeV]"),
           variable("sublM/eta",  "Subleading Mass Dijet #eta"),
           variable("sublM/phi",  "Subleading Mass Dijet #phi"),
           variable("lead/m",    "Leading P_{T} Dijet Mass [GeV]"),
           variable("lead/dR",   "Leading p_{T} Dijet #DeltaR(j,j)"),
           #variable("lead/dPhi",   "Leading p_{T} Dijet #DeltaPhi(j,j)"),
           variable("lead/pt_m", "Leading p_{T} Dijet p_{T} [GeV]"),
           variable("lead/eta",  "Leading p_{T} Dijet #eta"),
           variable("lead/phi",  "Leading p_{T} Dijet #phi"),
           variable("subl/m",    "Subleading p_{T} Dijet Mass [GeV]"),
           variable("subl/dR",   "Subleading p_{T} Dijet #DeltaR(j,j)"),
           #variable("subl/dPhi",   "Subleading p_{T} Dijet #DeltaPhi(j,j)"),
           variable("subl/pt_m", "Subleading p_{T} Dijet p_{T} [GeV]"),
           variable("subl/eta",  "Subleading p_{T} Dijet #eta"),
           variable("subl/phi",  "Subleading p_{T} Dijet #phi"),
           variable("close/m",    "Minimum #DeltaR(j,j) Dijet Mass [GeV]"),
           variable("close/dR",   "Minimum #DeltaR(j,j) Dijet #DeltaR(j,j)"),
           #variable("close/dPhi",   "Minimum #DeltaR(j,j) Dijet #DeltaPhi(j,j)"),
           variable("close/pt_m", "Minimum #DeltaR(j,j) Dijet p_{T} [GeV]"),
           variable("close/eta",  "Minimum #DeltaR(j,j) Dijet #eta"),
           variable("close/phi",  "Minimum #DeltaR(j,j) Dijet #phi"),
           variable("other/m",    "Complement of Minimum #DeltaR(j,j) Dijet Mass [GeV]"),
           variable("other/dR",   "Complement of Minimum #DeltaR(j,j) Dijet #DeltaR(j,j)"),
           #variable("other/dPhi",   "Complement of Minimum #DeltaR(j,j) Dijet #DeltaPhi(j,j)"),
           variable("other/pt_m", "Complement of Minimum #DeltaR(j,j) Dijet p_{T} [GeV]"),
           variable("other/eta",  "Complement of Minimum #DeltaR(j,j) Dijet #eta"),
           variable("other/phi",  "Complement of Minimum #DeltaR(j,j) Dijet #phi"),
           ]

if o.doMain and not onlySignal2D:
    if  'fourTag' in o.histDetailLevel: 
        for cut in cuts:
            for view in views:
                for region in regions:
                    for var in variables:
                        plots.append(standardPlot(o.year, cut, view, region, var))

    if 'threeTag' in o.histDetailLevel: 
        for cut in cuts:
            for view in views:
                for region in regions:
                    for var in variables:
                        plots.append(threeTagPlot(o.year, cut, view, region, var))

    if "ZZ4b"+o.year in files and "bothZH4b"+o.year in files:
        for cut in cuts:
            for view in views:
                for region in regions:
                    for var in variables:
                        plots.append(      mcPlot(o.year, cut, view, region, var))

    if o.doJECSyst and "ZZ4b"+o.year in files and "bothZH4b"+o.year in files:
        for cut in cuts:
            for view in views:
                for region in regions:
                    for var in variables:
                            plots.append(     JECPlot(o.year, cut, view, region, var))

    if  'mixedVsData' in o.histDetailLevel: 
        for cut in cuts:
            for view in views:
                for region in regions:
                    for var in variables:
                        plots.append(mixedVsDataPlot(o.year, cut, view, region, var))




variables2d = [variable("leadSt_m_vs_sublSt_m", "Leading S_{T} Dijet Mass [GeV]", "Subleading S_{T} Dijet Mass [GeV]"),
               variable("leadM_m_vs_sublM_m", "Leading Mass Dijet Mass [GeV]", "Subleading Mass Dijet Mass [GeV]"),
               variable("close_m_vs_other_m", "Minimum #DeltaR(j,j) Dijet Mass [GeV]", "Other Dijet Mass [GeV]"),
               variable('m4j_vs_leadSt_dR', 'm_{4j} [GeV]', 'Leading S_{T} Boson Candidate #DeltaR(j,j)'),
               variable('m4j_vs_sublSt_dR', 'm_{4j} [GeV]', 'Subleading S_{T} Boson Candidate #DeltaR(j,j)'),
               variable("m4j_vs_nViews", "m_{4j} [GeV]", "Number of Pairings"),
               variable("t/mW_vs_mt", "W Boson Candidate Mass [GeV]", "Top Quark Candidate Mass [GeV]"),
               variable("t/mW_vs_mbW", "W Boson Candidate Mass [GeV]", "m_{b,W} [GeV]"),
               variable("t/mW_vs_mbW", "W Boson Candidate Mass [GeV]", "m_{b,W} [GeV]"),
               variable("t/xW_vs_xt", "x_{W}", "x_{t}"),
               variable("t/xW_vs_xbW", "x_{W}", "x_{bW}"),
               variable("t0/mW_vs_mt", "W Boson Candidate Mass [GeV]", "Top Quark Candidate Mass [GeV]"),
               variable("t0/mW_vs_mbW", "W Boson Candidate Mass [GeV]", "m_{b,W} [GeV]"),
               variable("t0/mW_vs_mbW", "W Boson Candidate Mass [GeV]", "m_{b,W} [GeV]"),
               variable("t0/xW_vs_xt", "x_{W}", "x_{t}"),
               variable("t0/xW_vs_xbW", "x_{W}", "x_{bW}"),                               
               variable("t1/mW_vs_mt", "W Boson Candidate Mass [GeV]", "Top Quark Candidate Mass [GeV]"),
               variable("t1/mW_vs_mbW", "W Boson Candidate Mass [GeV]", "m_{b,W} [GeV]"),
               variable("t1/mW_vs_mbW", "W Boson Candidate Mass [GeV]", "m_{b,W} [GeV]"),
               variable("t1/xW_vs_xt", "x_{W}", "x_{t}"),
               variable("t1/xW_vs_xbW", "x_{W}", "x_{bW}"),                               
               ]

if o.doMain and not onlySignal2D:
    for cut in cuts:
        for view in views:
            for region in regions:
                #if True:
                for var in variables2d:
                    sample = nameTitle("data"+o.year, ("Data %.1f/fb, "+o.year)%(lumi))
                    plots.append(TH2Plot("data", sample, o.year, cut, "fourTag", view, region, var))
                    if 'fourTag' in o.histDetailLevel:
                        sample = nameTitle(None, "Background")
                        plots.append(TH2Plot("data", sample, o.year, cut, sample.title, view, region, var, debug=False))
                        
                    if 'ttbar3b' in o.histDetailLevel:
                        sample = nameTitle("TT"+o.year, "t#bar{t} (three-tag)")
                        plots.append(TH2Plot("ttbar", sample, o.year, cut, "threeTag", view, region, var))

                    sample = nameTitle("TT"+o.year, "t#bar{t} (four-tag)")
                    plots.append(TH2Plot("ttbar", sample, o.year, cut, "fourTag", view, region, var))



# cuts = [nameTitle("passDijetMass", "Pass m(j,j)")] + cuts
# views = ["allViews"] + views
# regions = [nameTitle("inclusive", "")] + regions
if o.doMain:
    for cut in cuts:
        for view in views:
            for region in regions:
                #if True:
                for var in variables2d:
                    for process in ['HH4b','bothZH4b','ZZ4b','ZZZHHH4b']:
                        if process+o.year in files:
                            decay = []
                            if 'ZZ' in process: decay.append('ZZ')
                            if 'ZH' in process: decay.append('ZH')
                            if 'HH' in process: decay.append('HH')
                            decay = ','.join(decay) + '#rightarrowb#bar{b}b#bar{b}'
                            sample = nameTitle(process+o.year, decay)
                            TH2 = TH2Plot(process, sample, o.year, cut, "fourTag", view, region, var)
                            plots.append(TH2)




class accxEffPlot:
    def __init__(self, topDir, fileName, year, region, denominator = nameTitle('all', ''), tag='_fourTag', weight=''):
        self.samplesAbs=collections.OrderedDict()
        self.samplesAbs[files[fileName.name]] = collections.OrderedDict()
        self.samplesAbs[files[fileName.name]]["jetMultiplicity_over_"+denominator.name+tag+weight] = {
            "label"      : "#geq4 selected jets",
            "legend"     : 1,
            "color"      : "ROOT.kViolet",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}
        self.samplesAbs[files[fileName.name]]["bTags_over_"+denominator.name+tag+weight] = {
            "label"      : "#geq4 b-tagged jets" if tag=='_fourTag' else "3 loose b-tags",
            "legend"     : 2,
            "color"      : "ROOT.kBlue",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}
        self.samplesAbs[files[fileName.name]]["DijetMass_over_"+denominator.name+tag+weight] = {
            "label"      : "m(j,j)",
            "legend"     : 3,
            "color"      : "ROOT.kGreen+3",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}
        # self.samplesAbs[files[fileName.name]]["MDRs_over_"+denominator.name+tag+weight] = {
        #     "label"      : "#DeltaR(j,j)",
        #     "legend"     : 4,
        #     "color"      : "ROOT.kGreen+3",
        #     "drawOptions" : "HIST PC",
        #     'errorBands' : False,
        #     "marker"      : "20"}
        self.samplesAbs[files[fileName.name]]["DijetMass_SR_over_"+denominator.name+tag+weight] = {
            "label"      : "SR",
            "legend"     : 6,
            "color"      : "ROOT.kRed",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}
        self.samplesAbs[files[fileName.name]]["DijetMass_SR_HLT_over_"+denominator.name+tag+weight] = {
            "label"      : "Trigger",
            "legend"     : 7,
            "color"      : "ROOT.kBlack",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}

        xmin = 160
        if topDir=='HH4b': xmin = 250

        self.parametersAbs = {"titleLeft"       : "#scale[0.7]{#bf{CMS} Simulation Internal}",
                              "titleCenter"      : "#scale[0.7]{"+denominator.title+"}",
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
                              "xMin" :  xmin,
                              "xMax" : 2000,
                              "yMin"       : 0.00002,
                              "yMax"       : 20,
                              "xleg"       : [0.15,0.71],
                              "yleg"       : [0.77,0.92],
                              "labelSize"  : 16,
                              "logY"       : True,
                              "outputDir"   : outputPlot+topDir+"/"+year+"/",
                              "outputName" : "absoluteAccxEff"+tag+weight,
                              }

        self.samplesRel=collections.OrderedDict()
        self.samplesRel[files[fileName.name]] = collections.OrderedDict()
        self.samplesRel[files[fileName.name]]["jetMultiplicity_over_all"+tag+weight] = {
            "label"      : "#geq4 jets",
            "legend"     : 1,
            "color"      : "ROOT.kViolet",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}
        self.samplesRel[files[fileName.name]]["bTags_over_jetMultiplicity"+tag+weight] = {
            "label"      : "#geq4 b-tags / #geq4 jets" if tag=='_fourTag' else "3 loose b-tags / #geq4 jets",
            "legend"     : 2,
            "color"      : "ROOT.kBlue",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}
        self.samplesRel[files[fileName.name]]["DijetMass_over_bTags"+tag+weight] = {
            "label"      : "m(j,j) / #geq4 b-tags" if tag=='_fourTag' else "m(j,j) / 3 loose b-tags",
            "legend"     : 3,
            "color"      : "ROOT.kGreen+3",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}
        # self.samplesRel[files[fileName.name]]["MDRs_over_DijetMass"+tag+weight] = {
        #     "label"      : "#DeltaR(j,j) / m(j,j)",
        #     "legend"     : 4,
        #     "color"      : "ROOT.kGreen+3",
        #     "drawOptions" : "HIST PC",
        #     'errorBands' : False,
        #     "marker"      : "20"}
        # self.samplesRel[files[fileName.name]]["MDRs_over_bTags"+tag+weight] = {
        #     "label"      : "#DeltaR(j,j) / #geq4 b-Tags",
        #     "legend"     : 3,
        #     "color"      : "ROOT.kGreen+3",
        #     "drawOptions" : "HIST PC",
        #     'errorBands' : False,
        #     "marker"      : "20"}
        self.samplesRel[files[fileName.name]]["DijetMass_SR_over_DijetMass"+tag+weight] = {
            "label"      : "SR / #m(j,j)",
            "legend"     : 5,
            "color"      : "ROOT.kRed",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}
        self.samplesRel[files[fileName.name]]["DijetMass_SR_HLT_over_DijetMass_SR"+tag+weight] = {
            "label"      : "Trigger / SR",
            "legend"     : 6,
            "color"      : "ROOT.kBlack",
            "drawOptions" : "HIST PC",
            'errorBands' : False,
            "marker"      : "20"}


        self.parametersRel = {"titleLeft"       : "#scale[0.7]{#bf{CMS} Simulation Internal}",
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
                              "xMin" :  xmin,
                              "xMax" : 2000,
                              "yMin"       : 0.0,
                              "yMax"       : 1.3,
                              "xleg"       : [0.15,0.71],
                              "yleg"       : [0.77,0.92],
                              "labelSize"  : 16,
                              "logY"       : False,
                              "drawLines"  : [[xmin,1,2000,1]],
                              "outputDir"   : outputPlot+topDir+"/"+year+"/",
                              "outputName" : "relativeAccxEff"+tag+weight,
                              }
    
    def plot(self, debug = False):
        PlotTools.plot(self.samplesAbs, self.parametersAbs, o.debug or debug)
        PlotTools.plot(self.samplesRel, self.parametersRel, o.debug or debug)





if o.doAccxEff:
    region = nameTitle("SR", "SR")
    fileName = nameTitle("ZZ4b"+o.year, "ZZ#rightarrowb#bar{b}b#bar{b}")
    plots.append(accxEffPlot("ZZ4b", fileName, o.year, region))
    plots.append(accxEffPlot("ZZ4b", fileName, o.year, region, tag='_threeTag'))
    plots.append(accxEffPlot("ZZ4b", fileName, o.year, region, weight='_unitWeight'))
    plots.append(accxEffPlot("ZZ4b", fileName, o.year, region, tag='_threeTag', weight='_unitWeight'))

    fileName = nameTitle("bothZH4b"+o.year, "ZH#rightarrowb#bar{b}b#bar{b}")
    plots.append(accxEffPlot("bothZH4b", fileName, o.year, region))
    plots.append(accxEffPlot("bothZH4b", fileName, o.year, region, tag='_threeTag'))
    plots.append(accxEffPlot("bothZH4b", fileName, o.year, region, weight='_unitWeight'))
    plots.append(accxEffPlot("bothZH4b", fileName, o.year, region, tag='_threeTag', weight='_unitWeight'))

    fileName = nameTitle("HH4b"+o.year, "HH#rightarrowb#bar{b}b#bar{b}")
    plots.append(accxEffPlot("HH4b", fileName, o.year, region))
    plots.append(accxEffPlot("HH4b", fileName, o.year, region, tag='_threeTag'))
    plots.append(accxEffPlot("HH4b", fileName, o.year, region, weight='_unitWeight'))
    plots.append(accxEffPlot("HH4b", fileName, o.year, region, tag='_threeTag', weight='_unitWeight'))

    fileName = nameTitle("ZZZHHH4b"+o.year, "ZZ, ZH, HH#rightarrowb#bar{b}b#bar{b}")
    plots.append(accxEffPlot("ZZZHHH4b", fileName, o.year, region))
    plots.append(accxEffPlot("ZZZHHH4b", fileName, o.year, region, tag='_threeTag'))
    plots.append(accxEffPlot("ZZZHHH4b", fileName, o.year, region, weight='_unitWeight'))
    plots.append(accxEffPlot("ZZZHHH4b", fileName, o.year, region, tag='_threeTag', weight='_unitWeight'))


nPlots=len(plots)
start = time.time()
rate = -1
for p, thisPlot in enumerate(plots):

    # try:
    thisPlot.plot()

    # except:
    #     print "ERROR"
    #     print thisPlot
    #     pass

    if (p%10)==9:
        elapsedTime = time.time()-start
        start = time.time()
        rate = 0.1 * 10/elapsedTime + 0.9 * rate if rate>0 else 0.1 * 10/elapsedTime
        hours = int((nPlots-p-1)/rate/60/60)
        minutes = int((nPlots-p-1)/rate/60)-60*hours
        seconds = int((nPlots-p-1)/rate)%60
        sys.stdout.write("\rMade %4d of %4d | %3.0f%% | %4.1f plots/sec | %02d:%02d:%02d remaining | %s"%(p+1, nPlots, 100.0*(p+1)/nPlots, rate, hours, minutes, seconds, thisPlot.__class__.__name__))
        sys.stdout.flush()
    if (p%1000)==999:
        gc.collect()
print
