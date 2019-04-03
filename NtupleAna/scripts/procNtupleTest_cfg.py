import sys
import optparse
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import FWCore.ParameterSet.Types as CfgTypes
sys.path.insert(0, 'ZZ4b/NtupleAna/scripts/')
from cfgHelper import *

parser = optparse.OptionParser()
parser.add_option('-d', '--debug',                dest="debug",         action="store_true", default=False, help="debug")
parser.add_option('-m', '--isMC',                 dest="isMC",          action="store_true", default=False, help="isMC")
parser.add_option('-y', '--year',                 dest="year",          default="2016", help="Year specifies trigger (and lumiMask for data)")
parser.add_option('-l', '--lumi',                 dest="lumi",          default="1",    help="Luminosity for MC normalization: units [pb]")
#parser.add_option(      '--bTagger',              dest="bTagger",       default="CSVv2", help="bTagging algorithm")
#parser.add_option('-b', '--bTag',                 dest="bTag",          default="0.8484", help="bTag cut value: default is medium WP for CSVv2 https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80XReReco")
parser.add_option(      '--bTagger',              dest="bTagger",       default="deepFlavB", help="bTagging algorithm")
parser.add_option('-b', '--bTag',                 dest="bTag",          default="0.2770", help="bTag cut value: default is medium WP for deepFlavB (DeepJet?) https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X")
parser.add_option('-i', '--input',                dest="input",         default="ZZ4b/fileLists/data2016H.txt", help="Input file(s). If it ends in .txt, will treat it as a list of input files.")
parser.add_option('-o', '--outputBase',           dest="outputBase",    default="/uscms/home/bryantp/nobackup/ZZ4b/", help="Base path for storing output histograms and picoAOD")
parser.add_option('-p', '--createPicoAOD',        dest="createPicoAOD", action="store_true", default=False, help="Create picoAOD from original NANOAOD even if picoAOD already exists")
parser.add_option('-n', '--nevents',              dest="nevents",       default="-1", help="Number of events to process. Default -1 for no limit.")
parser.add_option(      '--histogramming',        dest="histogramming", default="1e6", help="Histogramming level. 0 to make no kinematic histograms. 1: only make histograms for full event selection, larger numbers add hists in reverse cutflow order.")
parser.add_option('-r', '--reweight',             dest="reweight",      default="", help="Reweight file containing TSpline3 of nTagClassifier ratio")
o, a = parser.parse_args()

#
# Basic Configuration
#
debug      = o.debug
outputBase = o.outputBase + ("/" if o.outputBase[-1] != "/" else "") # make sure it ends with a slash
isMC       = o.isMC
isData     = not isMC
bTagger    = o.bTagger
bTag       = float(o.bTag)
blind      = True and isData
year       = o.year
JSONfiles  = {'2015':'',
              '2016':'ZZ4b/lumiMasks/Cert_271036-284044_13TeV_PromptReco_Collisions16_JSON.txt', #Final, unlikely to change
              '2017':'',
              '2018':'ZZ4b/lumiMasks/Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt'} #Not Final, should be updated at some point
reweight   = o.reweight
# Calculated lumi per lumiBlock from brilcalc. See README
lumiData   = {'2015':'',
              '2016':'ZZ4b/lumiMasks/', 
              '2017':'',
              '2018':'ZZ4b/lumiMasks/brilcalc_2018_HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5.csv'} 

# for MC we need to normalize the sample to the recommended cross section * BR times the target luminosity
lumi       = float(o.lumi)
## ZH cross sections https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ZH_Process
## ZZ cross section 15.0 +0.7 -0.6 +/-0.2 (MCFM at NLO in QCD with additional contributions from LO gg -> ZZ diagrams) or 16.2 +0.6 -0.4 (calculated at NNLO in QCD via MATRIX) https://arxiv.org/pdf/1607.08834.pdf pg 10
## Higgs BR(mH=125.0) = 0.5824, BR(mH=125.09) = 0.5809: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR
## Z BR = 0.1512+/-0.0005 from PDG
## store all process cross sections in pb. Can compute xs of sample with GenXsecAnalyzer. Example: 
## cd genproductions/test/calculateXSectionAndFilterEfficiency; ./calculateXSectionAndFilterEfficiency.sh -f ../../../ZZ_dataset.txt -c RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1 -d MINIAODSIM -n -1 
xsDictionary = {"ggZH4b":  0.1227*0.5824*0.1512, #0.0432 from GenXsecAnalyzer, does not include BR for H, does include BR(Z->hadrons) = 0.69911. 0.0432/0.69911 = 0.0618, almost exactly half the LHCXSWG value... NNLO = 2x NLO??
                  "ZH4b":  0.7612*0.5824*0.1512, #0.5540 from GenXsecAnalyzer, does not include BR for H, does include BR(Z->hadrons) = 0.69911. 0.5540/0.69911 = 0.7924, 4% larger than the LHCXSWG value.
                  "ZZ4b": 15.5   *0.1512*0.1512} #0.3688 from GenXsecAnalyzer gives 16.13 dividing by BR^2. mcEventSumw/mcEventCount * FxFx Jet Matching eff. = 542638/951791 * 0.647 = 0.3688696216. Jet matching not included in genWeight!

## figure out what sample is being run from the name of the input
sample = ""
if "ggZH" in o.input.split("/")[-1]: sample = "ggZH4b"
elif "ZH" in o.input.split("/")[-1]: sample =   "ZH4b"
elif "ZZ" in o.input.split("/")[-1]: sample =   "ZZ4b"
xs = 1
if isMC: 
    xs = xsDictionary[sample] if sample in xsDictionary else 1.0
    print "Simulated sample:",sample,"| xs =",xs


fileNames = []
inputList=False
if ".txt" in o.input:
    inputList = True
    for line in open(o.input, 'r').readlines():
        line = line.replace('\n','').strip()
        if line    == '' : continue
        if line[0] == '#': continue
        fileNames.append(line.replace('\n',''))
else:
    fileNames.append(o.input)

pathOut = outputBase
pathOut = pathOut + fileNames[0].replace("root://cmsxrootd-site.fnal.gov//store/", "") #make it a local path
pathOut = '/'.join(pathOut.split("/")[:-1])+"/" #remove <fileName>.root
if inputList: #use simplified directory structure based on grouping of filelists
    pathOut = outputBase+o.input.split("/")[-1].replace(".txt","/")

if not os.path.exists(pathOut): 
    mkpath(pathOut)

histOut = pathOut+"hists.root"
picoAOD = pathOut+"picoAOD.root"
exists  = os.path.isfile(picoAOD) # picoAOD already exists
use     = exists and not o.createPicoAOD  # if picoAOD already existed use it unlesss otherwise specified in the command line
create  = not use # if not using the picoAOD, let's create it
if create: histOut = histOut.replace("hists.root", "histsFromNanoAOD.root") #store elsewhere to get the correct cutflow numbers prior to the preselection applied to create the picoAOD

#
# ParameterSets for use in bin/<script>.cc 
#
process = cms.PSet()

#Setup framework lite input file object
process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring(fileNames),
    maxEvents   = cms.int32(int(o.nevents)),
    )

# LumiMask
process.inputs = cms.PSet(
    lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
    )
if isData:
    # get JSON file correctly parced
    myList = LumiList.LumiList(filename = JSONfiles[year]).getCMSSWString().split(',')
    process.inputs.lumisToProcess.extend(myList)

# Setup picoAOD
process.picoAOD = cms.PSet(
    fileName = cms.string(picoAOD),
    create   = cms.bool(create),
    use      = cms.bool(use),
    )

# Setup framwork lite output file object
process.fwliteOutput = cms.PSet(
    fileName  = cms.string(histOut),
    )

#Setup event loop object
process.procNtupleTest = cms.PSet(
    debug   = cms.bool(debug),
    isMC    = cms.bool(isMC),
    blind   = cms.bool(blind),
    year    = cms.string(year),
    lumi    = cms.double(lumi),
    xs      = cms.double(xs),
    bTag    = cms.double(bTag),
    bTagger = cms.string(bTagger),
    lumiData= cms.string(lumiData[year]),
    histogramming = cms.int32(int(o.histogramming)),
    reweight= cms.string(reweight),
    )

