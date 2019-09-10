
import sys
import optparse
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import FWCore.ParameterSet.Types as CfgTypes
sys.path.insert(0, 'ZZ4b/nTupleAnalysis/scripts/')
from cfgHelper import *

parser = optparse.OptionParser()
parser.add_option('-d', '--debug',                dest="debug",         action="store_true", default=False, help="debug")
parser.add_option('-m', '--isMC',                 dest="isMC",          action="store_true", default=False, help="isMC")
parser.add_option('-y', '--year',                 dest="year",          default="2016", help="Year specifies trigger (and lumiMask for data)")
parser.add_option(      '--firstEvent',           default=0, help="First event in the data set to proccess")
parser.add_option('-l', '--lumi', type="float",   dest="lumi",          default=1.0,    help="Luminosity for MC normalization: units [pb]")
#parser.add_option(      '--bTagger',              dest="bTagger",       default="CSVv2", help="bTagging algorithm")
#parser.add_option('-b', '--bTag',                 dest="bTag",          default="0.8484", help="bTag cut value: default is medium WP for CSVv2 https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80XReReco")
parser.add_option(      '--bTagger',              dest="bTagger",       default="deepFlavB", help="bTagging algorithm")
parser.add_option('-b', '--bTag', type="float",   dest="bTag",          default="0.2770", help="bTag cut value: default is medium WP for deepFlavB (DeepJet?) https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X")
parser.add_option('-i', '--input',                dest="input",         default="ZZ4b/fileLists/data2016H.txt", help="Input file(s). If it ends in .txt, will treat it as a list of input files.")
parser.add_option('-o', '--outputBase',           dest="outputBase",    default="/uscms/home/bryantp/nobackup/ZZ4b/", help="Base path for storing output histograms and picoAOD")
parser.add_option('-p', '--createPicoAOD',        dest="createPicoAOD", type="string", help="Create picoAOD with given name")
parser.add_option('-f', '--fastSkim',             dest="fastSkim",      action="store_true", default=False, help="Do minimal computation to maximize event loop rate for picoAOD production")
parser.add_option('-n', '--nevents',              dest="nevents",       default="-1", help="Number of events to process. Default -1 for no limit.")
parser.add_option(      '--histogramming',        dest="histogramming", default="1e6", help="Histogramming level. 0 to make no kinematic histograms. 1: only make histograms for full event selection, larger numbers add hists in reverse cutflow order.")
parser.add_option(   '--createHemisphereLibrary',    action="store_true", default=False, help="create Output Hemisphere library")
parser.add_option(   '--inputHLib3Tag',           help="Base path for storing output histograms and picoAOD")
parser.add_option(   '--inputHLib4Tag',           help="Base path for storing output histograms and picoAOD")
parser.add_option(   '--loadHemisphereLibrary',    action="store_true", default=False, help="load Hemisphere library")
parser.add_option(   '--maxNHemis',    default=10000, help="Max nHemis to load")
parser.add_option(      '--histFile',             dest="histFile",      default="hists.root", help="name of ouptut histogram file")
parser.add_option('-r', '--doReweight',           dest="doReweight",    action="store_true", default=False, help="boolean  to toggle using FvT reweight")
#parser.add_option('-r', '--reweight',             dest="reweight",      default="", help="Reweight file containing TSpline3 of nTagClassifier ratio")
parser.add_option('-j', '--jetCombinatoricModel', dest="jetCombinatoricModel", default="", help="file containing jet combinatoric model parameters")
o, a = parser.parse_args()


#
# Basic Configuration
#
outputBase = o.outputBase + ("/" if o.outputBase[-1] != "/" else "") # make sure it ends with a slash
isData     = not o.isMC
blind      = True and isData
#https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions17/13TeV/
JSONfiles  = {'2015':'',
              '2016':'ZZ4b/lumiMasks/Cert_271036-284044_13TeV_PromptReco_Collisions16_JSON.txt', #Final, unlikely to change
              '2017':'ZZ4b/lumiMasks/Cert_294927-306462_13TeV_PromptReco_Collisions17_JSON.txt', #Final, unlikely to change
              '2018':'ZZ4b/lumiMasks/Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt'} #Not Final, should be updated at some point
# Calculated lumi per lumiBlock from brilcalc. See README
lumiData   = {'2015':'',
              '2016':'ZZ4b/lumiMasks/brilcalc_2016_HLT_QuadJet45_TripleBTagCSV_p087.csv', 
              '2017':'ZZ4b/lumiMasks/brilcalc_2017_HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5.csv',
              '2018':'ZZ4b/lumiMasks/brilcalc_2018_HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5.csv'} 

# for MC we need to normalize the sample to the recommended cross section * BR times the target luminosity
## Higgs BRs https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR BR(h125->bb) = 0.5824 BR(h125->\tau\tau) = 0.06272 BR(Z->bb) = 0.1512, BR(Z->\tau\tau) = 0.03696
## ZH cross sections https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ZH_Process
## ZZ cross section 15.0 +0.7 -0.6 +/-0.2 (MCFM at NLO in QCD with additional contributions from LO gg -> ZZ diagrams) or 16.2 +0.6 -0.4 (calculated at NNLO in QCD via MATRIX) https://arxiv.org/pdf/1607.08834.pdf pg 10
## ZH->bb\tau\tau xs = (0.7612+0.1227)*(0.58*0.036+0.15*0.067) = 27/fb ~ 10x HH cross section
## HH->bb\tau\tau xs = 34*0.58*0.067*2 = 2.6/fb
## Higgs BR(mH=125.0) = 0.5824, BR(mH=125.09) = 0.5809: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR
## Z BR = 0.1512+/-0.0005 from PDG
## store all process cross sections in pb. Can compute xs of sample with GenXsecAnalyzer. Example: 
## cd genproductions/test/calculateXSectionAndFilterEfficiency; ./calculateXSectionAndFilterEfficiency.sh -f ../../../ZZ_dataset.txt -c RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1 -d MINIAODSIM -n -1 
## tt xs NNLO and measurement in dilep and semilep tt+jets, tt+bb: https://cds.cern.ch/record/2684606/files/TOP-18-002-paper-v19.pdf
xsDictionary = {"ggZH4b":  0.1227*0.5824*0.1512, #0.0432 from GenXsecAnalyzer, does not include BR for H, does include BR(Z->hadrons) = 0.69911. 0.0432/0.69911 = 0.0618, almost exactly half the LHCXSWG value... NNLO = 2x NLO??
                  "ZH4b":  0.7612*0.5824*0.1512, #0.5540 from GenXsecAnalyzer, does not include BR for H, does include BR(Z->hadrons) = 0.69911. 0.5540/0.69911 = 0.7924, 4% larger than the LHCXSWG value.
              "bothZH4b": (0.1227+0.7612)*0.5824*0.1512,
                  "ZZ4b": 15.5   *0.1512*0.1512,#0.3688 from GenXsecAnalyzer gives 16.13 dividing by BR^2. mcEventSumw/mcEventCount * FxFx Jet Matching eff. = 542638/951791 * 0.647 = 0.3688696216. Jet matching not included in genWeight!
                "TTJets": 831.76, #749.5 get xs from GenXsecAnalyzer, McM is just wrong... TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8. Apply 4b scale k-factor 5.5/3.6=1.53 https://cds.cern.ch/record/2687373/files/TOP-18-011-paper-v15.pdf
                "TTToHadronic": 377.9607353256, #313.9 from McM. NNLO tt xs = 831.76, W hadronic BR = 0.6741 => NNLO = 831.76*0.6741^2 = 377.9607353256
                "TTToSemiLeptonic": 365.7826460496, #300.9 from McM. NNLO = 831.76*2*(1-0.6741)*0.6747 = 365.7826460496
                "TTTo2L2Nu": 88.3419033256, #72.1 from McM. NNLO = 831.76*(1-0.6741)^2 = 88.3419033256
                } 

## figure out what sample is being run from the name of the input
sample = ""
if "TTJets" in o.input: sample = "TTJets"
elif "TTToHadronic" in o.input: sample = "TTToHadronic"
elif "TTToSemiLeptonic" in o.input: sample = "TTToSemiLeptonic"
elif "TTTo2L2Nu" in o.input: sample = "TTTo2L2Nu"
elif "ggZH" in o.input: sample = "ggZH4b"
elif "bothZH" in o.input: sample = "bothZH4b"
elif "ZH" in o.input: sample =   "ZH4b"
elif "ZZ" in o.input: sample =   "ZZ4b" #make sure this is last, ZZ in path name...
xs = 1
if o.isMC: 
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


fourbkfactor = 1.0
for name in fileNames:
    if "TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8" in name: 
        #fourbkfactor = 5.5/3.6 # from https://cds.cern.ch/record/2687373/files/TOP-18-011-paper-v15.pdf
        fourbkfactor = 4.7/4.1 # 2.9/2.4 dilepton channel, 4.7/4.1 lepton+jets channel https://cds.cern.ch/record/2684606/files/TOP-18-002-paper-v19.pdf 
        print "Four b-jet k-Factor: TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8",fourbkfactor
    if "TTTo" in name and "powheg-pythia8" in name:
        #fourbkfactor = 5.5/3.5 # from https://cds.cern.ch/record/2687373/files/TOP-18-011-paper-v15.pdf
        fourbkfactor = 4.7/3.9 # 2.9/2.3 dilepton channel, 4.7/3.9 lepton+jets channel https://cds.cern.ch/record/2684606/files/TOP-18-002-paper-v19.pdf
        print "Four b-jet k-Factor: TTTo*powheg-pythia8",fourbkfactor


useOtherPicoAOD = True if "picoAOD" in o.input else False

pathOut = outputBase
if "root://cmsxrootd-site.fnal.gov//store/" in pathOut: 
    pathOut = pathOut + fileNames[0].replace("root://cmsxrootd-site.fnal.gov//store/", "") #make it a local path
if useOtherPicoAOD:
    pathOut = o.input
pathOut = '/'.join(pathOut.split("/")[:-1])+"/" #remove <fileName>.root
    
if inputList: #use simplified directory structure based on grouping of filelists
    pathOut = outputBase+o.input.split("/")[-1].replace(".txt","/")

if not os.path.exists(pathOut): 
    mkpath(pathOut)

histOut = pathOut+o.histFile
defaultPicoAOD = "picoAOD.root"
createDefaultPicoAOD = o.createPicoAOD == defaultPicoAOD
defaultPicoAODExists = os.path.isfile(pathOut + defaultPicoAOD)
useDefaultPicoAOD = defaultPicoAODExists and not createDefaultPicoAOD and not useOtherPicoAOD
if useDefaultPicoAOD: fileNames = [pathOut+defaultPicoAOD]
picoAOD = pathOut+(o.createPicoAOD if o.createPicoAOD else "picoAOD.root")
create  = o.createPicoAOD != None or not defaultPicoAODExists # create this picoAOD if the user specified one or if the default picoAOD.root does not exist
#if : histOut = histOut.replace("hists.root", "histsFromNanoAOD.root") #store elsewhere to get the correct cutflow numbers prior to the preselection applied to create the picoAOD

if fileNames[0] == picoAOD and create:
    print "ERROR: Trying to overwrite input picoAOD:",picoAOD
    sys.exit()


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
    myList = LumiList.LumiList(filename = JSONfiles[o.year]).getCMSSWString().split(',')
    process.inputs.lumisToProcess.extend(myList)

# Setup picoAOD
process.picoAOD = cms.PSet(
    fileName = cms.string(picoAOD),
    create   = cms.bool(create),
    fastSkim = cms.bool(o.fastSkim),
    )

inputHFiles_3Tag = []
inputHFiles_4Tag = []
if o.loadHemisphereLibrary:

    fileList_3Tag = os.popen("ls "+o.inputHLib3Tag).readlines()
    for i in fileList_3Tag:
        inputHFiles_3Tag.append(i.rstrip())


    fileList_4Tag = os.popen("ls "+o.inputHLib4Tag).readlines()
    for i in fileList_4Tag:
        inputHFiles_4Tag.append(i.rstrip())


# Setup hemisphere Mixing files
hSphereLib = pathOut+"hemiSphereLib"
process.hSphereLib = cms.PSet(
    fileName = cms.string(hSphereLib),
    create   = cms.bool(o.createHemisphereLibrary),
    load     = cms.bool(o.loadHemisphereLibrary),
    inputHLibs_3tag     = cms.vstring(inputHFiles_3Tag),
    inputHLibs_4tag     = cms.vstring(inputHFiles_4Tag),
    maxNHemis  = cms.int32(int(o.maxNHemis)),
    )


# Setup framwork lite output file object
process.fwliteOutput = cms.PSet(
    fileName  = cms.string(histOut),
    )

#Setup event loop object
process.nTupleAnalysis = cms.PSet(
    debug   = cms.bool(o.debug),
    isMC    = cms.bool(o.isMC),
    blind   = cms.bool(blind),
    year    = cms.string(o.year),
    lumi    = cms.double(o.lumi),
    firstEvent  = cms.int32(int(o.firstEvent)),
    xs      = cms.double(xs),
    fourbkfactor = cms.double(fourbkfactor),
    bTag    = cms.double(o.bTag),
    bTagger = cms.string(o.bTagger),
    lumiData= cms.string(lumiData[o.year]),
    histogramming = cms.int32(int(o.histogramming)),
    jetCombinatoricModel = cms.string(o.jetCombinatoricModel),
    doReweight= cms.bool(o.doReweight),
    #reweight= cms.string(o.reweight),
    )

