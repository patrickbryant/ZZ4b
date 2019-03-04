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
parser.add_option(      '--bTagger',              dest="bTagger",       default="CSVv2", help="bTagging algorithm")
parser.add_option('-b', '--bTag',                 dest="bTag",          default="0.8484", help="bTag cut value: default is medium WP for CSVv2 https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80XReReco")
parser.add_option('-i', '--input',                dest="input",         default="ZZ4b/fileLists/data2016H.txt", help="Input file(s). If it ends in .txt, will treat it as a list of input files.")
parser.add_option('-o', '--outputBase',           dest="outputBase",    default="/uscms/home/bryantp/nobackup/ZZ4b/", help="Base path for storing output histograms and picoAOD")
parser.add_option('-p', '--createPicoAOD',        dest="createPicoAOD", action="store_true", default=False, help="Create picoAOD from original NANOAOD even if picoAOD already exists")
parser.add_option('-n', '--nevents',              dest="nevents",       default="-1", help="Number of events to process. Default -1 for no limit.")
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
# Calculated lumi per lumiBlock from brilcalc. See README
lumiData   = {'2015':'',
              '2016':'ZZ4b/lumiMasks/', 
              '2017':'',
              '2018':'ZZ4b/lumiMasks/brilcalc_2018_HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5.csv'} 
lumi       = float(o.lumi)

fileNames = []
if ".txt" in o.input:
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
if not os.path.exists(pathOut): 
    mkpath(pathOut)

histOut = pathOut+"hists.root"
picoAOD = pathOut+"picoAOD.root"
exists  = os.path.isfile(picoAOD) # picoAOD already exists
use     = exists and not o.createPicoAOD  # if picoAOD already existed use it unlesss otherwise specified in the command line
create  = not use # if not using the picoAOD, let's create it

#
# Create ParameterSets for use in bin/<script>.cc 
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
    bTag    = cms.double(bTag),
    bTagger = cms.string(bTagger),
    lumiData= cms.string(lumiData[year]),
    )

