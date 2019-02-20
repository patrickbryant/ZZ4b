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
o, a = parser.parse_args()

#
# Basic Configuration
#
debug      = o.debug
outputBase = o.outputBase
inputFile, isMC  = "ZZ4b/fileLists/data2016H.txt", False
#inputFile, isMC  = "ZZ4b/fileLists/ZH_bbqq.txt", True
#inputFile, isMC  = "ZZ4b/fileLists/ZZ_bbbb.txt", True
bTag, bTagger    = float(o.bTag), o.bTagger
isData     = not isMC
blind      = True and isData
year       = o.year
JSONfile   = 'ZZ4b/lumiMasks/Cert_271036-284044_13TeV_PromptReco_Collisions16_JSON.txt'
lumi       = float(o.lumi)

fileNames = []
if ".txt" in o.input:
    for line in open(inputFile, 'r').readlines():
        line = line.replace('\n','').strip()
        if line    == '' : continue
        if line[0] == '#': continue
        fileNames.append(line.replace('\n',''))
else:
    fileNames.append(o.input)
fileNames = cms.vstring(fileNames)


process = cms.PSet()

#Setup framework lite input file object
process.fwliteInput = cms.PSet(
    fileNames   = fileNames,
    maxEvents   = cms.int32(-1),                             ## optional, -1 for no max
    )

#
# LumiMask
#
process.inputs = cms.PSet(
    lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
    )
if not isMC:
    # get JSON file correctly parced
    myList = LumiList.LumiList(filename = JSONfile).getCMSSWString().split(',')
    process.inputs.lumisToProcess.extend(myList)


#
# Setup picoAOD and output location. 
#    
pathOut = fileNames[0].replace("root://cmsxrootd-site.fnal.gov//store/", outputBase)
pathOut = '/'.join(pathOut.split("/")[:-1])+"/"
if not os.path.exists(pathOut): 
    mkpath(pathOut)
picoAOD = pathOut+"picoAOD.root"
exists  = os.path.isfile(picoAOD) # picoAOD already exists

use    = exists and not o.createPicoAOD  # if picoAOD already existed use it unlesss otherwise specified in the command line
create = not use # if not using the picoAOD, let's create it
process.picoAOD = cms.PSet(
    fileName = cms.string(picoAOD),
    create   = cms.bool(create),
    use      = cms.bool(use),
    )

#
# Setup framwork lite output file object
#
process.fwliteOutput = cms.PSet(
    fileName  = cms.string(pathOut+"hists.root"),  ## mandatory
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
    )

