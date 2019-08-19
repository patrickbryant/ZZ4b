import sys
import optparse
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
import FWCore.ParameterSet.Types as CfgTypes
sys.path.insert(0, 'ZZ4b/nTupleAnalysis/scripts/')
from cfgHelper import *

parser = optparse.OptionParser()
parser.add_option('-d', '--debug',                dest="debug",         action="store_true", default=False, help="debug")
#parser.add_option(      '--bTagger',              dest="bTagger",       default="CSVv2", help="bTagging algorithm")
#parser.add_option('-b', '--bTag',                 dest="bTag",          default="0.8484", help="bTag cut value: default is medium WP for CSVv2 https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80XReReco")
parser.add_option('-i', '--inputHLib',           help="Base path for storing output histograms and picoAOD")
parser.add_option('-o', '--outputBase',           dest="outputBase",    default="/uscms/home/bryantp/nobackup/ZZ4b/", help="Base path for storing output histograms and picoAOD")
parser.add_option('-n', '--nevents',              dest="nevents",       default="-1", help="Number of events to process. Default -1 for no limit.")
parser.add_option(   '--createHemisphereLibrary',    action="store_true", default=False, help="create Output Hemisphere library")
parser.add_option(   '--loadHemisphereLibrary',    action="store_true", default=False, help="load Hemisphere library")
parser.add_option(   '--loadJetFourVecs',    action="store_true", default=False, help="load Hemisphere library")
parser.add_option(      '--histFile',             dest="histFile",      default="hists.root", help="name of ouptut histogram file")
o, a = parser.parse_args()


#
# Basic Configuration
#
outputBase = o.outputBase + ("/" if o.outputBase[-1] != "/" else "") # make sure it ends with a slash




pathOut = outputBase
if "root://cmsxrootd-site.fnal.gov//store/" in pathOut: 
    pathOut = pathOut + fileNames[0].replace("root://cmsxrootd-site.fnal.gov//store/", "") #make it a local path

pathOut = '/'.join(pathOut.split("/")[:-1])+"/" #remove <fileName>.root
    

if not os.path.exists(pathOut): 
    mkpath(pathOut)

histOut = pathOut+o.histFile


#
# ParameterSets for use in bin/<script>.cc 
#
process = cms.PSet()


inputFileNames = []
inputTest = os.popen("ls "+o.inputHLib).readlines()
for i in inputTest:
    inputFileNames.append(i.rstrip())



# Setup hemisphere Mixing files
hSphereLib = pathOut+"hemiSphereLib"
process.hSphereLib = cms.PSet(
    #fileName  = cms.string(o.inputHLib),
    fileNames = cms.vstring(inputFileNames),
    create   = cms.bool(False),
    load     = cms.bool(True),
    maxHemi  = cms.int32(int(o.nevents)),
    )


# Setup framwork lite output file object
process.fwliteOutput = cms.PSet(
    fileName  = cms.string(histOut),
    )

#Setup event loop object
process.hemisphereAnalysis = cms.PSet(
    debug   = cms.bool(o.debug),
    loadJetFourVecs  = cms.bool(o.loadJetFourVecs),
    )

