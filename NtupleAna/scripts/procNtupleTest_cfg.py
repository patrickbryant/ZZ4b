import FWCore.ParameterSet.Config as cms

process = cms.PSet()

fileNames = ["~/nobackup/ZZ4b/ZH_HToBB_ZToLL_M125_13TeV_powheg_herwigpp/NANOAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/00000/380F111E-F342-E811-B579-0025905D1E02.root"]
fileNames = cms.vstring(fileNames)

#Setup framework lite input file object
process.fwliteInput = cms.PSet(
    fileNames   = fileNames,
    maxEvents   = cms.int32(100),                             ## optional, -1 for no max
)

#Setup framwork lite output file object
process.fwliteOutput = cms.PSet(
    fileName  = cms.string('test.root'),  ## mandatory
)

#Setup event loop object
process.procNtupleTest = cms.PSet(
    ## input specific for this analyzer
    debug = cms.bool(True),
)

