import FWCore.ParameterSet.Config as cms

process = cms.PSet()

#fileNames = ["root://cmsxrootd-site.fnal.gov//store/mc/RunIISummer16NanoAOD/ZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8/NANOAODSIM/PUMoriond17_05Feb2018_94X_mcRun2_asymptotic_v2-v1/90000/8E9245DE-0616-E811-A45D-141877638F39.root",
#             "root://cmsxrootd-site.fnal.gov//store/mc/RunIISummer16NanoAOD/ZH_HToBB_ZToQQ_M125_13TeV_powheg_pythia8/NANOAODSIM/PUMoriond17_05Feb2018_94X_mcRun2_asymptotic_v2-v1/90000/46052100-4316-E811-898E-44A84224053C.root"]
fileNames = ["~/nobackup/ZZ4b/ZH_HToBB_ZToLL_M125_13TeV_powheg_herwigpp/NANOAODSIM/PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/00000/380F111E-F342-E811-B579-0025905D1E02.root"]
fileNames = cms.vstring(fileNames)

#Setup framework lite input file object
process.fwliteInput = cms.PSet(
    fileNames   = fileNames,
    maxEvents   = cms.int32(-1),                             ## optional, -1 for no max
)

#Setup framwork lite output file object
process.fwliteOutput = cms.PSet(
    fileName  = cms.string('test.root'),  ## mandatory
)

#Setup event loop object
process.procNtupleTest = cms.PSet(
    ## input specific for this analyzer
    debug = cms.bool(False),
    lumi  = cms.double(150e3),
)

