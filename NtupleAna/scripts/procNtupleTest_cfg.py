import FWCore.ParameterSet.Config as cms
from glob import glob

process = cms.PSet()

fileNames = []
paths = glob("/eos/uscms/store/user/jda102/Data_2018B/MuonEG/Data_2018B/181112_172327/0000/*.root")
for p in paths:
    p = p.replace("/eos/uscms/","root://cmsxrootd.fnal.gov//")
    fileNames.append(p)
fileNames = cms.vstring(fileNames)

process.fwliteInput = cms.PSet(
    fileNames   = fileNames,
    maxEvents   = cms.int32(-1),                             ## optional
    outputEvery = cms.uint32(10),                            ## optional
)

process.fwliteOutput = cms.PSet(
    fileName  = cms.string('Data_2018B.root'),  ## mandatory
)


process.procNtupleExample = cms.PSet(
    ## input specific for this analyzer
    debug = cms.bool(False),
    MakeEventDisplays = cms.bool(False),
)

