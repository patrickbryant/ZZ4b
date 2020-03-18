#sigNumber=193240
#sigNumber=483100
sigNumber=$1
#mcName=ZH4b_nEvts_966200
#mcName=ZH4b_nEvts_483100 # REDO
#mcName=ZH4b_nEvts_1932400 # REDO
#mcName=ZH4b_nEvts_4831000 # REDO
mcName=ZH4b_nEvts_${sigNumber} # REDO

inputFileName=data18_wMCBranches_${mcName}
inputFileNameEm=data18_${mcName}_4bEmulated
inputFileNameEm4b=data18_4bEmulatedwMCBranches_${mcName}
inputFileSignal3b=${mcName}_4bEmulated

#
# Make Hemis (3b and 4b)
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileName}.txt -p picoAOD_makeHemis.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 1 --histFile hists.root  --nevents -1  --isDataMCMix --createHemisphereLibrary |tee HemiSignalInjectionStudies_4bA/logCreateHemis_${inputFileName}


#
#  Make 3bA Hemis
# 

# first make 3bA input dataset for signal
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${mcName}.txt -p picoAOD_4bEmulated_PS.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile hists.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTTPreSel/jetCombinatoricModel_SB_00-00-01.txt  --emulate4bFrom3b --noDiJetMassCutInPicoAOD --isDataMCMix  |tee HemiSignalInjectionStudies_4bA/log_${mcName}_4bFrom3b


# then make hemis
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm}.txt -p picoAOD_makeHemis.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 1 --histFile hists.root  --nevents -1  --isDataMCMix --createHemisphereLibrary |tee HemiSignalInjectionStudies_4bA/logCreateHemis_${inputFileNameEm}



# 
#  Mix events
#

# 3b -> 4b
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm}.txt  -p picoAOD_3bMixed4b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed_3bMix4b.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA//data18_wMCBranches_ZH4b_nEvts_${sigNumber}/hemiSphereLib_4TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_data_${mcName}_3bMix4b

# 3b -> 3b
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm}.txt  -p picoAOD_3bMixed3b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed_3bMix3b.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 100000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA//data18_wMCBranches_ZH4b_nEvts_${sigNumber}/hemiSphereLib_3TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_data_${mcName}_3bMix3b

# 3b -> 3bA
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm}.txt  -p picoAOD_3bMixed3bA.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed_3bMix3bA.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA/data18_ZH4b_nEvts_${sigNumber}_4bEmulated/hemiSphereLib_4TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_data_${mcName}_3bMix3bA

# 4b -> 4b  (Use 3bA for 4b, so we dont unblind the SR)
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm4b}.txt  -p picoAOD_3bAMixed4b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed_3bAMix4b.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA//data18_wMCBranches_ZH4b_nEvts_${sigNumber}/hemiSphereLib_4TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_data_${mcName}_3bAMix4b

# 4b -> 3b  (Use 3bA for 4b, so we dont unblind the SR)
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm4b}.txt  -p picoAOD_3bAMixed3b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed_3bAMix3b.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 100000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA//data18_wMCBranches_ZH4b_nEvts_${sigNumber}/hemiSphereLib_3TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_data_${mcName}_3bAMix3b

# 4b -> 3bA (Use 3bA for 4b, so we dont unblind the SR)
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm4b}.txt  -p picoAOD_3bAMixed3bA.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed_3bAMix3bA.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA/data18_ZH4b_nEvts_${sigNumber}_4bEmulated/hemiSphereLib_4TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_data_${mcName}_3bAMix3bA


#
# Make plots
#

# Non-mixed
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm}.txt -p picoAOD_nonMixed.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile hists_nonMixed.root  --nevents -1  --isDataMCMix  |tee HemiSignalInjectionStudies_4bA/logPlotNonMixed_${inputFileNameEm}

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileNameEm4b}.txt -p picoAOD_nonMixed.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile hists_nonMixed.root  --nevents -1  --isDataMCMix  |tee HemiSignalInjectionStudies_4bA/logPlotNonMixed_${inputFileNameEm4b}

# Signal 
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${inputFileSignal3b}.txt -p picoAOD_nonMixed.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile hists_nonMixed.root  --nevents -1  --isDataMCMix  |tee HemiSignalInjectionStudies_4bA/logPlotNonMixed_${inputFileSignal3b}

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/${mcName}.txt -p picoAOD_nonMixed.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile hists_nonMixed.root  --nevents -1  --isDataMCMix  |tee HemiSignalInjectionStudies_4bA/logPlotNonMixed_${mcName}