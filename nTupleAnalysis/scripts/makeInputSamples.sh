# 
#  Make Hists with all 2018 data
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents/ -y 2018  --histogramming 10 --histFile hists.root  --nevents -1  |tee HemiSignalInjectionStudies_4bA/log_dataOnlyAll

# 
#  Make the emulated 4b sample (3b subsampled) we call this 3bA
#

#
#  Make the JCM-weights at PS-level
#
python ZZ4b/nTupleAnalysis/scripts/makeWeights.py -d HemiSignalInjectionStudies_4bA/dataAllEvents/data18/hists.root -c passPreSel   -o ZZ4b/nTupleAnalysis/weights/data2018noTTPreSel/  -r SB -w 00-00-01

#nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_4bEmulated.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_4bFrom3b -y 2018  --histogramming 10 --histFile hists.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018/jetCombinatoricModel_SB_00-00-02.txt --emulate4bFrom3b --noDiJetMassCutInPicoAOD |tee HemiSignalInjectionStudies_4bA/log_dataOnlyAll_4bFrom3b
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_4bEmulated_PS.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_4bFrom3b -y 2018  --histogramming 10 --histFile hists.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTTPreSel/jetCombinatoricModel_SB_00-00-01.txt  --emulate4bFrom3b --noDiJetMassCutInPicoAOD |tee HemiSignalInjectionStudies_4bA/log_dataOnlyAll_4bFrom3b


#
#  Add MC branches
#
#py ZZ4b/nTupleAnalysis/scripts/addMCBranchesToDataPicoAOD.py -i HemiSignalInjectionStudies_4bA/dataAllEvents_4bFrom3b/data18/picoAOD_4bEmulated.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_4bFrom3b/data18/picoAOD_4bEmulated_wMCBranches.root 
py ZZ4b/nTupleAnalysis/scripts/addMCBranchesToDataPicoAOD.py -i HemiSignalInjectionStudies_4bA/dataAllEvents_4bFrom3b/data18/picoAOD_4bEmulated_PS.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_4bFrom3b/data18/picoAOD_4bEmulated_PS_wMCBranches.root 

#
#  Make hists of the subdamples data
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches.txt -p picoAOD_NonMixed.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA// -y 2018  --histogramming 10 --histFile hists.root  --nevents -1  --isDataMCMix  |tee HemiSignalInjectionStudies_4bA/logMake_subSampledHists

#
#  Make Hemis
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18wMCBranches.txt -p picoAOD.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataHemis/ -y 2018  --histogramming 1 --histFile hists.root  --nevents -1 --createHemisphereLibrary  |tee HemiSignalInjectionStudies_4bA/log_makeHemisData

#
#  Make Hemis of subdampled data
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches.txt -p picoAOD_NonMixed.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA// -y 2018  --histogramming 10 --histFile hists.root  --nevents -1  --isDataMCMix --createHemisphereLibrary  |tee HemiSignalInjectionStudies_4bA/logMake_subSampledHemis



#
#  Mix "3bA" with 4b hemis to make "4bA" evnets
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches.txt  -p picoAOD_Mixed.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA/dataHemis/data18wMCBranches/hemiSphereLib_4TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_dataOnlyAll


#
#  Mix "3bA" with 3b hemis to make "4bA_p" evnets
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches.txt  -p picoAOD_Mixed_3TagHemis.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed_3TagHemis.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 10000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA/dataHemis/data18wMCBranches/hemiSphereLib_3TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_dataOnlyAll_3Tag

#
#  Mix "3bA" with 3bA hemis to make "3bA_Mixed" evnets
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches.txt  -p picoAOD_Mixed_3TagEmulatedHemis.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/ -y 2018  --histogramming 10 --histFile histsMixed_3TagEmulatedHemis.root  --nevents -1 --isDataMCMix --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches/hemiSphereLib_4TagEvents_*root"| tee HemiSignalInjectionStudies_4bA/logMixHemis_dataOnlyAll_3TagEmulated

#
# Mix 4b  with 4b hemis
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18wMCBranches.txt -p picoAOD_Mixed_4bInto4b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/data4bMixed -y 2018  --histogramming 10 --histFile hists_Mixed_4bInto4b.root  --nevents -1  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA/dataHemis/data18wMCBranches/hemiSphereLib_4TagEvents_*root"  |tee HemiSignalInjectionStudies_4bA/log_dataOnlyAll_Mix_4bInto4b

#
# Mix 4b  with 3b hemis
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18wMCBranches.txt -p picoAOD_Mixed_4bInto3b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/data4bMixed -y 2018  --histogramming 10 --histFile hists_Mixed_4bInto3b.root  --nevents -1  --loadHemisphereLibrary --maxNHemis 10000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA/dataHemis/data18wMCBranches/hemiSphereLib_3TagEvents_*root"  |tee HemiSignalInjectionStudies_4bA/log_dataOnlyAll_Mix_4bInto3b


#
# Mix 4b  with 3bA hemis
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18wMCBranches.txt -p picoAOD_Mixed_4bInto3bA.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/data4bMixed -y 2018  --histogramming 10 --histFile hists_Mixed_4bInto3bA.root  --nevents -1  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "$PWD/HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches/hemiSphereLib_4TagEvents_*root"  |tee HemiSignalInjectionStudies_4bA/log_dataOnlyAll_Mix_4bInto3bA

#
#  Make weights for classifier comparison
#
python ZZ4b/nTupleAnalysis/scripts/makeWeights.py -d HemiSignalInjectionStudies_4bA/dataAllEvents/data18/hists.root    -c passXWt   -o ZZ4b/nTupleAnalysis/weights/data2018noTT/  -r SB -w 00-00-01

python ZZ4b/nTupleAnalysis/scripts/makeWeights.py -d HemiSignalInjectionStudies_4bA/dataAllEvents/data18/hists.root --data4b HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches/histsMixed_PS.root   -c passXWt   -o ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo3bMix4b/  -r SB -w 00-00-01

python ZZ4b/nTupleAnalysis/scripts/makeWeights.py -d HemiSignalInjectionStudies_4bA/dataAllEvents/data18/hists.root --data4b HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches/histsMixed_PS_3TagHemis.root   -c passXWt   -o ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo3bMix3b/  -r SB -w 00-00-01

python ZZ4b/nTupleAnalysis/scripts/makeWeights.py -d HemiSignalInjectionStudies_4bA/dataAllEvents/data18/hists.root --data4b HemiSignalInjectionStudies_4bA/data18_4bEmulatedwMCBranches/histsMixed_3TagEmulatedHemis_PS.root   -c passXWt   -o ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo3bMix3bEm/  -r SB -w 00-00-01

python ZZ4b/nTupleAnalysis/scripts/makeWeights.py -d HemiSignalInjectionStudies_4bA/dataAllEvents/data18/hists.root --data4b HemiSignalInjectionStudies_4bA/data4bMixed/data18wMCBranches/hists_Mixed_4bInto4b.root   -c passXWt   -o ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo4bMix4b/  -r SB -w 00-00-01

python ZZ4b/nTupleAnalysis/scripts/makeWeights.py -d HemiSignalInjectionStudies_4bA/dataAllEvents/data18/hists.root --data4b HemiSignalInjectionStudies_4bA/data4bMixed/data18wMCBranches/hists_Mixed_4bInto3b.root   -c passXWt   -o ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo4bMix3b/  -r SB -w 00-00-01

python ZZ4b/nTupleAnalysis/scripts/makeWeights.py -d HemiSignalInjectionStudies_4bA/dataAllEvents/data18/hists.root --data4b HemiSignalInjectionStudies_4bA/data4bMixed/data18wMCBranches/hists_PS_Mixed_4bInto3bA.root   -c passXWt   -o ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo4bMix3bEm/  -r SB -w 00-00-01

#
# Make picoAODS of 3b data with weights applied
#
nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_JCM_3bTo4b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_3b_with_JCM -y 2018  --histogramming 10 --histFile hists_JCM_3bTo4b.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTT/jetCombinatoricModel_SB_00-00-01.txt  |tee HemiSignalInjectionStudies_4bA/log_3bTo4b_JCM

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_JCM_3bTo3bMix4b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_3b_with_JCM -y 2018  --histogramming 10 --histFile hists_JCM_3bTo3bMix4b.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo3bMix4b/jetCombinatoricModel_SB_00-00-01.txt  |tee HemiSignalInjectionStudies_4bA/log_3bTo3bMix4b_JCM

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_JCM_3bTo3bMix3b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_3b_with_JCM -y 2018  --histogramming 10 --histFile hists_JCM_3bTo3bMix3b.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo3bMix3b/jetCombinatoricModel_SB_00-00-01.txt  |tee HemiSignalInjectionStudies_4bA/log_3bTo3bMix3b_JCM

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_JCM_3bTo3bMix3bA.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_3b_with_JCM -y 2018  --histogramming 10 --histFile hists_JCM_3bTo3bMix3bA.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo3bMix3bEm/jetCombinatoricModel_SB_00-00-01.txt  |tee HemiSignalInjectionStudies_4bA/log_3bTo3bMix3bA_JCM

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_JCM_3bTo4bMix4b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_3b_with_JCM -y 2018  --histogramming 10 --histFile hists_JCM_3bTo4bMix4b.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo4bMix4b/jetCombinatoricModel_SB_00-00-01.txt  |tee HemiSignalInjectionStudies_4bA/log_3bTo4bMix4b_JCM

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_JCM_3bTo4bMix3b.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_3b_with_JCM -y 2018  --histogramming 10 --histFile hists_JCM_3bTo4bMix3b.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo4bMix3b/jetCombinatoricModel_SB_00-00-01.txt  |tee HemiSignalInjectionStudies_4bA/log_3bTo4bMix3b_JCM

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt -p picoAOD_JCM_3bTo4bMix3bA.root  -o /uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/HemiSignalInjectionStudies_4bA/dataAllEvents_3b_with_JCM -y 2018  --histogramming 10 --histFile hists_JCM_3bTo4bMix3bA.root  --nevents -1  -j ZZ4b/nTupleAnalysis/weights/data2018noTT_3bTo4bMix3bEm/jetCombinatoricModel_SB_00-00-01.txt  |tee HemiSignalInjectionStudies_4bA/log_3bTo4bMix3bA_JCM






