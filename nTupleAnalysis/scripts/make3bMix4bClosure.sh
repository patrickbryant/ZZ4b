outputDir=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b
outputDirNom=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal
outputDirMix=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/mixed

# mixed
mixedName=3bMix4b


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'

data2018_3bSubSampled=${outputDirMix}/fileLists/data2018_3bSubSampled.txt
data2017_3bSubSampled=${outputDirMix}/fileLists/data2017_3bSubSampled.txt
data2016_3bSubSampled=${outputDirMix}/fileLists/data2016_3bSubSampled.txt


YEAR2018=' -y 2018 --bTag 0.2770 '
YEAR2017=' -y 2017 --bTag 0.3033 '
YEAR2016=' -y 2016 --bTag 0.3093 '


#
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
#for i in 0 1 2 3 4 
#do
#    #$runCMD -i ${outputDirMix}/data2018/picoAOD_data2018_3bSubSampled_v${i}.root  -p picoAOD_${mixedName}_noTTVeto_v${i}.root  $YEAR2018  --histogramming 10 --histFile hists_${mixedName}_noTTVeto_v${i}.root   --is3bMixed --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2018/hemiSphereLib_4TagEvents_*root" | tee ${outputDir}/logMix_${mixedName}_2018_v${i}  & 
#    $runCMD -i ${outputDirMix}/data2017/picoAOD_data2017_3bSubSampled_v${i}.root  -p picoAOD_${mixedName}_noTTVeto_v${i}.root $YEAR2017  --histogramming 10 --histFile hists_${mixedName}_noTTVeto_v${i}.root   --is3bMixed --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2017/hemiSphereLib_4TagEvents_*root" | tee ${outputDir}/logMix_${mixedName}_2017_v${i}  & 
#    $runCMD -i ${outputDirMix}/data2016/picoAOD_data2016_3bSubSampled_v${i}.root  -p picoAOD_${mixedName}_noTTVeto_v${i}.root $YEAR2016  --histogramming 10 --histFile hists_${mixedName}_noTTVeto_v${i}.root   --is3bMixed --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2016/hemiSphereLib_4TagEvents_*root" | tee ${outputDir}/logMix_${mixedName}_2016_v${i}  & 
#done

#mkdir ${outputDir}/data2018
#mkdir ${outputDir}/data2017
#mkdir ${outputDir}/data2016

#mv ${outputDirMix}/data2018/picoAOD_${mixedName}_noTTVeto_v*  ${outputDir}/data2018
#mv ${outputDirMix}/data2018/hists_${mixedName}_noTTVeto_v*    ${outputDir}/data2018

#mv ${outputDirMix}/data2017/picoAOD_${mixedName}_noTTVeto_v*  ${outputDir}/data2017
#mv ${outputDirMix}/data2017/hists_${mixedName}_noTTVeto_v*    ${outputDir}/data2017
#
#
#mv ${outputDirMix}/data2016/picoAOD_${mixedName}_noTTVeto_v*  ${outputDir}/data2016
#mv ${outputDirMix}/data2016/hists_${mixedName}_noTTVeto_v*    ${outputDir}/data2016


#
#  Mix "4b" ttbar
#
#for i in 0 1 2 3 4 
#do
#    $runCMD -i ${outputDirMix}/TTToSemiLeptonic2018_noMjj/picoAOD_3bSubSampled_v${i}.root -p picoAOD_${mixedName}_v${i}.root $YEAR2018MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2018/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToSemiLeptonic2018_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/TTToHadronic2018_noMjj/picoAOD_3bSubSampled_v${i}.root     -p picoAOD_${mixedName}_v${i}.root $YEAR2018MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2018/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToHadronic2018_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/TTTo2L2Nu2018_noMjj/picoAOD_3bSubSampled_v${i}.root            -p picoAOD_${mixedName}_v${i}.root $YEAR2018MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2018/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTTo2L2Nu2018_${mixedName}_v${i} &
#
#
#    $runCMD -i ${outputDirMix}/TTToSemiLeptonic2017_noMjj/picoAOD_3bSubSampled_v${i}.root -p picoAOD_${mixedName}_v${i}.root $YEAR2017MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2017/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToSemiLeptonic2017_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/TTToHadronic2017_noMjj/picoAOD_3bSubSampled_v${i}.root     -p picoAOD_${mixedName}_v${i}.root $YEAR2017MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2017/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToHadronic2017_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/TTTo2L2Nu2017_noMjj/picoAOD_3bSubSampled_v${i}.root            -p picoAOD_${mixedName}_v${i}.root $YEAR2017MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2017/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTTo2L2Nu2017_${mixedName}_v${i} &
#
#
#    $runCMD -i ${outputDirMix}/TTToSemiLeptonic2016_noMjj/picoAOD_3bSubSampled_v${i}.root -p picoAOD_${mixedName}_v${i}.root $YEAR2016MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2016/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToSemiLeptonic2016_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/TTToHadronic2016_noMjj/picoAOD_3bSubSampled_v${i}.root     -p picoAOD_${mixedName}_v${i}.root $YEAR2016MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2016/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToHadronic2016_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/TTTo2L2Nu2016_noMjj/picoAOD_3bSubSampled_v${i}.root            -p picoAOD_${mixedName}_v${i}.root $YEAR2016MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2016/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTTo2L2Nu2016_${mixedName}_v${i} &
#done


#mkdir ${outputDir}/TTToSemiLeptonic2018
#mkdir ${outputDir}/TTToHadronic2018
#mkdir ${outputDir}/TTTo2L2Nu2018
#
#mkdir ${outputDir}/TTToSemiLeptonic2017
#mkdir ${outputDir}/TTToHadronic2017
#mkdir ${outputDir}/TTTo2L2Nu2017
#
#mkdir ${outputDir}/TTToSemiLeptonic2016
#mkdir ${outputDir}/TTToHadronic2016
#mkdir ${outputDir}/TTTo2L2Nu2016



#mv ${outputDirMix}/TTToSemiLeptonic2018_noMjj/picoAOD_${mixedName}_v*  ${outputDir}/TTToSemiLeptonic2018
#mv ${outputDirMix}/TTToSemiLeptonic2018_noMjj/hists_${mixedName}_v*    ${outputDir}/TTToSemiLeptonic2018
#
#mv ${outputDirMix}/TTToHadronic2018_noMjj/picoAOD_${mixedName}_v*      ${outputDir}/TTToHadronic2018
#mv ${outputDirMix}/TTToHadronic2018_noMjj/hists_${mixedName}_v*        ${outputDir}/TTToHadronic2018

#mv ${outputDirMix}/TTTo2L2Nu2018_noMjj/picoAOD_${mixedName}_v*         ${outputDir}/TTTo2L2Nu2018
#mv ${outputDirMix}/TTTo2L2Nu2018_noMjj/hists_${mixedName}_v*           ${outputDir}/TTTo2L2Nu2018
#
#
#
#mv ${outputDirMix}/TTToSemiLeptonic2017_noMjj/picoAOD_${mixedName}_v*  ${outputDir}/TTToSemiLeptonic2017
#mv ${outputDirMix}/TTToSemiLeptonic2017_noMjj/hists_${mixedName}_v*    ${outputDir}/TTToSemiLeptonic2017
#
#mv ${outputDirMix}/TTToHadronic2017_noMjj/picoAOD_${mixedName}_v*      ${outputDir}/TTToHadronic2017
#mv ${outputDirMix}/TTToHadronic2017_noMjj/hists_${mixedName}_v*        ${outputDir}/TTToHadronic2017

#mv ${outputDirMix}/TTTo2L2Nu2017_noMjj/picoAOD_${mixedName}_v*         ${outputDir}/TTTo2L2Nu2017
#mv ${outputDirMix}/TTTo2L2Nu2017_noMjj/hists_${mixedName}_v*           ${outputDir}/TTTo2L2Nu2017
#
#
#
#mv ${outputDirMix}/TTToSemiLeptonic2016_noMjj/picoAOD_${mixedName}_v*  ${outputDir}/TTToSemiLeptonic2016
#mv ${outputDirMix}/TTToSemiLeptonic2016_noMjj/hists_${mixedName}_v*    ${outputDir}/TTToSemiLeptonic2016
#
#mv ${outputDirMix}/TTToHadronic2016_noMjj/picoAOD_${mixedName}_v*      ${outputDir}/TTToHadronic2016
#mv ${outputDirMix}/TTToHadronic2016_noMjj/hists_${mixedName}_v*        ${outputDir}/TTToHadronic2016
#
#mv ${outputDirMix}/TTTo2L2Nu2016_noMjj/picoAOD_${mixedName}_v*         ${outputDir}/TTTo2L2Nu2016
#mv ${outputDirMix}/TTTo2L2Nu2016_noMjj/hists_${mixedName}_v*           ${outputDir}/TTTo2L2Nu2016

#for i in 0 1 2 3 4 
#do
#    hadd -f ${outputDir}/TT2018/hists_${mixedName}_v${i}.root ${outputDir}/TTToHadronic2018/hists_${mixedName}_v${i}.root ${outputDir}/TTToSemiLeptonic2018/hists_${mixedName}_v${i}.root ${outputDir}/TTTo2L2Nu2018/hists_${mixedName}_v${i}.root &
#    hadd -f ${outputDir}/TT2017/hists_${mixedName}_v${i}.root ${outputDir}/TTToHadronic2017/hists_${mixedName}_v${i}.root ${outputDir}/TTToSemiLeptonic2017/hists_${mixedName}_v${i}.root ${outputDir}/TTTo2L2Nu2017/hists_${mixedName}_v${i}.root &
#    hadd -f ${outputDir}/TT2016/hists_${mixedName}_v${i}.root ${outputDir}/TTToHadronic2016/hists_${mixedName}_v${i}.root ${outputDir}/TTToSemiLeptonic2016/hists_${mixedName}_v${i}.root ${outputDir}/TTTo2L2Nu2016/hists_${mixedName}_v${i}.root &
#done



#
#  Fit JCM
#
#for i in 0 1 2 3 4 
#do
#    $weightCMD -d ${outputDirNom}/data2018/hists.root  --data4b ${outputDir}/data2018/hists_${mixedName}_noTTVeto_v${i}.root  --tt ${outputDirNom}/TT2018/hists.root  --tt4b ${outputDir}/TT2018/hists_${mixedName}_v${i}.root  -c passXWt   -o ${outputDir}/weights/data2018_${mixedName}_v${i}/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_makeWeights_2018_v${i}
#    $weightCMD -d ${outputDirNom}/data2017/hists.root  --data4b ${outputDir}/data2017/hists_${mixedName}_noTTVeto_v${i}.root  --tt ${outputDirNom}/TT2017/hists.root  --tt4b ${outputDir}/TT2017/hists_${mixedName}_v${i}.root  -c passXWt   -o ${outputDir}/weights/data2017_${mixedName}_v${i}/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_makeWeights_2017_v${i}
#    $weightCMD -d ${outputDirNom}/data2016/hists.root  --data4b ${outputDir}/data2016/hists_${mixedName}_noTTVeto_v${i}.root  --tt ${outputDirNom}/TT2016/hists.root  --tt4b ${outputDir}/TT2016/hists_${mixedName}_v${i}.root  -c passXWt   -o ${outputDir}/weights/data2016_${mixedName}_v${i}/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_makeWeights_2016_v${i}
#done




#### OLD


#
## Helpers
#convertToH5JOB=ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py
#SvBModel=ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1638_lr0.008_epochs40_stdscale_epoch39_loss0.1985.pkl
#trainJOB=ZZ4b/nTupleAnalysis/scripts/multiClassifier.py
#convertToROOTJOB=ZZ4b/nTupleAnalysis/scripts/convert_h52root.py
#
#
#######################
#### TTBar Mixing
#######################
#

#
##
## Make picoAODS of 3b data with weights applied
##
##$runCMD $runJOB  -i ${fileAllEvents}        -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix_3bTo${mixedName}.root -y 2018  --histogramming 10 --histFile hists_wJCM_TTmix_3bTo${mixedName}.root --skip4b    -j ${JCMNameTTmixed}  |tee ${outputDir}/log_3bTo${mixedName}_JCM_TTmix  &
##$runCMD $runJOB  -i ${fileTTToHadronic}     -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix.root --skip4b   --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixed}  |tee ${outputDir}/log_TTToHadronic2018_JCM_TTmix &
##$runCMD $runJOB  -i ${fileTTToSemiLeptonic} -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix.root --skip4b   --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixed}  |tee ${outputDir}/log_TTToSemiLeptonic2018_JCM_TTmix &
##$runCMD $runJOB  -i ${fileTTTo2L2Nu}        -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix.root --skip4b   --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixed}  |tee ${outputDir}/log_TTTo2L2Nu2018_JCM_TTmix &
#
##hadd -f ${outputPath}/${outputDir}/TT2018/hists_wJCM_TTmix.root ${outputPath}/${outputDir}/TTToHadronic2018/hists_wJCM_TTmix.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_wJCM_TTmix.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_wJCM_TTmix.root
#
#
#
##
## Convert root to hdf5 
##   (with conversion enviorment)
### 4b
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root"                      2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTTo2L2Nu_mixed &         
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root"                   2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToHadronic_mixed &      
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root"               2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToSemiLep_mixede &       
### 3b
##python $convertToH5JOB -i "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}.root"   2>&1 |tee ${outputDir}/log_Convert_ROOTh5_wJCM_TTmix_3bTo${mixedName} &         
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.root"                   2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTTo2L2Nu_3b_mixed &         
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.root"                2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToHadronic_3b_mixed &      
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.root"            2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToSemiLep_3b_mixed &       
#
#
#
##
## Train
##   (with GPU conversion enviorment)
##mkdir ${outputDir}/dataAllEvents_TTmix
## 4b
##ln -s ${outputPath}/${outputDir}/${name3bUnmixed}dataAllEvents_TTmix
##cp ${outputPath}/${outputDir}/${name3bUnmixed}/picoAOD_${mixedName}.h5 ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.h5
##cp ${outputPath}/${outputDir}/${name3bUnmixed}/picoAOD_${mixedName}.root ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.root
##mv ${outputPath}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}.h5   ${outputPath}/data2018AllEvents_TTmix/
##mv ${outputPath}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}.root ${outputPath}/data2018AllEvents_TTmix/
##python $trainJOB -c FvT -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD*${mixedName}*.h5" -t "${outputPath}/${outputDir}/TTTo*2018*/picoAOD_*ix*.h5" -e 40 --outputName 3bTo${mixedName}TTmix 2>&1 |tee ${outputDir}/log_Train_FvT_3bTo${mixedName}_TTmix
#
#
##
## Add FvT
##
##reweightModel_TTmix=ZZ4b/nTupleAnalysis/pytorchModels/3bTo3bMix4b_TTmixFvT_ResNet+multijetAttention_9_9_9_np1904_lr0.008_epochs40_stdscale_epoch40_loss0.1754.pkl
##reweightModel_TTmix=ZZ4b/nTupleAnalysis/pytorchModels/3bTo3bMix4bTTmixFvT_ResNet+multijetAttention_9_9_9_np1904_lr0.008_epochs40_stdscale_epoch6_loss0.1765.pkl
#reweightModel_TTmix=ZZ4b/nTupleAnalysis/pytorchModels/3bTo3bMix4bTTmixFvT_ResNet+multijetAttention_8_8_8_np1579_lr0.004_epochs40_stdscale_epoch9_loss0.1760.pkl
##python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.h5"              -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_${mixedName}_TTmix
##python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.h5"     -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_3bTo${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"     -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToHad_${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5" -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"        -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.h5"                          -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTTo2L2Nu _TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.h5"                       -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToHadronicc_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.h5"                   -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_TTmix
#
###
### Add SvB
###
##python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.h5"                -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_${mixedName}_TTmix
##python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.h5" -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_3bTo${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"       -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToHad_${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"   -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"          -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.h5"                            -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTTo2L2Nu _TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.h5"                         -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToHadronicc_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.h5"                     -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_TTmix
#
##
## Convert hdf5 to root
##   (with conversion enviorment)
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.h5"                 2>&1 |tee ${outputDir}/log_Convert_h5ROOT_${mixedName}_TTmix
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.h5"  2>&1 |tee ${outputDir}/log_Convert_h5ROOT_3bTo${mixedName}_TTmix
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"        2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToHad_${mixedName}_TTmix
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"    2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_${mixedName}_TTmix
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"           2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_${mixedName}_TTmix
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.h5"                             2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTTo2L2Nu _TTmix
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.h5"                          2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToHadronicc_TTmix
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.h5"                      2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_TTmix
#
#
##
## Make hists of data with classifiers (Reweighted)
##
## 4b
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.root           -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_${mixedName}_wWeights.root       --is3bMixed      |tee ${outputDir}/log_${mixedName}_TTmix_wWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root        -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTTo2L2Nu_TTmix_wWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root     -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTToHadronic_TTmix_wWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_wWeights &
### 3b
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.root  -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_3bTo${mixedName}_wWeights.root   -j ${JCMNameTTmixed}   |tee ${outputDir}/log_3bTo${mixedName}_TTmix_wWeights &
#
##hadd -f ${outputPath}/${outputDir}/TT2018/hists_TTmix_wWeights.root ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/hists_wWeights_TTmix.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/hists_wWeights_TTmix.root ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/hists_wWeights_TTmix.root
#
##
## MAke Plots
##
##python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plots_TTmix -l 60.0e3 -y 2018 -m -r  --data ${outputDir}/data2018AllEvents_TTmix/hists_${mixedName}_wWeights.root --TT ${outputDir}/TT2018/hists_TTmix_wWeights.root --data3b  ${outputDir}/data2018AllEvents_TTmix/hists_3bTo${mixedName}_wWeights.root  --noSignal
##tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plots_TTmix.tar plots_TTmix
#
#
#
##
## Make hists of data with classifiers (Un=Reweighted)
##
## 4b
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.root            -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_${mixedName}_noWeights.root       --is3bMixed      |tee ${outputDir}/log_${mixedName}_TTmix_noWeights &
###$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root      -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed  |tee ${outputDir}/log_TTTo2L2Nu_TTmix_noWeights &
###$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root   -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed   |tee ${outputDir}/log_TTToHadronic_TTmix_noWeights &
###$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed  |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_noWeights &
#### 3b
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.root   -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_3bTo${mixedName}_noWeights.root   -j ${JCMNameTTmixed}   |tee ${outputDir}/log_3bTo${mixedName}_TTmix_noWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.root  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_noWeights_3b &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.root      -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTToHadronic_TTmix_noWeights_3b &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.root         -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTTo2L2Nu_TTmix_noWeights_3b &
#
#
#
##hadd -f ${outputPath}/${outputDir}/TT2018/hists_noWeights_TTmix.root \
##    ${outputPath}/${outputDir}/TTToHadronic2018/hists_noWeights_TTmix.root \
##    ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_noWeights_TTmix.root \
##    ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_noWeights_TTmix.root
#
##python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputPath}/${outputDir}/data2018AllEvents_TTmix/hists_3bTo${mixedName}_noWeights.root --tt ${outputPath}/${outputDir}/TT2018/hists_noWeights_TTmix.root -q   ${outputPath}/${outputDir}/qcd2018/hists_noWeights_TTmix.root
#
#
##python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plots_TTmixBeforeRW -l 60.0e3 -y 2018 -m  --data ${outputDir}/data2018AllEvents_TTmix/hists_${mixedName}_noWeights.root --TT ${outputDir}/TT2018/hists_TTmix_wWeights.root --qcd  ${outputDir}/qcd2018/hists_noWeights_TTmix.root  --noSignal
##tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plots_TTmixBeforeRW.tar plots_TTmixBeforeRW
#
#
#
#
##
##  Debugging 
##
##python -i ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD*${mixedName}*.h5" -t "${outputPath}/${outputDir}/TTTo*2018*/picoAOD_*ix*.h5" -o ${outputDir}/plotsHDF5_wTTmix
#
#
##
##  With TTbar Veto
##
#
##
##  Mix "3b" with 4b hemis to make "3bMix4b" evnets (With  TTVeto)
##
##$runCMD $runJOB -i ${file3bUnmixed}  -p picoAOD_${mixedName}_ttVeto.root  -o ${outputPath}/${outputDir}/ -y 2018  --histogramming 10 --histFile hists_${mixedName}_ttVeto.root   --is3bMixed --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputPath}/${outputDirMix}/dataHemis/data18wMCBranches/hemiSphereLib_4TagEvents_*root"| tee ${outputDir}/logMix_${mixedName}_ttVeto
#
#
##$runCMD $runJOB  -i ${fileTTToSemiLeptonic_noMjj_3bUnmixed} -p picoAOD_${mixedName}_ttVeto.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_${mixedName}_ttVeto.root   --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputPath}/${outputDirMix}/dataHemis/data18wMCBranches/hemiSphereLib_4TagEvents_*root" --bTagSF -l 60.0e3 --isMC  2>&1  |tee ${outputDir}/log_TTToSemiLeptonic2018_${mixedName}_ttVeto &
##$runCMD $runJOB  -i ${fileTTToHadronic_noMjj_3bUnmixed}     -p picoAOD_${mixedName}_ttVeto.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_${mixedName}_ttVeto.root   --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputPath}/${outputDirMix}/dataHemis/data18wMCBranches/hemiSphereLib_4TagEvents_*root" --bTagSF -l 60.0e3 --isMC  2>&1  |tee ${outputDir}/log_TTToHadronic2018_${mixedName}_ttVeto &
##$runCMD $runJOB  -i ${fileTTTo2L2Nu_noMjj_3bUnmixed}        -p picoAOD_${mixedName}_ttVeto.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_${mixedName}_ttVeto.root   --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputPath}/${outputDirMix}/dataHemis/data18wMCBranches/hemiSphereLib_4TagEvents_*root" --bTagSF -l 60.0e3 --isMC   2>&1 |tee ${outputDir}/log_TTTo2L2Nu2018_${mixedName}_ttVeto &
#
#
#
##hadd -f ${outputPath}/${outputDir}/TT2018/hists_${mixedName}_ttVeto.root ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/hists_${mixedName}_ttVeto.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/hists_${mixedName}_ttVeto.root ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/hists_${mixedName}_ttVeto.root
#
#
##
##  Fit JCM
##
##python $weightJOB -d ${outputPath}/${outputDirNom}/data2018AllEvents/data18/hists.root  --data4b ${outputDir}/${name3bUnmixed}/hists_${mixedName}_ttVeto.root  --tt ${outputPath}/${outputDirNom}/TT2018/hists.root  --tt4b ${outputPath}/${outputDir}/TT2018/hists_${mixedName}_ttVeto.root  -c passXWt   -o ${outputPath}/${outputDir}/weights/${mixedName}_wMixedTT_ttVeto/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_makeWeights_wMixedTT_ttVeto
#JCMNameTTmixedttVeto=${outputDir}/weights/${mixedName}_wMixedTT_ttVeto/jetCombinatoricModel_SB_00-00-01.txt
#
#
##
## Make picoAODS of 3b data with weights applied
##
##$runCMD $runJOB  -i ${fileAllEvents}        -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.root -y 2018  --histogramming 10 --histFile hists_wJCM_TTmix_3bTo${mixedName}_ttVeto.root --skip4b    -j ${JCMNameTTmixedttVeto}  |tee ${outputDir}/log_3bTo${mixedName}_JCM_TTmix_ttVeto  &
##$runCMD $runJOB  -i ${fileTTToHadronic}     -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix_ttVeto.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix_ttVeto.root --skip4b   --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixedttVeto}  |tee ${outputDir}/log_TTToHadronic2018_JCM_TTmix_ttVeto &
##$runCMD $runJOB  -i ${fileTTToSemiLeptonic} -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix_ttVeto.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix_ttVeto.root --skip4b   --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixedttVeto}  |tee ${outputDir}/log_TTToSemiLeptonic2018_JCM_TTmix_ttVeto &
##$runCMD $runJOB  -i ${fileTTTo2L2Nu}        -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix_ttVeto.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix_ttVeto.root --skip4b   --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixedttVeto}  |tee ${outputDir}/log_TTTo2L2Nu2018_JCM_TTmix_ttVeto &
#
#
##hadd -f ${outputPath}/${outputDir}/TT2018/hists_wJCM_TTmix_ttVeto.root ${outputPath}/${outputDir}/TTToHadronic2018/hists_wJCM_TTmix_ttVeto.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_wJCM_TTmix_ttVeto.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_wJCM_TTmix_ttVeto.root
#
#
#
##
## Convert root to hdf5 
##   (with conversion enviorment)
## 4b
##python $convertToH5JOB -i "${outputPath}/${outputDir}/${name3bUnmixed}/picoAOD_${mixedName}_ttVeto.root"                     2>&1 |tee ${outputDir}/log_Convert_ROOTh5_${mixedName}_ttVeto &	   
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.root"        2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTTo2L2Nu_mixed_ttVeto &         
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.root"     2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToHadronic_mixed_ttVeto &      
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.root" 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToSemiLep_mixede_ttVeto &       
### 3b
##python $convertToH5JOB -i "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.root"   2>&1 |tee ${outputDir}/log_Convert_ROOTh5_wJCM_TTmix_3bTo${mixedName}_ttVeto & 
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix_ttVeto.root"                        2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTTo2L2Nu_3b_mixed_ttVeto &
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix_ttVeto.root"                     2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToHadronic_3b_mixed_ttVeto &      
##python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix_ttVeto.root"                 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToSemiLep_3b_mixed_ttVeto &       
#
#
#
##
## Train
##   (with GPU conversion enviorment)
##mkdir ${outputDir}/data2018AllEvents_TTmix_ttVeto
## 4b
##ln -s ${outputPath}/${outputDir}/${name3bUnmixed}dataAllEvents_TTmix_ttVeto
##cp ${outputPath}/${outputDir}/${name3bUnmixed}/picoAOD_${mixedName}_ttVeto.h5 ${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_${mixedName}_ttVeto.h5
##cp ${outputPath}/${outputDir}/${name3bUnmixed}/picoAOD_${mixedName}_ttVeto.root ${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_${mixedName}_ttVeto.root
##mv ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.h5   ${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/
##mv ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.root ${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/
##python $trainJOB -c FvT -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD*${mixedName}*.h5" -t "${outputPath}/${outputDir}/TTTo*2018*/picoAOD_*ix*.h5" -e 40 --outputName 3bTo${mixedName}TTmixttVetov2 2>&1 |tee ${outputDir}/log_Train_FvT_3bTo${mixedName}_TTmix_ttVeto_v2
#
#
##
## Add FvT
##
##reweightModel_TTmix_ttVeto=ZZ4b/nTupleAnalysis/pytorchModels/3bTo3bMix4bTTmixFvTttVeto_ResNet+multijetAttention_8_8_8_np1579_lr0.004_epochs40_stdscale_epoch8_loss0.1598.pkl
#reweightModel_TTmix_ttVeto=ZZ4b/nTupleAnalysis/pytorchModels/3bTo3bMix4bTTmixttVetov2FvT_ResNet+multijetAttention_8_8_8_np1579_lr0.004_epochs40_stdscale_epoch6_loss0.1484.pkl
##python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_${mixedName}_ttVeto.h5"              -m $reweightModel_TTmix_ttVeto -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_${mixedName}_TTmix_ttVeto
##python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.h5"     -m $reweightModel_TTmix_ttVeto -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_3bTo${mixedName}_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5"     -m $reweightModel_TTmix_ttVeto -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToHad_${mixedName}_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5" -m $reweightModel_TTmix_ttVeto -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5"        -m $reweightModel_TTmix_ttVeto -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_${mixedName}_TTmix
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix_ttVeto.h5"                          -m $reweightModel_TTmix_ttVeto -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTTo2L2Nu_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix_ttVeto.h5"                       -m $reweightModel_TTmix_ttVeto -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToHadronicc_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix_ttVeto.h5"                   -m $reweightModel_TTmix_ttVeto -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_TTmix_ttVeto
#
###
### Add SvB
###
##python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_${mixedName}_ttVeto.h5"                -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_${mixedName}_TTmix_ttVeto
##python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.h5" -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_3bTo${mixedName}_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5"       -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToHad_${mixedName}_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5"   -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_${mixedName}_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5"          -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_${mixedName}_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix_ttVeto.h5"                            -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTTo2L2Nu _TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix_ttVeto.h5"                         -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToHadronicc_TTmix_ttVeto
##python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix_ttVeto.h5"                     -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_TTmix_ttVeto
#
#
##
##  Debugging 
##
##python -i ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD*${mixedName}*.h5" -t "${outputPath}/${outputDir}/TTTo*2018*/picoAOD_*ix*.h5" -o ${outputDir}/plotsHDF5_wTTmix_ttVeto_v2
#
#
#
#
##
## Convert hdf5 to root
##   (with conversion enviorment)
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_${mixedName}_ttVeto.h5"                 2>&1 |tee ${outputDir}/log_Convert_h5ROOT_${mixedName}_TTmix_ttVeto
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.h5"  2>&1 |tee ${outputDir}/log_Convert_h5ROOT_3bTo${mixedName}_TTmix_ttVeto
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5"        2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToHad_${mixedName}_TTmix_ttVeto
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5"    2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_${mixedName}_TTmix_ttVeto
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.h5"           2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_${mixedName}_TTmix_ttVeto
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix_ttVeto.h5"                             2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTTo2L2Nu _TTmix_ttVeto
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix_ttVeto.h5"                          2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToHadronicc_TTmix_ttVeto
##python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix_ttVeto.h5"                      2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_TTmix_ttVeto
#
#
##
## Make hists of data with classifiers (Reweighted)
##
## 4b
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_${mixedName}_ttVeto.root           -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_${mixedName}_wWeights_ttVeto.root       --is3bMixed      |tee ${outputDir}/log_${mixedName}_TTmix_wWeights_ttVeto &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.root        -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix_ttVeto.root              -j ${JCMNameTTmixedttVeto}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTTo2L2Nu_TTmix_wWeights_ttVeto &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.root     -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix_ttVeto.root              -j ${JCMNameTTmixedttVeto}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTToHadronic_TTmix_wWeights_ttVeto &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}_ttVeto.root -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix_ttVeto.root              -j ${JCMNameTTmixedttVeto}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_wWeights_ttVeto &
### 3b
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.root  -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_3bTo${mixedName}_wWeights_ttVeto.root   -j ${JCMNameTTmixedttVeto}   |tee ${outputDir}/log_3bTo${mixedName}_TTmix_wWeights_ttVeto &
#
##hadd -f ${outputPath}/${outputDir}/TT2018/hists_TTmix_wWeights_ttVeto.root ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/hists_wWeights_TTmix_ttVeto.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/hists_wWeights_TTmix_ttVeto.root ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/hists_wWeights_TTmix_ttVeto.root
#
#
##
## MAke Plots
##
##python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plots_TTmix_ttVeto -l 60.0e3 -y 2018 -m -r  --data ${outputDir}/data2018AllEvents_TTmix_ttVeto/hists_${mixedName}_wWeights_ttVeto.root --TT ${outputDir}/TT2018/hists_TTmix_wWeights_ttVeto.root --data3b  ${outputDir}/data2018AllEvents_TTmix_ttVeto/hists_3bTo${mixedName}_wWeights_ttVeto.root  --noSignal
##tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plots_TTmix_ttVeto.tar plots_TTmix_ttVeto
#
#
#
#
##
## Make hists of data with classifiers (Un=Reweighted)
##
## 4b
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.root            -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_${mixedName}_noWeights.root       --is3bMixed      |tee ${outputDir}/log_${mixedName}_TTmix_noWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root      -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed  |tee ${outputDir}/log_TTTo2L2Nu_TTmix_noWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root   -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed   |tee ${outputDir}/log_TTToHadronic_TTmix_noWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root              -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed  |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_noWeights &
### 3b
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix_ttVeto/picoAOD_wJCM_TTmix_3bTo${mixedName}_ttVeto.root   -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_3bTo${mixedName}_noWeights_ttVeto.root   -j ${JCMNameTTmixedttVeto}   |tee ${outputDir}/log_3bTo${mixedName}_TTmix_noWeights_ttVeto &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix_ttVeto.root  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix_ttVeto.root              -j ${JCMNameTTmixedttVeto}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_noWeights_3b_ttVeto &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix_ttVeto.root      -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix_ttVeto.root              -j ${JCMNameTTmixedttVeto}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTToHadronic_TTmix_noWeights_3b_ttVeto &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix_ttVeto.root         -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix_ttVeto.root              -j ${JCMNameTTmixedttVeto}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTTo2L2Nu_TTmix_noWeights_3b_ttVeto &
#
#
##hadd -f ${outputPath}/${outputDir}/TT2018/hists_noWeights_TTmix_ttVeto.root \
##    ${outputPath}/${outputDir}/TTToHadronic2018/hists_noWeights_TTmix_ttVeto.root \
##    ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_noWeights_TTmix_ttVeto.root \
##    ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_noWeights_TTmix_ttVeto.root
#
##
