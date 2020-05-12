#
# In the following "3b" refers to 3b subsampled to have the 4b statistics
#
outputPath=/uscms/home/jda102/nobackup/NanoAODs/CMSSW_10_2_0/src/
outputDir=HemiClosureTest3bMix3b/
outputDirNom=HemiClosureTest/
outputDirMix=HemiClosureTestMixed/

fileAllEvents=${outputPath}/${outputDirNom}/fileLists/data2018AllEvents.txt
fileTTToHadronic=${outputPath}/${outputDirNom}/fileLists/TTToHadronic2018.txt
fileTTToSemiLeptonic=${outputPath}/${outputDirNom}/fileLists/TTToSemiLeptonic2018.txt
fileTTTo2L2Nu=${outputPath}/${outputDirNom}/fileLists/TTTo2L2Nu2018.txt

fileTTToHadronic_noMjj_3bUnmixed=${outputPath}/${outputDirMix}/fileLists/TTToHadronic2018_noMjj_3bUnmixed.txt
fileTTToSemiLeptonic_noMjj_3bUnmixed=${outputPath}/${outputDirMix}/fileLists/TTToSemiLeptonic2018_noMjj_3bUnmixed.txt
fileTTTo2L2Nu_noMjj_3bUnmixed=${outputPath}/${outputDirMix}/fileLists/TTTo2L2Nu2018_noMjj_3bUnmixed.txt


# 3b Unmixed
name3bUnmixed=data2018_3bUnmixed
pico3bUnmixed=picoAOD_${name3bUnmixed}.root
path3bUnmixed=${name3bUnmixed}
file3bUnmixed=${outputDirMix}/fileLists/${name3bUnmixed}.txt

# mixed
mixedName=3bMix3b


# Helpers
runCMD=nTupleAnalysis 
runJOB=ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py
weightJOB=ZZ4b/nTupleAnalysis/scripts/makeWeights.py
convertToH5JOB=ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py
SvBModel=ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1638_lr0.008_epochs40_stdscale_epoch39_loss0.1985.pkl
trainJOB=ZZ4b/nTupleAnalysis/scripts/multiClassifier.py
convertToROOTJOB=ZZ4b/nTupleAnalysis/scripts/convert_h52root.py




#
#  Mix "3b" with 4b hemis to make "3bMix3b" evnets
#
#$runCMD $runJOB -i ${outputDir}/${file3bUnmixed}  -p picoAOD_${mixedName}.root  -o ${outputPath}/${outputDir}/ -y 2018  --histogramming 10 --histFile hists_${mixedName}.root  --nevents -1 --is3bMixed --loadHemisphereLibrary --maxNHemis 10000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputPath}/${outputDirMix}/dataHemis/data18wMCBranches/hemiSphereLib_3TagEvents_*root"| tee ${outputDir}/logMix_${mixedName}


#
#  Make the JCM-weights w/ttbar at passXWt
#
#mkdir ${outputDir}/weights
#python $weightJOB -d ${outputPath}/${outputDirNom}/data2018AllEvents/data18/hists.root  --data4b ${outputDir}/${name3bUnmixed}/hists_${mixedName}.root  --tt ${outputPath}/${outputDirNom}/TT2018/hists.root  -c passXWt   -o ${outputPath}/${outputDir}/weights/${mixedName}/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_makeWeights
JCMName=${outputDir}/weights/${mixedName}/jetCombinatoricModel_SB_00-00-01.txt

#
# Make picoAODS of 3b data with weights applied
#
#$runCMD $runJOB  -i ${fileAllEvents}        -o ${outputPath}/${outputDir} -p picoAOD_wJCM_3bTo${mixedName}.root -y 2018  --histogramming 10 --histFile hists_wJCM_3bTo${mixedName}.root --skip4b  --nevents -1  -j ${JCMName}  |tee ${outputDir}/log_3bTo${mixedName}_JCM  &
#$runCMD $runJOB  -i ${fileTTToHadronic}     -o ${outputPath}/${outputDir} -p picoAOD_wJCM.root -y 2018 --histogramming 10  --histFile hists_wJCM.root  --nevents -1 --bTagSF -l 60.0e3 --isMC  -j ${JCMName}  |tee ${outputDir}/log_TTToHadronic2018_JCM &
#$runCMD $runJOB  -i ${fileTTToSemiLeptonic} -o ${outputPath}/${outputDir} -p picoAOD_wJCM.root -y 2018 --histogramming 10  --histFile hists_wJCM.root  --nevents -1 --bTagSF -l 60.0e3 --isMC  -j ${JCMName}  |tee ${outputDir}/log_TTToSemiLeptonic2018_JCM &
#$runCMD $runJOB  -i ${fileTTTo2L2Nu}        -o ${outputPath}/${outputDir} -p picoAOD_wJCM.root -y 2018 --histogramming 10  --histFile hists_wJCM.root  --nevents -1 --bTagSF -l 60.0e3 --isMC  -j ${JCMName}  |tee ${outputDir}/log_TTTo2L2Nu2018_JCM &


#mkdir ${outputPath}/${outputDir}/TT2018
#hadd -f ${outputPath}/${outputDir}/TT2018/hists_wJCM.root ${outputPath}/${outputDir}/TTToHadronic2018/hists_wJCM.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_wJCM.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_wJCM.root

#
# Make QCD with out the reweights
#
#mkdir -p ${outputPath}/${outputDir}/qcd2018
#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputPath}/${outputDir}/data2018AllEvents/hists_wJCM_3bTo${mixedName}.root --tt ${outputPath}/${outputDir}/TT2018/hists_wJCM.root -q   ${outputPath}/${outputDir}/qcd2018/hists_wJCM.root

#
# Make Plots
#
#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plotsNoReWeight -l 60.0e3 -y 2018 -m
#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plotsTestNoReweight -l 60.0e3 -y 2018 -m  --data ${outputDir}/${name3bUnmixed}/hists_${mixedName}.root --TT ${outputDir}/TT2018/hists_wJCM.root  --qcd  ${outputDir}/qcd2018/hists_wJCM.root   --noSignal
#tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plotsTestNoReweight.tar plotsTestNoReweight


#
# Convert root to hdf5
#   (with conversion enviorment)
#python $convertToH5JOB -i "${outputPath}/${outputDir}/${name3bUnmixed}/picoAOD_${mixedName}.root"           2>&1 |tee ${outputDir}/log_Convert_ROOTh5_${mixedName} &
#python $convertToH5JOB -i "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_3bTo${mixedName}.root" 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bTo${mixedName} &
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.root"                      2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTTo2L2Nu & 	  
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.root" 	             2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToHadronic & 
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.root" 	             2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToSemiLep &       

#
# Train
#   (with GPU conversion enviorment)
#mv ${outputPath}/${outputDir}/${name3bUnmixed}/picoAOD_${mixedName}.h5 ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.h5
#python $trainJOB -c FvT -d "${outputPath}/${outputDir}/data2018AllEvents/picoAOD*${mixedName}*.h5" -t "${outputPath}/${outputDir}/TTTo*2018*/picoAOD_wJCM.h5" -e 40 -o 3bTo${mixedName} 2>&1 |tee ${outputDir}/log_Train_FvT_3bTo${mixedName}

#
#  Debugging 
#
#echo python -i ZZ4b/nTupleAnalysis/scripts/plotHDF5.py -d "${outputDir}/data2018AllEvents/picoAOD*3bMix3b*.h5" -t "${outputDir}/TTTo*2018*/picoAOD_wJCM.h5"

#
# Add FvT
#
reweightModel=ZZ4b/nTupleAnalysis/pytorchModels/3bTo3bMix3bFvT_ResNet+multijetAttention_9_9_9_np1904_lr0.008_epochs40_stdscale_epoch37_loss0.1675.pkl
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.h5"          -m $reweightModel -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_${mixedName}
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_3bTo${mixedName}.h5" -m $reweightModel -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_3bTo${mixedName}
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.h5"                      -m $reweightModel -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTTo2L2Nu       
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.h5"                   -m $reweightModel -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToHadronicc
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.h5"               -m $reweightModel -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic

#
# Add SvB
#
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_3bTo${mixedName}.h5" -m ${SvBModel}   -c SvB  2>&1 |tee  ${outputDir}/log_Add_SvB_3bTo${mixedName}
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.h5"          -m ${SvBModel}   -c SvB  2>&1 |tee  ${outputDir}/log_Add_SvB_${mixedName}
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.h5"                      -m ${SvBModel}   -c SvB  2>&1 |tee  ${outputDir}/log_Add_SvB_TTTo2L2Nu       
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.h5"                   -m ${SvBModel}   -c SvB  2>&1 |tee  ${outputDir}/log_Add_SvB_TTToHadronicc
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.h5"               -m ${SvBModel}   -c SvB  2>&1 |tee  ${outputDir}/log_Add_SvB_TTToSemiLeptonic

#
# Convert hdf5 to root
#   (with conversion enviorment)
#mv ${outputPath}/${outputDir}/${name3bUnmixed}/picoAOD_${mixedName}.root ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.root
#python $convertToROOTJOB  -i "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_3bTo${mixedName}.h5" 2>&1 |tee ${outputDir}/log_Convert_h5ROOT_3bTo${mixedName}  &
#python $convertToROOTJOB  -i "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.h5"          2>&1 |tee ${outputDir}/log_Convert_h5ROOT_${mixedName}
#python $convertToROOTJOB  -i "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.h5"                      2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTTo2L2Nu  & 
#python $convertToROOTJOB  -i "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.h5"                   2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToHadronic  & 
#python $convertToROOTJOB  -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.h5"               2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLep &   



#
# Make hists of data with classifiers
#
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_3bTo${mixedName}.root -r -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_3bTo${mixedName}_wWeights.root  --nevents -1 -j ${JCMName}   |tee ${outputDir}/log_3bTo${mixedName}_wWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.root          -r -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_${mixedName}_wWeights.root      --nevents -1 --is3bMixed      |tee ${outputDir}/log_${mixedName}_wWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.root                      -r -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root              --nevents -1 -j ${JCMName}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTTo2L2Nu_wWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.root                   -r -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root              --nevents -1 -j ${JCMName}  --bTagSF -l 60.0e3 --isMC  |tee ${outputDir}/log_TTToHadronic_wWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.root               -r -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root              --nevents -1 -j ${JCMName}  --bTagSF -l 60.0e3 --isMC  |tee ${outputDir}/log_TTToSemiLeptonic_wWeights &



#hadd -f ${outputPath}/${outputDir}/TT2018/hists_wWeights.root ${outputPath}/${outputDir}/TTToHadronic2018/hists_wWeights.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_wWeights.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_wWeights.root



#
# MAke Plots
#
#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plots -l 60.0e3 -y 2018 -m -r --data ${outputDir}/data2018AllEvents/hists_${mixedName}_wWeights.root --TT ${outputDir}/TT2018/hists_wWeights.root --data3b  ${outputDir}/data2018AllEvents/hists_3bTo${mixedName}_wWeights.root  --noSignal
#tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plots.tar plots



#
# Make hists of data with classifiers (Un-ReWeighted)
#
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_3bTo${mixedName}.root -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_3bTo${mixedName}_noWeights.root  --nevents -1 -j ${JCMName}   |tee ${outputDir}/log_3bTo${mixedName}_noWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.root          -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_${mixedName}_noWeights.root      --nevents -1 --is3bMixed      |tee ${outputDir}/log_${mixedName}_noWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.root                      -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_noWeights.root              --nevents -1 -j ${JCMName}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTTo2L2Nu_noWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.root                   -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_noWeights.root              --nevents -1 -j ${JCMName}  --bTagSF -l 60.0e3 --isMC  |tee ${outputDir}/log_TTToHadronic_noWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.root               -p "" -y 2018  --histogramming 10 --histDetail 7 --histFile hists_noWeights.root              --nevents -1 -j ${JCMName}  --bTagSF -l 60.0e3 --isMC  |tee ${outputDir}/log_TTToSemiLeptonic_noWeights &

#hadd -f ${outputPath}/${outputDir}/TT2018/hists_noWeights.root ${outputPath}/${outputDir}/TTToHadronic2018/hists_noWeights.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_noWeights.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_noWeights.root

#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputPath}/${outputDir}/data2018AllEvents/hists_3bTo${mixedName}_noWeights.root --tt ${outputPath}/${outputDir}/TT2018/hists_noWeights.root -q   ${outputPath}/${outputDir}/qcd2018/hists_noWeights.root

#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plotsBeforeRW -l 60.0e3 -y 2018 -m  --data ${outputDir}/data2018AllEvents/hists_${mixedName}_noWeights.root --TT ${outputDir}/TT2018/hists_noWeights.root --qcd  ${outputDir}/qcd2018/hists_noWeights.root  --noSignal
#tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plotsBeforeRW.tar plotsBeforeRW


#
#  Clean up 
#
#rm -rf  HemiClosureTest3bMix4b/TT*/picoAOD.root 

######################
### TTBar Mixing
######################


#
#  Mix "4b" ttbar
#
#$runCMD $runJOB  -i ${fileTTToSemiLeptonic_noMjj_3bUnmixed} -p picoAOD_${mixedName}.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_${mixedName}.root  --nevents -1 --loadHemisphereLibrary --maxNHemis 10000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputPath}/${outputDirMix}/dataHemis/data18wMCBranches/hemiSphereLib_3TagEvents_*root" --bTagSF -l 60.0e3 --isMC  2>&1  |tee ${outputDir}/log_TTToSemiLeptonic2018_${mixedName} &
#$runCMD $runJOB  -i ${fileTTToHadronic_noMjj_3bUnmixed}     -p picoAOD_${mixedName}.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_${mixedName}.root  --nevents -1 --loadHemisphereLibrary --maxNHemis 10000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputPath}/${outputDirMix}/dataHemis/data18wMCBranches/hemiSphereLib_3TagEvents_*root" --bTagSF -l 60.0e3 --isMC  2>&1  |tee ${outputDir}/log_TTToHadronic2018_${mixedName} &
#$runCMD $runJOB  -i ${fileTTTo2L2Nu_noMjj_3bUnmixed}        -p picoAOD_${mixedName}.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_${mixedName}.root  --nevents -1 --loadHemisphereLibrary --maxNHemis 10000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputPath}/${outputDirMix}/dataHemis/data18wMCBranches/hemiSphereLib_3TagEvents_*root" --bTagSF -l 60.0e3 --isMC   2>&1 |tee ${outputDir}/log_TTTo2L2Nu2018_${mixedName} &


#hadd -f ${outputPath}/${outputDir}/TT2018/hists_${mixedName}.root \
#    ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/hists_${mixedName}.root \
#    ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/hists_${mixedName}.root \
#    ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/hists_${mixedName}.root


#
#  Fit JCM
#
#python $weightJOB -d ${outputPath}/${outputDirNom}/data2018AllEvents/data18/hists.root  --data4b ${outputDir}/${name3bUnmixed}/hists_${mixedName}.root  --tt ${outputPath}/${outputDirNom}/TT2018/hists.root  --tt4b ${outputPath}/${outputDir}/TT2018/hists_${mixedName}.root  -c passXWt   -o ${outputPath}/${outputDir}/weights/${mixedName}_wMixedTT/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_makeWeights_wMixedTT
JCMNameTTmixed=${outputDir}/weights/${mixedName}_wMixedTT/jetCombinatoricModel_SB_00-00-01.txt

#
# Make picoAODS of 3b data with weights applied
#
#$runCMD $runJOB  -i ${fileAllEvents}        -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix_3bTo${mixedName}.root -y 2018  --histogramming 10 --histFile hists_wJCM_TTmix_3bTo${mixedName}.root --skip4b  --nevents -1  -j ${JCMNameTTmixed}  |tee ${outputDir}/log_3bTo${mixedName}_JCM_TTmix  &
#$runCMD $runJOB  -i ${fileTTToHadronic}     -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix.root --skip4b  --nevents -1 --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixed}  |tee ${outputDir}/log_TTToHadronic2018_JCM_TTmix &
#$runCMD $runJOB  -i ${fileTTToSemiLeptonic} -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix.root --skip4b  --nevents -1 --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixed}  |tee ${outputDir}/log_TTToSemiLeptonic2018_JCM_TTmix &
#$runCMD $runJOB  -i ${fileTTTo2L2Nu}        -o ${outputPath}/${outputDir} -p picoAOD_wJCM_TTmix.root -y 2018 --histogramming 10  --histFile hists_wJCM_TTmix.root --skip4b  --nevents -1 --bTagSF -l 60.0e3 --isMC  -j ${JCMNameTTmixed}  |tee ${outputDir}/log_TTTo2L2Nu2018_JCM_TTmix &


#
# Convert root to hdf5 
#   (with conversion enviorment)
## 4b
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root"                      2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTTo2L2Nu_mixed &         
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root"                   2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToHadronic_mixed &      
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root"               2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToSemiLep_mixede &       
### 3b
#python $convertToH5JOB -i "${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}.root"   2>&1 |tee ${outputDir}/log_Convert_ROOTh5_wJCM_TTmix_3bTo${mixedName} &         
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.root"                   2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTTo2L2Nu_3b_mixed &         
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.root"                2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToHadronic_3b_mixed &      
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.root"            2>&1 |tee ${outputDir}/log_Convert_ROOTh5_TTToSemiLep_3b_mixed &       


#
# Train
#   (with GPU conversion enviorment)
#mkdir ${outputDir}/data2018AllEvents_TTmix
# 4b
#cp ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.h5 ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.h5
#cp ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_${mixedName}.root ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.root
#mv ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}.h5   ${outputPath}/${outputDir}/data2018AllEvents_TTmix/
#mv ${outputPath}/${outputDir}/data2018AllEvents/picoAOD_wJCM_TTmix_3bTo${mixedName}.root ${outputPath}/${outputDir}/data2018AllEvents_TTmix/
#python $trainJOB -c FvT -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD*${mixedName}*.h5" -t "${outputPath}/${outputDir}/TTTo*2018*/picoAOD_*ix*.h5" -e 40 --outputName 3bTo${mixedName}TTmix 2>&1 |tee ${outputDir}/log_Train_FvT_3bTo${mixedName}_TTmix



#
# Add FvT
#
reweightModel_TTmix=ZZ4b/nTupleAnalysis/pytorchModels/3bTo3bMix3bTTmixFvT_ResNet+multijetAttention_8_8_8_np1579_lr0.004_epochs40_stdscale_epoch9_loss0.1729.pkl
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.h5"              -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_${mixedName}_TTmix
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.h5"     -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_3bTo${mixedName}_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"     -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToHad_${mixedName}_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5" -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_${mixedName}_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"        -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_${mixedName}_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.h5"                          -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTTo2L2Nu _TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.h5"                       -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToHadronicc_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.h5"                   -m $reweightModel_TTmix -c FvT  2>&1 |tee ${outputDir}/log_Add_FvT_TTToSemiLeptonic_TTmix

##
## Add SvB
##
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.h5"                -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_${mixedName}_TTmix
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.h5" -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_3bTo${mixedName}_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"       -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToHad_${mixedName}_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"   -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_${mixedName}_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"          -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_${mixedName}_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.h5"                            -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTTo2L2Nu _TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.h5"                         -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToHadronicc_TTmix
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.h5"                     -m ${SvBModel} -c SvB  2>&1 |tee ${outputDir}/log_Add_SvB_TTToSemiLeptonic_TTmix

#python -i ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py -d "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD*${mixedName}*.h5" -t "${outputPath}/${outputDir}/TTTo*2018*/picoAOD_*ix*.h5" -o ${outputDir}/plotsHDF5_wTTmix

#
# Convert hdf5 to root
#   (with conversion enviorment)
#python $convertToROOTJOB -i  "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.h5"                 2>&1 |tee ${outputDir}/log_Convert_h5ROOT_${mixedName}_TTmix
#python $convertToROOTJOB -i  "${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.h5"  2>&1 |tee ${outputDir}/log_Convert_h5ROOT_3bTo${mixedName}_TTmix
#python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"        2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToHad_${mixedName}_TTmix
#python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"    2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_${mixedName}_TTmix
#python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.h5"           2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_${mixedName}_TTmix
#python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.h5"                             2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTTo2L2Nu _TTmix
#python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.h5"                          2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToHadronicc_TTmix
#python $convertToROOTJOB -i  "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.h5"                      2>&1 |tee ${outputDir}/log_Convert_h5ROOT_TTToSemiLeptonic_TTmix



#
# Make hists of data with classifiers (Reweighted)
#
# 4b
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.root           -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_${mixedName}_wWeights.root      --nevents -1 --is3bMixed      |tee ${outputDir}/log_${mixedName}_TTmix_wWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root        -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix.root             --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTTo2L2Nu_TTmix_wWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root     -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix.root             --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTToHadronic_TTmix_wWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights_TTmix.root             --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC  --is3bMixed   |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_wWeights &
## 3b
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.root  -r -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_3bTo${mixedName}_wWeights.root  --nevents -1 -j ${JCMNameTTmixed}   |tee ${outputDir}/log_3bTo${mixedName}_TTmix_wWeights &

#hadd -f ${outputPath}/${outputDir}/TT2018/hists_TTmix_wWeights.root ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/hists_wWeights_TTmix.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/hists_wWeights_TTmix.root ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/hists_wWeights_TTmix.root

#
# MAke Plots
#
#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plots_TTmix -l 60.0e3 -y 2018 -m -r  --data ${outputDir}/data2018AllEvents_TTmix/hists_${mixedName}_wWeights.root --TT ${outputDir}/TT2018/hists_TTmix_wWeights.root --data3b  ${outputDir}/data2018AllEvents_TTmix/hists_3bTo${mixedName}_wWeights.root  --noSignal
#tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plots_TTmix.tar plots_TTmix



#
# Make hists of data with classifiers (Un=Reweighted)
#
# 4b
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_${mixedName}.root            -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_${mixedName}_noWeights.root      --nevents -1 --is3bMixed      |tee ${outputDir}/log_${mixedName}_TTmix_noWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root      -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root             --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed  |tee ${outputDir}/log_TTTo2L2Nu_TTmix_noWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root   -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root             --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed   |tee ${outputDir}/log_TTToHadronic_TTmix_noWeights &
##$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj_3bUnmixed/picoAOD_${mixedName}.root  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root             --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC --is3bMixed  |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_noWeights &
### 3b
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents_TTmix/picoAOD_wJCM_TTmix_3bTo${mixedName}.root   -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_3bTo${mixedName}_noWeights.root  --nevents -1 -j ${JCMNameTTmixed}   |tee ${outputDir}/log_3bTo${mixedName}_TTmix_noWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM_TTmix.root  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root  --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTToSemiLeptonic_TTmix_noWeights_3b &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM_TTmix.root      -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root  --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTToHadronic_TTmix_noWeights_3b &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM_TTmix.root         -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights_TTmix.root  --nevents -1 -j ${JCMNameTTmixed}  --bTagSF -l 60.0e3 --isMC   |tee ${outputDir}/log_TTTo2L2Nu_TTmix_noWeights_3b &




#hadd -f ${outputPath}/${outputDir}/TT2018/hists_noWeights_TTmix.root \
#    ${outputPath}/${outputDir}/TTToHadronic2018/hists_noWeights_TTmix.root \
#    ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_noWeights_TTmix.root \
#    ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_noWeights_TTmix.root

#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputPath}/${outputDir}/data2018AllEvents_TTmix/hists_3bTo${mixedName}_noWeights.root --tt ${outputPath}/${outputDir}/TT2018/hists_noWeights_TTmix.root -q   ${outputPath}/${outputDir}/qcd2018/hists_noWeights_TTmix.root


python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plots_TTmixBeforeRW -l 60.0e3 -y 2018 -m  --data ${outputDir}/data2018AllEvents_TTmix/hists_${mixedName}_noWeights.root --TT ${outputDir}/TT2018/hists_TTmix_wWeights.root --qcd  ${outputDir}/qcd2018/hists_noWeights_TTmix.root  --noSignal
tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plots_TTmixBeforeRW.tar plots_TTmixBeforeRW
