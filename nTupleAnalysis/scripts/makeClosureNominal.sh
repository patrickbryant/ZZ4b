#
# In the following "3b" refers to 3b subsampled to have the 4b statistics
#
outputDir=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal
outputDirComb=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/combined


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'
convertToH5JOB=ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py
SvBModel=ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1692_lr0.008_epochs40_stdscale_epoch40_loss0.2070.pkl
trainJOB=ZZ4b/nTupleAnalysis/scripts/multiClassifier.py
convertToROOTJOB=ZZ4b/nTupleAnalysis/scripts/convert_h52root.py

YEAR2018=' -y 2018 --bTag 0.2770 '
YEAR2017=' -y 2017 --bTag 0.3033 '
YEAR2016=' -y 2016 --bTag 0.3093 '

YEAR2018MC=${YEAR2018}' --bTagSF -l 60.0e3 --isMC '
YEAR2017MC=${YEAR2017}' --bTagSF -l 36.7e3 --isMC '
YEAR2016MC=${YEAR2016}' --bTagSF -l 35.9e3 --isMC '


### 
###  Make Hists with all data
###
##$runCMD  -i ${outputDir}/fileLists/data2018.txt -p picoAOD.root  -o ${outputDir} $YEAR2018  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2018 &
##$runCMD  -i ${outputDir}/fileLists/data2017.txt -p picoAOD.root  -o ${outputDir} $YEAR2017  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2017 &
##$runCMD  -i ${outputDir}/fileLists/data2016.txt -p picoAOD.root  -o ${outputDir} $YEAR2016  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2016 &
##
##
###
###  Make Hists with all ttbar
###
###
### 2018
##$runCMD -i ${outputDir}/fileLists/TTToHadronic2018_noMjj.txt     -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTHad2018   & 
##$runCMD -i ${outputDir}/fileLists/TTToSemiLeptonic2018_noMjj.txt -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTSemi2018  &
##$runCMD -i ${outputDir}/fileLists/TTTo2L2Nu2018_noMjj.txt        -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TT2L2Nu2018 &
##
### 2017
##$runCMD -i ${outputDir}/fileLists/TTToHadronic2017_noMjj.txt     -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTHad2017   & 
##$runCMD -i ${outputDir}/fileLists/TTToSemiLeptonic2017_noMjj.txt -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTSemi2017  &
##$runCMD -i ${outputDir}/fileLists/TTTo2L2Nu2017_noMjj.txt        -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TT2L2Nu2017 &
##
### 2016
##$runCMD -i ${outputDir}/fileLists/TTToHadronic2016_noMjj.txt     -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTHad2016   & 
##$runCMD -i ${outputDir}/fileLists/TTToSemiLeptonic2016_noMjj.txt -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTSemi2016  &
##$runCMD -i ${outputDir}/fileLists/TTTo2L2Nu2016_noMjj.txt        -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TT2L2Nu2016 &


#
#  Hadd ttbar
#
#mkdir ${outputDir}/TT2018
#mkdir ${outputDir}/TT2017
#mkdir ${outputDir}/TT2016
#hadd -f ${outputDir}/TT2018/hists.root ${outputDir}/TTToHadronic2018_noMjj/hists.root ${outputDir}/TTToSemiLeptonic2018_noMjj/hists.root ${outputDir}/TTTo2L2Nu2018_noMjj/hists.root & 
#hadd -f ${outputDir}/TT2017/hists.root ${outputDir}/TTToHadronic2017_noMjj/hists.root ${outputDir}/TTToSemiLeptonic2017_noMjj/hists.root ${outputDir}/TTTo2L2Nu2017_noMjj/hists.root & 
#hadd -f ${outputDir}/TT2016/hists.root ${outputDir}/TTToHadronic2016_noMjj/hists.root ${outputDir}/TTToSemiLeptonic2016_noMjj/hists.root ${outputDir}/TTTo2L2Nu2016_noMjj/hists.root &



#
#  Make the JCM-weights
#
# (3b -> 4b)
#$weightCMD -d ${outputDir}/data2018/hists.root  --tt ${outputDir}/TT2018/hists.root -c passXWt  -o ${outputDir}/weights/data2018/  -r SB -w 00-00-02 -y 2018 -l 60.0e3 2>&1 |tee ${outputDir}/log_JCM2018 
#$weightCMD -d ${outputDir}/data2017/hists.root  --tt ${outputDir}/TT2017/hists.root -c passXWt  -o ${outputDir}/weights/data2017/  -r SB -w 00-00-02 -y 2017 -l 36.7e3 2>&1 |tee ${outputDir}/log_JCM2017 
#$weightCMD -d ${outputDir}/data2016/hists.root  --tt ${outputDir}/TT2016/hists.root -c passXWt  -o ${outputDir}/weights/data2016/  -r SB -w 00-00-02 -y 2016 -l 35.9e3 2>&1 |tee ${outputDir}/log_JCM2016 



#
#  Adding JCM weights now done in makeClosureTestCombined
#


#
#  Make 3b Hists with JCM weights applied (for cut flow )
#
JCMNAME=Nominal

## 2018
#$runCMD -i ${outputDirComb}/data2018/picoAOD_3b_wJCM.root             -p NONE $YEAR2018   --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME}  2>&1 |tee ${outputDir}/log_2018_wJCM_${JCMNAME}  &
#$runCMD -i ${outputDirComb}/TTToHadronic2018/picoAOD_3b_wJCM.root     -p NONE $YEAR2018MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME}  2>&1 |tee ${outputDir}/log_TTHad2018_wJCM_${JCMNAME} & 
#$runCMD -i ${outputDirComb}/TTToSemiLeptonic2018/picoAOD_3b_wJCM.root -p NONE $YEAR2018MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME}  2>&1 |tee ${outputDir}/log_TTSem2018_wJCM_${JCMNAME} & 
#$runCMD -i ${outputDirComb}/TTTo2L2Nu2018/picoAOD_3b_wJCM.root        -p NONE $YEAR2018MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME}  2>&1 |tee ${outputDir}/log_TT2L2Nu2018_wJCM_${JCMNAME}  & 
#
## 2017
#$runCMD -i ${outputDirComb}/data2017/picoAOD_3b_wJCM.root             -p NONE $YEAR2017   --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME} 2>&1 |tee ${outputDir}/log_2017_wJCM_${JCMNAME}  &
#$runCMD -i ${outputDirComb}/TTToHadronic2017/picoAOD_3b_wJCM.root     -p NONE $YEAR2017MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME} 2>&1 |tee ${outputDir}/log_TTHad2017_wJCM_${JCMNAME}   & 
#$runCMD -i ${outputDirComb}/TTToSemiLeptonic2017/picoAOD_3b_wJCM.root -p NONE $YEAR2017MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME} 2>&1 |tee ${outputDir}/log_TTSem2017_wJCM_${JCMNAME}   & 
#$runCMD -i ${outputDirComb}/TTTo2L2Nu2017/picoAOD_3b_wJCM.root        -p NONE $YEAR2017MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME} 2>&1 |tee ${outputDir}/log_TT2L2Nu2017_wJCM_${JCMNAME}   & 
#
## 2016
#$runCMD -i ${outputDirComb}/data2016/picoAOD_3b_wJCM.root             -p NONE $YEAR2016   --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME} 2>&1 |tee ${outputDir}/log_2016_wJCM_${JCMNAME}  &
#$runCMD -i ${outputDirComb}/TTToHadronic2016/picoAOD_3b_wJCM.root     -p NONE $YEAR2016MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME} 2>&1 |tee ${outputDir}/log_TTHad2016_wJCM_${JCMNAME}   & 
#$runCMD -i ${outputDirComb}/TTToSemiLeptonic2016/picoAOD_3b_wJCM.root -p NONE $YEAR2016MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME} 2>&1 |tee ${outputDir}/log_TTSem2016_wJCM_${JCMNAME}   & 
#$runCMD -i ${outputDirComb}/TTTo2L2Nu2016/picoAOD_3b_wJCM.root        -p NONE $YEAR2016MC --histogramming 10 --histFile hists_3b_wJCM_${JCMNAME}.root --jcmNameLoad ${JCMNAME} 2>&1 |tee ${outputDir}/log_TT2L2Nu2016_wJCM_${JCMNAME}   & 


#hadd -f ${outputDirComb}/TT2018/hists_3b_wJCM_${JCMNAME}.root ${outputDirComb}/TTToHadronic2018/hists_3b_wJCM_${JCMNAME}.root  ${outputDirComb}/TTToSemiLeptonic2018/hists_3b_wJCM_${JCMNAME}.root  ${outputDirComb}/TTTo2L2Nu2018/hists_3b_wJCM_${JCMNAME}.root &

#hadd -f ${outputDirComb}/TT2017/hists_3b_wJCM_${JCMNAME}.root ${outputDirComb}/TTToHadronic2017/hists_3b_wJCM_${JCMNAME}.root  ${outputDirComb}/TTToSemiLeptonic2017/hists_3b_wJCM_${JCMNAME}.root  ${outputDirComb}/TTTo2L2Nu2017/hists_3b_wJCM_${JCMNAME}.root &

#hadd -f ${outputDirComb}/TT2016/hists_3b_wJCM_${JCMNAME}.root ${outputDirComb}/TTToHadronic2016/hists_3b_wJCM_${JCMNAME}.root  ${outputDirComb}/TTToSemiLeptonic2016/hists_3b_wJCM_${JCMNAME}.root  ${outputDirComb}/TTTo2L2Nu2016/hists_3b_wJCM_${JCMNAME}.root &

#for y in 2016
#do 
#    python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py \
#	--d4 ${outputDir}/data${y}/hists_4b.root \
#	--d3 ${outputDirComb}/data${y}/hists_3b_wJCM_Nominal.root \
#	--t4 ${outputDir}/TT${y}/hists_4b.root \
#	--t3 ${outputDirComb}/TT${y}/hists_3b_wJCM_Nominal.root \
#	--t4_s ${outputDir}/TTToSemiLeptonic${y}/hists_4b.root \
#	--t4_h ${outputDir}/TTToHadronic${y}/hists_4b.root \
#	--t4_d ${outputDir}/TTTo2L2Nu${y}/hists_4b.root \
#	--t3_s ${outputDirComb}/TTToSemiLeptonic${y}/hists_3b_wJCM_Nominal.root \
#	--t3_h ${outputDirComb}/TTToHadronic${y}/hists_3b_wJCM_Nominal.root \
#	--t3_d ${outputDirComb}/TTTo2L2Nu${y}/hists_3b_wJCM_Nominal.root \
#	--name closureTests/nominal/CutFlow_${y} \
#	--makePDF
#done


#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputDir}/data2018AllEvents/data18/hists_3bTo4b_noWeights.root  --tt ${outputPath}/${outputDir}/TT2018/hists_noWeights.root -q   ${outputPath}/${outputDir}/qcd2018/hists_noWeights.root

#hadd -f ${outputDir}/TT2018/hists_3b_wJCM_${JCMNAME}.root ${outputDir}/TTToHadronic2018/hists_3b_wJCM_${JCMNAME}.root  ${outputDir}/TTToSemiLeptonic2018/hists_3b_wJCM_${JCMNAME}.root  ${outputDir}/TTTo2L2Nu2018/hists_3b_wJCM_${JCMNAME}.root 
#hadd -f ${outputDirNom}/TT2017/hists_4b.root ${outputDirNom}/TTToHadronic2017/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2017/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2017/hists_4b.root 
#hadd -f ${outputDirNom}/TT2016/hists_4b.root ${outputDirNom}/TTToHadronic2016/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2016/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2016/hists_4b.root 


#done





#
#  OLD
#
#####################################3



#
# Make hists of data with classifiers
#
#$runCMD  -i  ${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.root     -r  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_3bTo4b_wWeights.root   -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  |tee ${outputDir}/log_3bTo4b_wWeights
#$runCMD  -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.root                           -r  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt   |tee ${outputDir}/log_TTTo2L2Nu_wWeights
#$runCMD  -i  ${outputPath}/${outputDir}/fileLists/TTToHadronic2018_wWeights_3bTo4b.txt       -r  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  |tee ${outputDir}/log_TTToHadronic_wWeights
#$runCMD  -i  ${outputPath}/${outputDir}/fileLists/TTToSemiLeptonic2018_wWeights_3bTo4b.txt   -r  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt |tee ${outputDir}/log_TTToSemiLeptonic_wWeights

#hadd -f ${outputPath}/${outputDir}/TT2018/hists_wWeights.root ${outputPath}/${outputDir}/TTToHadronic2018/hists_wWeights.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_wWeights.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_wWeights.root


#
# MAke Plots
#
#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o HemiClosureTest -p plotsRW -l 60.0e3 -y 2018 -m -j -r --data ${outputDir}/data2018AllEvents/data18/hists_3bTo4b_wWeights.root --TT ${outputDir}/TT2018/hists_wWeights.root  --noSignal
#tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plotsRW.tar plotsRW

#
# Make hists of data with classifiers (w/o Reweight)
#
#$runCMD  -i  ${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.root   -p "" -y 2018  -o ${outputDir} --histogramming 10 --histDetail 7 --histFile hists_3bTo4b_noWeights.root   -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  2>&1 |tee ${outputDir}/log_3bTo4b_noWeights &
#$runCMD  -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.root                         -p "" -y 2018  -o ${outputDir} --histogramming 10 --histDetail 7 --histFile hists_noWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt 2>&1 |tee ${outputDir}/log_TTTo2L2Nu_noWeights &
#$runCMD  -i  ${outputPath}/${outputDir}/fileLists/TTToHadronic2018_wWeights_3bTo4b.txt     -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt 2>&1 |tee ${outputDir}/log_TTToHadronic_noWeights &
#$runCMD  -i  ${outputPath}/${outputDir}/fileLists/TTToSemiLeptonic2018_wWeights_3bTo4b.txt -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt 2>&1 |tee ${outputDir}/log_TTToSemiLeptonic_noWeights &

#hadd -f ${outputPath}/${outputDir}/TT2018/hists_noWeights.root ${outputPath}/${outputDir}/TTToHadronic2018_wWeights_3bTo4b/hists_noWeights.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018_wWeights_3bTo4b/hists_noWeights.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_noWeights.root


#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputPath}/${outputDir}/data2018AllEvents/data18/hists_3bTo4b_noWeights.root  --tt ${outputPath}/${outputDir}/TT2018/hists_noWeights.root -q   ${outputPath}/${outputDir}/qcd2018/hists_noWeights.root


#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plotsBeforeRW -l 60.0e3 -y 2018 -m  --data ${outputDir}/data2018AllEvents/data18/hists_3bTo4b_noWeights.root --TT ${outputDir}/TT2018/hists_noWeights.root --qcd  ${outputDir}/qcd2018/hists_noWeights.root  --noSignal
#tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plotsBeforeRW.tar plotsBeforeRW


#
# MAke Plots
#
# python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o HemiClosureTest -p plots -l 60.0e3 -y 2018 -m -j -r

