#
# In the following "3b" refers to 3b subsampled to have the 4b statistics
#
outputDir=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightJOB=ZZ4b/nTupleAnalysis/scripts/makeWeights.py
convertToH5JOB=ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py
SvBModel=ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1692_lr0.008_epochs40_stdscale_epoch40_loss0.2070.pkl
trainJOB=ZZ4b/nTupleAnalysis/scripts/multiClassifier.py
convertToROOTJOB=ZZ4b/nTupleAnalysis/scripts/convert_h52root.py

YEAR2018=' -y 2018 --bTag 0.2770 '
YEAR2017=' -y 2017 --bTag 0.3033 '
YEAR2016=' -y 2016 --bTag 0.3093 '


# 
#  Make Hists with all data
#
#$runCMD  -i ${outputDir}/fileLists/data2018.txt -p picoAOD.root  -o ${outputDir} $YEAR2018  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2018 &
#$runCMD  -i ${outputDir}/fileLists/data2017.txt -p picoAOD.root  -o ${outputDir} $YEAR2017  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2017 &
#$runCMD  -i ${outputDir}/fileLists/data2016.txt -p picoAOD.root  -o ${outputDir} $YEAR2016  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2016 &




#
#  OLD
#
#####################################3

#
#  Make Hists with all 2018 ttbar
#
#$runCMD $runJOB  -i ZZ4b/fileLists/TTToHadronic2018.txt -o ${outputPath}/${outputDir} -y 2018 --histogramming 10  --histFile hists.root  --bTagSF -l 60.0e3 --isMC   &
#$runCMD $runJOB  -i ZZ4b/fileLists/TTToSemiLeptonic2018.txt -o ${outputPath}/${outputDir} -y 2018 --histogramming 10  --histFile hists.root  --bTagSF -l 60.0e3 --isMC  & 
#$runCMD $runJOB  -i ZZ4b/fileLists/TTTo2L2Nu2018.txt -o ${outputPath}/${outputDir} -y 2018 --histogramming 10  --histFile hists.root  --bTagSF -l 60.0e3 --isMC  &


#
#  Hadd ttbar
#
#mkdir ${outputPath}/${outputDir}/TT2018
#hadd -f ${outputPath}/${outputDir}/TT2018/hists.root ${outputPath}/${outputDir}/TTToHadronic2018/hists.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists.root


#
#  Make the JCM-weights
#
# (3b -> 4b)
#python $weightJOB -d ${outputPath}/${outputDir}/data2018AllEvents/data18/hists.root  --tt ${outputPath}/${outputDir}/TT2018/hists.root -c passXWt   -o ${outputPath}/${outputDir}/weights/data2018/  -r SB -w 00-00-01

#
# Make picoAODS of 3b data with weights applied  (for closure test)
#
#$runCMD $runJOB -i ${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD.root -p picoAOD_JCM_3bTo4b.root   -y 2018  --histogramming 10 --histFile hists_JCM_3bTo4b.root   -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  |tee ${outputDir}/log_3bTo4b_JCM
#$runCMD $runJOB  -i ${outputPath}/${outputDir}/TTToHadronic2018/picoAOD.root     -o ${outputPath}/${outputDir} -p picoAOD_wJCM.root -y 2018 --histogramming 10  --histFile hists_wJCM.root  --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  |tee ${outputDir}/log_TTToHadronic2018_JCM
#$runCMD $runJOB  -i ${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD.root -o ${outputPath}/${outputDir} -p picoAOD_wJCM.root -y 2018 --histogramming 10  --histFile hists_wJCM.root  --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  |tee ${outputDir}/log_TTToSemiLeptonic2018_JCM
#$runCMD $runJOB  -i ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD.root        -o ${outputPath}/${outputDir} -p picoAOD_wJCM.root -y 2018 --histogramming 10  --histFile hists_wJCM.root  --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  |tee ${outputDir}/log_TTTo2L2Nu2018_JCM

#hadd -f ${outputPath}/${outputDir}/TT2018/hists_wJCM.root ${outputPath}/${outputDir}/TTToHadronic2018/hists_wJCM.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_wJCM.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_wJCM.root


#
# Make QCD with out the reweights
#
#mkdir -p ${outputPath}/${outputDir}/qcd2018
#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputPath}/${outputDir}/data2018AllEvents/data18/hists_JCM_3bTo4b.root --tt ${outputPath}/${outputDir}/TT2018/hists_wJCM.root -q   ${outputPath}/${outputDir}/qcd2018/hists_wJCM.root

#
# Make Plots
#
#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o HemiClosureTest -p plotsNoReWeight -l 60.0e3 -y 2018 -m
#tar -C HemiClosureTest -zcf ${outputPath}/${outputDir}/plotsNoReWeight.tar plotsNoReWeight


#
# Convert root to hdf5
#   (with conversion enviorment)
#python $convertToH5JOB -i "${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.root"
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.root"
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.root"
#python $convertToH5JOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.root"

#
# Train
#   (with GPU conversion enviorment)
#python $trainJOB -c FvT -d "${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.h5" -t "${outputPath}/${outputDir}/TTTo*2018*/picoAOD_wJCM.h5" -e 40 -o 3bTo4b 2>&1 |tee log ${outputDir}/log_Train_FvT_3bTo4b

#
# Add FvT
#
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.h5"      -m ZZ4b/nTupleAnalysis/pytorchModels/3bTo4bFvT_ResNet+multijetAttention_9_9_9_np1904_lr0.008_epochs40_stdscale_epoch24_loss0.2405.pkl        -c FvT  2>&1 |tee log ${outputDir}/log_Add_FvT_3bTo4b       
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.h5"      -m ZZ4b/nTupleAnalysis/pytorchModels/3bTo4bFvT_ResNet+multijetAttention_9_9_9_np1904_lr0.008_epochs40_stdscale_epoch24_loss0.2405.pkl        -c FvT  2>&1 |tee log ${outputDir}/log_Add_FvT_3bTo4b_TTTo2L2Nu       
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.h5"      -m ZZ4b/nTupleAnalysis/pytorchModels/3bTo4bFvT_ResNet+multijetAttention_9_9_9_np1904_lr0.008_epochs40_stdscale_epoch24_loss0.2405.pkl        -c FvT  2>&1 |tee log ${outputDir}/log_Add_FvT_3bTo4b_TTToHadronicc
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.h5"      -m ZZ4b/nTupleAnalysis/pytorchModels/3bTo4bFvT_ResNet+multijetAttention_9_9_9_np1904_lr0.008_epochs40_stdscale_epoch24_loss0.2405.pkl        -c FvT  2>&1 |tee log ${outputDir}/log_Add_FvT_3bTo4b_TTToSemiLeptonic

#
# Add SvB
#
#python $trainJOB   -u  -d "${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.h5"      -m ${SvBModel}   -c SvB 2>&1 |tee log ${outputDir}/log_Add_SvB_3bTo4b
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.h5"         -m ${SvBModel}        -c SvB  2>&1 |tee log ${outputDir}/log_Add_SvB_3bTo4b_TTTo2L2Nu       
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.h5"      -m ${SvBModel}        -c SvB  2>&1 |tee log ${outputDir}/log_Add_SvB_3bTo4b_TTToHadronicc
#python $trainJOB   -u  -t "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.h5"  -m ${SvBModel}        -c SvB  2>&1 |tee log ${outputDir}/log_Add_SvB_3bTo4b_TTToSemiLeptonic

#
# Convert hdf5 to root
#   (with conversion enviorment)
#python $convertToROOTJOB -i "${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.h5"      
#python $convertToROOTJOB -i "${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.h5"
#python $convertToROOTJOB -i "${outputPath}/${outputDir}/TTToHadronic2018/picoAOD_wJCM.h5"
#python $convertToROOTJOB -i "${outputPath}/${outputDir}/TTToSemiLeptonic2018/picoAOD_wJCM.h5"



#
# Make hists of data with classifiers
#
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.root     -r  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_3bTo4b_wWeights.root   -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  |tee ${outputDir}/log_3bTo4b_wWeights
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.root                           -r  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt   |tee ${outputDir}/log_TTTo2L2Nu_wWeights
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/fileLists/TTToHadronic2018_wWeights_3bTo4b.txt       -r  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  |tee ${outputDir}/log_TTToHadronic_wWeights
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/fileLists/TTToSemiLeptonic2018_wWeights_3bTo4b.txt   -r  -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_wWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt |tee ${outputDir}/log_TTToSemiLeptonic_wWeights

#hadd -f ${outputPath}/${outputDir}/TT2018/hists_wWeights.root ${outputPath}/${outputDir}/TTToHadronic2018/hists_wWeights.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018/hists_wWeights.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_wWeights.root


#
# MAke Plots
#
#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o HemiClosureTest -p plotsRW -l 60.0e3 -y 2018 -m -j -r --data ${outputDir}/data2018AllEvents/data18/hists_3bTo4b_wWeights.root --TT ${outputDir}/TT2018/hists_wWeights.root  --noSignal
#tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plotsRW.tar plotsRW

#
# Make hists of data with classifiers (w/o Reweight)
#
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/data2018AllEvents/data18/picoAOD_JCM_3bTo4b.root   -p "" -y 2018  -o ${outputDir} --histogramming 10 --histDetail 7 --histFile hists_3bTo4b_noWeights.root   -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt  2>&1 |tee ${outputDir}/log_3bTo4b_noWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/TTTo2L2Nu2018/picoAOD_wJCM.root                         -p "" -y 2018  -o ${outputDir} --histogramming 10 --histDetail 7 --histFile hists_noWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt 2>&1 |tee ${outputDir}/log_TTTo2L2Nu_noWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/fileLists/TTToHadronic2018_wWeights_3bTo4b.txt     -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt 2>&1 |tee ${outputDir}/log_TTToHadronic_noWeights &
#$runCMD $runJOB -i  ${outputPath}/${outputDir}/fileLists/TTToSemiLeptonic2018_wWeights_3bTo4b.txt -p "" -y 2018 -o ${outputDir}  --histogramming 10 --histDetail 7 --histFile hists_noWeights.root    --bTagSF -l 60.0e3 --isMC  -j ${outputPath}/${outputDir}/weights/data2018/jetCombinatoricModel_SB_00-00-01.txt 2>&1 |tee ${outputDir}/log_TTToSemiLeptonic_noWeights &

#hadd -f ${outputPath}/${outputDir}/TT2018/hists_noWeights.root ${outputPath}/${outputDir}/TTToHadronic2018_wWeights_3bTo4b/hists_noWeights.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018_wWeights_3bTo4b/hists_noWeights.root ${outputPath}/${outputDir}/TTTo2L2Nu2018/hists_noWeights.root


#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputPath}/${outputDir}/data2018AllEvents/data18/hists_3bTo4b_noWeights.root  --tt ${outputPath}/${outputDir}/TT2018/hists_noWeights.root -q   ${outputPath}/${outputDir}/qcd2018/hists_noWeights.root


#python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o ${outputDir} -p plotsBeforeRW -l 60.0e3 -y 2018 -m  --data ${outputDir}/data2018AllEvents/data18/hists_3bTo4b_noWeights.root --TT ${outputDir}/TT2018/hists_noWeights.root --qcd  ${outputDir}/qcd2018/hists_noWeights.root  --noSignal
#tar -C ${outputDir} -zcf ${outputPath}/${outputDir}/plotsBeforeRW.tar plotsBeforeRW


#
# MAke Plots
#
# python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o HemiClosureTest -p plots -l 60.0e3 -y 2018 -m -j -r

