outputDir=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/combined
outputDirNom=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal
outputDir3bMix4b=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
convertToH5JOB='python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'
SvBModel=ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1692_lr0.008_epochs40_stdscale_epoch40_loss0.2070.pkl
trainJOB=ZZ4b/nTupleAnalysis/scripts/multiClassifier.py
convertToROOTJOB='python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py'

YEAR2018=' -y 2018 --bTag 0.2770 '
YEAR2017=' -y 2017 --bTag 0.3033 '
YEAR2016=' -y 2016 --bTag 0.3093 '

YEAR2018MC=${YEAR2018}' --bTagSF -l 60.0e3 --isMC '
YEAR2017MC=${YEAR2017}' --bTagSF -l 36.7e3 --isMC '
YEAR2016MC=${YEAR2016}' --bTagSF -l 35.9e3 --isMC '


jcmNameList=Nominal,3bMix4b_v0,3bMix4b_v1,3bMix4b_v2,3bMix4b_v3,3bMix4b_v4

fileJCM18_Nom=${outputDirNom}/weights/data2017/jetCombinatoricModel_SB_00-00-01.txt
fileJCM18_3bMix4b_v0=${outputDir3bMix4b}/weights/data2018_3bMix4b_v0/jetCombinatoricModel_SB_00-00-01.txt
fileJCM18_3bMix4b_v1=${outputDir3bMix4b}/weights/data2018_3bMix4b_v1/jetCombinatoricModel_SB_00-00-01.txt
fileJCM18_3bMix4b_v2=${outputDir3bMix4b}/weights/data2018_3bMix4b_v2/jetCombinatoricModel_SB_00-00-01.txt
fileJCM18_3bMix4b_v3=${outputDir3bMix4b}/weights/data2018_3bMix4b_v3/jetCombinatoricModel_SB_00-00-01.txt
fileJCM18_3bMix4b_v4=${outputDir3bMix4b}/weights/data2018_3bMix4b_v4/jetCombinatoricModel_SB_00-00-01.txt
jcmFileList18=${fileJCM18_Nom},${fileJCM18_3bMix4b_v0},${fileJCM18_3bMix4b_v1},${fileJCM18_3bMix4b_v2},${fileJCM18_3bMix4b_v3},${fileJCM18_3bMix4b_v4}

fileJCM17_Nom=${outputDirNom}/weights/data2017/jetCombinatoricModel_SB_00-00-01.txt
fileJCM17_3bMix4b_v0=${outputDir3bMix4b}/weights/data2017_3bMix4b_v0/jetCombinatoricModel_SB_00-00-01.txt
fileJCM17_3bMix4b_v1=${outputDir3bMix4b}/weights/data2017_3bMix4b_v1/jetCombinatoricModel_SB_00-00-01.txt
fileJCM17_3bMix4b_v2=${outputDir3bMix4b}/weights/data2017_3bMix4b_v2/jetCombinatoricModel_SB_00-00-01.txt
fileJCM17_3bMix4b_v3=${outputDir3bMix4b}/weights/data2017_3bMix4b_v3/jetCombinatoricModel_SB_00-00-01.txt
fileJCM17_3bMix4b_v4=${outputDir3bMix4b}/weights/data2017_3bMix4b_v4/jetCombinatoricModel_SB_00-00-01.txt
jcmFileList17=${fileJCM17_Nom},${fileJCM17_3bMix4b_v0},${fileJCM17_3bMix4b_v1},${fileJCM17_3bMix4b_v2},${fileJCM17_3bMix4b_v3},${fileJCM17_3bMix4b_v4}

fileJCM16_Nom=${outputDirNom}/weights/data2016/jetCombinatoricModel_SB_00-00-01.txt
fileJCM16_3bMix4b_v0=${outputDir3bMix4b}/weights/data2016_3bMix4b_v0/jetCombinatoricModel_SB_00-00-01.txt
fileJCM16_3bMix4b_v1=${outputDir3bMix4b}/weights/data2016_3bMix4b_v1/jetCombinatoricModel_SB_00-00-01.txt
fileJCM16_3bMix4b_v2=${outputDir3bMix4b}/weights/data2016_3bMix4b_v2/jetCombinatoricModel_SB_00-00-01.txt
fileJCM16_3bMix4b_v3=${outputDir3bMix4b}/weights/data2016_3bMix4b_v3/jetCombinatoricModel_SB_00-00-01.txt
fileJCM16_3bMix4b_v4=${outputDir3bMix4b}/weights/data2016_3bMix4b_v4/jetCombinatoricModel_SB_00-00-01.txt
jcmFileList16=${fileJCM16_Nom},${fileJCM16_3bMix4b_v0},${fileJCM16_3bMix4b_v1},${fileJCM16_3bMix4b_v2},${fileJCM16_3bMix4b_v3},${fileJCM16_3bMix4b_v4}

##
## Make picoAODS of 3b data with weights applied  (for closure test)
##
#$runCMD  -i ${outputDirNom}/data2018/picoAOD.root -p picoAOD_3b_wJCM.root $YEAR2018 --histogramming 10 --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --skip4b 2>&1|tee ${outputDir}/log_2018_JCM  &
#$runCMD  -i ${outputDirNom}/data2017/picoAOD.root -p picoAOD_3b_wJCM.root $YEAR2017 --histogramming 10 --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --skip4b 2>&1|tee ${outputDir}/log_2017_JCM  &
#$runCMD  -i ${outputDirNom}/data2016/picoAOD.root -p picoAOD_3b_wJCM.root $YEAR2016 --histogramming 10 --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --skip4b 2>&1|tee ${outputDir}/log_2016_JCM  &
#
#
### 2018
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2018.txt     -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2018MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --skip4b 2>&1 |tee ${outputDir}/log_TTHad2018_JCM   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2018.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2018MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --skip4b 2>&1 |tee ${outputDir}/log_TTSemi2018_JCM  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2018.txt        -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2018MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --skip4b 2>&1 |tee ${outputDir}/log_TT2L2Nu2018_JCM &
#
##2017
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2017.txt     -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2017MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --skip4b 2>&1 |tee ${outputDir}/log_TTHad2017_JCM   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2017.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2017MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --skip4b 2>&1 |tee ${outputDir}/log_TTSemi2017_JCM  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2017.txt        -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2017MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --skip4b 2>&1 |tee ${outputDir}/log_TT2L2Nu2017_JCM &
#
##2016
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2016.txt     -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2016MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --skip4b 2>&1 |tee ${outputDir}/log_TTHad2016_JCM   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2016.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2016MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --skip4b 2>&1 |tee ${outputDir}/log_TTSemi2016_JCM  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2016.txt        -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2016MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --skip4b 2>&1 |tee ${outputDir}/log_TT2L2Nu2016_JCM &




##
## skim to only have 4b events in the pico ADO (Needed for training) 
##
#$runCMD  -i root://cmseos.fnal.gov//store/user/johnda/closureTest/nominal/data2018/picoAOD.root -p picoAOD_4b.root   $YEAR2018  --histogramming 10 --histFile hists_4b.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 2>&1|tee ${outputDir}/log_2018_4b  &
#$runCMD  -i root://cmseos.fnal.gov//store/user/johnda/closureTest/nominal/data2017/picoAOD.root -p picoAOD_4b.root   $YEAR2017  --histogramming 10 --histFile hists_4b.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 2>&1|tee ${outputDir}/log_2017_4b  &
#$runCMD  -i root://cmseos.fnal.gov//store/user/johnda/closureTest/nominal/data2016/picoAOD.root -p picoAOD_4b.root   $YEAR2016  --histogramming 10 --histFile hists_4b.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 2>&1|tee ${outputDir}/log_2016_4b  &
#
#
#
### 2018
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2018.txt     -o ${outputDirNom} -p picoAOD_4b.root $YEAR2018MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18  --skip3b 2>&1 |tee ${outputDir}/log_TTHad2018_4b   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2018.txt -o ${outputDirNom} -p picoAOD_4b.root $YEAR2018MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18  --skip3b 2>&1 |tee ${outputDir}/log_TTSemi2018_4b  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2018.txt        -o ${outputDirNom} -p picoAOD_4b.root $YEAR2018MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18  --skip3b 2>&1 |tee ${outputDir}/log_TT2L2Nu2018_4b &
#
##2017
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2017.txt     -o ${outputDirNom} -p picoAOD_4b.root $YEAR2017MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17  --skip3b 2>&1 |tee ${outputDir}/log_TTHad2017_4b   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2017.txt -o ${outputDirNom} -p picoAOD_4b.root $YEAR2017MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17  --skip3b 2>&1 |tee ${outputDir}/log_TTSemi2017_4b  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2017.txt        -o ${outputDirNom} -p picoAOD_4b.root $YEAR2017MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17  --skip3b 2>&1 |tee ${outputDir}/log_TT2L2Nu2017_4b &
#
##2016
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2016.txt     -o ${outputDirNom} -p picoAOD_4b.root $YEAR2016MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16  --skip3b 2>&1 |tee ${outputDir}/log_TTHad2016_4b   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2016.txt -o ${outputDirNom} -p picoAOD_4b.root $YEAR2016MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16  --skip3b 2>&1 |tee ${outputDir}/log_TTSemi2016_4b  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2016.txt        -o ${outputDirNom} -p picoAOD_4b.root $YEAR2016MC --histogramming 10  --histFile hists_4b.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16  --skip3b 2>&1 |tee ${outputDir}/log_TT2L2Nu2016_4b &

##    $convertToH5JOB -i "${outputDir3bMix4b}/data2018/picoAOD_3bMix4b_noTTVeto_v${i}.root"             --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_data2018_v${i} &
#for i in 0 1 2 3 4 
#do
#    $runCMD  -i ${outputDir3bMix4b}/data2018/picoAOD_3bMix4b_noTTVeto_v${i}.root -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2018    --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --is3bMixed 2>&1|tee ${outputDir}/log_2018_3bMix4b_noTtVeto_4b_v${i}  &
#    $runCMD -i ${outputDir3bMix4b}/TTToHadronic2018/picoAOD_3bMix4b_v${i}.root       -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2018MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --is3bMixed 2>&1 |tee ${outputDir}/log_TTHad2018_4b_v${i}   & 
#    $runCMD -i ${outputDir3bMix4b}/TTToSemiLeptonic2018/picoAOD_3bMix4b_v${i}.root   -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2018MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --is3bMixed 2>&1 |tee ${outputDir}/log_TTSemi2018_4b_v${i}   & 
#    $runCMD -i ${outputDir3bMix4b}/TTTo2L2Nu2018/picoAOD_3bMix4b_v${i}.root          -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2018MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --is3bMixed 2>&1 |tee ${outputDir}/log_TT2L2N2018_4b_v${i}   & 

#    $runCMD  -i ${outputDir3bMix4b}/data2017/picoAOD_3bMix4b_noTTVeto_v${i}.root -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2017    --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --is3bMixed 2>&1|tee ${outputDir}/log_2017_3bMix4b_noTtVeto_4b_v${i}  &
#    $runCMD -i ${outputDir3bMix4b}/TTToHadronic2017/picoAOD_3bMix4b_v${i}.root       -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2017MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --is3bMixed 2>&1 |tee ${outputDir}/log_TTHad2017_4b_v${i}   & 
#    $runCMD -i ${outputDir3bMix4b}/TTToSemiLeptonic2017/picoAOD_3bMix4b_v${i}.root   -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2017MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --is3bMixed 2>&1 |tee ${outputDir}/log_TTSemi2017_4b_v${i}   & 
#    $runCMD -i ${outputDir3bMix4b}/TTTo2L2Nu2017/picoAOD_3bMix4b_v${i}.root          -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2017MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --is3bMixed 2>&1 |tee ${outputDir}/log_TT2L2N2017_4b_v${i}   & 

#    $runCMD  -i ${outputDir3bMix4b}/data2016/picoAOD_3bMix4b_noTTVeto_v${i}.root -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2016    --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --is3bMixed 2>&1|tee ${outputDir}/log_2016_3bMix4b_noTtVeto_4b_v${i}  &
#    $runCMD -i ${outputDir3bMix4b}/TTToHadronic2016/picoAOD_3bMix4b_v${i}.root       -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2016MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --is3bMixed 2>&1 |tee ${outputDir}/log_TTHad2016_4b_v${i}   & 
#    $runCMD -i ${outputDir3bMix4b}/TTToSemiLeptonic2016/picoAOD_3bMix4b_v${i}.root   -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2016MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --is3bMixed 2>&1 |tee ${outputDir}/log_TTSemi2016_4b_v${i}   & 
#    $runCMD -i ${outputDir3bMix4b}/TTTo2L2Nu2016/picoAOD_3bMix4b_v${i}.root          -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2016MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --is3bMixed 2>&1 |tee ${outputDir}/log_TT2L2N2016_4b_v${i}   & 
#done



#
# Convert root to hdf5
#   (with conversion enviorment)
#  "4b Data"
#
#$convertToH5JOB -i "${outputDirNom}/data2018/picoAOD_4b.root"               --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_data2018 &         
#$convertToH5JOB -i "${outputDirNom}/TTTo2L2Nu2018/picoAOD_4b.root"          --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTTo2L2Nu2018 &         
#$convertToH5JOB -i "${outputDirNom}/TTToHadronic2018/picoAOD_4b.root"       --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTTo2Had2018 &         
#$convertToH5JOB -i "${outputDirNom}/TTToSemiLeptonic2018/picoAOD_4b.root"   --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTToSemi2018 &         
#
#$convertToH5JOB -i "${outputDirNom}/data2017/picoAOD_4b.root"               --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_data2017 &         
#$convertToH5JOB -i "${outputDirNom}/TTTo2L2Nu2017/picoAOD_4b.root"          --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTTo2L2Nu2017 &         
#$convertToH5JOB -i "${outputDirNom}/TTToHadronic2017/picoAOD_4b.root"       --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTTo2Had2017 &         
#$convertToH5JOB -i "${outputDirNom}/TTToSemiLeptonic2017/picoAOD_4b.root"   --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTToSemi2017 &         
#
#$convertToH5JOB -i "${outputDirNom}/data2016/picoAOD_4b.root"               --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_data2016 &         
#$convertToH5JOB -i "${outputDirNom}/TTTo2L2Nu2016/picoAOD_4b.root"          --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTTo2L2Nu2016 &         
#$convertToH5JOB -i "${outputDirNom}/TTToHadronic2016/picoAOD_4b.root"       --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTTo2Had2016 &         
#$convertToH5JOB -i "${outputDirNom}/TTToSemiLeptonic2016/picoAOD_4b.root"   --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_4b_TTToSemi2016 &         

for i in 0 1 2 3 4 
do
#    $convertToH5JOB -i "${outputDir3bMix4b}/data2018/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"             --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_data2018_v${i} &
#    $convertToH5JOB -i "${outputDir3bMix4b}/TTToHadronic2018/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"     --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTToHad2018_v${i} &
#    $convertToH5JOB -i "${outputDir3bMix4b}/TTToSemiLeptonic2018/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root" --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTToSemi2018_v${i} &
#    $convertToH5JOB -i "${outputDir3bMix4b}/TTTo2L2Nu2018/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"        --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTTo2L2N2018_v${i} &

#    $convertToH5JOB -i "${outputDir3bMix4b}/data2017/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"             --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_data2017_v${i} &
#    $convertToH5JOB -i "${outputDir3bMix4b}/TTToHadronic2017/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"     --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTToHad2017_v${i} &
#    $convertToH5JOB -i "${outputDir3bMix4b}/TTToSemiLeptonic2017/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root" --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTToSemi2017_v${i} &
#    $convertToH5JOB -i "${outputDir3bMix4b}/TTTo2L2Nu2017/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"        --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTTo2L2N2017_v${i} &

#    $convertToH5JOB -i "${outputDir3bMix4b}/data2016/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"             --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_data2016_v${i} &
    $convertToH5JOB -i "${outputDir3bMix4b}/TTToHadronic2016/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"     --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTToHad2016_v${i} &
    $convertToH5JOB -i "${outputDir3bMix4b}/TTToSemiLeptonic2016/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root" --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTToSemi2016_v${i} &
    $convertToH5JOB -i "${outputDir3bMix4b}/TTTo2L2Nu2016/picoAOD_3bMix4b_noTTVeto_4b_v${i}.root"       --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3bmix4b_TTTo2L2N2016_v${i} &
done


#
#  3b with JCM weights
# 
#$convertToH5JOB -i "${outputDirNom}/data2018/picoAOD_3b_wJCM.root"            --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_data2018 &         
#$convertToH5JOB -i "${outputDir}/TTTo2L2Nu2018/picoAOD_3b_wJCM.root"          --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTTo2L2Nu2018 &         
#$convertToH5JOB -i "${outputDir}/TTToHadronic2018/picoAOD_3b_wJCM.root"       --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTTo2Had2018 &         
#$convertToH5JOB -i "${outputDir}/TTToSemiLeptonic2018/picoAOD_3b_wJCM.root"   --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTToSemi2018 &         
#
#$convertToH5JOB -i "${outputDirNom}/data2017/picoAOD_3b_wJCM.root"            --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_data2017 &         
#$convertToH5JOB -i "${outputDir}/TTTo2L2Nu2017/picoAOD_3b_wJCM.root"          --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTTo2L2Nu2017 &         
#$convertToH5JOB -i "${outputDir}/TTToHadronic2017/picoAOD_3b_wJCM.root"       --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTTo2Had2017 &         
#$convertToH5JOB -i "${outputDir}/TTToSemiLeptonic2017/picoAOD_3b_wJCM.root"   --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTToSemi2017 &         
#
#$convertToH5JOB -i "${outputDirNom}/data2016/picoAOD_3b_wJCM.root"            --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_data2016 &         
#$convertToH5JOB -i "${outputDir}/TTTo2L2Nu2016/picoAOD_3b_wJCM.root"          --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTTo2L2Nu2016 &         
#$convertToH5JOB -i "${outputDir}/TTToHadronic2016/picoAOD_3b_wJCM.root"       --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTTo2Had2016 &         
#$convertToH5JOB -i "${outputDir}/TTToSemiLeptonic2016/picoAOD_3b_wJCM.root"   --jcmNameList $jcmNameList 2>&1 |tee ${outputDir}/log_Convert_ROOTh5_3b_TTToSemi2016 &         
#







