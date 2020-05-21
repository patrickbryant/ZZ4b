outputDir=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/mixed
outputDirNom=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal

# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'

data2018_noMjj=${outputDirNom}/fileLists/data2018.txt
data2017_noMjj=${outputDirNom}/fileLists/data2017.txt
data2016_noMjj=${outputDirNom}/fileLists/data2016.txt


ttHad2018_noMjj=${outputDirNom}/fileLists/TTToHadronic2018_noMjj.txt
ttSem2018_noMjj=${outputDirNom}/fileLists/TTToSemiLeptonic2018_noMjj.txt
tt2LN2018_noMjj=${outputDirNom}/fileLists/TTTo2L2Nu2018_noMjj.txt

ttHad2017_noMjj=${outputDirNom}/fileLists/TTToHadronic2017_noMjj.txt
ttSem2017_noMjj=${outputDirNom}/fileLists/TTToSemiLeptonic2017_noMjj.txt
tt2LN2017_noMjj=${outputDirNom}/fileLists/TTTo2L2Nu2017_noMjj.txt

ttHad2016_noMjj=${outputDirNom}/fileLists/TTToHadronic2016_noMjj.txt
ttSem2016_noMjj=${outputDirNom}/fileLists/TTToSemiLeptonic2016_noMjj.txt
tt2LN2016_noMjj=${outputDirNom}/fileLists/TTTo2L2Nu2016_noMjj.txt


# 
#  Hists with all 2018 data no weights (Made in the nominal closure test)
#
dataHists2018RAW=${outputDirNom}/data2018/hists.root
dataHists2017RAW=${outputDirNom}/data2017/hists.root
dataHists2016RAW=${outputDirNom}/data2016/hists.root

YEAR2018=' -y 2018 --bTag 0.2770 '
YEAR2017=' -y 2017 --bTag 0.3033 '
YEAR2016=' -y 2016 --bTag 0.3093 '


YEAR2018MC=${YEAR2018}' --bTagSF -l 60.0e3 --isMC '
YEAR2017MC=${YEAR2017}' --bTagSF -l 36.7e3 --isMC '
YEAR2016MC=${YEAR2016}' --bTagSF -l 35.9e3 --isMC '


#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
#$weightCMD -d ${dataHists2018RAW} -c passPreSel   -o ${outputDir}/weights/noTT_data2018_PreSel/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_fitJCM_PS_2018
#$weightCMD -d ${dataHists2017RAW} -c passPreSel   -o ${outputDir}/weights/noTT_data2017_PreSel/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_fitJCM_PS_2017
#$weightCMD -d ${dataHists2016RAW} -c passPreSel   -o ${outputDir}/weights/noTT_data2016_PreSel/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_fitJCM_PS_2016
JCMName2018=${outputDir}/weights/noTT_data2018_PreSel/jetCombinatoricModel_SB_00-00-01.txt
JCMName2017=${outputDir}/weights/noTT_data2017_PreSel/jetCombinatoricModel_SB_00-00-01.txt
JCMName2016=${outputDir}/weights/noTT_data2016_PreSel/jetCombinatoricModel_SB_00-00-01.txt

# 
#  Make the 3b sample with the stats of the 4b sample
#
# In the following "3b" refers to 3b subsampled to have the 4b statistics
#for i in 0 1 2 3 4 5 6
#do
#    $runCMD -i $data2018_noMjj -p picoAOD_data2018_3bSubSampled_v${i}.root  -o ${outputDir} $YEAR2018  --histogramming 10 --histFile hists_v${i}.root   -j ${JCMName2018}  --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD  2>&1 |tee ${outputDir}/log_dataOnlyAll_make3b_2018_v${i} &
#    $runCMD -i $data2017_noMjj -p picoAOD_data2017_3bSubSampled_v${i}.root  -o ${outputDir} $YEAR2017  --histogramming 10 --histFile hists_v${i}.root   -j ${JCMName2017}  --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD  2>&1 |tee ${outputDir}/log_dataOnlyAll_make3b_2017_v${i} &
#    $runCMD -i $data2016_noMjj -p picoAOD_data2016_3bSubSampled_v${i}.root  -o ${outputDir} $YEAR2016  --histogramming 10 --histFile hists_v${i}.root   -j ${JCMName2016}  --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD  2>&1 |tee ${outputDir}/log_dataOnlyAll_make3b_2016_v${i} &
#done




#for i in 0 1 2 3 4 
#do
#    $runCMD -i $ttHad2018_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2018} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TTHad2018_v${i}   & 
#    $runCMD -i $ttSem2018_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2018} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TTSemi2018_v${i}  &
#    $runCMD -i $tt2LN2018_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2018} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TT2L2Nu2018_v${i} &
#
#    $runCMD -i $ttHad2017_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2017} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TTHad2017_v${i}   & 
#    $runCMD -i $ttSem2017_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2017} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TTSemi2017_v${i}  &
#    $runCMD -i $tt2LN2017_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2017} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TT2L2Nu2017_v${i} &
#
#    $runCMD -i $ttHad2016_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2016} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TTHad2016_v${i}   & 
#    $runCMD -i $ttSem2016_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2016} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TTSemi2016_v${i}  &
#    $runCMD -i $tt2LN2016_noMjj -p picoAOD_3bSubSampled_v${i}.root -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists_v${i}.root -j ${JCMName2016} --emulate4bFrom3b --emulationOffset ${i} --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_TT2L2Nu2016_v${i} &
#
#done



##
##  Make hists of the subdampled data
##    #(Optional: for sanity check
#for i in 0 1 2 3 4 5 6
#do
#    $runCMD -i ${outputDir}/data2018/picoAOD_data2018_3bSubSampled_v${i}.root -p "None"  -o ${outputDir} $YEAR2018  --histogramming 10 --histFile hists_3bSubSampled_v${i}.root   --is3bMixed  --writeEventTextFile 2>&1 |tee ${outputDir}/log_subSampledHists_2018_v${i} &
#    $runCMD -i ${outputDir}/data2017/picoAOD_data2017_3bSubSampled_v${i}.root -p "None"  -o ${outputDir} $YEAR2017  --histogramming 10 --histFile hists_3bSubSampled_v${i}.root   --is3bMixed  --writeEventTextFile 2>&1 |tee ${outputDir}/log_subSampledHists_2017_v${i} &
#    $runCMD -i ${outputDir}/data2016/picoAOD_data2016_3bSubSampled_v${i}.root -p "None"  -o ${outputDir} $YEAR2016  --histogramming 10 --histFile hists_3bSubSampled_v${i}.root   --is3bMixed  --writeEventTextFile 2>&1 |tee ${outputDir}/log_subSampledHists_2016_v${i} &
#done



#
# Make Hemisphere library from all hemispheres
#
#$runCMD -i $data2018_noMjj -p "None" $YEAR2018  -o ${outputDir}/dataHemis  --histogramming 1 --histFile hists.root  --createHemisphereLibrary  2>&1 |tee ${outputDir}/log_makeHemisData2018 &
#$runCMD -i $data2017_noMjj -p "None" $YEAR2017  -o ${outputDir}/dataHemis  --histogramming 1 --histFile hists.root  --createHemisphereLibrary  2>&1 |tee ${outputDir}/log_makeHemisData2017 &
#$runCMD -i $data2016_noMjj -p "None" $YEAR2016  -o ${outputDir}/dataHemis  --histogramming 1 --histFile hists.root  --createHemisphereLibrary  2>&1 |tee ${outputDir}/log_makeHemisData2016 &


#
# OLD
####################################


#
#  Make 3b subsampled ttbar
#
#$runCMD $runJOB  -i ${fileTTToSemiLeptonic_noMjj} -p picoAOD_3bUnmixed.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_3bUnmixed.root   --bTagSF -l 60.0e3 --isMC --emulate4bFrom3b --noDiJetMassCutInPicoAOD -j ${JCMName}  2>&1 |tee ${outputDir}/log_TTToSemiLeptonic2018_3bUnmixed  &
#$runCMD $runJOB  -i ${fileTTToHadronic_noMjj}     -p picoAOD_3bUnmixed.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_3bUnmixed.root   --bTagSF -l 60.0e3 --isMC --emulate4bFrom3b --noDiJetMassCutInPicoAOD -j ${JCMName} 2>&1 |tee ${outputDir}/log_TTToHadronic2018_3bUnmixed  &
#$runCMD $runJOB  -i ${fileTTTo2L2Nu_noMjj}        -p picoAOD_3bUnmixed.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_3bUnmixed.root   --bTagSF -l 60.0e3 --isMC --emulate4bFrom3b --noDiJetMassCutInPicoAOD -j ${JCMName} 2>&1 |tee ${outputDir}/log_TTTo2L2Nu2018_3bUnmixed  &


#hadd -f ${outputPath}/${outputDir}/TT2018/hists_3bUnmixed.root ${outputPath}/${outputDir}/TTToHadronic2018_noMjj/hists_3bUnmixed.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj/hists_3bUnmixed.root ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj/hists_3bUnmixed.root


