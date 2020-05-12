outputDir=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/mixed
outputDirNom=/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal

# 3b Unmixed
name3bSubSampled=data2018_3bSubSampled
pico3bSubSampled=picoAOD_${name3bSubSampled}.root

# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'

data2018_noMjj=${outputDirNom}/fileLists/data2018.txt

fileTTToHadronic=${outputPath}/${outputDirNom}/fileLists/TTToHadronic2018.txt
fileTTToSemiLeptonic=${outputPath}/${outputDirNom}/fileLists/TTToSemiLeptonic2018.txt
fileTTTo2L2Nu=${outputPath}/${outputDirNom}/fileLists/TTTo2L2Nu2018.txt


fileTTToHadronic_noMjj=${outputPath}/${outputDir}/fileLists/TTToHadronic2018_noMjj.txt
fileTTToSemiLeptonic_noMjj=${outputPath}/${outputDir}/fileLists/TTToSemiLeptonic2018_noMjj.txt
fileTTTo2L2Nu_noMjj=${outputPath}/${outputDir}/fileLists/TTTo2L2Nu2018_noMjj.txt


# 
#  Hists with all 2018 data no weights (Made in the nominal closure test)
#
dataHistsRAW=${outputDirNom}/data2018/hists.root

#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
#$weightCMD -d ${dataHistsRAW} -c passPreSel   -o ${outputDir}/weights/noTT_data2018_PreSel/  -r SB -w 00-00-01 2>&1 |tee ${outputDir}/log_fitJCM_PS_2018
JCMName2018=${outputDir}/weights/noTT_data2018_PreSel/jetCombinatoricModel_SB_00-00-01.txt

# 
#  Make the 3b sample with the stats of the 4b sample
#
# In the following "3b" refers to 3b subsampled to have the 4b statistics
#$runCMD -i $data2018_noMjj -p ${pico3bSubSampled}  -o ${outputDir} -y 2018  --histogramming 10 --histFile hists.root  --nevents -1  -j ${JCMName2018}  --emulate4bFrom3b --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_dataOnlyAll_make3b_2018
data2018_3bSubSampled=${outputDir}/fileLists/data2018_3bSubSampled.txt



#
#  Make hists of the subdampled data
#    #(Optional: for sanity check
#$runCMD -i $data2018_3bSubSampled -p "None"  -o ${outputDir} -y 2018  --histogramming 10 --histFile hists.root  --nevents -1  --is3bMixed  2>&1 |tee ${outputDir}/logMake_subSampledHists_2018


#
# Make Hemisphere library
#
#$runCMD -i $data2018_noMjj -p picoAOD.root  -o ${outputDir}/dataHemis -y 2018  --histogramming 1 --histFile hists.root  --nevents -1 --createHemisphereLibrary  2>&1 |tee ${outputDir}/log_makeHemisData2018 



#
#  Make skimmed ttbar 
#
#$runCMD $runJOB  -i ZZ4b/fileLists/TTToHadronic2018.txt     -o ${outputPath}/${outputDir} -y 2018 --histogramming 10  --histFile hists_No4b.root  --nevents -1 --bTagSF -l 60.0e3 --isMC --skip4b --fastSkim --noDiJetMassCutInPicoAOD -p picoAOD_NoMjj_No4b.root  2>&1 |tee ${outputDir}/log_TTToHadronic &	  
#$runCMD $runJOB  -i ZZ4b/fileLists/TTToSemiLeptonic2018.txt -o ${outputPath}/${outputDir} -y 2018 --histogramming 10  --histFile hists_No4b.root  --nevents -1 --bTagSF -l 60.0e3 --isMC --skip4b --fastSkim --noDiJetMassCutInPicoAOD -p picoAOD_NoMjj_No4b.root  2>&1 |tee ${outputDir}/log_TTToSemiLeptonic &	  
#$runCMD $runJOB  -i ZZ4b/fileLists/TTTo2L2Nu2018.txt        -o ${outputPath}/${outputDir} -y 2018 --histogramming 10  --histFile hists_No4b.root  --nevents -1 --bTagSF -l 60.0e3 --isMC --skip4b --fastSkim --noDiJetMassCutInPicoAOD -p picoAOD_NoMjj_No4b.root  2>&1 |tee ${outputDir}/log_TTTo2L2Nu &


#
#  Make "4b" ttbar from 3b and JCM
#
#$runCMD $runJOB  -i ${fileTTToSemiLeptonic_noMjj} -p picoAOD_3bUnmixed.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_3bUnmixed.root  --nevents -1  --bTagSF -l 60.0e3 --isMC --emulate4bFrom3b --noDiJetMassCutInPicoAOD -j ${JCMName}  2>&1 |tee ${outputDir}/log_TTToSemiLeptonic2018_3bUnmixed  &
#$runCMD $runJOB  -i ${fileTTToHadronic_noMjj}     -p picoAOD_3bUnmixed.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_3bUnmixed.root  --nevents -1  --bTagSF -l 60.0e3 --isMC --emulate4bFrom3b --noDiJetMassCutInPicoAOD -j ${JCMName} 2>&1 |tee ${outputDir}/log_TTToHadronic2018_3bUnmixed  &
#$runCMD $runJOB  -i ${fileTTTo2L2Nu_noMjj}        -p picoAOD_3bUnmixed.root -o ${outputPath}/${outputDir}  -y 2018 --histogramming 10  --histFile hists_3bUnmixed.root  --nevents -1  --bTagSF -l 60.0e3 --isMC --emulate4bFrom3b --noDiJetMassCutInPicoAOD -j ${JCMName} 2>&1 |tee ${outputDir}/log_TTTo2L2Nu2018_3bUnmixed  &


#hadd -f ${outputPath}/${outputDir}/TT2018/hists_3bUnmixed.root ${outputPath}/${outputDir}/TTToHadronic2018_noMjj/hists_3bUnmixed.root ${outputPath}/${outputDir}/TTToSemiLeptonic2018_noMjj/hists_3bUnmixed.root ${outputPath}/${outputDir}/TTTo2L2Nu2018_noMjj/hists_3bUnmixed.root


