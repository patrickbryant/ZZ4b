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
# Make skims with out the di-jet Mass cuts
#

# 2018
#$runCMD  -i ZZ4b/fileLists/data2018A.txt -o ${outputDir} $YEAR2018 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2018A & 
#$runCMD  -i ZZ4b/fileLists/data2018B.txt -o ${outputDir} $YEAR2018 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2018B &
#$runCMD  -i ZZ4b/fileLists/data2018C.txt -o ${outputDir} $YEAR2018 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2018C &
#$runCMD  -i ZZ4b/fileLists/data2018D.txt -o ${outputDir} $YEAR2018 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee  ${outputDir}/log_skim_2018D &

#$runCMD -i ZZ4b/fileLists/TTToHadronic2018.txt     -o ${outputDir} $YEAR2018 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 60.0e3 --isMC 2>&1 |tee ${outputDir}/log_skim_TTHad2018 & 
#$runCMD -i ZZ4b/fileLists/TTToSemiLeptonic2018.txt -o ${outputDir} $YEAR2018 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 60.0e3 --isMC 2>&1 |tee ${outputDir}/log_skim_TTSemi2018 &
#$runCMD -i ZZ4b/fileLists/TTTo2L2Nu2018.txt        -o ${outputDir} $YEAR2018 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 60.0e3 --isMC 2>&1 |tee ${outputDir}/log_skim_TT2L2Nu2018 &



# 2017
#$runCMD -i ZZ4b/fileLists/data2017B.txt -o ${outputDir} $YEAR2017 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2017B &   
#$runCMD -i ZZ4b/fileLists/data2017C.txt -o ${outputDir} $YEAR2017 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2017C &   
#$runCMD -i ZZ4b/fileLists/data2017D.txt -o ${outputDir} $YEAR2017 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2017D &   
#$runCMD -i ZZ4b/fileLists/data2017E.txt -o ${outputDir} $YEAR2017 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2017E &   
#$runCMD -i ZZ4b/fileLists/data2017F.txt -o ${outputDir} $YEAR2017 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2017F &   

#$runCMD -i ZZ4b/fileLists/TTToHadronic2017.txt     -o ${outputDir} $YEAR2017 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 36.7e3 --isMC   2>&1 |tee ${outputDir}/log_skim_TTHad2017 &  
#$runCMD -i ZZ4b/fileLists/TTToSemiLeptonic2017.txt -o ${outputDir} $YEAR2017 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 36.7e3 --isMC   2>&1 |tee ${outputDir}/log_skim_TTSemi2017 & 
#$runCMD -i ZZ4b/fileLists/TTTo2L2Nu2017.txt        -o ${outputDir} $YEAR2017 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 36.7e3 --isMC   2>&1 |tee ${outputDir}/log_skim_TT2L2Nu2017 &


# 2016
#$runCMD -i ZZ4b/fileLists/data2016B.txt -o ${outputDir} $YEAR2016 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2016B &   
#$runCMD -i ZZ4b/fileLists/data2016C.txt -o ${outputDir} $YEAR2016 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2016C &   
#$runCMD -i ZZ4b/fileLists/data2016D.txt -o ${outputDir} $YEAR2016 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2016D &   
#$runCMD -i ZZ4b/fileLists/data2016E.txt -o ${outputDir} $YEAR2016 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2016E &   
#$runCMD -i ZZ4b/fileLists/data2016F.txt -o ${outputDir} $YEAR2016 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2016F &   
#$runCMD -i ZZ4b/fileLists/data2016G.txt -o ${outputDir} $YEAR2016 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2016G &   
#$runCMD -i ZZ4b/fileLists/data2016H.txt -o ${outputDir} $YEAR2016 --histogramming 1 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f  --noDiJetMassCutInPicoAOD 2>&1 |tee ${outputDir}/log_skim_2016H &   

#$runCMD -i ZZ4b/fileLists/TTToHadronic2016.txt     -o ${outputDir} $YEAR2016 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 35.9e3 --isMC 2>&1 |tee ${outputDir}/log_skim_TTHad2016 &  
#$runCMD -i ZZ4b/fileLists/TTToSemiLeptonic2016.txt -o ${outputDir} $YEAR2016 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 35.9e3 --isMC 2>&1 |tee ${outputDir}/log_skim_TTSemi2016 & 
#$runCMD -i ZZ4b/fileLists/TTTo2L2Nu2016.txt        -o ${outputDir} $YEAR2016 --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root -p picoAOD_noDiJetMjj.root -f --noDiJetMassCutInPicoAOD --bTagSF -l 35.9e3 --isMC 2>&1 |tee ${outputDir}/log_skim_TT2L2Nu2016 &


#
# These outputs go into following
#
data2018_noMjj=${outputDir}/fileLists/data2018.txt
data2018_noMjj=${outputDir}/fileLists/data2017.txt
data2018_noMjj=${outputDir}/fileLists/data2016.txt


