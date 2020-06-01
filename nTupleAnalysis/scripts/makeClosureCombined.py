import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
o, a = parser.parse_args()

doRun = o.execute
convertH5ToROOT=True
convertROOTToH5=False


baseDir="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src"
outputDir=baseDir+"/closureTests/combined"
outputDirNom=baseDir+"/closureTests/nominal"
outputDir3bMix4b=baseDir+"/closureTests/3bMix4b"


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
convertToH5JOB='python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'
convertToROOTJOB='python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py'

years = ["2018","2017","2016"]

yearOpts = {}
yearOpts["2018"]=' -y 2018 --bTag 0.2770 '
yearOpts["2017"]=' -y 2017 --bTag 0.3033 '
yearOpts["2016"]=' -y 2016 --bTag 0.3093 '

MCyearOpts = {}
MCyearOpts["2018"]=yearOpts["2018"]+' --bTagSF -l 60.0e3 --isMC '
MCyearOpts["2017"]=yearOpts["2017"]+' --bTagSF -l 36.7e3 --isMC '
MCyearOpts["2016"]=yearOpts["2016"]+' --bTagSF -l 35.9e3 --isMC '

subSamples = ["0", "1", "2", "3", "4"]

#
#  Get JCM Files
#    (Might be able to kill...)
jcmNameList="Nominal"
jcmFileList18 = outputDirNom+"/weights/data2018/jetCombinatoricModel_SB_00-00-02.txt"
jcmFileList17 = outputDirNom+"/weights/data2017/jetCombinatoricModel_SB_00-00-02.txt"
jcmFileList16 = outputDirNom+"/weights/data2016/jetCombinatoricModel_SB_00-00-02.txt"

for s in subSamples:
    jcmNameList   += ",3bMix4b_v"+s
    jcmFileList18 += ","+outputDir3bMix4b+"/weights/data2018_3bMix4b_v"+s+"/jetCombinatoricModel_SB_00-00-02.txt"
    jcmFileList17 += ","+outputDir3bMix4b+"/weights/data2017_3bMix4b_v"+s+"/jetCombinatoricModel_SB_00-00-02.txt"
    jcmFileList16 += ","+outputDir3bMix4b+"/weights/data2016_3bMix4b_v"+s+"/jetCombinatoricModel_SB_00-00-02.txt"



#
# Make picoAODS of 3b data with weights applied  (for closure test)
#
#$runCMD  -i ${outputDir}/fileLists/data2018.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2018 --histogramming 10 --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --skip4b 2>&1|tee ${outputDir}/log_2018_JCM  &
#$runCMD  -i ${outputDir}/fileLists/data2017.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2017 --histogramming 10 --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --skip4b 2>&1|tee ${outputDir}/log_2017_JCM  &
#$runCMD  -i ${outputDir}/fileLists/data2016.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2016 --histogramming 10 --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --skip4b 2>&1|tee ${outputDir}/log_2016_JCM  &


### 2018
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2018.txt     -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2018MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --skip4b 2>&1 |tee ${outputDir}/log_TTHad2018_JCM   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2018.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2018MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --skip4b 2>&1 |tee ${outputDir}/log_TTSemi2018_JCM  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2018.txt        -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2018MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --skip4b 2>&1 |tee ${outputDir}/log_TT2L2Nu2018_JCM &

###2017
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2017.txt     -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2017MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --skip4b 2>&1 |tee ${outputDir}/log_TTHad2017_JCM   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2017.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2017MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --skip4b 2>&1 |tee ${outputDir}/log_TTSemi2017_JCM  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2017.txt        -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2017MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --skip4b 2>&1 |tee ${outputDir}/log_TT2L2Nu2017_JCM &
#
###2016
#$runCMD -i ${outputDirNom}/fileLists/TTToHadronic2016.txt     -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2016MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --skip4b 2>&1 |tee ${outputDir}/log_TTHad2016_JCM   & 
#$runCMD -i ${outputDirNom}/fileLists/TTToSemiLeptonic2016.txt -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2016MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --skip4b 2>&1 |tee ${outputDir}/log_TTSemi2016_JCM  &
#$runCMD -i ${outputDirNom}/fileLists/TTTo2L2Nu2016.txt        -o ${outputDir} -p picoAOD_3b_wJCM.root $YEAR2016MC --histogramming 10  --histFile hists_3b_wJCM.root --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --skip4b 2>&1 |tee ${outputDir}/log_TT2L2Nu2016_JCM &


##
## skim to only have 4b events in the pico ADO (Needed for training) 
##
#$runCMD  -i ${outputDir}/fileLists/data2018.txt -p picoAOD_4b.root   $YEAR2018  --histogramming 10 --histFile hists_4b.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 2>&1|tee ${outputDir}/log_2018_4b  &
#$runCMD  -i ${outputDir}/fileLists/data2017.txt -p picoAOD_4b.root   $YEAR2017  --histogramming 10 --histFile hists_4b.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 2>&1|tee ${outputDir}/log_2017_4b  &
#$runCMD  -i ${outputDir}/fileLists/data2016.txt -p picoAOD_4b.root   $YEAR2016  --histogramming 10 --histFile hists_4b.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 2>&1|tee ${outputDir}/log_2016_4b  &
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


#mkdir ${outputDirNom}/TT2018
#mkdir ${outputDirNom}/TT2017
#mkdir ${outputDirNom}/TT2016
#
#hadd -f ${outputDirNom}/TT2018/hists_4b.root ${outputDirNom}/TTToHadronic2018/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2018/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2018/hists_4b.root 
#hadd -f ${outputDirNom}/TT2017/hists_4b.root ${outputDirNom}/TTToHadronic2017/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2017/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2017/hists_4b.root 
#hadd -f ${outputDirNom}/TT2016/hists_4b.root ${outputDirNom}/TTToHadronic2016/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2016/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2016/hists_4b.root 



#for i in 0 1 2 3 4 
#do
##    $runCMD  -i ${outputDir3bMix4b}/data2018_v${i}/picoAOD_3bMix4b_noTTVeto_v${i}.root -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2018    --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --is3bMixed 2>&1|tee ${outputDir}/log_2018_3bMix4b_noTtVeto_4b_v${i}  &
##    $runCMD -i ${outputDir3bMix4b}/TTToHadronic2018_v${i}/picoAOD_3bMix4b_v${i}.root       -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2018MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --is3bMixed 2>&1 |tee ${outputDir}/log_TTHad2018_4b_v${i}   & 
##    $runCMD -i ${outputDir3bMix4b}/TTToSemiLeptonic2018_v${i}/picoAOD_3bMix4b_v${i}.root   -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2018MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --is3bMixed 2>&1 |tee ${outputDir}/log_TTSemi2018_4b_v${i}   & 
##    $runCMD -i ${outputDir3bMix4b}/TTTo2L2Nu2018_v${i}/picoAOD_3bMix4b_v${i}.root          -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2018MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList18 --is3bMixed 2>&1 |tee ${outputDir}/log_TT2L2N2018_4b_v${i}   & 
#
##    $runCMD  -i ${outputDir3bMix4b}/data2017_v${i}/picoAOD_3bMix4b_noTTVeto_v${i}.root -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2017    --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --is3bMixed 2>&1|tee ${outputDir}/log_2017_3bMix4b_noTtVeto_4b_v${i}  &
##    $runCMD -i ${outputDir3bMix4b}/TTToHadronic2017_v${i}/picoAOD_3bMix4b_v${i}.root       -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2017MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --is3bMixed 2>&1 |tee ${outputDir}/log_TTHad2017_4b_v${i}   & 
##    $runCMD -i ${outputDir3bMix4b}/TTToSemiLeptonic2017_v${i}/picoAOD_3bMix4b_v${i}.root   -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2017MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --is3bMixed 2>&1 |tee ${outputDir}/log_TTSemi2017_4b_v${i}   & 
##    $runCMD -i ${outputDir3bMix4b}/TTTo2L2Nu2017_v${i}/picoAOD_3bMix4b_v${i}.root          -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2017MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList17 --is3bMixed 2>&1 |tee ${outputDir}/log_TT2L2N2017_4b_v${i}   & 
#
#    $runCMD  -i ${outputDir3bMix4b}/data2016_v${i}/picoAOD_3bMix4b_noTTVeto_v${i}.root -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2016    --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --is3bMixed 2>&1|tee ${outputDir}/log_2016_3bMix4b_noTtVeto_4b_v${i}  &
#    $runCMD -i ${outputDir3bMix4b}/TTToHadronic2016_v${i}/picoAOD_3bMix4b_v${i}.root       -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2016MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --is3bMixed 2>&1 |tee ${outputDir}/log_TTHad2016_4b_v${i}   & 
#    $runCMD -i ${outputDir3bMix4b}/TTToSemiLeptonic2016_v${i}/picoAOD_3bMix4b_v${i}.root   -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2016MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --is3bMixed 2>&1 |tee ${outputDir}/log_TTSemi2016_4b_v${i}   & 
#    $runCMD -i ${outputDir3bMix4b}/TTTo2L2Nu2016_v${i}/picoAOD_3bMix4b_v${i}.root          -p picoAOD_3bMix4b_noTTVeto_4b_v${i}.root   $YEAR2016MC  --histogramming 10  --histFile hists_3bMix4b_noTTVeto_4b_v${i}.root  --skip3b --jcmNameList $jcmNameList --jcmFileList $jcmFileList16 --is3bMixed 2>&1 |tee ${outputDir}/log_TT2L2N2016_4b_v${i}   & 
#done



#
# Convert root to hdf5
#   (with conversion enviorment)
#  "4b Data"
#
if convertROOTToH5: 
    cmds = []
    
    for y in years:
        jcmName = "Nominal"
        cmds.append(convertToH5JOB+" -i "+outputDirNom+"/data"+y+"/picoAOD_4b.root               --jcmNameList "+jcmName)
        cmds.append(convertToH5JOB+" -i "+outputDirNom+"/TTTo2L2Nu"+y+"/picoAOD_4b.root          --jcmNameList "+jcmName)
        cmds.append(convertToH5JOB+" -i "+outputDirNom+"/TTToHadronic"+y+"/picoAOD_4b.root       --jcmNameList "+jcmName)
        cmds.append(convertToH5JOB+" -i "+outputDirNom+"/TTToSemiLeptonic"+y+"/picoAOD_4b.root   --jcmNameList "+jcmName)


    babySit(cmds, doRun)
    

    cmds = []
    for s in subSamples:
        picoIn="picoAOD_3bMix4b_noTTVeto_4b_v"+s+".root"
        jcmName = "3bMix4b_v"+s

        for y in years:
            cmds.append(convertToH5JOB+" -i "+outputDir3bMix4b+"/data"+y+"_v"+s+"/"+picoIn+"               --jcmNameList "+jcmName)
            cmds.append(convertToH5JOB+" -i "+outputDir3bMix4b+"/TTTo2L2Nu"+y+"_v"+s+"/"+picoIn+"          --jcmNameList "+jcmName)
            cmds.append(convertToH5JOB+" -i "+outputDir3bMix4b+"/TTToHadronic"+y+"_v"+s+"/"+picoIn+"       --jcmNameList "+jcmName)
            cmds.append(convertToH5JOB+" -i "+outputDir3bMix4b+"/TTToSemiLeptonic"+y+"_v"+s+"/"+picoIn+"   --jcmNameList "+jcmName)

        babySit(cmds, doRun)


    #
    #  3b with JCM weights
    #
    cmds = []
    jcmList = "Nominal"
    for s in subSamples:
        jcmList += ",3bMix4b_v"+s

    for y in years:
        cmds.append(convertToH5JOB+" -i "+outputDir+"/data"+y+"/picoAOD_3b_wJCM.root               --jcmNameList "+jcmList)
        cmds.append(convertToH5JOB+" -i "+outputDir+"/TTTo2L2Nu"+y+"/picoAOD_3b_wJCM.root          --jcmNameList "+jcmList)
        cmds.append(convertToH5JOB+" -i "+outputDir+"/TTToHadronic"+y+"/picoAOD_3b_wJCM.root       --jcmNameList "+jcmList)
        cmds.append(convertToH5JOB+" -i "+outputDir+"/TTToSemiLeptonic"+y+"/picoAOD_3b_wJCM.root   --jcmNameList "+jcmList)


    babySit(cmds, doRun)




#
#  Run commands in makeClosureCombinedTraining.sh
#


#
# Convert hdf5 to root
#   (with conversion enviorment)
#  "4b Data"
#
if convertH5ToROOT: 


    #
    #   4b Nominal
    #
    cmds = []
    fvtList = "_Nominal"
    for y in years:
        cmds.append(convertToROOTJOB+" -i "+outputDirNom+"/data"+y+"/picoAOD_4b.h5               --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDirNom+"/TTTo2L2Nu"+y+"/picoAOD_4b.h5          --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDirNom+"/TTToHadronic"+y+"/picoAOD_4b.h5       --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDirNom+"/TTToSemiLeptonic"+y+"/picoAOD_4b.h5   --fvtNameList "+fvtList)


    babySit(cmds, doRun)
    

    #
    #   3bMix4b
    #
    cmds = []
    for s in subSamples:
        picoIn="picoAOD_3bMix4b_noTTVeto_4b_v"+s+".h5"
        fvtList = "_3bMix4b_v"+s

        for y in years:
            cmds.append(convertToROOTJOB+" -i "+outputDir3bMix4b+"/data"+y+"_v"+s+"/"+picoIn+"               --fvtNameList "+fvtList)
            cmds.append(convertToROOTJOB+" -i "+outputDir3bMix4b+"/TTTo2L2Nu"+y+"_v"+s+"/"+picoIn+"          --fvtNameList "+fvtList)
            cmds.append(convertToROOTJOB+" -i "+outputDir3bMix4b+"/TTToHadronic"+y+"_v"+s+"/"+picoIn+"       --fvtNameList "+fvtList)
            cmds.append(convertToROOTJOB+" -i "+outputDir3bMix4b+"/TTToSemiLeptonic"+y+"_v"+s+"/"+picoIn+"   --fvtNameList "+fvtList)

        babySit(cmds, doRun)


    #
    #  3b with JCM weights
    #
    cmds = []
    fvtList = "_Nominal"
    for s in subSamples:
        fvtList += ",_3bMix4b_v"+s

    for y in years:
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/data"+y+"/picoAOD_3b_wJCM.h5               --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/TTTo2L2Nu"+y+"/picoAOD_3b_wJCM.h5          --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/TTToHadronic"+y+"/picoAOD_3b_wJCM.h5       --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/TTToSemiLeptonic"+y+"/picoAOD_3b_wJCM.h5   --fvtNameList "+fvtList)


    babySit(cmds, doRun)

