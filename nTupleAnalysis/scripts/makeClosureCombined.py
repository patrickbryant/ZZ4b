import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4", help="Year or comma separated list of subsamples")
parser.add_option('-d',            action="store_true", dest="doData",         default=False, help="Run data")
parser.add_option('-t',            action="store_true", dest="doTT",       default=False, help="Run ttbar MC")
parser.add_option('-n',            action="store_true", dest="doNominal",       default=False, help="Run nominal 4b samples")
parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--email',            default=None,      help="")
parser.add_option('--addFvT', action="store_true",      help="Should be obvious")
parser.add_option('--convertH5ToROOT', action="store_true",      help="Should be obvious")
parser.add_option('--convertROOTToH5', action="store_true",      help="Should be obvious")
o, a = parser.parse_args()

doRun = o.execute


years = o.year.split(",")
subSamples = o.subSamples.split(",")
mixedName=o.mixedName

baseDir="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src"
outputDir=baseDir+"/closureTests/combined"
outputDirNom=baseDir+"/closureTests/nominal"
outputDir3bMix4b=baseDir+"/closureTests/3bMix4b"


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
convertToH5JOB='python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'
convertToROOTJOB='python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py'


ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]

yearOpts = {}
yearOpts["2018"]=' -y 2018 --bTag 0.2770 '
yearOpts["2017"]=' -y 2017 --bTag 0.3033 '
yearOpts["2016"]=' -y 2016 --bTag 0.3093 '

MCyearOpts = {}
MCyearOpts["2018"]=yearOpts["2018"]+' --bTagSF -l 60.0e3 --isMC '
MCyearOpts["2017"]=yearOpts["2017"]+' --bTagSF -l 36.7e3 --isMC '
MCyearOpts["2016"]=yearOpts["2016"]+' --bTagSF -l 35.9e3 --isMC '



#
#  Get JCM Files
#    (Might be able to kill...)
jcmNameList="Nominal"
jcmFileList = {}

jcmFileList["2018"] = outputDirNom+"/weights/data2018/jetCombinatoricModel_SB_00-00-02.txt"
jcmFileList["2017"] = outputDirNom+"/weights/data2017/jetCombinatoricModel_SB_00-00-02.txt"
jcmFileList["2016"] = outputDirNom+"/weights/data2016/jetCombinatoricModel_SB_00-00-02.txt"


for s in subSamples:
    jcmNameList   += ","+mixedName+"_v"+s
    jcmFileList["2018"] += ","+outputDir3bMix4b+"/weights/data2018_"+mixedName+"_v"+s+"/jetCombinatoricModel_SB_00-00-04.txt"
    jcmFileList["2017"] += ","+outputDir3bMix4b+"/weights/data2017_"+mixedName+"_v"+s+"/jetCombinatoricModel_SB_00-00-04.txt"
    jcmFileList["2016"] += ","+outputDir3bMix4b+"/weights/data2016_"+mixedName+"_v"+s+"/jetCombinatoricModel_SB_00-00-04.txt"



#
# Make picoAODS of 3b data with weights applied  (for closure test)
#
if o.addFvT:
    cmds = []
    logs = []

    #
    #  3b Files
    #
    picoOut3b    = " -p picoAOD_3b_wJCM.root "
    h10        = " --histogramming 10 "
    histOut3b    = " --histFile hists_3b_wJCM.root "

    for y in years:
        
        jcmList = "Nominal"
        for s in subSamples:
            jcmList += ","+mixedName+"_v"+s


        fileListIn = " -i "+outputDir+"/fileLists/data"+y+".txt"
        cmd = runCMD+ fileListIn + " -o "+outputDir + picoOut3b + yearOpts[y] + h10 + histOut3b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip4b "

        cmds.append(cmd)
        logs.append(outputDir+"/log_"+y+"_JCM")
        
        for tt in ttbarSamples:

            fileListIn = " -i "+outputDirNom+"/fileLists/"+tt+y+".txt "
            cmd = runCMD + fileListIn + " -o "+outputDir+ picoOut3b + MCyearOpts[y] + h10 + histOut3b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip4b "

            cmds.append(cmd)
            logs.append(outputDir+"/log_"+tt+y+"_JCM")




    #
    #  4b Files
    #   (skim to only have 4b events in the pico ADO (Needed for training) )
    #
    picoOut4b    = " -p picoAOD_4b.root "
    histOut4b    = " --histFile hists_4b.root "

    for y in years:
        
        #
        #  Nominal 
        #
        if o.doNominal:
            fileListIn = " -i "+outputDir+"/fileLists/data"+y+".txt"
            cmd = runCMD+ fileListIn + " -o "+outputDir + picoOut4b + yearOpts[y] + h10 + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "
    
            cmds.append(cmd)
            logs.append(outputDir+"/log_"+y+"_4b")
    
            for tt in ttbarSamples:
    
                fileListIn = " -i "+outputDirNom+"/fileLists/"+tt+y+".txt "
                cmd = runCMD + fileListIn + " -o "+outputDir+ picoOut4b + MCyearOpts[y] + h10 + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "
    
                cmds.append(cmd)
                logs.append(outputDir+"/log_"+tt+y+"_4b")
    
        #
        #  Mixed Samples
        #
        for s in subSamples:

            fileListIn = " -i "+outputDir3bMix4b+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_v"+s+".root"
            picoOutMixed = " -p picoAOD_"+mixedName+"_4b_v"+s+".root "
            histOutMixed = " --histFile hists_"+mixedName+"_4b_v"+s+".root"
            cmd = runCMD + fileListIn + " -o "+outputDir + picoOutMixed + yearOpts[y] + h10 + histOutMixed + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b --is3bMixed "

            cmds.append(cmd)
            logs.append(outputDir+"/log_"+y+mixedName+"_wJCM_v"+s)

            for tt in ttbarSamples:

                fileListIn = " -i "+outputDir3bMix4b+"/"+tt+y+"_v"+s+"/picoAOD_"+mixedName+"_v"+s+".root"
                cmd = runCMD + fileListIn + " -o "+outputDir+ picoOutMixed + MCyearOpts[y] + h10 + histOutMixed + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b --is3bMixed "

                cmds.append(cmd)
                logs.append(outputDir+"/log_"+tt+y+"_4b_v"+s)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombined] addFvT  Done" | sendmail '+o.email,doRun)


#
# Convert root to hdf5
#   (with conversion enviorment)
#
if o.convertROOTToH5: 
    cmds = []
    logs = []
    
    for y in years:
        jcmName = "Nominal"

#HACK put back!         cmds.append(convertToH5JOB+" -i "+outputDirNom+"/data"+y+"/picoAOD_4b.root               --jcmNameList "+jcmName)
#HACK put back!         logs.append(outputDir+"/log_ConvertH5ToROOT_data"+y)
#HACK put back! 
#HACK put back!         for tt in ttbarSamples:
#HACK put back!             cmds.append(convertToH5JOB+" -i "+outputDirNom+"/"+tt+y+"/picoAOD_4b.root          --jcmNameList "+jcmName)
#HACK put back!             logs.append(outputDir+"/log_ConvertH5ToROOT_"+tt+y)
#HACK put back! 
#HACK put back!         for s in subSamples:
        for s in ["1"]:
            picoIn="picoAOD_"+mixedName+"_4b_v"+s+".root"
            jcmName = mixedName+"_v"+s
    
            cmds.append(convertToH5JOB+" -i "+outputDir3bMix4b+"/data"+y+"_v"+s+"/"+picoIn+"               --jcmNameList "+jcmName)
            logs.append(outputDir+"/log_ConvertH5ToROOT_"+mixedName+"_v"+s+"_data"+y)

            for tt in ttbarSamples:
                cmds.append(convertToH5JOB+" -i "+outputDir3bMix4b+"/"+tt+y+"_v"+s+"/"+picoIn+"          --jcmNameList "+jcmName)
                logs.append(outputDir+"/log_ConvertH5ToROOT_"+mixedName+"_v"+s+"_"+tt+y)
    

    #
    #  3b with JCM weights
    #
    jcmList = "Nominal"
    for s in subSamples:
        jcmList += ","+mixedName+"_v"+s

    for y in years:
        cmds.append(convertToH5JOB+" -i "+outputDir+"/data"+y+"/picoAOD_3b_wJCM.root               --jcmNameList "+jcmList)
        logs.append(outputDir+"/log_ConvertH5ToROOT_3b_wJCM_data"+y)

        for tt in ttbarSamples:
            cmds.append(convertToH5JOB+" -i "+outputDir+"/"+tt+y+"/picoAOD_3b_wJCM.root          --jcmNameList "+jcmList)
            logs.append(outputDir+"/log_ConvertH5ToROOT_3b_wJCM_"+tt+y)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombined] convertROOTToH5  Done" | sendmail '+o.email,doRun)



#
#  Run commands in makeClosureCombinedTraining.sh
#


#
# Convert hdf5 to root
#   (with conversion enviorment)
#  "4b Data"
#
if o.convertH5ToROOT: 


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
        picoIn="picoAOD_"+mixedName+"_4b_v"+s+".h5"
        fvtList = "_"+mixedName+"_v"+s

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
        fvtList += ",_"+mixedName+"_v"+s

    for y in years:
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/data"+y+"/picoAOD_3b_wJCM.h5               --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/TTTo2L2Nu"+y+"/picoAOD_3b_wJCM.h5          --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/TTToHadronic"+y+"/picoAOD_3b_wJCM.h5       --fvtNameList "+fvtList)
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/TTToSemiLeptonic"+y+"/picoAOD_3b_wJCM.h5   --fvtNameList "+fvtList)


    babySit(cmds, doRun)

