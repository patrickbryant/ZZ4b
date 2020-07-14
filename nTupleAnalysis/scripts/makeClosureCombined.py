import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6", help="Year or comma separated list of subsamples")
parser.add_option('-d',            action="store_true", dest="doData",         default=False, help="Run data")
parser.add_option('-t',            action="store_true", dest="doTT",       default=False, help="Run ttbar MC")
#parser.add_option('-n',            action="store_true", dest="doNominal",       default=False, help="Run nominal 4b samples")
parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--email',            default=None,      help="")
parser.add_option('--addJCM', action="store_true",      help="Should be obvious")
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
#yearOpts["2018"]=' -y 2018 --bTag 0.2770 '
#yearOpts["2017"]=' -y 2017 --bTag 0.3033 '
#yearOpts["2016"]=' -y 2016 --bTag 0.3093 '
yearOpts["2018"]=' -y 2018 --bTag 0.6 '
yearOpts["2017"]=' -y 2017 --bTag 0.6 '
yearOpts["2016"]=' -y 2016 --bTag 0.6 '


MCyearOpts = {}
MCyearOpts["2018"]=yearOpts["2018"]+' --bTagSF -l 60.0e3 --isMC '
MCyearOpts["2017"]=yearOpts["2017"]+' --bTagSF -l 36.7e3 --isMC '
MCyearOpts["2016"]=yearOpts["2016"]+' --bTagSF -l 35.9e3 --isMC '



#
#  Get JCM Files
#    (Might be able to kill...)
jcmNameList="Nominal"
jcmFileList = {}

#jcmFileList["2018"] = outputDirNom+"/weights/data2018/jetCombinatoricModel_SB_00-00-02.txt"
#jcmFileList["2017"] = outputDirNom+"/weights/data2017/jetCombinatoricModel_SB_00-00-02.txt"
#jcmFileList["2016"] = outputDirNom+"/weights/data2016/jetCombinatoricModel_SB_00-00-02.txt"
jcmFileList["2018"] = outputDirNom+"/weights/data2018_b0p6/jetCombinatoricModel_SB_00-00-05.txt"
jcmFileList["2017"] = outputDirNom+"/weights/data2017_b0p6/jetCombinatoricModel_SB_00-00-05.txt"
jcmFileList["2016"] = outputDirNom+"/weights/data2016_b0p6/jetCombinatoricModel_SB_00-00-05.txt"


for s in subSamples:
    jcmNameList   += ","+mixedName+"_v"+s
    jcmFileList["2018"] += ","+outputDir3bMix4b+"/weights/data2018_"+mixedName+"_b0p6_v"+s+"/jetCombinatoricModel_SB_00-00-05.txt"
    jcmFileList["2017"] += ","+outputDir3bMix4b+"/weights/data2017_"+mixedName+"_b0p6_v"+s+"/jetCombinatoricModel_SB_00-00-05.txt"
    jcmFileList["2016"] += ","+outputDir3bMix4b+"/weights/data2016_"+mixedName+"_b0p6_v"+s+"/jetCombinatoricModel_SB_00-00-05.txt"



#
# Make picoAODS of 3b data with weights applied
#
if o.addJCM:
    cmds = []
    logs = []

    #
    #  3b Files
    #
    picoOut3b    = " -p picoAOD_3b_wJCM_b0p6.root "
    h10        = " --histogramming 10 "
    histOut3b    = " --histFile hists_3b_wJCM_b0p6.root "

    for y in years:
        
        jcmList = "Nominal"
        for s in subSamples:
            jcmList += ","+mixedName+"_v"+s


        fileListIn = " -i "+outputDir+"/fileLists/data"+y+"_b0p6.txt"
        cmd = runCMD+ fileListIn + " -o "+outputDir + picoOut3b + yearOpts[y] + h10 + histOut3b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip4b "

        cmds.append(cmd)
        logs.append(outputDir+"/log_"+y+"_JCM_b0p6")
        
        for tt in ttbarSamples:

            fileListIn = " -i "+outputDirNom+"/fileLists/"+tt+y+"_b0p6.txt "
            cmd = runCMD + fileListIn + " -o "+outputDir+ picoOut3b + MCyearOpts[y] + h10 + histOut3b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip4b "

            cmds.append(cmd)
            logs.append(outputDir+"/log_"+tt+y+"_JCM_b0p6")




    #
    #  4b Files
    #   (skim to only have 4b events in the pico ADO (Needed for training) )
    #
    picoOut4b    = " -p picoAOD_4b_b0p6.root "
    histOut4b    = " --histFile hists_4b_b0p6.root "

    for y in years:
        
        #
        #  Nominal 
        #
        fileListIn = " -i "+outputDir+"/fileLists/data"+y+"_b0p6.txt"
        cmd = runCMD+ fileListIn + " -o "+outputDir + picoOut4b + yearOpts[y] + h10 + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "
    
        cmds.append(cmd)
        logs.append(outputDir+"/log_"+y+"_4b_b0p6")
    
        for tt in ttbarSamples:
    
            fileListIn = " -i "+outputDirNom+"/fileLists/"+tt+y+"_b0p6.txt "
            cmd = runCMD + fileListIn + " -o "+outputDir+ picoOut4b + MCyearOpts[y] + h10 + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "
    
            cmds.append(cmd)
            logs.append(outputDir+"/log_"+tt+y+"_4b_b0p6")
    
        #
        #  Mixed Samples
        #
        for s in subSamples:

            fileListIn = " -i "+outputDir3bMix4b+"/data"+y+"_b0p6_v"+s+"/picoAOD_"+mixedName+"_b0p6_v"+s+".root"
            picoOutMixed = " -p picoAOD_"+mixedName+"_4b_b0p6_v"+s+".root "
            histOutMixed = " --histFile hists_"+mixedName+"_4b_b0p6_v"+s+".root"
            cmd = runCMD + fileListIn + " -o "+outputDir + picoOutMixed + yearOpts[y] + h10 + histOutMixed + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b --is3bMixed "

            cmds.append(cmd)
            logs.append(outputDir+"/log_"+y+mixedName+"_wJCM_b0p6_v"+s)

            for tt in ttbarSamples:

                fileListIn = " -i "+outputDir3bMix4b+"/"+tt+y+"_b0p6_v"+s+"/picoAOD_"+mixedName+"_b0p6_v"+s+".root"
                cmd = runCMD + fileListIn + " -o "+outputDir+ picoOutMixed + MCyearOpts[y] + h10 + histOutMixed + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b --is3bMixed "

                cmds.append(cmd)
                logs.append(outputDir+"/log_"+tt+y+"_4b_b0p6_v"+s)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombined] addJCM  Done" | sendmail '+o.email,doRun)


#
# Convert root to hdf5
#   (with conversion enviorment)
#
if o.convertROOTToH5: 
    cmds = []
    logs = []
    
    #picoAOD = "picoAOD_4b.root"
    picoAOD = "picoAOD_4b_b0p6.root"

    for y in years:
        jcmName = "Nominal"

        cmds.append(convertToH5JOB+" -i "+outputDir+"/data"+y+"_b0p6/"+picoAOD+"               --jcmNameList "+jcmName)
        logs.append(outputDir+"/log_ConvertROOTToH5_data"+y+"_b0p6")
        
        for tt in ttbarSamples:
            cmds.append(convertToH5JOB+" -i "+outputDir+"/"+tt+y+"_b0p6/"+picoAOD+"          --jcmNameList "+jcmName)
            logs.append(outputDir+"/log_ConvertROOTToH5_"+tt+y+"_b0p6")

        for s in subSamples:
            #picoIn="picoAOD_"+mixedName+"_4b_v"+s+".root"
            picoIn="picoAOD_"+mixedName+"_4b_b0p6_v"+s+".root"
            jcmName = mixedName+"_v"+s

            cmds.append(convertToH5JOB+" -i "+outputDir3bMix4b+"/data"+y+"_b0p6_v"+s+"/"+picoIn+"               --jcmNameList "+jcmName)
            logs.append(outputDir+"/log_ConvertROOTToH5_"+mixedName+"_b0p6_v"+s+"_data"+y)

            for tt in ttbarSamples:
                cmds.append(convertToH5JOB+" -i "+outputDir3bMix4b+"/"+tt+y+"_b0p6_v"+s+"/"+picoIn+"          --jcmNameList "+jcmName)
                logs.append(outputDir+"/log_ConvertROOTToH5_"+mixedName+"_b0p6_v"+s+"_"+tt+y)
    

    #
    #  3b with JCM weights
    #
    #picoAOD = "picoAOD_3b_wJCM.root"
    picoAOD = "picoAOD_3b_wJCM_b0p6.root"

    jcmList = "Nominal"
    for s in subSamples:
        jcmList += ","+mixedName+"_v"+s

    for y in years:
        cmds.append(convertToH5JOB+" -i "+outputDir+"/data"+y+"_b0p6/"+picoAOD+"               --jcmNameList "+jcmList)
        logs.append(outputDir+"/log_ConvertROOTToH5_3b_wJCM_data"+y+"_b0p6")

        for tt in ttbarSamples:
            cmds.append(convertToH5JOB+" -i "+outputDir+"/"+tt+y+"_b0p6/"+picoAOD+"          --jcmNameList "+jcmList)
            logs.append(outputDir+"/log_ConvertROOTToH5_3b_wJCM_"+tt+y+"_b0p6")

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
    logs = []
    fvtList = "_Nominal"
    #picoAOD = "picoAOD_4b.h5"
    picoAOD = "picoAOD_4b_b0p6.h5"

    for y in years:
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/data"+y+"_b0p6/"+picoAOD+"               --fvtNameList "+fvtList)
        logs.append(outputDir+"/log_ConvertToROOT_data"+y+"_b0p6")

        for tt in ttbarSamples:
            cmds.append(convertToROOTJOB+" -i "+outputDir+"/"+tt+y+"_b0p6/"+picoAOD+"          --fvtNameList "+fvtList)
            logs.append(outputDir+"/log_ConvertToROOT_"+tt+y+"_b0p6")

    #
    #   3bMix4b
    #
    for s in subSamples:
        #picoIn="picoAOD_"+mixedName+"_4b_v"+s+".h5"
        picoIn="picoAOD_"+mixedName+"_4b_b0p6_v"+s+".h5"
        fvtList = "_"+mixedName+"_v"+s

        for y in years:
            cmds.append(convertToROOTJOB+" -i "+outputDir3bMix4b+"/data"+y+"_b0p6_v"+s+"/"+picoIn+"               --fvtNameList "+fvtList)
            logs.append(outputDir+"/log_ConvertToROOT_data"+y+"_b0p6_v"+s)

            for tt in ttbarSamples:
                cmds.append(convertToROOTJOB+" -i "+outputDir3bMix4b+"/"+tt+y+"_b0p6_v"+s+"/"+picoIn+"          --fvtNameList "+fvtList)
                logs.append(outputDir+"/log_ConvertToROOT_"+tt+y+"_b0p6_v"+s)
            


    #
    #  3b with JCM weights
    #
    #picoAOD = "picoAOD_3b_wJCM.h5"
    picoAOD = "picoAOD_3b_wJCM_b0p6.h5"

    fvtList = "_Nominal"
    for s in subSamples:
        fvtList += ",_"+mixedName+"_v"+s

    for y in years:
        cmds.append(convertToROOTJOB+" -i "+outputDir+"/data"+y+"_b0p6/"+picoAOD+"               --fvtNameList "+fvtList)
        logs.append(outputDir+"/log_ConvertToROOT_3b_wJCM_data"+y+"_b0p6")

        for tt in ttbarSamples:
            cmds.append(convertToROOTJOB+" -i "+outputDir+"/"+tt+y+"_b0p6/"+picoAOD+"          --fvtNameList "+fvtList)
            logs.append(outputDir+"/log_ConvertToROOT_3b_wJCM_"+tt+y+"_b0p6")


    babySit(cmds, doRun, logFiles=logs)

