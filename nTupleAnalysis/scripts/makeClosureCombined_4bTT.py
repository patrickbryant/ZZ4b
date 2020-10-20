import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9", help="Year or comma separated list of subsamples")
#parser.add_option('-n',            action="store_true", dest="doNominal",       default=False, help="Run nominal 4b samples")
parser.add_option('--mixedName',                        default="3bMix4b_4bTT", help="Year or comma separated list of subsamples")
parser.add_option('--email',            default=None,      help="")
parser.add_option('--addJCM', action="store_true",      help="Should be obvious")
parser.add_option('--convertH5ToROOT', action="store_true",      help="Should be obvious")
parser.add_option('--convertROOTToH5', action="store_true",      help="Should be obvious")
parser.add_option('--makeInputFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--makeOutputFileLists',  action="store_true",      help="make Output file lists")
parser.add_option('--makeTarball',  action="store_true",      help="make Output file lists")
parser.add_option('--copyLocally',  action="store_true",      help="make Input file lists")
parser.add_option('--copyToEOS',  action="store_true",      help="make Input file lists")
parser.add_option('-c',   '--condor',   action="store_true", default=False,           help="Run on condor")

o, a = parser.parse_args()

doRun = o.execute


years = o.year.split(",")
subSamples = o.subSamples.split(",")
mixedName=o.mixedName

baseDir="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src"
outputDir="closureTests/combined_4bTT/"
outputDirNom="closureTests/nominal/"
outputDir3bMix4b="closureTests/"+mixedName+"/"

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


#tagID = "b0p6"
tagID = "b0p60p3"

from condorHelpers import *



CMSSW = getCMSSW()
USER = getUSER()
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/combined_4bTT/"
EOSOUTDIRMIXED = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/mixed/"
EOSOUTDIRNOM = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/nominal/"
EOSOUTDIR3BMIX4B = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+mixedName+"/"

TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"


def getOutDir():
    if o.condor:
        return EOSOUTDIR
    return outputDir

def getOutDirNom():
    if o.condor:
        return EOSOUTDIRNOM
    return outputDirNom

def getOutDirMixed():
    if o.condor:
        return EOSOUTDIRMIXED
    return outputDirMix

def getOutDir3bMix4b():
    if o.condor:
        return EOSOUTDIR3BMIX4B
    return outputDir3bMix4b

if o.makeTarball:
    print "Remove old Tarball"
    rmTARBALL(o.execute)

if o.condor:
    print "Making Tarball"
    makeTARBALL(o.execute)



#
#  Get JCM Files
#    (Might be able to kill...)
jcmNameList="Nominal"
jcmFileList = {}

JCMTagNom = "01-00-00"
JCMTagMixed = "02-01-00"


for y in years:
    jcmFileList[y] = outputDirNom+"/weights/dataRunII_"+tagID+"/jetCombinatoricModel_SB_"+JCMTagNom+".txt"


for s in subSamples:
    #jcmNameList   += ","+mixedName+"_v"+s
    #jcmFileList["2018"] += ","+outputDir3bMix4b+"/weights/data2018_"+mixedName+"_"+tagID+"_v"+s+"/jetCombinatoricModel_SB_"+JCMTag+".txt"
    #jcmFileList["2017"] += ","+outputDir3bMix4b+"/weights/data2017_"+mixedName+"_"+tagID+"_v"+s+"/jetCombinatoricModel_SB_"+JCMTag+".txt"
    #jcmFileList["2016"] += ","+outputDir3bMix4b+"/weights/data2016_"+mixedName+"_"+tagID+"_v"+s+"/jetCombinatoricModel_SB_"+JCMTag+".txt"

    jcmNameList   += ","+mixedName+"_v"+s
    for y in years:
        jcmFileList[y] += ","+outputDir3bMix4b+"/weights/dataRunII_"+mixedName+"_"+tagID+"_v"+s+"/jetCombinatoricModel_SB_"+JCMTagMixed+".txt"


#
# Make Input file lists
#
if o.makeInputFileLists:
    
    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/nominal/"    
    eosDirMixed = "root://cmseos.fnal.gov//store/user/johnda/condor/mixed/"    

    for y in years:

        fileList = outputDir+"/fileLists/data"+y+"_"+tagID+".txt"    
        run("rm "+fileList)

        run("echo "+eosDir+"/data"+y+"_"+tagID+"/picoAOD_"+tagID+".root >> "+fileList)


        for sample in ttbarSamples:
            
            # 3b TT
            fileList = outputDir+"/fileLists/"+sample+y+"_"+tagID+".txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+sample+y+"_noMjj_"+tagID+"/picoAOD_"+tagID+".root >> "+fileList)

            # 4b TT
            fileList = outputDir+"/fileLists/"+sample+y+"_"+tagID+"_noPSData.txt"    
            run("rm "+fileList)
            run("echo "+eosDirMixed+"/"+sample+y+"_noMjj_"+tagID+"/picoAOD_noPSData_"+tagID+".root >> "+fileList)

#
# Make picoAODS of 3b data with weights applied
#
if o.addJCM:
    cmds = []
    logs = []
    dag_config = []
    condor_jobs = []

    #
    #  3b Files
    #
    picoOut3b    = " -p picoAOD_3b_wJCM_"+tagID+".root "
    h10        = " --histogramming 10 "
    histOut3b    = " --histFile hists_3b_wJCM_"+tagID+".root "

    for y in years:

        fileListIn = " -i "+outputDir+"/fileLists/data"+y+"_"+tagID+".txt"
        cmd = runCMD+ fileListIn + " -o "+getOutDir() + picoOut3b + yearOpts[y] + h10 + histOut3b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip4b "

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="addJCM_3b_"))
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_"+y+"_JCM_"+tagID)
        
        for tt in ttbarSamples:

            fileListIn = " -i "+outputDir+"/fileLists/"+tt+y+"_"+tagID+".txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ picoOut3b + MCyearOpts[y] + h10 + histOut3b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip4b "

            if o.condor:
                cmd += " --condor"
                condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="addJCM_3b_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_"+tt+y+"_JCM_"+tagID)


    #
    #  4b Files
    #   (skim to only have 4b events in the pico ADO (Needed for training) )
    #
    picoOut4b    = " -p picoAOD_4b_"+tagID+".root "
    histOut4b    = " --histFile hists_4b_"+tagID+".root "

    for y in years:
        
        #
        #  Nominal 
        #
        fileListIn = " -i "+outputDir+"/fileLists/data"+y+"_"+tagID+".txt"
        cmd = runCMD+ fileListIn + " -o "+getOutDir() + picoOut4b + yearOpts[y] + h10 + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "

        if o.condor:
            cmd += " --condor"
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="addJCM_4b_"))
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_"+y+"_4b_"+tagID)
    
        for tt in ttbarSamples:
    
            fileListIn = " -i "+outputDir+"/fileLists/"+tt+y+"_"+tagID+"_noPSData.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ picoOut4b + MCyearOpts[y] + h10 + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "

            if o.condor:
                cmd += " --condor"
                condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="addJCM_4b_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_"+tt+y+"_4b_"+tagID)
    
        #
        #  Mixed Samples
        #
        for s in subSamples:

            fileListIn = " -i "+outputDir3bMix4b+"/fileLists/mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s+".txt"
            picoOutMixed = " -p picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".root "
            histOutMixed = " --histFile hists_"+mixedName+"_4b_"+tagID+"_v"+s+".root"
            cmd = runCMD + fileListIn + " -o "+getOutDir() + picoOutMixed + yearOpts[y] + h10 + histOutMixed + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b --is3bMixed --isDataMCMix "

            if o.condor:
                cmd += " --condor"
                condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="addJCM_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_"+y+mixedName+"_wJCM_"+tagID+"_v"+s)

    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)



    if o.condor:
        execute("rm "+outputDir+"addJCM_All.dag", doRun)
        execute("rm "+outputDir+"addJCM_All.dag.*", doRun)


        dag_file = makeDAGFile("addJCM_All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)

    else:
        if o.email: execute('echo "Subject: [makeClosureCombined] addJCM  Done" | sendmail '+o.email,doRun)


#
# Convert root to hdf5
#   (with conversion enviorment)
#
if o.convertROOTToH5: 
    cmds = []
    logs = []
    dag_config = []
    condor_jobs = []
    
    #picoAOD = "picoAOD_4b.root"
    picoAOD = "picoAOD_4b_"+tagID+".root"
    picoAODH5 = "picoAOD_4b_"+tagID+".h5"

    for y in years:
        jcmName = "Nominal"

        cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"_"+tagID+"/"+picoAOD+"  -o "+picoAODH5+"             --jcmNameList "+jcmName

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="convertROOTToH5_4b_"))
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_ConvertROOTToH5_data"+y+"_"+tagID)
        
        for tt in ttbarSamples:
            cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+y+"_"+tagID+"_noPSData/"+picoAOD+"  -o "+picoAODH5+"          --jcmNameList "+jcmNameList

            if o.condor:
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID+"_noPSData", outputDir=outputDir, filePrefix="convertROOTToH5_4b_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_ConvertROOTToH5_"+tt+y+"_"+tagID)

        #
        # Mixed events
        #
        for s in subSamples:
            picoIn="picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".root"
            picoOut="picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".h5"
            jcmName = mixedName+"_v"+s

            cmd = convertToH5JOB+" -i "+getOutDir()+"/mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/"+picoIn+"  -o "+picoOut+"             --jcmNameList "+jcmName

            if o.condor:
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="convertROOTToH5_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_ConvertROOTToH5_"+mixedName+"_"+tagID+"_v"+s+"_data"+y)


    #
    #  3b with JCM weights
    #
    #picoAOD = "picoAOD_3b_wJCM.root"
    picoAOD = "picoAOD_3b_wJCM_"+tagID+".root"
    picoAODH5 = "picoAOD_3b_wJCM_"+tagID+".h5"

    for y in years:
        cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"_"+tagID+"/"+picoAOD+"  -o "+picoAODH5+"             --jcmNameList "+jcmNameList

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="convertROOTToH5_3b_"))
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_ConvertROOTToH5_3b_wJCM_data"+y+"_"+tagID)

        for tt in ttbarSamples:
            cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+y+"_"+tagID+"/"+picoAOD+"  -o "+picoAODH5+"        --jcmNameList "+jcmNameList

            if o.condor:
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID, outputDir=outputDir, filePrefix="convertROOTToH5_3b_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_ConvertROOTToH5_3b_wJCM_"+tt+y+"_"+tagID)

    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)

    if o.condor:
        execute("rm "+outputDir+"convertROOTToH5_All.dag", doRun)
        execute("rm "+outputDir+"convertROOTToH5_All.dag.*", doRun)


        dag_file = makeDAGFile("convertROOTToH5_All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)
    else:
        if o.email: execute('echo "Subject: [makeClosureCombined] convertROOTToH5  Done" | sendmail '+o.email,doRun)



#
# Copy h5 files locally
#
if o.copyLocally: 

    def cpLocal(condorBase, localBase, subDir, picoH5):
        cmd  = "xrdcp -f "+condorBase+"/"+subDir+"/"+picoH5+" "+localBase+"/"+subDir+"/"+picoH5
    
        if doRun:
            os.system(cmd)
        else:
            print cmd

    
    picoAODH5 = "picoAOD_4b_"+tagID+".h5"

    for y in years:
        cpLocal(EOSOUTDIR, outputDir, "data"+y+"_"+tagID, picoAODH5)

        for tt in ttbarSamples:
            cpLocal(EOSOUTDIR, outputDir, tt+y+"_"+tagID+"_noPSData", picoAODH5)

        #
        # Mixed events
        #
        for s in subSamples:
            picoOut="picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".h5"

            cpLocal(EOSOUTDIR, outputDir, "mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s, picoOut)


    #
    #  3b with JCM weights
    #
    picoAODH5 = "picoAOD_3b_wJCM_"+tagID+".h5"

    for y in years:
        cpLocal(EOSOUTDIR, outputDir, "data"+y+"_"+tagID, picoAODH5)

        for tt in ttbarSamples:
            cpLocal(EOSOUTDIR, outputDir, tt+y+"_"+tagID, picoAODH5)



#
#  Run commands in makeClosureCombinedTraining.sh
#



#
# Copy h5 files to EOS
#
if o.copyToEOS: 

    def cpEOS(condorBase, localBase, subDir, picoH5):
        cmd  = "xrdcp -f "+localBase+"/"+subDir+"/"+picoH5+" "+condorBase+"/"+subDir+"/"+picoH5
    
        if doRun:
            os.system(cmd)
        else:
            print cmd

    
    picoAODH5 = "picoAOD_4b_"+tagID+".h5"

    for y in years:
        cpEOS(EOSOUTDIR, outputDir, "data"+y+"_"+tagID, picoAODH5)

        for tt in ttbarSamples:
            cpEOS(EOSOUTDIR, outputDir, tt+y+"_"+tagID+"_noPSData", picoAODH5)

        #
        # Mixed events
        #
        for s in subSamples:
            picoOut="picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".h5"

            cpEOS(EOSOUTDIR, outputDir, "mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s, picoOut)


    #
    #  3b with JCM weights
    #
    picoAODH5 = "picoAOD_3b_wJCM_"+tagID+".h5"

    for y in years:
        cpEOS(EOSOUTDIR, outputDir, "data"+y+"_"+tagID, picoAODH5)

        for tt in ttbarSamples:
            cpEOS(EOSOUTDIR, outputDir, tt+y+"_"+tagID, picoAODH5)



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
    dag_config = []
    condor_jobs = []

    #fvtList = "_Nominal"
    #picoAOD = "picoAOD_4b.h5"
    picoAOD = "picoAOD_4b_"+tagID+".h5"

    fvtList = "_Nominal"
    for s in subSamples:
        fvtList += ",_"+mixedName+"_v"+s


    for y in years:
        cmd = convertToROOTJOB+" -i "+getOutDir()+"/data"+y+"_"+tagID+"/"+picoAOD+"               --fvtNameList "+"_Nominal"

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="convertH5ToROOT_4b_"))
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_ConvertToROOT_data"+y+"_"+tagID)


        for tt in ttbarSamples:
            cmd = convertToROOTJOB+" -i "+getOutDir()+"/"+tt+y+"_"+tagID+"_noPSData/"+picoAOD+"          --fvtNameList "+fvtList
            
            if o.condor:
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID+"_noPSData", outputDir=outputDir, filePrefix="convertH5ToROOT_4b_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_ConvertToROOT_"+tt+y+"_"+tagID)


    #
    #   3bMix4b
    #
    for s in subSamples:
        #picoIn="picoAOD_"+mixedName+"_4b_v"+s+".h5"
        picoIn="picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".h5"

        for y in years:
            cmd = convertToROOTJOB+" -i "+getOutDir()+"/mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/"+picoIn+"               --fvtNameList "+"_"+mixedName+"_v"+s

            if o.condor:
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="convertH5ToROOT_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_ConvertToROOT_data"+y+"_"+tagID+"_v"+s)



    #
    #  3b with JCM weights
    #
    #picoAOD = "picoAOD_3b_wJCM.h5"
    picoAOD = "picoAOD_3b_wJCM_"+tagID+".h5"


    for y in years:
        cmd = convertToROOTJOB+" -i "+getOutDir()+"/data"+y+"_"+tagID+"/"+picoAOD+"               --fvtNameList "+fvtList

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="convertH5ToROOT_3b_"))
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_ConvertToROOT_3b_wJCM_data"+y+"_"+tagID)

        for tt in ttbarSamples:
            cmd = convertToROOTJOB+" -i "+getOutDir()+"/"+tt+y+"_"+tagID+"/"+picoAOD+"          --fvtNameList "+fvtList
            
            if o.condor:
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID, outputDir=outputDir, filePrefix="convertH5ToROOT_3b_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_ConvertToROOT_3b_wJCM_"+tt+y+"_"+tagID)


    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)


    if o.condor:
        execute("rm "+outputDir+"convertH5ToROOT_All.dag", doRun)
        execute("rm "+outputDir+"convertH5ToROOT_All.dag.*", doRun)


        dag_file = makeDAGFile("convertH5ToROOT_All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)
    else:
        if o.email: execute('echo "Subject: [makeClosureCombined] convertH5ToROOT  Done" | sendmail '+o.email,doRun)









#
# Make Input file lists
#
if o.makeOutputFileLists:
    
    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/combined_4bTT/"    

    for y in years:

        #
        #  3b 
        #
        picoName = "picoAOD_3b_wJCM_b0p60p3.root"

        fileList = outputDir+"/fileLists/data"+y+"_"+tagID+"_3b_wFvT.txt"    
        run("rm "+fileList)

        run("echo "+eosDir+"/data"+y+"_"+tagID+"/"+picoName+" >> "+fileList)

        for sample in ttbarSamples:
            
            fileList = outputDir+"/fileLists/"+sample+y+"_"+tagID+"_3b_wFvT.txt"    
            run("rm "+fileList)

            run("echo "+eosDir+"/"+sample+y+"_"+tagID+"/"+picoName+" >> "+fileList)


        #
        #  4b 
        #
        picoName = "picoAOD_4b_b0p60p3.root"

        fileList = outputDir+"/fileLists/data"+y+"_"+tagID+"_4b_wFvT.txt"    
        run("rm "+fileList)

        run("echo "+eosDir+"/data"+y+"_"+tagID+"/"+picoName+" >> "+fileList)

        for sample in ttbarSamples:
            
            fileList = outputDir+"/fileLists/"+sample+y+"_"+tagID+"_noPSData_4b_wFvT.txt"    
            run("rm "+fileList)

            run("echo "+eosDir+"/"+sample+y+"_"+tagID+"_noPSData/"+picoName+" >> "+fileList)


        #
        #  Mixed Samples
        #
        for s in subSamples:
            picoName = "picoAOD_"+mixedName+"_4b_b0p60p3_v"+s+".root"

            fileList = outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT.txt"    
            run("rm "+fileList)

            run("echo "+eosDir+"/mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/"+picoName+" >> "+fileList)

