
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('--copySkims',  action="store_true",      help="Make input skims")
parser.add_option('--makeInputFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--inputsForDataVsTT',  action="store_true",      help="makeInputs for Dave Vs TTbar")
parser.add_option('--noConvert',  action="store_true",      help="")
parser.add_option('--makeTarball',  action="store_true",      help="make Output file lists")

parser.add_option('--makeAutonDirs', action="store_true",      help="Setup auton dirs")
parser.add_option('--copyToAuton', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyFromAuton', action="store_true",      help="copy h5 picos from Auton ")


#parser.add_option('--noTT',       action="store_true",      help="Skip TTbar")
parser.add_option('-c',   '--condor',   action="store_true", default=False,           help="Run on condor")
parser.add_option(     '--doTTbarPtReweight',        action="store_true", help="boolean  to toggle using FvT reweight")

o, a = parser.parse_args()

doRun = o.execute

from condorHelpers import *

CMSSW = getCMSSW()
USER = getUSER()
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/ZH4b/UL/"
TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"


outputDir="closureTests/UL/"

# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'

#ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]
signalSamples = ["ZZ4b","ZH4b","ggZH4b"]

years = o.year.split(",")

def getOutDir():
    if o.condor:
        return EOSOUTDIR
    return outputDir


if o.condor:
    print "Making Tarball"
    makeTARBALL(o.execute)

if o.makeTarball:
    print "Remove old Tarball"
    rmTARBALL(o.execute)
    makeTARBALL(o.execute, debug=True)


yearOpts = {}
#yearOpts["2018"]=' -y 2018 --bTag 0.2770 '
#yearOpts["2017"]=' -y 2017 --bTag 0.3033 '
#yearOpts["2016"]=' -y 2016 --bTag 0.3093 '
yearOpts["2018"]=' -y 2018 --bTag 0.6 '
yearOpts["2017"]=' -y 2017 --bTag 0.6 '
yearOpts["2016"]=' -y 2016 --bTag 0.6 '


__MCyearOpts = {}
__MCyearOpts["2018"]=yearOpts["2018"]+' --bTagSF -l 60.0e3 --isMC '
__MCyearOpts["2017"]=yearOpts["2017"]+' --bTagSF -l 36.7e3 --isMC '
__MCyearOpts["2016_preVFP"]=yearOpts["2016"]+' --bTagSF -l 19.5e3 --isMC '
__MCyearOpts["2016_postVFP"]=yearOpts["2016"]+' --bTagSF -l 16.5e3 --isMC '

def MCyearOpts(tt):
    for y in ["2018","2017","2016_preVFP","2016_postVFP"]:
        if not tt.find(y) == -1 :
            return __MCyearOpts[y]


dataPeriods = {}
# All
dataPeriods["2018"] = ["A","B","C","D"]
#dataPeriods["2017"] = ["B","C","D","E","F"]
dataPeriods["2017"] = ["C","D","E","F"]
dataPeriods["2016"] = ["B","C","D","E","F","G","H"]
# for skimming 
#dataPeriods["2018"] = []
#dataPeriods["2017"] = []
#dataPeriods["2016"] = []

# for skimming
ttbarSamplesByYear = {}
ttbarSamplesByYear["2018"] = ["TTToHadronic2018","TTToSemiLeptonic2018","TTTo2L2Nu2018"]
ttbarSamplesByYear["2017"] = ["TTToHadronic2017","TTToSemiLeptonic2017","TTTo2L2Nu2017"]
ttbarSamplesByYear["2016"] = ["TTToHadronic2016_preVFP", "TTToSemiLeptonic2016_preVFP","TTTo2L2Nu2016_preVFP",
                              "TTToHadronic2016_postVFP","TTToSemiLeptonic2016_postVFP","TTTo2L2Nu2016_postVFP",
                              ]

eosls = "eos root://cmseos.fnal.gov ls"
eoslslrt = "eos root://cmseos.fnal.gov ls -lrt"
eosmkdir = "eos root://cmseos.fnal.gov mkdir "

convertToH5JOB='python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'



#
# Make skims with out the di-jet Mass cuts
#
if o.copySkims:
    cmds = []

    for y in years:

        picoName = "picoAOD.root"

        #
        #  Data
        #
        for p in dataPeriods[y]:
            cmds.append("xrdcp -f root://cmseos.fnal.gov//store/user/bryantp/condor/data"+y+p+"/"+picoName+" root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/UL/data"+y+p+"/"+picoName)

        #
        #  TTbar
        # 
        for tt in ttbarSamplesByYear[y]:
            cmds.append("xrdcp -f root://cmseos.fnal.gov//store/user/bryantp/condor/"+tt+"/"+picoName+" root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/UL/"+tt+"/"+picoName)

    babySit(cmds, doRun)


#
#   Make inputs fileLists
#
if o.makeInputFileLists:

    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/UL/"

    for y in years:
        fileList = outputDir+"/fileLists/data"+y+".txt"    
        run("rm "+fileList)

        for p in dataPeriods[y]:
            run("echo "+eosDir+"/data"+y+p+"/picoAOD.root >> "+fileList)


        for tt in ttbarSamplesByYear[y]:
            fileList = outputDir+"/fileLists/"+tt+".txt"    
            run("rm "+fileList)

            run("echo "+eosDir+"/"+tt+"/picoAOD.root >> "+fileList)


# 
#  Separate 3b and 4b for data vs ttbar training
#
if o.inputsForDataVsTT:
    # In the following "3b" refers to 3b subsampled to have the 4b statistics

    dag_config = []
    condor_jobs = []

    histDetailStr        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "

    pico4b    = "picoAOD_4b.root"
    pico3b    = "picoAOD_3b.root"

    picoOut4b = " -p " + pico4b + " "
    histOut4b = " --histFile hists_4b.root"


    picoOut3b = " -p " + pico3b + " " 
    histOut3b = " --histFile hists_3b.root"


    for y in years:

        #
        #  4b 
        #
        cmd = runCMD+" -i "+outputDir+"/fileLists/data"+y+".txt"+ picoOut4b + " -o "+getOutDir()+ yearOpts[y]+  histDetailStr+  histOut4b + " --skip3b "
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix="inputsForDataVsTT_4b_"))

        #
        #  3b
        #
        cmd = runCMD+" -i "+outputDir+"/fileLists/data"+y+".txt"+ picoOut3b + " -o "+getOutDir()+ yearOpts[y]+  histDetailStr+  histOut3b + " --skip4b "
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix="inputsForDataVsTT_3b_"))

        for tt in ttbarSamplesByYear[y]:
            
            #
            # 4b
            #
            cmd = runCMD+" -i "+outputDir+"/fileLists/"+tt+".txt" + picoOut4b + " -o "+getOutDir() + MCyearOpts(tt) +histDetailStr + histOut4b + " --skip3b "
            if o.doTTbarPtReweight:
                cmd += " --doTTbarPtReweight "

            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix="inputsForDataVsTT_4b_"))                    

            #
            # 3b
            #
            cmd = runCMD+" -i "+outputDir+"/fileLists/"+tt+".txt" + picoOut3b + " -o "+getOutDir() + MCyearOpts(tt) +histDetailStr + histOut3b + " --skip4b "
            if o.doTTbarPtReweight:
                cmd += " --doTTbarPtReweight "

            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix="inputsForDataVsTT_3b_"))                    


    dag_config.append(condor_jobs)

    #
    #  Convert root to h5
    #
    if not o.noConvert:

        condor_jobs = []
        pico4b_h5 = "picoAOD_4b.h5"
        pico3b_h5 = "picoAOD_3b.h5"
    
        for y in years:
    
            cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"/"+pico4b+"  -o "+pico4b_h5
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y, outputDir=outputDir, filePrefix="inputsForDataVsTT_convert_4b_"))
    
            cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"/"+pico3b+"  -o "+pico3b_h5
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y, outputDir=outputDir, filePrefix="inputsForDataVsTT_convert_3b_"))
    
            
            for tt in ttbarSamplesByYear[y]:
                cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+"/"+pico4b+"  -o "+pico4b_h5
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt, outputDir=outputDir, filePrefix="inputsForDataVsTT_convert_4b_"))
    
                cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+"/"+pico3b+"  -o "+pico3b_h5
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt, outputDir=outputDir, filePrefix="inputsForDataVsTT_convert_3b_"))
    
        dag_config.append(condor_jobs)    
    

    execute("rm "+outputDir+"inputsForDataVsTT_All.dag", doRun)
    execute("rm "+outputDir+"inputsForDataVsTT_All.dag.*", doRun)

    dag_file = makeDAGFile("inputsForDataVsTT_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




# 
#  Copy to AUTON
#
if o.copyToAuton or o.makeAutonDirs or o.copyFromAuton:
    
    import os
    autonAddr = "gpu13"
    
    
    def run(cmd):
        if doRun:
            os.system(cmd)
        else:
            print cmd
    
    def runA(cmd):
        print "> "+cmd
        run("ssh "+autonAddr+" "+cmd)
    
    def scp(local, auton):
        cmd = "scp "+local+" "+autonAddr+":hh4b/"+auton
        print "> "+cmd
        run(cmd)

    def scpFrom(auton, local):
        cmd = "scp "+autonAddr+":hh4b/"+auton+" "+local
        print "> "+cmd
        run(cmd)

    def scpEOS(eosDir, subdir, pName, autonDir):

        tempPath = "/uscms/home/jda102/nobackup/forSCP/"

        cmd = "xrdcp "+eosDir+"/"+subdir+"/"+pName+"  "+tempPath+pName
        run(cmd)

        cmd = "scp "+tempPath+pName+" "+autonAddr+":hh4b/"+autonDir+"/"+subdir+"/"+pName
        run(cmd)

        cmd = "rm "+tempPath+pName
        run(cmd)

    
    #
    # Setup directories
    #
    if o.makeAutonDirs:

        runA("mkdir hh4b/closureTests/UL")
    
        for y in years:
            runA("mkdir hh4b/closureTests/UL/data"+y)
    
            for tt in ttbarSamplesByYear[y]:
                runA("mkdir hh4b/closureTests/UL/"+tt)
    
    #
    # Copy Files
    #
    if o.copyToAuton:
        for y in years:
            scpEOS(EOSOUTDIR,"data"+y,"picoAOD_3b.h5","hh4b/closureTests/UL")
            scpEOS(EOSOUTDIR,"data"+y,"picoAOD_4b.h5","hh4b/closureTests/UL")
            
            for tt in ttbarSamplesByYear[y]:
                scpEOS(EOSOUTDIR,tt,"picoAOD_3b.h5","hh4b/closureTests/UL")
                scpEOS(EOSOUTDIR,tt,"picoAOD_4b.h5","hh4b/closureTests/UL")



    #
    # Copy Files
    #
    if o.copyFromAuton:
        #pName = "picoAOD_3b_"+tagID+"_DvT3.h5"


        for y in ["2018","2017","2016"]:
            #pName = "picoAOD_3b_"+tagID+"_DvT3_with_rwTT.h5"
            pName = "picoAOD_4b_"+tagID+"_DvT4_no_rwTT.h5"
            scpFrom("closureTests/"+dirName+"/data"+y+"_"+tagID+"/"+pName, "closureTests/mixed/data"+y+"_"+tagID+"/"+pName)


            for tt in ttbarSamples:
                #pName = "picoAOD_3b_"+tagID+"_DvT3_with_rwTT.h5"

                scpFrom("closureTests/"+dirName+"/"+tt+y+"_"+tagID+"/"+pName, "closureTests/mixed/"+tt+y+"_"+tagID+"/"+pName )
                #scp("closureTests/mixed/"+tt+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5", "closureTests/"+dirName+"/"+tt+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5")


