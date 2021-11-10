
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse
from glob import glob

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('--makeSkims',  action="store_true",      help="Make input skims")
parser.add_option('--haddChunks',  action="store_true",      help="Hadd chunks")
parser.add_option('--makeTarball',  action="store_true",      help="make Output file lists")
parser.add_option('--makeInputFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--makeHists',  action="store_true",      help="make Input file lists")

#parser.add_option('--noTT',       action="store_true",      help="Skip TTbar")
parser.add_option('-c',   '--condor',   action="store_true", default=False,           help="Run on condor")
parser.add_option(     '--doTTbarPtReweight',        action="store_true", help="boolean  to toggle using FvT reweight")

o, a = parser.parse_args()

doRun = o.execute

from condorHelpers import *

CMSSW = getCMSSW()
USER = getUSER()
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/ZH4b/TTStudy/"
TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"


outputDir="closureTests/TTStudy/"

# Helpers
runCMD='tTbarNTupleAnalysis ZZ4b/nTupleAnalysis/scripts/tTbarAnalysis_cfg.py'

years = o.year.split(",")
streams = ["MuonEgData","SingleMuonData"]



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

def getFileChunks(tag):
    files = glob('ZZ4b/fileLists/'+tag+'_chunk*.txt')
    return files


dataPeriods = {}
# All
dataPeriods["2018"] = ["A","B","C","D"]
dataPeriods["2017"] = ["B","C","D","E","F"]
dataPeriods["2016"] = ["B","C","D","E","F","G","H"]

#dataPeriods["2018"] = ["D"]
#dataPeriods["2017"] = []
#dataPeriods["2016"] = []


# for skimming
ttbarSamplesByYear = {}
ttbarSamplesByYear["2018"] = ["TTToHadronic2018","TTToSemiLeptonic2018","TTTo2L2Nu2018"]
ttbarSamplesByYear["2017"] = ["TTToHadronic2017","TTToSemiLeptonic2017","TTTo2L2Nu2017"]
ttbarSamplesByYear["2016"] = ["TTToHadronic2016_preVFP", "TTToSemiLeptonic2016_preVFP","TTTo2L2Nu2016_preVFP",
                              "TTToHadronic2016_postVFP","TTToSemiLeptonic2016_postVFP","TTTo2L2Nu2016_postVFP",
                              ]

#ttbarSamplesByYear["2018"] = ["TTToHadronic2018","TTToSemiLeptonic2018"]
#ttbarSamplesByYear["2017"] = ["TTToHadronic2017","TTToSemiLeptonic2017"]
#ttbarSamplesByYear["2016"] = [                              ]


eosls = "eos root://cmseos.fnal.gov ls"
eoslslrt = "eos root://cmseos.fnal.gov ls -lrt"
eosmkdir = "eos root://cmseos.fnal.gov mkdir "

convertToH5JOB='python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'



#
# Make skims with out the di-jet Mass cuts
#
if o.makeSkims:

    dag_config = []
    condor_jobs = []
    jobName = "makeSkims_"

    for y in years:
        
        histConfig = " --histDetailLevel allEvents.passPreSel --histFile histsFromNanoAOD.root "
        picoOut = " -p picoAOD.root "

        #
        #  Data
        #
        for p in dataPeriods[y]:

            for s in streams:
                chunckedFiles = getFileChunks(s+y+p)
                for ic, cf in enumerate(chunckedFiles):
                    cmd = runCMD+"  -i "+cf+" -o "+getOutDir() +  yearOpts[y] + histConfig + picoOut
                    condor_jobs.append(makeCondorFile(cmd, "None", s+y+p+"_c"+str(ic), outputDir=outputDir, filePrefix=jobName))


        #
        #  TTbar
        # 
        for tt in ttbarSamplesByYear[y]:
            chunckedFiles = getFileChunks(tt)
            for ic, cf in enumerate(chunckedFiles):
                cmd = runCMD+" -i "+cf+" -o "+getOutDir()+  MCyearOpts(tt) + histConfig + picoOut
                condor_jobs.append(makeCondorFile(cmd, "None", tt+"_c"+str(ic), outputDir=outputDir, filePrefix=jobName))                    


    dag_config.append(condor_jobs)
    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
# Make skims with out the di-jet Mass cuts
#
if o.haddChunks:

    dag_config = []
    condor_jobs = []
    jobName = "haddChunks" 

    for y in years:

        picoName = "picoAOD.root"

        #
        #  Data
        #
        for p in dataPeriods[y]:

            for s in streams:

                cmdPico = "hadd -f "+getOutDir()+"/"+s+y+p+"/picoAOD.root "
                cmdHist = "hadd -f "+getOutDir()+"/"+s+y+p+"/histsFromNanoAOD.root "

                chunckedFiles = getFileChunks(s+y+p)
                for ic, cf in enumerate(chunckedFiles):
                
                    chIdx = ic + 1
                    chunkName = str(chIdx) if chIdx > 9 else "0"+str(chIdx)
                    cmdPico += getOutDir()+"/"+s+y+p+"_chunk"+str(chunkName)+"/picoAOD.root "
                    cmdHist += getOutDir()+"/"+s+y+p+"_chunk"+str(chunkName)+"/histsFromNanoAOD.root "

                condor_jobs.append(makeCondorFile(cmdPico, "None", s+y+p+"_pico", outputDir=outputDir, filePrefix=jobName))
                condor_jobs.append(makeCondorFile(cmdHist, "None", s+y+p+"_hist", outputDir=outputDir, filePrefix=jobName))


        #
        #  TTbar
        # 
        for tt in ttbarSamplesByYear[y]:

            cmdPico = "hadd -f "+getOutDir()+"/"+tt+"/picoAOD.root "
            cmdHist = "hadd -f "+getOutDir()+"/"+tt+"/histsFromNanoAOD.root "

            chunckedFiles = getFileChunks(tt)
            for ic, cf in enumerate(chunckedFiles):

                chIdx = ic + 1
                chunkName = str(chIdx) if chIdx > 9 else "0"+str(chIdx)
                cmdPico += getOutDir()+"/"+tt+"_chunk"+str(chunkName)+"/picoAOD.root "
                cmdHist += getOutDir()+"/"+tt+"_chunk"+str(chunkName)+"/histsFromNanoAOD.root "

            condor_jobs.append(makeCondorFile(cmdPico, "None", tt+"_pico", outputDir=outputDir, filePrefix=jobName))
            condor_jobs.append(makeCondorFile(cmdHist, "None", tt+"_hist", outputDir=outputDir, filePrefix=jobName))



    dag_config.append(condor_jobs)
    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#   Make inputs fileLists
#
if o.makeInputFileLists:

    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    for y in years:

        for s in streams:

            fileList = outputDir+"/fileLists/"+s+y+".txt"    
            run("rm "+fileList)

            for p in dataPeriods[y]:
                run("echo "+EOSOUTDIR+"/"+s+y+p+"/picoAOD.root >> "+fileList)


        for tt in ttbarSamplesByYear[y]:
            fileList = outputDir+"/fileLists/"+tt+".txt"    
            run("rm "+fileList)

            run("echo "+EOSOUTDIR+"/"+tt+"/picoAOD.root >> "+fileList)


# 
#  Make Hists
#
if o.makeHists:

    jobName = "makeHists_" 

    dag_config = []
    condor_jobs = []

    histDetailStr = " --histDetailLevel allEvents.passPreSel.passEMuSel.passMuSel.allMeT.triggerStudy "

    picoOut = " -p None "
    histName = "hists.root"
    histOut = " --histFile "+histName+" "

    for y in years:

        for s in streams: 

            cmd = runCMD+" -i "+outputDir+"/fileLists/"+s+y+".txt"+ picoOut + " -o "+getOutDir()+ yearOpts[y]+  histDetailStr+  histOut
            cmd += " --doTrigEmulation "
            condor_jobs.append(makeCondorFile(cmd, "None", s+y, outputDir=outputDir, filePrefix=jobName))


        for tt in ttbarSamplesByYear[y]:
            
            cmd = runCMD+" -i "+outputDir+"/fileLists/"+tt+".txt" + picoOut + " -o "+getOutDir() + MCyearOpts(tt) +histDetailStr + histOut
            cmd += " --doTrigEmulation "
            if o.doTTbarPtReweight:
                cmd += " --doTTbarPtReweight "

            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName))                    


    dag_config.append(condor_jobs)


    #
    #  Hadd ttbar
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
        for tt in ttbarSamplesByYear[y]: 
            cmd += getOutDir()+"/"+tt+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y, outputDir=outputDir, filePrefix=jobName))

    dag_config.append(condor_jobs)

    #
    #   Hadd years
    #
    condor_jobs = []        

    mkdir(outputDir+"/TTRunII",   doRun)

    cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName+" "
    for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName+" "
    condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII", outputDir=outputDir, filePrefix=jobName))             


    for s in streams:
        mkdir(outputDir+"/"+s+"RunII",   doRun)
        cmd = "hadd -f "+getOutDir()+"/"+s+"RunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/"+s+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", s+"RunII", outputDir=outputDir, filePrefix=jobName))             

   
    dag_config.append(condor_jobs)            

    

    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
#
## 
##  Copy to AUTON
##
#if o.copyToAuton or o.makeAutonDirs or o.copyFromAuton:
#    
#    import os
#    autonAddr = "gpu13"
#    
#    
#    def run(cmd):
#        if doRun:
#            os.system(cmd)
#        else:
#            print cmd
#    
#    def runA(cmd):
#        print "> "+cmd
#        run("ssh "+autonAddr+" "+cmd)
#    
#    def scp(local, auton):
#        cmd = "scp "+local+" "+autonAddr+":hh4b/"+auton
#        print "> "+cmd
#        run(cmd)
#
#    def scpFrom(auton, local):
#        cmd = "scp "+autonAddr+":hh4b/"+auton+" "+local
#        print "> "+cmd
#        run(cmd)
#
#    def scpEOS(eosDir, subdir, pName, autonDir):
#
#        tempPath = "/uscms/home/jda102/nobackup/forSCP/"
#
#        cmd = "xrdcp "+eosDir+"/"+subdir+"/"+pName+"  "+tempPath+pName
#        run(cmd)
#
#        cmd = "scp "+tempPath+pName+" "+autonAddr+":hh4b/"+autonDir+"/"+subdir+"/"+pName
#        run(cmd)
#
#        cmd = "rm "+tempPath+pName
#        run(cmd)
#
#    
#    #
#    # Setup directories
#    #
#    if o.makeAutonDirs:
#
#        runA("mkdir hh4b/closureTests/UL")
#    
#        for y in years:
#            runA("mkdir hh4b/closureTests/UL/data"+y)
#    
#            for tt in ttbarSamplesByYear[y]:
#                runA("mkdir hh4b/closureTests/UL/"+tt)
#    
#    #
#    # Copy Files
#    #
#    if o.copyToAuton:
#        for y in years:
#            scpEOS(EOSOUTDIR,"data"+y,"picoAOD_3b.h5","hh4b/closureTests/UL")
#            scpEOS(EOSOUTDIR,"data"+y,"picoAOD_4b.h5","hh4b/closureTests/UL")
#            
#            for tt in ttbarSamplesByYear[y]:
#                scpEOS(EOSOUTDIR,tt,"picoAOD_3b.h5","hh4b/closureTests/UL")
#                scpEOS(EOSOUTDIR,tt,"picoAOD_4b.h5","hh4b/closureTests/UL")
#
#
#
#    #
#    # Copy Files
#    #
#    if o.copyFromAuton:
#        #pName = "picoAOD_3b_"+tagID+"_DvT3.h5"
#
#
#        for y in ["2018","2017","2016"]:
#            #pName = "picoAOD_3b_"+tagID+"_DvT3_with_rwTT.h5"
#            pName = "picoAOD_4b_"+tagID+"_DvT4_no_rwTT.h5"
#            scpFrom("closureTests/"+dirName+"/data"+y+"_"+tagID+"/"+pName, "closureTests/mixed/data"+y+"_"+tagID+"/"+pName)
#
#
#            for tt in ttbarSamples:
#                #pName = "picoAOD_3b_"+tagID+"_DvT3_with_rwTT.h5"
#
#                scpFrom("closureTests/"+dirName+"/"+tt+y+"_"+tagID+"/"+pName, "closureTests/mixed/"+tt+y+"_"+tagID+"/"+pName )
#                #scp("closureTests/mixed/"+tt+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5", "closureTests/"+dirName+"/"+tt+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5")
#
#
