
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse
from glob import glob

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9", help="Year or comma separated list of subsamples")
parser.add_option('--doWeightsNominal',               action="store_true", default=False, help="Fit jetCombinatoricModel and nJetClassifier TSpline")

parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--makeTarball',  action="store_true",      help="make Output file lists")

parser.add_option('--makeSkims',  action="store_true",      help="Make input skims")
parser.add_option('--haddChunks',  action="store_true",      help="Hadd chunks")
parser.add_option('--copySkims',  action="store_true",      help="Make input skims")

parser.add_option('--makeInputFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--inputsForDataVsTT',  action="store_true",      help="makeInputs for Dave Vs TTbar")
parser.add_option('--noConvert',  action="store_true",      help="")



parser.add_option('--makeAutonDirs', action="store_true",      help="Setup auton dirs")
parser.add_option('--copyToAuton', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyFromAuton', action="store_true",      help="copy h5 picos from Auton ")

parser.add_option('--writeOutDvTWeights',  action="store_true",      help=" ")

#parser.add_option('--noTT',       action="store_true",      help="Skip TTbar")
parser.add_option('-c',   '--condor',   action="store_true", default=False,           help="Run on condor")
parser.add_option(     '--doTTbarPtReweight',        action="store_true", help="boolean  to toggle using FvT reweight")
parser.add_option('--makeDvTFileLists',  action="store_true",      help="make Input file lists")

parser.add_option('--testDvTWeights',  action="store_true",      help="make Input file lists")
parser.add_option(     '--no3b',        action="store_true", help="boolean  to toggle using FvT reweight")
parser.add_option(     '--doDvTReweight',        action="store_true", help="boolean  to toggle using FvT reweight")

parser.add_option('--doWeightsQCD',     action="store_true",      help="")
parser.add_option('--doWeightsData',    action="store_true",      help="")


parser.add_option('--subSample3bQCD',  action="store_true",      help="Subsample 3b to look like 4b")
parser.add_option('--subSample3bData',  action="store_true",      help="Subsample 3b to look like 4b")

parser.add_option('--make4bHemisWithDvT',  action="store_true",      help="make 4b Hemisphere library ")
parser.add_option('--make4bHemiTarballDvT',  action="store_true",      help="make 4b Hemi Tarball  ")

parser.add_option('--makeInputFileListsSubSampledQCD',  action="store_true",      help="make file lists  ")

parser.add_option('--mixInputs',  action="store_true",      help="")
parser.add_option('--mixInputsDvT3',  action="store_true",      help="")
parser.add_option('--mixInputsDvT3DvT4',  action="store_true",      help="")

parser.add_option('--makeTTPseudoData',  action="store_true",      help="make PSeudo data  ")
parser.add_option('--makeTTPSDataFilesLists',  action="store_true",      help="make Input file lists")
parser.add_option('--checkPSData',  action="store_true",      help="")
parser.add_option('--checkOverlap',  action="store_true",      help="make Output file lists")

parser.add_option('--makeInputFileListsMixedData',  action="store_true",      help="make file lists  ")

parser.add_option('--histsForJCM',  action="store_true",      help="Make hist.root for JCM")
parser.add_option('--doWeightsMixed',    action="store_true",      help="")

parser.add_option('--addJCM', action="store_true",      help="Should be obvious")

o, a = parser.parse_args()

doRun = o.execute

from condorHelpers import *

CMSSW = getCMSSW()
USER = getUSER()
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/ZH4b/UL/"
TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"


outputDir="closureTests/UL/"


#ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]
signalSamples = ["ZZ4b","ZH4b","ggZH4b"]

years = o.year.split(",")
subSamples = o.subSamples.split(",")
mixedName=o.mixedName

def getOutDir():
    if o.condor:
        return EOSOUTDIR
    return outputDir

def run(cmd):
    if doRun: os.system(cmd)
    else:     print cmd



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

# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'
convertToH5JOB='python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'
convertToROOTWEIGHTFILE = 'python ZZ4b/nTupleAnalysis/scripts/convert_h52rootWeightFile.py'

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

            chunckedFiles = getFileChunks("data"+y+p)
            for ic, cf in enumerate(chunckedFiles):
                cmd = runCMD+"  -i "+cf+" -o "+getOutDir() +  yearOpts[y] + histConfig + picoOut + " -f "
                condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+p+"_c"+str(ic), outputDir=outputDir, filePrefix=jobName))


        #
        #  TTbar
        # 
        for tt in ttbarSamplesByYear[y]:
            chunckedFiles = getFileChunks(tt)
            for ic, cf in enumerate(chunckedFiles):
                cmd = runCMD+" -i "+cf+" -o "+getOutDir()+  MCyearOpts(tt) + histConfig + picoOut  + " -f "
                condor_jobs.append(makeCondorFile(cmd, "None", tt+"_c"+str(ic), outputDir=outputDir, filePrefix=jobName))                    


    
    #
    #  Hadd Chunks
    #
    # Do to put here


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

            cmdPico = "hadd -f "+getOutDir()+"/data"+y+p+"/picoAOD.root "
            cmdHist = "hadd -f "+getOutDir()+"/data"+y+p+"/histsFromNanoAOD.root "

            chunckedFiles = getFileChunks("data"+y+p)
            for ic, cf in enumerate(chunckedFiles):
                
                chIdx = ic + 1
                chunkName = str(chIdx) if chIdx > 9 else "0"+str(chIdx)
                cmdPico += getOutDir()+"/data"+y+p+"_chunk"+str(chunkName)+"/picoAOD.root "
                cmdHist += getOutDir()+"/data"+y+p+"_chunk"+str(chunkName)+"/histsFromNanoAOD.root "

            condor_jobs.append(makeCondorFile(cmdPico, "None", "data"+y+p+"_pico", outputDir=outputDir, filePrefix=jobName))
            condor_jobs.append(makeCondorFile(cmdHist, "None", "data"+y+p+"_hist", outputDir=outputDir, filePrefix=jobName))


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
    jobName = "inputsForDataVsTT_"

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
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+"4b_"))

        #
        #  3b
        #
        cmd = runCMD+" -i "+outputDir+"/fileLists/data"+y+".txt"+ picoOut3b + " -o "+getOutDir()+ yearOpts[y]+  histDetailStr+  histOut3b + " --skip4b "
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+"3b_"))

        for tt in ttbarSamplesByYear[y]:
            
            #
            # 4b
            #
            cmd = runCMD+" -i "+outputDir+"/fileLists/"+tt+".txt" + picoOut4b + " -o "+getOutDir() + MCyearOpts(tt) +histDetailStr + histOut4b + " --skip3b "
            if o.doTTbarPtReweight:
                cmd += " --doTTbarPtReweight "

            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"4b_"))                    

            #
            # 3b
            #
            cmd = runCMD+" -i "+outputDir+"/fileLists/"+tt+".txt" + picoOut3b + " -o "+getOutDir() + MCyearOpts(tt) +histDetailStr + histOut3b + " --skip4b "
            if o.doTTbarPtReweight:
                cmd += " --doTTbarPtReweight "

            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"3b_"))                    


    dag_config.append(condor_jobs)

    #
    #  Convert root to h5
    #
    if not o.noConvert:

        condor_jobs = []
        pico4b_h5 = "picoAOD_4b.h5"
        pico3b_h5 = "picoAOD_3b.h5"
    
        for y in years:
    
            cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"/"+pico4b+"  -o "+getOutDir()+"/data"+y+"/"+pico4b_h5
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+"convert_4b_"))
    
            cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"/"+pico3b+"  -o "+getOutDir()+"/data"+y+"/"+pico3b_h5
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+"convert_3b_"))
    
            
            for tt in ttbarSamplesByYear[y]:
                cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+"/"+pico4b+"  -o "+getOutDir()+"/"+tt+"/"+pico4b_h5
                condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"convert_4b_"))
    
                cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+"/"+pico3b+"  -o "+getOutDir()+"/"+tt+"/"+pico3b_h5
                condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"convert_3b_"))
    
        dag_config.append(condor_jobs)    
    

    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




# 
#  Copy to AUTON
#
if o.copyToAuton or o.makeAutonDirs or o.copyFromAuton:
    
    import os
    autonAddr = "gpu13"
    
    
    
    def runA(cmd):
        print "> "+cmd
        run("ssh "+autonAddr+" "+cmd)
    
    def scp(local, auton):
        cmd = "scp "+local+" "+autonAddr+":hh4b/"+auton
        print "> "+cmd
        run(cmd)

    def scpFromEOS(pName, autonPath, eosPath):

        tempPath = "/uscms/home/jda102/nobackup/forSCP/"

        localFile = tempPath+"/"+pName

        cmd = "scp "+autonAddr+":hh4b/"+autonPath+"/"+pName+" "+localFile
        print "> "+cmd
        run(cmd)

        cmd = "xrdcp -f "+localFile+" "+eosPath+"/"+pName
        run(cmd)

        cmd = "rm "+localFile
        run(cmd)

        

    def scpEOS(eosDir, subdir, pName, autonDir):

        tempPath = "/uscms/home/jda102/nobackup/forSCP/"

        cmd = "xrdcp "+eosDir+"/"+subdir+"/"+pName+"  "+tempPath+pName
        run(cmd)

        cmd = "scp "+tempPath+pName+" "+autonAddr+":"+autonDir+"/"+subdir+"/"+pName
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
        for tag in ["3b","4b"]:
            for y in years:
                scpEOS(EOSOUTDIR,"data"+y,"picoAOD_"+tag+".h5","hh4b/closureTests/UL")
            
                for tt in ttbarSamplesByYear[y]:
                    scpEOS(EOSOUTDIR,tt,"picoAOD_"+tag+".h5","hh4b/closureTests/UL")


    #
    # Copy Files
    #
    if o.copyFromAuton:
        for tag in [("3b","_DvT3"),("4b","_DvT4")]:
            DvTName = tag[1]
            tagName = tag[0]
            pName = "picoAOD_"+tagName+DvTName+".h5"

            for y in years:

                scpFromEOS(pName, "closureTests/UL/data"+y , EOSOUTDIR+"data"+y)

                for tt in ttbarSamplesByYear[y]:
                    scpFromEOS(pName,"closureTests/UL/"+tt, EOSOUTDIR+tt)







#
# Convert hdf5 to root
#
if o.writeOutDvTWeights: 

    dag_config = []
    condor_jobs = []
    jobName = "writeOutDvTWeights_"

    for tag in [("3b","DvT3",",_pt3"),("4b","DvT4",",_pt4")]:
        
        weightList = tag[2]#",_pt3"
    
        picoAOD_h5 = "picoAOD_"+tag[0]+"_"+tag[1]+".h5"
        picoAOD_root = "picoAOD_"+tag[0]+"_"+tag[1]+".root"

        for y in years:
            cmd = convertToROOTWEIGHTFILE+" -i "+getOutDir()+"/data"+y+"/"+picoAOD_h5+" --outFile "+getOutDir()+"/data"+y+"/"+picoAOD_root + " --classifierName "+tag[1]+"   --fvtNameList "+weightList
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+tag[1]+"_"))
    
    
            for tt in ttbarSamplesByYear[y]:
                cmd = convertToROOTWEIGHTFILE+" -i "+getOutDir()+"/"+tt+"/"+picoAOD_h5+" --outFile "+getOutDir()+"/"+tt+"/"+picoAOD_root +" --classifierName "+tag[1]+"      --fvtNameList "+weightList
                condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+tag[1]+"_"))
    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag",   doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)


    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)





#
# Make Input file lists
#
if o.makeDvTFileLists:
    


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    weightName = ""

    for tag in [("3b","DvT3",",_pt3"),("4b","DvT4",",_pt4")]:

        picoAOD_DvT_root = "picoAOD_"+tag[0]+"_"+tag[1]+".root"
        picoAOD_root     = "picoAOD_"+tag[0]+".root"

        for y in years:


            fileList = outputDir+"/fileLists/data"+y+"_"+tag[0]+".txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIR+"/data"+y+"/"+picoAOD_root+" >> "+fileList)

            fileList = outputDir+"/fileLists/data"+y+"_"+tag[0]+"_"+tag[1]+".txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIR+"/data"+y+"/"+picoAOD_DvT_root+" >> "+fileList)


            for tt in ttbarSamplesByYear[y]:

                fileList = outputDir+"/fileLists/"+tt+"_"+tag[0]+".txt"    
                run("rm "+fileList)
                run("echo "+EOSOUTDIR+"/"+tt+"/"+picoAOD_root+" >> "+fileList)

                fileList = outputDir+"/fileLists/"+tt+"_"+tag[0]+"_"+tag[1]+".txt"    
                run("rm "+fileList)
                run("echo "+EOSOUTDIR+"/"+tt+"/"+picoAOD_DvT_root+" >> "+fileList)




# 
#  Test DvT Weights
#
if o.testDvTWeights:
    # In the following "3b" refers to 3b subsampled to have the 4b statistics

    dag_config = []
    condor_jobs = []

    jobName = "testDvTWeights_"
    if o.doDvTReweight:
        jobName = "testDvTWeights_wDvT_"


    histDetail3b        = " --histDetailLevel allEvents.passPreSel.passMDRs.threeTag.failrWbW2.passMuon.passDvT05 "
    histDetail4b        = " --histDetailLevel allEvents.passPreSel.passMDRs.fourTag.failrWbW2.passMuon.passDvT05 "

    picoOut = " -p None " 


    tagList = []
    if not o.no3b:
        tagList.append( ("3b","DvT3","_pt3",histDetail3b))
    tagList.append( ("4b","DvT4","_pt4", histDetail4b) )

    for tag in tagList:

        histName = "hists_"+tag[0]+".root"
        if o.doDvTReweight:
            histName = "hists_"+tag[0]+"_rwDvT.root"


        histOut  = " --histFile "+histName
        histDetail = tag[3]

        for y in years:
        
            inputFile = " -i  "+outputDir+"/fileLists/data"+y+"_"+tag[0]+".txt "
            inputWeights = " --inputWeightFilesDvT "+outputDir+"/fileLists/data"+y+"_"+tag[0]+"_"+tag[1]+".txt "
            DvTName      = " --reweightDvTName weight_"+tag[1]+tag[2]

            cmd = runCMD+ inputFile + inputWeights + DvTName + picoOut + " -o "+getOutDir()+ yearOpts[y]+ histDetail +  histOut

            if o.doDvTReweight:  cmd += " --doDvTReweight "

            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))

            #
            # Only to ttbare if we are not doing the DvT Weighting
            #
            if not o.doDvTReweight:

                for tt in ttbarSamplesByYear[y]:
                
                    #
                    # 4b
                    #
                    inputFile = " -i  "+outputDir+"/fileLists/"+tt+"_"+tag[0]+".txt "
                    inputWeights = " --inputWeightFilesDvT "+outputDir+"/fileLists/"+tt+"_"+tag[0]+"_"+tag[1]+".txt "
    
                    cmd = runCMD+ inputFile + inputWeights + DvTName + picoOut + " -o "+getOutDir() + MCyearOpts(tt) +histDetail + histOut

                    condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))                    
    
    

    dag_config.append(condor_jobs)


    #
    #  Hadd ttbar
    #
    if not o.doDvTReweight:
        condor_jobs = []

        for tag in tagList:

            histName = "hists_"+tag[0]+".root"

            for y in years:
            
                cmd = "hadd -f "+ getOutDir()+"/TT"+y+"/"+histName+" "
                for tt in ttbarSamplesByYear[y]:        
                    cmd += getOutDir()+"/"+tt+"_"+tag[0]+"/"+histName+" "
                condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y, outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))            
    
    
        dag_config.append(condor_jobs)
        

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        for tag in tagList:

            histName = "hists_"+tag[0]+".root"

            #
            #  TTbar
            #
            if not o.doDvTReweight:

                cmd = "hadd -f " + getOutDir()+"/TTRunII/"+ histName+" "
                for y in years:
                    cmd += getOutDir()+"/TT"+y+"/"  +histName+" "
    
                condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII", outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))            


            if o.doDvTReweight:
                histName = "hists_"+tag[0]+"_rwDvT.root"
    
            #
            #  Data
            #
            cmd = "hadd -f " + getOutDir()+"/dataRunII/"+ histName+" "
            for y in years:
                cmd += getOutDir()+"/data"+y+"_"+tag[0]+"/"  +histName+" "

            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII", outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))            



        dag_config.append(condor_jobs)            



    #
    # Subtract QCD 
    #
    if not o.doDvTReweight:

        condor_jobs = []
    
        for tag in tagList:
            histName = "hists_"+tag[0]+".root"

            cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
            cmd += " -d "+getOutDir()+"/dataRunII/"+histName
            cmd += " --tt "+getOutDir()+"/TTRunII/"+histName
            cmd += " -q "+getOutDir()+"/QCDRunII/"+histName
            
            condor_jobs.append(makeCondorFile(cmd, getOutDir(), "QCDRunII", outputDir=outputDir, filePrefix=jobName+tag[0]+"_") )

    
        dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
if o.doWeightsQCD:

    cmds = []

    mkdir(outputDir+"/weights", doRun)


    #/store/user/jda102/condor/mixed/QCDRunII/
    dataFile3b = getOutDir()+"/QCDRunII/hists_3b.root"
    dataFile4b = getOutDir()+"/QCDRunII/hists_4b.root"

    cmd  = weightCMD
    cmd += " -d "+dataFile3b
    cmd += " --data4b "+dataFile4b
    cmd += " -c passPreSel   -o "+outputDir+"/weights/QCDRunII_PreSel/  -r SB -w 03-00-00"
    #cmd += " -c passMDRs   -o "+outputDir+"/weights/QCDRunII_MRDs/  -r SB -w 03-00-00"
    
    cmds.append(cmd)


    babySit(cmds, doRun)


#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
if o.doWeightsData:

    cmds = []

    mkdir(outputDir+"/weights", doRun)


    #/store/user/jda102/condor/mixed/QCDRunII/
    dataFile3b = getOutDir()+"/dataRunII/hists_3b.root"
    dataFile4b = getOutDir()+"/dataRunII/hists_4b.root"

    cmd  = weightCMD
    cmd += " -d "+dataFile3b
    cmd += " --data4b "+dataFile4b
    cmd += " -c passPreSel   -o "+outputDir+"/weights/dataRunII_PreSel/  -r SB -w 03-00-00"
    
    cmds.append(cmd)


    babySit(cmds, doRun)



# 
#  Make the 3b sample with the stats of the 4b sample
#
if o.subSample3bQCD:

    dag_config = []
    condor_jobs = []

    jobName = "subSample3bQCD_"

    for s in subSamples:

        for y in years:
            
            picoOut = " -p picoAOD_3bSubSampled_v"+s+".root "
            h10        = " --histDetailLevel allEvents.passPreSel.passMDRs.threeTag "
            histOut = " --histFile hists_v"+s+".root"

            inputFile = " -i  "+outputDir+"/fileLists/data"+y+"_3b.txt "
            inputWeights = " --inputWeightFilesDvT "+outputDir+"/fileLists/data"+y+"_3b_DvT3.txt "
            DvTName3b = " --reweightDvTName weight_DvT3_pt3 "

            cmd = runCMD + inputFile + inputWeights + DvTName3b + picoOut + " -o "+getOutDir()+ yearOpts[y]+  h10+  histOut + " -j closureTests/UL/weights/QCDRunII_PreSel/jetCombinatoricModel_SB_03-00-00.txt --emulate4bFrom3b --emulationOffset "+s
            cmd += " --doDvTReweight "
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


# 
#  Make the 3b sample with the stats of the 4b sample
#
if o.subSample3bData:

    dag_config = []
    condor_jobs = []

    jobName = "subSample3bData_"

    for s in subSamples:

        for y in years:
            
            picoOut = " -p picoAOD_3bSubSampled_v"+s+"_Data.root "
            h10        = " --histDetailLevel allEvents.passPreSel.passMDRs.threeTag "
            histOut = " --histFile hists_v"+s+"_Data.root"

            inputFile = " -i  "+outputDir+"/fileLists/data"+y+"_3b.txt "
            inputWeights = " --inputWeightFilesDvT "+outputDir+"/fileLists/data"+y+"_3b_DvT3.txt "
            DvTName3b = " --reweightDvTName weight_DvT3_pt3 "

            cmd = runCMD + inputFile + inputWeights + DvTName3b + picoOut + " -o "+getOutDir()+ yearOpts[y]+  h10+  histOut + " -j closureTests/UL/weights/dataRunII_PreSel/jetCombinatoricModel_SB_03-00-00.txt --emulate4bFrom3b --emulationOffset "+s
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)





#
# Make Hemisphere library from all hemispheres
#   (Should run locally)
if o.make4bHemisWithDvT:
    
    cmds = []
    logs = []

    jobName = "make4bHemisWithDvT_"

    picoOut = "  -p 'None' "
    histDetailLevel     = " --histDetailLevel allEvents.threeTag.fourTag "
    histOut = " --histFile hists.root " 

    for y in years:
        inputFile = " -i  "+outputDir+"/fileLists/data"+y+"_4b.txt "

        inputWeights = " --inputWeightFilesDvT "+outputDir+"/fileLists/data"+y+"_4b_DvT4.txt "
        DvTName4b = " --reweightDvTName weight_DvT4_pt4 "
        
        cmd = runCMD+ inputFile + inputWeights + DvTName4b + picoOut + " -o "+os.getcwd()+"/"+outputDir+"/dataHemisDvT " + yearOpts[y]+  histDetailLevel +  histOut + " --createHemisphereLibrary --doDvTReweight "

        cmds.append(cmd)
        logs.append(outputDir+"/log_"+jobName+y)

    babySit(cmds, doRun, logFiles=logs)



if o.make4bHemiTarballDvT:

    for y in years:

        tarballName = 'data'+y+'_hemisDvT.tgz'
        localTarball = outputDir+"/"+tarballName

        cmd  = 'tar -C '+outputDir+"/dataHemisDvT -zcvf "+ localTarball +' data'+y+"_4b"
        cmd += ' --exclude="hist*root"  '
        cmd += ' --exclude-vcs --exclude-caches-all'

        execute(cmd, doRun)
        cmd  = 'ls -hla '+localTarball
        execute(cmd, doRun)
        cmd = "xrdfs root://cmseos.fnal.gov/ mkdir /store/user/"+getUSER()+"/condor"
        execute(cmd, doRun)
        cmd = "xrdcp -f "+localTarball+ " root://cmseos.fnal.gov//store/user/"+getUSER()+"/condor/"+tarballName
        execute(cmd, doRun)





#
#   Make inputs fileLists
#
if o.makeInputFileListsSubSampledQCD:

    for s in subSamples:

        for y in years:

            fileList = outputDir+"/fileLists/data"+y+"_v"+s+".txt"    
            run("rm "+fileList)
            picoName = "picoAOD_3bSubSampled_v"+s+".root"
            run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+picoName+" >> "+fileList)


            fileList = outputDir+"/fileLists/data"+y+"_v"+s+"_Data.txt"    
            run("rm "+fileList)
            picoName = "picoAOD_3bSubSampled_v"+s+"_Data.root"
            run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+picoName+" >> "+fileList)




#
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
if o.mixInputs or o.mixInputsDvT3 or o.mixInputsDvT3DvT4:

    dag_config = []
    condor_jobs = []

    if o.mixInputs:
        jobName   = "mixInputs_"
        mixedName = "3bMix4b"
    if o.mixInputsDvT3:
        jobName = "mixInputsDvT3_"
        mixedName = "3bDvTMix4b"
    if o.mixInputsDvT3DvT4:    
        jobName = "mixInputsDvT3DvT4_"
        mixedName = "3bDvTMix4bDvT"

    for s in subSamples:

        for y in years:

            picoOut    = " -p picoAOD_"+mixedName+"_v"+s+".root "
            h10        = " --histDetailLevel passPreSel.passMDRs.threeTag.fourTag "
            histOut    = " --histFile hists_"+mixedName+"_v"+s+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 1000000 "
            hemiLoad   += '--inputHLib3Tag \\"NONE\\" --inputHLib4Tag \\"data'+y+'_4b/hemiSphereLib_4TagEvents_*root\\"'

            if o.mixInputsDvT3DvT4:
                hemiLoad += " --useHemiWeights "

            #
            #  Data
            #
            if o.mixInputs:
                inFileList = outputDir+"/fileLists/data"+y+"_v"+s+"_Data.txt"
            if o.mixInputsDvT3 or o.mixInputsDvT3DvT4:
                inFileList = outputDir+"/fileLists/data"+y+"_v"+s+".txt"

            # The --is3bMixed here just turns off blinding of the data
            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --is3bMixed "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", "data"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName, 
                                                        HEMINAME="data"+y+"_hemisDvT", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/data"+y+"_hemisDvT.tgz"))

    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)





#
#
#
if o.makeTTPseudoData:

    dag_config = []
    condor_jobs = []
    jobName = "makeTTPseudoData_"

    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
    picoOutMake = " -p picoAOD_PSData.root "
    picoOutRemove = " -p picoAOD_noPSData.root "

    histOutMake = " --histFile hists_PSData.root"
    histOutRemove = " --histFile hists_noPSData.root"

    #
    #  Make  ttbar PSData
    #
    for y in years:
       for tt in ttbarSamplesByYear[y]:
           cmd = runCMD+" -i "+outputDir+"/fileLists/"+tt+"_4b.txt "+ picoOutMake +" -o "+getOutDir()+ MCyearOpts(tt) + h10 + histOutMake +" --makePSDataFromMC --mcUnitWeight  "
           condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"PSData_"))

           cmd = runCMD+" -i "+outputDir+"/fileLists/"+tt+"_4b.txt "+ picoOutRemove +" -o "+getOutDir()+ MCyearOpts(tt) + h10 + histOutRemove +" --removePSDataFromMC  "
           condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"noPSData_"))
   

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)


    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.makeTTPSDataFilesLists:
    mkdir(outputDir+"/fileLists", doExecute=doRun)

    for y in years:

        for tt in ttbarSamplesByYear[y]:

            fileList = outputDir+"/fileLists/"+tt+"_4b_PSData.txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIR+"/"+tt+"_4b/picoAOD_PSData.root >> "+fileList)

            fileList = outputDir+"/fileLists/"+tt+"_4b_noPSData.txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIR+"/"+tt+"_4b/picoAOD_noPSData.root >> "+fileList)




if o.checkPSData:
    dag_config = []
    condor_jobs = []
    jobName = "checkPSData_"
    
    noPico    = " -p NONE "
    h10        = " --histDetailLevel allEvents.passMDRs.fourTag "


    histNameNoPSData = "hists_4b_noPSData.root"
    histNamePSData =   "hists_4b_PSData.root"
    histNameNom =      "hists_4b_nominal.root"

    for y in years:

        for tt in ttbarSamplesByYear[y]:

            # 
            # No PSData
            #
            fileListIn = " -i "+outputDir+"/fileLists/"+tt+"_4b_noPSData.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ noPico + MCyearOpts(tt) + h10 + " --histFile " + histNameNoPSData +"  --writeOutEventNumbers "
            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"noPS_"))

            # 
            # PSData
            #
            fileListIn = " -i "+outputDir+"/fileLists/"+tt+"_4b_PSData.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ noPico + yearOpts[y] + h10 + " --histFile " + histNamePSData +"  --is3bMixed --isDataMCMix --writeOutEventNumbers "
            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"PS_"))

            #
            #  Nominal
            #
            fileListIn = " -i "+outputDir+"/fileLists/"+tt+"_4b.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir() + noPico  + MCyearOpts(tt)+ h10 + " --histFile " + histNameNom +"  --writeOutEventNumbers "
            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"Nom_"))


    
    dag_config.append(condor_jobs)            


    #
    #  Hadd ttbar
    #
    condor_jobs = []

    for y in years:

        for h in [(histNameNoPSData,"_noPSData") , (histNamePSData,"_PSData"), (histNameNom,"" )]:
            cmd = "hadd -f "+ getOutDir()+"/TT"+y+"/"+h[0]+" "
            for tt in ttbarSamplesByYear[y]:        
                cmd += getOutDir()+"/"+tt+"_4b"+h[1]+"/"+h[0]+" "

            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+h[1], outputDir=outputDir, filePrefix=jobName))            

    dag_config.append(condor_jobs)


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        
    
        for h in [histNameNoPSData, histNamePSData, histNameNom]:
            cmd = "hadd -f " + getOutDir()+"/TTRunII/"+ h+" "
            for y in years:
                cmd += getOutDir()+"/TT"+y+"/"  +h+" "

            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_"+h.replace(".root",""), outputDir=outputDir, filePrefix=jobName))            


        dag_config.append(condor_jobs)            


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.checkOverlap:

    for y in years:
        for tt in ttbarSamplesByYear[y]:        
            cmd = "python ZZ4b/nTupleAnalysis/scripts/compEventCounts.py "
            cmd += " --file1 "+getOutDir()+tt+"_4b_noPSData/hists_4b_noPSData.root "
            cmd += " --file2 "+getOutDir()+tt+"_4b_PSData/hists_4b_PSData.root "

            execute(cmd, o.execute)

            
            cmd = "python ZZ4b/nTupleAnalysis/scripts/compEventCounts.py "
            cmd += " --file1 "+getOutDir()+tt+"_4b_noPSData/hists_4b_noPSData.root "
            cmd += " --file2 "+getOutDir()+tt+"_4b/hists_4b_nominal.root "
            
            execute(cmd, o.execute)


#
#   Make inputs fileLists
#
if o.makeInputFileListsMixedData:

    for m in ["3bDvTMix4bDvT"]:
    
        for s in subSamples:
    
            for y in years:
    
                fileList = outputDir+"/fileLists/mixed"+y+"_"+m+"_v"+s+".txt"    
                run("rm "+fileList)

                run("echo "+EOSOUTDIR+"/data"+y+"_v"+s+"/picoAOD_"+m+"_v"+s+".root >> "+fileList)

                for tt in ttbarSamplesByYear[y]:
                    run("echo "+EOSOUTDIR+"/"+tt+"_4b/picoAOD_PSData.root >> "+fileList)
    



#
#  Make Hists of mixed Datasets
#
if o.histsForJCM: 

    if mixedName not in ["3bDvTMix4bDvT"]:
        print "ERRROR mixedName=",mixedName,"Not valid"
        #import sys
        sys.exit(-1)

    #
    # Already have 3b hists and 4b TT
    #
    # Data and noPSDAta

    #
    #  Mixed data
    #
    for s in subSamples:

        dag_config = []
        condor_jobs = []
        jobName = "histsForJCM_"+mixedName+"_v"+s+"_"

        histName = "hists_"+mixedName+"_v"+s+".root "

        for y in years:

            picoOut    = " -p NONE "
            h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
            inFileList = outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_v"+s+".txt"
            histOut    = " --histFile "+histName

            # The --is3bMixed here just turns off blinding of the data
            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --is3bMixed --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y, outputDir=outputDir, filePrefix=jobName))
                                   

        dag_config.append(condor_jobs)


        #
        #   Hadd years
        #
        if "2016" in years and "2017" in years and "2018" in years:
    
            mkdir(outputDir+"/mixedRunII_"+mixedName,   doRun)
            condor_jobs = []        

            cmd = "hadd -f " + getOutDir()+"/mixedRunII_"+mixedName+"/"+ histName+" "
            for y in years:
                cmd += getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/"  +histName+" "

        condor_jobs.append(makeCondorFile(cmd, "None", "mixedRunII", outputDir=outputDir, filePrefix=jobName))            
        dag_config.append(condor_jobs)            


        execute("rm "+outputDir+jobName+"All.dag", doRun)
        execute("rm "+outputDir+jobName+"All.dag.*", doRun)

        dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)



#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
if o.doWeightsMixed:

    if mixedName not in ["3bDvTMix4bDvT"]:
        print "ERRROR mixedName=",mixedName,"Not valid"
        #import sys
        sys.exit(-1)


    cmds = []

    mkdir(outputDir+"/weights", doRun)

    dataFile3b = getOutDir()+"/dataRunII/hists_3b.root"
    ttbar3bFile = getOutDir()+"/TTRunII/hists_3b.root"

    ttbar4bFile = getOutDir()+"/TTRunII/hists_4b_noPSData.root"

    for s in subSamples:


        dataFile4b = getOutDir()+"/mixedRunII_"+mixedName+"/hists_"+mixedName+"_v"+s+".root"

        cmd  = weightCMD
        cmd += " -d "+dataFile3b
        cmd += " --data4b "+dataFile4b
        cmd += " --tt "+ttbar3bFile
        cmd += " --tt4b "+ttbar4bFile
        cmd += " -c passMDRs   -o "+outputDir+"/weights/mixedRunII_"+mixedName+"_v"+s+"/  -r SB -w 03-00-00"
    
        cmds.append(cmd)


    babySit(cmds, doRun)


#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
if o.doWeightsNominal:

    cmds = []

    mkdir(outputDir+"/weights", doRun)

    dataFile3b = getOutDir()+"/dataRunII/hists_3b.root"
    ttbar3bFile = getOutDir()+"/TTRunII/hists_3b.root"

    ttbar4bFile = getOutDir()+"/TTRunII/hists_4b.root"
    dataFile4b = getOutDir()+"/dataRunII/hists_4b.root"

    cmd  = weightCMD
    cmd += " -d "+dataFile3b
    cmd += " --data4b "+dataFile4b
    cmd += " --tt "+ttbar3bFile
    cmd += " --tt4b "+ttbar4bFile
    cmd += " -c passMDRs   -o "+outputDir+"/weights/dataRunII/  -r SB -w 03-00-00"
        
    cmds.append(cmd)


    babySit(cmds, doRun)




#
#  Get JCM Files
#    (Might be able to kill...)
jcmNameList="Nominal"
jcmFileList = {}

JCMTagNom = "03-00-00"
JCMTagMixed = "03-00-00"


for y in years:
    jcmFileList[y] = outputDir+"/weights/dataRunII/jetCombinatoricModel_SB_"+JCMTagNom+".txt"


for s in subSamples:
    jcmNameList   += ","+mixedName+"_v"+s
    for y in years:
        jcmFileList[y] += ","+outputDir+"/weights/mixedRunII_"+mixedName+"_v"+s+"/jetCombinatoricModel_SB_"+JCMTagMixed+".txt"



#
# Make picoAODS of 3b data with weights applied
#
if o.addJCM:


    dag_config = []
    condor_jobs = []
    
    jobName = "addJCM_"

    #
    #  3b Files
    #
    picoOut3b    = " -p picoAOD_3b_wJCM.root "
    histDetailLevel3b        = " --histDetailLevel allEvents.passMDRs.threeTag "

    histOut3b    = " --histFile hists_3b_wJCM.root "

    for y in years:

        fileListIn = " -i "+outputDir+"/fileLists/data"+y+"_3b.txt"
        cmd = runCMD+ fileListIn + " -o "+getOutDir() + picoOut3b + yearOpts[y] + histDetailLevel3b + histOut3b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip4b "
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+"3b_"))
        
        for tt in ttbarSamplesByYear[y]:

            fileListIn = " -i "+outputDir+"/fileLists/"+tt+"_3b.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ picoOut3b + MCyearOpts(tt) + histDetailLevel3b + histOut3b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip4b "
            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"3b_"))



    #
    #  4b Files
    #   (skim to only have 4b events in the pico ADO (Needed for training) )
    #
    picoOut4b    = " -p picoAOD_4b_wJCM.root "
    histOut4b    = " --histFile hists_4b_wJCM.root "
    histDetailLevel4b        = " --histDetailLevel allEvents.passMDRs.fourTag "

    for y in years:
        
        #
        #  Nominal 
        #
        fileListIn = " -i "+outputDir+"/fileLists/data"+y+"_4b.txt"
        cmd = runCMD+ fileListIn + " -o "+getOutDir() + picoOut4b + yearOpts[y] + histDetailLevel4b + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+"4b_"))
    
        for tt in ttbarSamplesByYear[y]:
    
            fileListIn = " -i "+outputDir+"/fileLists/"+tt+"_4b.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ picoOut4b + MCyearOpts(tt) + histDetailLevel4b + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "
            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"4b_"))

            fileListIn = " -i "+outputDir+"/fileLists/"+tt+"_4b_noPSData.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ picoOut4b + MCyearOpts(tt) + histDetailLevel4b + histOut4b + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b "
            condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"4b_noPSData_"))

    
        #
        #  Mixed Samples
        #
        for s in subSamples:

            fileListIn = " -i "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_v"+s+".txt"
            picoOutMixed = " -p picoAOD_"+mixedName+"_4b_wJCM_v"+s+".root "
            histOutMixed = " --histFile hists_"+mixedName+"_4b_wJCM_v"+s+".root"
            cmd = runCMD + fileListIn + " -o "+getOutDir() + picoOutMixed + yearOpts[y] + histDetailLevel4b + histOutMixed + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b --is3bMixed --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName))



    dag_config.append(condor_jobs)

    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)
