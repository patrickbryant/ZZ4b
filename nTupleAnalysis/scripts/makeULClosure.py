
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
parser.add_option('--histDetailStr',                    default="allEvents.passMDRs", help="Year or comma separated list of subsamples")

parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--makeTarball',  action="store_true",      help="make Output file lists")

parser.add_option('--makeSkims',  action="store_true",      help="Make input skims")
parser.add_option('--haddChunks',  action="store_true",      help="Hadd chunks")
parser.add_option('--copySkims',  action="store_true",      help="Make input skims")

parser.add_option('--makeInputFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--inputsForDataVsTT',  action="store_true",      help="makeInputs for Dave Vs TTbar")
parser.add_option('--noConvert',  action="store_true",      help="")
parser.add_option('--onlyConvert',  action="store_true",      help="")



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

parser.add_option('--make3bHemisWithDvT',  action="store_true",      help="make 4b Hemisphere library ")
parser.add_option('--make3bHemiTarballDvT',  action="store_true",      help="make 4b Hemi Tarball  ")


parser.add_option('--makeInputFileListsSubSampledQCD',  action="store_true",      help="make file lists  ")

parser.add_option('--mixInputs',  action="store_true",      help="")
parser.add_option('--mixInputsDvT3',  action="store_true",      help="")
parser.add_option('--mixInputsDvT3DvT4',  action="store_true",      help="")

parser.add_option('--mixInputs3b',  action="store_true",      help="")
parser.add_option('--mixInputs3bDvT3',  action="store_true",      help="")
parser.add_option('--mixInputs3bDvT3DvT3',  action="store_true",      help="")

parser.add_option('--convertMixedSamples',  action="store_true",      help="")

parser.add_option('--makeTTPseudoData',  action="store_true",      help="make PSeudo data  ")
parser.add_option('--makeTTPSDataFilesLists',  action="store_true",      help="make Input file lists")
parser.add_option('--checkPSData',  action="store_true",      help="")
parser.add_option('--checkOverlap',  action="store_true",      help="make Output file lists")

parser.add_option('--makeInputFileListsMixedData',  action="store_true",      help="make file lists  ")

parser.add_option('--histsForJCM',  action="store_true",      help="Make hist.root for JCM")
parser.add_option('--doWeightsMixed',    action="store_true",      help="")

parser.add_option('--addJCM', action="store_true",      help="Should be obvious")

parser.add_option('--makeAutonDirsForFvT', action="store_true",      help="Setup auton dirs")
parser.add_option('--copyToAutonForFvT', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyFromAutonForFvT', action="store_true",      help="copy h5 picos to Auton")

parser.add_option('--makeAutonDirsForMixedSamples', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyMixedSamplesToAuton', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyMixedSamplesFromAuton', action="store_true",      help="copy h5 picos to Auton")


parser.add_option('--writeOutSvBFvTWeights',  action="store_true",      help=" ")
parser.add_option('--makeInputFileListsSvBFvT',  action="store_true",      help=" ")

parser.add_option('--writeOutSvBFvTWeightsAllMixedSamples',  action="store_true",      help=" ")
parser.add_option('--makeInputFileListsSvBFvTAllMixedSamples',  action="store_true",      help=" ")

parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--plotsWithFvTVHH', action="store_true",      help="Make pdfs with FvT")

parser.add_option('--histsWithFvTAllMixedSamples', action="store_true",      help="Make hist.root with FvT")

parser.add_option('--makeInputsForCombine', action="store_true",      help="Make inputs for the combined tool")

parser.add_option('--plotsMixedVsNominal', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--plotsMixedVsNominalAllMixedSamples', action="store_true",      help="Make pdfs with FvT")

parser.add_option('--histsNoFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsNoFvT', action="store_true",      help="Make pdfs with FvT")

parser.add_option('--makeInputFileListsSignal',  action="store_true",      help="make Input file lists")
parser.add_option('--histsSignal',  action="store_true",      help="Make hist.root for JCM")

parser.add_option('--subSample3bSignal',  action="store_true",      help="Subsample 3b to look like 4b")
parser.add_option('--makeInputFileListsSignalSubSampled',  action="store_true",      help="make Input file lists")
parser.add_option('--histsSignal3bSubSamples',  action="store_true",      help="make Input file lists")

parser.add_option('--mixSignalDataHemis',  action="store_true",      help="make Input file lists")
parser.add_option('--makeInputFileListsSignalMixData',  action="store_true",      help="make Input file lists")
parser.add_option('--histsSignalMixData',  action="store_true",      help="make Input file lists")
parser.add_option('--convertSignalMixData',  action="store_true",      help="make Input file lists")

parser.add_option('--makeAutonDirsForSignalMixData', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copySignalMixDataToAuton', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copySignalMixDataFromAuton', action="store_true",      help="copy h5 picos to Auton")

parser.add_option('--writeOutSvBFvTWeightsSignalMixData',  action="store_true",      help=" ")


#
# PS Signal for Hemisphere
#
parser.add_option('--makeSignalPseudoData',  action="store_true",      help="make PSeudo data  ")
parser.add_option('--makeSignalPSFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--checkSignalPSData',  action="store_true",      help="make Output file lists")
parser.add_option('--makeHemisSignalOnly',  action="store_true",      help="make 4b Hemisphere library ")
parser.add_option('--makeHemiTarballSignal',  action="store_true",      help="make 4b Hemi Tarball  ")

parser.add_option('--mixSignalSignalHemis',  action="store_true",      help="make Input file lists")

parser.add_option('--makeHemisSignalAndData',  action="store_true",      help="make 4b Hemisphere library ")
parser.add_option('--makeHemiTarballSignalAndData',  action="store_true",      help="make 4b Hemi Tarball  ")
parser.add_option('--mixSignalAndData',  action="store_true",      help="make Input file lists")
parser.add_option('--mcHemiWeight',  default=1.0,      help="make Input file lists")

parser.add_option('--convertMixedSignalAndData',  action="store_true",      help="make Input file lists")
parser.add_option('--makeAutonDirsForMixedSignalAndData', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyMixedSignalAndDataToAuton', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyMixedSignalAndDataFromAuton', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--writeOutSvBFvTWeightsMixedSignalAndData',  action="store_true",      help=" ")

parser.add_option('--makeInputFileListsMixedSignalAndData',  action="store_true",      help="make Input file lists")
parser.add_option('--histsMixedSignalAndData',  action="store_true",      help="make Input file lists")

parser.add_option('--mix4bSignal',  action="store_true",      help="make Input file lists")
parser.add_option('--convertMixed4bSignal',  action="store_true",      help="make Input file lists")
parser.add_option('--makeAutonDirsForMixed4bSignal', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyMixed4bSignalToAuton', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--copyMixed4bSignalFromAuton', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--writeOutSvBFvTWeightsMixed4bSignal',  action="store_true",      help=" ")
parser.add_option('--makeInputFileListsMixed4bSignal',  action="store_true",      help="make Input file lists")
parser.add_option('--histsMixed4bSignal',  action="store_true",      help="make Input file lists")


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
__MCyearOpts["2016"]=yearOpts["2016"]+' --bTagSF -l 36.0e3 --isMC '
__MCyearOpts["2016_preVFP"]=yearOpts["2016"]+' --bTagSF -l 19.5e3 --isMC '
__MCyearOpts["2016_postVFP"]=yearOpts["2016"]+' --bTagSF -l 16.5e3 --isMC '


__MCyearOptsMu1000 = {}
__MCyearOptsMu1000["2018"]=yearOpts["2018"]+' --bTagSF -l 60000.0e3 --isMC '
__MCyearOptsMu1000["2017"]=yearOpts["2017"]+' --bTagSF -l 36000.7e3 --isMC '
__MCyearOptsMu1000["2016"]=yearOpts["2016"]+' --bTagSF -l 36000.0e3 --isMC '
__MCyearOptsMu1000["2016_preVFP"] =yearOpts["2016"]+' --bTagSF -l 19000.5e3 --isMC '
__MCyearOptsMu1000["2016_postVFP"]=yearOpts["2016"]+' --bTagSF -l 16000.5e3 --isMC '


def MCyearOpts(tt):
    for y in ["2018","2017","2016_preVFP","2016_postVFP"]:
        if not tt.find(y) == -1 :
            return __MCyearOpts[y]

def MCyearOptsSignal(a):
    for y in ["2018","2017","2016"]:
        if not a.find(y) == -1 :
            return __MCyearOpts[y]

def MCyearOptsSignalMu1000(a):
    for y in ["2018","2017","2016"]:
        if not a.find(y) == -1 :
            return __MCyearOptsMu1000[y]



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

plotOpts = {}
plotOpts["2018"]=" -l 60.0e3 -y 2018"
plotOpts["2017"]=" -l 36.7e3 -y 2017"
plotOpts["2016"]=" -l 35.9e3 -y 2016"
plotOpts["RunII"]=" -l 132.6e3 -y RunII"


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
 

    print "Needs updated with the skiming"
#
#    dag_config = []
#    condor_jobs = []
#    jobName = "writeOutDvTWeights_"
#
#    for tag in [("3b","DvT3",",_pt3"),("4b","DvT4",",_pt4")]:
#        
#        weightList = tag[2]#",_pt3"
#    
#        picoAOD_h5 = "picoAOD_"+tag[0]+"_"+tag[1]+".h5"
#        picoAOD_root = "picoAOD_"+tag[0]+"_"+tag[1]+".root"
#
#        for y in years:
#            cmd = convertToROOTWEIGHTFILE+" -i "+getOutDir()+"/data"+y+"/"+picoAOD_h5+" --outFile "+getOutDir()+"/data"+y+"/"+picoAOD_root + " --classifierName "+tag[1]+"   --fvtNameList "+weightList
#            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+tag[1]+"_"))
#    
#    
#            for tt in ttbarSamplesByYear[y]:
#                cmd = convertToROOTWEIGHTFILE+" -i "+getOutDir()+"/"+tt+"/"+picoAOD_h5+" --outFile "+getOutDir()+"/"+tt+"/"+picoAOD_root +" --classifierName "+tag[1]+"      --fvtNameList "+weightList
#                condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+tag[1]+"_"))
#    
#
#    dag_config.append(condor_jobs)
#
#
#    execute("rm "+outputDir+jobName+"All.dag",   doRun)
#    execute("rm "+outputDir+jobName+"All.dag.*", doRun)
#
#
#    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
#    cmd = "condor_submit_dag "+dag_file
#    execute(cmd, o.execute)
#




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



#
# Make Hemisphere library from all hemispheres
#   (Should run locally)
if o.make3bHemisWithDvT:
    
    cmds = []
    logs = []

    jobName = "make3bHemisWithDvT_"

    picoOut = "  -p 'None' "
    histDetailLevel     = " --histDetailLevel allEvents.threeTag.fourTag "
    histOut = " --histFile hists.root " 

    for y in years:
        inputFile = " -i  "+outputDir+"/fileLists/data"+y+"_3b.txt "

        inputWeights = " --inputWeightFilesDvT "+outputDir+"/fileLists/data"+y+"_3b_DvT3.txt "
        DvTName3b = " --reweightDvTName weight_DvT3_pt3 "
        
        cmd = runCMD+ inputFile + inputWeights + DvTName3b + picoOut + " -o "+os.getcwd()+"/"+outputDir+"/dataHemis3bDvT " + yearOpts[y]+  histDetailLevel +  histOut + " --createHemisphereLibrary --doDvTReweight "

        cmds.append(cmd)
        logs.append(outputDir+"/log_"+jobName+y)

    babySit(cmds, doRun, logFiles=logs)




if o.make3bHemiTarballDvT:

    for y in years:

        tarballName = 'data'+y+'_hemis3bDvT.tgz'
        localTarball = outputDir+"/"+tarballName

        cmd  = 'tar -C '+outputDir+"/dataHemis3bDvT -zcvf "+ localTarball +' data'+y+"_3b"
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

            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --unBlind "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", "data"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName, 
                                                        HEMINAME="data"+y+"_hemisDvT", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/data"+y+"_hemisDvT.tgz"))

    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)





#
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
if o.mixInputs3b or o.mixInputs3bDvT3 or o.mixInputs3bDvT3DvT3:

    dag_config = []
    condor_jobs = []

    if o.mixInputs3b:
        jobName   = "mixInputs3b_"
        mixedName = "3bMix3b"
    if o.mixInputs3bDvT3:
        jobName = "mixInputs3bDvT3_"
        mixedName = "3bDvTMix3b"
    if o.mixInputs3bDvT3DvT3:    
        jobName = "mixInputs3bDvT3DvT3_"
        mixedName = "3bDvTMix3bDvT"

    for s in subSamples:

        for y in years:

            picoOut    = " -p picoAOD_"+mixedName+"_v"+s+".root "
            h10        = " --histDetailLevel passPreSel.passMDRs.threeTag.fourTag "
            histOut    = " --histFile hists_"+mixedName+"_v"+s+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 50000 "
            hemiLoad   += '--inputHLib4Tag \\"data'+y+'_3b/hemiSphereLib_3TagEvents_*root\\" --inputHLib3Tag  \\"NONE\\"'

            if o.mixInputs3bDvT3DvT3:
                hemiLoad += " --useHemiWeights "

            #
            #  Data
            #
            if o.mixInputs3b:
                inFileList = outputDir+"/fileLists/data"+y+"_v"+s+"_Data.txt"
            if o.mixInputs3bDvT3 or o.mixInputs3bDvT3DvT3:
                inFileList = outputDir+"/fileLists/data"+y+"_v"+s+".txt"

            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --unBlind "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", "data"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName, 
                                                        HEMINAME="data"+y+"_hemis3bDvT", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/data"+y+"_hemis3bDvT.tgz"))

    

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
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ noPico + yearOpts[y] + h10 + " --histFile " + histNamePSData +"  --unBlind --isDataMCMix --writeOutEventNumbers "
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

    for mData in [("3bMix4b","_Data"),("3bDvTMix4b",""),("3bDvTMix4bDvT",""),
                  ("3bMix3b","_Data"),("3bDvTMix3b",""),("3bDvTMix3bDvT","")]:
    
    
        m = mData[0]

        for s in subSamples:
    
            for y in years:
    
                fileList = outputDir+"/fileLists/mixed"+y+"_"+m+"_v"+s+".txt"    
                run("rm "+fileList)

                run("echo "+EOSOUTDIR+"/data"+y+"_v"+s+mData[1]+"/picoAOD_"+m+"_v"+s+".root >> "+fileList)

                for tt in ttbarSamplesByYear[y]:
                    run("echo "+EOSOUTDIR+"/"+tt+"_4b/picoAOD_PSData.root >> "+fileList)


                fileList = outputDir+"/fileLists/data"+y+"_"+m+"_v"+s+".txt"    
                run("rm "+fileList)

                run("echo "+EOSOUTDIR+"/data"+y+"_v"+s+mData[1]+"/picoAOD_"+m+"_v"+s+".root >> "+fileList)
    



#
# Make picoAODS of 3b data with weights applied
#
if o.convertMixedSamples:


    dag_config = []
    condor_jobs = []
    
    jobName = "convertMixedSamples_"

    for mixedName in ["3bMix4b","3bDvTMix4b",
                      "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:

        
        #
        #  4b Files
        #   (skim to only have 4b events in the pico ADO (Needed for training) )
        #
        histDetailLevel4b        = " --histDetailLevel allEvents.passMDRs.fourTag "

        for y in years:
            
            #
            #  Mixed Samples
            #
            for s in subSamples:
    
                fileListIn = " -i "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_v"+s+".txt"
                picoOutMixed = " -p picoAOD_"+mixedName+"_4b_v"+s+".root "
                histOutMixed = " --histFile hists_"+mixedName+"_4b_v"+s+".root"
                cmd = runCMD + fileListIn + " -o "+getOutDir() + picoOutMixed + yearOpts[y] + histDetailLevel4b + histOutMixed + " --skip3b --unBlind --isDataMCMix "
                condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+"_"+mixedName+"_v"+s, outputDir=outputDir, filePrefix=jobName))
    

    #
    #  Making the root files
    #
    dag_config.append(condor_jobs)
    if o.onlyConvert: 
        dag_config = []


    #
    #  Convert the root files
    #
    if not o.noConvert:

        condor_jobs = []

        for mixedName in ["3bMix4b","3bDvTMix4b",
                          "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:

    
            for y in years:
    
                #
                # Mixed events
                #
                for s in subSamples:
                    #picoIn="picoAOD_"+mixedName+"_4b_v"+s+".root"
                    picoAOD="picoAOD_"+mixedName+"_4b_v"+s+".root"
                    picoAODH5="picoAOD_"+mixedName+"_4b_v"+s+".h5"
        
                    cmd = convertToH5JOB+" -i "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/"+picoAOD+"  -o "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/"+picoAODH5
                    condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+"_"+mixedName+"_v"+s, outputDir=outputDir, filePrefix=jobName+"convert_"))
            
    
        dag_config.append(condor_jobs)






    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



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

            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --unBlind --isDataMCMix "
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
            cmd = runCMD + fileListIn + " -o "+getOutDir() + picoOutMixed + yearOpts[y] + histDetailLevel4b + histOutMixed + " --jcmNameList "+jcmNameList+" --jcmFileList "+jcmFileList[y]+" --skip3b --unBlind --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName))


    #
    #  Making the root files
    #
    dag_config.append(condor_jobs)
    if o.onlyConvert: 
        dag_config = []


    #
    #  Convert the root files
    #
    if not o.noConvert:

        condor_jobs = []
        
    
        for y in years:

            #
            #  3b with JCM weights
            #
            picoAOD = "picoAOD_3b_wJCM.root"
            picoAODH5 = "picoAOD_3b_wJCM.h5"
    
            cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"_3b/"+picoAOD+"  -o "+getOutDir()+"/data"+y+"_3b/"+picoAODH5+"             --jcmNameList "+jcmNameList
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+"convert_3b_"))
    
            for tt in ttbarSamplesByYear[y]:
                cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+"_3b/"+picoAOD+"  -o "+getOutDir()+"/"+tt+"_3b/"+picoAODH5+"        --jcmNameList "+jcmNameList
                condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"convert_3b_"))


            #
            #  4b
            #
            picoAOD = "picoAOD_4b_wJCM.root"
            picoAODH5 = "picoAOD_4b_wJCM.h5"

            jcmName = "Nominal"
            cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"_4b/"+picoAOD+"  -o "+getOutDir()+"/data"+y+"_4b/"+picoAODH5+"             --jcmNameList "+jcmName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y, outputDir=outputDir, filePrefix=jobName+"convert_4b_"))
            
            for tt in ttbarSamplesByYear[y]:
                cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+"_4b/"+picoAOD+"  -o "+getOutDir()+"/"+tt+"_4b/"+picoAODH5+"          --jcmNameList "+jcmName
                condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"convert_4b_"))
    
                cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+"_4b_noPSData/"+picoAOD+"  -o "+getOutDir()+"/"+tt+"_4b_noPSData/"+picoAODH5+"          --jcmNameList "+jcmNameList
                condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+"convert_4b_noPSData_"))


            #
            # Mixed events
            #
            for s in subSamples:
                #picoIn="picoAOD_"+mixedName+"_4b_v"+s+".root"
                picoAOD="picoAOD_"+mixedName+"_4b_wJCM_v"+s+".root"
                picoAODH5="picoAOD_"+mixedName+"_4b_wJCM_v"+s+".h5"
                jcmName = mixedName+"_v"+s
    
                cmd = convertToH5JOB+" -i "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/"+picoAOD+"  -o "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/"+picoAODH5+"             --jcmNameList "+jcmName
                condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+"_"+mixedName+"_v"+s, outputDir=outputDir, filePrefix=jobName+"convert_"))
        
    
        dag_config.append(condor_jobs)






    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)





# 
#  Copy to AUTON
#
if o.copyFromAutonForFvT or o.copyToAutonForFvT or o.makeAutonDirsForFvT:
    
    import os
    autonAddr = "gpu13"
    
    #
    # Setup directories
    #
    if o.makeAutonDirsForFvT:

        runA("mkdir hh4b/closureTests/UL")
        for y in years:
            for tag in ["3b","4b"]:    
                runA("mkdir hh4b/closureTests/UL/data"+y+"_"+tag)
    
                for tt in ttbarSamplesByYear[y]:
                    runA("mkdir hh4b/closureTests/UL/"+tt+"_"+tag)

            for s in subSamples:
                runA("mkdir hh4b/closureTests/UL/mixed"+y+"_"+mixedName+"_v"+s)

            for tag in ["4b_noPSData"]:    
    
                for tt in ttbarSamplesByYear[y]:
                    runA("mkdir hh4b/closureTests/UL/"+tt+"_"+tag)



    #
    # Copy Files
    #
    if o.copyToAutonForFvT:
        for y in years:

            for tag in ["3b","4b"]:
                scpEOS(EOSOUTDIR,"data"+y+"_"+tag,"picoAOD_"+tag+"_wJCM.h5","hh4b/closureTests/UL")
            
                for tt in ttbarSamplesByYear[y]:
                    scpEOS(EOSOUTDIR,tt+"_"+tag,"picoAOD_"+tag+"_wJCM.h5","hh4b/closureTests/UL")



            for s in subSamples:
                scpEOS(EOSOUTDIR,"mixed"+y+"_"+mixedName+"_v"+s,"picoAOD_"+mixedName+"_4b_wJCM_v"+s+".h5","hh4b/closureTests/UL")                    

            for tag in ["4b_noPSData"]:    
    
                for tt in ttbarSamplesByYear[y]:
                    scpEOS(EOSOUTDIR,tt+"_"+tag,"picoAOD_4b_wJCM.h5","hh4b/closureTests/UL")                    



    #
    # Copy Files
    #
    if o.copyFromAutonForFvT:
        for y in years:

            for tag in ["3b","4b"]:
                scpFromEOS("picoAOD_"+tag+"_wJCM_SvB_FvT.h5", "closureTests/UL/data"+y+"_"+tag , EOSOUTDIR+"data"+y+"_"+tag)
            
                for tt in ttbarSamplesByYear[y]:
                    scpFromEOS("picoAOD_"+tag+"_wJCM_SvB_FvT.h5", "closureTests/UL/"+tt+"_"+tag, EOSOUTDIR+tt+"_"+tag)

            for s in subSamples:
                scpFromEOS("picoAOD_"+mixedName+"_4b_wJCM_v"+s+"_SvB_FvT.h5","closureTests/UL/mixed"+y+"_"+mixedName+"_v"+s,EOSOUTDIR+"mixed"+y+"_"+mixedName+"_v"+s)                    

            for tag in ["4b_noPSData"]:    
                for tt in ttbarSamplesByYear[y]:
                    scpFromEOS("picoAOD_4b_wJCM_SvB_FvT.h5", "closureTests/UL/"+tt+"_"+tag, EOSOUTDIR+tt+"_"+tag)





# 
#  Copy to AUTON
#
if o.copyMixedSamplesFromAuton or o.copyMixedSamplesToAuton or o.makeAutonDirsForMixedSamples:
    
    import os
    autonAddr = "gpu13"
    
    #
    # Setup directories
    #
    if o.makeAutonDirsForMixedSamples:

        for mixedName in ["3bMix4b","3bDvTMix4b",
                          "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:


            for y in years:

                for s in subSamples:
                    runA("mkdir hh4b/closureTests/UL/mixed"+y+"_"+mixedName+"_v"+s)


    #
    # Copy Files
    #
    if o.copyMixedSamplesToAuton:

        for mixedName in ["3bMix4b","3bDvTMix4b",
                          "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:


            for y in years:

                for s in subSamples:
                    scpEOS(EOSOUTDIR,"mixed"+y+"_"+mixedName+"_v"+s,"picoAOD_"+mixedName+"_4b_v"+s+".h5","hh4b/closureTests/UL")                    




    #
    # Copy Files
    #
    if o.copyMixedSamplesFromAuton:

        for mixedName in ["3bMix4b","3bDvTMix4b",
                          "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:

            for y in years:

                for s in subSamples:
                    scpFromEOS("picoAOD_"+mixedName+"_4b_v"+s+"_SvB_FvT.h5","closureTests/UL/mixed"+y+"_"+mixedName+"_v"+s,EOSOUTDIR+"mixed"+y+"_"+mixedName+"_v"+s)                    






if o.writeOutSvBFvTWeights: 

    dag_config = []
    condor_jobs = []
    jobName = "writeOutSvBFvTWeights_"


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
    ]

    def getFvTList(fvtName):
        return ["FvT"+fvtName,
                "FvT"+fvtName+"_pd3",
                "FvT"+fvtName+"_pt4",
                "FvT"+fvtName+"_pt3"
        ]

    #
    #  Make the var lists
    #
    varList3b = list(varListSvB) + getFvTList("_Nominal")
    for s in subSamples:
        fvtName = "_"+mixedName+"_v"+s
        varList3b += getFvTList(fvtName)
    varList3b += getFvTList("_"+mixedName+"_vAll")

    varList4b = list(varListSvB) + getFvTList("_Nominal")

    varList4b_noPS = list(varListSvB )
    for s in subSamples:
        fvtName = "_"+mixedName+"_v"+s
        varList4b_noPS += getFvTList(fvtName)
    varList4b_noPS += getFvTList("_"+mixedName+"_vAll")

    #
    #  Now convert
    #
    for y in years:
        for tag in [("3b",varList3b),("4b",varList4b)]:
            

            cmd = convertToROOTWEIGHTFILE 
            cmd += " --inFileH5 "+getOutDir()+"/data"+y+"_"+tag[0]+"/picoAOD_"+tag[0]+"_wJCM_SvB_FvT.h5"
            cmd += " --inFileROOT "+getOutDir()+"/data"+y+"_"+tag[0]+"/picoAOD_"+tag[0]+"_wJCM.root"
            cmd += " --outFile "+getOutDir()+"/data"+y+"_"+tag[0]+"/picoAOD_"+tag[0]+"_wJCM_SvB_FvT.root"
            cmd += " --varList "+",".join(tag[1])
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tag[0], outputDir=outputDir, filePrefix=jobName))


            for tt in ttbarSamplesByYear[y]:
                cmd = convertToROOTWEIGHTFILE 
                cmd += " --inFileH5 "+getOutDir()+"/"+tt+"_"+tag[0]+"/picoAOD_"+tag[0]+"_wJCM_SvB_FvT.h5"
                cmd += " --inFileROOT "+getOutDir()+"/"+tt+"_"+tag[0]+"/picoAOD_"+tag[0]+"_wJCM.root"
                cmd += " --outFile "+getOutDir()+"/"+tt+"_"+tag[0]+"/picoAOD_"+tag[0]+"_wJCM_SvB_FvT.root"
                cmd += " --varList "+",".join(tag[1])

                condor_jobs.append(makeCondorFile(cmd, "None", tt+"_"+tag[0], outputDir=outputDir, filePrefix=jobName))


        #
        # Mixed
        #
        for tt in ttbarSamplesByYear[y]:
            cmd = convertToROOTWEIGHTFILE 
            cmd += " --inFileH5 "+getOutDir()+"/"+tt+"_4b_noPSData/picoAOD_4b_wJCM_SvB_FvT.h5"
            cmd += " --inFileROOT "+getOutDir()+"/"+tt+"_4b_noPSData/picoAOD_4b_wJCM.root"
            cmd += " --outFile "+getOutDir()+"/"+tt+"_4b_noPSData/picoAOD_4b_wJCM_SvB_FvT.root"
            cmd += " --varList "+",".join(varList4b_noPS)
            condor_jobs.append(makeCondorFile(cmd, "None", tt+"_4b_noPSData", outputDir=outputDir, filePrefix=jobName))


        for s in subSamples:
            varListMixed = list(varListSvB)
            fvtName = "_"+mixedName+"_v"+s
            varListMixed += getFvTList(fvtName)
            varListMixed += getFvTList("_"+mixedName+"_vAll")

            cmd = convertToROOTWEIGHTFILE 
            cmd += " --inFileH5 "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/picoAOD_"+mixedName+"_4b_wJCM_v"+s+"_SvB_FvT.h5"
            cmd += " --inFileROOT "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/picoAOD_"+mixedName+"_4b_wJCM_v"+s+".root"
            cmd += " --outFile "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/picoAOD_"+mixedName+"_4b_wJCM_v"+s+"_SvB_FvT.root"
            cmd += " --varList "+",".join(varListMixed)
            condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+"_"+mixedName+"_v"+s, outputDir=outputDir, filePrefix=jobName))




    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag",   doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)


    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.writeOutSvBFvTWeightsAllMixedSamples: 

    dag_config = []
    condor_jobs = []
    jobName = "writeOutSvBFvTWeightsAllMixedSamples_"


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
    ]


    #
    #  Now convert
    #
    for mixedName in ["3bMix4b","3bDvTMix4b",
                      "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:

        for y in years:

            for s in subSamples:
                varListMixed = list(varListSvB)
    
                cmd = convertToROOTWEIGHTFILE 
                cmd += " --inFileH5 "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+"_SvB_FvT.h5"
                cmd += " --inFileROOT "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+".root"
                cmd += " --outFile "+getOutDir()+"/mixed"+y+"_"+mixedName+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+"_SvB_FvT.root"
                cmd += " --varList "+",".join(varListMixed)
                condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+"_"+mixedName+"_v"+s, outputDir=outputDir, filePrefix=jobName))
    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag",   doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)


    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#   Make inputs fileLists
#
if o.makeInputFileListsSvBFvT:

    for y in years:
    
        for fileName in ["wJCM","wJCM_SvB_FvT"]:
    
            for tag in ["3b","4b"]:
                
                fileList = outputDir+"/fileLists/data"+y+"_"+tag+"_"+fileName+".txt"    
                run("rm "+fileList)
                run("echo "+EOSOUTDIR+"/data"+y+"_"+tag+"/picoAOD_"+tag+"_"+fileName+".root >> "+fileList)
    
                for tt in ttbarSamplesByYear[y]:
        
                    fileList = outputDir+"/fileLists/"+tt+"_"+tag+"_"+fileName+".txt"    
                    run("rm "+fileList)
                    run("echo "+EOSOUTDIR+"/"+tt+"_"+tag+"/picoAOD_"+tag+"_"+fileName+".root >> "+fileList)
    
            #
            # Mixed
            #
            for tt in ttbarSamplesByYear[y]:
    
                fileList = outputDir+"/fileLists/"+tt+"_4b_noPSData_"+fileName+".txt"    
                run("rm "+fileList)
                run("echo "+EOSOUTDIR+"/"+tt+"_4b_noPSData/picoAOD_4b_"+fileName+".root >> "+fileList)
    
    
    
        for s in subSamples:
            for fileName in ["wJCM_v"+s,"wJCM_v"+s+"_SvB_FvT"]:
    
                fileList = outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_"+fileName+".txt"    
                run("rm "+fileList)
                run("echo "+EOSOUTDIR+"/mixed"+y+"_"+mixedName+"_v"+s+"/picoAOD_"+mixedName+"_4b_"+fileName+".root >> "+fileList)



#
#   Make inputs fileLists
#
if o.makeInputFileListsSvBFvTAllMixedSamples:

    for mixedName in ["3bMix4b","3bDvTMix4b",
                      "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:

        for y in years:
        
            for s in subSamples:

                for fileName in ["_v"+s,"_v"+s+"_SvB_FvT"]:
    
                    fileList = outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wSvB"+fileName+".txt"    
                    run("rm "+fileList)
                    run("echo "+EOSOUTDIR+"/mixed"+y+"_"+mixedName+"_v"+s+"/picoAOD_"+mixedName+"_4b"+fileName+".root >> "+fileList)
    




#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithFvT: 

    dag_config = []
    condor_jobs = []
    jobName = "histsWithFvT_"

    
    noPico = " -p NONE "
    hist3b        = " --histDetailLevel threeTag."+o.histDetailStr
    hist4b        = " --histDetailLevel fourTag."+o.histDetailStr
    outDir = " -o "+getOutDir()+" "


    for y in years:

        #
        # Nominal
        #

        JCMName="Nominal"
        FvTName="_Nominal"

        for tagData in [("3b",hist3b),("4b",hist4b)]:

            tag = tagData[0]
            histDetail = tagData[1]

            histName = "hists_"+tag+"_wFvT"+FvTName+".root"

            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM.txt "
            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM_SvB_FvT.txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +  yearOpts[y] + " --histFile "+histName + histDetail + " --jcmNameLoad "+JCMName+" -r --FvTName  FvT"+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            

            # 3b TTbar not needed 
            if tag == "4b":

                for tt in ttbarSamplesByYear[y]:
                    inputFile = " -i "+outputDir+"/fileLists/"+tt+"_"+tag+"_wJCM.txt "
                    inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/"+tt+"_"+tag+"_wJCM_SvB_FvT.txt"                

                    cmd = runCMD + inputFile + inputWeights + outDir + noPico  + MCyearOpts(tt) + " --histFile " + histName + histDetail  + " --jcmNameLoad "+JCMName+ " -r --FvTName FvT"+FvTName
                    condor_jobs.append(makeCondorFile(cmd, "None", tt+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            
        
        #
        #  SubSamples
        #
        for s in subSamples:

            JCMName=mixedName+"_v"+s
            FvTName="_"+mixedName+"_v"+s

            histName = "hists_wFvT"+FvTName+".root"    
    
            #
            # 3b
            #
            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_3b_wJCM.txt "
            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/data"+y+"_3b_wJCM_SvB_FvT.txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico + yearOpts[y] + " --histFile " + histName + hist3b + " --jcmNameLoad "+JCMName+ " -r --FvTName FvT"+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_3b"+FvTName, outputDir=outputDir, filePrefix=jobName))


            #
            # 4b
            #
            inputFile = " -i "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wJCM_v"+s+".txt"
            inputWeights = " --inputWeightFiles "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wJCM_v"+s+"_SvB_FvT.txt"

            cmd = runCMD + inputFile + inputWeights + outDir +  noPico + yearOpts[y] + " --histFile " + histName + hist4b + "  --FvTName FvT"+FvTName + " --unBlind  --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+FvTName, outputDir=outputDir, filePrefix=jobName))
            
            for tt in ttbarSamplesByYear[y]:

                histName = "hists_4b_noPSData_wFvT"+FvTName+".root"    
                inputFile = " -i "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wJCM.txt"
                inputWeights = " --inputWeightFiles "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wJCM_SvB_FvT.txt"
                
                cmd = runCMD + inputFile + inputWeights + outDir + noPico + MCyearOpts(tt)+ " --histFile " + histName + hist4b + "  --FvTName FvT"+FvTName
                condor_jobs.append(makeCondorFile(cmd, "None", tt+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))

        #
        #  vAll
        #
        JCMName=mixedName+"_v4"
        FvTName="_"+mixedName+"_vAll"
        histName = "hists_wFvT"+FvTName+"_oneFit.root"    
    
        #
        # 3b
        #
        inputFile = " -i "+outputDir+"/fileLists/data"+y+"_3b_wJCM.txt "
        inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/data"+y+"_3b_wJCM_SvB_FvT.txt"

        cmd = runCMD + inputFile + inputWeights + outDir + noPico + yearOpts[y] + " --histFile " + histName + hist3b + " --jcmNameLoad "+JCMName+ " -r --FvTName FvT"+FvTName
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_3b"+FvTName, outputDir=outputDir, filePrefix=jobName))

        #
        # 4b
        #
        for tt in ttbarSamplesByYear[y]:

            histName = "hists_4b_noPSData_wFvT"+FvTName+"_oneFit.root"    
            inputFile = " -i "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wJCM.txt"
            inputWeights = " --inputWeightFiles "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wJCM_SvB_FvT.txt"
                
            cmd = runCMD + inputFile + inputWeights + outDir + noPico + MCyearOpts(tt)+ " --histFile " + histName + hist4b + "  --FvTName FvT"+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", tt+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))



    dag_config.append(condor_jobs)

    #
    #  Hadd TTbar
    #
    condor_jobs = []

    for y in years:
        
        FvTName="_Nominal"
        histName = "hists_4b_wFvT"+FvTName+".root"

        cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
        for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_4b_wJCM/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_4b"+FvTName, outputDir=outputDir, filePrefix=jobName))

        for s in subSamples:

            FvTName="_"+mixedName+"_v"+s
            histName = "hists_4b_noPSData_wFvT"+FvTName+".root"    

            cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
            for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_4b_noPSData_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))


        FvTName="_"+mixedName+"_vAll"
        histName = "hists_4b_noPSData_wFvT"+FvTName+"_oneFit.root"    

        cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
        for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_4b_noPSData_wJCM/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))



    dag_config.append(condor_jobs)
    condor_jobs = []        

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/mixedRunII_"+mixedName, doRun)
        mkdir(outputDir+"/TTRunII",   doRun)
        

        #
        #  Nominal
        #
        for tag in ["3b","4b"]:

            FvTName="_Nominal"
            histName = "hists_"+tag+"_wFvT"+FvTName+".root"

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+tag+"_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))            

            if tag == "4b":
                cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName+" "
                for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName+" "
                condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_4b"+FvTName, outputDir=outputDir, filePrefix=jobName))            


        #
        #  Mixed
        #
        for s in subSamples:

            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wFvT"+FvTName+".root"    

            cmd = "hadd -f "+getOutDir()+"/mixedRunII_"+mixedName+"/"+histName+" "
            for y in years: cmd += getOutDir()+"/mixed"+y+"_"+mixedName+"_wJCM_v"+s+"/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "mixedRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_3b_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            

            histName = "hists_4b_noPSData_wFvT"+FvTName+".root"    
            cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))            


        FvTName="_"+mixedName+"_vAll"
        histName = "hists_wFvT"+FvTName+"_oneFit.root"    

        cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/data"+y+"_3b_wJCM/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            

        histName = "hists_4b_noPSData_wFvT"+FvTName+"_oneFit.root"    
        cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))            


        dag_config.append(condor_jobs)


    #
    #  Hadd SubSamples
    #
    condor_jobs = []

    histNameAll = "hists_wFvT_"+mixedName+"_vAll.root"    

    cmdData3b    = "hadd -f "+getOutDir()+"/dataRunII/"+histNameAll+" "
    cmdDataMixed = "hadd -f "+getOutDir()+"/mixedRunII/"+histNameAll+" "

    for s in subSamples:

        FvTName="_"+mixedName+"_v"+s
        histName = "hists_wFvT"+FvTName+".root"    

        cmdData3b    += getOutDir()+"/dataRunII/"+histName+" "
        cmdDataMixed += getOutDir()+"/mixedRunII_"+mixedName+"/"+histName+" "


    condor_jobs.append(makeCondorFile(cmdData3b,    "None", "dataRunII_vAll",  outputDir=outputDir, filePrefix=jobName))            
    condor_jobs.append(makeCondorFile(cmdDataMixed, "None", "mixedRunII_vAll", outputDir=outputDir, filePrefix=jobName))            
    dag_config.append(condor_jobs)

    #
    #  Scale SubSample
    #
    condor_jobs = []

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "

    cmd = cmdScale + " -i "+getOutDir()+"/dataRunII/"+histNameAll+" "
    condor_jobs.append(makeCondorFile(cmd, getOutDir(), "dataRunII", outputDir=outputDir, filePrefix=jobName+"scale_"))            

    cmd = cmdScale + " -i "+getOutDir()+"/mixedRunII/"+histNameAll+" "
    condor_jobs.append(makeCondorFile(cmd, getOutDir(), "mixedRunII", outputDir=outputDir, filePrefix=jobName+"scale_"))            

    dag_config.append(condor_jobs)



    

    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithFvTAllMixedSamples: 

    dag_config = []
    condor_jobs = []
    jobName = "histsWithFvTAllMixedSamples_"

    
    noPico = " -p NONE "
    hist4b        = " --histDetailLevel fourTag."+o.histDetailStr
    outDir = " -o "+getOutDir()+" "

    for mixedName in ["3bMix4b","3bDvTMix4b",
                      "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:


        for y in years:
    
            
            #
            #  SubSamples
            #
            for s in subSamples:

                FvTName="_"+mixedName+"_v"+s
                histName = "hists_wFvT"+FvTName+".root"    
        
                #
                # 4b
                #
                inputFile = " -i "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wSvB_v"+s+".txt"
                inputWeights = " --inputWeightFiles "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wSvB_v"+s+"_SvB_FvT.txt"
    
                cmd = runCMD + inputFile + inputWeights + outDir +  noPico + yearOpts[y] + " --histFile " + histName + hist4b + " --unBlind  --isDataMCMix "
                condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+FvTName, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)
    condor_jobs = []        

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        for mixedName in ["3bMix4b","3bDvTMix4b",
                          "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:


            mkdir(outputDir+"/mixedRunII_"+mixedName, doRun)
            
    
            #
            #  Mixed
            #
            for s in subSamples:
    
                FvTName="_"+mixedName+"_v"+s
                histName = "hists_wFvT"+FvTName+".root"    
    
                cmd = "hadd -f "+getOutDir()+"/mixedRunII_"+mixedName+"/"+histName+" "
                for y in years: cmd += getOutDir()+"/mixed"+y+"_"+mixedName+"_wSvB_v"+s+"/"+histName+" "
                condor_jobs.append(makeCondorFile(cmd, "None", "mixedRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            
    
    
            dag_config.append(condor_jobs)
    

    #
    #  Hadd SubSamples
    #
    condor_jobs = []

    for mixedName in ["3bMix4b","3bDvTMix4b",
                      "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:


        histNameAll = "hists_wFvT_"+mixedName+"_vAll.root"    
        cmdDataMixed = "hadd -f "+getOutDir()+"/mixedRunII/"+histNameAll+" "

        for s in subSamples:
    
            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wFvT"+FvTName+".root"    
    
            cmdDataMixed += getOutDir()+"/mixedRunII_"+mixedName+"/"+histName+" "
    
        condor_jobs.append(makeCondorFile(cmdDataMixed, "None", "mixedRunII_"+mixedName+"_vAll", outputDir=outputDir, filePrefix=jobName))            
        dag_config.append(condor_jobs)
    
    #
    #  Scale SubSample
    #
    condor_jobs = []

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "

    for mixedName in ["3bMix4b","3bDvTMix4b",
                      "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:

        histNameAll = "hists_wFvT_"+mixedName+"_vAll.root"    
        cmd = cmdScale + " -i "+getOutDir()+"/mixedRunII/"+histNameAll+" "

        condor_jobs.append(makeCondorFile(cmd, getOutDir(), "mixedRunII_"+mixedName, outputDir=outputDir, filePrefix=jobName+"scale_"))            

    dag_config.append(condor_jobs)
    

    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)





#
#  Make Plots with FvT
#
if o.plotsWithFvT:
    cmds = []

    histDetailLevel = "passMDRs,passMjjOth,passSvB,fourTag,SB,CR,SRNoHH,HHSR"

    for y in ["RunII"]:

        #
        #  Nominal
        #
        FvTName = "_Nominal"
        
        data3bFile  = getOutDir()+"/data"+y+"/hists_3b_wFvT"+FvTName+".root"
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wFvT"+FvTName+".root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+".root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+FvTName+plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)


        #
        #  Mixed Samples Combined
        #
        data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+".root"
        data3bFile  = getOutDir()+"/data"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+"_vAll_"+mixedName + plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)

        #
        #  Mixed Samples Combined vs OneFit
        #
        data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+".root"
        data3bFile  = getOutDir()+"/data"+y+"/hists_wFvT_"+mixedName+"_vAll_oneFit.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+"_vAll_oneFit_"+mixedName + plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)



        for s in subSamples:

            #
            #  Mixed 
            #
            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wFvT"+FvTName+".root"    

            data3bFile  = getOutDir()+"/data"+y+"/"+histName
            data4bFile  = getOutDir()+"/mixed"+y+"_"+mixedName+"/"+histName
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+".root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+FvTName + plotOpts[y]+" -m -j -r --noSignal "
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)


            #
            #
            #
            data3bFile  = getOutDir()+"/data"+y+"/"+histName
            data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+".root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+"_vAll_"+mixedName+"_vs_v"+s + plotOpts[y]+" -m -j -r --noSignal "
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)

    babySit(cmds, doRun)

    cmds = []

    for y in ["RunII"]:
        FvTName = "_Nominal"
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+FvTName+".tar plotsWithFvT_"+y+FvTName)

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+"_vAll_"+mixedName+".tar plotsWithFvT_"+y+"_vAll_"+mixedName)

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+"_vAll_oneFit_"+mixedName+".tar plotsWithFvT_"+y+"_vAll_oneFit_"+mixedName)

        for s in subSamples:
            FvTName="_"+mixedName+"_v"+s

            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+FvTName+".tar plotsWithFvT_"+y+FvTName)
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+"_vAll_"+mixedName+"_vs_v"+s+".tar plotsWithFvT_"+y+"_vAll_"+mixedName+"_vs_v"+s)



    
    babySit(cmds, doRun)    



#
#  Make Plots with FvT
#
if o.plotsWithFvTVHH:
    cmds = []

    histDetailLevel = "passMDRs,fourTag,SB,HHSR,passMjjOth"

    for y in ["RunII"]:

        #
        #  Nominal
        #
        FvTName = "_Nominal"
        
        data3bFile  = getOutDir()+"/data"+y+"/hists_3b_wFvT"+FvTName+".root"
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wFvT"+FvTName+".root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+".root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_VHH_"+y+FvTName+plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)


        #
        #  Mixed Samples Combined
        #
        data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+".root"
        data3bFile  = getOutDir()+"/data"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_VHH_"+y+"_vAll_"+mixedName + plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)


        for s in subSamples:

            #
            #  Mixed 
            #
            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wFvT"+FvTName+".root"    

            data3bFile  = getOutDir()+"/data"+y+"/"+histName
            data4bFile  = getOutDir()+"/mixed"+y+"_"+mixedName+"/"+histName
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+".root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_VHH_"+y+FvTName + plotOpts[y]+" -m -j -r --noSignal "
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)


            #
            #
            #
            data3bFile  = getOutDir()+"/data"+y+"/"+histName
            data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+".root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_VHH_"+y+"_vAll_"+mixedName+"_vs_v"+s + plotOpts[y]+" -m -j -r --noSignal "
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)

    babySit(cmds, doRun)

    cmds = []

    for y in ["RunII"]:
        FvTName = "_Nominal"
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_VHH_"+y+FvTName+".tar plotsWithFvT_VHH_"+y+FvTName)

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_VHH_"+y+"_vAll_"+mixedName+".tar plotsWithFvT_VHH_"+y+"_vAll_"+mixedName)

        for s in subSamples:
            FvTName="_"+mixedName+"_v"+s

            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_VHH_"+y+FvTName+".tar plotsWithFvT_VHH_"+y+FvTName)
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_VHH_"+y+"_vAll_"+mixedName+"_vs_v"+s+".tar plotsWithFvT_VHH_"+y+"_vAll_"+mixedName+"_vs_v"+s)



    
    babySit(cmds, doRun)    
    


#
#  Make Plots with FvT
#
if o.plotsNoFvT:
    cmds = []

    histDetailLevel = "passMDRs,passMjjOth,passSvB,fourTag,SB,CR,SRNoHH,HHSR"
    #histDetailLevel = "passMDRs,fourTag,SB"

    for y in ["RunII"]:

        #
        #  Nominal
        #
        FvTName = "_Nominal"
        
        qcdFile  = getOutDir()+"/QCD"+y+"/hists_3b_noFvT"+FvTName+".root"
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wFvT"+FvTName+".root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+".root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+y+FvTName+plotOpts[y]+" -m -j --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --qcd "+qcdFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)


        #
        #  Nominal Overllaaid with Mixed
        #
        mixedFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_Nominal_vs_"+mixedName+plotOpts[y]+"  -m -j --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --qcd "+qcdFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmd += " --mixedName  " + mixedName 
        cmd += " --mixedSamples " + mixedFile
        cmd += " --mixedSamplesDen " + data4bFile
        cmds.append(cmd)
        




        #
        #  Mixed Samples Combined
        #
        data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+".root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+y+"_vAll_"+mixedName + plotOpts[y]+" -m -j --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --qcd "+qcdFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)



        for s in subSamples:

            #
            #  Mixed 
            #
            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wFvT"+FvTName+".root"    

            data4bFile  = getOutDir()+"/mixed"+y+"_"+mixedName+"/"+histName
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+".root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+y+FvTName + plotOpts[y]+" -m -j  --noSignal "
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --qcd "+qcdFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)


            #
            #
            #
            data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+".root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+y+"_vAll_"+mixedName+"_vs_v"+s + plotOpts[y]+" -m -j  --noSignal "
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --qcd "+qcdFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)

    babySit(cmds, doRun)

    cmds = []

    for y in ["RunII"]:
        FvTName = "_Nominal"
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+y+FvTName+".tar plotsNoFvT_"+y+FvTName)

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_Nominal_vs_"+mixedName+".tar plotsNoFvT_Nominal_vs_"+mixedName)

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+y+"_vAll_"+mixedName+".tar plotsNoFvT_"+y+"_vAll_"+mixedName)




        for s in subSamples:
            FvTName="_"+mixedName+"_v"+s

            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+y+FvTName+".tar plotsNoFvT_"+y+FvTName)
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+y+"_vAll_"+mixedName+"_vs_v"+s+".tar plotsNoFvT_"+y+"_vAll_"+mixedName+"_vs_v"+s)



    
    babySit(cmds, doRun)    





if o.makeInputsForCombine:

    import ROOT

    def getHistForCombine(in_File,tag,proc,outName,region):
        hist = in_File.Get("passMDRs/"+tag+"/mainView/"+region+"/SvB_ps_"+proc).Clone()
        hist.SetName(outName)
        return hist


    def makeInputsForRegion(region, noFvT=False):
        
        noFvT = False

        if noFvT:
            outFile = ROOT.TFile(outputDir+"/hists_closure_"+mixedName+"_"+region+"_noFvT.root","RECREATE")
        else:
            outFile = ROOT.TFile(outputDir+"/hists_closure_"+mixedName+"_"+region+".root","RECREATE")
    
        procs = ["zz","zh"]
        
        for s in subSamples: 
            
            #
            #  "+tagID+" with combined JCM 
            #
            #weightPostFix = "_comb"
            #tagName = "_"+tagID
            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wFvT"+FvTName+".root"    
            
            histName3b = "hists_wFvT"+FvTName+".root"    
            histName4b = "hists_wFvT"+FvTName+".root"    
            histName4bTT = "hists_4b_noPSData_wFvT"+FvTName+".root" 
            
            sampleDir = outFile.mkdir(mixedName+"_v"+s)

            multiJet_Files = []
            data_obs_Files = []
            ttbar_Files    = []

            multiJet_Hists = {}
            data_obs_Hists = {}
            ttbar_Hists    = {}

            for p in procs:
                multiJet_Hists[p] = []
                data_obs_Hists[p] = []
                ttbar_Hists   [p] = []

            for y in years:
    
                multiJet_Files .append(ROOT.TFile.Open(getOutDir()+"/data"+y+"_3b_wJCM/"+histName3b))
                data_obs_Files .append(ROOT.TFile.Open(getOutDir()+"/mixed"+y+"_"+mixedName+"_wJCM_v"+s+"/"+histName4b))
                ttbar_Files    .append(ROOT.TFile.Open(getOutDir()+"/TT"+y+"/"+histName4bTT))
        
                for p in procs:
        
                    multiJet_Hists[p].append( getHistForCombine(multiJet_Files[-1],"threeTag",p,"multijet", region) )
                    data_obs_Hists[p].append( getHistForCombine(data_obs_Files[-1],"fourTag",p, "data_obs", region) )
                    ttbar_Hists[p]   .append( getHistForCombine(ttbar_Files[-1],   "fourTag",p, "ttbar",    region) )
    
                    sampleDir.cd()
                    procDir = sampleDir.mkdir(p+y)
                    procDir.cd()
                    
                    #multiJet_Hist.SetDirectory(procDir)
                    multiJet_Hists[p][-1].Write()
                    data_obs_Hists[p][-1].Write()
                    ttbar_Hists   [p][-1].Write()
                    


            # Combined Run2
            for p in procs:
                
                multiJet_HistRunII = multiJet_Hists[p][0].Clone()
                data_obs_HistRunII = data_obs_Hists[p][0].Clone()
                ttbar_HistRunII    = ttbar_Hists   [p][0].Clone()

                for i in [1,2]:
                    multiJet_HistRunII.Add(multiJet_Hists[p][i])
                    data_obs_HistRunII.Add(data_obs_Hists[p][i])
                    ttbar_HistRunII   .Add(ttbar_Hists   [p][i])
                    
                
                sampleDir.cd()
                procDir = sampleDir.mkdir(p+"RunII")
                procDir.cd()

                #multiJet_Hist.SetDirectory(procDir)
                multiJet_HistRunII.Write()
                data_obs_HistRunII.Write()
                ttbar_HistRunII   .Write()



        #
        #  vAll
        #
        sampleDir = outFile.mkdir(mixedName+"_vAll_oneFit")

        FvTName="_"+mixedName+"_vAll"
        histName3b = "hists_wFvT"+FvTName+"_oneFit.root"    

        multiJet_Files = []
        multiJet_Hists = {}

        for p in procs:
            multiJet_Hists[p] = []

        for y in years:

            multiJet_Files .append(ROOT.TFile.Open(getOutDir()+"/data"+y+"_3b_wJCM/"+histName3b))
    
            for p in procs:
    
                multiJet_Hists[p].append( getHistForCombine(multiJet_Files[-1],"threeTag",p,"multijet", region) )

                sampleDir.cd()
                procDir = sampleDir.mkdir(p+y)
                procDir.cd()
                
                #multiJet_Hist.SetDirectory(procDir)
                multiJet_Hists[p][-1].Write()


        # Combined Run2
        for p in procs:
            
            multiJet_HistRunII = multiJet_Hists[p][0].Clone()

            for i in [1,2]:
                multiJet_HistRunII.Add(multiJet_Hists[p][i])
            
            sampleDir.cd()
            procDir = sampleDir.mkdir(p+"RunII")
            procDir.cd()

            #multiJet_Hist.SetDirectory(procDir)
            multiJet_HistRunII.Write()


    makeInputsForRegion("SR")
    makeInputsForRegion("SRNoHH")
    makeInputsForRegion("CR")
    makeInputsForRegion("SB")

    makeInputsForRegion("SR",noFvT=True)
    makeInputsForRegion("SRNoHH",noFvT=True)
    makeInputsForRegion("CR",noFvT=True)
    makeInputsForRegion("SB",noFvT=True)



#
#  Make Plots with FvT
#
if o.plotsMixedVsNominal:
    cmds = []

    histDetailLevel = "passMDRs,mixedVsData,SB,CR,passMjjOth"
    #histDetailLevel = "passMDRs,mixedVsData,SB"


    for y in ["RunII"]:
             
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wFvT_Nominal.root"
        mixedFile   = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_vAll_scaled.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+"_vAll_"+mixedName+"_vs_Nominal "+plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data "+data4bFile

        cmd += " --mixedSamples "+mixedFile
        cmd += " --mixedName "+"Data"
        cmd += " --mixedSamplesDen "+data4bFile
        cmds.append(cmd)


    babySit(cmds, doRun)


    cmds = []

    for y in ["RunII"]:

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+"_vAll_"+mixedName+"_vs_Nominal.tar plotsWithFvT_"+y+"_vAll_"+mixedName+"_vs_Nominal")

    
    babySit(cmds, doRun)    



#
#  Make Plots with FvT
#
if o.plotsMixedVsNominalAllMixedSamples:
    cmds = []

    histDetailLevel = "passMDRs,mixedVsData,SB,CR,passMjjOth,SRNoHH"


    for y in ["RunII"]:
             
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wFvT_Nominal.root"

        mixedFile3bDvTMix4bDvT   = getOutDir()+"/mixed"+y+"/hists_wFvT_3bDvTMix4bDvT_vAll_scaled.root"
        mixedFile3bDvTMix4b      = getOutDir()+"/mixed"+y+"_3bDvTMix4b/hists_wFvT_3bDvTMix4b_vAll_scaled.root"
        mixedFile3bMix4b         = getOutDir()+"/mixed"+y+"_3bMix4b/hists_wFvT_3bMix4b_vAll_scaled.root"

        mixedFile3bDvTMix3bDvT   = getOutDir()+"/mixed"+y+"_3bDvTMix3bDvT/hists_wFvT_3bDvTMix3bDvT_vAll_scaled.root"
        mixedFile3bDvTMix3b      = getOutDir()+"/mixed"+y+"_3bDvTMix3b/hists_wFvT_3bDvTMix3b_vAll_scaled.root"
        mixedFile3bMix3b         = getOutDir()+"/mixed"+y+"_3bMix3b/hists_wFvT_3bMix3b_vAll_scaled.root"


        #
        # 3bMix4b
        #
        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+"_vAll_MixedStudy_3bMix4b "+plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data "+data4bFile
        cmd += " --mixedSamples "+mixedFile3bDvTMix4bDvT+","+mixedFile3bDvTMix4b+","+mixedFile3bMix4b
        cmd += " --mixedName 3bDvTMix4bDvT,3bDvTMix4b,3bMix4b"
        cmd += " --mixedSamplesDen "+data4bFile+","+data4bFile+","+data4bFile
        cmds.append(cmd)


        #
        # 3bMix3b
        #
        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+"_vAll_MixedStudy_3bMix3b "+plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data "+data4bFile
        cmd += " --mixedSamples "+mixedFile3bDvTMix3bDvT+","+mixedFile3bDvTMix3b+","+mixedFile3bMix3b
        cmd += " --mixedName 3bDvTMix3bDvT,3bDvTMix3b,3bMix3b"
        cmd += " --mixedSamplesDen "+data4bFile+","+data4bFile+","+data4bFile
        cmds.append(cmd)


        #
        # 3bMix3b vs 3bMix4b
        #
        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+"_vAll_MixedStudy_3bVs4b "+plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data "+data4bFile
        cmd += " --mixedSamples "+mixedFile3bDvTMix4bDvT+","+mixedFile3bDvTMix3bDvT
        cmd += " --mixedName 3bDvTMix4bDvT,3bDvTMix3bDvT"
        cmd += " --mixedSamplesDen "+data4bFile+","+data4bFile
        cmds.append(cmd)




    babySit(cmds, doRun)


    cmds = []

    for y in ["RunII"]:

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+"_vAll_MixedStudy_3bMix4b.tar plotsWithFvT_"+y+"_vAll_MixedStudy_3bMix4b")
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+"_vAll_MixedStudy_3bMix3b.tar plotsWithFvT_"+y+"_vAll_MixedStudy_3bMix3b")
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+"_vAll_MixedStudy_3bVs4b.tar  plotsWithFvT_"+y+"_vAll_MixedStudy_3bVs4b")

    
    babySit(cmds, doRun)    








#
#  Make Hists with JCM and FvT weights applied
#
if o.histsNoFvT: 

    dag_config = []
    condor_jobs = []
    jobName = "histsNoFvT_"

    
    noPico = " -p NONE "
    hist3b        = " --histDetailLevel threeTag."+o.histDetailStr
    hist4b        = " --histDetailLevel fourTag."+o.histDetailStr
    outDir = " -o "+getOutDir()+" "


    for y in years:

        #
        # Nominal
        #

        JCMName="Nominal"
        FvTName="_Nominal"

        # Can reuse the 4b data when running iwth FvT weights
        for tagData in [("3b",hist3b)]:

            tag = tagData[0]
            histDetail = tagData[1]

            histName = "hists_"+tag+"_noFvT"+FvTName+".root"

            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM.txt "
            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM_SvB_FvT.txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +  yearOpts[y] + " --histFile "+histName + histDetail + " --jcmNameLoad "+JCMName+" --FvTName  FvT"+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            

            for tt in ttbarSamplesByYear[y]:
                inputFile = " -i "+outputDir+"/fileLists/"+tt+"_"+tag+"_wJCM.txt "
                inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/"+tt+"_"+tag+"_wJCM_SvB_FvT.txt"                

                cmd = runCMD + inputFile + inputWeights + outDir + noPico  + MCyearOpts(tt) + " --histFile " + histName + histDetail  + " --jcmNameLoad "+JCMName+ "  --FvTName FvT"+FvTName
                condor_jobs.append(makeCondorFile(cmd, "None", tt+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            
        
        #
        #  SubSamples
        #
        #  Can reuse the nominal 4b samples
        

    dag_config.append(condor_jobs)

    #
    #  Hadd TTbar
    #
    condor_jobs = []

    for y in years:
        
        FvTName="_Nominal"


        for tag in ["3b"]:
            histName = "hists_"+tag+"_noFvT"+FvTName+".root"
            cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
            for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_"+tag+"_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))

    dag_config.append(condor_jobs)
    condor_jobs = []        


    #
    # Subtract QCD 
    #
    for y in years:
        mkdir(outputDir+"/QCD"+y, doRun)
        FvTName="_Nominal"
        histName3b = "hists_3b_noFvT"+FvTName+".root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
        cmd += " -d "+getOutDir()+"/data"+y+"_3b_wJCM/"+histName3b
        cmd += " --tt "+getOutDir()+"/TT"+y+"/"+histName3b
        cmd += " -q "+getOutDir()+"/QCD"+y+"/"+histName3b
        condor_jobs.append(makeCondorFile(cmd, getOutDir(), "QCD"+y, outputDir=outputDir, filePrefix=jobName) )

    dag_config.append(condor_jobs)
    condor_jobs = []        


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/QCDRunII",   doRun)
        
        #
        #  Nominal
        #
        for tag in ["3b"]:

            FvTName="_Nominal"
            histName = "hists_"+tag+"_noFvT"+FvTName+".root"

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+tag+"_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))            

            cmd = "hadd -f "+getOutDir()+"/QCDRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/QCD"+y+"/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "QCDRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            



        dag_config.append(condor_jobs)

    

    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)





if o.makeInputFileListsSignal: 

    eosDir = "root://cmseos.fnal.gov//store/user/bryantp/condor/"

    for y in years: 

        for s in signalSamples:
            
            fileList = outputDir+"/fileLists/"+s+y+".txt"    
            run("rm "+fileList)

            run("echo "+eosDir+"/"+s+y+"/picoAOD.root >> "+fileList)





#
#  Make Hists Signal Hists
#
if o.histsSignal: 

    jobName = "histsSignal_"

    #
    #  Make Hists
    #
    dag_config = []
    condor_jobs = []

    histName = "hists.root "

    noPico = " -p NONE "
    histOut = " --histFile "+histName
    histDetail  = " --histDetailLevel threeTag.fourTag."+o.histDetailStr

    outDir = " -o "+getOutDir()+" "

    for y in years:

        for sig in signalSamples:
        
            inputFile = " -i "+outputDir+"/fileLists/"+sig+y+".txt "
            cmd = runCMD + inputFile + outDir + noPico  +   MCyearOptsSignal(y)+ histDetail + histOut + " -r -j "+outputDir+"/weights/dataRunII/jetCombinatoricModel_SB_"+JCMTagNom+".txt"
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)

    
    #
    #  Hadd Signal
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4b"+y, outputDir=outputDir, filePrefix=jobName))

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZZ4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4b"+y, outputDir=outputDir, filePrefix=jobName))

    dag_config.append(condor_jobs)




    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        cmd = "hadd -f "+getOutDir()+"/ZZ4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZ4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZ4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ggZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ggZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ggZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/bothZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




# 
#  Make the 3b sample with the stats of the 4b sample
#
if o.subSample3bSignal:

    dag_config = []
    condor_jobs = []

    jobName = "subSample3bSignal_"

    picoOut = " -p picoAOD_3bSubSampled.root "
    h10        = " --histDetailLevel allEvents.passPreSel.passMDRs.threeTag "
    histOut = " --histFile hists_3bSubSampled.root"

    for sig in signalSamples:

        for y in years:

            inputFile = " -i  "+outputDir+"/fileLists/"+sig+y+".txt "

            cmd = runCMD + inputFile + picoOut + " -o "+getOutDir()+ MCyearOptsSignal(y)+  h10+  histOut + " -j closureTests/UL/weights/dataRunII_PreSel/jetCombinatoricModel_SB_03-00-00.txt --skip4b --emulate4bFrom3b --emulationOffset 0 "
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.makeInputFileListsSignalSubSampled: 

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/UL/"

    for y in years: 

        for s in signalSamples:
            
            fileList = outputDir+"/fileLists/"+s+y+"_3bSubSampled.txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+s+y+"/picoAOD_3bSubSampled.root >> "+fileList)




#
#  Make Hists Signal Hists
#
if o.mixSignalDataHemis: 


    dag_config = []
    condor_jobs = []

    jobName = "mixSignalDataHemis_"

    for sig in signalSamples:

        for y in years:

            picoOut    = " -p picoAOD_"+mixedName+".root "
            h10        = " --histDetailLevel passPreSel.passMDRs.threeTag.fourTag "
            histOut    = " --histFile hists_"+mixedName+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 1000000 "
            hemiLoad   += '--inputHLib3Tag \\"NONE\\" --inputHLib4Tag \\"data'+y+'_4b/hemiSphereLib_4TagEvents_*root\\"'
            hemiLoad += " --useHemiWeights "

            inFileList = " -i "+outputDir+"/fileLists/"+sig+y+"_3bSubSampled.txt "

            cmd = runCMD + inFileList+" -o "+getOutDir() + picoOut + MCyearOptsSignal(y) + h10 + histOut+" --usePreCalcBTagSFs "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName, 
                                                        HEMINAME="data"+y+"_hemisDvT", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/data"+y+"_hemisDvT.tgz"))

    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




if o.makeInputFileListsSignalMixData:

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/UL/"

    for y in years: 

        for s in signalSamples:
            
            fileList = outputDir+"/fileLists/"+s+y+"_"+mixedName+".txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+".root >> "+fileList)

            fileList = outputDir+"/fileLists/"+s+y+"_"+mixedName+"_SvB_FvT.txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+"_SvB_FvT.root >> "+fileList)


#
#  Make Hists Signal Hists
#
if o.convertSignalMixData: 

    jobName = "convertSignalMixData_"

    #
    #  Make Hists
    #
    dag_config = []
    condor_jobs = []

    
    for y in years:
    
        
        for sig in signalSamples:
            #picoIn="picoAOD_"+mixedName+"_4b_v"+s+".root"
            picoAOD="picoAOD_"+mixedName+".root"
            picoAODH5="picoAOD_"+mixedName+".h5"
        
            cmd = convertToH5JOB+" -i "+getOutDir()+"/"+sig+y+"_3bSubSampled/"+picoAOD+"  -o "+getOutDir()+"/"+sig+y+"_3bSubSampled/"+picoAODH5
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y+"_"+mixedName, outputDir=outputDir, filePrefix=jobName))
            
    
    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




#
#  Make Hists Signal Hists
#
if o.histsSignalMixData: 

    jobName = "histsSignalMixData_"

    #
    #  Make Hists
    #
    dag_config = []
    condor_jobs = []

    histName = "hists"+mixedName+".root "

    noPico = " -p NONE "
    histOut = " --histFile "+histName
    histDetail  = " --histDetailLevel threeTag.fourTag."+o.histDetailStr

    outDir = " -o "+getOutDir()+" "

    for y in years:

        for sig in signalSamples:
        
            inputFile = " -i "+outputDir+"/fileLists/"+sig+y+"_"+mixedName+".txt "
            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/"+sig+y+"_"+mixedName+"_SvB_FvT.txt"
            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +   MCyearOptsSignal(y)+ histDetail + histOut  + " --usePreCalcBTagSFs "
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)

    
    #
    #  Hadd Signal
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4b"+y, outputDir=outputDir, filePrefix=jobName))

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ZZ4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4b"+y, outputDir=outputDir, filePrefix=jobName))

    dag_config.append(condor_jobs)




    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        cmd = "hadd -f "+getOutDir()+"/ZZ4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZ4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZ4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZH4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ggZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ggZH4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ggZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/bothZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#  Make Hists Signal Hists
#
if o.histsSignal3bSubSamples: 

    jobName = "histsSignal3bSubSamples_"

    #
    #  Make Hists
    #
    dag_config = []
    condor_jobs = []

    histName = "hists_3bSubSampled.root "

    noPico = " -p NONE "
    histOut = " --histFile "+histName
    histDetail  = " --histDetailLevel threeTag.fourTag."+o.histDetailStr

    outDir = " -o "+getOutDir()+" "

    for y in years:

        for sig in signalSamples:
        
            inputFile = " -i "+outputDir+"/fileLists/"+sig+y+"_3bSubSampled.txt "
            cmd = runCMD + inputFile + outDir + noPico  +   MCyearOptsSignal(y)+ histDetail + histOut  + " --usePreCalcBTagSFs " 
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)

    
    #
    #  Hadd Signal
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"_3bSubSampled/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_3bSubSampled/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4b"+y, outputDir=outputDir, filePrefix=jobName))

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"_3bSubSampled/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_3bSubSampled/"+histName+" "
        cmd += getOutDir()+"/ZZ4b"+y+"_3bSubSampled/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4b"+y, outputDir=outputDir, filePrefix=jobName))

    dag_config.append(condor_jobs)




    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        cmd = "hadd -f "+getOutDir()+"/ZZ4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZ4b"+y+"_3bSubSampled/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZ4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZH4b"+y+"_3bSubSampled/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ggZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ggZH4b"+y+"_3bSubSampled/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ggZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/bothZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



# 
#  Copy to AUTON
#
if o.copySignalMixDataToAuton or o.makeAutonDirsForSignalMixData or o.copySignalMixDataFromAuton:
    
    import os
    autonAddr = "gpu13"
    
    #
    # Setup directories
    #
    if o.makeAutonDirsForSignalMixData:

        for y in years:

            for s in signalSamples:
                runA("mkdir hh4b/closureTests/UL/"+s+y+"_3bSubSampled")


    #
    # Copy Files
    #
    if o.copySignalMixDataToAuton:

        for y in years:

            for s in signalSamples:
                scpEOS(EOSOUTDIR,s+y+"_3bSubSampled","picoAOD_"+mixedName+".h5","hh4b/closureTests/UL")                    



    #
    # Copy Files
    #
    if o.copySignalMixDataFromAuton:

        for y in years:

            for s in signalSamples:
                scpFromEOS("picoAOD_"+mixedName+"_SvB_FvT.h5","closureTests/UL/"+s+y+"_3bSubSampled",EOSOUTDIR+s+y+"_3bSubSampled")                    



if o.writeOutSvBFvTWeightsSignalMixData: 

    dag_config = []
    condor_jobs = []
    jobName = "writeOutSvBFvTWeightsSignalMixData_"


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
    ]


    #
    #  Now convert
    #
    for y in years:

        for s in signalSamples:
            varListMixed = list(varListSvB)
    
            cmd = convertToROOTWEIGHTFILE 
            cmd += " --inFileH5 "+getOutDir()+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+"_SvB_FvT.h5"
            cmd += " --inFileROOT "+getOutDir()+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+".root"
            cmd += " --outFile "+getOutDir()+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+"_SvB_FvT.root"
            cmd += " --varList "+",".join(varListMixed)
            condor_jobs.append(makeCondorFile(cmd, "None", s+y+"_"+mixedName, outputDir=outputDir, filePrefix=jobName))
    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag",   doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)


    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#
#
if o.makeSignalPseudoData:

    jobName = "makeSignalPseudoData_"

    dag_config = []
    condor_jobs = []

    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
    picoOut = " -p picoAOD_PseudoData_Mu1000.root "
    histName = "hists_PseudoData_Mu1000.root"
    histOut = " --histFile "+histName

    #
    #  Make Hists for ttbar
    #
    for y in years:
        for sig in signalSamples:

            inputFile = " -i  "+outputDir+"/fileLists/"+sig+y+".txt "
            cmd = runCMD + inputFile + picoOut + " -o "+getOutDir()+ MCyearOptsSignalMu1000(y)+  h10+  histOut + " --skip3b --makePSDataFromMC --mcUnitWeight "
   
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName))
   

    dag_config.append(condor_jobs)



    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)


    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




if o.makeSignalPSFileLists:

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/UL/"

    for y in years: 

        fileListAll = outputDir+"/fileLists/Signal4b"+y+"_PseudoData_Mu1000.txt"    
        run("rm "+fileListAll)

        fileListCombined = outputDir+"/fileLists/dataAndSignal4b"+y+"_PseudoData_Mu1000.txt"    
        run("rm "+fileListCombined)


        for s in signalSamples:
            
            fileList = outputDir+"/fileLists/"+s+y+"_PseudoData_Mu1000.txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+s+y+"/picoAOD_PseudoData_Mu1000.root >> "+fileList)

            run("echo "+eosDir+"/"+s+y+"/picoAOD_PseudoData_Mu1000.root >> "+fileListAll)

            run("echo "+eosDir+"/"+s+y+"/picoAOD_PseudoData_Mu1000.root >> "+fileListCombined)


        run("echo "+ eosDir+"/data"+y+"/picoAOD_4b.root >> "+fileListCombined)
                    



if o.checkSignalPSData:

    jobName = "checkSignalPSData_PS_"

    dag_config = []
    condor_jobs = []

    noPico    = " -p NONE "
    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "

    histNamePSData =   "hists_4b_PseudoDataMu1000.root"

    for y in years:

        for sig in signalSamples:

            # 
            # PSData
            #
            fileListIn = " -i "+outputDir+"/fileLists/"+sig+y+"_PseudoData_Mu1000.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ noPico + yearOpts[y] + h10 + " --histFile " + histNamePSData +"  --unBlind --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName))

            #
            #  Nominal
            #
            # Already Ran Nominal above
    
    dag_config.append(condor_jobs)            

    #
    #  Hadd Signal
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/bothZH4b"+y+"/"+histNamePSData+" "
        cmd += getOutDir()+"/ZH4b"+y+"_PseudoData_Mu1000/"+histNamePSData+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_PseudoData_Mu1000/"+histNamePSData+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4b"+y, outputDir=outputDir, filePrefix=jobName))

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4b"+y+"/"+histNamePSData+" "
        cmd += getOutDir()+"/ZH4b"+y+"_PseudoData_Mu1000/"+histNamePSData+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_PseudoData_Mu1000/"+histNamePSData+" "
        cmd += getOutDir()+"/ZZ4b"+y+"_PseudoData_Mu1000/"+histNamePSData+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4b"+y, outputDir=outputDir, filePrefix=jobName))

    dag_config.append(condor_jobs)




    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        cmd = "hadd -f "+getOutDir()+"/ZZ4bRunII/"+histNamePSData+" "
        for y in years: cmd += getOutDir()+"/ZZ4b"+y+"_PseudoData_Mu1000/"+histNamePSData+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZ4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZH4bRunII/"+histNamePSData+" "
        for y in years: cmd += getOutDir()+"/ZH4b"+y+"_PseudoData_Mu1000/"+histNamePSData+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ggZH4bRunII/"+histNamePSData+" "
        for y in years: cmd += getOutDir()+"/ggZH4b"+y+"_PseudoData_Mu1000/"+histNamePSData+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ggZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/bothZH4bRunII/"+histNamePSData+" "
        for y in years: cmd += getOutDir()+"/bothZH4b"+y+"/"+histNamePSData+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4bRunII/"+histNamePSData+" "
        for y in years: cmd += getOutDir()+"/ZZandZH4b"+y+"/"+histNamePSData+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        dag_config.append(condor_jobs)






    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
# Make Hemisphere library from all hemispheres
#   (Should run locally)
if o.makeHemisSignalOnly:


    
    cmds = []
    logs = []

    picoOut = "  -p 'None' "
    h1     = " --histDetailLevel allEvents.threeTag.fourTag "
    histOut = " --histFile hists.root " 

    for y in years:
        
        cmds.append(runCMD+" -i "+outputDir+"/fileLists/Signal4b"+y+"_PseudoData_Mu1000.txt"+ picoOut + " -o "+os.getcwd()+"/"+outputDir+"/Signal4bHemis_PseudoDataMu1000"+ yearOpts[y]+  h1 +  histOut + " --createHemisphereLibrary --skip3b --isDataMCMix")
        logs.append(outputDir+"/log_makeHemisSignal"+y)
    


    babySit(cmds, doRun, logFiles=logs)


if o.makeHemiTarballSignal:

    for y in years:

        tarballName = 'Signal4bHemis_PseudoDataMu1000_'+y+'_hemis.tgz'
        localTarball = outputDir+"/"+tarballName


        cmd  = 'tar -C '+outputDir+'/Signal4bHemis_PseudoDataMu1000 -zcvf '+ localTarball +' Signal4b'+y+'_PseudoData_Mu1000'
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
#  Make Hists Signal Hists
#
if o.mixSignalSignalHemis: 


    dag_config = []
    condor_jobs = []

    jobName = "mixSignalSignalHemis_"

    for sig in signalSamples:

        for y in years:

            picoOut    = " -p picoAOD_"+mixedName+".root "
            h10        = " --histDetailLevel passPreSel.passMDRs.threeTag.fourTag "
            histOut    = " --histFile hists_"+mixedName+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 1000000 "
            hemiLoad   += '--inputHLib3Tag \\"NONE\\" --inputHLib4Tag \\"Signal4b'+y+'_PseudoData_Mu1000/hemiSphereLib_4TagEvents_*root\\"'
            hemiLoad += " --useHemiWeights "

            inFileList = " -i "+outputDir+"/fileLists/"+sig+y+"_3bSubSampled.txt "

            cmd = runCMD + inFileList+" -o "+getOutDir() + picoOut + MCyearOptsSignal(y) + h10 + histOut+" --usePreCalcBTagSFs "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName, 
                                                        HEMINAME="Signal4bHemis_PseudoDataMu1000_"+y+"_hemis", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/Signal4bHemis_PseudoDataMu1000_"+y+"_hemis.tgz"))

    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
# Make Hemisphere library from all hemispheres
#   (Should run locally)
if o.makeHemisSignalAndData:
    
    cmds = []
    logs = []

    jobName = "makeHemisSignalAndData_"

    picoOut = "  -p 'None' "
    histDetailLevel     = " --histDetailLevel allEvents.threeTag.fourTag "
    histOut = " --histFile hists.root " 

    for y in years:
        inputFile = " -i  "+outputDir+"/fileLists/dataAndSignal4b"+y+"_PseudoData_Mu1000.txt "

        
        cmd = runCMD+ inputFile + picoOut + " -o "+os.getcwd()+"/"+outputDir+"/dataAndSignal4bHemis_PseudoDataMu1000 " + yearOpts[y]+  histDetailLevel +  histOut + " --createHemisphereLibrary  --skip3b --isDataMCMix "

        cmds.append(cmd)
        logs.append(outputDir+"/log_"+jobName+y)

    babySit(cmds, doRun, logFiles=logs)


if o.makeHemiTarballSignalAndData:

    for y in years:

        tarballName = 'dataAndSignal4bHemis_PseudoDataMu1000_'+y+'_hemis.tgz'
        localTarball = outputDir+"/"+tarballName


        cmd  = 'tar -C '+outputDir+'/dataAndSignal4bHemis_PseudoDataMu1000 -zcvf '+ localTarball +' dataAndSignal4b'+y+'_PseudoData_Mu1000'
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
#  Make Hists Signal Hists
#
if o.mixSignalAndData: 


    dag_config = []
    condor_jobs = []

    jobName = "mixSignalAndData_"

    h10        = " --histDetailLevel passPreSel.passMDRs.threeTag.fourTag "


    for sig in signalSamples:

        for y in years:
            
            #
            #  3b SubSampled
            #
            picoOut    = " -p picoAOD_"+mixedName+".root "
            histOut    = " --histFile hists_"+mixedName+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 1000000 "
            hemiLoad   += '--inputHLib3Tag \\"NONE\\" --inputHLib4Tag \\"dataAndSignal4b'+y+'_PseudoData_Mu1000/hemiSphereLib_4TagEvents_*root\\"'
            hemiLoad   += " --useHemiWeights "
            hemiLoad   += " --mcHemiWeight "+o.mcHemiWeight

            inFileList = " -i "+outputDir+"/fileLists/"+sig+y+"_3bSubSampled.txt "

            cmd = runCMD + inFileList+" -o "+getOutDir() + picoOut + MCyearOptsSignal(y) + h10 + histOut+" --usePreCalcBTagSFs "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName, 
                                                        HEMINAME="dataAndSignal4bHemis_PseudoDataMu1000_"+y+"_hemis", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/dataAndSignal4bHemis_PseudoDataMu1000_"+y+"_hemis.tgz"))





    #
    #  Add Data
    #
    for s in subSamples:

        for y in years:

            picoOut    = " -p picoAOD_"+mixedName+"_v"+s+".root "
            histOut    = " --histFile hists_"+mixedName+"_v"+s+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 1000000 "
            hemiLoad   += '--inputHLib3Tag \\"NONE\\" --inputHLib4Tag \\"dataAndSignal4b'+y+'_PseudoData_Mu1000/hemiSphereLib_4TagEvents_*root\\"'
            hemiLoad   += " --useHemiWeights "
            hemiLoad   += " --mcHemiWeight "+o.mcHemiWeight


            #
            #  Data
            #
            inFileList = " -i " + outputDir+"/fileLists/data"+y+"_v"+s+".txt"

            cmd = runCMD + inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --unBlind "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", "data"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName, 
                                                        HEMINAME="dataAndSignal4bHemis_PseudoDataMu1000_"+y+"_hemis", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/dataAndSignal4bHemis_PseudoDataMu1000_"+y+"_hemis.tgz"))



    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#  Make Hists Signal Hists
#
if o.mix4bSignal: 


    dag_config = []
    condor_jobs = []

    jobName = "mix4bSignal_"

    h10        = " --histDetailLevel passPreSel.passMDRs.threeTag.fourTag "


    for sig in signalSamples:

        for y in years:
            
            #
            #  4b
            #
            picoOut    = " -p picoAOD_"+mixedName+".root "
            histOut    = " --histFile hists_"+mixedName+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 1000000 "
            hemiLoad   += '--inputHLib3Tag \\"NONE\\" --inputHLib4Tag \\"dataAndSignal4b'+y+'_PseudoData_Mu1000/hemiSphereLib_4TagEvents_*root\\"'
            hemiLoad   += " --useHemiWeights "
            hemiLoad   += " --mcHemiWeight "+o.mcHemiWeight

            inFileList = " -i "+outputDir+"/fileLists/"+sig+y+".txt "

            cmd = runCMD + inFileList+" -o "+getOutDir() + picoOut + MCyearOptsSignal(y) + h10 + histOut+" "+hemiLoad + " --skip3b " 
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName+"4b_", 
                                                        HEMINAME="dataAndSignal4bHemis_PseudoDataMu1000_"+y+"_hemis", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/dataAndSignal4bHemis_PseudoDataMu1000_"+y+"_hemis.tgz"))



    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#  Make Hists Signal Hists
#
if o.convertMixedSignalAndData: 

    jobName = "convertMixedSignalAndData_"

    #
    #  Make Hists
    #
    dag_config = []
    condor_jobs = []

    
    for y in years:
    
        
        for sig in signalSamples:
            picoAOD="picoAOD_"+mixedName+".root"
            picoAODH5="picoAOD_"+mixedName+".h5"
        
            cmd = convertToH5JOB+" -i "+getOutDir()+"/"+sig+y+"_3bSubSampled/"+picoAOD+"  -o "+getOutDir()+"/"+sig+y+"_3bSubSampled/"+picoAODH5
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y+"_"+mixedName, outputDir=outputDir, filePrefix=jobName))


        for s in subSamples:
            picoAOD="picoAOD_"+mixedName+"_v"+s+".root"
            picoAODH5="picoAOD_"+mixedName+"_v"+s+".h5"
        
            cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"_v"+s+"/"+picoAOD+"  -o "+getOutDir()+"/data"+y+"_v"+s+"/"+picoAODH5
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_v"+s+"_"+mixedName, outputDir=outputDir, filePrefix=jobName))

            
    
    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
#  Make Hists Signal Hists
#
if o.convertMixed4bSignal: 

    jobName = "convertMixed4bSignal_"

    #
    #  Make Hists
    #
    dag_config = []
    condor_jobs = []

    
    for y in years:
    
        
        for sig in signalSamples:
            picoAOD="picoAOD_"+mixedName+".root"
            picoAODH5="picoAOD_"+mixedName+".h5"
        
            cmd = convertToH5JOB+" -i "+getOutDir()+"/"+sig+y+"/"+picoAOD+"  -o "+getOutDir()+"/"+sig+y+"/"+picoAODH5
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y+"_"+mixedName, outputDir=outputDir, filePrefix=jobName))


    
    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



# 
#  Copy to AUTON
#
if o.copyMixedSignalAndDataToAuton or o.copyMixedSignalAndDataFromAuton or o.makeAutonDirsForMixedSignalAndData: # or  o.makeAutonDirsForSignalMixData or o.copySignalMixDataFromAuton:
    
    import os
    autonAddr = "gpu13"
    
    #
    # Setup directories
    #
    if o.makeAutonDirsForMixedSignalAndData:

        for y in years:

            for s in subSamples:
                #scpEOS(EOSOUTDIR,"data"+y+"_v"+s,"picoAOD_"+mixedName+"_v"+s+".h5","hh4b/closureTests/UL")                    

                runA("mkdir hh4b/closureTests/UL/data"+y+"_v"+s)


    #
    # Copy Files
    #
    if o.copyMixedSignalAndDataToAuton:

        for y in years:

            for s in signalSamples:
                scpEOS(EOSOUTDIR,s+y+"_3bSubSampled","picoAOD_"+mixedName+".h5","hh4b/closureTests/UL")                    

            for s in subSamples:
                scpEOS(EOSOUTDIR,"data"+y+"_v"+s,"picoAOD_"+mixedName+"_v"+s+".h5","hh4b/closureTests/UL")                    



    #
    # Copy Files
    #
    if o.copyMixedSignalAndDataFromAuton:

        for y in years:

            for s in signalSamples:
                scpFromEOS("picoAOD_"+mixedName+"_SvB_FvT.h5","closureTests/UL/"+s+y+"_3bSubSampled",EOSOUTDIR+s+y+"_3bSubSampled")                    

            for s in subSamples:
                scpFromEOS("picoAOD_"+mixedName+"_v"+s+"_SvB_FvT.h5","closureTests/UL/data"+y+"_v"+s,EOSOUTDIR+"data"+y+"_v"+s)                    




# 
#  Copy to AUTON
#
if o.copyMixed4bSignalToAuton or o.copyMixed4bSignalFromAuton or o.makeAutonDirsForMixed4bSignal: 
    
    import os
    autonAddr = "gpu13"
    
    #
    # Setup directories
    #
    if o.makeAutonDirsForMixed4bSignal:

        for y in years:

            for s in signalSamples:
                #scpEOS(EOSOUTDIR,"data"+y+"_v"+s,"picoAOD_"+mixedName+"_v"+s+".h5","hh4b/closureTests/UL")                    

                runA("mkdir hh4b/closureTests/UL/"+s+y)


    #
    # Copy Files
    #
    if o.copyMixed4bSignalToAuton:

        for y in years:

            for s in signalSamples:
                scpEOS(EOSOUTDIR,s+y,"picoAOD_"+mixedName+".h5","hh4b/closureTests/UL")                    




    #
    # Copy Files
    #
    if o.copyMixed4bSignalFromAuton:

        for y in years:

            for s in signalSamples:
                scpFromEOS("picoAOD_"+mixedName+"_SvB_FvT.h5","closureTests/UL/"+s+y,EOSOUTDIR+s+y)                    





if o.writeOutSvBFvTWeightsMixedSignalAndData: 

    dag_config = []
    condor_jobs = []
    jobName = "writeOutSvBFvTWeightsMixedSignalAndData_"


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
    ]


    #
    #  Now convert
    #
    for y in years:

        for s in signalSamples:
            varListMixed = list(varListSvB)
    
            cmd = convertToROOTWEIGHTFILE 
            cmd += " --inFileH5 "+getOutDir()+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+"_SvB_FvT.h5"
            cmd += " --inFileROOT "+getOutDir()+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+".root"
            cmd += " --outFile "+getOutDir()+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+"_SvB_FvT.root"
            cmd += " --varList "+",".join(varListMixed)
            condor_jobs.append(makeCondorFile(cmd, "None", s+y+"_"+mixedName, outputDir=outputDir, filePrefix=jobName))

        for s in subSamples:
            varListMixed = list(varListSvB)
    
            cmd = convertToROOTWEIGHTFILE 
            cmd += " --inFileH5 "+getOutDir()+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_v"+s+"_SvB_FvT.h5"
            cmd += " --inFileROOT "+getOutDir()+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_v"+s+".root"
            cmd += " --outFile "+getOutDir()+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_v"+s+"_SvB_FvT.root"
            cmd += " --varList "+",".join(varListMixed)
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+mixedName+"_v"+s, outputDir=outputDir, filePrefix=jobName))

    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag",   doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)


    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.writeOutSvBFvTWeightsMixed4bSignal: 

    dag_config = []
    condor_jobs = []
    jobName = "writeOutSvBFvTWeightsMixed4bSignal_"


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
    ]


    #
    #  Now convert
    #
    for y in years:

        for s in signalSamples:
            varListMixed = list(varListSvB)
    
            cmd = convertToROOTWEIGHTFILE 
            cmd += " --inFileH5 "+getOutDir()+"/"+s+y+"/picoAOD_"+mixedName+"_SvB_FvT.h5"
            cmd += " --inFileROOT "+getOutDir()+"/"+s+y+"/picoAOD_"+mixedName+".root"
            cmd += " --outFile "+getOutDir()+"/"+s+y+"/picoAOD_"+mixedName+"_SvB_FvT.root"
            cmd += " --varList "+",".join(varListMixed)
            condor_jobs.append(makeCondorFile(cmd, "None", s+y+"_"+mixedName, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag",   doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)


    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.makeInputFileListsMixedSignalAndData:

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/UL/"

    for y in years: 

        for s in signalSamples:
            
            fileList = outputDir+"/fileLists/"+s+y+"_"+mixedName+".txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+".root >> "+fileList)

            fileList = outputDir+"/fileLists/"+s+y+"_"+mixedName+"_SvB_FvT.txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+s+y+"_3bSubSampled/picoAOD_"+mixedName+"_SvB_FvT.root >> "+fileList)


        for s in subSamples:
            
            fileList = outputDir+"/fileLists/data"+y+"_"+mixedName+"_v"+s+".txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_v"+s+".root >> "+fileList)

            fileList = outputDir+"/fileLists/data"+y+"_"+mixedName+"_v"+s+"_SvB_FvT.txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_v"+s+"_SvB_FvT.root >> "+fileList)



if o.makeInputFileListsMixed4bSignal:

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/UL/"

    for y in years: 

        for s in signalSamples:
            
            fileList = outputDir+"/fileLists/"+s+y+"_4b_"+mixedName+".txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+s+y+"/picoAOD_"+mixedName+".root >> "+fileList)

            fileList = outputDir+"/fileLists/"+s+y+"_4b_"+mixedName+"_SvB_FvT.txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+s+y+"/picoAOD_"+mixedName+"_SvB_FvT.root >> "+fileList)





#
#  Make Hists Signal Hists
#
if o.histsMixedSignalAndData: 

    jobName = "histsMixedSignalAndData_"

    #
    #  Make Hists
    #
    dag_config = []
    condor_jobs = []

    histName = "hists"+mixedName+".root "

    noPico = " -p NONE "
    histOut = " --histFile "+histName
    histDetail  = " --histDetailLevel threeTag.fourTag."+o.histDetailStr

    outDir = " -o "+getOutDir()+" "

    for y in years:

        for sig in signalSamples:
        
            inputFile = " -i "+outputDir+"/fileLists/"+sig+y+"_"+mixedName+".txt "
            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/"+sig+y+"_"+mixedName+"_SvB_FvT.txt"
            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +   MCyearOptsSignal(y)+ histDetail + histOut + " --usePreCalcBTagSFs " 
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName))

        for s in subSamples:
        
            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_"+mixedName+"_v"+s+".txt "
            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/data"+y+"_"+mixedName+"_v"+s+"_SvB_FvT.txt"
            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +   yearOpts[y]+ histDetail + histOut  + " --unBlind " 
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_v"+s, outputDir=outputDir, filePrefix=jobName))



    dag_config.append(condor_jobs)

    
    #
    #  Hadd Signal
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4b"+y, outputDir=outputDir, filePrefix=jobName))

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ZZ4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4b"+y, outputDir=outputDir, filePrefix=jobName))

    dag_config.append(condor_jobs)




    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        cmd = "hadd -f "+getOutDir()+"/ZZ4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZ4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZ4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZH4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ggZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ggZH4b"+y+"_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ggZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/bothZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        #
        #  Mixed
        #
        for s in subSamples:
            histNameRunII = "hists"+mixedName+"_v"+s+".root"
    
            cmd = "hadd -f "+getOutDir()+"/dataRunII_"+mixedName+"/"+histNameRunII+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+mixedName+"_v"+s+"/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix=jobName))            
    
    
        dag_config.append(condor_jobs)


    #
    #  Hadd SubSamples
    #
    condor_jobs = []

    histNameAll = "hists"+mixedName+"_vAll.root "

    cmdDataMixed = "hadd -f "+getOutDir()+"/dataRunII_"+mixedName+"/"+histNameAll+" "

    for s in subSamples:
        histNameRunII = "hists"+mixedName+"_v"+s+".root"    
        cmdDataMixed += getOutDir()+"/dataRunII_"+mixedName+"/"+histNameRunII+" "
    
    condor_jobs.append(makeCondorFile(cmdDataMixed, "None", "mixedRunII_"+mixedName+"_vAll", outputDir=outputDir, filePrefix=jobName))            
    dag_config.append(condor_jobs)
    
    #
    #  Scale SubSample
    #
    condor_jobs = []

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "

    cmd = cmdScale + " -i "+getOutDir()+"/dataRunII_"+mixedName+"/"+histNameAll+" "

    condor_jobs.append(makeCondorFile(cmd, getOutDir(), "mixedRunII_"+mixedName, outputDir=outputDir, filePrefix=jobName+"scale_"))            
    dag_config.append(condor_jobs)

    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)





#
#  Make Hists Signal Hists
#
if o.histsMixed4bSignal: 

    jobName = "histsMixed4bSignal_"

    #
    #  Make Hists
    #
    dag_config = []
    condor_jobs = []

    histName = "hists_4b_"+mixedName+".root "

    noPico = " -p NONE "
    histOut = " --histFile "+histName
    histDetail  = " --histDetailLevel threeTag.fourTag."+o.histDetailStr

    outDir = " -o "+getOutDir()+" "

    for y in years:

        for sig in signalSamples:
        
            inputFile = " -i "+outputDir+"/fileLists/"+sig+y+"_4b_"+mixedName+".txt "
            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/"+sig+y+"_4b_"+mixedName+"_SvB_FvT.txt"
            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +   MCyearOptsSignal(y)+ histDetail + histOut + " --usePreCalcBTagSFs "
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)

    
    #
    #  Hadd Signal
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"_4b_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_4b_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4b"+y, outputDir=outputDir, filePrefix=jobName))

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        cmd += getOutDir()+"/ZH4b"+y+"_4b_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ggZH4b"+y+"_4b_"+mixedName+"/"+histName+" "
        cmd += getOutDir()+"/ZZ4b"+y+"_4b_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4b"+y, outputDir=outputDir, filePrefix=jobName))

    dag_config.append(condor_jobs)




    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        cmd = "hadd -f "+getOutDir()+"/ZZ4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZ4b"+y+"_4b_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZ4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZH4b"+y+"_4b_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ggZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ggZH4b"+y+"_4b_"+mixedName+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ggZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/bothZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/bothZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "bothZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

        cmd = "hadd -f "+getOutDir()+"/ZZandZH4bRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/ZZandZH4b"+y+"/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "ZZandZH4bRunII", outputDir=outputDir, filePrefix=jobName))            

    
        dag_config.append(condor_jobs)


    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


