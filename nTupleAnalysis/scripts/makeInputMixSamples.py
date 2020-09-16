
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse


parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9", help="Year or comma separated list of subsamples")
parser.add_option('-w',            action="store_true", dest="doWeights",      default=False, help="Fit jetCombinatoricModel and nJetClassifier TSpline")
parser.add_option('--subSample3b',  action="store_true",      help="Subsample 3b to look like 4b")
parser.add_option('--histSubSample3b',  action="store_true",      help="plot hists of the Subsampled 3b ")
parser.add_option('--make4bHemis',  action="store_true",      help="make 4b Hemisphere library ")
parser.add_option('--copyToEOS',  action="store_true",      help="Copy 3b subsampled data to eos ")
parser.add_option('--cleanPicoAODs',  action="store_true",      help="rm 3b subsampled data  ")
parser.add_option('--makeInputFileLists',  action="store_true",      help="make file lists  ")
parser.add_option('--make4bHemiTarball',  action="store_true",      help="make 4b Hemi Tarball  ")
#parser.add_option('--histsWithJCM', action="store_true",      help="Make hist.root with JCM")
#parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
#parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
#parser.add_option('--plotsWithJCM', action="store_true",      help="Make pdfs with JCM")
parser.add_option('-c',   '--condor',   action="store_true", default=False,           help="Run on condor")
#parser.add_option('-l',   '--cd',   action="store_true", default=True,           help="Run on condor")
parser.add_option('--email',            default="johnalison@cmu.edu",      help="")

o, a = parser.parse_args()


doRun = o.execute

years = o.year.split(",")
subSamples = o.subSamples.split(",")
ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]

outputDir="closureTests/mixed/"
outputDirNom="closureTests/nominal/"

# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'


# 
#  Hists with all 2018 data no weights (Made in the nominal closure test)
#

yearOpts={}
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
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/mixed/"
EOSOUTDIRNOM = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/nominal/"
TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"


def getOutDir():
    if o.condor:
        return EOSOUTDIR
    return outputDir

def getOutDirNom():
    if o.condor:
        return EOSOUTDIRNOM
    return outputDirNom


if o.condor:
    print "Making Tarball"
    makeTARBALL(o.execute)


#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
if o.doWeights:

    cmds = []
    logs = []

    mkdir(outputDir+"/weights", doRun)

    # histName = "hists.root"
    histName = "hists_"+tagID+".root " 

    for y in years:

        dataFile = getOutDirNom()+"/data"+y+"_"+tagID+"/"+histName

        cmd  = weightCMD
        cmd += " -d "+dataFile
        cmd += " -c passPreSel   -o "+outputDir+"/weights/noTT_data"+y+"_"+tagID+"_PreSel/  -r SB -w 01-00-00"

        cmds.append(cmd)
        logs.append(outputDir+"/log_fitJCM_"+tagID+"_PS_"+y)


    babySit(cmds, doRun, logFiles=logs)

    rmTARBALL(o.execute)

jcmFileList = {}
jcmFileList["2018"] = outputDir+"/weights/noTT_data2018_"+tagID+"_PreSel/jetCombinatoricModel_SB_01-00-00.txt"
jcmFileList["2017"] = outputDir+"/weights/noTT_data2017_"+tagID+"_PreSel/jetCombinatoricModel_SB_01-00-00.txt"
jcmFileList["2016"] = outputDir+"/weights/noTT_data2016_"+tagID+"_PreSel/jetCombinatoricModel_SB_01-00-00.txt"


# 
#  Make the 3b sample with the stats of the 4b sample
#
if o.subSample3b:
    # In the following "3b" refers to 3b subsampled to have the 4b statistics

    cmds = []
    logs = []
    dag_config = []
    condor_jobs = []

    # histName = "hists.root"
    histName = "hists_"+tagID+".root " 

    for s in subSamples:

        for y in years:
            
            picoOut = " -p picoAOD_3bSubSampled_"+tagID+"_v"+s+".root "
            h10     = " --histogramming 10 "
            histOut = " --histFile hists_"+tagID+"_v"+s+".root"

            cmd = runCMD+" -i "+outputDirNom+"/fileLists/data"+y+"_"+tagID+".txt"+ picoOut + " -o "+outputDir+ yearOpts[y]+  h10+  histOut + " -j "+jcmFileList[y]+" --emulate4bFrom3b --emulationOffset "+s+" --noDiJetMassCutInPicoAOD "

            if o.condor:
                cmd += " --condor"
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="subSample3b_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_dataOnlyAll_make3b_"+y+"_"+tagID+"_v"+s)

            for tt in ttbarSamples:

                cmd = runCMD+" -i "+outputDirNom+"/fileLists/"+tt+y+"_noMjj_"+tagID+".txt" + picoOut + " -o "+outputDir + MCyearOpts[y] +h10 + histOut + " -j "+jcmFileList[y]+" --emulate4bFrom3b --emulationOffset "+s+" --noDiJetMassCutInPicoAOD"
                
                if o.condor:
                    cmd += " --condor"
                    condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="subSample3b_"))                    
                else:
                    cmds.append(cmd)
                    logs.append(outputDir+"/log_"+tt+y+"_v"+s)


    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)


    if o.condor:
        rmdir(outputDir+"subSample3b_All.dag", doRun)
        rmdir(outputDir+"subSample3b_All.dag.*", doRun)

        dag_file = makeDAGFile("subSample3b_All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)

    else:
        if o.email: execute('echo "Subject: [makeInputMixSamples] subSample3b  Done" | sendmail '+o.email,doRun)


#
#   Make inputs fileLists
#
if o.makeInputFileLists:

    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/mixed/"

    for s in subSamples:

        for y in years:

            for sample in ["data","TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"]:
                
                fileList = outputDir+"/fileLists/"+sample+y+"_"+tagID+"_v"+s+".txt"    
                run("rm "+fileList)

                subdir = sample+y+"_"+tagID+"_v"+s
                picoName = "picoAOD_3bSubSampled_"+tagID+"_v"+s+".root"

                run("echo "+eosDir+"/"+subdir+"/"+picoName+" >> "+fileList)




#
#  Make hists of the subdampled data
#    #(Optional: for sanity check
if o.histSubSample3b:

    cmds = []
    logs = []
    dag_config = []
    condor_jobs = []

    for s in subSamples:

        for y in years:
            
            picoIn  = "picoAOD_3bSubSampled_"+tagID+"_v"+s+".root "
            picoOut = "  -p 'None' "
            h10     = " --histogramming 10 "
            histOut = " --histFile hists_3bSubSampled_"+tagID+"_v"+s+".root "
            cmd = runCMD+" -i "+outputDir+"/fileLists/data"+y+"_"+tagID+"_v"+s+".txt " + picoOut +" -o "+outputDir+ yearOpts[y] + h10 + histOut + " --is3bMixed  --writeOutEventNumbers "

            if o.condor:
                cmd += " --condor"
                condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histSubSample3b_"))
            else:
                cmds.append(cmd)
                logs.append(outputDir+"/log_subSampledHists_"+tagID+"_"+y+"_v"+s)

    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)


    if o.condor:
        rmdir(outputDir+"histSubSample3b_All.dag", doRun)
        rmdir(outputDir+"histSubSample3b_All.dag.*", doRun)

        dag_file = makeDAGFile("histSubSample3b_All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)

    else:
        if o.email: execute('echo "Subject: [makeInputMixSamples] histSubSample3b  Done" | sendmail '+o.email,doRun)



 
#
# Copy subsamples to EOS
#   (Not needed with condor:)
if o.copyToEOS:

    def copy(fileName, subDir, outFileName):
        cmd  = "xrdcp  "+fileName+" root://cmseos.fnal.gov//store/user/johnda/closureTest/mixed/"+subDir+"/"+outFileName
    
        if doRun:
            os.system(cmd)
        else:
            print cmd


    for y in years:
        for tt in ["data","TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"]:
            subDir = tt+y if tt == "data" else tt+y+"_noMjj"
            for s in subSamples:
                copy("closureTests/mixed/"+subDir+"_"+tagID+"/picoAOD_3bSubSampled_"+tagID+"_v"+s+".root", subDir,"picoAOD_3bSubSampled_"+tagID+"_v"+s+".root")

#   (Not needed with condor:)
if o.cleanPicoAODs:
    
    def rm(fileName):
        cmd  = "rm  "+fileName
    
        if doRun: os.system(cmd)
        else:     print cmd

    for y in years:
        for tt in ["data","TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"]:
            subDir = tt+y if tt == "data" else tt+y+"_noMjj"
            for s in subSamples:
                rm("closureTests/mixed/"+subDir+"_"+tagID+"/picoAOD_3bSubSampled_"+tagID+"_v"+s+".root")



#
# Make Hemisphere library from all hemispheres
#   (Should run locally)
if o.make4bHemis:
    
    cmds = []
    logs = []

    picoOut = "  -p 'None' "
    h1     = " --histogramming 1 "
    histOut = " --histFile hists_"+tagID+".root " 

    for y in years:
        
        cmds.append(runCMD+" -i "+outputDirNom+"/fileLists/data"+y+"_"+tagID+".txt"+ picoOut + " -o "+outputDir+"/dataHemis_"+tagID+ yearOpts[y]+  h1 +  histOut + " --createHemisphereLibrary --skip3b")
        logs.append(outputDir+"/log_makeHemisData"+y+"_"+tagID)
    


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeInputMixSamples] make4bHemis  Done" | sendmail '+o.email,doRun)



if o.make4bHemiTarball:

    for y in years:
    #tar -C closureTests/mixed/dataHemis_b0p60p3 -zcvf closureTests/mixed/data2018_b0p60p3_hemis.tgz data2018_b0p60p3  --exclude="hist*root" --exclude-vcs --exclude-caches-all

        tarballName = 'data'+y+'_'+tagID+'_hemis.tgz'
        localTarball = outputDir+"/"+tarballName

        cmd  = 'tar -C '+outputDir+"/dataHemis_"+tagID+' -zcvf '+ localTarball +' data'+y+'_'+tagID
        cmd += ' --exclude="hist*root"  '
        cmd += ' --exclude-vcs --exclude-caches-all'

        execute(cmd, doRun)
        cmd  = 'ls -hla '+localTarball
        execute(cmd, doRun)
        cmd = "xrdfs root://cmseos.fnal.gov/ mkdir /store/user/"+getUSER()+"/condor"
        execute(cmd, doRun)
        cmd = "xrdcp -f "+localTarball+ " root://cmseos.fnal.gov//store/user/"+getUSER()+"/condor/"+tarballName
        execute(cmd, doRun)
