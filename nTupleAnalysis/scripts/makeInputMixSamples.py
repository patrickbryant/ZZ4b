
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
parser.add_option('--subSample3bQCD',  action="store_true",      help="Subsample 3b to look like 4b")
parser.add_option('--doWeightsQCD',  action="store_true",      help="")
parser.add_option('--inputsForDataVsTT',  action="store_true",      help="makeInputs for Dave Vs TTbar")
parser.add_option('--subSample3bSignal',  action="store_true",      help="Subsample 3b to look like 4b")
parser.add_option('--histSubSample3b',  action="store_true",      help="plot hists of the Subsampled 3b ")
parser.add_option('--make4bHemis',  action="store_true",      help="make 4b Hemisphere library ")
parser.add_option('--make4bHemisWithDvT',  action="store_true",      help="make 4b Hemisphere library ")
parser.add_option('--make4bHemisSignalMu10',  action="store_true",      help="make 4b Hemisphere library ")
parser.add_option('--copyToEOS',  action="store_true",      help="Copy 3b subsampled data to eos ")
parser.add_option('--copyWeightsToEOS',  action="store_true",      help="Copy 3b subsampled data to eos ")
parser.add_option('--writeOutDvT3Weights',  action="store_true",      help=" ")
parser.add_option('--writeOutDvT4Weights',  action="store_true",      help=" ")
parser.add_option('--cleanPicoAODs',  action="store_true",      help="rm 3b subsampled data  ")
parser.add_option('--makeInputFileLists',  action="store_true",      help="make file lists  ")
parser.add_option('--makeInputFileListsSubSampledQCD',  action="store_true",      help="make file lists  ")
parser.add_option('--make4bHemiTarball',  action="store_true",      help="make 4b Hemi Tarball  ")
parser.add_option('--make4bHemiTarballDvT',  action="store_true",      help="make 4b Hemi Tarball  ")
parser.add_option('--make4bHemiTarballSignalMu10',  action="store_true",      help="make 4b Hemi Tarball  ")
parser.add_option('--makeTTPseudoData',  action="store_true",      help="make PSeudo data  ")
parser.add_option('--removeTTPseudoData',  action="store_true",      help="make PSeudo data  ")
#parser.add_option('--histsWithJCM', action="store_true",      help="Make hist.root with JCM")
#parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
#parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
#parser.add_option('--plotsWithJCM', action="store_true",      help="Make pdfs with JCM")
parser.add_option('--makeTarball',  action="store_true",      help="make Output file lists")
parser.add_option('--checkPSData',  action="store_true",      help="make Output file lists")
parser.add_option('--checkSignalPSData',  action="store_true",      help="make Output file lists")
parser.add_option('--checkOverlap',  action="store_true",      help="make Output file lists")
parser.add_option('--makeSignalPseudoData',  action="store_true",      help="make PSeudo data  ")
parser.add_option('--makeSignalPSFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('-c',   '--condor',   action="store_true", default=False,           help="Run on condor")
#parser.add_option('-l',   '--cd',   action="store_true", default=True,           help="Run on condor")
parser.add_option('--copyToAuton', action="store_true",      help="Should be obvious")
parser.add_option('--copyFromAuton', action="store_true",      help="Should be obvious")
parser.add_option('--makeAutonDirs', action="store_true",      help="Should be obvious")
parser.add_option('--copyLocally',  action="store_true",      help="make Input file lists")
parser.add_option('--makeDvT3FileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--makeDvT4FileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--testDvTWeights',  action="store_true",      help="make Input file lists")
parser.add_option('--mixInputsDvT',  action="store_true",      help="")
parser.add_option('--email',            default="johnalison@cmu.edu",      help="")
parser.add_option(     '--doDvTReweight',        action="store_true", help="boolean  to toggle using FvT reweight")
parser.add_option(     '--doTTbarPtReweight',        action="store_true", help="boolean  to toggle using FvT reweight")
parser.add_option(     '--no3b',        action="store_true", help="boolean  to toggle using FvT reweight")

o, a = parser.parse_args()


doRun = o.execute

years = o.year.split(",")
subSamples = o.subSamples.split(",")
ttbarSamples  = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]
signalSamples = ["ZZ4b","ZH4b","ggZH4b"]

outputDir="closureTests/mixed/"
outputDir3bMix4b="closureTests/3bMix4b_4bTT/"
outputDirNom="closureTests/nominal/"
outputDirComb="closureTests/combined_4bTT/"
outputDirCombOLD="closureTests/combined/"

# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'
convertToH5JOB='python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'
convertToROOTWEIGHTFILE = 'python ZZ4b/nTupleAnalysis/scripts/convert_h52rootWeightFile.py'

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

MCyearOptsMu10 = {}
MCyearOptsMu10["2018"]=yearOpts["2018"]+' --bTagSF -l 600.0e3 --isMC '
MCyearOptsMu10["2017"]=yearOpts["2017"]+' --bTagSF -l 367.0e3 --isMC '
MCyearOptsMu10["2016"]=yearOpts["2016"]+' --bTagSF -l 359.0e3 --isMC '


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


if o.makeTarball:
    print "Remove old Tarball"
    rmTARBALL(o.execute)
    makeTARBALL(o.execute)


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

    dag_config = []
    condor_jobs = []

    # histName = "hists.root"
    histName = "hists_"+tagID+".root " 

    for s in subSamples:

        for y in years:
            
            picoOut = " -p picoAOD_3bSubSampled_"+tagID+"_v"+s+".root "
            h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
            histOut = " --histFile hists_"+tagID+"_v"+s+".root"

            cmd = runCMD+" -i "+outputDirNom+"/fileLists/data"+y+"_"+tagID+".txt"+ picoOut + " -o "+getOutDir()+ yearOpts[y]+  h10+  histOut + " -j "+jcmFileList[y]+" --emulate4bFrom3b --emulationOffset "+s+" --noDiJetMassCutInPicoAOD "

            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="subSample3b_"))

            for tt in ttbarSamples:
            
                cmd = runCMD+" -i "+outputDirNom+"/fileLists/"+tt+y+"_noMjj_"+tagID+".txt" + picoOut + " -o "+getOutDir() + MCyearOpts[y] +h10 + histOut + " -j "+jcmFileList[y]+" --emulate4bFrom3b --emulationOffset "+s+" --noDiJetMassCutInPicoAOD"
                
                condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="subSample3b_"))                    


    dag_config.append(condor_jobs)


    execute("rm "+outputDir+"subSample3b_All.dag", doRun)
    execute("rm "+outputDir+"subSample3b_All.dag.*", doRun)

    dag_file = makeDAGFile("subSample3b_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)






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

                subdir = sample+y+"_"+tagID
                if sample in ttbarSamples:
                    subdir = sample+y+"_noMjj_"+tagID

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
            h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
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
    h1     = " --histDetailLevel allEvents.threeTag.fourTag "
    histOut = " --histFile hists_"+tagID+".root " 

    for y in years:
        
        cmds.append(runCMD+" -i "+outputDirNom+"/fileLists/data"+y+"_"+tagID+".txt"+ picoOut + " -o "+outputDir+"/dataHemisFixPt_"+tagID+ yearOpts[y]+  h1 +  histOut + " --createHemisphereLibrary --skip3b")
        logs.append(outputDir+"/log_makeHemisData"+y+"_"+tagID)
    


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeInputMixSamples] make4bHemis  Done" | sendmail '+o.email,doRun)



if o.make4bHemiTarball:

    for y in years:
    #tar -C closureTests/mixed/dataHemis_b0p60p3 -zcvf closureTests/mixed/data2018_b0p60p3_hemis.tgz data2018_b0p60p3  --exclude="hist*root" --exclude-vcs --exclude-caches-all

        tarballName = 'data'+y+'_'+tagID+'_hemis.tgz'
        localTarball = outputDir+"/"+tarballName

        cmd  = 'tar -C '+outputDir+"/dataHemisFixPt_"+tagID+' -zcvf '+ localTarball +' data'+y+'_'+tagID
        cmd += ' --exclude="hist*root"  '
        cmd += ' --exclude-vcs --exclude-caches-all'

        execute(cmd, doRun)
        cmd  = 'ls -hla '+localTarball
        execute(cmd, doRun)
        cmd = "xrdfs root://cmseos.fnal.gov/ mkdir /store/user/"+getUSER()+"/condor"
        execute(cmd, doRun)
        cmd = "xrdcp -f "+localTarball+ " root://cmseos.fnal.gov//store/user/"+getUSER()+"/condor/"+tarballName
        execute(cmd, doRun)


if o.make4bHemiTarballSignalMu10:

    for y in years:
    #tar -C closureTests/mixed/dataHemis_b0p60p3 -zcvf closureTests/mixed/data2018_b0p60p3_hemis.tgz data2018_b0p60p3  --exclude="hist*root" --exclude-vcs --exclude-caches-all

        tarballName = 'dataWithMu10Signal'+y+'_'+tagID+'_hemis.tgz'
        localTarball = outputDir+"/"+tarballName


        cmd  = 'tar -C '+outputDir+"/dataWithMu10SignalHemis_"+tagID+' -zcvf '+ localTarball +' dataWithMu10Signal'+y+'_'+tagID
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
#
#
if o.makeTTPseudoData:
    cmds = []
    logs = []
    dag_config = []
    condor_jobs = []

    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
    picoOut = " -p picoAOD_4bPseudoData_"+tagID+".root "
    histName = "hists_4bPseudoData_"+tagID+".root"
    histOut = " --histFile "+histName

    #
    #  Make Hists for ttbar
    #
    for y in years:
       for tt in ttbarSamples:
           cmd = runCMD+" -i "+outputDirNom+"/fileLists/"+tt+y+"_noMjj_"+tagID+".txt "+ picoOut +" -o "+getOutDir()+ MCyearOpts[y] + h10 + histOut +" --skip3b --makePSDataFromMC --mcUnitWeight  "
   
           if o.condor:
               condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_noMjj_"+tagID, outputDir=outputDir, filePrefix="makeTTPseudoData_"))
           else:
               cmds.append(cmd)
               logs.append(outputDir+"/log_"+tt+y+"_"+tagID)
   

    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)


    #
    #  Hadd ttbar
    #
    cmds = [] 
    logs = []
    condor_jobs = []

    for y in years:
        mkdir(outputDir+"/TT"+y, doRun)
        
        cmd = "hadd -f "+ getOutDir()+"/TT"+y+"/"+histName+" "
        for tt in ttbarSamples:        
            cmd += getOutDir()+"/"+tt+y+"_noMjj_"+tagID+"/"+histName+" "

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y, outputDir=outputDir, filePrefix="makeTTPseudoData_"))            
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_HaddTT"+y+"_"+tagID)

    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/TTRunII",   doRun)

        cmds = []
        logs = []
        condor_jobs = []        
    

        cmd = "hadd -f " + getOutDir()+"/TTRunII/"+ histName+" "
        for y in years:
            cmd += getOutDir()+"/TT"+y+"/"  +histName+" "

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII", outputDir=outputDir, filePrefix="makeTTPseudoData_"))            
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_haddTTRunII_"+tagID)

        if o.condor:
            dag_config.append(condor_jobs)            
        else:
            babySit(cmds, doRun, logFiles=logs)


    if o.condor:
        execute("rm "+outputDir+"makeTTPseudoData_All.dag", doRun)
        execute("rm "+outputDir+"makeTTPseudoData_All.dag.*", doRun)


        dag_file = makeDAGFile("makeTTPseudoData_All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)

    else:
        if o.email: execute('echo "Subject: [makeInputMixedSamples] makeTTPseudoData  Done" | sendmail '+o.email,doRun)


#
#
#
if o.removeTTPseudoData:

    cmds = []
    logs = []
    dag_config = []
    condor_jobs = []

    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
    picoOut = " -p picoAOD_noPSData_"+tagID+".root "
    histName = "hists_noPSData_"+tagID+".root"
    histOut = " --histFile "+histName

    #
    #  Make Hists for ttbar
    #
    for y in years:
       for tt in ttbarSamples:

           cmd = runCMD+" -i "+outputDirNom+"/fileLists/"+tt+y+"_noMjj_"+tagID+".txt "+ picoOut +" -o "+getOutDir()+ MCyearOpts[y] + h10 + histOut +" --skip3b --removePSDataFromMC  "
   
           if o.condor:
               condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_noMjj_"+tagID, outputDir=outputDir, filePrefix="removeTTPseudoData_"))
           else:
               cmds.append(cmd)
               logs.append(outputDir+"/log_"+tt+y+"_"+tagID)



    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)


    #
    #  Hadd ttbar
    #
    cmds = [] 
    logs = []
    condor_jobs = []

    for y in years:
        mkdir(outputDir+"/TT"+y, doRun)
        
        cmd = "hadd -f "+ getOutDir()+"/TT"+y+"/"+histName+" "
        for tt in ttbarSamples:        
            cmd += getOutDir()+"/"+tt+y+"_noMjj_"+tagID+"/"+histName+" "

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y, outputDir=outputDir, filePrefix="removeTTPseudoData_"))            
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_HaddTT"+y+"_"+tagID)

    if o.condor:
        dag_config.append(condor_jobs)
    else:
        babySit(cmds, doRun, logFiles=logs)


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/TTRunII",   doRun)

        cmds = []
        logs = []
        condor_jobs = []        
    

        cmd = "hadd -f " + getOutDir()+"/TTRunII/"+ histName+" "
        for y in years:
            cmd += getOutDir()+"/TT"+y+"/"  +histName+" "

        if o.condor:
            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII", outputDir=outputDir, filePrefix="removeTTPseudoData_"))            
        else:
            cmds.append(cmd)
            logs.append(outputDir+"/log_haddTTRunII_"+tagID)

        if o.condor:
            dag_config.append(condor_jobs)            
        else:
            babySit(cmds, doRun, logFiles=logs)



    if o.condor:
        execute("rm "+outputDir+"removeTTPseudoData_All.dag", doRun)
        execute("rm "+outputDir+"removeTTPseudoData_All.dag.*", doRun)


        dag_file = makeDAGFile("removeTTPseudoData_All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)

    else:
        if o.email: execute('echo "Subject: [makeInputMixedSamples] removeTTPseudoData  Done" | sendmail '+o.email,doRun)





if o.checkPSData:
    dag_config = []
    condor_jobs = []

    noPico    = " -p NONE "
    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "


    histNameNoPSData = "hists_4b_noPSData_"+tagID+".root"
    histNamePSData =   "hists_4b_PSData_"+tagID+".root"
    histNameNom =      "hists_4b_nominal_"+tagID+".root"
    for y in years:

        for tt in ttbarSamples:

            # 
            # No PSData
            #
            fileListIn = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_noPSData.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ noPico + MCyearOpts[y] + h10 + " --histFile " + histNameNoPSData +"  --writeOutEventNumbers "
            condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="checkPSData_noPS_"))

            # 
            # PSData
            #
            fileListIn = " -i "+outputDir3bMix4b+"/fileLists/"+tt+y+"_PSData_"+tagID+".txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ noPico + yearOpts[y] + h10 + " --histFile " + histNamePSData +"  --is3bMixed --isDataMCMix --writeOutEventNumbers "
            condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="checkPSData_PS_"))

            #
            #  Nominal
            #
            fileListIn = " -i "+outputDirCombOLD+"/fileLists/"+tt+y+"_"+tagID+"_4b_wFvT.txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir() + noPico  + MCyearOpts[y]+ h10 + " --histFile " + histNameNom +"  --writeOutEventNumbers "
            condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="checkPSData_Nom_"))


    
    dag_config.append(condor_jobs)            


    #
    #  Hadd ttbar
    #
    condor_jobs = []

    for y in years:

        for h in [(histNameNoPSData,tagID+"_noPSData") , (histNamePSData,"PSData_"+tagID), (histNameNom,tagID+"_4b_wFvT" )]:
            cmd = "hadd -f "+ getOutDir()+"/TT"+y+"/"+h[0]+" "
            for tt in ttbarSamples:        
                cmd += getOutDir()+"/"+tt+y+"_"+h[1]+"/"+h[0]+" "

            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_"+h[1], outputDir=outputDir, filePrefix="checkPSData_"))            

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

            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_"+h, outputDir=outputDir, filePrefix="checkPSData_"))            


        dag_config.append(condor_jobs)            




    execute("rm "+outputDir+"checkPSData_All.dag", doRun)
    execute("rm "+outputDir+"checkPSData_All.dag.*", doRun)

    dag_file = makeDAGFile("checkPSData_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.checkOverlap:

    for y in years:
        for tt in ttbarSamples:        
            cmd = "python ZZ4b/nTupleAnalysis/scripts/compEventCounts.py "
            cmd += "--file1 "+getOutDir()+tt+y+"_"+tagID+"_noPSData/hists_4b_noPSData_"+tagID+".root "
            cmd += " --file2 "+getOutDir()+tt+y+"_PSData_"+tagID+"/hists_4b_PSData_"+tagID+".root "

            execute(cmd, o.execute)

            
            cmd = "python ZZ4b/nTupleAnalysis/scripts/compEventCounts.py "
            cmd += " --file1 "+getOutDir()+tt+y+"_"+tagID+"_noPSData/hists_4b_noPSData_"+tagID+".root "
            cmd += " --file2 "+getOutDir()+tt+y+"_"+tagID+"_4b_wFvT/hists_4b_nominal_"+tagID+".root "
            
            execute(cmd, o.execute)
    




#
#
#
if o.makeSignalPseudoData:

    dag_config = []
    condor_jobs = []

    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
    picoOut = " -p picoAOD_PseudoData_"+tagID+".root "
    histName = "hists_PseudoData_"+tagID+".root"
    histOut = " --histFile "+histName

    #
    #  Make Hists for ttbar
    #
    for y in years:
        for sig in signalSamples:
            cmd = runCMD+" -i "+outputDirNom+"/fileLists/"+sig+y+"_noMjj_"+tagID+".txt "+ picoOut +" -o "+getOutDir()+ MCyearOptsMu10[y] + h10 + histOut +"  --makePSDataFromMC --mcUnitWeight  "
   
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y+"_noMjj_"+tagID, outputDir=outputDir, filePrefix="makeSignalPseudoData_"))
   

    dag_config.append(condor_jobs)



    execute("rm "+outputDir+"makeSignalPseudoData_All.dag", doRun)
    execute("rm "+outputDir+"makeSignalPseudoData_All.dag.*", doRun)


    dag_file = makeDAGFile("makeSignalPseudoData_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




#
#   Make inputs fileLists
#
if o.makeSignalPSFileLists:

    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    #mkdir(outputDir+"/fileLists", execute=doRun)

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/mixed/"    


    for y in years:

        fileListDataAndSignal = outputDir+"/fileLists/dataWithMu10Signal"+y+"_"+tagID+".txt"    
        run("rm "+fileListDataAndSignal)

        for sig in signalSamples:
            fileList = outputDir+"/fileLists/"+sig+y+"_PSDataMu10_"+tagID+".txt"    
            run("rm "+fileList)
            run("echo "+eosDir+"/"+sig+y+"_noMjj_"+tagID+"/picoAOD_PseudoData_"+tagID+".root >> "+fileList)

            fileList = outputDir+"/fileLists/"+sig+y+"_3bSubSampled_"+tagID+".txt"
            run("rm "+fileList)
            run("echo "+eosDir+"/"+sig+y+"_noMjj_"+tagID+"/picoAOD_3bSubSampled_"+tagID+".root >> "+fileList)

            run("echo "+eosDir+"/"+sig+y+"_noMjj_"+tagID+"/picoAOD_PseudoData_"+tagID+".root >> "+fileListDataAndSignal)
            

        run("cat  "+outputDirNom+"/fileLists/data"+y+"_"+tagID+".txt >> "+fileListDataAndSignal)            
            




if o.checkSignalPSData:
    dag_config = []
    condor_jobs = []

    noPico    = " -p NONE "
    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "

    histNamePSData =   "hists_4b_PSDataMu10_"+tagID+".root"
    histNameNom =      "hists_4b_nominal_"+tagID+".root"
    for y in years:

        for sig in signalSamples:

            # 
            # PSData
            #
            fileListIn = " -i "+outputDir+"/fileLists/"+sig+y+"_PSDataMu10_"+tagID+".txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir()+ noPico + yearOpts[y] + h10 + " --histFile " + histNamePSData +"  --is3bMixed --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y+"_"+tagID, outputDir=outputDir, filePrefix="checkSignalPSData_PS_"))

            #
            #  Nominal
            #
            fileListIn = " -i "+outputDirNom+"/fileLists/"+sig+y+"_noMjj_"+tagID+".txt "
            cmd = runCMD + fileListIn + " -o "+getOutDir() + noPico  + MCyearOpts[y]+ h10 + " --histFile " + histNameNom
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y+"_"+tagID, outputDir=outputDir, filePrefix="checkSignalPSData_Nom_"))

    
    dag_config.append(condor_jobs)            


    execute("rm "+outputDir+"checkSignalPSData_All.dag", doRun)
    execute("rm "+outputDir+"checkSignalPSData_All.dag.*", doRun)

    dag_file = makeDAGFile("checkSignalPSData_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
# Make Hemisphere library from all hemispheres
#   (Should run locally)
if o.make4bHemisSignalMu10:


    
    cmds = []
    logs = []

    picoOut = "  -p 'None' "
    h1     = " --histDetailLevel allEvents.threeTag.fourTag "
    histOut = " --histFile hists_"+tagID+".root " 

    for y in years:
        
        cmds.append(runCMD+" -i "+outputDir+"/fileLists/dataWithMu10Signal"+y+"_"+tagID+".txt"+ picoOut + " -o $PWD/"+outputDir+"/dataWithMu10SignalHemis_"+tagID+ yearOpts[y]+  h1 +  histOut + " --createHemisphereLibrary --skip3b --isDataMCMix")
        logs.append(outputDir+"/log_makeHemisData"+y+"_"+tagID)
    


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeInputMixSamples] make4bHemis  Done" | sendmail '+o.email,doRun)



# 
#  Make the 3b sample with the stats of the 4b sample
#
if o.subSample3bSignal:
    # In the following "3b" refers to 3b subsampled to have the 4b statistics

    dag_config = []
    condor_jobs = []

    # histName = "hists.root"
    histName = "hists_"+tagID+".root " 

    for y in years:
        
        picoOut = " -p picoAOD_3bSubSampled_"+tagID+".root "
        h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
        histOut = " --histFile hists_3bSubSampled"+tagID+".root"

        for sig in signalSamples:
        
            cmd = runCMD+" -i "+outputDirNom+"/fileLists/"+sig+y+"_noMjj_"+tagID+".txt" + picoOut + " -o "+getOutDir() + MCyearOpts[y] +h10 + histOut + " -j "+jcmFileList[y]+" --emulate4bFrom3b --emulationOffset 0 --noDiJetMassCutInPicoAOD  "
            
            condor_jobs.append(makeCondorFile(cmd, "None", sig+y+"_"+tagID, outputDir=outputDir, filePrefix="subSample3bSignal_"))                    


    dag_config.append(condor_jobs)


    execute("rm "+outputDir+"subSample3bSignal_All.dag", doRun)
    execute("rm "+outputDir+"subSample3bSignal_All.dag.*", doRun)

    dag_file = makeDAGFile("subSample3bSignal_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



# 
#  Separate 3b and 4b for data vs ttbar training
#
if o.inputsForDataVsTT:
    # In the following "3b" refers to 3b subsampled to have the 4b statistics

    dag_config = []
    condor_jobs = []

    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "

    pico4b    = "picoAOD_4b_"+tagID+".root"
    pico3b    = "picoAOD_3b_"+tagID+".root"

    picoOut4b = " -p " + pico4b + " "
    histOut4b = " --histFile hists_4b_"+tagID+".root"


    picoOut3b = " -p " + pico3b + " " 
    histOut3b = " --histFile hists_3b_"+tagID+".root"


    for y in years:
        
        #
        #  4b 
        #
        cmd = runCMD+" -i "+outputDirNom+"/fileLists/data"+y+"_"+tagID+".txt"+ picoOut4b + " -o "+getOutDir()+ yearOpts[y]+  h10+  histOut4b + " --noDiJetMassCutInPicoAOD --skip3b "
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="inputsForDataVsTT_4b_"))

        #
        #  3b
        #
        cmd = runCMD+" -i "+outputDirNom+"/fileLists/data"+y+"_"+tagID+".txt"+ picoOut3b + " -o "+getOutDir()+ yearOpts[y]+  h10+  histOut3b + " --noDiJetMassCutInPicoAOD --skip4b "
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="inputsForDataVsTT_3b_"))


        for tt in ttbarSamples:
            
            #
            # 4b
            #
            cmd = runCMD+" -i "+outputDirNom+"/fileLists/"+tt+y+"_noMjj_"+tagID+".txt" + picoOut4b + " -o "+getOutDir() + MCyearOpts[y] +h10 + histOut4b + " --noDiJetMassCutInPicoAOD --skip3b "
            if o.doTTbarPtReweight:
                cmd += " --doTTbarPtReweight "

            condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="inputsForDataVsTT_4b_"))                    

            #
            # 3b
            #
            cmd = runCMD+" -i "+outputDirNom+"/fileLists/"+tt+y+"_noMjj_"+tagID+".txt" + picoOut3b + " -o "+getOutDir() + MCyearOpts[y] +h10 + histOut3b + " --noDiJetMassCutInPicoAOD --skip4b "
            if o.doTTbarPtReweight:
                cmd += " --doTTbarPtReweight "

            condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="inputsForDataVsTT_3b_"))                    


#    dag_config.append(condor_jobs)

    #
    #  Convert root to h5
    #
    condor_jobs = []
    pico4b_h5 = "picoAOD_4b_"+tagID+".h5"
    pico3b_h5 = "picoAOD_3b_"+tagID+".h5"


    for y in years:

        cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"_"+tagID+"/"+pico4b+"  -o "+pico4b_h5
        condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="inputsForDataVsTT_convert_4b_"))

        cmd = convertToH5JOB+" -i "+getOutDir()+"/data"+y+"_"+tagID+"/"+pico3b+"  -o "+pico3b_h5
        condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="inputsForDataVsTT_convert_3b_"))

        
#        for tt in ttbarSamples:
#            cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+y+"_noMjj_"+tagID+"/"+pico4b+"  -o "+pico4b_h5
#            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID, outputDir=outputDir, filePrefix="inputsForDataVsTT_convert_4b_"))
#
#            cmd = convertToH5JOB+" -i "+getOutDir()+"/"+tt+y+"_noMjj_"+tagID+"/"+pico3b+"  -o "+pico3b_h5
#            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID, outputDir=outputDir, filePrefix="inputsForDataVsTT_convert_3b_"))

    dag_config.append(condor_jobs)    


    execute("rm "+outputDir+"inputsForDataVsTT_All.dag", doRun)
    execute("rm "+outputDir+"inputsForDataVsTT_All.dag.*", doRun)

    dag_file = makeDAGFile("inputsForDataVsTT_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


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

    
    picoAOD_4b = "picoAOD_4b_"+tagID+".h5"
    picoAOD_3b = "picoAOD_3b_"+tagID+".h5"


    for y in years:
        mkdir(outputDir+"/"+"data"+y+"_"+tagID, doExecute=doRun)

        #if not o.doTTbarPtReweight:
        cpLocal(EOSOUTDIR, outputDir, "data"+y+"_"+tagID, picoAOD_4b)
        cpLocal(EOSOUTDIR, outputDir, "data"+y+"_"+tagID, picoAOD_3b)

        for tt in ttbarSamples:
            mkdir(outputDir+"/"+tt+y+"_"+tagID, doExecute=doRun)
            cpLocal(EOSOUTDIR, outputDir, tt+y+"_"+tagID, picoAOD_4b)
            cpLocal(EOSOUTDIR, outputDir, tt+y+"_"+tagID, picoAOD_3b)





# 
#  Copy to AUTON
#
if o.copyToAuton or o.makeAutonDirs or o.copyFromAuton:
    
    import os
    #autonAddr = "jalison@lop2.autonlab.org"
    autonAddr = "gpu13"
    dirName = "inputs"
    
    
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


    
    #
    # Setup directories
    #
    if o.makeAutonDirs:

        runA("mkdir hh4b/closureTests/")
        runA("mkdir hh4b/closureTests/"+dirName)
    
        for y in ["2018","2017","2016"]:
            runA("mkdir hh4b/closureTests/"+dirName+"/data"+y+"_"+tagID)
    
            for tt in ttbarSamples:
                runA("mkdir hh4b/closureTests/"+dirName+"/"+tt+y+"_"+tagID)
    
    #
    # Copy Files
    #
    if o.copyToAuton:
        for y in ["2018","2017","2016"]:
            scp("closureTests/mixed/data"+y+"_"+tagID+"/picoAOD_3b_"+tagID+".h5", "closureTests/"+dirName+"/data"+y+"_"+tagID+"/picoAOD_3b_"+tagID+".h5")
            scp("closureTests/mixed/data"+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5", "closureTests/"+dirName+"//data"+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5")

            for tt in ttbarSamples:
                scp("closureTests/mixed/"+tt+y+"_"+tagID+"/picoAOD_3b_"+tagID+".h5", "closureTests/"+dirName+"/"+tt+y+"_"+tagID+"/picoAOD_3b_"+tagID+".h5")
                scp("closureTests/mixed/"+tt+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5", "closureTests/"+dirName+"/"+tt+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5")


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



#    #
#    # Copy Files
#    #
#    if o.copyFromAuton:
#        for y in ["2018","2017","2016"]:
#            scpFrom("closureTests/"+combinedDirName+"/data"+y+"_"+tagID+"/picoAOD_3b_wJCM_"+tagID+".h5")
#            scpFrom("closureTests/"+combinedDirName+"//data"+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5")
#
#            for tt in ttbarSamples:
#                scpFrom("closureTests/"+combinedDirName+"/"+tt+y+"_"+tagID+"/picoAOD_3b_wJCM_"+tagID+".h5")
#                #scpFrom("closureTests/"+combinedDirName+"/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll/picoAOD_"+o.mixedName+"_4b_"+tagID+"_vAll.h5")
#                scpFrom("closureTests/"+combinedDirName+"/"+tt+y+"_PSData_"+tagID+"/picoAOD_4b_PSData_"+tagID+".h5")
#                scpFrom("closureTests/"+combinedDirName+"/"+tt+y+"_noPSData_"+tagID+"/picoAOD_4b_noPSData_"+tagID+".h5")
#
#
#            for s in subSamples:
#                scpFrom("closureTests/"+combinedDirName+"/data"+y+"_"+o.mixedName+"_"+tagID+"_v"+s+"/picoAOD_"+o.mixedName+"_4b_"+tagID+"_v"+s+".h5")



#
# Copy h5 files to EOS
#
if o.copyWeightsToEOS:


    def cpEOS(condorBase, localBase, subDir, picoH5):
        cmd  = "xrdcp -f "+localBase+"/"+subDir+"/"+picoH5+" "+condorBase+"/"+subDir+"/"+picoH5+" "
    
        if doRun:
            os.system(cmd)
        else:
            print cmd

    
    #picoAOD_4b = "picoAOD_4b_"+tagID+".h5"
    #picoAOD_3b = "picoAOD_3b_"+tagID+"_DvT3.h5"
    picoAOD_3b = "picoAOD_3b_"+tagID+"_DvT3_with_rwTT.h5"
    #picoAOD_3b_tt = "picoAOD_3b_"+tagID+"_DvT3_rwTT.h5"
    picoAOD_4b = "picoAOD_4b_"+tagID+"_DvT4_no_rwTT.h5"

    for y in years:

        #cpEOS(EOSOUTDIR, outputDir, "data"+y+"_"+tagID, picoAOD_3b)
        cpEOS(EOSOUTDIR, outputDir, "data"+y+"_"+tagID, picoAOD_4b)

        for tt in ttbarSamples:

            #cpEOS(EOSOUTDIR, outputDir, tt+y+"_"+tagID, picoAOD_3b)
            cpEOS(EOSOUTDIR, outputDir, tt+y+"_"+tagID, picoAOD_4b)



#
# Convert hdf5 to root
#
if o.writeOutDvT3Weights: 

    dag_config = []
    condor_jobs = []

    weightList = "_3b_with_rwTT_pt3"
    #picoAOD = "picoAOD_4b.h5"
    #picoAOD = "picoAOD_3b_"+tagID+"_DvT3.h5"

    picoAOD_3b = "picoAOD_3b_"+tagID+"_DvT3_with_rwTT.h5"
    #picoAOD_3b_tt = "picoAOD_3b_"+tagID+"_rwTT_DvT3_rwTT.h5"

    for y in years:
        cmd = convertToROOTWEIGHTFILE+" -i "+getOutDir()+"/data"+y+"_"+tagID+"/"+picoAOD_3b+"     --classifierName DvT3   --fvtNameList "+weightList
        condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="writeOutDvTWeights_3b_"))


        for tt in ttbarSamples:
            cmd = convertToROOTWEIGHTFILE+" -i "+getOutDir()+"/"+tt+y+"_"+tagID+"/"+picoAOD_3b+"    --classifierName DvT3      --fvtNameList "+weightList
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID, outputDir=outputDir, filePrefix="writeOutDvTWeights_3b_"))



    dag_config.append(condor_jobs)


    execute("rm "+outputDir+"writeOutDvTWeights_All.dag", doRun)
    execute("rm "+outputDir+"writeOutDvTWeights_All.dag.*", doRun)


    dag_file = makeDAGFile("writeOutDvTWeights_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
# Convert hdf5 to root
#
if o.writeOutDvT4Weights: 

    dag_config = []
    condor_jobs = []


    #picoAOD = "picoAOD_4b.h5"
    #picoAOD = "picoAOD_3b_"+tagID+"_DvT3.h5"

    #picoAOD_4b = "picoAOD_4b_"+tagID+"_DvT4_with_rwTT.h5"
    #weightList = "_4b_with_rwTT_pt3"
    
    picoAOD_4b = "picoAOD_4b_"+tagID+"_DvT4_no_rwTT.h5"
    weightList = "_4b_no_rwTT_pt3"

    for y in years:
        cmd = convertToROOTWEIGHTFILE+" -i "+getOutDir()+"/data"+y+"_"+tagID+"/"+picoAOD_4b+"     --classifierName DvT4   --fvtNameList "+weightList
        condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="writeOutDvTWeights_4b_"))


        for tt in ttbarSamples:
            cmd = convertToROOTWEIGHTFILE+" -i "+getOutDir()+"/"+tt+y+"_"+tagID+"/"+picoAOD_4b+"    --classifierName DvT4      --fvtNameList "+weightList
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIR, tt+y+"_"+tagID, outputDir=outputDir, filePrefix="writeOutDvTWeights_4b_"))



    dag_config.append(condor_jobs)


    execute("rm "+outputDir+"writeOutDvTWeights_All.dag", doRun)
    execute("rm "+outputDir+"writeOutDvTWeights_All.dag.*", doRun)


    dag_file = makeDAGFile("writeOutDvTWeights_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




#
# Make Input file lists
#
if o.makeDvT3FileLists:
    
    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    weightName = ""

    for y in years:

        #
        #  3b 
        #
        picoName3b    = "picoAOD_3b_"+tagID+".root"

        #picoNameDvT3b = "picoAOD_3b_"+tagID+"_DvT3_weights_.root"
        #picoNameDvT3b_tt = "picoAOD_3b_"+tagID+"_DvT3_weights_.root"

        picoNameDvT3b = "picoAOD_3b_"+tagID+"_DvT3_no_rwTT_weights_.root"
        #picoNameDvT3b_tt = "picoAOD_3b_"+tagID+"_rwTT_DvT3_rwTT_weights_.root"

        fileList = outputDirNom+"/fileLists/data"+y+"_"+tagID+"_3b.txt"    
        run("rm "+fileList)
        run("echo "+EOSOUTDIR+"/data"+y+"_"+tagID+"/"+picoName3b+" >> "+fileList)

        fileList = outputDirNom+"/fileLists/data"+y+"_"+tagID+"_3b_DvT3_no_rwTT.txt"    
        run("rm "+fileList)
        run("echo "+EOSOUTDIR+"/data"+y+"_"+tagID+"/"+picoNameDvT3b+" >> "+fileList)

        for tt in ttbarSamples:

            fileList = outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_3b.txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIR+"/"+tt+y+"_noMjj_"+tagID+"/"+picoName3b+" >> "+fileList)

            fileList = outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_3b_DvT3_no_rwTT.txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIR+"/"+tt+y+"_"+tagID+"/"+picoNameDvT3b+" >> "+fileList)




#
# Make Input file lists
#
if o.makeDvT4FileLists:
    
    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    weightName = ""

    for y in years:

        picoName4b    = "picoAOD_4b_"+tagID+".root"
        picoNameDvT4b = "picoAOD_4b_"+tagID+"_DvT4_with_rwTT_weights_.root"

        fileList = outputDirNom+"/fileLists/data"+y+"_"+tagID+"_4b.txt"    
        run("rm "+fileList)
        run("echo "+EOSOUTDIR+"/data"+y+"_"+tagID+"/"+picoName4b+" >> "+fileList)

        fileList = outputDirNom+"/fileLists/data"+y+"_"+tagID+"_4b_DvT4_with_rwTT.txt"    
        run("rm "+fileList)
        run("echo "+EOSOUTDIR+"/data"+y+"_"+tagID+"/"+picoNameDvT4b+" >> "+fileList)

        for tt in ttbarSamples:

            fileList = outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_4b.txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIR+"/"+tt+y+"_noMjj_"+tagID+"/"+picoName4b+" >> "+fileList)

            fileList = outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_4b_DvT4_with_rwTT.txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIR+"/"+tt+y+"_"+tagID+"/"+picoNameDvT4b+" >> "+fileList)





# 
#  Test DvT Weights
#
if o.testDvTWeights:
    # In the following "3b" refers to 3b subsampled to have the 4b statistics

    dag_config = []
    condor_jobs = []

    histDetail3b        = " --histDetailLevel allEvents.passPreSel.passMDRs.threeTag.failrWbW2 "
    histDetail4b        = " --histDetailLevel allEvents.passPreSel.passMDRs.fourTag.failrWbW2 "

    picoOut = " -p None " 


    histName3b = "hists_3b_"+tagID+".root"
    histName4b = "hists_4b_"+tagID+".root"

    if o.doTTbarPtReweight:
        histName3b = "hists_3b_"+tagID+"_rwTT.root"
        histName4b = "hists_4b_"+tagID+"_rwTT.root"

    if o.doDvTReweight:
        if o.doTTbarPtReweight:
            histName3b = "hists_3b_"+tagID+"_rwDvT_with_rwTT.root"
            histName4b = "hists_4b_"+tagID+"_rwDvT_with_rwTT.root"
        else:
            histName3b = "hists_3b_"+tagID+"_rwDvT_no_rwTT.root"
            histName4b = "hists_4b_"+tagID+"_rwDvT_no_rwTT.root"

    histOut3b = " --histFile "+histName3b
    histOut4b = " --histFile "+histName4b



    for y in years:
        
        ##
        ##  4b 
        ##
        inputFile = " -i  "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_4b.txt "
        if o.doTTbarPtReweight:
            inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_4b_DvT4_with_rwTT.txt "
            DvTName4b = " --reweightDvTName weight_DvT4_4b_with_rwTT_pt3 "
        else:
            inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_4b_DvT4_no_rwTT.txt "
            DvTName4b = " --reweightDvTName weight_DvT4_4b_no_rwTT_pt3 "            


        cmd = runCMD+ inputFile + inputWeights + DvTName4b + picoOut + " -o "+getOutDir()+ yearOpts[y]+  histDetail4b+  histOut4b + " --noDiJetMassCutInPicoAOD  "

        if o.doDvTReweight:
            cmd += " --doDvTReweight "

        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="testDvTWeights_4b_"))


        #
        #  3b
        #
        inputFile = " -i  "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_3b.txt "
        if o.doTTbarPtReweight:
            inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_3b_DvT3_with_rwTT.txt "
            DvTName3b = " --reweightDvTName weight_DvT3_3b_with_rwTT_pt3 "
        else:
            inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_3b_DvT3_no_rwTT.txt "
            DvTName3b = " --reweightDvTName weight_DvT3_3b_rwTT_pt3 "            

        cmd = runCMD+ inputFile + inputWeights + DvTName3b + picoOut + " -o "+getOutDir()+ yearOpts[y]+  histDetail3b+  histOut3b + " --noDiJetMassCutInPicoAOD  "

        if o.doDvTReweight:
            cmd += " --doDvTReweight "

        if not o.no3b: condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID, outputDir=outputDir, filePrefix="testDvTWeights_3b_"))

        #
        # Only to ttbare if we are not doing the DvT Weighting
        #
        if not o.doDvTReweight:

            for tt in ttbarSamples:
                
                #
                # 4b
                #
                inputFile = " -i  "+outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_4b.txt "
                if o.doTTbarPtReweight:
                    inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_4b_DvT4_with_rwTT.txt "
                else:
                    inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_4b_DvT4_no_rwTT.txt "

    
                cmd = runCMD+ inputFile + inputWeights + DvTName4b + picoOut + " -o "+getOutDir() + MCyearOpts[y] +histDetail4b + histOut4b + " --noDiJetMassCutInPicoAOD  "
    
                if o.doTTbarPtReweight:
                    cmd += " --doTTbarPtReweight "
    
                condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="testDvTWeights_4b_"))                    
    
                #
                # 3b
                #
                inputFile = " -i  "+outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_3b.txt "
                if o.doTTbarPtReweight:
                    inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_3b_DvT3_with_rwTT.txt "
                else:
                    inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/"+tt+y+"_"+tagID+"_3b_DvT3_no_rwTT.txt "

                cmd = runCMD+ inputFile + inputWeights + DvTName3b + picoOut + " -o "+getOutDir() + MCyearOpts[y] +histDetail3b + histOut3b + " --noDiJetMassCutInPicoAOD  "
    
                if o.doTTbarPtReweight:
                    cmd += " --doTTbarPtReweight "
    
                if not o.no3b: condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID, outputDir=outputDir, filePrefix="testDvTWeights_3b_"))                    
    

    dag_config.append(condor_jobs)


    #
    #  Hadd ttbar
    #
    if not o.doDvTReweight:
        condor_jobs = []
    
        for y in years:
            
            cmd = "hadd -f "+ getOutDir()+"/TT"+y+"/"+histName3b+" "
            for tt in ttbarSamples:        
                cmd += getOutDir()+"/"+tt+y+"_"+tagID+"_3b/"+histName3b+" "
            if not o.no3b: condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y, outputDir=outputDir, filePrefix="testDvTWeights_3b_"))            
    
            cmd = "hadd -f "+ getOutDir()+"/TT"+y+"/"+histName4b+" "
            for tt in ttbarSamples:        
                cmd += getOutDir()+"/"+tt+y+"_"+tagID+"_4b/"+histName4b+" "
    
            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y, outputDir=outputDir, filePrefix="testDvTWeights4b_"))            
    
    
    
    
        dag_config.append(condor_jobs)
        

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        if not o.doDvTReweight:
            cmd = "hadd -f " + getOutDir()+"/TTRunII/"+ histName3b+" "
            for y in years:
                cmd += getOutDir()+"/TT"+y+"/"  +histName3b+" "
    
            if not o.no3b: condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII", outputDir=outputDir, filePrefix="testDvTWeights_3b_"))            
    
            cmd = "hadd -f " + getOutDir()+"/TTRunII/"+ histName4b+" "
            for y in years:
                cmd += getOutDir()+"/TT"+y+"/"  +histName4b+" "
    
            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII", outputDir=outputDir, filePrefix="testDvTWeights_4b_"))            
    


        cmd = "hadd -f " + getOutDir()+"/dataRunII/"+ histName3b+" "
        for y in years:
            cmd += getOutDir()+"/data"+y+"_"+tagID+"_3b/"  +histName3b+" "

        if not o.no3b: condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII", outputDir=outputDir, filePrefix="testDvTWeights_3b_"))            


        cmd = "hadd -f " + getOutDir()+"/dataRunII/"+ histName4b+" "
        for y in years:
            cmd += getOutDir()+"/data"+y+"_"+tagID+"_4b/"  +histName4b+" "

        condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII", outputDir=outputDir, filePrefix="testDvTWeights_4b_"))            

        dag_config.append(condor_jobs)            



    #
    # Subtract QCD 
    #
    if not o.doDvTReweight:

        condor_jobs = []
    
        cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
        cmd += " -d "+getOutDir()+"/dataRunII/"+histName3b
        cmd += " --tt "+getOutDir()+"/TTRunII/"+histName3b
        cmd += " -q "+getOutDir()+"/QCDRunII/"+histName3b
        if not o.no3b: condor_jobs.append(makeCondorFile(cmd, getOutDir(), "QCDRunII", outputDir=outputDir, filePrefix="testDvTWeights_3b_") )


        cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
        cmd += " -d "+getOutDir()+"/dataRunII/"+histName4b
        cmd += " --tt "+getOutDir()+"/TTRunII/"+histName4b
        cmd += " -q "+getOutDir()+"/QCDRunII/"+histName4b
        condor_jobs.append(makeCondorFile(cmd, getOutDir(), "QCDRunII", outputDir=outputDir, filePrefix="testDvTWeights_4b_") )

    
        dag_config.append(condor_jobs)


    execute("rm "+outputDir+"testDvTWeights_All.dag", doRun)
    execute("rm "+outputDir+"testDvTWeights_All.dag.*", doRun)

    dag_file = makeDAGFile("testDvTWeights_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
if o.doWeightsQCD:

    cmds = []

    mkdir(outputDir+"/weights", doRun)

    # histName = "hists.root"
    histName = "hists_"+tagID+".root " 


    #/store/user/jda102/condor/mixed/QCDRunII/
    dataFile3b = getOutDir()+"/QCDRunII/hists_3b_"+tagID+".root"
    dataFile4b = getOutDir()+"/QCDRunII/hists_4b_"+tagID+".root"

    cmd  = weightCMD
    cmd += " -d "+dataFile3b
    cmd += " --data4b "+dataFile4b
    #cmd += " -c passPreSel   -o "+outputDir+"/weights/noTT_data"+y+"_"+tagID+"_PreSel/  -r SB -w 01-00-00"
    cmd += " -c passMDRs   -o "+outputDir+"/weights/QCDRunII_"+tagID+"_MRDs/  -r SB -w 02-00-00"
    
    cmds.append(cmd)


    babySit(cmds, doRun)


# 
#  Make the 3b sample with the stats of the 4b sample
#
if o.subSample3bQCD:
    # In the following "3b" refers to 3b subsampled to have the 4b statistics

    dag_config = []
    condor_jobs = []

    for s in subSamples:

        for y in years:
            
            picoOut = " -p picoAOD_3bSubSampled_"+tagID+"_v"+s+".root "
            h10        = " --histDetailLevel allEvents.passPreSel.passMDRs.threeTag "
            histOut = " --histFile hists_"+tagID+"_v"+s+".root"

            inputFile = " -i  "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_3b.txt "
            inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_3b_DvT3_with_rwTT.txt "
            DvTName3b = " --reweightDvTName weight_DvT3_3b_with_rwTT_pt3 "


            cmd = runCMD+ inputFile + inputWeights + DvTName3b + picoOut + " -o "+getOutDir()+ yearOpts[y]+  h10+  histOut + " -j closureTests/mixed/weights/QCDRunII_"+tagID+"_PreSel/jetCombinatoricModel_SB_02-01-00.txt --emulate4bFrom3b --emulationOffset "+s+" --noDiJetMassCutInPicoAOD "
            cmd += " --doDvTReweight "
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="subSample3bQCD_"))


    dag_config.append(condor_jobs)


    execute("rm "+outputDir+"subSample3bQCD_All.dag", doRun)
    execute("rm "+outputDir+"subSample3bQCD_All.dag.*", doRun)

    dag_file = makeDAGFile("subSample3bQCD_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
#   Make inputs fileLists
#
if o.makeInputFileListsSubSampledQCD:

    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd

    eosDir = "root://cmseos.fnal.gov//store/user/johnda/condor/mixed/"

    for s in subSamples:

        for y in years:

            for sample in ["data"]:
                
                fileList = outputDir+"/fileLists/"+sample+y+"DvT_"+tagID+"_v"+s+".txt"    
                run("rm "+fileList)

                subdir = sample+y+"_"+tagID+"_3b"

                picoName = "picoAOD_3bSubSampled_"+tagID+"_v"+s+".root"
                
                run("echo "+eosDir+"/"+subdir+"/"+picoName+" >> "+fileList)



#
# Make Hemisphere library from all hemispheres
#   (Should run locally)
if o.make4bHemisWithDvT:
    
    cmds = []
    logs = []

    picoOut = "  -p 'None' "
    h1     = " --histDetailLevel allEvents.threeTag.fourTag "
    histOut = " --histFile hists_"+tagID+"_wDvT.root " 

    for y in years:
        inputFile = " -i  "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_4b.txt "

        if o.doTTbarPtReweight:
            inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_4b_DvT4_with_rwTT.txt "
            DvTName4b = " --reweightDvTName weight_DvT4_4b_with_rwTT_pt3 "
        else:
            inputWeights = " --inputWeightFilesDvT "+outputDirNom+"/fileLists/data"+y+"_"+tagID+"_4b_DvT4_no_rwTT.txt "
            DvTName4b = " --reweightDvTName weight_DvT4_4b_no_rwTT_pt3 "            

        
        cmd = runCMD+ inputFile + inputWeights + DvTName4b + picoOut + " -o "+os.getcwd()+"/"+outputDir+"/dataHemisDvT_"+tagID+ yearOpts[y]+  h1 +  histOut + " --createHemisphereLibrary --doDvTReweight"

        cmds.append(cmd)
        logs.append(outputDir+"/log_makeHemisData"+y+"_"+tagID)
    


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeInputMixSamples] make4bHemis  Done" | sendmail '+o.email,doRun)


if o.make4bHemiTarballDvT:

    for y in years:
    #tar -C closureTests/mixed/dataHemis_b0p60p3 -zcvf closureTests/mixed/data2018_b0p60p3_hemis.tgz data2018_b0p60p3  --exclude="hist*root" --exclude-vcs --exclude-caches-all

        tarballName = 'data'+y+'_'+tagID+'_hemisDvT.tgz'
        localTarball = outputDir+"/"+tarballName

        cmd  = 'tar -C '+outputDir+"/dataHemisDvT_"+tagID+' -zcvf '+ localTarball +' data'+y+'_'+tagID+"_4b"
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
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
if o.mixInputsDvT:

    dag_config = []
    condor_jobs = []

    mixedName = "3bDvTMix4b"

    for s in subSamples:

        for y in years:

            picoOut    = " -p picoAOD_"+mixedName+"_"+tagID+"_v"+s+".root "
            h10        = " --histDetailLevel passPreSel.passMDRs.threeTag.fourTag "
            histOut    = " --histFile hists_"+mixedName+"_"+tagID+"_v"+s+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 1000000 "
            if o.condor:
                hemiLoad += '--inputHLib3Tag \\"NONE\\" --inputHLib4Tag \\"data'+y+'_'+tagID+'_4b/hemiSphereLib_4TagEvents_*root\\"'
            else:
                hemiLoad += '--inputHLib3Tag "NONE" --inputHLib4Tag "data'+y+'_'+tagID+'_4b/hemiSphereLib_4TagEvents_*root"'                

            #
            #  Data
            #
            inFileList = outputDir+"/fileLists/data"+y+"DvT_"+tagID+"_v"+s+".txt"

            # The --is3bMixed here just turns off blinding of the data
            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --is3bMixed "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="mixInputs_", 
                                                        HEMINAME="data"+y+"_"+tagID+"_hemisDvT", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/data"+y+"_"+tagID+"_hemisDvT.tgz"))
    
    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+"mixInputs_All.dag", doRun)
    execute("rm "+outputDir+"mixInputs_All.dag.*", doRun)

    dag_file = makeDAGFile("mixInputs_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)
