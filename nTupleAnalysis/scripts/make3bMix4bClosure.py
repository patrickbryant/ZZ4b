
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9", help="Year or comma separated list of subsamples")
parser.add_option('-w',            action="store_true", dest="doWeights",      default=False, help="Fit jetCombinatoricModel and nJetClassifier TSpline")
parser.add_option('--doWeightsOneFvTFit',            action="store_true", help="Fit jetCombinatoricModel and nJetClassifier TSpline")
parser.add_option('--histsForJCM',  action="store_true",      help="Make hist.root for JCM")
parser.add_option('--histsForJCMOneFvTFit',  action="store_true",      help="Make hist.root for JCM when fitting all vX samples together")
parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--mixInputs',    action="store_true",      help="Make Mixed Samples")
parser.add_option('--makeInputFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--plotUniqueHemis',    action="store_true",      help="Do Some Mixed event analysis")
parser.add_option('--histsWithJCM', action="store_true",      help="Make hist.root with JCM")
parser.add_option('--plotsWithJCM', action="store_true",      help="Make pdfs with JCM")
parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--histsNoFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsNoFvT', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--histsRW', action="store_true",      help="Make hist.root with RW")
parser.add_option('--plotsRW', action="store_true",      help="Make pdfs with RW")
parser.add_option('--cutFlowBeforeJCM', action="store_true",      help="Make 4b cut flow before JCM")
parser.add_option('--makeInputsForCombine', action="store_true",      help="Make inputs for the combined tool")
parser.add_option('--moveFinalPicoAODsToEOS', action="store_true",      help="Move Final AODs to EOS")
parser.add_option('--cleanFinalPicoAODsToEOS', action="store_true",      help="Move Final AODs to EOS")
parser.add_option('--haddSubSamples', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--haddSubSamplesRW', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--scaleCombSubSamples', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--scaleCombSubSamplesOneFvTFit', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--scaleCombSubSamplesRW', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--plotsCombinedSamples', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--plotsCombinedSamplesRW', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--makeInput3bWeightFiles',  action="store_true",      help="make Input file lists")
parser.add_option('--writeOutMixedToUnmixedWeights',  action="store_true",      help="")
parser.add_option('--makeMixedToUnMixedInputFiles',  action="store_true",      help="")
parser.add_option('--histsWithMixedUnMixedWeights',  action="store_true",      help="")
parser.add_option('--email',            default=None,      help="")
parser.add_option('-c',   '--condor',   action="store_true", default=False,           help="Run on condor")


o, a = parser.parse_args()

doRun = o.execute
years = o.year.split(",")
subSamples = o.subSamples.split(",")
# mixed
mixedName=o.mixedName

outputDir="closureTests/"+mixedName+"/"
mkdir(outputDir, doExecute=True)

outputDirNom="closureTests/nominal/"
outputDirMix="closureTests/mixed/"
#outputDirComb="closureTests/combined/"
outputDirComb="closureTests/combined_"+mixedName+"/"



# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'
mixedAnalysisCMD='mixedEventAnalysis ZZ4b/nTupleAnalysis/scripts/mixedEventAnalysis_cfg.py'
convertToROOTJOB = 'python ZZ4b/nTupleAnalysis/scripts/convert_h52rootWeightFile.py'

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



plotOpts = {}
plotOpts["2018"]=" -l 60.0e3 -y 2018"
plotOpts["2017"]=" -l 36.7e3 -y 2017"
plotOpts["2016"]=" -l 35.9e3 -y 2016"
plotOpts["RunII"]=" -l 132.6e3 -y RunII"

#tagID = "b0p6"
tagID = "b0p60p3"

from condorHelpers import *


CMSSW = getCMSSW()
USER = getUSER()
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+mixedName+"/"
EOSOUTDIRMIXED = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/mixed/"
EOSOUTDIRNOM = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/nominal/"
EOSOUTDIRCOMB = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/combined_"+mixedName+"/"
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


def getOutDirComb():
    if o.condor:
        return EOSOUTDIRCOMB
    return outputDirComb


if o.condor:
    print "Making Tarball"
    makeTARBALL(o.execute)




#
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
if o.mixInputs:

    dag_config = []
    condor_jobs = []

    for s in subSamples:

        for y in years:

            picoOut    = " -p picoAOD_"+mixedName+"_"+tagID+"_v"+s+".root "
            h10        = " --histogramming 10 "
            histOut    = " --histFile hists_"+mixedName+"_"+tagID+"_v"+s+".root "
            hemiLoad   = " --loadHemisphereLibrary --maxNHemis 1000000 "
            if o.condor:
                hemiLoad += '--inputHLib3Tag \\"NONE\\" --inputHLib4Tag \\"data'+y+'_'+tagID+'/hemiSphereLib_4TagEvents_*root\\"'
            else:
                hemiLoad += '--inputHLib3Tag "NONE" --inputHLib4Tag "data'+y+'_'+tagID+'/hemiSphereLib_4TagEvents_*root"'                

            #
            #  Data
            #
            inFileList = outputDirMix+"/fileLists/data"+y+"_"+tagID+"_v"+s+".txt"

            # The --is3bMixed here just turns off blinding of the data
            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --is3bMixed "+hemiLoad
            condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="mixInputs_", 
                                                        HEMINAME="data"+y+"_"+tagID+"_hemis", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/data"+y+"_"+tagID+"_hemis.tgz"))
    

            for tt in ttbarSamples:
                fileListTT = outputDirMix+"/fileLists/"+tt+y+"_"+tagID+"_v"+s+".txt"
    
                cmd = runCMD+" -i "+fileListTT +" -o "+getOutDir()+ picoOut + MCyearOpts[y] + h10  + histOut + " --is3bMixed " + hemiLoad
                condor_jobs.append(makeCondorFileHemiMixing(cmd, "None", tt+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="mixInputs_",
                                                            HEMINAME="data"+y+"_"+tagID+"_hemis", HEMITARBALL="root://cmseos.fnal.gov//store/user/johnda/condor/data"+y+"_"+tagID+"_hemis.tgz"))                        
    

    dag_config.append(condor_jobs)


    execute("rm "+outputDir+"mixInputs_All.dag", doRun)
    execute("rm "+outputDir+"mixInputs_All.dag.*", doRun)

    dag_file = makeDAGFile("mixInputs_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
#  Make Input file Lists
# 
if o.makeInputFileLists:
    
    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    eosDirMixed = "root://cmseos.fnal.gov//store/user/johnda/condor/"+mixedName+"/"    

    for y in years:

        #
        #  Mixed Samples
        #
        for s in subSamples:
            fileList = outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+".txt"    
            run("rm "+fileList)
            run("echo "+eosDirMixed+"/data"+y+"_"+tagID+"_v"+s+"/picoAOD_"+mixedName+"_"+tagID+"_v"+s+".root >> "+fileList)

        for tt in ttbarSamples:
            fileList = outputDir+"/fileLists/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll.txt"    
            run("rm "+fileList)
            
            for s in subSamples:
                run("echo "+eosDirMixed+"/"+tt+y+"_"+tagID+"_v"+s+"/picoAOD_"+mixedName+"_"+tagID+"_v"+s+".root >> "+fileList)



#
#  Make Hists of mixed Datasets
#
if o.histsForJCM: 

    #
    #  Mixed data
    #
    for s in subSamples:

        dag_config = []
        condor_jobs = []

        histName = "hists_"+mixedName+"_"+tagID+"_v"+s+".root "

        for y in years:

            picoOut    = " -p NONE "
            h10        = " --histogramming 10 "
            inFileList = outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+".txt"
            histOut    = " --histFile "+histName

            # The --is3bMixed here just turns off blinding of the data
            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --is3bMixed --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsForJCM_v"+s))
                                   

        dag_config.append(condor_jobs)


        #
        #   Hadd years
        #
        if "2016" in years and "2017" in years and "2018" in years:
    
            mkdir(outputDir+"/dataRunII",   doRun)
            condor_jobs = []        

            cmd = "hadd -f " + getOutDir()+"/dataRunII_"+mixedName+"_"+tagID+"_v"+s+"/"+ histName+" "
            for y in years:
                cmd += getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/"  +histName+" "

        condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsForJCM_v"+s))            

        dag_config.append(condor_jobs)            


        execute("rm "+outputDir+"histsForJCM_v"+s+"_All.dag", doRun)
        execute("rm "+outputDir+"histsForJCM_v"+s+"_All.dag.*", doRun)

        dag_file = makeDAGFile("histsForJCM_v"+s+"_All.dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)


    #
    # Ttbar
    #
    dag_config = []
    condor_jobs = []

    histName = "hists_"+mixedName+"_"+tagID+"_vAll.root "

    for y in years:

        for tt in ttbarSamples:
            picoOut    = " -p NONE "
            h10        = " --histogramming 10 "
            inFileList = outputDir+"/fileLists/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll.txt"
            histOut    = " --histFile "+histName

            # The --is3bMixed here just turns off blinding of the data
            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + MCyearOpts[y] + h10 + histOut+" --is3bMixed --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID+"_vAll", outputDir=outputDir, filePrefix="histsForJCM_"))
                                   

    dag_config.append(condor_jobs)

    #
    #  Hadd ttbar
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
        for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll/"+histName
        condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_vAll", outputDir=outputDir, filePrefix="histsForJCM_"))

    dag_config.append(condor_jobs)

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/TTRunII",   doRun)
        condor_jobs = []        

        cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName+" "
        for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName+" "

        condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_vAll", outputDir=outputDir, filePrefix="histsForJCM_"))             
 
        dag_config.append(condor_jobs)            
 
 
    execute("rm "+outputDir+"histsForJCM_TT_All.dag",   doRun)
    execute("rm "+outputDir+"histsForJCM_TT_All.dag.*", doRun)
 
    dag_file = makeDAGFile("histsForJCM_TT_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)
 




#
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
if o.plotUniqueHemis:

    cmds = []
    logs = []

    for y in years:

        histOut = " --hist hMixedAnalysis_"+tagID+".root "
        cmds.append(mixedAnalysisCMD + " -i "+outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+".txt -o "+outputDir + histOut)
        logs.append(outputDir+"/log_mixAnalysis_data"+y+"_"+mixedName+"_"+tagID)
            
        #for s in subSamples:
        #
        #    cmds.append(mixedAnalysisCMD + " -i "+outputDir+"/data"+y+"_"+tagID+"_v"+s+"/picoAOD_"+mixedName+"_"+tagID+"_v"+s+".root -o "+outputDir+"/data"+y+"_"+tagID+"_v"+s+  histOut)
        #    logs.append(outputDir+"/log_mixAnalysis_"+y+"_"+mixedName+"_"+tagID+"_v"+s)            

    babySit(cmds, doRun, logFiles=logs)


#
#  Cut flow to comp TTBar Fraction
#
if o.cutFlowBeforeJCM:
    cmds = []
    logs = []

    yearsToPlot = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToPlot.append("RunII")

    for s in subSamples:

        histName = "hists_"+mixedName+"_"+tagID+"_v"+s+".root " 
        histName3b = "hists_"+tagID+".root"
        for y in years:
    
            #
            # MAke Plots
            #
            data4bFile  = outputDir+"/data"+y+"_"+tagID+"_v"+s+"/"+histName if not y == "RunII" else outputDir+"/data"+y+"/"+histName
            data3bFile  = outputDir+"/data"+y+"_"+tagID+"_v"+s+"/"+histName  if not y == "RunII" else outputDir+"/data"+y+"/"+histName
            ttbar4bFile = outputDir+"/TT"+y+"/"+histName
            ttbar3bFile = outputDir+"/TT"+y+"/"+histName

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
            cmd += " --d4 "+data4bFile
            cmd += " --d3 "+data3bFile
            cmd += " --t4 "+ttbar4bFile
            cmd += " --t3 "+ttbar3bFile
            cmd += " --name "+outputDir+"/CutFlow_4tagOnly_"+y+"_"+mixedName+"_"+tagID+"_v"+s
            cmd += " --makePDF "
            cmds.append(cmd)
            logs.append(outputDir+"/log_cutFlow_"+y+"_v"+s)

    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    for s in subSamples:
        for y in years:
            cmds.append("mv CutFlow_4tagOnly_"+y+"_"+mixedName+"_"+tagID+"_v"+s+".pdf "+outputDir+"/")
            
    babySit(cmds, doRun)    




#
#  Fit JCM
#
if o.doWeights:
    
    cmds = []
    logs = []

    mkdir(outputDir+"/weights", doRun)

    yearsToFit = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToFit.append("RunII")

    for s in subSamples:
        
        for y in yearsToFit:

            #histName4b = "hists_"+mixedName+"_v"+s+".root " 
            #histName3b = "hists.root "

            histName4b = "hists_"+mixedName+"_"+tagID+"_v"+s+".root "             
            histName4bTT = "hists_"+mixedName+"_"+tagID+"_vAll.root "             
            histName3b = "hists_"+tagID+".root "

            data3bFile  = getOutDirNom()+"/data"+y+"_"+tagID+"/"+histName3b     if not y == "RunII" else getOutDirNom()+"/data"+y+"/"+histName3b               
            data4bFile  = getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/"+histName4b #if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName4b                
            ttbar4bFile = getOutDir()+"/TT"+y+"/"+histName4bTT
            ttbar3bFile = getOutDirNom()+"/TT"+y+"/"+histName3b
            
            cmd = weightCMD
            cmd += " -d "+data3bFile
            cmd += " --data4b "+data4bFile
            cmd += " --tt "+ttbar3bFile
            cmd += " --tt4b "+ttbar4bFile
            cmd += " -c passMDRs   -o "+outputDir+"/weights/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/  -r SB -w 02-03-00 "+plotOpts[y]
            
            cmds.append(cmd)
            logs.append(outputDir+"/log_makeWeights_"+y+"_"+tagID+"_v"+s)

    babySit(cmds, doRun, logFiles=logs)



#
#  Adding JCM weights now done in makeClosureCombined
#



#
#  Make Hists with JCM weights applied
#
if o.histsWithJCM: 

    #
    #  Make Hists
    #
    cmds = []
    logs = []

    for s in subSamples:

        JCMName=mixedName+"_v"+s
    
        histName3b = "hists_3b_wJCM_"+JCMName+".root "
    
        for y in years:
    
            #
            # 3b
            #
            pico3b = "picoAOD_3b_wJCM.root"
            picoOut = " -p NONE "
            h10 = " --histogramming 10 --histDetail 7 "    
            histOut3b = " --histFile "+histName3b
    
            #
            #  3b Data
            # 
            cmds.append(runCMD+" -i "+outputDirComb+"/data"+y+"/"+pico3b+             picoOut  +   yearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName)    
            logs.append(outputDir+"/log_"+y+"_3b_wJCM_v"+s)

            #
            #  TTbar
            #
            for tt in ttbarSamples:
                cmds.append(runCMD+" -i "+outputDirComb+"/"+tt+y+"/"+pico3b+     picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName)    
                logs.append(outputDir+"/log_"+tt+y+"_3b_wJCM_v"+s)
    

    babySit(cmds, doRun, logFiles=logs)



    
    #
    #  Hadd TTbar
    #
    cmds = []
    logs = []
    for s in subSamples:

        JCMName=mixedName+"_v"+s
        histName3b = "hists_3b_wJCM_"+JCMName+".root "

        for y in years:
            cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName3b+" "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b+"  "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b+" "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b)
            logs.append(outputDir+"/log_haddTT_3b_wJCM_"+y+"_v"+s)
    
    babySit(cmds, doRun, logFiles=logs)



    #
    # Subtract QCD 
    #
    cmds = []
    logs = []
    for s in subSamples:

        JCMName=mixedName+"_v"+s
        histName3b = "hists_3b_wJCM_"+JCMName+".root "

        for y in years:
            mkdir(outputDir+"/QCD"+y+"_v"+s, doRun)
    
            cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
            cmd += " -d   "+outputDirComb+"/data"+y+"/"+histName3b
            cmd += " --tt "+outputDir+"/TT"+y+"/"+histName3b
            cmd += " -q   "+outputDir+"/QCD"+y+"_v"+s+"/"+histName3b
            cmds.append(cmd)
            
            logs.append(outputDir+"/log_SubTT"+y+"_v"+s)
    
    babySit(cmds, doRun, logFiles=logs)    



    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/TTRunII",   doRun)
        mkdir(outputDir+"/QCDRunII",  doRun)

        cmds = []
        logs = []
        
        for s in subSamples:
    
            JCMName=mixedName+"_v"+s
        
            histName3b = "hists_3b_wJCM_"+JCMName+".root "
            histName4b = "hists_"+mixedName+"_v"+s+".root " 
    
    
            cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName3b+" "+outputDirComb+"/data2016/"+histName3b+" "+outputDirComb+"/data2017/"+histName3b+" "+outputDirComb+"/data2018/"+histName3b)
            cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName4b+" "+outputDir+"/data2016_v"+s+"/"+histName4b+" "+outputDir+"/data2017_v"+s+"/"+histName4b+" "+outputDir+"/data2018_v"+s+"/"+histName4b)
            cmds.append("hadd -f "+outputDir+"/TTRunII/"  +histName4b+" "+outputDir+"/TT2016/"  +histName4b+" "+outputDir+"/TT2017/"  +histName4b+" "+outputDir+"/TT2018/"  +histName4b)
            cmds.append("hadd -f "+outputDir+"/QCDRunII/"  +histName3b+" "+outputDir+"/QCD2016_v"+s+"/"  +histName3b+" "+outputDir+"/QCD2017_v"+s+"/"  +histName3b+" "+outputDir+"/QCD2018_v"+s+"/"  +histName3b)
            cmds.append("hadd -f "+outputDir+"/TTRunII/"  +histName3b+" "+outputDir+"/TT2016/"  +histName3b+" "+outputDir+"/TT2017/"  +histName3b+" "+outputDir+"/TT2018/"  +histName3b)

            logs.append(outputDir+"/log_haddDataRunII_3b_v"+s)
            logs.append(outputDir+"/log_haddDataRunII_4b_v"+s)
            logs.append(outputDir+"/log_haddDataRunII_v"+s)
            logs.append(outputDir+"/log_haddQCDRunII_v"+s)
            logs.append(outputDir+"/log_haddDataRunII_TT_3b_v"+s)

        babySit(cmds, doRun, logFiles=logs)




    if o.email: execute('echo "Subject: [make3bMix4bClosure] makeHistsWithJCM Done" | sendmail '+o.email,doRun)


#
#  Make Plots with JCM
#
if o.plotsWithJCM:
    cmds = []
    logs = []

    yearsToPlot = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToPlot.append("RunII")

    for s in subSamples:

        JCMName=mixedName+"_v"+s
        
        histName3b = "hists_3b_wJCM_"+JCMName+".root "
        histName4b = "hists_"+mixedName+"_v"+s+".root " 

        for y in yearsToPlot:
    
            #
            # MAke Plots
            #
            qcdFile     = outputDir+"/QCD"+y+"_v"+s+"/"+histName3b  if not y == "RunII" else outputDir+"/QCD"+y+"/"+histName3b                
            data4bFile  = outputDir+"/data"+y+"_v"+s+"/"+histName4b if not y == "RunII" else outputDir+"/data"+y+"/"+histName4b                
            data3bFile  = outputDirComb+"/data"+y+"/"+histName3b    if not y == "RunII" else outputDir+"/data"+y+"/"+histName3b                
            ttbar4bFile = outputDir+"/TT"+y+"/"+histName4b
            ttbar3bFile = outputDir+"/TT"+y+"/"+histName3b

            
            #cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
            #cmd += " --d4 "+outputDir+"/data"+y+"_v"+s+"/"+histName4b
            #cmd += " --d3 "+outputDirComb+"/data"+y+"/"+histName3b
            #cmd += " --t4 "+outputDir+"/TT"+y+"/"+histName4b
            #cmd += " --t3 "+outputDir+"/TT"+y+"/"+histName3b
            #cmd += " --t4_s "+outputDir+"/TTToSemiLeptonic"+y+"_v"+s+"/"+histName4b
            #cmd += " --t4_h "+outputDir+"/TTToHadronic"+y+"_v"+s+"/"+histName4b
            #cmd += " --t4_d "+outputDir+"/TTTo2L2Nu"+y+"_v"+s+"/"+histName4b
            #cmd += " --t3_s "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b
            #cmd += " --t3_h "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b
            #cmd += " --t3_d "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b
            #cmd += " --name "+outputDir+"/CutFlow_wFvT_"+y+"_v"+s
            #cmd += " --makePDF -r"
            #cmds.append(cmd)
            #logs.append(outputDir+"/log_cutFlow_wFVT_"+y+"_v"+s)


            cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
            cmd += " --d4 "+data4bFile
            cmd += " --d3 "+data3bFile
            cmd += " --t4 "+ttbar4bFile
            cmd += " --t3 "+ttbar3bFile
            cmd += " --name "+outputDir+"/CutFlow_"+y+"_"+mixedName+"_v"+s
            cmd += " --makePDF "
            cmds.append(cmd)
            logs.append(outputDir+"/log_cutFlow_"+y+"_v"+s)


            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithJCM_"+y+"_v"+s+plotOpts[y]+" -m -j  --noSignal "
            cmd += " --qcd " +qcdFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)
            logs.append(outputDir+"/log_makePlots_wJCM_"+y+"_v"+s)
    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    for s in subSamples:
        for y in years:
            cmds.append("mv CutFlow_"+y+"_"+mixedName+"_v"+s+".pdf "+outputDir+"/")
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithJCM_"+y+"_v"+s+".tar plotsWithJCM_"+y+"_v"+s)
            
    babySit(cmds, doRun)    
            



# 
#  Tracining done in makeClosureTestCombinedTraining
#
    


#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithFvT: 

    weightPostFix = ""
    #weightPostFix = "_comb"

    pico3b = "picoAOD_3b_wJCM_"+tagID+".root"
    picoOut = " -p NONE "
    h10 = " --histogramming 10 --histDetail 7 "    
    outDir = " -o "+getOutDir()+" "

    histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "
    histOut4bTT = " --histFile "+histName4bTT

    for s in subSamples:
        dag_config = []
        condor_jobs = []

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix
    
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
    
        histOut3b = " --histFile "+histName3b
        histOut4b = " --histFile "+histName4b

        for y in years:
    
            #
            # 3b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+tagID+"_3b_wFvT.txt "
            cmd = runCMD + inputFile + outDir + picoOut  +  yearOpts[y] + h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))

            # 3b TTbar not needed... Run it anyway for cut flow
            for tt in ttbarSamples:
                inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_3b_wFvT.txt "
                
                cmd = runCMD + inputFile + outDir + picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName
                condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))

            #
            # 4b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT.txt"

            cmd = runCMD + inputFile + outDir +  picoOut  +   yearOpts[y]+ h10 + histOut4b + "  --FvTName "+FvTName + " --is3bMixed"
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+mixedName+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_4b_"))
            
            if s in ["0"]:
                for tt in ttbarSamples:
                    inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll_wFvT.txt"
                
                    cmd = runCMD + inputFile + outDir + picoOut  + MCyearOpts[y]+ h10 + histOut4bTT + "  --FvTName "+FvTName + " --is3bMixed"
                    condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+mixedName+"_"+tagID+"_vAll", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))

        dag_config.append(condor_jobs)

        #
        #  Hadd TTbar
        #
        condor_jobs = []

        for y in years:
            cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName3b+" "
            for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_"+tagID+"_3b_wFvT/"+histName3b+" "
                
            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))

            if s in ["0"]:    
                cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName4bTT+" "
                for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll_wFvT/"+histName4bTT

                condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_vAll", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))


        dag_config.append(condor_jobs)


        condor_jobs = []        

        #
        #   Hadd years
        #
        if "2016" in years and "2017" in years and "2018" in years:
    
            mkdir(outputDir+"/dataRunII", doRun)
            mkdir(outputDir+"/TTRunII",   doRun)
        
            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName3b+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))            

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName4b+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_4b_"))            

            if s in ["0"]:                
                cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName4bTT+" "
                for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName4bTT+" "
                condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_vAll", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))            


            cmd = "hadd -f "+getOutDir()+"/TTRunII/"  +histName3b+" "
            for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName3b+" "

            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))            


        dag_config.append(condor_jobs)

        execute("rm "+outputDir+"histsWithFvT_All_v"+s+".dag", doRun)
        execute("rm "+outputDir+"histsWithFvT_All_v"+s+".dag.*", doRun)

        dag_file = makeDAGFile("histsWithFvT_All_v"+s+".dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)



#
#  Make Plots with FvT
#
if o.plotsWithFvT:
    cmds = []
    logs = []

    weightPostFix = ""

    histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "
    
    yearsToPlot = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToPlot.append("RunII")

    for s in subSamples:

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix

        #histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        #histName4b = "hists_4b_wFVT"+FvTName+".root "
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "

        for y in yearsToPlot:
    
            #
            # MAke Plots
            #
            data3bFile  = getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b         if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName3b               
            data4bFile  = getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b     if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName4b                
            ttbar4bFile = getOutDir()+"/TT"+y+"/"+histName4bTT
            ttbar3bFile = getOutDir()+"/TT"+y+"/"+histName3b
            
            #cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
            #cmd += " --d4 "+outputDir+"/data"+y+"_v"+s+"/"+histName4b
            #cmd += " --d3 "+outputDirComb+"/data"+y+"/"+histName3b
            #cmd += " --t4 "+outputDir+"/TT"+y+"/"+histName4b
            #cmd += " --t3 "+outputDir+"/TT"+y+"/"+histName3b
            #cmd += " --t4_s "+outputDir+"/TTToSemiLeptonic"+y+"_v"+s+"/"+histName4b
            #cmd += " --t4_h "+outputDir+"/TTToHadronic"+y+"_v"+s+"/"+histName4b
            #cmd += " --t4_d "+outputDir+"/TTTo2L2Nu"+y+"_v"+s+"/"+histName4b
            #cmd += " --t3_s "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b
            #cmd += " --t3_h "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b
            #cmd += " --t3_d "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b
            #cmd += " --name "+outputDir+"/CutFlow_wFvT_"+y+"_v"+s
            #cmd += " --makePDF -r"
            #cmds.append(cmd)
            #logs.append(outputDir+"/log_cutFlow_wFVT_"+y+"_v"+s)


            ##cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
            ##cmd += " --d4 "+data4bFile
            ##cmd += " --d3 "+data3bFile
            ##cmd += " --t4 "+ttbar4bFile
            ##cmd += " --t3 "+ttbar3bFile
            ##cmd += " --name "+outputDir+"/CutFlow_wFvT_"+y+FvTName+"_"+tagID
            ##cmd += " --makePDF -r"
            ##cmds.append(cmd)
            ##logs.append(outputDir+"/log_cutFlow_wFVT_"+y+FvTName+"_"+tagID)

    

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+FvTName+"_"+tagID+plotOpts[y]+" -m -j -r --noSignal "
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)
            logs.append(outputDir+"/log_makePlots_wFVT_"+y+FvTName+"_"+tagID)
    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    for s in subSamples:
        FvTName="_"+mixedName+"_v"+s+weightPostFix
        for y in years:
            #cmds.append("mv CutFlow_wFvT_"+y+FvTName+"_"+tagID+".pdf "+outputDir+"/")
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+FvTName+"_"+tagID+".tar plotsWithFvT_"+y+FvTName+"_"+tagID)
            
    babySit(cmds, doRun)    



        

#
#  Make Hists with JCM and FvT weights applied
#
if o.histsNoFvT: 

    weightPostFix = ""

    pico3b = "picoAOD_3b_wJCM_"+tagID+".root"
    picoOut = " -p NONE "
    h10 = " --histogramming 10 --histDetail 7 "    
    outDir = " -o "+getOutDir()+" "


    for s in subSamples:

        dag_config = []
        condor_jobs = []

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix
    
        histName3b = "hists_3b_wJCM_"+JCMName+"_wNoFVT_"+tagID+".root "
        histName4b = "hists_4b_noFVT_"+tagID+".root "


        histOut3b = " --histFile "+histName3b
        histOut4b = " --histFile "+histName4b

        for y in years:
    
            #
            # 3b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+tagID+"_3b_wFvT.txt "

            cmd = runCMD + inputFile + outDir + picoOut  + yearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " --FvTName "+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsNoFvT_3b_"))
    

            # 3b TTbar not needed... Run it anyway for cut flow
            for tt in ttbarSamples:
                inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_3b_wFvT.txt "
                cmd = runCMD + inputFile + outDir + picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " --FvTName "+FvTName
                condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsNoFvT_3b_"))


            #
            # 4b
            #
            # Can reuse the hists when running w/FvT
    
        dag_config.append(condor_jobs)        

    
        #
        #  Hadd TTbar
        #
        condor_jobs = []
        for y in years:
            cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName3b+" "
            for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_"+tagID+"_3b_wFvT/"+histName3b+" "

            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_v"+s, outputDir=outputDir, filePrefix="histsNoFvT_3b_"))


        dag_config.append(condor_jobs)


        #
        # Subtract QCD 
        #
        condor_jobs = []
        for y in years:
            mkdir(outputDir+"/QCD"+y, doRun)

            cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
            cmd += " -d "+getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b
            cmd += " --tt "+getOutDir()+"/TT"+y+"/"+histName3b
            cmd += " -q "+getOutDir()+"/QCD"+y+"/"+histName3b
            condor_jobs.append(makeCondorFile(cmd, getOutDir(), "QCD"+y, outputDir=outputDir, filePrefix="histsNoFvT_QCD_v"+s+"_") )

        dag_config.append(condor_jobs)

        condor_jobs = []        

        #
        #   Hadd years
        #
        if "2016" in years and "2017" in years and "2018" in years:
    
            mkdir(outputDir+"/dataRunII", doRun)
            mkdir(outputDir+"/TTRunII",   doRun)

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName3b+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsNoFvT_3b_"))            

            cmd = "hadd -f "+getOutDir()+"/TTRunII/"  +histName3b+" "
            for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName3b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_v"+s, outputDir=outputDir, filePrefix="histsNoFvT_3b_"))            

            cmd = "hadd -f "+getOutDir()+"/QCDRunII/"  +histName3b+" "
            for y in years: cmd += getOutDir()+"/QCD"+y+"/"+histName3b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "QCDRunII_v"+s, outputDir=outputDir, filePrefix="histsNoFvT_3b_"))            

        dag_config.append(condor_jobs)


        execute("rm "+outputDir+"histsNoFvT_All_v"+s+".dag", doRun)
        execute("rm "+outputDir+"histsNoFvT_All_v"+s+".dag.*", doRun)

        dag_file = makeDAGFile("histsNoFvT_All_v"+s+".dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)



#
#  Make Plots with No FvT
#
if o.plotsNoFvT:
    cmds = []
    logs = []

    weightPostFix = ""
    #weightPostFix = "_comb"

    yearsToPlot = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToPlot.append("RunII")

    for s in subSamples:

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix

        histName3b = "hists_3b_wJCM_"+JCMName+"_wNoFVT_"+tagID+".root "
        #histName4b = "hists_4b_noFVT_"+tagID+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
        histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "

        for y in yearsToPlot:
    
            #
            # MAke Plots
            #
            qcdFile     = getOutDir()+"/QCD"+y+"/"+histName3b
            data3bFile  = getOutDir()+"/data"+y+"_"+tagID+"/"+histName3b    if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName3b               
            data4bFile  = getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName4b                
            ttbar4bFile = getOutDir()+"/TT"+y+"/"+histName4bTT
            ttbar3bFile = getOutDir()+"/TT"+y+"/"+histName3b
            


            #cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
            #cmd += " --d4 "+data4bFile
            #cmd += " --d3 "+data3bFile
            #cmd += " --t4 "+ttbar4bFile
            #cmd += " --t3 "+ttbar3bFile
            #cmd += " --name "+outputDir+"/CutFlow_noFvT_"+y+FvTName+"_"+tagID
            #cmd += " --makePDF "
            #cmds.append(cmd)
            #logs.append(outputDir+"/log_cutFlow_noFVT_"+y+FvTName+"_"+tagID)

    

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+y+FvTName+"_"+tagID+plotOpts[y]+" -m -j  --noSignal "
            cmd += " --qcd "+qcdFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)
            logs.append(outputDir+"/log_makePlots_noFVT_"+y+FvTName+"_"+tagID)
    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    for s in subSamples:
        FvTName="_"+mixedName+"_v"+s+weightPostFix
        for y in years:
            #cmds.append("mv CutFlow_noFvT_"+y+FvTName+"_"+tagID+".pdf "+outputDir+"/")
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+y+FvTName+"_"+tagID+".tar plotsNoFvT_"+y+FvTName+"_"+tagID)
            
    babySit(cmds, doRun)    



        
if o.makeInputsForCombine:

    import ROOT

    def getHistForCombine(in_File,tag,proc,outName,region):
        hist = in_File.Get("passMDRs/"+tag+"/mainView/"+region+"/SvB_ps_"+proc).Clone()
        hist.SetName(outName)
        return hist


    def makeInputsForRegion(region, noFvT=False):

        if noFvT:
            outFile = ROOT.TFile(outputDir+"/hists_closure_"+mixedName+"_"+tagID+"_"+region+"_noFvT.root","RECREATE")
        else:
            outFile = ROOT.TFile(outputDir+"/hists_closure_"+mixedName+"_"+tagID+"_"+region+".root","RECREATE")
    
        procs = ["zz","zh"]
        
        for s in subSamples: 
            
            weightPostFix = ""
            
            #
            #  "+tagID+" with combined JCM 
            #
            #weightPostFix = "_comb"
            #tagName = "_"+tagID
            JCMName=mixedName+"_v"+s+weightPostFix
            FvTName="_"+mixedName+"_v"+s+weightPostFix
            
            histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
            histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
            histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "
            
            #if noFvT:
            #    histName3b = "hists_3b_wJCM_"+JCMName+"_wNoFVT_"+tagID+".root "
            #    histName4b = "hists_4b_noFVT_"+tagID+".root "

    
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
    
                multiJet_Files .append(ROOT.TFile.Open(getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b))
                data_obs_Files .append(ROOT.TFile.Open(getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b))
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


                #multiJet_File.Close()
                #data_obs_File.Close()
                #ttbar_File   .Close()

    makeInputsForRegion("SR")
    makeInputsForRegion("SRNoHH")
    makeInputsForRegion("CR")
    makeInputsForRegion("SB")

    makeInputsForRegion("SR",noFvT=True)
    makeInputsForRegion("SRNoHH",noFvT=True)
    makeInputsForRegion("CR",noFvT=True)
    makeInputsForRegion("SB",noFvT=True)



#
#  Make Hists with JCM and FvT weights applied
#
if o.moveFinalPicoAODsToEOS: 

    def copy(fileName, subDir, outFileName):
        cmd  = "xrdcp  "+fileName+" root://cmseos.fnal.gov//store/user/johnda/closureTest/results/3bAnd4b_"+tagID+"/"+subDir+"/"+outFileName
    
        if doRun:
            os.system(cmd)
        else:
            print cmd


    for s in subSamples:
        for y in years:
            for sample in ["data","TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"]:

                subDir = sample+y+"_"+tagID+"_v"+s
                
                #
                # 4b
                #
                pico4b = "picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".root"
                copy(outputDir+"/"+subDir+"/"+pico4b, subDir,pico4b)



#
#  Make Hists with JCM and FvT weights applied
#
if o.cleanFinalPicoAODsToEOS: 

    def rm(fileName):
        cmd  = "rm  "+fileName
    
        if doRun: os.system(cmd)
        else:     print cmd



    for s in subSamples:
        for y in years:
            for sample in ["data","TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"]:

                subDir = sample+y+"_"+tagID+"_v"+s
                
                #
                # 4b
                #
                pico4b = "picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".root"
                rm(outputDir+"/"+subDir+"/"+pico4b)





if o.haddSubSamples: 

    dag_config = []
    condor_jobs = []

    histNameComb3bwFvT  = "hists_3b_wJCM_wFVT_vAll_"+mixedName+"_"+tagID+".root "
    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+".root "
    histNameComb3bnoFvT = "hists_3b_wJCM_wNoFVT_"+mixedName+"_"+tagID+".root "

    cmdData3bwFvT = "hadd -f "+getOutDir()+"/dataRunII/"+histNameComb3bwFvT+" "
    cmdData4b     = "hadd -f "+getOutDir()+"/dataRunII/"+histNameComb4b+" "
    cmdQCD3b      = "hadd -f "+getOutDir()+"/QCDRunII/"+histNameComb3bnoFvT+" "



    for s in subSamples:
        weightPostFix = ""
        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix

        histName3bwFvT = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        cmdData3bwFvT += getOutDir()+"/dataRunII/"+histName3bwFvT+" "

        histName4b     = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
        cmdData4b += getOutDir()+"/dataRunII/"+histName4b+" "


        histName3bnoFvT = "hists_3b_wJCM_"+JCMName+"_wNoFVT_"+tagID+".root "
        cmdQCD3b += getOutDir()+"/QCDRunII/"+histName3bnoFvT+" "


    condor_jobs.append(makeCondorFile(cmdData3bwFvT, "None", "data3bwFvT", outputDir=outputDir, filePrefix="haddSubSample_"))            
    condor_jobs.append(makeCondorFile(cmdData4b, "None", "data4b", outputDir=outputDir, filePrefix="haddSubSample_"))            
    condor_jobs.append(makeCondorFile(cmdQCD3b, "None", "QCD", outputDir=outputDir, filePrefix="haddSubSample_"))            

    
    dag_config.append(condor_jobs)
    execute("rm "+outputDir+"haddSubSample_All.dag", doRun)
    execute("rm "+outputDir+"haddSubSample_All.dag.*", doRun)

    dag_file = makeDAGFile("haddSubSample_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)

if o.haddSubSamplesRW: 

    dag_config = []
    condor_jobs = []


    FvTName="weight_rw_iter10"
    histName3bComb = "hists_3b_wJCM_vAll_"+mixedName+"_"+tagID+"_"+FvTName+".root "

    cmdQCD = "hadd -f "+getOutDir()+"/QCDRunII/"+histName3bComb+" "


    for s in subSamples:
        weightPostFix = ""
        JCMName=mixedName+"_v"+s+weightPostFix

        histName3b = "hists_3b_wJCM_"+JCMName+"_"+tagID+"_"+FvTName+".root "

        cmdQCD += getOutDir()+"/QCDRunII/"+histName3b+" "


    condor_jobs.append(makeCondorFile(cmdQCD, "None", "qcdRW", outputDir=outputDir, filePrefix="haddSubSampleRW_"))            
    
    dag_config.append(condor_jobs)
    execute("rm "+outputDir+"haddSubSampleRW_All.dag", doRun)
    execute("rm "+outputDir+"haddSubSampleRW_All.dag.*", doRun)

    dag_file = makeDAGFile("haddSubSampleRW_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.scaleCombSubSamples: 

    dag_config = []
    condor_jobs = []

    histNameComb3bwFvT  = "hists_3b_wJCM_wFVT_vAll_"+mixedName+"_"+tagID+".root "
    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+".root "
    histNameComb3bnoFvT = "hists_3b_wJCM_wNoFVT_"+mixedName+"_"+tagID+".root "

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "

    cmdData3bwFvT = cmdScale + " -i "+getOutDir()+"/dataRunII/"+histNameComb3bwFvT+" "
    condor_jobs.append(makeCondorFile(cmdData3bwFvT, getOutDir(), "dataRunII", outputDir=outputDir, filePrefix="scaleCombSubSamples_3b_"))            

    cmdData4b     = cmdScale + " -i "+getOutDir()+"/dataRunII/"+histNameComb4b+" "
    condor_jobs.append(makeCondorFile(cmdData4b, getOutDir(), "dataRunII", outputDir=outputDir, filePrefix="scaleCombSubSamples_4b_"))            

    cmdQCD3b      = cmdScale + " -i "+getOutDir()+"/QCDRunII/"+histNameComb3bnoFvT+" "
    condor_jobs.append(makeCondorFile(cmdQCD3b, getOutDir(), "QCDRunII", outputDir=outputDir, filePrefix="scaleCombSubSamples_"))            

    dag_config.append(condor_jobs)
    execute("rm "+outputDir+"scaleCombSubSamples_All.dag", doRun)
    execute("rm "+outputDir+"scaleCombSubSamples_All.dag.*", doRun)

    dag_file = makeDAGFile("scaleCombSubSamples_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



if o.scaleCombSubSamplesRW: 

    dag_config = []
    condor_jobs = []

    FvTName="weight_rw_iter10"
    histName3bComb = "hists_3b_wJCM_vAll_"+mixedName+"_"+tagID+"_"+FvTName+".root "

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "

    cmdQCD3b      = cmdScale + " -i "+getOutDir()+"/QCDRunII/"+histName3bComb+" "
    condor_jobs.append(makeCondorFile(cmdQCD3b, getOutDir(), "QCDRunII", outputDir=outputDir, filePrefix="scaleCombSubSamplesRW_"))            

    dag_config.append(condor_jobs)
    execute("rm "+outputDir+"scaleCombSubSamplesRW_All.dag", doRun)
    execute("rm "+outputDir+"scaleCombSubSamplesRW_All.dag.*", doRun)

    dag_file = makeDAGFile("scaleCombSubSamplesRW_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#  Make Plots with FvT
#
if o.plotsCombinedSamples:
    cmds = []
    logs = []

    histNameComb3bwFvT  = "hists_3b_wJCM_wFVT_vAll_"+mixedName+"_"+tagID+"_scaled.root "
    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+"_scaled.root "
    histNameComb3bnoFvT = "hists_3b_wJCM_wNoFVT_"+mixedName+"_"+tagID+"_scaled.root "
    histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "

    #
    # Make Plots
    #
    data4bFile  = getOutDir()+"/dataRunII/"+histNameComb4b
    ttbar4bFile = getOutDir()+"/TTRunII/"+histName4bTT
    data3bFile  = getOutDir()+"/dataRunII/"+histNameComb3bwFvT
    qcd3bFile   = getOutDir()+"/QCDRunII/"+histNameComb3bnoFvT

    cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID+plotOpts["RunII"]+" -m -j -r --noSignal "
    cmd += " --data3b "+data3bFile
    cmd += " --data "+data4bFile
    cmd += " --TT "+ttbar4bFile
    cmds.append(cmd)
    logs.append(outputDir+"/log_makePlots_wFVT_RunII")


    cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_RunII_Combined_"+mixedName+"_"+tagID+plotOpts["RunII"]+" -m -j --noSignal "
    cmd += " --qcd "+qcd3bFile
    cmd += " --data "+data4bFile
    cmd += " --TT "+ttbar4bFile
    cmds.append(cmd)
    logs.append(outputDir+"/log_makePlots_noFVT_RunII")

    #
    #  Comparison of each fit vs the average
    #
    for s in subSamples:
        
        weightPostFix = ""
        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix
        histName3bvX = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "

        data3bFilevX  = getOutDir()+"/dataRunII/"+histName3bvX

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_RunII_Combined"+FvTName+"_"+tagID+plotOpts["RunII"]+" -m -j -r --noSignal "
        cmd += " --data3b "+data3bFilevX
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)
        logs.append(outputDir+"/log_makePlots_wFVT_RunII"+FvTName+"_"+tagID)

    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID+".tar plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID)
    cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_RunII_Combined_"+mixedName+"_"+tagID+".tar plotsNoFvT_RunII_Combined_"+mixedName+"_"+tagID)
            
    for s in subSamples:
        FvTName="_"+mixedName+"_v"+s+weightPostFix
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_RunII_Combined"+FvTName+"_"+tagID+".tar plotsWithFvT_RunII_Combined"+FvTName+"_"+tagID)

    babySit(cmds, doRun)    


#
#  Make Plots with FvT
#
if o.plotsCombinedSamplesRW:
    cmds = []
    logs = []

    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+"_scaled.root "

    histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "

    FvTName="weight_rw_iter10"
    histNameCombQCD = "hists_3b_wJCM_vAll_"+mixedName+"_"+tagID+"_"+FvTName+"_scaled.root "


    #
    # Make Plots
    #
    data4bFile  = getOutDir()+"/dataRunII/"+histNameComb4b
    ttbar4bFile = getOutDir()+"/TTRunII/"+histName4bTT
    qcd3bFile   = getOutDir()+"/QCDRunII/"+histNameCombQCD


    cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsRW_RunII_Combined_"+mixedName+"_"+tagID+plotOpts["RunII"]+" -m -j --noSignal "
    cmd += " --qcd "+qcd3bFile
    cmd += " --data "+data4bFile
    cmd += " --TT "+ttbar4bFile
    cmds.append(cmd)
    logs.append(outputDir+"/log_makePlots_RW_RunII")

    #
    #  Comparison of each fit vs the average
    #
    for s in subSamples:
        
        weightPostFix = ""
        JCMName=mixedName+"_v"+s+weightPostFix
        histNameQCDvX = "hists_3b_wJCM_"+JCMName+"_"+tagID+"_"+FvTName+".root "

        qcdFilevX  = getOutDir()+"/QCDRunII/"+histNameQCDvX

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsRW_RunII_Combined_"+JCMName+"_"+tagID+plotOpts["RunII"]+" -m -j  --noSignal "
        cmd += " --qcd "+qcdFilevX
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)
        logs.append(outputDir+"/log_makePlots_RW_RunII"+FvTName+"_"+tagID)

    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsRW_RunII_Combined_"+mixedName+"_"+tagID+".tar plotsRW_RunII_Combined_"+mixedName+"_"+tagID)
            
    for s in subSamples:
        JCMName=mixedName+"_v"+s+weightPostFix
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsRW_RunII_Combined"+JCMName+"_"+tagID+".tar plotsRW_RunII_Combined_"+JCMName+"_"+tagID)

    babySit(cmds, doRun)    



#
#  For Reweighting
# 
if o.makeInput3bWeightFiles:

    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd

    mkdir(outputDir+"/fileLists", doExecute=doRun)

    eosDirMixed = "root://cmseos.fnal.gov//store/user/jda102/condor/OT/"
    rwName = "weight_rw_iter10"
        
    for y in years:

        #
        #  Mixed Samples
        #
        for s in subSamples:
            #fileList = outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+".txt"    
            #run("rm "+fileList)
            #run("echo "+eosDirMixed+"/data"+y+"_"+tagID+"_v"+s+"/picoAOD_"+mixedName+"_"+tagID+"_v"+s+".root >> "+fileList)

            for tt in ttbarSamples + ["data"]:
                fileList = outputDir+"/fileLists/"+tt+y+"_"+mixedName+"_"+tagID+"_"+rwName+"_v"+s+".txt"    
                run("rm "+fileList)

                run("echo "+eosDirMixed+"/rw_v"+s+"/"+tt+y+"_"+tagID+"_picoAOD_3b_wJCM_"+tagID+"_"+rwName+".root >> "+fileList)



#
#  Make Hists with JCM and FvT weights applied
#
if o.histsRW: 

    weightPostFix = ""

    pico3b = "picoAOD_3b_wJCM_"+tagID+".root"
    picoOut = " -p NONE "
    h10 = " --histogramming 10 --histDetail 7 "    
    outDir = " -o "+getOutDir()+" "


    for s in subSamples:

        dag_config = []
        condor_jobs = []

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="weight_rw_iter10"
        rwName = "weight_rw_iter10_v"+s
    
        histName3b = "hists_3b_wJCM_"+JCMName+"_"+tagID+"_"+FvTName+".root "

        histOut3b = " --histFile "+histName3b

    
        for y in years:
    
            #
            # 3b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+tagID+"_3b_wFvT.txt "

            input3bWeightFile = " --inputWeight "+outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_"+rwName+".txt "

            cmd = runCMD + inputFile + input3bWeightFile + outDir + picoOut  + yearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsRW_3b_"))
    

            # 3b TTbar
            for tt in ttbarSamples:
                inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_3b_wFvT.txt "
                input3bWeightFile = " --inputWeight "+outputDir+"/fileLists/"+tt+y+"_"+mixedName+"_"+tagID+"_"+rwName+".txt "

                cmd = runCMD + inputFile + input3bWeightFile + outDir + picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName
                condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsRW_3b_"))


            #
            # 4b
            #
            # Can reuse the hists when running w/FvT
    
        dag_config.append(condor_jobs)        

    
        #
        #  Hadd TTbar
        #
        condor_jobs = []
        for y in years:
            cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName3b+" "
            for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_"+tagID+"_3b_wFvT/"+histName3b+" "

            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_v"+s, outputDir=outputDir, filePrefix="histsRW_3b_"))


        dag_config.append(condor_jobs)


        #
        # Subtract QCD 
        #
        condor_jobs = []
        for y in years:
            mkdir(outputDir+"/QCD"+y, doRun)

            cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
            cmd += " -d "+getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b
            cmd += " --tt "+getOutDir()+"/TT"+y+"/"+histName3b
            cmd += " -q "+getOutDir()+"/QCD"+y+"/"+histName3b
            condor_jobs.append(makeCondorFile(cmd, getOutDir(), "QCD"+y, outputDir=outputDir, filePrefix="histsRW_QCD_v"+s+"_") )

        dag_config.append(condor_jobs)

        condor_jobs = []        

        #
        #   Hadd years
        #
        if "2016" in years and "2017" in years and "2018" in years:
    
            mkdir(outputDir+"/dataRunII", doRun)
            mkdir(outputDir+"/TTRunII",   doRun)

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName3b+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsRW_3b_"))            

            cmd = "hadd -f "+getOutDir()+"/TTRunII/"  +histName3b+" "
            for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName3b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_v"+s, outputDir=outputDir, filePrefix="histsRW_3b_"))            

            cmd = "hadd -f "+getOutDir()+"/QCDRunII/"  +histName3b+" "
            for y in years: cmd += getOutDir()+"/QCD"+y+"/"+histName3b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "QCDRunII_v"+s, outputDir=outputDir, filePrefix="histsRW_3b_"))            

        dag_config.append(condor_jobs)


        execute("rm "+outputDir+"histsRW_All_v"+s+".dag", doRun)
        execute("rm "+outputDir+"histsRW_All_v"+s+".dag.*", doRun)

        dag_file = makeDAGFile("histsRW_All_v"+s+".dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)



#
#  Make Plots with RW
#
if o.plotsRW:
    cmds = []
    logs = []

    weightPostFix = ""

    yearsToPlot = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToPlot.append("RunII")

    for s in subSamples:

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix
        rwName = "weight_rw_iter10"

        histName3b = "hists_3b_wJCM_"+JCMName+"_"+tagID+"_"+rwName+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
        histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "

        for y in yearsToPlot:
    
            #
            # MAke Plots
            #
            qcdFile     = getOutDir()+"/QCD"+y+"/"+histName3b
            data3bFile  = getOutDir()+"/data"+y+"_"+tagID+"/"+histName3b    if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName3b               
            data4bFile  = getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName4b                
            ttbar4bFile = getOutDir()+"/TT"+y+"/"+histName4bTT
            ttbar3bFile = getOutDir()+"/TT"+y+"/"+histName3b
            

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsRW_"+y+FvTName+"_"+tagID+plotOpts[y]+" -m -j  --noSignal "
            cmd += " --qcd "+qcdFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)
            logs.append(outputDir+"/log_makePlots_RW_"+y+FvTName+"_"+tagID)
    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    for s in subSamples:
        FvTName="_"+mixedName+"_v"+s+weightPostFix
        for y in yearsToPlot:
            #cmds.append("mv CutFlow_noFvT_"+y+FvTName+"_"+tagID+".pdf "+outputDir+"/")
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsRW_"+y+FvTName+"_"+tagID+".tar plotsRW_"+y+FvTName+"_"+tagID)
            
    babySit(cmds, doRun)    


if o.writeOutMixedToUnmixedWeights:

    #
    #   4b Nominal
    #
    dag_config = []
    condor_jobs = []


    #
    #   3bMix4b
    #
    for y in years:

        for s in subSamples:
            #picoIn="picoAOD_"+mixedName+"_4b_v"+s+".h5"
            picoIn="picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".h5"

            #fvtList = "_"+mixedName+"toUnmixed_v"+s
            fvtList = "_"+mixedName+"toUnmixed"

            cmd = convertToROOTJOB+" --i "+getOutDirComb()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/"+picoIn +" --outName MixedToUnmixed  --fvtNameList "+fvtList 
            condor_jobs.append(makeCondorFile(cmd, EOSOUTDIRCOMB, "data"+y+"_"+mixedName+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="writeOutMixedToUnmixedWeights_"))


    dag_config.append(condor_jobs)


    execute("rm "+outputDir+"writeOutMixedToUnmixedWeights_All.dag", doRun)
    execute("rm "+outputDir+"writeOutMixedToUnmixedWeights_All.dag.*", doRun)


    dag_file = makeDAGFile("writeOutMixedToUnmixedWeights_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


if o.makeMixedToUnMixedInputFiles:
    
    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    
    for y in years:

        #
        #  Mixed Samples
        #
        for s in subSamples:

            fileList = outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_weights_MixedToUnmixed.txt"    
            run("rm "+fileList)
            run("echo "+EOSOUTDIRCOMB+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+"_weights_MixedToUnmixed.root >> "+fileList)




#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithMixedUnMixedWeights: 


    picoOut = " -p NONE "
    h10 = " --histogramming 10 --histDetail 7 "    
    outDir = " -o "+getOutDir()+" "


    for s in subSamples:
        dag_config = []
        condor_jobs = []

        histName = "hists_"+mixedName+"toUnmixed_"+tagID+"_v"+s+".root "
    
        histOut = " --histFile "+histName

        pico4b = "picoAOD_"+mixedName+"_4b_"+tagID+"_v"+s+".root "

        for y in years:

            #
            # 4b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT.txt"
            inputWeight4b = " --inputWeightFiles4b "+outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_weights_MixedToUnmixed.txt "
            reweight4bName = "--reweight4bName weight_FvT_"+mixedName+"toUnmixed "

            cmd = runCMD + inputFile + inputWeight4b + outDir +  picoOut  +   yearOpts[y]+ h10 + histOut + reweight4bName + " --is3bMixed  "
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+mixedName+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsWithMixedUnMixedWeights_"))


        dag_config.append(condor_jobs)


        condor_jobs = []        

        #
        #   Hadd years
        #
        if "2016" in years and "2017" in years and "2018" in years:
    
            mkdir(outputDir+"/dataRunII", doRun)
        
            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsWithMixedUnMixedWeights_"))            


        dag_config.append(condor_jobs)

        execute("rm "+outputDir+"histsWithMixedUnMixedWeights_All_v"+s+".dag", doRun)
        execute("rm "+outputDir+"histsWithMixedUnMixedWeights_All_v"+s+".dag.*", doRun)

        dag_file = makeDAGFile("histsWithMixedUnMixedWeights_All_v"+s+".dag",dag_config, outputDir=outputDir)
        cmd = "condor_submit_dag "+dag_file
        execute(cmd, o.execute)




#
#  Make Hists of mixed Datasets
#
if o.histsForJCMOneFvTFit: 


    dag_config = []
    condor_jobs = []

    histName = "hists_"+mixedName+"_"+tagID+"_vOneFit.root "

    for y in years:

        picoOut    = " -p NONE "
        h10        = " --histogramming 10 "
        inFileList = outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_vOneFit.txt"
        histOut    = " --histFile "+histName
        
        # The --is3bMixed here just turns off blinding of the data
        cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --is3bMixed --isDataMCMix "
        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_vOneFit", outputDir=outputDir, filePrefix="histsForJCM_vOneFit"))
                                   

    dag_config.append(condor_jobs)


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII",   doRun)
        condor_jobs = []        

        cmd = "hadd -f " + getOutDir()+"/dataRunII_"+mixedName+"_"+tagID+"_vOneFit/"+ histName+" "
        for y in years:
            cmd += getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_vOneFit/"  +histName+" "

        condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_vOneFit", outputDir=outputDir, filePrefix="histsForJCM_vOneFit"))            

    dag_config.append(condor_jobs)            


    execute("rm "+outputDir+"histsForJCM_vOneFit_All.dag", doRun)
    execute("rm "+outputDir+"histsForJCM_vOneFit_All.dag.*", doRun)

    dag_file = makeDAGFile("histsForJCM_vOneFit_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


if o.scaleCombSubSamplesOneFvTFit: 

    dag_config = []
    condor_jobs = []

    histName = "hists_"+mixedName+"_"+tagID+"_vOneFit.root "

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "

    cmdMixedData = cmdScale + " -i "+getOutDir()+"/dataRunII_"+mixedName+"_"+tagID+"_vOneFit/"+histName+" "
    condor_jobs.append(makeCondorFile(cmdMixedData, getOutDir(), "dataRunII_"+mixedName+"_"+tagID+"_vOneFit", outputDir=outputDir, filePrefix="scaleCombSubSamplesOneFvTFit_"))            

    dag_config.append(condor_jobs)
    execute("rm "+outputDir+"scaleCombSubSamplesOneFvTFit_All.dag", doRun)
    execute("rm "+outputDir+"scaleCombSubSamplesOneFvTFit_All.dag.*", doRun)

    dag_file = makeDAGFile("scaleCombSubSamplesOneFvTFit_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#  Fit JCM
#
if o.doWeightsOneFvTFit:
    
    cmds = []
    logs = []

    mkdir(outputDir+"/weights", doRun)

    for y in ["RunII"]:


        histName4b = "hists_"+mixedName+"_"+tagID+"_vOneFit_scaled.root "             
        histName4bTT = "hists_"+mixedName+"_"+tagID+"_vAll.root "             
        histName3b = "hists_"+tagID+".root "

        data3bFile  = getOutDirNom()+"/data"+y+"_"+tagID+"/"+histName3b     if not y == "RunII" else getOutDirNom()+"/data"+y+"/"+histName3b               
        data4bFile  = getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_vOneFit"+"/"+histName4b #if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName4b                
        ttbar4bFile = getOutDir()+"/TT"+y+"/"+histName4bTT
        ttbar3bFile = getOutDirNom()+"/TT"+y+"/"+histName3b
        
        cmd = weightCMD
        cmd += " -d "+data3bFile
        cmd += " --data4b "+data4bFile
        cmd += " --tt "+ttbar3bFile
        cmd += " --tt4b "+ttbar4bFile
        cmd += " -c passMDRs   -o "+outputDir+"/weights/data"+y+"_"+mixedName+"_"+tagID+"_vOneFit"+"/  -r SB -w 02-03-00 "+plotOpts[y]
        
        cmds.append(cmd)
        logs.append(outputDir+"/log_makeWeights_"+y+"_"+tagID+"_vOneFit")

    babySit(cmds, doRun, logFiles=logs)
