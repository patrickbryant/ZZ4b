
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
parser.add_option('--makeInputFileListsTTPseudoData',  action="store_true",      help="make Input file lists")
parser.add_option('--plotUniqueHemis',    action="store_true",      help="Do Some Mixed event analysis")
parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--histsNoFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsNoFvT', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--cutFlowBeforeJCM', action="store_true",      help="Make 4b cut flow before JCM")
parser.add_option('--makeInputsForCombine', action="store_true",      help="Make inputs for the combined tool")
parser.add_option('--moveFinalPicoAODsToEOS', action="store_true",      help="Move Final AODs to EOS")
parser.add_option('--cleanFinalPicoAODsToEOS', action="store_true",      help="Move Final AODs to EOS")
parser.add_option('--haddSubSamples', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--haddSubSamplesForOneFvT', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--scaleCombSubSamples', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--scaleCombSubSamplesOneFvTFit', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--plotsCombinedSamples', action="store_true",      help="Make pdfs with FvT")
#parser.add_option('--makeInput3bWeightFiles',  action="store_true",      help="make Input file lists")
#parser.add_option('--writeOutMixedToUnmixedWeights',  action="store_true",      help="")
#parser.add_option('--writeOutMixedToUnmixedWeightsMu10Signal',  action="store_true",      help="")
parser.add_option('--makeMixedToUnMixedInputFiles',  action="store_true",      help="")
parser.add_option('--histsWithMixedUnMixedWeights',  action="store_true",      help="")
parser.add_option('--histDetailLevel',  default="allEvents.passMDRs.threeTag.fourTag",      help="")
parser.add_option('--rMin',  default=0.9,      help="")
parser.add_option('--rMax',  default=1.1,      help="")
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
#  Make Input file Lists
# 
if o.makeInputFileListsTTPseudoData:
    
    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    mkdir(outputDir+"/fileLists", doExecute=doRun)

    eosDirTTbar = "root://cmseos.fnal.gov//store/user/johnda/condor/mixed/"    

    for y in years:

        for tt in ttbarSamples:
            PSFileList = outputDir+"/fileLists/"+tt+y+"_PSData_"+tagID+".txt"    
            run("rm "+PSFileList)
            
            run("echo "+eosDirTTbar+"/"+tt+y+"_noMjj_"+tagID+"/picoAOD_4bPseudoData_"+tagID+".root >> "+PSFileList)




#
#  Make Hists with JCM and FvT weights applied
#
if o.histsForJCM: 


    picoOut = " -p NONE "
    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
    outDir = " -o "+getOutDir()+" "

    dag_config = []
    condor_jobs = []


    histNameTT = "hists_PSData_"+tagID+".root "

    # 
    #  Hists of the Mixed data
    # 
    for s in subSamples:

        histName = "hists_"+mixedName+"toUnmixed_"+tagID+"_v"+s+".root "
        histOut = " --histFile "+histName

        for y in years:

            #
            # 4b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT.txt"

            inputWeight4b = " --inputWeightFiles4b "+outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_weights_MixedToUnmixed.txt "
            reweight4bName = "--reweight4bName weight_FvT_"+mixedName+"_MixedtoUnmixed"

            cmd = runCMD + inputFile + inputWeight4b + outDir +  picoOut  +   yearOpts[y]+ h10 + histOut + reweight4bName +" --unBlind  "
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+mixedName+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsForJCM_"))



    #
    #  Hists of the TTbar PS data
    #
    for y in years:

        for tt in ttbarSamples:

            inFileList = outputDir+"/fileLists/"+tt+y+"_PSData_"+tagID+".txt"
            histOut    = " --histFile "+histNameTT

            cmd = runCMD+" -i "+inFileList+" -o "+getOutDir() + picoOut + yearOpts[y] + h10 + histOut+" --unBlind --isDataMCMix "
            condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_PSData_"+tagID, outputDir=outputDir, filePrefix="histsForJCM_"))


    dag_config.append(condor_jobs)

    # 
    # Hadd TTbar
    #
    condor_jobs = []

    for y in years:
        cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histNameTT+" "
        for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_PSData_"+tagID+"/"+histNameTT+" "
        
        condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_PSData_"+tagID, outputDir=outputDir, filePrefix="histsForJCM_"))


    dag_config.append(condor_jobs)


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
        condor_jobs = []        

        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/TTRunII",   doRun)
        
        for s in subSamples:
            histName = "hists_"+mixedName+"toUnmixed_"+tagID+"_v"+s+".root "
            
            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsForJCM_"))            

        cmd = "hadd -f "+getOutDir()+"/TTRunII/"  +histNameTT+" "
        for y in years: cmd += getOutDir()+"/TT"+y+"/"+histNameTT+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII", outputDir=outputDir, filePrefix="histsForJCM_"))            

        dag_config.append(condor_jobs)


    #
    #  Hadd for "Combined Mixed dataset"
    # 
    condor_jobs = []        

    for s in subSamples:
        histName = "hists_"+mixedName+"toUnmixed_"+tagID+"_v"+s+".root "
            
        cmd = "hadd -f "+getOutDir()+"/mixedRunII/"+histName+" "
        cmd += getOutDir()+"/dataRunII/"+histName+" "
        cmd += getOutDir()+"/TTRunII/"+histNameTT+" "

        condor_jobs.append(makeCondorFile(cmd, "None", "mixedRunII_v"+s, outputDir=outputDir, filePrefix="histsForJCM_"))            



    dag_config.append(condor_jobs)

    execute("rm "+outputDir+"histsForJCM_All.dag", doRun)
    execute("rm "+outputDir+"histsForJCM_All.dag.*", doRun)

    dag_file = makeDAGFile("histsForJCM_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


if o.haddSubSamplesForOneFvT: 

    dag_config = []
    condor_jobs = []

    histNameComb = "hists_"+mixedName+"toUnmixed_"+tagID+"_vOneFvT.root "

    cmd  = "hadd -f "+getOutDir()+"/mixedRunII/"+histNameComb+" "

    for s in subSamples:

        histName = "hists_"+mixedName+"toUnmixed_"+tagID+"_v"+s+".root "

        cmd += getOutDir()+"/mixedRunII/"+histName+" "

    condor_jobs.append(makeCondorFile(cmd, "None", "mixedRunII", outputDir=outputDir, filePrefix="haddSubSampleOneFvT_"))            
    
    dag_config.append(condor_jobs)

    #
    #  Scale Hadded sample
    #
    condor_jobs = []
    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "
    cmd = cmdScale + " -i "+getOutDir()+"/mixedRunII/"+histNameComb+" "
    condor_jobs.append(makeCondorFile(cmd, getOutDir(), "mixedRunII", outputDir=outputDir, filePrefix="haddSubSampleOneFvT_scale_"))            
    dag_config.append(condor_jobs)

    execute("rm "+outputDir+"haddSubSampleOneFvT_All.dag", doRun)
    execute("rm "+outputDir+"haddSubSampleOneFvT_All.dag.*", doRun)

    dag_file = makeDAGFile("haddSubSampleOneFvT_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)


#
#  Fit JCM
#
if o.doWeights:
    
    cmds = []
    logs = []

    mkdir(outputDir+"/weights", doRun)

    histName3b = "hists_"+tagID+".root "
    histNameTT4b   = "hists_4b_noPSData_"+tagID+".root"        

    yearsToFit = ["RunII"]

    subSamplesAndOneFvT = subSamples + ["OneFvT"]

    for s in subSamplesAndOneFvT:

        histName4b = "hists_"+mixedName+"toUnmixed_"+tagID+"_v"+s+".root "
        if s in ["OneFvT"]: histName4b = "hists_"+mixedName+"toUnmixed_"+tagID+"_v"+s+"_scaled.root "
        

        for y in yearsToFit:

            data3bFile  = getOutDirNom()+"/data"+y+"/"+histName3b     
            ttbar3bFile = getOutDirNom()+"/TT"+y+"/"+histName3b

            data4bFile  = getOutDir()+"/mixed"+y+"/"+histName4b
            ttbar4bFile = getOutDirMixed()+"/TT"+y+"/"+histNameTT4b
            
            cmd = weightCMD
            cmd += " -d "+data3bFile
            cmd += " --data4b "+data4bFile
            cmd += " --tt "+ttbar3bFile
            cmd += " --tt4b "+ttbar4bFile
            cmd += " -c passMDRs   -o "+outputDir+"/weights/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"/  -r SB -w 03-01-00 "+plotOpts[y]
            
            cmds.append(cmd)
            logs.append(outputDir+"/log_makeWeights_"+y+"_"+tagID+"_v"+s)

    babySit(cmds, doRun, logFiles=logs)


#
#  Adding JCM weights now done in makeClosureCombined
#



# 
#  Tracining done in makeClosureTestCombinedTraining
#



#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithFvT: 

    dag_config = []
    condor_jobs = []

    weightPostFix = ""
    #weightPostFix = "_comb"
    FvToffsetName = "_os012"
    

    picoOut = " -p NONE "
    h10        = " --histDetailLevel "+o.histDetailLevel+" "
    outDir = " -o "+getOutDir()+" "

    histName4bTTPSData = "hists_4b_wFVT_PSData_"+tagID+FvToffsetName+".root "
    histOut4bTTPSData = " --histFile "+histName4bTTPSData

    histName4bTTNoPSData = "hists_4b_wFVT_noPSData_"+tagID+FvToffsetName+".root "
    histOut4bTTNoPSData = " --histFile "+histName4bTTNoPSData


    for s in subSamples:

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName
        
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
    
        histOut3b = " --histFile "+histName3b
        histOut4b = " --histFile "+histName4b

        for y in years:
    
            #
            # 3b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+tagID+"_3b_wFvT.txt "
            inputWeights = " --inputWeightFiles "+outputDirComb+"/fileLists/data"+y+"_"+tagID+"_3b_weights_FvT.txt "

            cmd = runCMD + inputFile + inputWeights + outDir + picoOut  +  yearOpts[y] + h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName weight_FvT"+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))

            ### 3b TTbar not needed... Run it anyway for cut flow
            ##for tt in ttbarSamples:
            ##    inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_3b_wFvT.txt "
            ##    inputWeights = " --inputWeightFiles "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_3b_weights_FvT.txt "
            ##    
            ##    cmd = runCMD + inputFile + inputWeights + outDir + picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName weight_FvT"+FvTName
            ##    condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))

            #
            # 4b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT.txt"
            inputWeights4b = " --inputWeightFiles4b "+outputDir+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_weights_MixedToUnmixed.txt "
            reweight4bName = " --reweight4bName weight_FvT_"+mixedName+"_MixedtoUnmixed "

            inputWeights   = " --inputWeightFiles "+outputDirComb+"/fileLists/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_weights_FvT.txt"

            cmd = runCMD + inputFile + inputWeights + inputWeights4b + reweight4bName + outDir +  picoOut  +   yearOpts[y]+ h10 + histOut4b + "  --FvTName weight_FvT"+FvTName + " --unBlind"
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+mixedName+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_4b_"))
            
            if s in ["0"]:
                for tt in ttbarSamples:
                    #inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll_wFvT.txt"
                    #cmd = runCMD + inputFile + outDir + picoOut  + MCyearOpts[y]+ h10 + histOut4bTT + "  --FvTName "+FvTName + " --usePreCalcBTagSFs"
                    #condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_"+mixedName+"_"+tagID+"_vAll", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))

                    inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_PSData_"+tagID+"_wFvT.txt"
                    inputWeights = " --inputWeightFiles "+outputDirComb+"/fileLists/"+tt+y+"_PSData_"+tagID+"_weights_FvT.txt ss"
                    cmd = runCMD + inputFile + inputWeights + outDir + picoOut  + yearOpts[y]+ h10 + histOut4bTTPSData + "  --FvTName weight_FvT"+FvTName + " --unBlind --isDataMCMix "
                    condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_PSData_"+tagID, outputDir=outputDir, filePrefix="histsWithFvT_4b_"))

                    inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_noPSData_wFvT.txt"
                    inputWeights = " --inputWeightFiles "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_noPSData_weights_FvT.txt "
                    cmd = runCMD + inputFile + inputWeights + outDir + picoOut  + MCyearOpts[y]+ h10 + histOut4bTTNoPSData + "  --FvTName weight_FvT"+FvTName + " " # --usePreCalcBTagSFs ??
                    condor_jobs.append(makeCondorFile(cmd, "None", tt+y+"_noPSData_"+tagID, outputDir=outputDir, filePrefix="histsWithFvT_4b_"))



    dag_config.append(condor_jobs)

    #
    #  Hadd TTbar
    #
    condor_jobs = []

    for s in subSamples:

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName

        #histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "

        for y in years:

            #cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName3b+" "
            #for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_"+tagID+"_3b_wFvT/"+histName3b+" "
            #condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))

            if s in ["0"]:    
                #cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName4bTT+" "
                #for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll_wFvT/"+histName4bTT
                #condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_vAll", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))

                cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName4bTTPSData+" "
                for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_PSData_"+tagID+"_wFvT/"+histName4bTTPSData
                condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_PSData", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))

                cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName4bTTNoPSData+" "
                for tt in ttbarSamples: cmd += getOutDir()+"/"+tt+y+"_"+tagID+"_noPSData_wFvT/"+histName4bTTNoPSData
                condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_NoPSData", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))

    dag_config.append(condor_jobs)


    condor_jobs = []        

    #
    #   Hadd years
    #
    mkdir(outputDir+"/dataRunII", doRun)
    mkdir(outputDir+"/TTRunII",   doRun)

    for s in subSamples:

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName

        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "


        if "2016" in years and "2017" in years and "2018" in years:
    
            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName3b+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))            

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName4b+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_4b_"))            

            if s in ["0"]:                
                #cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName4bTT+" "
                #for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName4bTT+" "
                #condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_vAll", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))            

                cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName4bTTPSData+" "
                for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName4bTTPSData+" "
                condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_PSData", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))            

                cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName4bTTNoPSData+" "
                for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName4bTTNoPSData+" "
                condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_NoPSData", outputDir=outputDir, filePrefix="histsWithFvT_4b_"))            


            #cmd = "hadd -f "+getOutDir()+"/TTRunII/"  +histName3b+" "
            #for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName3b+" "
            #condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_3b_"))            

    dag_config.append(condor_jobs)
    

    #
    #  Hadd for "Combined Mixed dataset"
    # 
    condor_jobs = []        

    for s in subSamples:

        yearsToHadd = years + ["RunII"]
        for y in yearsToHadd:


            JCMName=mixedName+"_v"+s+weightPostFix
            FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName
            
            histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
            histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
            
            cmd = "hadd -f "+getOutDir()+"/mixed"+y+"/"+histName4b+" " 
            cmd += getOutDir()+"/data"+y+"/"+histName4b+" "  if y in ["RunII"] else getOutDir()+"/data"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b
            cmd += getOutDir()+"/TT"+y+"/"+histName4bTTPSData+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+"_v"+s, outputDir=outputDir, filePrefix="histsWithFvT_"))            

    dag_config.append(condor_jobs)

    #
    #  Hadd subsamples
    #
    condor_jobs = []        

    histNameComb3bwFvT  = "hists_3b_wJCM_wFVT_vAll_"+mixedName+"_"+tagID+FvToffsetName+".root "
    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+FvToffsetName+".root "

    cmdData3bwFvT = "hadd -f "+getOutDir()+"/dataRunII/"+histNameComb3bwFvT+" "
    cmdData4b     = "hadd -f "+getOutDir()+"/mixedRunII/"+histNameComb4b+" "

    for s in subSamples:
        weightPostFix = ""
        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName

        histName3bwFvT = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        cmdData3bwFvT += getOutDir()+"/dataRunII/"+histName3bwFvT+" "

        histName4b     = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
        cmdData4b += getOutDir()+"/mixedRunII/"+histName4b+" "

    condor_jobs.append(makeCondorFile(cmdData3bwFvT, "None", "data3bwFvT", outputDir=outputDir, filePrefix="histsWithFvT_comb_"))            
    condor_jobs.append(makeCondorFile(cmdData4b,     "None", "data4b",     outputDir=outputDir, filePrefix="histsWithFvT_comb_"))            

    dag_config.append(condor_jobs)


    #
    #  Scale Subsamples
    #
    condor_jobs = []

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "

    cmdData3bwFvT = cmdScale + " -i "+getOutDir()+"/dataRunII/"+histNameComb3bwFvT+" "
    condor_jobs.append(makeCondorFile(cmdData3bwFvT, getOutDir(), "dataRunII", outputDir=outputDir, filePrefix="histsWithFvT_scale_"))            

    cmdData4b     = cmdScale + " -i "+getOutDir()+"/mixedRunII/"+histNameComb4b+" "
    condor_jobs.append(makeCondorFile(cmdData4b, getOutDir(), "mixedRunII", outputDir=outputDir, filePrefix="histsWithFvT_scale_"))            

    dag_config.append(condor_jobs)

        
    execute("rm "+outputDir+"histsWithFvT_All.dag", doRun)
    execute("rm "+outputDir+"histsWithFvT_All.dag.*", doRun)

    dag_file = makeDAGFile("histsWithFvT_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)

    



#
#  Make Plots with FvT
#
if o.plotsWithFvT:
    cmds = []
    logs = []

    weightPostFix = ""
    FvToffsetName = "_os012"

    histName4bTTNoPSData = "hists_4b_wFVT_noPSData_"+tagID+FvToffsetName+".root "
    
    yearsToPlot = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToPlot.append("RunII")

    yearsToPlot = ["RunII"]

    for s in subSamples:

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName

        #histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        #histName4b = "hists_4b_wFVT"+FvTName+".root "
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "


        for y in yearsToPlot:
    
            #
            # MAke Plots
            #
            data3bFile  = getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b         if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName3b               
            data4bFile  = getOutDir()+"/mixed"+y+"/"+histName4b                
            ttbar4bFile = getOutDir()+"/TT"+y+"/"+histName4bTTNoPSData
            #ttbar3bFile = getOutDir()+"/TT"+y+"/"+histName3b
            
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
            cmd += " --histDetailLevel "+o.histDetailLevel
            cmd += " --rMin "+o.rMin
            cmd += " --rMax "+o.rMax
            cmds.append(cmd)
            logs.append(outputDir+"/log_makePlots_wFVT_"+y+FvTName+"_"+tagID)
    



    #
    #  Combined Plots
    #
    histNameComb3bwFvT  = "hists_3b_wJCM_wFVT_vAll_"+mixedName+"_"+tagID+FvToffsetName+"_scaled.root "
    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+FvToffsetName+"_scaled.root "
    histName4bTTNoPSData = "hists_4b_wFVT_noPSData_"+tagID+FvToffsetName+".root "

    data4bFile  = getOutDir()+"/mixedRunII/"+histNameComb4b
    ttbar4bFile = getOutDir()+"/TTRunII/"+histName4bTTNoPSData
    data3bFile  = getOutDir()+"/dataRunII/"+histNameComb3bwFvT

    cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID+FvToffsetName+plotOpts["RunII"]+" -m -j -r --noSignal "
    cmd += " --data3b "+data3bFile
    cmd += " --data "+data4bFile
    cmd += " --TT "+ttbar4bFile
    cmd += " --histDetailLevel "+o.histDetailLevel
    cmd += " --rMin "+o.rMin
    cmd += " --rMax "+o.rMax
    cmds.append(cmd)
    logs.append(outputDir+"/log_makePlots_wFVT_RunII")


    #
    #  Comparison of each fit vs the average
    #
    for s in subSamples:
        
        weightPostFix = ""
        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName
        histName3bvX = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        data3bFilevX  = getOutDir()+"/dataRunII/"+histName3bvX

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_RunII_Combined"+FvTName+"_"+tagID+plotOpts["RunII"]+" -m -j -r --noSignal "
        cmd += " --data3b "+data3bFilevX
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmd += " --histDetailLevel "+o.histDetailLevel
        cmd += " --rMin "+o.rMin
        cmd += " --rMax "+o.rMax
        cmds.append(cmd)
        logs.append(outputDir+"/log_makePlots_wFVT_RunII"+FvTName+"_"+tagID)

    
    babySit(cmds, doRun, logFiles=logs)    


    #
    #  Tarballs
    #
    cmds = []
    cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID+FvToffsetName+".tar plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID+FvToffsetName)

    for s in subSamples:
        FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_RunII_Combined"+FvTName+"_"+tagID+".tar plotsWithFvT_RunII_Combined"+FvTName+"_"+tagID)

        for y in yearsToPlot:
            #cmds.append("mv CutFlow_wFvT_"+y+FvTName+"_"+tagID+".pdf "+outputDir+"/")
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+FvTName+"_"+tagID+".tar plotsWithFvT_"+y+FvTName+"_"+tagID)

    babySit(cmds, doRun)    




if o.makeInputsForCombine:

    import ROOT

    def getHistForCombine(in_File,tag,proc,outName,region):
        hist = in_File.Get("passMDRs/"+tag+"/mainView/"+region+"/SvB_ps_"+proc).Clone()
        hist.SetName(outName)
        return hist


    def makeInputsForRegion(region, noFvT=False):

        FvToffsetName = "_os012"

        if noFvT:
            outFile = ROOT.TFile(outputDir+"/hists_closure_MixedToUnmixed_"+mixedName+"_"+tagID+"_"+region+"_noFvT.root","RECREATE")
        else:
            outFile = ROOT.TFile(outputDir+"/hists_closure_MixedToUnmixed_"+mixedName+"_"+tagID+"_"+region+FvToffsetName+".root","RECREATE")
    
        procs = ["zz","zh"]
        
        for s in subSamples: 
            
            weightPostFix = ""
            
            #
            #  "+tagID+" with combined JCM 
            #
            #weightPostFix = "_comb"
            #tagName = "_"+tagID
            JCMName=mixedName+"_v"+s+weightPostFix
            FvTName="_"+mixedName+"_v"+s+weightPostFix+FvToffsetName
            
            histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
            histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
            histName4bTT = "hists_4b_wFVT_noPSData_"+tagID+FvToffsetName+".root"
            
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
            bkgTot_Hists    = {}

            for p in procs:
                multiJet_Hists[p] = []
                data_obs_Hists[p] = []
                ttbar_Hists   [p] = []
                bkgTot_Hists  [p] = []

            for y in years:
    
                multiJet_Files .append(ROOT.TFile.Open(getOutDir()+"/data"+y+"_"+tagID+"_3b_wFvT/"+histName3b))
                #data_obs_Files .append(ROOT.TFile.Open(getOutDir()+"/mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b))
                data_obs_Files .append(ROOT.TFile.Open(getOutDir()+"/mixed"+y+"/"+histName4b))
                ttbar_Files    .append(ROOT.TFile.Open(getOutDir()+"/TT"+y+"/"+histName4bTT))
        
                for p in procs:
    
    
                    multiJet_Hists[p].append( getHistForCombine(multiJet_Files[-1],"threeTag",p,"multijet", region) )
                    data_obs_Hists[p].append( getHistForCombine(data_obs_Files[-1],"fourTag",p, "data_obs", region) )
                    ttbar_Hists[p]   .append( getHistForCombine(ttbar_Files[-1],   "fourTag",p, "ttbar",    region) )

                    bkgTot_Hists[p]  .append( multiJet_Hists[p][-1].Clone())
                    bkgTot_Hists[p][-1].SetName("bkgTot")
                    bkgTot_Hists[p][-1].Add(ttbar_Hists[p][-1])
                    
    
                    sampleDir.cd()
                    procDir = sampleDir.mkdir(p+y)
                    procDir.cd()
                    
                    #multiJet_Hist.SetDirectory(procDir)
                    multiJet_Hists[p][-1].Write()
                    data_obs_Hists[p][-1].Write()
                    ttbar_Hists   [p][-1].Write()
                    bkgTot_Hists  [p][-1].Write()
                    


            # Combined Run2
            for p in procs:
                

                multiJet_HistRunII = multiJet_Hists[p][0].Clone()
                data_obs_HistRunII = data_obs_Hists[p][0].Clone()
                ttbar_HistRunII    = ttbar_Hists   [p][0].Clone()

                for i in [1,2]:
                    multiJet_HistRunII.Add(multiJet_Hists[p][i])
                    data_obs_HistRunII.Add(data_obs_Hists[p][i])
                    ttbar_HistRunII   .Add(ttbar_Hists   [p][i])


                bkgTot_HistRunII = multiJet_HistRunII.Clone()
                bkgTot_HistRunII.SetName("bkgTot")
                bkgTot_HistRunII.Add(ttbar_HistRunII)
                
                sampleDir.cd()
                procDir = sampleDir.mkdir(p+"RunII")
                procDir.cd()

                #multiJet_Hist.SetDirectory(procDir)
                multiJet_HistRunII.Write()
                data_obs_HistRunII.Write()
                ttbar_HistRunII   .Write()
                bkgTot_HistRunII  .Write()

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
if o.histsNoFvT: 


    weightPostFix = ""

    picoOut = " -p NONE "
    h10        = " --histDetailLevel allEvents.passMDRs.threeTag.fourTag "
    outDir = " -o "+getOutDir()+" "


    for s in subSamples:

        dag_config = []
        condor_jobs = []


        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix
    
        histName3b = "hists_3b_wJCM_"+JCMName+"_wNoFVT_"+tagID+".root "

        histOut3b = " --histFile "+histName3b

        for y in years:
    
            #
            # 3b
            #
            inputFile = " -i "+outputDirComb+"/fileLists/data"+y+"_"+tagID+"_3b_wFvT.txt "
            inputWeights = " --inputWeightFiles "+outputDirComb+"/fileLists/data"+y+"_"+tagID+"_3b_weights_FvT.txt "

            cmd = runCMD + inputFile + inputWeights + outDir + picoOut  + yearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " --FvTName weight_FvT"+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tagID+"_v"+s, outputDir=outputDir, filePrefix="histsNoFvT_3b_"))
    

            # 3b TTbar 
            for tt in ttbarSamples:
                inputFile = " -i "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_3b_wFvT.txt "
                inputWeights = " --inputWeightFiles "+outputDirComb+"/fileLists/"+tt+y+"_"+tagID+"_3b_weights_FvT.txt "
                cmd = runCMD + inputFile + inputWeights + outDir + picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " --FvTName weight_FvT"+FvTName
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

    yearsToPlot = ["RunII"]

    for s in subSamples:

        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix

        histName3b = "hists_3b_wJCM_"+JCMName+"_wNoFVT_"+tagID+".root "
        #histName4b = "hists_4b_noFVT_"+tagID+".root "
        histName4b = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
        #histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "
        histName4bTTNoPSData = "hists_4b_wFVT_noPSData_"+tagID+".root "

        for y in yearsToPlot:
    
            #
            # MAke Plots
            #
            qcdFile     = getOutDir()+"/QCD"+y+"/"+histName3b
            data3bFile  = getOutDir()+"/data"+y+"_"+tagID+"/"+histName3b    if not y == "RunII" else getOutDir()+"/data"+y+"/"+histName3b               
            data4bFile  = getOutDir()+"/mixed"+y+"_"+mixedName+"_"+tagID+"_v"+s+"_wFvT/"+histName4b if not y == "RunII" else getOutDir()+"/mixed"+y+"/"+histName4b                
            ttbar4bFile = getOutDir()+"/TT"+y+"/"+histName4bTTNoPSData
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
        for y in yearsToPlot:
            #cmds.append("mv CutFlow_noFvT_"+y+FvTName+"_"+tagID+".pdf "+outputDir+"/")
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+y+FvTName+"_"+tagID+".tar plotsNoFvT_"+y+FvTName+"_"+tagID)
            
    babySit(cmds, doRun)    






if o.haddSubSamples: 

    dag_config = []
    condor_jobs = []

    histNameComb3bwFvT  = "hists_3b_wJCM_wFVT_vAll_"+mixedName+"_"+tagID+"_e25_os012.root "
    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+"_e25_os012.root "
    histNameComb3bnoFvT = "hists_3b_wJCM_wNoFVT_"+mixedName+"_"+tagID+".root "

    cmdData3bwFvT = "hadd -f "+getOutDir()+"/dataRunII/"+histNameComb3bwFvT+" "
    cmdData4b     = "hadd -f "+getOutDir()+"/mixedRunII/"+histNameComb4b+" "
    cmdQCD3b      = "hadd -f "+getOutDir()+"/QCDRunII/"+histNameComb3bnoFvT+" "


    for s in subSamples:
        weightPostFix = ""
        JCMName=mixedName+"_v"+s+weightPostFix
        FvTName="_"+mixedName+"_v"+s+weightPostFix+"_e25_os012"

        histName3bwFvT = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_"+tagID+".root "
        cmdData3bwFvT += getOutDir()+"/dataRunII/"+histName3bwFvT+" "

        histName4b     = "hists_4b_wFVT"+FvTName+"_"+tagID+".root "
        cmdData4b += getOutDir()+"/mixedRunII/"+histName4b+" "


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




if o.scaleCombSubSamples: 

    dag_config = []
    condor_jobs = []

    histNameComb3bwFvT  = "hists_3b_wJCM_wFVT_vAll_"+mixedName+"_"+tagID+"_e25_os012.root "
    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+"_e25_os012.root "
    histNameComb3bnoFvT = "hists_3b_wJCM_wNoFVT_"+mixedName+"_"+tagID+".root "

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor 0.1 "

    cmdData3bwFvT = cmdScale + " -i "+getOutDir()+"/dataRunII/"+histNameComb3bwFvT+" "
    condor_jobs.append(makeCondorFile(cmdData3bwFvT, getOutDir(), "dataRunII", outputDir=outputDir, filePrefix="scaleCombSubSamples_3b_"))            

    cmdData4b     = cmdScale + " -i "+getOutDir()+"/mixedRunII/"+histNameComb4b+" "
    condor_jobs.append(makeCondorFile(cmdData4b, getOutDir(), "mixedRunII", outputDir=outputDir, filePrefix="scaleCombSubSamples_4b_"))            

    cmdQCD3b      = cmdScale + " -i "+getOutDir()+"/QCDRunII/"+histNameComb3bnoFvT+" "
    condor_jobs.append(makeCondorFile(cmdQCD3b, getOutDir(), "QCDRunII", outputDir=outputDir, filePrefix="scaleCombSubSamples_"))            

    dag_config.append(condor_jobs)
    execute("rm "+outputDir+"scaleCombSubSamples_All.dag", doRun)
    execute("rm "+outputDir+"scaleCombSubSamples_All.dag.*", doRun)

    dag_file = makeDAGFile("scaleCombSubSamples_All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)




#
#  Make Plots with FvT
#
if o.plotsCombinedSamples:
    cmds = []
    logs = []

    histNameComb3bwFvT  = "hists_3b_wJCM_wFVT_vAll_"+mixedName+"_"+tagID+"_e25_os012_scaled.root "
    histNameComb4b      = "hists_4b_wFVT_vAll_"+mixedName+"_"+tagID+"_e25_os012_scaled.root "
    histNameComb3bnoFvT = "hists_3b_wJCM_wNoFVT_"+mixedName+"_"+tagID+"_scaled.root "
    histName4bTTNoPSData = "hists_4b_wFVT_noPSData_"+tagID+"_e25_os012.root "
    #histName4bTT = "hists_4b_wFVT_"+mixedName+"vAll_"+tagID+".root "

    #
    # Make Plots
    #
    data4bFile  = getOutDir()+"/mixedRunII/"+histNameComb4b
    ttbar4bFile = getOutDir()+"/TTRunII/"+histName4bTTNoPSData
    data3bFile  = getOutDir()+"/dataRunII/"+histNameComb3bwFvT
    qcd3bFile   = getOutDir()+"/QCDRunII/"+histNameComb3bnoFvT

    cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID+""+plotOpts["RunII"]+" -m -j -r --noSignal "
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
        FvTName="_"+mixedName+"_v"+s+weightPostFix+"_e25_os012"
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
    cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID+"_e25_os012.tar plotsWithFvT_RunII_Combined_"+mixedName+"_"+tagID+"_e25_os012")
    cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_RunII_Combined_"+mixedName+"_"+tagID+".tar plotsNoFvT_RunII_Combined_"+mixedName+"_"+tagID)
            
    for s in subSamples:
        FvTName="_"+mixedName+"_v"+s+weightPostFix+"_e25_os012"
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_RunII_Combined"+FvTName+"_"+tagID+".tar plotsWithFvT_RunII_Combined"+FvTName+"_"+tagID)

    babySit(cmds, doRun)    






#
# The rest is not updated....
#

################################################################################################




        






        



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
