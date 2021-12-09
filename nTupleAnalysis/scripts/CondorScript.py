
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse
from glob import glob

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9", help="Year or comma separated list of subsamples")

parser.add_option('--histDetailStr',                    default="allEvents.passMDRs", help="Year or comma separated list of subsamples")
parser.add_option('--mixedName',                        default="3bDvTMix4bDvT", help="Year or comma separated list of subsamples")
parser.add_option('--makeTarball',  action="store_true",      help="make Output file lists")
parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
# parser.add_option('--noFvT', action="store_true",      help="Make hist.root without FvT corrections")

o, a = parser.parse_args()

doRun = o.execute

from condorHelpers import *

CMSSW = getCMSSW()
USER = getUSER()
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/ZH4b/UL/"
#EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/ZH4b/TestSindhu/"
TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"

outputDir="closureTests/UL/"

runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'

years = o.year.split(",")
subSamples = o.subSamples.split(",")
mixedName=o.mixedName


def getOutDir():
    return EOSOUTDIR



yearOpts = {}
#yearOpts["2018"]=' -y 2018 --bTag 0.2770 '
#yearOpts["2017"]=' -y 2017 --bTag 0.3033 '
#yearOpts["2016"]=' -y 2016 --bTag 0.3093 '
yearOpts["2018"]=' -y 2018 --bTag 0.6 '
yearOpts["2017"]=' -y 2017 --bTag 0.6 '
yearOpts["2016"]=' -y 2016 --bTag 0.6 '

if o.makeTarball:
    #print "Remove old Tarball"
    rmTARBALL(o.execute)
    makeTARBALL(o.execute, debug=True)


#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithFvT: 

    dag_config = []
    condor_jobs = []
    jobName = "histsWithFvT_"

    
    noPico = " -p NONE "
    hist3b        = " --histDetailLevel threeTag."+o.histDetailStr#+"passSRvsSB1p.passSRvsSB10p"
    hist4b        = " --histDetailLevel fourTag."+o.histDetailStr#+"passSRvsSB1p.passSRvsSB10p"
    outDir = " -o "+getOutDir()+" "


    for y in years:

        #
        # Nominal
        #

##        JCMName="Nominal"
##        FvTName="_Nominal"
##
##        for tagData in [("3b",hist3b),("4b",hist4b)]:
##
##            tag = tagData[0]
##            histDetail = tagData[1]
##
##            histName = "hists_"+tag+"_wFvT"+FvTName+".root"
##
##            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM.txt "
##            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM_weights.txt"
##
##            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +  yearOpts[y] + " --histFile "+histName + histDetail + " --jcmNameLoad "+JCMName+" -r --FvTName  FvT"+FvTName
##            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
##            
##
##            # 3b TTbar not needed 
##            if tag == "4b":
##
##                for tt in ttbarSamplesByYear[y]:
##                    inputFile = " -i "+outputDir+"/fileLists/"+tt+"_"+tag+"_wJCM.txt "
##                    inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/"+tt+"_"+tag+"_wJCM_weights.txt"                
##
##                    cmd = runCMD + inputFile + inputWeights + outDir + noPico  + MCyearOpts(tt) + " --histFile " + histName + histDetail  + " --jcmNameLoad "+JCMName+ " -r --FvTName FvT"+FvTName
##                    condor_jobs.append(makeCondorFile(cmd, "None", tt+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            
        
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
            inputWeights   = " --inputWeightFiles "+outputDir+"/fileLists/data"+y+"_3b_wJCM_weights.txt"
            cmd = runCMD + inputFile + inputWeights + outDir + noPico + yearOpts[y] + " --histFile " + histName + hist3b + " --jcmNameLoad "+JCMName+ " -r "#--FvTName FvT"+FvTName
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_3b"+FvTName, outputDir=outputDir, filePrefix=jobName))


            #
            # 4b
            #
            inputFile = " -i "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wJCM_v"+s+".txt"
            inputWeights = " --inputWeightFiles "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wJCM_v"+s+"_weights.txt"

            cmd = runCMD + inputFile + inputWeights + outDir +  noPico + yearOpts[y] + " --histFile " + histName + hist4b + " --unBlind  --isDataMCMix " #+ "  --FvTName FvT"+FvTName 
            condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+FvTName, outputDir=outputDir, filePrefix=jobName))
            
##            for tt in ttbarSamplesByYear[y]:
##
##                histName = "hists_4b_noPSData_wFvT"+FvTName+".root"    
##                inputFile = " -i "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wJCM.txt"
##                inputWeights = " --inputWeightFiles "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wJCM_weights.txt"
##                
##                cmd = runCMD + inputFile + inputWeights + outDir + noPico + MCyearOpts(tt)+ " --histFile " + histName + hist4b + "  --FvTName FvT"+FvTName
##                condor_jobs.append(makeCondorFile(cmd, "None", tt+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))
##
    

    dag_config.append(condor_jobs)

    #
    #  Hadd TTbar
    #
##    condor_jobs = []
##
##    for y in years:
##        
##        FvTName="_Nominal"
##        histName = "hists_4b_wFvT"+FvTName+".root"
##
##        cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
##        for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_4b_wJCM/"+histName+" "
##        condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_4b"+FvTName, outputDir=outputDir, filePrefix=jobName))
##
##        for s in subSamples:
##
##            FvTName="_"+mixedName+"_v"+s
##            histName = "hists_4b_noPSData_wFvT"+FvTName+".root"    
##
##            cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
##            for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_4b_noPSData_wJCM/"+histName+" "
##            condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))
##
##
##        FvTName="_"+mixedName+"_vAll"
##        histName = "hists_4b_noPSData_wFvT"+FvTName+"_oneFit.root"    
##
##        cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
##        for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_4b_noPSData_wJCM/"+histName+" "
##        condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))
##
##
##
##    dag_config.append(condor_jobs)


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:

        condor_jobs = []            

        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/mixedRunII_"+mixedName, doRun)
        #mkdir(outputDir+"/TTRunII",   doRun)
        

##        #
##        #  Nominal
##        #
##        for tag in ["3b","4b"]:
##
##            FvTName="_Nominal"
##            histName = "hists_"+tag+"_wFvT"+FvTName+".root"
##
##            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
##            for y in years: cmd += getOutDir()+"/data"+y+"_"+tag+"_wJCM/"+histName+" "
##            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))            
##
##            if tag == "4b":
##                cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName+" "
##                for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName+" "
##                condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_4b"+FvTName, outputDir=outputDir, filePrefix=jobName))            


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

##            histName = "hists_4b_noPSData_wFvT"+FvTName+".root"    
##            cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName+" "
##            for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName+" "
##            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))            


        dag_config.append(condor_jobs)



    

    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)
