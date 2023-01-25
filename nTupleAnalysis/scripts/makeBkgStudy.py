import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse
from glob import glob

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14", help="Year or comma separated list of subsamples")
parser.add_option('--histDetailStr',                    default="allEvents.passMDRs", help="Year or comma separated list of subsamples")


parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--makeTarball',  action="store_true",      help="make Output file lists")


parser.add_option('--makeInputFileListsFriendsRW',  action="store_true",      help=" ")

parser.add_option('--makeInputFileListsFriendsOT',  action="store_true",      help=" ")

parser.add_option('--makeInputFileListsFriendsNN',  action="store_true",      help=" ")

parser.add_option('-c',   '--condor',   action="store_true", default=False,           help="Run on condor")

parser.add_option('--testDvTWeightsWJCM',  action="store_true",      help="make Input file lists")
parser.add_option(     '--no3b',        action="store_true", help="boolean  to toggle using FvT reweight")
parser.add_option(     '--doDvTReweight',        action="store_true", help="boolean  to toggle using FvT reweight")

parser.add_option('--copyFromAutonForDvTROOT', action="store_true",      help="copy h5 picos to Auton")
parser.add_option('--gpuName',                    default="", help="")
parser.add_option('--weightName', default="weights",      help="copy h5 picos to Auton")

parser.add_option('--makeInputFileListsFriends',  action="store_true",      help=" ")


parser.add_option('--histsWithRW', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsWithRW', action="store_true",      help="Make pdfs with FvT")

parser.add_option('--histsWithOT', action="store_true",      help="Make hist.root with OT")
parser.add_option('--plotsWithOT', action="store_true",      help="Make pdfs with FvT")

parser.add_option('--histsWithNN', action="store_true",      help="Make hist.root with OT")
parser.add_option('--plotsWithNN', action="store_true",      help="Make pdfs with FvT")

parser.add_option('--histsNoFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsNoFvT', action="store_true",      help="Make pdfs with FvT")

parser.add_option('--makeInputsForCombine', action="store_true",      help="Make inputs for the combined tool")

o, a = parser.parse_args()

doRun = o.execute

from condorHelpers import *

CMSSW = getCMSSW()
USER = getUSER()
#EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/ZH4b/UL/"
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/ZH4b/ULTrig/"
TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"


outputDir="closureTests/ULTrig/"
outputAutonDir ="hh4b/closureTests/ULTrig"

#ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]
signalSamples = ["ZZ4b","ZH4b","ggZH4b"]
#signalSamplesVHH = ["WHHTo4B_CV_1_0_C2V_1_0_C3_1_0_","ZHHTo4B_CV_1_0_C2V_1_0_C3_1_0_"]

years = o.year.split(",")
subSamples = o.subSamples.split(",")
mixedName=o.mixedName

if o.subSamples == "30":
    subSamples = [str(i) for i in range(30)]
    print subSamples

if o.subSamples == "None":
    subSamples = []

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


pscAddr = "alison@vera.psc.edu"
def runPSC(cmd):
    print "> "+cmd
    run("ssh "+pscAddr+" "+cmd)


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



def scpFromScratchToEOS(pName, gpuName, autonPath, eosPath):

    tempPath = "/uscms/home/jda102/nobackup/forSCP/"

    localFile = tempPath+"/"+pName

    cmd = "scp "+gpuName+":"+autonPath+"/"+pName+" "+localFile
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


def scpPSC(eosDir, subdir, pName, autonDir):

    tempPath = "/uscms/home/jda102/nobackup/forSCP/"

    cmd = "xrdcp "+eosDir+"/"+subdir+"/"+pName+"  "+tempPath+pName
    run(cmd)


    cmd = "scp "+tempPath+pName+" "+pscAddr+":/hildafs/projects/phy210037p/alison/"+autonDir+"/"+subdir+"/"+pName
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

    print "ERROR cant find year",tt
    import sys
    sys.exit(-1)




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
mixedAnalysisCMD='mixedEventAnalysis ZZ4b/nTupleAnalysis/scripts/mixedEventAnalysis_cfg.py'

plotOpts = {}
plotOpts["2018"]=" -l 60.0e3 -y 2018"
plotOpts["2017"]=" -l 36.7e3 -y 2017"
plotOpts["2016"]=" -l 35.9e3 -y 2016"
plotOpts["RunII"]=" -l 132.6e3 -y RunII"




# 
#  Test DvT Weights
#
if o.testDvTWeightsWJCM:

    dag_config = []
    condor_jobs = []

    jobName = "testDvTWeightsWJCM_"
    if o.doDvTReweight:
        jobName = "testDvTWeightsWJCM_wDvT_"


    histDetail3b        = " --histDetailLevel allEvents.passPreSel.passMDRs.threeTag.failrWbW2.passMuon.passDvT05.DvT "
    histDetail4b        = " --histDetailLevel allEvents.passPreSel.passMDRs.fourTag.failrWbW2.passMuon.passDvT05.DvT "

    picoOut = " -p None " 
    outDir = " -o "+getOutDir()+" "

    tagList = []
    if not o.no3b:
        tagList.append( ("3b","DvT3","_pt3",histDetail3b))
    tagList.append( ("4b","DvT4","_pt4", histDetail4b) )

    for tag in tagList:

        histName = "hists_"+tag[0]+"_wJCM.root"
        if o.doDvTReweight:
            histName = "hists_"+tag[0]+"_wJCM_rwDvT.root"


        histOut  = " --histFile "+histName
        histDetail = tag[3]


        JCMName="Nominal"
        FvTName="_Nominal"

        for y in years:
        
            inputFile = " -i  "+outputDir+"/fileLists/data"+y+"_"+tag[0]+"_wJCM.txt "
            inputWeights = " --friends "+outputDir+"/fileLists/data"+y+"_"+tag[0]+"_wJCM_friends_Nominal_DvT.txt " 
            DvTName      = " --reweightDvTName "+tag[1]+"_Nominal"

            cmd = runCMD+ inputFile + inputWeights + DvTName + picoOut + outDir + yearOpts[y]+ histDetail +  histOut  + " --jcmNameLoad "+JCMName+" --FvTName  FvT"+FvTName

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
                    inputFile = " -i  "+outputDir+"/fileLists/"+tt+"_"+tag[0]+"_wTrigW_wJCM.txt "
                    inputWeights = " --friends "+outputDir+"/fileLists/"+tt+"_"+tag[0]+"_wTrigW_wJCM_friends_Nominal_DvT.txt "
    
                    cmd = runCMD+ inputFile + inputWeights + DvTName + picoOut + outDir + MCyearOpts(tt) +histDetail + histOut + " --jcmNameLoad "+JCMName+ " --FvTName FvT"+FvTName + " --doTrigEmulation "

                    condor_jobs.append(makeCondorFile(cmd, "None", tt, outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))                    
    
    

    dag_config.append(condor_jobs)


    #
    #  Hadd ttbar
    #
    if not o.doDvTReweight:
        condor_jobs = []

        for tag in tagList:

            histName = "hists_"+tag[0]+"_wJCM.root"

            for y in years:
            
                cmd = "hadd -f "+ getOutDir()+"/TT"+y+"/"+histName+" "
                for tt in ttbarSamplesByYear[y]:        
                    cmd += getOutDir()+"/"+tt+"_"+tag[0]+"_wTrigW_wJCM/"+histName+" "
                condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y, outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))            
    
    
        dag_config.append(condor_jobs)
        

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        condor_jobs = []        

        for tag in tagList:

            histName = "hists_"+tag[0]+"_wJCM.root"

            #
            #  TTbar
            #
            if not o.doDvTReweight:

                cmd = "hadd -f " + getOutDir()+"/TTRunII/"+ histName+" "
                for y in years:
                    cmd += getOutDir()+"/TT"+y+"/"  +histName+" "
    
                condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII", outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))            


            if o.doDvTReweight:
                histName = "hists_"+tag[0]+"_wJCM_rwDvT.root"
    
            #
            #  Data
            #
            cmd = "hadd -f " + getOutDir()+"/dataRunII/"+ histName+" "
            for y in years:
                cmd += getOutDir()+"/data"+y+"_"+tag[0]+"_wJCM/"  +histName+" "

            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII", outputDir=outputDir, filePrefix=jobName+tag[0]+"_"))            



        dag_config.append(condor_jobs)            



    #
    # Subtract QCD 
    #
    if not o.doDvTReweight:

        condor_jobs = []
    
        for tag in tagList:
            histName = "hists_"+tag[0]+"_wJCM.root"

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
#  Get JCM Files
#    (Might be able to kill...)
jcmNameList="Nominal"
jcmFileList = {}

JCMTagNom = "05-00-00"
JCMTagMixed = "05-00-00"


for y in years:
    jcmFileList[y] = outputDir+"/weights/dataRunII/jetCombinatoricModel_SB_"+JCMTagNom+".txt"


for s in subSamples:
    jcmNameList   += ","+mixedName+"_v"+s
    for y in years:
        jcmFileList[y] += ","+outputDir+"/weights/mixedRunII_"+mixedName+"_v"+s+"/jetCombinatoricModel_SB_"+JCMTagMixed+".txt"



# 
#  Copy to AUTON
#
if o.copyFromAutonForDvTROOT:
    
    import os
    autonAddr = "gpu13"
    


    #
    # Copy Files
    #
    if o.copyFromAutonForDvTROOT:

        if o.gpuName:
            outputAutonDir =  "/home/scratch/jalison/closureTests/ULTrig/"

        for y in years:

            #
            #  4b
            #
            
            for outFile in ["DvT4_Nominal"]:
                
                scpFromScratchToEOS(outFile+".root", o.gpuName, outputAutonDir+"/data"+y+"_4b" , EOSOUTDIR+"data"+y+"_4b")
                
                for tt in ttbarSamplesByYear[y]:
                    scpFromScratchToEOS(outFile+".root", o.gpuName, outputAutonDir+"/"+tt+"_4b_wTrigW", EOSOUTDIR+tt+"_4b_wTrigW")
    

            #
            #  3b
            # 
            for outFile in  ["DvT3_Nominal"]:
                scpFromScratchToEOS(outFile+".root", o.gpuName, outputAutonDir+"/data"+y+"_3b" , EOSOUTDIR+"data"+y+"_3b")

                for tt in ttbarSamplesByYear[y]:
                    scpFromScratchToEOS(outFile+".root", o.gpuName, outputAutonDir+"/"+tt+"_3b_wTrigW", EOSOUTDIR+tt+"_3b_wTrigW")


            for s in subSamples:
                for outFile in ["DvT4_"+mixedName+"_v"+s]:
                    scpFromScratchToEOS(outFile+".root",o.gpuName, outputAutonDir+"/mixed"+y+"_"+mixedName+"_v"+s,EOSOUTDIR+"mixed"+y+"_"+mixedName+"_v"+s)

            for tt in ttbarSamplesByYear[y]:
                for outFile in ["DvT4_"+mixedName+"_v0"]:
                    scpFromScratchToEOS(outFile+".root", o.gpuName, outputAutonDir+"/"+tt+"_4b_noPSData_wTrigW", EOSOUTDIR+tt+"_4b_noPSData_wTrigW")







#
#   Make inputs fileLists
#
if o.makeInputFileListsFriends:
    

    for y in years:
        fileName = "friends_Nominal"
    
        #
        #  4b
        #
        fileList = outputDir+"/fileLists/data"+y+"_4b_wJCM_"+fileName+".txt"    
        run("rm "+fileList)
        for outFile in ["FvT_Nominal","SvB","SvB_MA","SvB_MA_VHH"]:                
            run("echo "+EOSOUTDIR+"/data"+y+"_4b/"+outFile+".root >> "+fileList)
    
        for tt in ttbarSamplesByYear[y]:
            fileList = outputDir+"/fileLists/"+tt+"_4b_wTrigW_wJCM_"+fileName+".txt"    
            run("rm "+fileList)
            for outFile in ["FvT_Nominal","SvB","SvB_MA","SvB_MA_VHH"]:                
                run("echo "+EOSOUTDIR+"/"+tt+"_4b_wTrigW/"+outFile+".root >> "+fileList)

        #
        # 3b
        # 
        fileList = outputDir+"/fileLists/data"+y+"_3b_wJCM_"+fileName+".txt"    
        run("rm "+fileList)
        for outFile in ["FvT_Nominal","SvB","SvB_MA","SvB_MA_VHH"]:                
            run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+outFile+".root >> "+fileList)
    
        for tt in ttbarSamplesByYear[y]:
            fileList = outputDir+"/fileLists/"+tt+"_3b_wTrigW_wJCM_"+fileName+".txt"    
            run("rm "+fileList)
            for outFile in ["FvT_Nominal","SvB","SvB_MA","SvB_MA_VHH"]:                
                run("echo "+EOSOUTDIR+"/"+tt+"_3b_wTrigW/"+outFile+".root >> "+fileList)

    
        #
        # Mixed
        #
        allSubSamples = ["v"+s for s in subSamples] 
        allSubSamples += ["vAll"]
        for vs in allSubSamples:

            fileName = "friends_"+mixedName+"_"+vs
    
            fileList = outputDir+"/fileLists/data"+y+"_3b_wJCM_"+fileName+".txt"    
            run("rm "+fileList)
            for outFile in ["FvT_"+mixedName+"_"+vs,"SvB","SvB_MA","SvB_MA_VHH"]:    
                run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+outFile+".root >> "+fileList)


            if vs not in ["vAll"]:
                fileList = outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wJCM_"+fileName+".txt"    
                run("rm "+fileList)
                for outFile in ["FvT_"+mixedName+"_"+vs,"SvB","SvB_MA","SvB_MA_VHH"]:    
                    run("echo "+EOSOUTDIR+"/mixed"+y+"_"+mixedName+"_"+vs+"/"+outFile+".root >> "+fileList)
     

            for tt in ttbarSamplesByYear[y]:
                fileList = outputDir+"/fileLists/"+tt+"_4b_noPSData_wTrigW_wJCM_"+fileName+".txt"    
                run("rm "+fileList)

                for outFile in ["FvT_"+mixedName+"_"+vs,"SvB","SvB_MA","SvB_MA_VHH"]:    
                    run("echo "+EOSOUTDIR+"/"+tt+"_4b_noPSData_wTrigW/"+outFile+".root >> "+fileList)




#
#   Make inputs fileLists
#
if o.makeInputFileListsFriendsRW:
    

    for y in years:
        fileName = "friends_Nominal_"+o.weightName
    
        #
        #  4b
        #
        fileList = outputDir+"/fileLists/data"+y+"_4b_wJCM_"+fileName+".txt"    
        run("rm "+fileList)
        for outFile in ["FvT_Nominal_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT4_Nominal_newSBDef"]:                
            run("echo "+EOSOUTDIR+"/data"+y+"_4b/"+outFile+".root >> "+fileList)
    
        for tt in ttbarSamplesByYear[y]:
            fileList = outputDir+"/fileLists/"+tt+"_4b_wTrigW_wJCM_"+fileName+".txt"    
            run("rm "+fileList)
            for outFile in ["FvT_Nominal_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT4_Nominal_newSBDef"]:                
                run("echo "+EOSOUTDIR+"/"+tt+"_4b_wTrigW/"+outFile+".root >> "+fileList)

        #
        # 3b
        # 
        fileList = outputDir+"/fileLists/data"+y+"_3b_wJCM_"+fileName+".txt"    
        run("rm "+fileList)
        for outFile in ["FvT_Nominal_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT3_Nominal_newSBDef"]:                
            run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+outFile+".root >> "+fileList)

        run("echo root://cmseos.fnal.gov//store/user/jda102/condor/OT/RW_"+y+"_Nominal/data"+y+"_3b_picoAOD_3b_wJCM_newSBDef_weight_rw_iter10.root >> " +fileList)

        for tt in ttbarSamplesByYear[y]:
            fileList = outputDir+"/fileLists/"+tt+"_3b_wTrigW_wJCM_"+fileName+".txt"    
            run("rm "+fileList)
            for outFile in ["FvT_Nominal_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT3_Nominal_newSBDef"]:                
                run("echo "+EOSOUTDIR+"/"+tt+"_3b_wTrigW/"+outFile+".root >> "+fileList)

    
        #
        # Mixed
        #
        allSubSamples = ["v"+s for s in subSamples] 
        #allSubSamples += ["vAll"]
        for vs in allSubSamples:

            fileName = "friends_"+mixedName+"_"+vs+"_"+o.weightName
    
            fileList = outputDir+"/fileLists/data"+y+"_3b_wJCM_"+fileName+".txt"    
            run("rm "+fileList)
            #for outFile in ["FvT_"+mixedName+"_"+vs+"_newSB","SvB_newSB","SvB_MA_newSB","SvB_MA_VHH_newSB"]:    
            for outFile in ["FvT_"+mixedName+"_"+vs+"_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT3_Nominal_newSBDef"]:    
                run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+outFile+".root >> "+fileList)

            run("echo root://cmseos.fnal.gov//store/user/jda102/condor/OT/RW_"+y+"_"+vs+"/data"+y+"_3b_picoAOD_3b_wJCM_newSBDef_weight_rw_iter10.root >> " +fileList)

            if vs not in ["vAll"]:
                fileList = outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wJCM_"+fileName+".txt"    
                run("rm "+fileList)
                for outFile in ["FvT_"+mixedName+"_"+vs+"_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT4_"+mixedName+"_"+vs+"_newSBDef"]:    
                    run("echo "+EOSOUTDIR+"/mixed"+y+"_"+mixedName+"_"+vs+"/"+outFile+".root >> "+fileList)
     

            for tt in ttbarSamplesByYear[y]:
                fileList = outputDir+"/fileLists/"+tt+"_4b_noPSData_wTrigW_wJCM_"+fileName+".txt"    
                run("rm "+fileList)

                #for outFile in ["FvT_"+mixedName+"_"+vs+"_newSB","SvB_newSB","SvB_MA_newSB","SvB_MA_VHH_newSB","DvT4_"+mixedName+"_"+vs+"_newSB"]:
                for outFile in ["FvT_"+mixedName+"_"+vs+"_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT4_"+mixedName+"_"+vs+"_newSBDef"]:
                    run("echo "+EOSOUTDIR+"/"+tt+"_4b_noPSData_wTrigW/"+outFile+".root >> "+fileList)



#
#   Make inputs fileLists
#
if o.makeInputFileListsFriendsOT:
    

    for y in years:
        fileName = "friends_Nominal_"+o.weightName
    

        #
        # 3b
        # 
        fileList = outputDir+"/fileLists/data"+y+"_3b_wJCM_"+fileName+".txt"    
        run("rm "+fileList)
        for outFile in ["FvT_Nominal_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT3_Nominal_newSBDef"]:                
            run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+outFile+".root >> "+fileList)

        run("echo root://cmseos.fnal.gov//store/user/jda102/condor/OT/OT_Nominal/data"+y+"_3b_picoAOD_3b_wJCM_newSBDef_OTWeights_Random.root >> "+fileList)
    
        #
        # Mixed
        #
        allSubSamples = ["v"+s for s in subSamples] 
        #allSubSamples += ["vAll"]
        for vs in allSubSamples:

            fileName = "friends_"+mixedName+"_"+vs+"_"+o.weightName
    
            fileList = outputDir+"/fileLists/data"+y+"_3b_wJCM_"+fileName+".txt"    
            run("rm "+fileList)
            #for outFile in ["FvT_"+mixedName+"_"+vs+"_newSB","SvB_newSB","SvB_MA_newSB","SvB_MA_VHH_newSB"]:    
            for outFile in ["FvT_"+mixedName+"_"+vs+"_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT3_Nominal_newSBDef"]:    
                run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+outFile+".root >> "+fileList)


            run("echo root://cmseos.fnal.gov//store/user/jda102/condor/OT/OT_"+vs+"/data"+y+"_3b_picoAOD_3b_wJCM_newSBDef_OTWeights_3bDvTMix4bDvT_"+vs+"_Random.root >> " + fileList)


#
#   Make inputs fileLists
#
if o.makeInputFileListsFriendsNN:
    

    for y in years:
        fileName = "friends_Nominal_"+o.weightName
    

        #
        # 3b
        # 
        fileList = outputDir+"/fileLists/data"+y+"_3b_wJCM_"+fileName+".txt"    
        run("rm "+fileList)
        for outFile in ["FvT_Nominal_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT3_Nominal_newSBDef"]:                
            run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+outFile+".root >> "+fileList)

        run("echo root://cmseos.fnal.gov//store/user/jda102/condor/OT/NN_Nominal/data"+y+"_3b_picoAOD_3b_wJCM_newSBDef_OTWeights_FvTClosure.root >> "+fileList)
    
        #
        # Mixed
        #
        allSubSamples = ["v"+s for s in subSamples] 
        #allSubSamples += ["vAll"]
        for vs in allSubSamples:

            fileName = "friends_"+mixedName+"_"+vs+"_"+o.weightName
    
            fileList = outputDir+"/fileLists/data"+y+"_3b_wJCM_"+fileName+".txt"    
            run("rm "+fileList)
            #for outFile in ["FvT_"+mixedName+"_"+vs+"_newSB","SvB_newSB","SvB_MA_newSB","SvB_MA_VHH_newSB"]:    
            for outFile in ["FvT_"+mixedName+"_"+vs+"_newSBDef","SvB_newSBDef","SvB_MA_newSBDef","DvT3_Nominal_newSBDef"]:    
                run("echo "+EOSOUTDIR+"/data"+y+"_3b/"+outFile+".root >> "+fileList)


            run("echo root://cmseos.fnal.gov//store/user/jda102/condor/OT/NN_"+vs+"/data"+y+"_3b_picoAOD_3b_wJCM_newSBDef_OTWeights_3bDvTMix4bDvT_"+vs+"_FvTClosure.root >> " + fileList)






#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithRW: 

    dag_config = []
    condor_jobs = []
    jobName = "histsWithRW_"+o.weightName+"_"

    
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

            histName = "hists_"+tag+"_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"

            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM.txt "
            inputWeights   = " --friends "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM_friends_Nominal_"+o.weightName+".txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +  yearOpts[y] + " --histFile "+histName + histDetail + " --jcmNameLoad "+JCMName+" --FvTName  FvT"+FvTName+"_newSBDef"
            cmd += " --runKlBdt "
            if tag == "3b":
                cmd += " --reweightDvTName DvT3_Nominal_newSBDef"
                cmd += " --doDvTReweight "
                cmd += " --otherWeights weight_rw_iter10"

            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            
        
            #
            #  4b ttbar
            #

            # 3b TTbar not needed 
            if tag == "4b":

                for tt in ttbarSamplesByYear[y]:
                    inputFile = " -i "+outputDir+"/fileLists/"+tt+"_"+tag+"_wTrigW_wJCM.txt "
                    inputWeights   = " --friends "+outputDir+"/fileLists/"+tt+"_"+tag+"_wTrigW_wJCM_friends_Nominal_"+o.weightName+".txt"                
                 
                    cmd = runCMD + inputFile + inputWeights + outDir + noPico  + MCyearOpts(tt) + " --histFile " + histName + histDetail  + " --jcmNameLoad "+JCMName+ " --FvTName FvT"+FvTName+"_newSBDef" + " --doTrigEmulation "
                    cmd += " --runKlBdt "
                    condor_jobs.append(makeCondorFile(cmd, "None", tt+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))



        #
        #  SubSamples
        #
        for s in subSamples:

            JCMName=mixedName+"_v"+s
            FvTName="_"+mixedName+"_v"+s

            histName = "hists_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"

            #
            # 3b
            #
            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_3b_wJCM.txt "
            inputWeights   = " --friends "+outputDir+"/fileLists/data"+y+"_3b_wJCM_friends_"+JCMName+"_"+o.weightName+".txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico + yearOpts[y] + " --histFile " + histName + hist3b + " --jcmNameLoad "+JCMName+ " --FvTName FvT"+FvTName+"_newSBDef"
            cmd += " --runKlBdt "
            cmd += " --reweightDvTName DvT3_Nominal_newSBDef"
            cmd += " --doDvTReweight "
            cmd += " --otherWeights weight_rw_iter10"

            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_3b"+FvTName, outputDir=outputDir, filePrefix=jobName))


            #
            # 4b
            #
            inputFile = " -i "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wJCM_v"+s+".txt"
            inputWeights = " --friends "+outputDir+"/fileLists/mixed"+y+"_"+mixedName+"_wJCM_friends_"+JCMName+"_"+o.weightName+".txt"

            cmd = runCMD + inputFile + inputWeights + outDir +  noPico + yearOpts[y] + " --histFile " + histName + hist4b + "  --FvTName FvT"+FvTName+"_newSBDef" + " --unBlind  --isDataMCMix "
            cmd += " --runKlBdt "
            condor_jobs.append(makeCondorFile(cmd, "None", "mixed"+y+FvTName, outputDir=outputDir, filePrefix=jobName))
            
            for tt in ttbarSamplesByYear[y]:

                histName = "hists_4b_noPSData_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"

                inputFile = " -i "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wTrigW_wJCM.txt"
                inputWeights = " --friends "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wTrigW_wJCM_friends_"+JCMName+"_"+o.weightName+".txt"
                
                cmd = runCMD + inputFile + inputWeights + outDir + noPico + MCyearOpts(tt)+ " --histFile " + histName + hist4b + "  --FvTName FvT"+FvTName+"_newSBDef" + " --doTrigEmulation "
                cmd += " --runKlBdt "
                condor_jobs.append(makeCondorFile(cmd, "None", tt+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))

###        #
###        #  vAll
###        #
###        JCMName=mixedName+"_v4"
###        FvTName="_"+mixedName+"_vAll"
###        histName = "hists_wFvT"+FvTName+"_"+o.weightName+"_oneFit.root"
###    
###        #
###        # 3b
###        #
###        inputFile = " -i "+outputDir+"/fileLists/data"+y+"_3b_wJCM.txt "
###        inputWeights   = " --friends "+outputDir+"/fileLists/data"+y+"_3b_wJCM_friends_"+mixedName+"_vAll.txt"
###
###        cmd = runCMD + inputFile + inputWeights + outDir + noPico + yearOpts[y] + " --histFile " + histName + hist3b + " --jcmNameLoad "+JCMName+ " -r --FvTName FvT"+FvTName
###        cmd += " --runKlBdt "
###        condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_3b"+FvTName, outputDir=outputDir, filePrefix=jobName))
###
###        #
###        # 4b
###        #
###        for tt in ttbarSamplesByYear[y]:
###
###            histName = "hists_4b_noPSData_wFvT"+FvTName+"_"+o.weightName+"_oneFit.root"    
###            inputFile = " -i "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wTrigW_wJCM.txt"
###            inputWeights = " --friends "+outputDir+"/fileLists/"+tt+"_4b_noPSData_wTrigW_wJCM_friends_"+mixedName+"_vAll.txt"
###            
###            cmd = runCMD + inputFile + inputWeights + outDir + noPico + MCyearOpts(tt)+ " --histFile " + histName + hist4b + "  --FvTName FvT"+FvTName + " --doTrigEmulation "
###            cmd += " --runKlBdt "
###            condor_jobs.append(makeCondorFile(cmd, "None", tt+"_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))


    dag_config.append(condor_jobs)

    #
    #  Hadd TTbar
    #
    condor_jobs = []

    for y in years:
        
        FvTName="_Nominal"
        histName = "hists_"+tag+"_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"

        cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
        for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_4b_wTrigW_wJCM/"+histName+" "
        condor_jobs.append(makeCondorFile(cmd, "None", "TT"+y+"_4b"+FvTName, outputDir=outputDir, filePrefix=jobName))

        for s in subSamples:

            FvTName="_"+mixedName+"_v"+s
            histName = "hists_4b_noPSData_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"    

            cmd = "hadd -f "+getOutDir()+"/TT"+y+"/"+histName+" "
            for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_4b_noPSData_wTrigW_wJCM/"+histName+" "
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
            histName = "hists_"+tag+"_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"

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
            histName = "hists_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"    

            cmd = "hadd -f "+getOutDir()+"/mixedRunII_"+mixedName+"/"+histName+" "
            for y in years: cmd += getOutDir()+"/mixed"+y+"_"+mixedName+"_wJCM_v"+s+"/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "mixedRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_3b_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            

            histName = "hists_4b_noPSData_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"    
            cmd = "hadd -f "+getOutDir()+"/TTRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/TT"+y+"/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "TTRunII_4b_noPSData"+FvTName, outputDir=outputDir, filePrefix=jobName))            


####        FvTName="_"+mixedName+"_vAll"
####        histName = "hists_wFvT"+FvTName+"_"+o.weightName+"_oneFit.root"    
####
####        cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
####        for y in years: cmd += getOutDir()+"/data"+y+"_3b_wJCM/"+histName+" "
####        condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            


        dag_config.append(condor_jobs)

    #
    #  Hadd SubSamples
    #
    condor_jobs = []

    histNameAll = "hists_wRW_"+mixedName+"_"+o.weightName+"_vAll_newSBDef.root"    

    cmdData3b    = "hadd -f "+getOutDir()+"/dataRunII/"+histNameAll+" "
    cmdDataMixed = "hadd -f "+getOutDir()+"/mixedRunII/"+histNameAll+" "

    for s in subSamples:

        FvTName="_"+mixedName+"_v"+s
        histName = "hists_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"    

        cmdData3b    += getOutDir()+"/dataRunII/"+histName+" "
        cmdDataMixed += getOutDir()+"/mixedRunII_"+mixedName+"/"+histName+" "


    condor_jobs.append(makeCondorFile(cmdData3b,    "None", "dataRunII_vAll",  outputDir=outputDir, filePrefix=jobName))            
    condor_jobs.append(makeCondorFile(cmdDataMixed, "None", "mixedRunII_vAll", outputDir=outputDir, filePrefix=jobName))            
    dag_config.append(condor_jobs)

    #
    #  Scale SubSample
    #
    condor_jobs = []

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor  "+str(1.0/len(subSamples))

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
if o.histsWithOT: 

    dag_config = []
    condor_jobs = []
    jobName = "histsWithOT_"+o.weightName+"_"

    
    noPico = " -p NONE "
    hist3b        = " --histDetailLevel threeTag."+o.histDetailStr
    outDir = " -o "+getOutDir()+" "


    for y in years:

        #
        # Nominal
        #

        JCMName="Nominal"
        FvTName="_Nominal"

        for tagData in [("3b",hist3b)]:

            tag = tagData[0]
            histDetail = tagData[1]

            histName = "hists_"+tag+"_wOT"+FvTName+"_"+o.weightName+"_newSBDef.root"

            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM.txt "
            inputWeights   = " --friends "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM_friends_Nominal_"+o.weightName+".txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +  yearOpts[y] + " --histFile "+histName + histDetail + " --jcmNameLoad "+JCMName+" --FvTName  FvT"+FvTName+"_newSBDef"
            cmd += " --runKlBdt "
            if tag == "3b":
                cmd += " --otherWeights OTWeight"

            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            
        
            #
            #  4b and ttbar from RW hists
            #


        #
        #  SubSamples
        #
        for s in subSamples:

            JCMName=mixedName+"_v"+s
            FvTName="_"+mixedName+"_v"+s

            histName = "hists_wOT"+FvTName+"_"+o.weightName+"_newSBDef.root"

            #
            # 3b
            #
            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_3b_wJCM.txt "
            inputWeights   = " --friends "+outputDir+"/fileLists/data"+y+"_3b_wJCM_friends_"+JCMName+"_"+o.weightName+".txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico + yearOpts[y] + " --histFile " + histName + hist3b + " --jcmNameLoad "+JCMName+ " --FvTName FvT"+FvTName+"_newSBDef"
            cmd += " --runKlBdt "
            cmd += " --otherWeights OTWeight"

            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_3b"+FvTName, outputDir=outputDir, filePrefix=jobName))


            #
            # 4b
            #
            # From RW hists



    dag_config.append(condor_jobs)

    condor_jobs = []        

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/mixedRunII_"+mixedName, doRun)

        #
        #  Nominal
        #
        for tag in ["3b"]:

            FvTName="_Nominal"
            histName = "hists_"+tag+"_wOT"+FvTName+"_"+o.weightName+"_newSBDef.root"

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+tag+"_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))            



        #
        #  Mixed
        #
        for s in subSamples:

            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wOT"+FvTName+"_"+o.weightName+"_newSBDef.root"    


            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_3b_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            


        dag_config.append(condor_jobs)

    #
    #  Hadd SubSamples
    #
    condor_jobs = []

    histNameAll = "hists_wOT_"+mixedName+"_"+o.weightName+"_vAll_newSBDef.root"    

    cmdData3b    = "hadd -f "+getOutDir()+"/dataRunII/"+histNameAll+" "

    for s in subSamples:

        FvTName="_"+mixedName+"_v"+s
        histName = "hists_wOT"+FvTName+"_"+o.weightName+"_newSBDef.root"    

        cmdData3b    += getOutDir()+"/dataRunII/"+histName+" "


    condor_jobs.append(makeCondorFile(cmdData3b,    "None", "dataRunII_vAll",  outputDir=outputDir, filePrefix=jobName))            
    dag_config.append(condor_jobs)

    #
    #  Scale SubSample
    #
    condor_jobs = []

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor  "+str(1.0/len(subSamples))

    cmd = cmdScale + " -i "+getOutDir()+"/dataRunII/"+histNameAll+" "
    condor_jobs.append(makeCondorFile(cmd, getOutDir(), "dataRunII", outputDir=outputDir, filePrefix=jobName+"scale_"))            

    dag_config.append(condor_jobs)



    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)



#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithNN: 

    dag_config = []
    condor_jobs = []
    jobName = "histsWithNN_"+o.weightName+"_"
    if o.doDvTReweight: jobName += "rwDvT_"
        

    
    noPico = " -p NONE "
    hist3b        = " --histDetailLevel threeTag."+o.histDetailStr
    outDir = " -o "+getOutDir()+" "


    for y in years:

        #
        # Nominal
        #

        JCMName="Nominal"
        FvTName="_Nominal"

        for tagData in [("3b",hist3b)]:

            tag = tagData[0]
            histDetail = tagData[1]
            
            histName = "hists_"+tag+"_wNN"+FvTName+"_"+o.weightName+"_newSBDef.root"
            if o.doDvTReweight: histName = histName.replace(".root","_rwDvT.root")

            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM.txt "
            inputWeights   = " --friends "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM_friends_Nominal_"+o.weightName+".txt"
            DvTName      = " --reweightDvTName "+"DvT3_Nominal_newSBDef"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +  yearOpts[y] + " --histFile "+histName + histDetail + " --jcmNameLoad "+JCMName+" --FvTName  FvT"+FvTName+"_newSBDef"
            cmd += " --runKlBdt "
            if tag == "3b":
                cmd += " --otherWeights NNWeight"

            if o.doDvTReweight:   cmd += " --reweightDvTName DvT3_Nominal_newSBDef  --doDvTReweight "        

            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            
        
            #
            #  4b and ttbar from RW hists
            #


        #
        #  SubSamples
        #
        for s in subSamples:

            JCMName=mixedName+"_v"+s
            FvTName="_"+mixedName+"_v"+s

            histName = "hists_wNN"+FvTName+"_"+o.weightName+"_newSBDef.root"
            if o.doDvTReweight: histName = histName.replace(".root","_rwDvT.root")

            #
            # 3b
            #
            inputFile = " -i "+outputDir+"/fileLists/data"+y+"_3b_wJCM.txt "
            inputWeights   = " --friends "+outputDir+"/fileLists/data"+y+"_3b_wJCM_friends_"+JCMName+"_"+o.weightName+".txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico + yearOpts[y] + " --histFile " + histName + hist3b + " --jcmNameLoad "+JCMName+ " --FvTName FvT"+FvTName+"_newSBDef"
            cmd += " --runKlBdt "
            cmd += " --otherWeights NNWeight"
            if o.doDvTReweight:   cmd += " --reweightDvTName DvT3_Nominal_newSBDef  --doDvTReweight "        

            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_3b"+FvTName, outputDir=outputDir, filePrefix=jobName))


            #
            # 4b
            #
            # From RW hists



    dag_config.append(condor_jobs)

    condor_jobs = []        

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/mixedRunII_"+mixedName, doRun)

        #
        #  Nominal
        #
        for tag in ["3b"]:

            FvTName="_Nominal"
            histName = "hists_"+tag+"_wNN"+FvTName+"_"+o.weightName+"_newSBDef.root"
            if o.doDvTReweight: histName = histName.replace(".root","_rwDvT.root")

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_"+tag+"_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))            



        #
        #  Mixed
        #
        for s in subSamples:

            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wNN"+FvTName+"_"+o.weightName+"_newSBDef.root"    
            if o.doDvTReweight: histName = histName.replace(".root","_rwDvT.root")

            cmd = "hadd -f "+getOutDir()+"/dataRunII/"+histName+" "
            for y in years: cmd += getOutDir()+"/data"+y+"_3b_wJCM/"+histName+" "
            condor_jobs.append(makeCondorFile(cmd, "None", "dataRunII"+FvTName, outputDir=outputDir, filePrefix=jobName))            


        dag_config.append(condor_jobs)

    #
    #  Hadd SubSamples
    #
    condor_jobs = []

    histNameAll = "hists_wNN_"+mixedName+"_"+o.weightName+"_vAll_newSBDef.root"    
    if o.doDvTReweight: histNameAll = histNameAll.replace(".root","_rwDvT.root")

    cmdData3b    = "hadd -f "+getOutDir()+"/dataRunII/"+histNameAll+" "

    for s in subSamples:

        FvTName="_"+mixedName+"_v"+s
        histName = "hists_wNN"+FvTName+"_"+o.weightName+"_newSBDef.root"    
        if o.doDvTReweight: histName = histName.replace(".root","_rwDvT.root")

        cmdData3b    += getOutDir()+"/dataRunII/"+histName+" "


    condor_jobs.append(makeCondorFile(cmdData3b,    "None", "dataRunII_vAll",  outputDir=outputDir, filePrefix=jobName))            
    dag_config.append(condor_jobs)

    #
    #  Scale SubSample
    #
    condor_jobs = []

    cmdScale = "python ZZ4b/nTupleAnalysis/scripts/scaleFile.py --scaleFactor  "+str(1.0/len(subSamples))

    cmd = cmdScale + " -i "+getOutDir()+"/dataRunII/"+histNameAll+" "
    condor_jobs.append(makeCondorFile(cmd, getOutDir(), "dataRunII", outputDir=outputDir, filePrefix=jobName+"scale_"))            

    dag_config.append(condor_jobs)



    execute("rm "+outputDir+jobName+"All.dag", doRun)
    execute("rm "+outputDir+jobName+"All.dag.*", doRun)

    dag_file = makeDAGFile(jobName+"All.dag",dag_config, outputDir=outputDir)
    cmd = "condor_submit_dag "+dag_file
    execute(cmd, o.execute)






#
#  Make Plots with RW
#
if o.plotsWithRW:
    cmds = []

    #histDetailLevel = "passMDRs,passMjjOth,passSvB,fourTag,SB,CR,SRNoHH,HHSR,notSR"
    histDetailLevel = "passPreSel,fourTag,SB,SR"

    for y in ["RunII"]:

        #
        #  Nominal
        #
        FvTName = "_Nominal"
        data3bFile = getOutDir()+"/data"+y+"/hists_3b_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithRW_"+y+FvTName+"_"+o.weightName+"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)


        #
        #  Mixed Samples Combined
        #
        data4bFile  = getOutDir()+"/mixed"+y+"/hists_wRW_"+mixedName+"_"+o.weightName+"_vAll_newSBDef_scaled.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"
        data3bFile  = getOutDir()+"/data"+y+"/hists_wRW_"+mixedName+"_"+o.weightName+"_vAll_newSBDef_scaled.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
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

            data3bFile  = getOutDir()+"/data"+y+"/"+"hists_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"    
            data4bFile  = getOutDir()+"/mixed"+y+"_"+mixedName+"/"+"hists_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root"    
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wRW"+FvTName+"_"+o.weightName+"_newSBDef.root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithRW_"+o.weightName+"_"+y+FvTName+"_"+o.weightName +"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)


#            #
#            #
#            #
#            data3bFile  = getOutDir()+"/data"+y+"/"+"hists_wRW"+FvTName+"_"+o.weightName+".root"    
#            data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_"+weightName4b+"_vAll_scaled.root"
#            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+"_"+weightName4b+".root" 
#
#            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s + plotOpts[y]+" -m -j -r --noSignal "
#            cmd += " --histDetailLevel  "+histDetailLevel
#            cmd += " --data3b "+data3bFile
#            cmd += " --data "+data4bFile
#            cmd += " --TT "+ttbar4bFile
#            cmds.append(cmd)
#
    babySit(cmds, doRun)

    cmds = []

    for y in ["RunII"]:
        FvTName = "_Nominal"
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithRW_"+y+FvTName+"_"+o.weightName+"_newSBDef"+".tar plotsWithRW_"+y+FvTName+"_"+o.weightName+"_newSBDef")

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef"+".tar plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef")

        for s in subSamples:
            FvTName="_"+mixedName+"_v"+s

            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithRW_"+o.weightName+"_"+y+FvTName+"_"+o.weightName+"_newSBDef"+".tar plotsWithRW_"+o.weightName+"_"+y+FvTName+"_"+o.weightName+"_newSBDef")
#            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s+".tar plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s)
#


    
    babySit(cmds, doRun)    


#
#  Make Plots with RW
#
if o.plotsWithOT:
    cmds = []

    #histDetailLevel = "passMDRs,passMjjOth,passSvB,fourTag,SB,CR,SRNoHH,HHSR,notSR"
    histDetailLevel = o.histDetailStr 

    for y in ["RunII"]:

        #
        #  Nominal
        #
        FvTName = "_Nominal"
        data3bFile = getOutDir()+"/data"+y+"/hists_3b_wOT"+FvTName+"_"+o.weightName+"_newSBDef.root"
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wFvT"+FvTName+"_weights_newSBDef.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+"_weights_newSBDef.root"


        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithOT_"+y+FvTName+"_"+o.weightName+"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)


        #
        #  Mixed Samples Combined
        #
        data3bFile  = getOutDir()+"/data"+y+"/hists_wOT_"+mixedName+"_"+o.weightName+"_vAll_newSBDef_scaled.root"
        data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_weights_vAll_newSBDef_scaled.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+"_weights_newSBDef.root"


        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithOT_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
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

            data3bFile  = getOutDir()+"/data"+y+"/"+"hists_wOT"+FvTName+"_"+o.weightName+"_newSBDef.root"    
            data4bFile  = getOutDir()+"/mixed"+y+"_"+mixedName+"/"+"hists_wFvT"+FvTName+"_weights_newSBDef.root"    
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+"_weights_newSBDef.root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithOT_"+o.weightName+"_"+y+FvTName+"_"+o.weightName +"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)


    babySit(cmds, doRun)

    cmds = []

    for y in ["RunII"]:
        FvTName = "_Nominal"
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithOT_"+y+FvTName+"_"+o.weightName+"_newSBDef"+".tar plotsWithOT_"+y+FvTName+"_"+o.weightName+"_newSBDef")

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithOT_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef"+".tar plotsWithOT_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef")

        for s in subSamples:
            FvTName="_"+mixedName+"_v"+s

            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithOT_"+o.weightName+"_"+y+FvTName+"_"+o.weightName+"_newSBDef"+".tar plotsWithOT_"+o.weightName+"_"+y+FvTName+"_"+o.weightName+"_newSBDef")
#            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s+".tar plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s)
#


    
    babySit(cmds, doRun)    


#
#  Make Plots with RW
#
if o.plotsWithNN:
    cmds = []

    #histDetailLevel = "passMDRs,passMjjOth,passSvB,fourTag,SB,CR,SRNoHH,HHSR,notSR"
    histDetailLevel = "passPreSel,fourTag,SB,SR"

    for y in ["RunII"]:

        #
        #  Nominal
        #
        FvTName = "_Nominal"
        data3bFile = getOutDir()+"/data"+y+"/hists_3b_wNN"+FvTName+"_"+o.weightName+"_newSBDef.root"
        if o.doDvTReweight: data3bFile = data3bFile.replace(".root","_rwDvT.root")
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wRW"+FvTName+"_RW_newSBDef.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wRW"+FvTName+"_RW_newSBDef.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithNN_"+y+FvTName+"_"+o.weightName+"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
        if o.doDvTReweight:
            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithNN_"+y+FvTName+"_"+o.weightName+"_newSBDef_rwDvT" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)


        #
        #  Mixed Samples Combined
        #
        data3bFile  = getOutDir()+"/data"+y+"/hists_wNN_"+mixedName+"_"+o.weightName+"_vAll_newSBDef_scaled.root"
        if o.doDvTReweight: data3bFile = data3bFile.replace("_scaled.root","_rwDvT_scaled.root")
        data4bFile  = getOutDir()+"/mixed"+y+"/hists_wRW_"+mixedName+"_RW_vAll_newSBDef_scaled.root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wRW"+FvTName+"_RW_newSBDef.root"


        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithNN_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
        if o.doDvTReweight:
            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithNN_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef_rwDvT" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
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

            data3bFile  = getOutDir()+"/data"+y+"/"+"hists_wNN"+FvTName+"_"+o.weightName+"_newSBDef.root"    
            if o.doDvTReweight: data3bFile = data3bFile.replace(".root","_rwDvT.root")
            data4bFile  = getOutDir()+"/mixed"+y+"_"+mixedName+"/"+"hists_wRW"+FvTName+"_RW_newSBDef.root"    
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wRW"+FvTName+"_RW_newSBDef.root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithNN_"+o.weightName+"_"+y+FvTName+"_"+o.weightName +"_newSBDef" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
            if o.doDvTReweight:
                cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithNN_"+o.weightName+"_"+y+FvTName+"_"+o.weightName +"_newSBDef_rwDvT" + plotOpts[y]+" -m -j -r --noSignal --rMin 0.5 --rMax 1.5"
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)


    babySit(cmds, doRun)

    cmds = []

    for y in ["RunII"]:
        FvTName = "_Nominal"
        if o.doDvTReweight:

            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithNN_"+y+FvTName+"_"+o.weightName+"_newSBDef_rwDvT"+".tar plotsWithNN_"+y+FvTName+"_"+o.weightName+"_newSBDef_rwDvT")    
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithNN_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef_rwDvT"+".tar plotsWithNN_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef_rwDvT")
    
            for s in subSamples:
                FvTName="_"+mixedName+"_v"+s
    
                cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithNN_"+o.weightName+"_"+y+FvTName+"_"+o.weightName+"_newSBDef_rwDvT"+".tar plotsWithNN_"+o.weightName+"_"+y+FvTName+"_"+o.weightName+"_newSBDef_rwDvT")
    #            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s+".tar plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s)
    #
    

        else:
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithNN_"+y+FvTName+"_"+o.weightName+"_newSBDef"+".tar plotsWithNN_"+y+FvTName+"_"+o.weightName+"_newSBDef")
            
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithNN_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef"+".tar plotsWithNN_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_"+o.weightName+"_newSBDef")
            
            for s in subSamples:
                FvTName="_"+mixedName+"_v"+s
                
                cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithNN_"+o.weightName+"_"+y+FvTName+"_"+o.weightName+"_newSBDef"+".tar plotsWithNN_"+o.weightName+"_"+y+FvTName+"_"+o.weightName+"_newSBDef")
#            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s+".tar plotsWithRW_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s)
#


    
    babySit(cmds, doRun)    





#
#  Make Plots with FvT
#
if o.plotsNoFvT:
    cmds = []

    histDetailLevel = "passMDRs,passMjjOth,fourTag,SB,CR,SRNoHH,HHSR"
    #histDetailLevel = "passMDRs,fourTag,SB"

    for y in ["RunII"]:

        #
        #  Nominal
        #
        FvTName = "_Nominal"
        
        qcdFile  = getOutDir()+"/QCD"+y+"/hists_3b_noFvT"+FvTName+".root"
        data4bFile  = getOutDir()+"/data"+y+"/hists_4b_wFvT"+FvTName+"_"+o.weightName+".root"
        ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_wFvT"+FvTName+"_"+o.weightName+".root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+o.weightName+"_"+y+FvTName+plotOpts[y]+" -m -j --noSignal "
        cmd += " --histDetailLevel  "+histDetailLevel
        cmd += " --qcd "+qcdFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)


        #
        #  Nominal Overllaaid with Mixed
        #
        mixedFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_"+o.weightName+"_vAll_scaled.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+o.weightName+"_Nominal_vs_"+mixedName+plotOpts[y]+"  -m -j --noSignal "
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
        data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_"+o.weightName+"_vAll_scaled.root"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+o.weightName+"_"+y+"_vAll_"+mixedName + plotOpts[y]+" -m -j --noSignal "
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
            histName = "hists_wFvT"+FvTName+"_"+o.weightName+".root"    

            data4bFile  = getOutDir()+"/mixed"+y+"_"+mixedName+"/"+histName
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+"_"+o.weightName+".root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+o.weightName+"_"+y+FvTName + plotOpts[y]+" -m -j  --noSignal "
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --qcd "+qcdFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)


            #
            #
            #
            data4bFile  = getOutDir()+"/mixed"+y+"/hists_wFvT_"+mixedName+"_"+o.weightName+"_vAll_scaled.root"
            ttbar4bFile = getOutDir()+"/TT"+y+"/hists_4b_noPSData_wFvT"+FvTName+"_"+o.weightName+".root" 

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsNoFvT_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s + plotOpts[y]+" -m -j  --noSignal "
            cmd += " --histDetailLevel  "+histDetailLevel
            cmd += " --qcd "+qcdFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)

    babySit(cmds, doRun)

    cmds = []

    for y in ["RunII"]:
        FvTName = "_Nominal"
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+o.weightName+"_"+y+FvTName+".tar plotsNoFvT_"+o.weightName+"_"+y+FvTName)

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+o.weightName+"_Nominal_vs_"+mixedName+".tar plotsNoFvT_"+o.weightName+"_Nominal_vs_"+mixedName)

        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+o.weightName+"_"+y+"_vAll_"+mixedName+".tar plotsNoFvT_"+o.weightName+"_"+y+"_vAll_"+mixedName)




        for s in subSamples:
            FvTName="_"+mixedName+"_v"+s

            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+o.weightName+"_"+y+FvTName+".tar plotsNoFvT_"+o.weightName+"_"+y+FvTName)
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsNoFvT_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s+".tar plotsNoFvT_"+o.weightName+"_"+y+"_vAll_"+mixedName+"_vs_v"+s)



    
    babySit(cmds, doRun)    





if o.makeInputsForCombine:

    import ROOT

    def getHistForCombine(in_File,tag,proc,outName,region, doMA=False):
        if doMA:
            hist = in_File.Get("passMDRs/"+tag+"/mainView/"+region+"/SvB_MA_ps_"+proc).Clone()
        else:
            hist = in_File.Get("passMDRs/"+tag+"/mainView/"+region+"/SvB_ps_"+proc).Clone()

        hist.SetName(outName)
        return hist


    def makeInputsForRegion(region, noFvT=False, doMA=False):
        
        #noFvT = False

        if noFvT:
            outFile = ROOT.TFile(outputDir+"/hists_closure_"+mixedName+"_"+region+"_noFvT.root","RECREATE")
        elif doMA:
            outFile = ROOT.TFile(outputDir+"/hists_closure_"+mixedName+"_"+region+"_"+o.weightName+"_MA.root","RECREATE")
        else:
            outFile = ROOT.TFile(outputDir+"/hists_closure_"+mixedName+"_"+region+"_"+o.weightName+".root","RECREATE")

        procs = ["zz","zh","hh"]
        
        for s in subSamples: 
            
            #
            #  "+tagID+" with combined JCM 
            #
            #weightPostFix = "_comb"
            #tagName = "_"+tagID
            FvTName="_"+mixedName+"_v"+s
            histName = "hists_wFvT"+FvTName+"_"+o.weightName+".root"    
            
            histName3b = "hists_wFvT"+FvTName+"_"+o.weightName+".root"    
            histName4b = "hists_wFvT"+FvTName+"_"+o.weightName+".root"    
            histName4bTT = "hists_4b_noPSData_wFvT"+FvTName+"_"+o.weightName+".root" 
            
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
    
                print "Reading ",getOutDir()+"/data"+y+"_3b_wJCM/"+histName3b
                print "Reading ",getOutDir()+"/mixed"+y+"_"+mixedName+"_wJCM_v"+s+"/"+histName4b
                print "Reading ",getOutDir()+"/TT"+y+"/"+histName4bTT
                multiJet_Files .append(ROOT.TFile.Open(getOutDir()+"/data"+y+"_3b_wJCM/"+histName3b))
                data_obs_Files .append(ROOT.TFile.Open(getOutDir()+"/mixed"+y+"_"+mixedName+"_wJCM_v"+s+"/"+histName4b))
                ttbar_Files    .append(ROOT.TFile.Open(getOutDir()+"/TT"+y+"/"+histName4bTT))
        
                for p in procs:
        
                    multiJet_Hists[p].append( getHistForCombine(multiJet_Files[-1],"threeTag",p,"multijet", region, doMA) )
                    data_obs_Hists[p].append( getHistForCombine(data_obs_Files[-1],"fourTag",p, "data_obs", region, doMA) )
                    ttbar_Hists[p]   .append( getHistForCombine(ttbar_Files[-1],   "fourTag",p, "ttbar",    region, doMA) )
    
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
        histName3b = "hists_wFvT"+FvTName+"_"+o.weightName+"_oneFit.root"    

        multiJet_Files = []
        multiJet_Hists = {}

        for p in procs:
            multiJet_Hists[p] = []

        for y in years:

            multiJet_Files .append(ROOT.TFile.Open(getOutDir()+"/data"+y+"_3b_wJCM/"+histName3b))
    
            for p in procs:
    
                multiJet_Hists[p].append( getHistForCombine(multiJet_Files[-1],"threeTag",p,"multijet", region, doMA) )

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
    makeInputsForRegion("notSR")
    makeInputsForRegion("SRNoHH")
    makeInputsForRegion("CR")
    makeInputsForRegion("SB")

    makeInputsForRegion("SR",    doMA=True)
    makeInputsForRegion("notSR", doMA=True)
    makeInputsForRegion("SRNoHH",doMA=True)
    makeInputsForRegion("CR",    doMA=True)
    makeInputsForRegion("SB",    doMA=True)

#    makeInputsForRegion("SR",    noFvT=True)
#    makeInputsForRegion("notSR", noFvT=True)
#    makeInputsForRegion("SRNoHH",noFvT=True)
#    makeInputsForRegion("CR",    noFvT=True)
#    makeInputsForRegion("SB",    noFvT=True)



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
            inputWeights   = " --friends "+outputDir+"/fileLists/data"+y+"_"+tag+"_wJCM_friends_Nominal.txt"

            cmd = runCMD + inputFile + inputWeights + outDir + noPico  +  yearOpts[y] + " --histFile "+histName + histDetail + " --jcmNameLoad "+JCMName+" --FvTName  FvT"+FvTName
            cmd += " --runKlBdt "
            condor_jobs.append(makeCondorFile(cmd, "None", "data"+y+"_"+tag+FvTName, outputDir=outputDir, filePrefix=jobName))
            

            for tt in ttbarSamplesByYear[y]:
                inputFile = " -i "+outputDir+"/fileLists/"+tt+"_"+tag+"_wTrigW_wJCM.txt "
                inputWeights   = " --friends "+outputDir+"/fileLists/"+tt+"_"+tag+"_wTrigW_wJCM_friends_Nominal.txt"                

                cmd = runCMD + inputFile + inputWeights + outDir + noPico  + MCyearOpts(tt) + " --histFile " + histName + histDetail  + " --jcmNameLoad "+JCMName+ "  --FvTName FvT"+FvTName + " --doTrigEmulation "
                cmd += " --runKlBdt "
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
            for tt in ttbarSamplesByYear[y]: cmd += getOutDir()+"/"+tt+"_"+tag+"_wTrigW_wJCM/"+histName+" "
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



