
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('--makeSkims',  action="store_true",      help="Make input skims")
parser.add_option('--copyToEOS',  action="store_true",      help="Copy to EOS")
parser.add_option('--cleanPicoAODs',  action="store_true",      help="rm local picoAODs")
parser.add_option('--makeInputFileLists',  action="store_true",      help="make Input file lists")
parser.add_option('--noTT',       action="store_true",      help="Skip TTbar")
parser.add_option('--email',            default=None,      help="")

o, a = parser.parse_args()

doRun = o.execute

#
# In the following "3b" refers to 3b subsampled to have the 4b statistics
#
outputDir="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal"

# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'

ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]

if o.noTT:
    ttbarSamples = []

years = o.year.split(",")

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


dataPeriods = {}
# All
#dataPeriods["2018"] = ["A","B","C","D"]
#dataPeriods["2017"] = ["B","C","D","E","F"]
#dataPeriods["2016"] = ["B","C","D","E","F","G","H"]
# for skimming 
dataPeriods["2018"] = []
dataPeriods["2017"] = []
dataPeriods["2016"] = ["B"]


#
# Make skims with out the di-jet Mass cuts
#
if o.makeSkims:

    cmds = []
    logs = []

    for y in years:
        
        histConfig = " --histogramming 0 --histDetailLevel 1 --histFile histsFromNanoAOD.root "
        picoOut = " -p picoAOD_noDiJetMjj_b0p6.root "

        #
        #  Data
        #
        for p in dataPeriods[y]:
            cmds.append(runCMD+"  -i ZZ4b/fileLists/data"+y+p+".txt -o "+outputDir+  yearOpts[y] + histConfig + picoOut + " --fastSkim  --noDiJetMassCutInPicoAOD ")
            logs.append(outputDir+"/log_skim_b0p6_"+y+p)

        #
        #  TTbar
        # 
        for tt in ttbarSamples:
            cmds.append(runCMD+" -i ZZ4b/fileLists/"+tt+y+".txt -o"+outputDir+  MCyearOpts[y] + histConfig + picoOut +" --fastSkim --noDiJetMassCutInPicoAOD ")
            logs.append(outputDir+"/log_skim_b0p6_"+tt+y)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [make3bMix4bClosure] mixInputs  Done" | sendmail '+o.email,doRun)






if o.copyToEOS:

    def copy(fileName, subDir, outFileName):
        cmd  = "xrdcp  "+fileName+" root://cmseos.fnal.gov//store/user/johnda/closureTest/skims/"+subDir+"/"+outFileName
    
        if doRun:
            os.system(cmd)
        else:
            print cmd
    
    for y in years:
    
        for p in dataPeriods[y]:
            copy("closureTests/nominal/data"+y+p+"/picoAOD_noDiJetMjj_b0p6.root", "data"+y, "picoAOD_noDiJetMjj_b0p6_"+y+p+".root")

        for tt in ttbarSamples:
            copy("closureTests/nominal/"+tt+y+"/picoAOD_noDiJetMjj_b0p6.root", tt+y, "picoAOD_noDiJetMjj_b0p6.root")


if o.cleanPicoAODs:
    
    def rm(fileName):
        cmd  = "rm  "+fileName
    
        if doRun: os.system(cmd)
        else:     print cmd


    for y in years:
    
        for p in dataPeriods[y]:
            rm("closureTests/nominal/data"+y+p+"/picoAOD_noDiJetMjj_b0p6.root")

        for tt in ttbarSamples:
            rm("closureTests/nominal/"+tt+y+"/picoAOD_noDiJetMjj_b0p6.root")




#
#   Make inputs fileLists
#
if o.makeInputFileLists:

    def run(cmd):
        if doRun: os.system(cmd)
        else:     print cmd


    eosDir = "root://cmseos.fnal.gov//store/user/johnda/closureTest/skims/"    

    for y in years:
        fileList = outputDir+"/fileLists/data"+y+"_b0p6.txt"    
        run("rm "+fileList)

        for p in dataPeriods[y]:
            run("echo "+eosDir+"/data"+y+"/picoAOD_noDiJetMjj_b0p6_"+y+p+".root >> "+fileList)


        for tt in ttbarSamples:
            fileList = outputDir+"/fileLists/"+tt+y+"_noMjj_b0p6.txt"    
            run("rm "+fileList)

            run("echo "+eosDir+"/"+tt+y+"/picoAOD_noDiJetMjj_b0p6.root >> "+fileList)

    
        




