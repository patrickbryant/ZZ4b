
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
#parser.add_option('--histsWithJCM', action="store_true",      help="Make hist.root with JCM")
#parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
#parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
#parser.add_option('--plotsWithJCM', action="store_true",      help="Make pdfs with JCM")
parser.add_option('--email',            default=None,      help="")

o, a = parser.parse_args()


doRun = o.execute

years = o.year.split(",")
subSamples = o.subSamples.split(",")
ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]

outputDir="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/mixed"
outputDirNom="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal"

# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'

#data2018_noMjj=outputDirNom+"/fileLists/data2018.txt"
#data2017_noMjj=outputDirNom+"/fileLists/data2017.txt"
#data2016_noMjj=outputDirNom+"/fileLists/data2016.txt"
#
#
#ttHad2018_noMjj=outputDirNom+"/fileLists/TTToHadronic2018_noMjj.txt"
#ttSem2018_noMjj=outputDirNom+"/fileLists/TTToSemiLeptonic2018_noMjj.txt"
#tt2LN2018_noMjj=outputDirNom+"/fileLists/TTTo2L2Nu2018_noMjj.txt"
#
#ttHad2017_noMjj=outputDirNom+"/fileLists/TTToHadronic2017_noMjj.txt"
#ttSem2017_noMjj=outputDirNom+"/fileLists/TTToSemiLeptonic2017_noMjj.txt"
#tt2LN2017_noMjj=outputDirNom+"/fileLists/TTTo2L2Nu2017_noMjj.txt"
#
#ttHad2016_noMjj=outputDirNom+"/fileLists/TTToHadronic2016_noMjj.txt"
#ttSem2016_noMjj=outputDirNom+"/fileLists/TTToSemiLeptonic2016_noMjj.txt"
#tt2LN2016_noMjj=outputDirNom+"/fileLists/TTTo2L2Nu2016_noMjj.txt"


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

#
#  Make the JCM-weights at PS-level (Needed for making the 3b sample)
#
if o.doWeights:

    cmds = []
    logs = []

    # histName = "hists.root"
    histName = "hists_b0p6.root " 

    for y in years:

        cmds.append(weightCMD+" -d "+outputDirNom+"/data"+y+"_b0p6/"+histName+" -c passPreSel   -o "+outputDir+"/weights/noTT_data"+y+"_b0p6_PreSel/  -r SB -w 00-00-03")
        logs.append(outputDir+"/log_fitJCM_b0p6_PS_"+y)


    babySit(cmds, doRun, logFiles=logs)

jcmFileList = {}
jcmFileList["2018"] = outputDir+"/weights/noTT_data2018_b0p6_PreSel/jetCombinatoricModel_SB_00-00-03.txt"
jcmFileList["2017"] = outputDir+"/weights/noTT_data2017_b0p6_PreSel/jetCombinatoricModel_SB_00-00-03.txt"
jcmFileList["2016"] = outputDir+"/weights/noTT_data2016_b0p6_PreSel/jetCombinatoricModel_SB_00-00-03.txt"


# 
#  Make the 3b sample with the stats of the 4b sample
#
if o.subSample3b:
    # In the following "3b" refers to 3b subsampled to have the 4b statistics

    cmds = []
    logs = []

    # histName = "hists.root"
    histName = "hists_b0p6.root " 


    for s in subSamples:

        for y in years:
            
            picoOut = " -p picoAOD_3bSubSampled_b0p6_v"+s+".root "
            h10     = " --histogramming 10 "
            histOut = " --histFile hists_b0p6_v"+s+".root"

            cmds.append(runCMD+" -i "+outputDirNom+"/fileLists/data"+y+"_b0p6.txt"+ picoOut + " -o "+outputDir+ yearOpts[y]+  h10+  histOut + " -j "+jcmFileList[y]+" --emulate4bFrom3b --emulationOffset "+s+" --noDiJetMassCutInPicoAOD ")
            logs.append(outputDir+"/log_dataOnlyAll_make3b_"+y+"_b0p6_v"+s)

            for tt in ttbarSamples:
                cmds.append(runCMD+" -i "+outputDirNom+"/fileLists/"+tt+y+"_noMjj_b0p6.txt" + picoOut + " -o "+outputDir + MCyearOpts[y] +h10 + histOut + " -j "+jcmFileList[y]+" --emulate4bFrom3b --emulationOffset "+s+" --noDiJetMassCutInPicoAOD")
                logs.append(outputDir+"/log_"+tt+y+"_v"+s)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeInputMixSamples] subSample3b  Done" | sendmail '+o.email,doRun)



#
#  Make hists of the subdampled data
#    #(Optional: for sanity check
if o.histSubSample3b:

    cmds = []
    logs = []

    for s in subSamples:

        for y in years:
            
            picoIn  = "picoAOD_3bSubSampled_b0p6_v"+s+".root "
            picoOut = "  -p 'None' "
            h10     = " --histogramming 10 "
            histOut = " --histFile hists_3bSubSampled_b0p6_v"+s+".root "

            cmds.append(runCMD+" -i "+outputDir+"/data"+y+"_b0p6/"+picoIn + picoOut +" -o "+outputDir+ yearOpts[y] + h10 + histOut + " --is3bMixed  --writeEventTextFile ")
            logs.append(outputDir+"/log_subSampledHists_b0p6_"+y+"_v"+s)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeInputMixSamples] histSubSample3b  Done" | sendmail '+o.email,doRun)


 
#
# Copy subsamples to EOS
#
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
                copy("closureTests/mixed/"+subDir+"_b0p6/picoAOD_3bSubSampled_b0p6_v"+s+".root", subDir,"picoAOD_3bSubSampled_b0p6_v"+s+".root")


if o.cleanPicoAODs:
    
    def rm(fileName):
        cmd  = "rm  "+fileName
    
        if doRun: os.system(cmd)
        else:     print cmd

    for y in years:
        for tt in ["data","TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"]:
            subDir = tt+y if tt == "data" else tt+y+"_noMjj"
            for s in subSamples:
                rm("closureTests/mixed/"+subDir+"_b0p6/picoAOD_3bSubSampled_b0p6_v"+s+".root")

#
#   Make inputs fileLists
#
#if o.makeInputFileLists:
#
#    def run(cmd):
#        if doRun: os.system(cmd)
#        else:     print cmd
#
#
#    eosDir = "root://cmseos.fnal.gov//store/user/johnda/closureTest/mixed/"  #+subDir+"/"+outFileName
#
#    for y in years:
#        fileList = outputDir+"/fileLists/data"+y+"_b0p6.txt"    
#        run("rm "+fileList)
#
#
#        for y in years:
#            for tt in ["data","TTTo2L2Nu","TTToHadronic","TTToSemiLeptonic"]:
#                subDir = tt+y if tt == "data" else tt+y+"_noMjj"
#                for s in subSamples:
#                    rm("echo "+closureTests/mixed/"+subDir+"_b0p6/picoAOD_3bSubSampled_b0p6_v"+s+".root")
#        for p in dataPeriods[y]:
#            run("echo "+eosDir+"/data"+y+"/picoAOD_noDiJetMjj_b0p6_"+y+p+".root >> "+fileList)
#
#
#        for tt in ttbarSamples:
#            fileList = outputDir+"/fileLists/"+tt+y+"_noMjj_b0p6.txt"    
#            run("rm "+fileList)
#
#            run("echo "+eosDir+"/"+tt+y+"/picoAOD_noDiJetMjj_b0p6.root >> "+fileList)




#
# Make Hemisphere library from all hemispheres
#
if o.make4bHemis:
    
    cmds = []
    logs = []

    picoOut = "  -p 'None' "
    h1     = " --histogramming 1 "
    histOut = " --histFile hists_b0p6.root " 

    for y in years:
        
        cmds.append(runCMD+" -i "+outputDirNom+"/fileLists/data"+y+"_b0p6.txt"+ picoOut + " -o "+outputDir+"/dataHemis_b0p6"+ yearOpts[y]+  h1 +  histOut + " --createHemisphereLibrary")
        logs.append(outputDir+"/log_makeHemisData"+y+"_b0p6")
    


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeInputMixSamples] make4bHemis  Done" | sendmail '+o.email,doRun)
