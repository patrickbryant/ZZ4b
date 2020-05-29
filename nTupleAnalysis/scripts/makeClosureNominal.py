makeHistsWithFvT=True
makeCutFlowsWithFvT=True
makeHistsWithJCM=True
makeCutFlowsWithJCM=True

import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
o, a = parser.parse_args()

doRun = o.execute

#
# In the following "3b" refers to 3b subsampled to have the 4b statistics
#
outputDir="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal"
outputDirComb="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/combined"


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'


years = ["2018","2017","2016"]

yearOpts = {}
yearOpts["2018"]=' -y 2018 --bTag 0.2770 '
yearOpts["2017"]=' -y 2017 --bTag 0.3033 '
yearOpts["2016"]=' -y 2016 --bTag 0.3093 '

MCyearOpts = {}
MCyearOpts["2018"]=yearOpts["2018"]+' --bTagSF -l 60.0e3 --isMC '
MCyearOpts["2017"]=yearOpts["2017"]+' --bTagSF -l 36.7e3 --isMC '
MCyearOpts["2016"]=yearOpts["2016"]+' --bTagSF -l 35.9e3 --isMC '

plotOpts = {}
plotOpts["2018"]=" -l 60.0e3 -y 2018"
plotOpts["2017"]=" -l 36.7e3 -y 2017"
plotOpts["2016"]=" -l 35.9e3 -y 2016"

### 
###  Make Hists with all data
###
##$runCMD  -i ${outputDir}/fileLists/data2018.txt -p picoAOD.root  -o ${outputDir} $YEAR2018  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2018 &
##$runCMD  -i ${outputDir}/fileLists/data2017.txt -p picoAOD.root  -o ${outputDir} $YEAR2017  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2017 &
##$runCMD  -i ${outputDir}/fileLists/data2016.txt -p picoAOD.root  -o ${outputDir} $YEAR2016  --histogramming 10 --histFile hists.root   2>&1|tee ${outputDir}/log_data2016 &
##
##
###
###  Make Hists with all ttbar
###
###
### 2018
##$runCMD -i ${outputDir}/fileLists/TTToHadronic2018_noMjj.txt     -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTHad2018   & 
##$runCMD -i ${outputDir}/fileLists/TTToSemiLeptonic2018_noMjj.txt -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTSemi2018  &
##$runCMD -i ${outputDir}/fileLists/TTTo2L2Nu2018_noMjj.txt        -o ${outputDir} $YEAR2018MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TT2L2Nu2018 &
##
### 2017
##$runCMD -i ${outputDir}/fileLists/TTToHadronic2017_noMjj.txt     -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTHad2017   & 
##$runCMD -i ${outputDir}/fileLists/TTToSemiLeptonic2017_noMjj.txt -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTSemi2017  &
##$runCMD -i ${outputDir}/fileLists/TTTo2L2Nu2017_noMjj.txt        -o ${outputDir} $YEAR2017MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TT2L2Nu2017 &
##
### 2016
##$runCMD -i ${outputDir}/fileLists/TTToHadronic2016_noMjj.txt     -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTHad2016   & 
##$runCMD -i ${outputDir}/fileLists/TTToSemiLeptonic2016_noMjj.txt -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TTSemi2016  &
##$runCMD -i ${outputDir}/fileLists/TTTo2L2Nu2016_noMjj.txt        -o ${outputDir} $YEAR2016MC --histogramming 10  --histFile hists.root 2>&1 |tee ${outputDir}/log_TT2L2Nu2016 &


#
#  Hadd ttbar
#
#mkdir ${outputDir}/TT2018
#mkdir ${outputDir}/TT2017
#mkdir ${outputDir}/TT2016
#hadd -f ${outputDir}/TT2018/hists.root ${outputDir}/TTToHadronic2018_noMjj/hists.root ${outputDir}/TTToSemiLeptonic2018_noMjj/hists.root ${outputDir}/TTTo2L2Nu2018_noMjj/hists.root & 
#hadd -f ${outputDir}/TT2017/hists.root ${outputDir}/TTToHadronic2017_noMjj/hists.root ${outputDir}/TTToSemiLeptonic2017_noMjj/hists.root ${outputDir}/TTTo2L2Nu2017_noMjj/hists.root & 
#hadd -f ${outputDir}/TT2016/hists.root ${outputDir}/TTToHadronic2016_noMjj/hists.root ${outputDir}/TTToSemiLeptonic2016_noMjj/hists.root ${outputDir}/TTTo2L2Nu2016_noMjj/hists.root &



#
#  Make the JCM-weights
#
# (3b -> 4b)
#$weightCMD -d ${outputDir}/data2018/hists.root  --tt ${outputDir}/TT2018/hists.root -c passXWt  -o ${outputDir}/weights/data2018/  -r SB -w 00-00-02 -y 2018 -l 60.0e3 2>&1 |tee ${outputDir}/log_JCM2018 
#$weightCMD -d ${outputDir}/data2017/hists.root  --tt ${outputDir}/TT2017/hists.root -c passXWt  -o ${outputDir}/weights/data2017/  -r SB -w 00-00-02 -y 2017 -l 36.7e3 2>&1 |tee ${outputDir}/log_JCM2017 
#$weightCMD -d ${outputDir}/data2016/hists.root  --tt ${outputDir}/TT2016/hists.root -c passXWt  -o ${outputDir}/weights/data2016/  -r SB -w 00-00-02 -y 2016 -l 35.9e3 2>&1 |tee ${outputDir}/log_JCM2016 


#
#  Adding JCM weights now done in makeClosureTestCombined
#



#
#  Make 3b Hists with JCM weights applied
#
if makeHistsWithJCM: 

    #
    #  Make Hists
    #
    cmds = []
    logs = []

    JCMName="Nominal"
    histName = "hists_3b_wJCM_"+JCMName+".root "
    histOut = " --histFile "+histName

    for y in years:

        pico3b = "picoAOD_3b_wJCM.root"
        picoOut = " -p NONE "
        h10 = " --histogramming 10 --histDetail 7 "    

        cmds.append(runCMD+" -i "+outputDirComb+"/data"+y+"/"+pico3b+             picoOut  +   yearOpts[y]+ h10 + histOut + " --jcmNameLoad "+JCMName)    #outputDir+"/log_"+y+"_wJCM_wFVT
        cmds.append(runCMD+" -i "+outputDirComb+"/TTToHadronic"+y+"/"+pico3b+     picoOut  + MCyearOpts[y]+ h10 + histOut + " --jcmNameLoad "+JCMName)    #outputDir+"/log_TTHad"+y+"_wJCM_wFVT
        cmds.append(runCMD+" -i "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+pico3b+ picoOut  + MCyearOpts[y]+ h10 + histOut + " --jcmNameLoad "+JCMName)    #outputDir+"/log_TTSem"+y+"_wJCM_wFVT
        cmds.append(runCMD+" -i "+outputDirComb+"/TTTo2L2Nu"+y+"/"+pico3b+        picoOut  + MCyearOpts[y]+ h10 + histOut + " --jcmNameLoad "+JCMName)    #outputDir+"/log_TT2L2Nu"+y+"_wJCM_wFVT
        
    babySit(cmds, doRun)

    
    #
    #  Hadd TTbar
    #
    cmds = []

    for y in years:
        cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName+" "+outputDirComb+"/TTToHadronic"+y+"/"+histName+"  "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName+" "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName)

    babySit(cmds, doRun)


    #
    # Subtract QCD 
    #
    cmds = []
    for y in years:
        mkdir(outputDir+"/QCD"+y, o.execute)


        cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
        cmd += " -d "+outputDirComb+"/data"+y+"/"+histName
        cmd += " -tt "+outputDirComb+"/TT"+y+"/"+histName
        cmd += " -q "+outputDir+"/QCD"+y+"/"+histName

    babySit(cmds, doRun)    



#
#  Make CutFlows
#
if makeCutFlowsWithJCM:
    cmds = []
    for y in years:
            
        histName4b="hists_4b.root" 

        JCMName="Nominal"
        histName3b = "hists_3b_wJCM_"+JCMName+".root "

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
        cmd += " --d4 "+outputDir+"/data"+y+"/"+histName4b
        cmd += " --d3 "+outputDirComb+"/data"+y+"/"+histName3b
        cmd += " --t4 "+outputDir+"/TT"+y+"/"+histName4b
        cmd += " --t3 "+outputDirComb+"/TT"+y+"/"+histName3b
        cmd += " --t4_s "+outputDir+"/TTToSemiLeptonic"+y+"/"+histName4b
        cmd += " --t4_h "+outputDir+"/TTToHadronic"+y+"/"+histName4b
        cmd += " --t4_d "+outputDir+"/TTTo2L2Nu"+y+"/"+histName4b
        cmd += " --t3_s "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b
        cmd += " --t3_h "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b
        cmd += " --t3_d "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b
        cmd += " --name "+outputDir+"/CutFlow_wJCM_"+y
        cmd += " --makePDF"
        cmds.append(cmd)

        #
        # MAke Plots
        #
        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithJCM_"+y+ plotOpts[y]+" -m -j  --noSignal "
        cmd += " --qcd "+outputDir+"/QCD"+y+"/"+histName3b    
        cmd += " --data "+outputDir+"/data"+y+"/"+histName4b
        cmd += " --TT "+outputDir+"/TT"+y+"/"+histName4b
        cmds.append(cmd)

    babySit(cmds, doRun)    

    cmds = []
    for y in years:
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithJCM_"+y+".tar plotsWithJCM"+y)
    babySit(cmds, doRun)    







#
#  Make 3b Hists with JCM and FvT weights applied
#
if makeHistsWithFvT: 

    #
    #  Make Hists
    #
    cmds = []
    logs = []

    JCMName="Nominal"
    FvTName="_Nominal"
    histName = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
    histOut = " --histFile "+histName

    for y in years:

        pico3b = "picoAOD_3b_wJCM.root"
        picoOut = " -p NONE "
        h10 = " --histogramming 10 --histDetail 7 "    

        cmds.append(runCMD+" -i "+outputDirComb+"/data"+y+"/"+pico3b+             picoOut  +   yearOpts[y]+ h10 + histOut + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    #outputDir+"/log_"+y+"_wJCM_wFVT
        cmds.append(runCMD+" -i "+outputDirComb+"/TTToHadronic"+y+"/"+pico3b+     picoOut  + MCyearOpts[y]+ h10 + histOut + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    #outputDir+"/log_TTHad"+y+"_wJCM_wFVT
        cmds.append(runCMD+" -i "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+pico3b+ picoOut  + MCyearOpts[y]+ h10 + histOut + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    #outputDir+"/log_TTSem"+y+"_wJCM_wFVT
        cmds.append(runCMD+" -i "+outputDirComb+"/TTTo2L2Nu"+y+"/"+pico3b+        picoOut  + MCyearOpts[y]+ h10 + histOut + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    #outputDir+"/log_TT2L2Nu"+y+"_wJCM_wFVT
        
    babySit(cmds, doRun)

    
    #
    #  Hadd TTbar
    #
    cmds = []

    for y in years:
        cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName+" "+outputDirComb+"/TTToHadronic"+y+"/"+histName+"  "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName+" "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName)

    babySit(cmds, doRun)


#
#  Make CutFlows
#
if makeCutFlowsWithFvT:
    cmds = []
    for y in years:
            
        histName4b="hists_4b.root" 

        JCMName="Nominal"
        FvTName="_Nominal"
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
        cmd += " --d4 "+outputDir+"/data"+y+"/"+histName4b
        cmd += " --d3 "+outputDirComb+"/data"+y+"/"+histName3b
        cmd += " --t4 "+outputDir+"/TT"+y+"/"+histName4b
        cmd += " --t3 "+outputDirComb+"/TT"+y+"/"+histName3b
        cmd += " --t4_s "+outputDir+"/TTToSemiLeptonic"+y+"/"+histName4b
        cmd += " --t4_h "+outputDir+"/TTToHadronic"+y+"/"+histName4b
        cmd += " --t4_d "+outputDir+"/TTTo2L2Nu"+y+"/"+histName4b
        cmd += " --t3_s "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b
        cmd += " --t3_h "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b
        cmd += " --t3_d "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b
        cmd += " --name "+outputDir+"/CutFlow_wFvT_"+y
        cmd += " --makePDF -r"
        cmds.append(cmd)

        #
        # MAke Plots
        #
        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y +plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --data3b "+outputDirComb+"/data"+y+"/"+histName3b    
        cmd += " --data "+outputDir+"/data"+y+"/"+histName4b
        cmd += " --TT "+outputDir+"/TT"+y+"/"+histName4b
        cmds.append(cmd)

    babySit(cmds, doRun)    

    cmds = []
    for y in years:
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+".tar plotsWithFvT_"+y)
        
    babySit(cmds, doRun)    









#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputDir}/data2018AllEvents/data18/hists_3bTo4b_noWeights.root  --tt ${outputPath}/${outputDir}/TT2018/hists_noWeights.root -q   ${outputPath}/${outputDir}/qcd2018/hists_noWeights.root

#hadd -f ${outputDir}/TT2018/hists_3b_wJCM_${JCMNAME}.root ${outputDir}/TTToHadronic2018/hists_3b_wJCM_${JCMNAME}.root  ${outputDir}/TTToSemiLeptonic2018/hists_3b_wJCM_${JCMNAME}.root  ${outputDir}/TTTo2L2Nu2018/hists_3b_wJCM_${JCMNAME}.root 
#hadd -f ${outputDirNom}/TT2017/hists_4b.root ${outputDirNom}/TTToHadronic2017/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2017/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2017/hists_4b.root 
#hadd -f ${outputDirNom}/TT2016/hists_4b.root ${outputDirNom}/TTToHadronic2016/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2016/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2016/hists_4b.root 


#done





#
#  OLD
#
#####################################3





#
# MAke Plots
#
# python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o HemiClosureTest -p plots -l 60.0e3 -y 2018 -m -j -r

