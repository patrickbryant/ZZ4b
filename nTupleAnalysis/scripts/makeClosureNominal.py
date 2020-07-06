
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4", help="Year or comma separated list of subsamples")
parser.add_option('-w',            action="store_true", dest="doWeights",      default=False, help="Fit jetCombinatoricModel and nJetClassifier TSpline")
parser.add_option('--histsForJCM',  action="store_true",      help="Make hist.root for JCM")
parser.add_option('--histsWithJCM', action="store_true",      help="Make hist.root with JCM")
parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--plotsWithJCM', action="store_true",      help="Make pdfs with JCM")
parser.add_option('--email',            default=None,      help="")

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

ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]

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

plotOpts = {}
plotOpts["2018"]=" -l 60.0e3 -y 2018"
plotOpts["2017"]=" -l 36.7e3 -y 2017"
plotOpts["2016"]=" -l 35.9e3 -y 2016"
plotOpts["RunII"]=" -l 132.6e3 -y RunII"

# 
#  Make Hists for JCM Calc
#
if o.histsForJCM: 

    #
    #  Make Hists
    #
    cmds = []
    logs = []

    histName = "hists_b0p6.root "
    histOut = " --histFile "+histName
    
    for y in years:
        picoOut = " -p picoAOD_b0p6.root "
        h10 = " --histogramming 10 --histDetail 7 "    

        cmds.append(runCMD+"  -i "+outputDir+"/fileLists/data"+y+".txt "+picoOut+" -o "+outputDir+" "+ yearOpts[y] + h10 + histOut )
        logs.append(outputDir+"/log_data"+y+"_b0p6")

        #
        #  Make Hists for ttbar
        #
        for tt in ttbarSamples:
            cmds.append(runCMD+" -i "+outputDir+"/fileLists/"+tt+y+"_noMjj.txt "+ picoOut +" -o "+outputDir+ MCyearOpts[y] + h10 + histOut )
            logs.append(outputDir+"/log_"+tt+y+"_b0p6")

    babySit(cmds, doRun, logFiles=logs)

    #
    #  Hadd ttbar
    #
    cmds = [] 
    logs = []
    
    histName = "hists_b0p6.root " 

    for y in years:
        mkdir(outputDir+"/TT"+y, doRun)
        
        cmd = "hadd -f "+outputDir+"/TT"+y+"/"+histName
        for tt in ttbarSamples:        
            cmd += outputDir+"/"+tt+y+"_noMjj/"+histName

        cmds.append(cmd)
        logs.append(outputDir+"/log_HaddTT"+y+"_b0p6")

    babySit(cmds, doRun, logFiles=logs)


#
#  Make the JCM-weights
#

#
#  Fit JCM
#
if o.doWeights:
    
    cmds = []
    logs = []

    for y in years:

        histName = "hists_b0p6.root " 
        
        cmd = weightCMD+" -d "+outputDir+"/data"+y+"/"+histName
        cmd += " --tt "+outputDir+"/TT"+y+"/"+histName
        cmd += " -c passMDRs   -o "+outputDir+"/weights/data"+y+"_"+"_/  -r SB -w 00-00-04 "+plotOpts[y]
        
        cmds.append(cmd)
        logs.append(outputDir+"/log_makeWeights_"+y+"_v"+s)
    
    babySit(cmds, doRun, logFiles=logs)


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
if o.histsWithJCM: 

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
        mkdir(outputDir+"/QCD"+y, doRun)

        cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
        cmd += " -d "+outputDirComb+"/data"+y+"/"+histName
        cmd += " -tt "+outputDirComb+"/TT"+y+"/"+histName
        cmd += " -q "+outputDir+"/QCD"+y+"/"+histName
        cmds.append(cmd)
        
    babySit(cmds, doRun)    



#
#  Make CutFlows
#
if o.plotsWithJCM:
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
#  Training  done seperately
#



#
#  Make Hists with JCM and FvT weights applied
#
if o.histsWithFvT: 

    #
    #  Make Hists
    #
    cmds = []
    logs = []

    JCMName="Nominal"
    FvTName="_Nominal"

    histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
    histName4b = "hists_4b_wFVT"+FvTName+".root "

    for y in years:

        pico3b = "picoAOD_3b_wJCM.root"
        picoOut = " -p NONE "
        h10 = " --histogramming 10 --histDetail 7 "    
        histOut3b = " --histFile "+histName3b

        cmds.append(runCMD+" -i "+outputDirComb+"/data"+y+"/"+pico3b+             picoOut  +   yearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    
        cmds.append(runCMD+" -i "+outputDirComb+"/TTToHadronic"+y+"/"+pico3b+     picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    
        cmds.append(runCMD+" -i "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+pico3b+ picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    
        cmds.append(runCMD+" -i "+outputDirComb+"/TTTo2L2Nu"+y+"/"+pico3b+        picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)
        
        logs.append(outputDir+"/log_"+y+"_3b_wJCM_wFVT")
        logs.append(outputDir+"/log_TTHad"+y+"_3b_wJCM_wFVT")
        logs.append(outputDir+"/log_TTSem"+y+"_3b_wJCM_wFVT")
        logs.append(outputDir+"/log_TT2L2Nu"+y+"_3b_wJCM_wFVT")

        pico4b = "picoAOD_4b.root"
        histOut4b = " --histFile "+histName4b

        cmds.append(runCMD+" -i "+outputDir+"/data"+y+"/"+pico4b+             picoOut  +   yearOpts[y]+ h10 + histOut4b + " -r --FvTName "+FvTName)    
        cmds.append(runCMD+" -i "+outputDir+"/TTToHadronic"+y+"/"+pico4b+     picoOut  + MCyearOpts[y]+ h10 + histOut4b + " -r --FvTName "+FvTName)    
        cmds.append(runCMD+" -i "+outputDir+"/TTToSemiLeptonic"+y+"/"+pico4b+ picoOut  + MCyearOpts[y]+ h10 + histOut4b + " -r --FvTName "+FvTName)    
        cmds.append(runCMD+" -i "+outputDir+"/TTTo2L2Nu"+y+"/"+pico4b+        picoOut  + MCyearOpts[y]+ h10 + histOut4b + " -r --FvTName "+FvTName)

        logs.append(outputDir+"/log_"+y+"_4b_wFVT")
        logs.append(outputDir+"/log_TTHad"+y+"_4b_wFVT")
        logs.append(outputDir+"/log_TTSem"+y+"_4b_wFVT")
        logs.append(outputDir+"/log_TT2L2Nu"+y+"_4b_wFVT")

        
    babySit(cmds, doRun, logFiles=logs)

    
    #
    #  Hadd TTbar
    #
    cmds = []
    logs = []
    for y in years:
        cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName3b+" "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b+"  "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b+" "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b)
        logs.append(outputDir+"/log_haddTT_3b_wJCM_wFvT_"+y)

        cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName4b+" "+outputDir+"/TTToHadronic"+y+"/"+histName4b+"  "+outputDir+"/TTToSemiLeptonic"+y+"/"+histName4b+" "+outputDir+"/TTTo2L2Nu"+y+"/"+histName4b)
        logs.append(outputDir+"/log_haddTT_4b_wFvT_"+y)

    babySit(cmds, doRun, logFiles=logs)


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/TTRunII",   doRun)

        cmds = []
        logs = []
        
        cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName3b+" "+outputDirComb+"/data2016/"+histName3b+" "+outputDirComb+"/data2017/"+histName3b+" "+outputDirComb+"/data2018/"+histName3b)
        cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName4b+" "+outputDir+"/data2016/"+histName4b+" "+outputDir+"/data2017/"+histName4b+" "+outputDir+"/data2018/"+histName4b)
        cmds.append("hadd -f "+outputDir+"/TTRunII/"  +histName4b+" "+outputDir+"/TT2016/"  +histName4b+" "+outputDir+"/TT2017/"  +histName4b+" "+outputDir+"/TT2018/"  +histName4b)
        cmds.append("hadd -f "+outputDir+"/TTRunII/"  +histName3b+" "+outputDir+"/TT2016/"  +histName3b+" "+outputDir+"/TT2017/"  +histName3b+" "+outputDir+"/TT2018/"  +histName3b)

        logs.append(outputDir+"/log_haddDataRunII_3b")
        logs.append(outputDir+"/log_haddDataRunII_4b")
        logs.append(outputDir+"/log_haddDataRunII_TT")
        logs.append(outputDir+"/log_haddDataRunII_TT_3b")

        babySit(cmds, doRun, logFiles=logs)


    if o.email: execute('echo "Subject: [makeClosureNominal] histsWithFvT Done" | sendmail johnalison@cmu.edu',doRun)




#
#  Make CutFlows
#
if o.plotsWithFvT:
    cmds = []
    logs = []
    
    yearsToPlot = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToPlot.append("RunII")

    for y in yearsToPlot:
            
        JCMName="Nominal"
        FvTName="_Nominal"
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        histName4b = "hists_4b_wFVT"+FvTName+".root "

        data3bFile  = outputDirComb+"/data"+y+"/"+histName3b    if not y == "RunII" else outputDir+"/data"+y+"/"+histName3b               
        data4bFile  = outputDir+"/data"+y+"/"+histName4b
        ttbar4bFile = outputDir+"/TT"+y+"/"+histName4b
        ttbar3bFile = outputDir+"/TT"+y+"/"+histName3b

        
        #cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
        #cmd += " --d4 "+outputDir+"/data"+y+"/"+histName4b
        #cmd += " --d3 "+outputDirComb+"/data"+y+"/"+histName3b
        #cmd += " --t4 "+outputDir+"/TT"+y+"/"+histName4b
        #cmd += " --t3 "+outputDir+"/TT"+y+"/"+histName3b
        #cmd += " --t4_s "+outputDir+"/TTToSemiLeptonic"+y+"/"+histName4b
        #cmd += " --t4_h "+outputDir+"/TTToHadronic"+y+"/"+histName4b
        #cmd += " --t4_d "+outputDir+"/TTTo2L2Nu"+y+"/"+histName4b
        #cmd += " --t3_s "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b
        #cmd += " --t3_h "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b
        #cmd += " --t3_d "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b
        #cmd += " --name "+outputDir+"/CutFlow_wFvT_"+y
        #cmd += " --makePDF -r"
        #cmds.append(cmd)
        #logs.append(outputDir+"/log_cutFlow_wFVT_"+y)

        cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
        cmd += " --d4 "+data4bFile
        cmd += " --d3 "+data3bFile
        cmd += " --t4 "+ttbar4bFile
        cmd += " --t3 "+ttbar3bFile
        cmd += " --name "+outputDir+"/CutFlow_wFvT_"+y
        cmd += " --makePDF -r"
        cmds.append(cmd)
        logs.append(outputDir+"/log_cutFlow_wFVT_"+y)


        cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y +plotOpts[y]+" -m -j -r --noSignal "
        cmd += " --data3b "+data3bFile
        cmd += " --data "+data4bFile
        cmd += " --TT "+ttbar4bFile
        cmds.append(cmd)
        logs.append(outputDir+"/log_makePlots_wFVT_"+y)


    babySit(cmds, doRun, logFiles=logs)    

    cmds = []
    for y in years:
        cmds.append("mv CutFlow_wFvT_"+y+".pdf "+outputDir+"/")
        cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+".tar plotsWithFvT_"+y)
        
    babySit(cmds, doRun)    









#python ZZ4b/nTupleAnalysis/scripts/subtractTT.py -d   ${outputDir}/data2018AllEvents/data18/hists_3bTo4b_noWeights.root  --tt ${outputPath}/${outputDir}/TT2018/hists_noWeights.root -q   ${outputPath}/${outputDir}/qcd2018/hists_noWeights.root

#hadd -f ${outputDir}/TT2018/hists_3b_wJCM_${JCMNAME}.root ${outputDir}/TTToHadronic2018/hists_3b_wJCM_${JCMNAME}.root  ${outputDir}/TTToSemiLeptonic2018/hists_3b_wJCM_${JCMNAME}.root  ${outputDir}/TTTo2L2Nu2018/hists_3b_wJCM_${JCMNAME}.root 
#hadd -f ${outputDirNom}/TT2017/hists_4b.root ${outputDirNom}/TTToHadronic2017/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2017/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2017/hists_4b.root 
#hadd -f ${outputDirNom}/TT2016/hists_4b.root ${outputDirNom}/TTToHadronic2016/hists_4b.root  ${outputDirNom}/TTToSemiLeptonic2016/hists_4b.root  ${outputDirNom}/TTTo2L2Nu2016/hists_4b.root 


#done





