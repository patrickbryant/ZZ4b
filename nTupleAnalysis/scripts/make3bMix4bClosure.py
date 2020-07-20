import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6", help="Year or comma separated list of subsamples")
parser.add_option('-d',            action="store_true", dest="doData",         default=False, help="Run data")
parser.add_option('-t',            action="store_true", dest="doTT",       default=False, help="Run ttbar MC")
parser.add_option('-w',            action="store_true", dest="doWeights",      default=False, help="Fit jetCombinatoricModel and nJetClassifier TSpline")
parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--mixInputs',    action="store_true",      help="Make Mixed Samples")
parser.add_option('--plotUniqueHemis',    action="store_true",      help="Do Some Mixed event analysis")
parser.add_option('--histsWithJCM', action="store_true",      help="Make hist.root with JCM")
parser.add_option('--plotsWithJCM', action="store_true",      help="Make pdfs with JCM")
parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
parser.add_option('--cutFlowBeforeJCM', action="store_true",      help="Make 4b cut flow before JCM")
parser.add_option('--email',            default=None,      help="")



o, a = parser.parse_args()

doRun = o.execute
years = o.year.split(",")
subSamples = o.subSamples.split(",")

outputDir="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b"
outputDirNom="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/nominal"
outputDirMix="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/mixed"
outputDirComb="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/combined"

# mixed
mixedName=o.mixedName


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'
mixedAnalysisCMD='mixedEventAnalysis ZZ4b/nTupleAnalysis/scripts/mixedEventAnalysis_cfg.py'


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




#
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
if o.mixInputs:

    cmds = []
    logs = []

    for s in subSamples:

        for y in years:

            picoOut    = " -p picoAOD_"+mixedName+"_b0p6_v"+s+".root "
            h10        = " --histogramming 10 "
            histOut    = " --histFile hists_"+mixedName+"_b0p6_v"+s+".root "
            hemiLoad   = ' --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "'+outputDirMix+'/dataHemis_b0p6/data'+y+'_b0p6/hemiSphereLib_4TagEvents_*root"'

            #
            #  Data
            #
            if o.doData:
                inFileList = outputDirMix+"/fileLists/data"+y+"_b0p6_v"+s+".txt"
                
                # The --is3bMixed here just turns off blinding of the data
                cmds.append(runCMD+" -i "+inFileList+" -o "+outputDir + picoOut + yearOpts[y] + h10 + histOut+" --is3bMixed "+hemiLoad)
                logs.append(outputDir+"/logMix_"+mixedName+"_"+y+"_b0p6_v"+s)
    
            if o.doTT:
                for tt in ttbarSamples:
                    fileListTT = outputDirMix+"/fileLists/"+tt+y+"_b0p6_v"+s+".txt"
    
                    cmds.append(runCMD+" -i "+fileListTT +" -o "+outputDir+ picoOut + MCyearOpts[y] + h10  + histOut + " --is3bMixed " + hemiLoad)
                    logs.append(outputDir+"/log_"+tt+y+"_"+mixedName+"_b0p6_v"+s)
    

    babySit(cmds, doRun, logFiles=logs)


    #
    #  Hadd TTbar
    #
    if o.doTT:
        cmds = [] 
        logs = []

        for s in subSamples:

            for y in years:

                histName = "hists_"+mixedName+"_b0p6_v"+s+".root " 

                cmd = "hadd -f "+outputDir+"/TT"+y+"/"+histName

                for tt in ttbarSamples:
                    cmd += outputDir+"/"+tt+y+"_b0p6_v"+s+"/"+histName
                cmds.append(cmd)
                logs.append(outputDir+"/log_HaddTT"+y+"_v"+s)
    babySit(cmds, doRun, logFiles=logs)


    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/TTRunII",   doRun)

        cmds = []
        logs = []
        
        for s in subSamples:
    
            histName = "hists_"+mixedName+"_b0p6_v"+s+".root " 
    
            cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName+" "+outputDir+"/data2016_b0p6_v"+s+"/"+histName+" "+outputDir+"/data2017_b0p6_v"+s+"/"+histName+" "+outputDir+"/data2018_b0p6_v"+s+"/"+histName)
            logs.append(outputDir+"/log_haddDataRunII_mixed_b0p6_v"+s)

            cmds.append("hadd -f "+outputDir+"/TTRunII/"  +histName+" "+outputDir+"/TT2016/"  +histName+" "+outputDir+"/TT2017/"  +histName+" "+outputDir+"/TT2018/"  +histName)
            logs.append(outputDir+"/log_haddTTRunII_mixed_b0p6_v"+s)

        babySit(cmds, doRun, logFiles=logs)

    if o.email: execute('echo "Subject: [make3bMix4bClosure] mixInputs  Done" | sendmail '+o.email,doRun)


#
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
if o.plotUniqueHemis:

    cmds = []
    logs = []

    for y in years:

        histOut = " --hist hMixedAnalysis_b0p6.root "
        cmds.append(mixedAnalysisCMD + " -i "+outputDir+"/fileLists/data"+y+"_"+mixedName+"_b0p6.txt -o "+outputDir + histOut)
        logs.append(outputDir+"/log_mixAnalysis_data"+y+"_"+mixedName+"_b0p6")
            
        for s in subSamples:

            cmds.append(mixedAnalysisCMD + " -i "+outputDir+"/data"+y+"_b0p6_v"+s+"/picoAOD_"+mixedName+"_b0p6_v"+s+".root -o "+outputDir+"/data"+y+"_b0p6_v"+s+  histOut)
            logs.append(outputDir+"/log_mixAnalysis_"+y+"_"+mixedName+"_b0p6_v"+s)            

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

        histName = "hists_"+mixedName+"_b0p6_v"+s+".root " 
        histName3b = "hists_b0p6.root"
        for y in years:
    
            #
            # MAke Plots
            #
            data4bFile  = outputDir+"/data"+y+"_b0p6_v"+s+"/"+histName if not y == "RunII" else outputDir+"/data"+y+"/"+histName
            data3bFile  = outputDir+"/data"+y+"_b0p6_v"+s+"/"+histName  if not y == "RunII" else outputDir+"/data"+y+"/"+histName
            ttbar4bFile = outputDir+"/TT"+y+"/"+histName
            ttbar3bFile = outputDir+"/TT"+y+"/"+histName

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
            cmd += " --d4 "+data4bFile
            cmd += " --d3 "+data3bFile
            cmd += " --t4 "+ttbar4bFile
            cmd += " --t3 "+ttbar3bFile
            cmd += " --name "+outputDir+"/CutFlow_4tagOnly_"+y+"_"+mixedName+"_b0p6_v"+s
            cmd += " --makePDF "
            cmds.append(cmd)
            logs.append(outputDir+"/log_cutFlow_"+y+"_v"+s)

    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    for s in subSamples:
        for y in years:
            cmds.append("mv CutFlow_4tagOnly_"+y+"_"+mixedName+"_b0p6_v"+s+".pdf "+outputDir+"/")
            
    babySit(cmds, doRun)    




#
#  Fit JCM
#
if o.doWeights:
    
    cmds = []
    logs = []

    yearsToFit = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToFit.append("RunII")

    for s in subSamples:
        
        for y in yearsToFit:


            #histName4b = "hists_"+mixedName+"_v"+s+".root " 
            #histName3b = "hists.root "

            histName4b = "hists_"+mixedName+"_b0p6_v"+s+".root "             
            histName3b = "hists_b0p6.root "

            data3bFile  = outputDirNom+"/data"+y+"_b0p6/"+histName3b     if not y == "RunII" else outputDirNom+"/data"+y+"/"+histName3b               
            data4bFile  = outputDir+"/data"+y+"_b0p6_v"+s+"/"+histName4b if not y == "RunII" else outputDir+"/data"+y+"/"+histName4b                
            ttbar4bFile = outputDir+"/TT"+y+"/"+histName4b
            ttbar3bFile = outputDirNom+"/TT"+y+"/"+histName3b
            
            cmd = weightCMD
            cmd += " -d "+data3bFile
            cmd += " --data4b "+data4bFile
            cmd += " --tt "+ttbar3bFile
            cmd += " --tt4b "+ttbar4bFile
            cmd += " -c passMDRs   -o "+outputDir+"/weights/data"+y+"_"+mixedName+"_b0p6_v"+s+"/  -r SB -w 00-00-07 "+plotOpts[y]
            
            cmds.append(cmd)
            logs.append(outputDir+"/log_makeWeights_"+y+"_b0p6_v"+s)

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

    #
    #  Make Hists
    #
    cmds = []
    logs = []

    for s in subSamples:

        JCMName=mixedName+"_v"+s
        FvTName="_"+mixedName+"_v"+s
    
        #histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        #histName4b = "hists_4b_wFVT"+FvTName+".root "
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_b0p6.root "
        histName4b = "hists_4b_wFVT"+FvTName+"_b0p6.root "
    
        for y in years:
    
            #
            # 3b
            #
            #pico3b = "picoAOD_3b_wJCM.root"
            pico3b = "picoAOD_3b_wJCM_b0p6.root"
            picoOut = " -p NONE "
            h10 = " --histogramming 10 --histDetail 7 "    
            histOut3b = " --histFile "+histName3b
    
            cmds.append(runCMD+" -i "+outputDirComb+"/data"+y+"_b0p6/"+pico3b+             picoOut  +   yearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    
            logs.append(outputDir+"/log_"+y+"_3b_wJCM_wFVT_b0p6_v"+s)

            # 3b TTbar not needed... Run it anyway for cut flow
            for tt in ttbarSamples:
                cmds.append(runCMD+" -i "+outputDirComb+"/"+tt+y+"_b0p6/"+pico3b+     picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    
                logs.append(outputDir+"/log_"+tt+y+"_3b_wJCM_wFVT_b0p6_v"+s)
    

            #
            # 4b
            #
            pico4b = "picoAOD_"+mixedName+"_4b_b0p6_v"+s+".root"
            histOut4b = " --histFile "+histName4b
    
            cmds.append(runCMD+" -i "+outputDir+"/data"+y+"_b0p6_v"+s+"/"+pico4b+             picoOut  +   yearOpts[y]+ h10 + histOut4b + "  --FvTName "+FvTName + " --is3bMixed")    
            logs.append(outputDir+"/log_"+y+"_4b_wFVT_b0p6_v"+s)

            for tt in ttbarSamples:
                cmds.append(runCMD+" -i "+outputDir+"/"+tt+y+"_b0p6_v"+s+"/"+pico4b+     picoOut  + MCyearOpts[y]+ h10 + histOut4b + "  --FvTName "+FvTName + " --is3bMixed")    
                logs.append(outputDir+"/log_"+tt+y+"_4b_wFVT_b0p6_v"+s)

        
    babySit(cmds, doRun, logFiles=logs)

    
    #
    #  Hadd TTbar
    #
    cmds = []
    logs = []
    for s in subSamples:

        JCMName=mixedName+"_v"+s
        FvTName="_"+mixedName+"_v"+s
    
        #histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        #histName4b = "hists_4b_wFVT"+FvTName+".root "
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_b0p6.root "
        histName4b = "hists_4b_wFVT"+FvTName+"_b0p6.root "


        for y in years:
            cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName3b+" "+outputDirComb+"/TTToHadronic"+y+"_b0p6/"+histName3b+"  "+outputDirComb+"/TTToSemiLeptonic"+y+"_b0p6/"+histName3b+" "+outputDirComb+"/TTTo2L2Nu"+y+"_b0p6/"+histName3b)
            logs.append(outputDir+"/log_haddTT_3b_wJCM_wFvT_"+y+"_b0p6_v"+s)
    
            cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName4b+" "+outputDir+"/TTToHadronic"+y+"_b0p6_v"+s+"/"+histName4b+"  "+outputDir+"/TTToSemiLeptonic"+y+"_b0p6_v"+s+"/"+histName4b+" "+outputDir+"/TTTo2L2Nu"+y+"_b0p6_v"+s+"/"+histName4b)
            logs.append(outputDir+"/log_haddTT_4b_wFvT_"+y+"_b0p6_v"+s)

    babySit(cmds, doRun, logFiles=logs)

    #
    #   Hadd years
    #
    if "2016" in years and "2017" in years and "2018" in years:
    
        mkdir(outputDir+"/dataRunII", doRun)
        mkdir(outputDir+"/TTRunII",   doRun)

        cmds = []
        logs = []
        
        for s in subSamples:
    
            JCMName=mixedName+"_v"+s
            FvTName="_"+mixedName+"_v"+s
        
            # histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
            # histName4b = "hists_4b_wFVT"+FvTName+".root "
            histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_b0p6.root "
            histName4b = "hists_4b_wFVT"+FvTName+"_b0p6.root "
    
    
            cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName3b+" "+outputDirComb+"/data2016_b0p6/"+histName3b+" "+outputDirComb+"/data2017_b0p6/"+histName3b+" "+outputDirComb+"/data2018_b0p6/"+histName3b)
            cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName4b+" "+outputDir+"/data2016_b0p6_v"+s+"/"+histName4b+" "+outputDir+"/data2017_b0p6_v"+s+"/"+histName4b+" "+outputDir+"/data2018_b0p6_v"+s+"/"+histName4b)
            cmds.append("hadd -f "+outputDir+"/TTRunII/"  +histName4b+" "+outputDir+"/TT2016/"  +histName4b+" "+outputDir+"/TT2017/"  +histName4b+" "+outputDir+"/TT2018/"  +histName4b)
            cmds.append("hadd -f "+outputDir+"/TTRunII/"  +histName3b+" "+outputDir+"/TT2016/"  +histName3b+" "+outputDir+"/TT2017/"  +histName3b+" "+outputDir+"/TT2018/"  +histName3b)

            logs.append(outputDir+"/log_haddDataRunII_3b_b0p6_v"+s)
            logs.append(outputDir+"/log_haddDataRunII_4b_b0p6_v"+s)
            logs.append(outputDir+"/log_haddDataRunII_TT_b0p6_v"+s)
            logs.append(outputDir+"/log_haddDataRunII_TT_3b_b0p6_v"+s)

        babySit(cmds, doRun, logFiles=logs)


    if o.email: execute('echo "Subject: [make3bMix4bClosure] makeHistsWithFvT Done" | sendmail '+o.email,doRun)


#
#  Make Plots with FvT
#
if o.plotsWithFvT:
    cmds = []
    logs = []

    yearsToPlot = years
    if "2016" in years and "2017" in years and "2018" in years:
        yearsToPlot.append("RunII")

    for s in subSamples:

        JCMName=mixedName+"_v"+s
        FvTName="_"+mixedName+"_v"+s

        #histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        #histName4b = "hists_4b_wFVT"+FvTName+".root "
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+"_b0p6.root "
        histName4b = "hists_4b_wFVT"+FvTName+"_b0p6.root "

        for y in yearsToPlot:
    
            #
            # MAke Plots
            #
            data3bFile  = outputDirComb+"/data"+y+"_b0p6/"+histName3b    if not y == "RunII" else outputDir+"/data"+y+"/"+histName3b               
            data4bFile  = outputDir+"/data"+y+"_b0p6_v"+s+"/"+histName4b if not y == "RunII" else outputDir+"/data"+y+"/"+histName4b                
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
            cmd += " --name "+outputDir+"/CutFlow_wFvT_"+y+FvTName+"_b0p6"
            cmd += " --makePDF -r"
            cmds.append(cmd)
            logs.append(outputDir+"/log_cutFlow_wFVT_"+y+FvTName+"_b0p6")

    

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+FvTName+"_b0p6"+plotOpts[y]+" -m -j -r --noSignal "
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)
            logs.append(outputDir+"/log_makePlots_wFVT_"+y+FvTName+"_b0p6")
    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    for s in subSamples:
        FvTName="_"+mixedName+"_v"+s
        for y in years:
            cmds.append("mv CutFlow_wFvT_"+y+FvTName+"_b0p6.pdf "+outputDir+"/")
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+FvTName+"_b0p6.tar plotsWithFvT_"+y+FvTName+"_b0p6")
            
    babySit(cmds, doRun)    



        
