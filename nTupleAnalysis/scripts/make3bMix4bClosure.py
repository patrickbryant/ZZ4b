

import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4", help="Year or comma separated list of subsamples")
parser.add_option('--histsWithJCM', action="store_true",      help="Make hist.root with JCM")
parser.add_option('--histsWithFvT', action="store_true",      help="Make hist.root with FvT")
parser.add_option('--plotsWithFvT', action="store_true",      help="Make pdfs with FvT")
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
mixedName="3bMix4b"


# Helpers
runCMD='nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
weightCMD='python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'

data2018_3bSubSampled=outputDirMix+"/fileLists/data2018_3bSubSampled.txt"
data2017_3bSubSampled=outputDirMix+"/fileLists/data2017_3bSubSampled.txt"
data2016_3bSubSampled=outputDirMix+"/fileLists/data2016_3bSubSampled.txt"




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
plotOpts["RunII"]=" -l 132.6e3 -y RunII"




#
#  Mix "3b" with 4b hemis to make "3bMix4b" evnets
#
#for i in 0 1 2 3 4 
#do
#    #$runCMD -i ${outputDirMix}/fileLists/data2018_v${i}.txt -o ${outputDir}  -p picoAOD_${mixedName}_noTTVeto_v${i}.root  $YEAR2018  --histogramming 10 --histFile hists_${mixedName}_noTTVeto_v${i}.root   --is3bMixed --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2018/hemiSphereLib_4TagEvents_*root" | tee ${outputDir}/logMix_${mixedName}_2018_v${i}  & 
#    #$runCMD -i ${outputDirMix}/fileLists/data2017_v${i}.txt -o ${outputDir}   -p picoAOD_${mixedName}_noTTVeto_v${i}.root $YEAR2017  --histogramming 10 --histFile hists_${mixedName}_noTTVeto_v${i}.root   --is3bMixed --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2017/hemiSphereLib_4TagEvents_*root" | tee ${outputDir}/logMix_${mixedName}_2017_v${i}  & 
#    #$runCMD -i ${outputDirMix}/fileLists/data2016_v${i}.txt -o ${outputDir}   -p picoAOD_${mixedName}_noTTVeto_v${i}.root $YEAR2016  --histogramming 10 --histFile hists_${mixedName}_noTTVeto_v${i}.root   --is3bMixed --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2016/hemiSphereLib_4TagEvents_*root" | tee ${outputDir}/logMix_${mixedName}_2016_v${i}  & 
#done


#
#  Mix "4b" ttbar
#
#for i in 0 1 2 3 4 
#do
#    $runCMD -i ${outputDirMix}/fileLists/TTToSemiLeptonic2018_v${i}.txt -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2018MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2018/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToSemiLeptonic2018_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/fileLists/TTToHadronic2018_v${i}.txt     -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2018MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2018/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToHadronic2018_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/fileLists/TTTo2L2Nu2018_v${i}.txt        -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2018MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2018/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTTo2L2Nu2018_${mixedName}_v${i} &

#    $runCMD -i ${outputDirMix}/fileLists/TTToSemiLeptonic2017_v${i}.txt -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2017MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2017/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToSemiLeptonic2017_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/fileLists/TTToHadronic2017_v${i}.txt     -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2017MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2017/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToHadronic2017_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/fileLists/TTTo2L2Nu2017_v${i}.txt        -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2017MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2017/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTTo2L2Nu2017_${mixedName}_v${i} &

#    $runCMD -i ${outputDirMix}/fileLists/TTToSemiLeptonic2016_v${i}.txt -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2016MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2016/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToSemiLeptonic2016_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/fileLists/TTToHadronic2016_v${i}.txt     -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2016MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2016/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTToHadronic2016_${mixedName}_v${i} &
#    $runCMD -i ${outputDirMix}/fileLists/TTTo2L2Nu2016_v${i}.txt        -o ${outputDir} -p picoAOD_${mixedName}_v${i}.root $YEAR2016MC --histogramming 10  --histFile hists_${mixedName}_v${i}.root  --loadHemisphereLibrary --maxNHemis 1000000 --inputHLib3Tag "NONE" --inputHLib4Tag "${outputDirMix}/dataHemis/data2016/hemiSphereLib_4TagEvents_*root" 2>&1  |tee ${outputDir}/log_TTTo2L2Nu2016_${mixedName}_v${i} &
#done


#for i in 0 1 2 3 4 
#do
#    #hadd -f ${outputDir}/TT2018/hists_${mixedName}_v${i}.root ${outputDir}/TTToHadronic2018_v${i}/hists_${mixedName}_v${i}.root ${outputDir}/TTToSemiLeptonic2018_v${i}/hists_${mixedName}_v${i}.root ${outputDir}/TTTo2L2Nu2018_v${i}/hists_${mixedName}_v${i}.root &
#    #hadd -f ${outputDir}/TT2017/hists_${mixedName}_v${i}.root ${outputDir}/TTToHadronic2017_v${i}/hists_${mixedName}_v${i}.root ${outputDir}/TTToSemiLeptonic2017_v${i}/hists_${mixedName}_v${i}.root ${outputDir}/TTTo2L2Nu2017_v${i}/hists_${mixedName}_v${i}.root &
#    #hadd -f ${outputDir}/TT2016/hists_${mixedName}_v${i}.root ${outputDir}/TTToHadronic2016_v${i}/hists_${mixedName}_v${i}.root ${outputDir}/TTToSemiLeptonic2016_v${i}/hists_${mixedName}_v${i}.root ${outputDir}/TTTo2L2Nu2016_v${i}/hists_${mixedName}_v${i}.root &
#done



#
#  Fit JCM
#
#for i in 0 1 2 3 4 
#do
#    $weightCMD -d ${outputDirNom}/data2018/hists.root  --data4b ${outputDir}/data2018_v${i}/hists_${mixedName}_noTTVeto_v${i}.root  --tt ${outputDirNom}/TT2018/hists.root  --tt4b ${outputDir}/TT2018/hists_${mixedName}_v${i}.root  -c passXWt   -o ${outputDir}/weights/data2018_${mixedName}_v${i}/  -r SB -w 00-00-02 -y 2018 -l 60.0e3  2>&1 |tee ${outputDir}/log_makeWeights_2018_v${i}
#    $weightCMD -d ${outputDirNom}/data2017/hists.root  --data4b ${outputDir}/data2017_v${i}/hists_${mixedName}_noTTVeto_v${i}.root  --tt ${outputDirNom}/TT2017/hists.root  --tt4b ${outputDir}/TT2017/hists_${mixedName}_v${i}.root  -c passXWt   -o ${outputDir}/weights/data2017_${mixedName}_v${i}/  -r SB -w 00-00-02 -y 2017 -l 36.7e3 2>&1 |tee ${outputDir}/log_makeWeights_2017_v${i}
#    $weightCMD -d ${outputDirNom}/data2016/hists.root  --data4b ${outputDir}/data2016_v${i}/hists_${mixedName}_noTTVeto_v${i}.root  --tt ${outputDirNom}/TT2016/hists.root  --tt4b ${outputDir}/TT2016/hists_${mixedName}_v${i}.root  -c passXWt   -o ${outputDir}/weights/data2016_${mixedName}_v${i}/  -r SB -w 00-00-02 -y 2016 -l 35.9e3 2>&1 |tee ${outputDir}/log_makeWeights_2016_v${i}
#done





#
#  Adding JCM weights now done in makeClosureTestCombined
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

        JCMName="3bMix4b_v"+s
    
        histName3b = "hists_3b_wJCM_"+JCMName+".root "
    
        for y in years:
    
            #
            # 3b
            #
            pico3b = "picoAOD_3b_wJCM.root"
            picoOut = " -p NONE "
            h10 = " --histogramming 10 --histDetail 7 "    
            histOut3b = " --histFile "+histName3b
    
            cmds.append(runCMD+" -i "+outputDirComb+"/data"+y+"/"+pico3b+             picoOut  +   yearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName)    
            cmds.append(runCMD+" -i "+outputDirComb+"/TTToHadronic"+y+"/"+pico3b+     picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName)    
            cmds.append(runCMD+" -i "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+pico3b+ picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName)    
            cmds.append(runCMD+" -i "+outputDirComb+"/TTTo2L2Nu"+y+"/"+pico3b+        picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName)
            
            logs.append(outputDir+"/log_"+y+"_3b_wJCM_v"+s)
            logs.append(outputDir+"/log_TTHad"+y+"_3b_wJCM_v"+s)
            logs.append(outputDir+"/log_TTSem"+y+"_3b_wJCM_v"+s)
            logs.append(outputDir+"/log_TT2L2Nu"+y+"_3b_wJCM_v"+s)
    

    babySit(cmds, doRun, logFiles=logs)



    
    #
    #  Hadd TTbar
    #
    cmds = []
    logs = []
    for s in subSamples:

        JCMName="3bMix4b_v"+s
        histName3b = "hists_3b_wJCM_"+JCMName+".root "

        for y in years:
            cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName3b+" "+outputDirComb+"/TTToHadronic"+y+"_v"+s+"/"+histName3b+"  "+outputDirComb+"/TTToSemiLeptonic"+y+"_v"+s+"/"+histName3b+" "+outputDirComb+"_v"+s+"/TTTo2L2Nu"+y+"/"+histName3b)
            logs.append(outputDir+"/log_haddTT_3b_wJCM_"+y+"_v"+s)
    
    babySit(cmds, doRun, logFiles=logs)



    #
    # Subtract QCD 
    #
    cmds = []
    logs = []
    for s in subSamples:

        JCMName="3bMix4b_v"+s
        histName3b = "hists_3b_wJCM_"+JCMName+".root "

        for y in years:
            mkdir(outputDir+"/QCD"+y+"_v"+s, doRun)
    
            cmd = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py "
            cmd += " -d "+outputDirComb+"/data"+y+"/"+histName3b
            cmd += " -tt "+outputDirComb+"/TT"+y+"/"+histName3b
            cmd += " -q "+outputDir+"/QCD"+y+"/"+histName3b
            cmds.append(cmd)
            
            logs.append(outputDir+"/log_SubTT"+y+"_v"+s)
    
    babySit(cmds, doRun, logFiles=logs)    


    if o.email: execute('echo "Subject: [make3bMix4bClosure] makeHistsWithJCM Done" | sendmail johnalison@cmu.edu',doRun)


            
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

        JCMName="3bMix4b_v"+s
        FvTName="_3bMix4b_v"+s
    
        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        histName4b = "hists_4b_wFVT"+FvTName+".root "
    
        for y in years:
    
            #
            # 3b
            #
            pico3b = "picoAOD_3b_wJCM.root"
            picoOut = " -p NONE "
            h10 = " --histogramming 10 --histDetail 7 "    
            histOut3b = " --histFile "+histName3b
    
            cmds.append(runCMD+" -i "+outputDirComb+"/data"+y+"/"+pico3b+             picoOut  +   yearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    

            # 3b TTbar not needed
            #cmds.append(runCMD+" -i "+outputDirComb+"/TTToHadronic"+y+"/"+pico3b+     picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    
            #cmds.append(runCMD+" -i "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+pico3b+ picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)    
            #cmds.append(runCMD+" -i "+outputDirComb+"/TTTo2L2Nu"+y+"/"+pico3b+        picoOut  + MCyearOpts[y]+ h10 + histOut3b + " --jcmNameLoad "+JCMName+ " -r --FvTName "+FvTName)
            
            logs.append(outputDir+"/log_"+y+"_3b_wJCM_wFVT_v"+s)
            #logs.append(outputDir+"/log_TTHad"+y+"_3b_wJCM_wFVT_v"+s)
            #logs.append(outputDir+"/log_TTSem"+y+"_3b_wJCM_wFVT_v"+s)
            #logs.append(outputDir+"/log_TT2L2Nu"+y+"_3b_wJCM_wFVT_v"+s)
    

            #
            # 4b
            #
            pico4b = "picoAOD_3bMix4b_noTTVeto_4b_v"+s+".root"
            histOut4b = " --histFile "+histName4b
    
            cmds.append(runCMD+" -i "+outputDir+"/data"+y+"_v"+s+"/"+pico4b+             picoOut  +   yearOpts[y]+ h10 + histOut4b + "  --FvTName "+FvTName + " --is3bMixed")    
            cmds.append(runCMD+" -i "+outputDir+"/TTToHadronic"+y+"_v"+s+"/"+pico4b+     picoOut  + MCyearOpts[y]+ h10 + histOut4b + "  --FvTName "+FvTName + " --is3bMixed")    
            cmds.append(runCMD+" -i "+outputDir+"/TTToSemiLeptonic"+y+"_v"+s+"/"+pico4b+ picoOut  + MCyearOpts[y]+ h10 + histOut4b + "  --FvTName "+FvTName + " --is3bMixed")    
            cmds.append(runCMD+" -i "+outputDir+"/TTTo2L2Nu"+y+"_v"+s+"/"+pico4b+        picoOut  + MCyearOpts[y]+ h10 + histOut4b + "  --FvTName "+FvTName + " --is3bMixed")
    
            logs.append(outputDir+"/log_"+y+"_4b_wFVT_v"+s)
            logs.append(outputDir+"/log_TTHad"+y+"_4b_wFVT_v"+s)
            logs.append(outputDir+"/log_TTSem"+y+"_4b_wFVT_v"+s)
            logs.append(outputDir+"/log_TT2L2Nu"+y+"_4b_wFVT_v"+s)
    
        
    babySit(cmds, doRun, logFiles=logs)

    
    #
    #  Hadd TTbar
    #
    cmds = []
    logs = []
    for s in subSamples:

        JCMName="3bMix4b_v"+s
        FvTName="_3bMix4b_v"+s
    
        #histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        histName4b = "hists_4b_wFVT"+FvTName+".root "

        for y in years:
            #cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName3b+" "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b+"  "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b+" "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b)
            #logs.append(outputDir+"/log_haddTT_3b_wJCM_wFvT_"+y+"_v"+s)
    
            cmds.append("hadd -f "+outputDir+"/TT"+y+"/"+histName4b+" "+outputDir+"/TTToHadronic"+y+"_v"+s+"/"+histName4b+"  "+outputDir+"/TTToSemiLeptonic"+y+"_v"+s+"/"+histName4b+" "+outputDir+"/TTTo2L2Nu"+y+"_v"+s+"/"+histName4b)
            logs.append(outputDir+"/log_haddTT_4b_wFvT_"+y+"_v"+s)

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
    
            JCMName="3bMix4b_v"+s
            FvTName="_3bMix4b_v"+s
        
            histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
            histName4b = "hists_4b_wFVT"+FvTName+".root "
    
    
            cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName3b+" "+outputDirComb+"/data2016/"+histName3b+" "+outputDirComb+"/data2017/"+histName3b+" "+outputDirComb+"/data2018/"+histName3b)
            cmds.append("hadd -f "+outputDir+"/dataRunII/"+histName4b+" "+outputDir+"/data2016_v"+s+"/"+histName4b+" "+outputDir+"/data2017_v"+s+"/"+histName4b+" "+outputDir+"/data2018_v"+s+"/"+histName4b)
            cmds.append("hadd -f "+outputDir+"/TTRunII/"  +histName4b+" "+outputDir+"/TT2016/"  +histName4b+" "+outputDir+"/TT2017/"  +histName4b+" "+outputDir+"/TT2018/"  +histName4b)

            logs.append(outputDir+"/log_haddDataRunII_3b_v"+s)
            logs.append(outputDir+"/log_haddDataRunII_4b_v"+s)
            logs.append(outputDir+"/log_haddDataRunII_TT_v"+s)

        babySit(cmds, doRun, logFiles=logs)


    if o.email: execute('echo "Subject: [make3bMix4bClosure] makeHistsWithFvT Done" | sendmail johnalison@cmu.edu',doRun)


#
#  Make Plots with FvT
#
if o.plotsWithFvT:
    cmds = []
    logs = []

    for s in subSamples:

        JCMName="3bMix4b_v"+s
        FvTName="_3bMix4b_v"+s

        histName3b = "hists_3b_wJCM_"+JCMName+"_wFVT"+FvTName+".root "
        histName4b = "hists_4b_wFVT"+FvTName+".root "

        for y in years:
    
            cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCutFlow.py "
            cmd += " --d4 "+outputDir+"/data"+y+"_v"+s+"/"+histName4b
            cmd += " --d3 "+outputDirComb+"/data"+y+"/"+histName3b
            cmd += " --t4 "+outputDir+"/TT"+y+"/"+histName4b
            cmd += " --t3 "+outputDir+"/TT"+y+"/"+histName3b
            cmd += " --t4_s "+outputDir+"/TTToSemiLeptonic"+y+"_v"+s+"/"+histName4b
            cmd += " --t4_h "+outputDir+"/TTToHadronic"+y+"_v"+s+"/"+histName4b
            cmd += " --t4_d "+outputDir+"/TTTo2L2Nu"+y+"_v"+s+"/"+histName4b
            cmd += " --t3_s "+outputDirComb+"/TTToSemiLeptonic"+y+"/"+histName3b
            cmd += " --t3_h "+outputDirComb+"/TTToHadronic"+y+"/"+histName3b
            cmd += " --t3_d "+outputDirComb+"/TTTo2L2Nu"+y+"/"+histName3b
            cmd += " --name "+outputDir+"/CutFlow_wFvT_"+y+"_v"+s
            cmd += " --makePDF -r"
            #cmds.append(cmd)
            #logs.append(outputDir+"/log_cutFlow_wFVT_"+y+"_v"+s)
    
            #
            # MAke Plots
            #
            data3bFile  = outputDirComb+"/data"+y+"/"+histName3b    if not y == "RunII" else outputDir+"/data"+y+"/"+histName3b               
            data4bFile  = outputDir+"/data"+y+"_v"+s+"/"+histName4b if not y == "RunII" else outputDir+"/data"+y+"/"+histName4b                
            ttbar4bFile = outputDir+"/TT"+y+"/"+histName4b

            cmd = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputDir+" -p plotsWithFvT_"+y+"_v"+s+plotOpts[y]+" -m -j -r --noSignal "
            cmd += " --data3b "+data3bFile
            cmd += " --data "+data4bFile
            cmd += " --TT "+ttbar4bFile
            cmds.append(cmd)
            logs.append(outputDir+"/log_makePlots_wFVT_"+y+"_v"+s)
    
    babySit(cmds, doRun, logFiles=logs)    
    
    cmds = []
    for s in subSamples:
        for y in years:
            #cmds.append("mv CutFlow_wFvT_"+y+"_v"+s+".pdf "+outputDir+"/")
            cmds.append("tar -C "+outputDir+" -zcf "+outputDir+"/plotsWithFvT_"+y+"_v"+s+".tar plotsWithFvT_"+y+"_v"+s)
            
    babySit(cmds, doRun)    



        
