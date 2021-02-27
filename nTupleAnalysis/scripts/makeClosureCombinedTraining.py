import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9", help="Year or comma separated list of subsamples")
parser.add_option('--doTrain', action="store_true",      help="Should be obvious")
parser.add_option('--plotFvTFits', action="store_true",      help="Should be obvious")
parser.add_option('--plotFvTFitsJackKnife', action="store_true",      help="")
parser.add_option('--addSvB', action="store_true",      help="Should be obvious")
parser.add_option('--addFvT', action="store_true",      help="Should be obvious")
parser.add_option('--addFvTComb', action="store_true",      help="Should be obvious")
parser.add_option('--addFvTStudies', action="store_true",      help="Should be obvious")
parser.add_option('--addFvTJackKnife', action="store_true",      help="Should be obvious")
parser.add_option('--makeClosurePlots', action="store_true",      help="Should be obvious")
parser.add_option('--makeJackKnifePlots', action="store_true",      help="Should be obvious")
parser.add_option('--skimH5', action="store_true",      help="Should be obvious")
parser.add_option('--email',            default=None,      help="")
parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--makeAutonDirs', action="store_true",      help="Should be obvious")
parser.add_option('--copyToAuton', action="store_true",      help="Should be obvious")
parser.add_option('--copySignalToAuton', action="store_true",      help="Should be obvious")
parser.add_option('--copyFromAuton', action="store_true",      help="Should be obvious")
parser.add_option('--trainOffset', default=1, help='training offset.')
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")

parser.add_option('--cuda', default=1, type=int, help='Which gpuid to use.')

o, a = parser.parse_args()

doRun = o.execute
subSamples = o.subSamples.split(",")
mixedName = o.mixedName
years = o.year.split(",")

ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]

CUDA=str(o.cuda)
#baseDir="/zfsauton2/home/jalison/hh4b/"
#baseDir="/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src"
outputDir="closureTests/combined/"
outputDirNom="closureTests/nominal/"
outputDir3bMix4b="closureTests/3bMix4b/"


# Helpers
#SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1713_lr0.008_epochs40_stdscale_epoch40_loss0.2138.pkl"
#SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.008_epochs40_stdscale_epoch39_loss0.1512.pkl"
#SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.008_epochs40_stdscale_epoch35_loss0.1461.pkl"
SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.01_epochs20_offset0_epoch20.pkl,ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.01_epochs20_offset2_epoch20.pkl,ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.01_epochs20_offset1_epoch20.pkl"

#SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.008_epochs40_stdscale_epoch38_loss0.1515.pkl"
trainJOB='python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py '
makeClosurePlots='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py'
makeJackKnifePlots='python  ZZ4b/nTupleAnalysis/scripts/jackKnifeStudyHDF5.py'
skimCMD="python   ZZ4b/nTupleAnalysis/scripts/skim_h5.py "
modelDir="ZZ4b/nTupleAnalysis/pytorchModels/"


#tagID = "b0p6"
tagID = "b0p60p3"


#
# Train
#   (with GPU enviorment)
if o.copyToAuton or o.makeAutonDirs or o.copyFromAuton:
    
    import os
    autonAddr = "jalison@lop2.autonlab.org"
    combinedDirName = "combined_"+o.mixedName
    
    
    def run(cmd):
        if doRun:
            os.system(cmd)
        else:
            print cmd
    
    def runA(cmd):
        print "> "+cmd
        run("ssh "+autonAddr+" "+cmd)
    
    def scp(fileName):
        cmd = "scp "+fileName+" "+autonAddr+":hh4b/"+fileName
        print "> "+cmd
        run(cmd)

    def scpFrom(fileName):
        cmd = "scp "+autonAddr+":hh4b/"+fileName+" "+fileName
        print "> "+cmd
        run(cmd)


    
    #
    # Setup directories
    #
    if o.makeAutonDirs:

        runA("mkdir hh4b/closureTests/")
        runA("mkdir hh4b/closureTests/"+combinedDirName)
    
        for y in ["2018","2017","2016"]:
            runA("mkdir hh4b/closureTests/"+combinedDirName+"/data"+y+"_"+tagID)
    
            for tt in ttbarSamples:
                runA("mkdir hh4b/closureTests/"+combinedDirName+"/"+tt+y+"_"+tagID)
                #runA("mkdir hh4b/closureTests/"+combinedDirName+"/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll")
                runA("mkdir hh4b/closureTests/"+combinedDirName+"/"+tt+y+"_PSData_"+tagID)
                runA("mkdir hh4b/closureTests/"+combinedDirName+"/"+tt+y+"_noPSData_"+tagID)

            for s in subSamples:
                runA("mkdir hh4b/closureTests/"+combinedDirName+"/data"+y+"_"+o.mixedName+"_"+tagID+"_v"+s)
                
    
    #
    # Copy Files
    #
    if o.copyToAuton:
        for y in ["2018","2017","2016"]:
            scp("closureTests/"+combinedDirName+"/data"+y+"_"+tagID+"/picoAOD_3b_wJCM_"+tagID+".h5")
            scp("closureTests/"+combinedDirName+"//data"+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5")

            for tt in ttbarSamples:
                scp("closureTests/"+combinedDirName+"/"+tt+y+"_"+tagID+"/picoAOD_3b_wJCM_"+tagID+".h5")
                scp("closureTests/"+combinedDirName+"/"+tt+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5")
                #scp("closureTests/"+combinedDirName+"/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll/picoAOD_"+o.mixedName+"_4b_"+tagID+"_vAll.h5")
                scp("closureTests/"+combinedDirName+"/"+tt+y+"_PSData_"+tagID+"/picoAOD_4b_PSData_"+tagID+".h5")
                scp("closureTests/"+combinedDirName+"/"+tt+y+"_noPSData_"+tagID+"/picoAOD_4b_noPSData_"+tagID+".h5")

            for s in subSamples:
                scp("closureTests/"+combinedDirName+"/data"+y+"_"+o.mixedName+"_"+tagID+"_v"+s+"/picoAOD_"+o.mixedName+"_4b_"+tagID+"_v"+s+".h5")


    #
    # Copy Files
    #
    if o.copyFromAuton:
        for y in ["2018","2017","2016"]:
            scpFrom("closureTests/"+combinedDirName+"/data"+y+"_"+tagID+"/picoAOD_3b_wJCM_"+tagID+".h5")
            scpFrom("closureTests/"+combinedDirName+"//data"+y+"_"+tagID+"/picoAOD_4b_"+tagID+".h5")

            for tt in ttbarSamples:
                scpFrom("closureTests/"+combinedDirName+"/"+tt+y+"_"+tagID+"/picoAOD_3b_wJCM_"+tagID+".h5")
                #scpFrom("closureTests/"+combinedDirName+"/"+tt+y+"_"+mixedName+"_"+tagID+"_vAll/picoAOD_"+o.mixedName+"_4b_"+tagID+"_vAll.h5")
                scpFrom("closureTests/"+combinedDirName+"/"+tt+y+"_PSData_"+tagID+"/picoAOD_4b_PSData_"+tagID+".h5")
                scpFrom("closureTests/"+combinedDirName+"/"+tt+y+"_noPSData_"+tagID+"/picoAOD_4b_noPSData_"+tagID+".h5")


            for s in subSamples:
                scpFrom("closureTests/"+combinedDirName+"/data"+y+"_"+o.mixedName+"_"+tagID+"_v"+s+"/picoAOD_"+o.mixedName+"_4b_"+tagID+"_v"+s+".h5")



#
# Train
#   (with GPU enviorment)
if o.copySignalToAuton:
    
    import os
    autonAddr = "jalison@lop2.autonlab.org"
    combinedDirName = "combined_"+o.mixedName
    
    def run(cmd):
        if doRun:
            os.system(cmd)
        else:
            print cmd
    
    def runA(cmd):
        print "> "+cmd
        run("ssh "+autonAddr+" "+cmd)
    
    def scp(localFile, autonFile):
        cmd = "scp "+localFile+" "+autonAddr+":hh4b/"+autonFile
        print "> "+cmd
        run(cmd)



    
    #
    # Setup directories
    #
    if True: 

        for sig in ["ZH4b","ZZ4b","ggZH4b"]:
    
            for y in ["2018","2017","2016"]:
                runA("mkdir hh4b/closureTests/"+combinedDirName+"/"+sig+y)
    
    
    #
    # Copy Files
    #
    if o.copySignalToAuton:

        for sig in ["ZH4b","ZZ4b","ggZH4b"]:

            for y in ["2018","2017","2016"]:
                scp("/uscms/home/bryantp/nobackup/ZZ4b/"+sig+y+"/picoAOD.h5", "closureTests/"+combinedDirName+"/"+sig+y+"/picoAOD.h5")



#
# Train
#   (with GPU enviorment)
if o.doTrain:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_'+tagID+'/pico*3b*JCM_'+tagID+'.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*3b_wJCM_'+tagID+'.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*4b_'+tagID+'.h5" '

    JCMPostFix = ""
    #JCMPostFix = "_comb"
    outName = "3bTo4b."+tagID+""+JCMPostFix.replace("_",".")
    cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight_Nominal"+JCMPostFix+"  --trainOffset 0,1,2 --train --update  --updatePostFix _Nominal "
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Train_FvT_3bTo4b_"+tagID+""+JCMPostFix)

    for s in subSamples:

        outName = (mixedName+"_v"+s+"."+tagID+""+JCMPostFix).replace("_",".")
        dataFiles4bMix = '"'+outputDir+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir+'/*TT*201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '        

        cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+JCMPostFix+"  --trainOffset 0,1,2 --train --update  --updatePostFix _"+mixedName+"_v"+s
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Train_FvT_3bMix4b_"+tagID+"_v"+s+JCMPostFix)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] FvT Training  Done" | sendmail '+o.email,doRun)






#
#  plotFvTFits
#
if o.plotFvTFits:
    cmds = []
    logs = []


    #modelsLogFiles  = modelDir+"3bTo4b."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v0."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v1."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v2."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v3."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v4."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v5."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v6."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"

    #modelsLogFiles = modelDir+"3bTo4b."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v0."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v1."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v2."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v3."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v4."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v5."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v6."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"

    modelsLogFiles = modelDir+"3bTo4b."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v1."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v2."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v3."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v4."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v5."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v6."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"

    
    modelNames = "Nominal,3bMix4bV0,3bMix4bV1,3bMix4bV2,3bMix4bV3,3bMix4bV4,3bMix4bV5,3bMix4bV6"
    #modelNames = "Nominal,3bMix4bv0,3bMix4br,3bMix4bV2,3bMix4bV3,3bMix4bV4"

    cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_FvTFits_"+mixedName+"_"+tagID
    cmd += " -i "+modelsLogFiles+" --names "+modelNames

    cmds.append(cmd)
    logs.append(outputDir+"/log_plotFvT_"+mixedName+"_"+tagID)


    babySit(cmds, doRun, logFiles=logs)


#
# Add SvB
#
if o.addSvB:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_'+tagID+'/pico*3b*JCM_'+tagID+'.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*3b_wJCM_'+tagID+'.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*4b_'+tagID+'.h5" '


    cmd = trainJOB+' -u -m '+SvBModel+' -c SvB --cuda '+CUDA  
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Add_SvB_Nominal_"+tagID+"")


    for s in subSamples:
        dataFiles4bMix = '"'+outputDir+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir+'/*TT*201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '        

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
        cmd += ' -d '+dataFiles4bMix
        cmd += ' -t '+ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_SvB_"+mixedName+"_v"+s+"_"+tagID+"")
        

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] Add SvB  Done" | sendmail '+o.email,doRun)


#
# Add FvT
#
FvTModel = {} 
#FvTModel["Nominal"]=modelDir+"/3bTo4bITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch40_loss0.1553.pkl"
#FvTModel["0"]=modelDir+"/3bMix4brWbW2v0FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch20_loss0.1425.pkl"
#FvTModel["1"]=modelDir+"/3bMix4brWbW2v1FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch29_loss0.1420.pkl"
#FvTModel["2"]=modelDir+"/3bMix4brWbW2v2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch18_loss0.1426.pkl"
#FvTModel["3"]=modelDir+"/3bMix4brWbW2v3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch28_loss0.1416.pkl"
#FvTModel["4"]=modelDir+"/3bMix4brWbW2v4FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch22_loss0.1420.pkl"

#FvTModel["Nominal"]=modelDir+"3bTo4b."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch35_loss0.1230.pkl"
#FvTModel["0"]=modelDir+"3bMix4b.v0."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1334.pkl"
#FvTModel["1"]=modelDir+"3bMix4b.v1."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch37_loss0.1351.pkl"
#FvTModel["2"]=modelDir+"3bMix4b.v2."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1357.pkl"
#FvTModel["3"]=modelDir+"3bMix4b.v3."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1353.pkl"
#FvTModel["4"]=modelDir+"3bMix4b.v4."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1349.pkl"
#FvTModel["5"]=modelDir+"3bMix4b.v5."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1349.pkl"
#FvTModel["6"]=modelDir+"3bMix4b.v6."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch47_loss0.1350.pkl"

FvTModel["Nominal"]=modelDir+"3bTo4b."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.01_epochs40_offset2_epoch20_loss0.8061.pkl,"+modelDir+"3bTo4b."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.01_epochs40_offset0_epoch20_loss0.8081.pkl,"+modelDir+"3bTo4b."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.01_epochs40_offset1_epoch20_loss0.8068.pkl"
FvTModel["0"]=modelDir+"3bMix4b.v0."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.01_epochs50_offset1_epoch20_loss0.9234.pkl,"+modelDir+"3bMix4b.v0."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.01_epochs50_offset2_epoch20_loss0.9206.pkl,"+modelDir+"3bMix4b.v0."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.01_epochs50_offset0_epoch20_loss0.9218.pkl"
FvTModel["1"]=modelDir+"3bMix4b.rWbW2.v1."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1180.pkl"
FvTModel["2"]=modelDir+"3bMix4b.rWbW2.v2."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1187.pkl"
FvTModel["3"]=modelDir+"3bMix4b.rWbW2.v3."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1183.pkl"
FvTModel["4"]=modelDir+"3bMix4b.rWbW2.v4."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1164.pkl"
FvTModel["5"]=modelDir+"3bMix4b.rWbW2.v5."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch30_loss0.1177.pkl"
FvTModel["6"]=modelDir+"3bMix4b.rWbW2.v6."+tagID+"FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1165.pkl"

#
# Add FvT
#
if o.addFvT:
    cmds = []
    logs = []


    dataFiles3b = '"'+outputDir+'/*data201*_'+tagID+'/pico*3b*JCM_'+tagID+'.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*3b_wJCM_'+tagID+'.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*4b_'+tagID+'.h5" '

    
    cmd = trainJOB+' -u -m '+FvTModel["Nominal"]+' -c FvT --cuda '+CUDA+' --updatePostFix _Nominal '
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Add_FvT_Nominal_"+tagID+"")


    for s in subSamples:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '        

        cmd = trainJOB+' -u  -m '+FvTModel[s]+' -c FvT  --cuda '+CUDA  + ' --updatePostFix _'+mixedName+'_v'+s
        cmd += ' -d '+dataFiles3b
        cmd += ' --data4b '+dataFiles4bMix
        cmd += ' --ttbar4b '+ttFile4bMix
        cmd += ' -t '+ttFile3b
        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_FvT_"+mixedName+"_"+tagID+"_v"+s)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] add FvT  Done" | sendmail '+o.email,doRun)

FvTModelComb = {} 
FvTModelComb["Nominal"]=modelDir+"3bTo4b."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch40_loss0.1264.pkl"
FvTModelComb["0"]=modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1164.pkl"
FvTModelComb["1"]=modelDir+"3bMix4b.rWbW2.v1."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1170.pkl"
FvTModelComb["2"]=modelDir+"3bMix4b.rWbW2.v2."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1177.pkl"
FvTModelComb["3"]=modelDir+"3bMix4b.rWbW2.v3."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch48_loss0.1175.pkl"
FvTModelComb["4"]=modelDir+"3bMix4b.rWbW2.v4."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1154.pkl"
FvTModelComb["5"]=modelDir+"3bMix4b.rWbW2.v5."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1169.pkl"
FvTModelComb["6"]=modelDir+"3bMix4b.rWbW2.v6."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1157.pkl"


#
# Add FvT
#
if o.addFvTComb:
    cmds = []
    logs = []


    dataFiles3b = '"'+outputDir+'/*data201*_'+tagID+'/pico*3b*JCM_'+tagID+'.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*3b_wJCM_'+tagID+'.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    
    cmd = trainJOB+' -u -m '+FvTModelComb["Nominal"]+' -c FvT --cuda '+CUDA+' --updatePostFix _Nominal '
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Add_FvT_Nominal_"+tagID+"")


    for s in subSamples:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '        

        cmd = trainJOB+' -u  -m '+FvTModelComb[s]+' -c FvT  --cuda '+CUDA  + ' --updatePostFix _'+mixedName+'_v'+s+'_comb'
        cmd += ' -d '+dataFiles3b
        cmd += ' --data4b '+dataFiles4bMix
        cmd += ' --ttbar4b '+ttFile4bMix
        cmd += ' -t '+ttFile3b
        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_FvT_comb"+mixedName+"_"+tagID+"_v"+s)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] add FvTComb  Done" | sendmail '+o.email,doRun)


FvTModelStudies = {} 
#FvTModelStudies["Nominal"]=modelDir+"3bTo4b."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch40_loss0.1264.pkl"
FvTModelStudies["0"] = {}
#Nominal FvTModelStudies["0"]=modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1164.pkl"
FvTModelStudies["0"]["e2"] = modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch2_loss0.1197.pkl"
FvTModelStudies["0"]["e9"] = modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch9_loss0.1177.pkl"
FvTModelStudies["0"]["e39"] = modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch39_loss0.1164.pkl"
FvTModelStudies["0"]["e45"] = modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch45_loss0.1164.pkl"
FvTModelStudies["0"]["e49"] = modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch49_loss0.1164.pkl"
FvTModelStudies["0"]["e50"] = modelDir+"3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1164.pkl"

#FvTModelStudies["1"]=modelDir+"3bMix4b.rWbW2.v1."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1170.pkl"
#FvTModelStudies["2"]=modelDir+"3bMix4b.rWbW2.v2."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1177.pkl"
#FvTModelStudies["3"]=modelDir+"3bMix4b.rWbW2.v3."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch48_loss0.1175.pkl"
#FvTModelStudies["4"]=modelDir+"3bMix4b.rWbW2.v4."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1154.pkl"
#FvTModelStudies["5"]=modelDir+"3bMix4b.rWbW2.v5."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1169.pkl"
#FvTModelStudies["6"]=modelDir+"3bMix4b.rWbW2.v6."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1157.pkl"


#
# Add FvT for studies
#
if o.addFvTStudies:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_'+tagID+'/pico*3b*JCM_'+tagID+'.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*3b_wJCM_'+tagID+'.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    
    #cmd = trainJOB+' -u -m '+FvTModelComb["Nominal"]+' -c FvT --cuda '+CUDA+' --updatePostFix _Nominal_comb '
    #cmd += ' -d '+dataFiles3b
    #cmd += ' --data4b '+dataFiles4b
    #cmd += ' -t '+ttFile3b
    #cmd += ' --ttbar4b '+ttFile4b
    #
    #cmds.append(cmd)
    #logs.append(outputDir+"/log_Add_FvT_Nominal_comb_"+tagID+"")


    #for s in subSamples:
    for s in ["0"]:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '        

        for e in FvTModelStudies[s]:
            cmd = trainJOB+' -u  -m '+FvTModelStudies[s][e]+' -c FvT  --cuda '+CUDA  + ' --updatePostFix _'+mixedName+'_v'+s+'_comb_'+e
            cmd += ' -d '+dataFiles3b
            cmd += ' --data4b '+dataFiles4bMix
            cmd += ' --ttbar4b '+ttFile4bMix
            cmd += ' -t '+ttFile3b
            cmds.append(cmd)
            logs.append(outputDir+"/log_Add_FvT_studies"+mixedName+"_"+tagID+"_v"+s)
    

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] add FvTStudies  Done" | sendmail '+o.email,doRun)




#
#  Make Closure Plots
#
if o.makeClosurePlots:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_'+tagID+'/pico*3b*JCM_'+tagID+'.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*3b_wJCM_'+tagID+'.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*4b_'+tagID+'.h5" '

    weightPostFix = ""
    #weightPostFix = "_comb"

    cmd = makeClosurePlots+"  --weightName mcPseudoTagWeight_Nominal"+weightPostFix+"  --FvTName FvT_Nominal"+weightPostFix+" -o "+outputDir+"/PlotsNominal_"+tagID+""+weightPostFix
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Train_FvT_3bTo4b_"+tagID+""+weightPostFix)

    for s in subSamples:

        dataFiles4bMix = '"'+outputDir+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir+'/*TT*201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '        

        cmd = makeClosurePlots+ "  --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+weightPostFix+"  --FvTName FvT_"+mixedName+"_v"+s+weightPostFix+"  -o "+outputDir+"/Plots_"+mixedName+"_"+tagID+"_v"+s+weightPostFix
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Train_FvT_3bMix4b_"+tagID+"_v"+s+weightPostFix)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] makeClosurePlots  Done" | sendmail '+o.email,doRun)


#
#  Skiming (Not usually needed)
#
if o.skimH5:
    cmds = []
    logs = []

    for y in years:
        #cmds.append("ls "+outputDirNom+"/data"+y+"/picoAOD_4b.h5")
        #cmds.append("mv "+outputDirNom+"/data"+y+"/picoAOD_4b.h5  "+outputDirNom+"/data"+y+"/picoAOD_4b_All.h5")
        cmds.append(skimCMD+" -o"+outputDirNom+"/data"+y+"/picoAOD_4b.h5 -i "+outputDirNom+"/data"+y+"/picoAOD_4b_All.h5")


        #cmds.append("ls "+outputDir+"/data"+y+"/picoAOD_3b_wJCM.h5")
        #cmds.append("mv "+outputDir+"/data"+y+"/picoAOD_3b_wJCM.h5  "+outputDir+"/data"+y+"/picoAOD_3b_wJCM_All.h5")
        cmds.append(skimCMD+" -o "+outputDir+"/data"+y+"/picoAOD_3b_wJCM.h5  -i "+outputDir+"/data"+y+"/picoAOD_3b_wJCM_All.h5")

        for t in ttbarSamples:
            #cmds.append("ls "+outputDirNom+"/"+t+y+"/picoAOD_4b.h5")
            #cmds.append("mv "+outputDirNom+"/"+t+y+"/picoAOD_4b.h5  "+outputDirNom+"/"+t+y+"/picoAOD_4b_All.h5")
            cmds.append(skimCMD+" -o "+outputDirNom+"/"+t+y+"/picoAOD_4b.h5  -i "+outputDirNom+"/"+t+y+"/picoAOD_4b_All.h5")
    
            #cmds.append("ls "+outputDir+"/"+t+y+"/picoAOD_3b_wJCM.h5")
            #cmds.append("mv "+outputDir+"/"+t+y+"/picoAOD_3b_wJCM.h5  "+outputDir+"/"+t+y+"/picoAOD_3b_wJCM_All.h5")
            cmds.append(skimCMD+" -o "+outputDir+"/"+t+y+"/picoAOD_3b_wJCM.h5  -i  "+outputDir+"/"+t+y+"/picoAOD_3b_wJCM_All.h5")

        for s in subSamples:
    
            #cmds.append("ls "+outputDir3bMix4b+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+".h5")
            #cmds.append("mv "+outputDir3bMix4b+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+".h5  "+outputDir3bMix4b+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+"_All.h5")
            cmds.append(skimCMD+" -o "+outputDir3bMix4b+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+".h5  -i "+outputDir3bMix4b+"/data"+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+"_All.h5")
            
            for t in ttbarSamples:
                #cmds.append("ls "+outputDir3bMix4b+"/"+t+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+".h5")
                #cmds.append("mv "+outputDir3bMix4b+"/"+t+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+".h5  "+outputDir3bMix4b+"/"+t+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+"_All.h5")
                cmds.append(skimCMD+" -o "+outputDir3bMix4b+"/"+t+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+".h5 -i "+outputDir3bMix4b+"/"+t+y+"_v"+s+"/picoAOD_"+mixedName+"_4b_v"+s+"_All.h5")
            
        


    #    
    #cmd = skimCMD+" -o /zfsauton2/home/jalison/hh4b//closureTests/combined/TTToHadronic2016/picoAOD_3b_wJCM.h5     -i  /zfsauton2/home/jalison/hh4b//closureTests/combined/TTToHadronic2016/picoAOD_3b_wJCM_All.h5
    babySit(cmds, doRun, logFiles=logs)




#
#  plotFvTFits
#
if o.plotFvTFitsJackKnife:
    cmds = []
    logs = []

    modelsLogFiles  = modelDir+"3bTo4b."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bTo4b."+tagID+".comb.offSet1FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bTo4b."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"


    modelNames = "Offset0,Offset1,Offset2"
    
    cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_FvTFits_"+mixedName+"_"+tagID+"_comb_jackKnife"
    cmd += " -i "+modelsLogFiles+" --names "+modelNames

    cmds.append(cmd)
    logs.append(outputDir+"/log_plotFvT_"+mixedName+"_"+tagID+"_comb")


    for s in subSamples:
    
        modelsLogFiles  = modelDir+"3bMix4b.rWbW2.v"+s+"."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
        modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v"+s+"."+tagID+".comb.offSet1FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
        modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v"+s+"."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"

        modelNames = "Offset0,Offset1,Offset2"
        #modelNames = "Offset0,Offset1"

        cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_FvTFits_"+mixedName+"_"+tagID+"_comb_jackKnife_v"+s
        cmd += " -i "+modelsLogFiles+" --names "+modelNames

        cmds.append(cmd)
        logs.append(outputDir+"/log_plotFvT_"+mixedName+"_"+tagID+"_comb")


    babySit(cmds, doRun, logFiles=logs)


FvTModelJK = {} 
FvTModelJK["Nominal"]=["3bTo4b."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch40_loss0.1260.pkl",
                       "3bTo4b."+tagID+".comb.offSet1FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch40_loss0.1261.pkl",
                       "3bTo4b."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch40_loss0.1261.pkl",
                       ]

FvTModelJK["0"]=["3bMix4b.rWbW2.v0."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1170.pkl",
                 "3bMix4b.rWbW2.v0."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1165.pkl",
                 "3bMix4b.rWbW2.v0."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1174.pkl"
                 ]
                 
FvTModelJK["1"]=["3bMix4b.rWbW2.v1."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1172.pkl",
                 "3bMix4b.rWbW2.v1."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1169.pkl",
                 "3bMix4b.rWbW2.v1."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1173.pkl",
                 ]

FvTModelJK["2"]=["3bMix4b.rWbW2.v2."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1178.pkl",
                 "3bMix4b.rWbW2.v2."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1177.pkl",
                 "3bMix4b.rWbW2.v2."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1174.pkl",
                 ]

FvTModelJK["3"]=["3bMix4b.rWbW2.v3."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch49_loss0.1174.pkl",
                 "3bMix4b.rWbW2.v3."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1174.pkl",
                 "3bMix4b.rWbW2.v3."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1174.pkl",
                 ]

FvTModelJK["4"]=["3bMix4b.rWbW2.v4."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1161.pkl",
                 "3bMix4b.rWbW2.v4."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1156.pkl",
                 "3bMix4b.rWbW2.v4."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1156.pkl",
                 ]

FvTModelJK["5"]=["3bMix4b.rWbW2.v5."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1166.pkl",
                 "3bMix4b.rWbW2.v5."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1168.pkl",
                 "3bMix4b.rWbW2.v5."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1166.pkl",
                 ]

FvTModelJK["6"]=["3bMix4b.rWbW2.v6."+tagID+".comb.offSet2FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1159.pkl",
                 "3bMix4b.rWbW2.v6."+tagID+".combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1158.pkl",
                 "3bMix4b.rWbW2.v6."+tagID+".comb.offSet0FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch45_loss0.1161.pkl",
                 ]


#
# Add FvT
#
if o.addFvTJackKnife:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_'+tagID+'/pico*3b*JCM_'+tagID+'.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*3b_wJCM_'+tagID+'.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*4b_'+tagID+'.h5" '


    for offSet in [0,1,2]:
        
        cmd = trainJOB+' -u -m '+modelDir+FvTModelJK["Nominal"][offSet]+' -c FvT --cuda '+CUDA+' --updatePostFix _Nominal_offSet'+str(offSet)
        cmd += ' -d '+dataFiles3b
        cmd += ' --data4b '+dataFiles4b
        cmd += ' -t '+ttFile3b
        cmd += ' --ttbar4b '+ttFile4b

        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_FvT_Nominal_"+tagID+"_offset"+str(offSet))


        for s in subSamples:
    
            dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
            ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '        
    
            cmd = trainJOB+' -u  -m '+modelDir+FvTModelJK[s][offSet]+' -c FvT  --cuda '+CUDA  + ' --updatePostFix _'+mixedName+'_v'+s+'_offset'+str(offSet)
            cmd += ' -d '+dataFiles3b
            cmd += ' --data4b '+dataFiles4bMix
            cmd += ' --ttbar4b '+ttFile4bMix
            cmd += ' -t '+ttFile3b
            cmds.append(cmd)
            logs.append(outputDir+"/log_Add_FvT_comb"+mixedName+"_"+tagID+"_v"+s+"_offset"+str(offSet))
    

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] add FvTJK  Done" | sendmail '+o.email,doRun)



#
#  Make JackKnife Plots
#
if o.makeJackKnifePlots:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_'+tagID+'/pico*3b*JCM_'+tagID+'.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_'+tagID+'/pico*4b_'+tagID+'.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*3b_wJCM_'+tagID+'.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_'+tagID+'/pico*4b_'+tagID+'.h5" '

    #weightPostFix = ""
    weightPostFix = "_comb"
    # note FVT wieghts use offsetX and have the combined JCM, but _comb is not in the FVT name

    cmd = makeJackKnifePlots+"  --weightName mcPseudoTagWeight_Nominal"+weightPostFix+"  --FvTName FvT_Nominal"+" -o "+outputDir+"/JackKnifePlotsNominal_"+tagID+""+weightPostFix
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Train_FvT_3bTo4b_"+tagID+""+weightPostFix)

    for s in subSamples:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '        

        cmd = makeJackKnifePlots+ "  --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+weightPostFix+"  --FvTName FvT_"+mixedName+"_v"+s+"  -o "+outputDir+"/JackKnifePlots_"+mixedName+"_"+tagID+"_v"+s+weightPostFix
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Train_FvT_3bMix4b_"+tagID+"_v"+s+weightPostFix)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] makeClosurePlots  Done" | sendmail '+o.email,doRun)
