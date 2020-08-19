import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6", help="Year or comma separated list of subsamples")
parser.add_option('--doTrain', action="store_true",      help="Should be obvious")
parser.add_option('--plotFvTFits', action="store_true",      help="Should be obvious")
parser.add_option('--addSvB', action="store_true",      help="Should be obvious")
parser.add_option('--addFvT', action="store_true",      help="Should be obvious")
parser.add_option('--addFvTComb', action="store_true",      help="Should be obvious")
parser.add_option('--addFvTStudies', action="store_true",      help="Should be obvious")
parser.add_option('--makeClosurePlots', action="store_true",      help="Should be obvious")
parser.add_option('--skimH5', action="store_true",      help="Should be obvious")
parser.add_option('--email',            default=None,      help="")
parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('-y',                                 dest="year",      default="2018,2017,2016", help="Year or comma separated list of years")

parser.add_option('--cuda', default=1, type=int, help='Which gpuid to use.')

o, a = parser.parse_args()

doRun = o.execute
subSamples = o.subSamples.split(",")
mixedName = o.mixedName
years = o.year.split(",")

ttbarSamples = ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]

CUDA=str(o.cuda)
baseDir="/zfsauton2/home/jalison/hh4b/"
outputDir=baseDir+"/closureTests/combined"
outputDirNom=baseDir+"/closureTests/nominal"
outputDir3bMix4b=baseDir+"/closureTests/3bMix4b"


# Helpers
#SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1713_lr0.008_epochs40_stdscale_epoch40_loss0.2138.pkl"
SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.008_epochs40_stdscale_epoch39_loss0.1512.pkl"
#SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.008_epochs40_stdscale_epoch38_loss0.1515.pkl"
trainJOB='python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py '
makeClosurePlots='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py'
skimCMD="python   ZZ4b/nTupleAnalysis/scripts/skim_h5.py "
modelDir="ZZ4b/nTupleAnalysis/pytorchModels/"



#
# Train
#   (with GPU enviorment)
if o.doTrain:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_b0p6/pico*3b*JCM_b0p6.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_b0p6/pico*4b_b0p6.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_b0p6/pico*3b_wJCM_b0p6.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_b0p6/pico*4b_b0p6.h5" '

    #JCMPostFix = ""
    JCMPostFix = "_comb"
    outName = "3bTo4b.b0p6"+JCMPostFix.replace("_",".")
    cmd = trainJOB+ " -c FvT -e 40 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight_Nominal"+JCMPostFix+" " 
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Train_FvT_3bTo4b_b0p6"+JCMPostFix)

    for s in subSamples:

        outName = (mixedName+"_v"+s+".b0p6"+JCMPostFix).replace("_",".")
        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '        

        cmd = trainJOB+ " -c FvT -e 50 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+JCMPostFix
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Train_FvT_3bMix4b_b0p6_v"+s+JCMPostFix)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] FvT Training  Done" | sendmail '+o.email,doRun)



#
#  plotFvTFits
#
if o.plotFvTFits:
    cmds = []
    logs = []


    #modelsLogFiles  = modelDir+"3bTo4b.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v0.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v1.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v2.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v3.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v4.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v5.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.v6.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"

    #modelsLogFiles = modelDir+"3bTo4b.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v0.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v1.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v2.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v3.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v4.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v5.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    #modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v6.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"

    modelsLogFiles = modelDir+"3bTo4b.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v1.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v2.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v3.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v4.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v5.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"3bMix4b.rWbW2.v6.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale.log"

    
    modelNames = "Nominal,3bMix4bV0,3bMix4bV1,3bMix4bV2,3bMix4bV3,3bMix4bV4,3bMix4bV5,3bMix4bV6"
    #modelNames = "Nominal,3bMix4bv0,3bMix4br,3bMix4bV2,3bMix4bV3,3bMix4bV4"

    cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_FvTFits_"+mixedName+"_b0p6_comb"
    cmd += " -i "+modelsLogFiles+" --names "+modelNames

    cmds.append(cmd)
    logs.append(outputDir+"/log_plotFvT_"+mixedName+"_b0p6_comb")


    babySit(cmds, doRun, logFiles=logs)


#
# Add SvB
#
if o.addSvB:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_b0p6/pico*3b*JCM_b0p6.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_b0p6/pico*4b_b0p6.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_b0p6/pico*3b_wJCM_b0p6.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_b0p6/pico*4b_b0p6.h5" '


    cmd = trainJOB+' -u -m '+SvBModel+' -c SvB --cuda '+CUDA  
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Add_SvB_Nominal_b0p6")


    for s in subSamples:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '        

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
        cmd += ' -d '+dataFiles4bMix
        cmd += ' -t '+ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_SvB_"+mixedName+"_v"+s+"_b0p6")
        

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

#FvTModel["Nominal"]=modelDir+"3bTo4b.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch35_loss0.1230.pkl"
#FvTModel["0"]=modelDir+"3bMix4b.v0.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1334.pkl"
#FvTModel["1"]=modelDir+"3bMix4b.v1.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch37_loss0.1351.pkl"
#FvTModel["2"]=modelDir+"3bMix4b.v2.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1357.pkl"
#FvTModel["3"]=modelDir+"3bMix4b.v3.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1353.pkl"
#FvTModel["4"]=modelDir+"3bMix4b.v4.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1349.pkl"
#FvTModel["5"]=modelDir+"3bMix4b.v5.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1349.pkl"
#FvTModel["6"]=modelDir+"3bMix4b.v6.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch47_loss0.1350.pkl"

FvTModel["Nominal"]=modelDir+"3bTo4b.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch40_loss0.1274.pkl"
FvTModel["0"]=modelDir+"3bMix4b.rWbW2.v0.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1175.pkl"
FvTModel["1"]=modelDir+"3bMix4b.rWbW2.v1.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1180.pkl"
FvTModel["2"]=modelDir+"3bMix4b.rWbW2.v2.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1187.pkl"
FvTModel["3"]=modelDir+"3bMix4b.rWbW2.v3.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1183.pkl"
FvTModel["4"]=modelDir+"3bMix4b.rWbW2.v4.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1164.pkl"
FvTModel["5"]=modelDir+"3bMix4b.rWbW2.v5.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch30_loss0.1177.pkl"
FvTModel["6"]=modelDir+"3bMix4b.rWbW2.v6.b0p6FvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1165.pkl"

#
# Add FvT
#
if o.addFvT:
    cmds = []
    logs = []


    dataFiles3b = '"'+outputDir+'/*data201*_b0p6/pico*3b*JCM_b0p6.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_b0p6/pico*4b_b0p6.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_b0p6/pico*3b_wJCM_b0p6.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_b0p6/pico*4b_b0p6.h5" '

    
    cmd = trainJOB+' -u -m '+FvTModel["Nominal"]+' -c FvT --cuda '+CUDA+' --updatePostFix _Nominal '
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Add_FvT_Nominal_b0p6")


    for s in subSamples:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '        

        cmd = trainJOB+' -u  -m '+FvTModel[s]+' -c FvT  --cuda '+CUDA  + ' --updatePostFix _'+mixedName+'_v'+s
        cmd += ' -d '+dataFiles3b
        cmd += ' --data4b '+dataFiles4bMix
        cmd += ' --ttbar4b '+ttFile4bMix
        cmd += ' -t '+ttFile3b
        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_FvT_"+mixedName+"_b0p6_v"+s)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] add FvT  Done" | sendmail '+o.email,doRun)

FvTModelComb = {} 
FvTModelComb["Nominal"]=modelDir+"3bTo4b.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch40_loss0.1264.pkl"
FvTModelComb["0"]=modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1164.pkl"
FvTModelComb["1"]=modelDir+"3bMix4b.rWbW2.v1.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1170.pkl"
FvTModelComb["2"]=modelDir+"3bMix4b.rWbW2.v2.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1177.pkl"
FvTModelComb["3"]=modelDir+"3bMix4b.rWbW2.v3.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch48_loss0.1175.pkl"
FvTModelComb["4"]=modelDir+"3bMix4b.rWbW2.v4.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1154.pkl"
FvTModelComb["5"]=modelDir+"3bMix4b.rWbW2.v5.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1169.pkl"
FvTModelComb["6"]=modelDir+"3bMix4b.rWbW2.v6.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1157.pkl"


#
# Add FvT
#
if o.addFvTComb:
    cmds = []
    logs = []


    dataFiles3b = '"'+outputDir+'/*data201*_b0p6/pico*3b*JCM_b0p6.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_b0p6/pico*4b_b0p6.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_b0p6/pico*3b_wJCM_b0p6.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_b0p6/pico*4b_b0p6.h5" '
    
    cmd = trainJOB+' -u -m '+FvTModelComb["Nominal"]+' -c FvT --cuda '+CUDA+' --updatePostFix _Nominal_comb '
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Add_FvT_Nominal_comb_b0p6")


    for s in subSamples:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '        

        cmd = trainJOB+' -u  -m '+FvTModelComb[s]+' -c FvT  --cuda '+CUDA  + ' --updatePostFix _'+mixedName+'_v'+s+'_comb'
        cmd += ' -d '+dataFiles3b
        cmd += ' --data4b '+dataFiles4bMix
        cmd += ' --ttbar4b '+ttFile4bMix
        cmd += ' -t '+ttFile3b
        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_FvT_comb"+mixedName+"_b0p6_v"+s)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] add FvTComb  Done" | sendmail '+o.email,doRun)


FvTModelStudies = {} 
#FvTModelStudies["Nominal"]=modelDir+"3bTo4b.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs40_stdscale_epoch40_loss0.1264.pkl"
FvTModelStudies["0"] = {}
#Nominal FvTModelStudies["0"]=modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1164.pkl"
FvTModelStudies["0"]["e2"] = modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch2_loss0.1197.pkl"
FvTModelStudies["0"]["e9"] = modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch9_loss0.1177.pkl"
FvTModelStudies["0"]["e39"] = modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch39_loss0.1164.pkl"
FvTModelStudies["0"]["e45"] = modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch45_loss0.1164.pkl"
FvTModelStudies["0"]["e49"] = modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch49_loss0.1164.pkl"
FvTModelStudies["0"]["e50"] = modelDir+"3bMix4b.rWbW2.v0.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1164.pkl"

#FvTModelStudies["1"]=modelDir+"3bMix4b.rWbW2.v1.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1170.pkl"
#FvTModelStudies["2"]=modelDir+"3bMix4b.rWbW2.v2.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1177.pkl"
#FvTModelStudies["3"]=modelDir+"3bMix4b.rWbW2.v3.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch48_loss0.1175.pkl"
#FvTModelStudies["4"]=modelDir+"3bMix4b.rWbW2.v4.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1154.pkl"
#FvTModelStudies["5"]=modelDir+"3bMix4b.rWbW2.v5.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1169.pkl"
#FvTModelStudies["6"]=modelDir+"3bMix4b.rWbW2.v6.b0p6.combFvT_ResNet+multijetAttention_8_8_8_np1494_lr0.008_epochs50_stdscale_epoch50_loss0.1157.pkl"


#
# Add FvT for studies
#
if o.addFvTStudies:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_b0p6/pico*3b*JCM_b0p6.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_b0p6/pico*4b_b0p6.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_b0p6/pico*3b_wJCM_b0p6.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_b0p6/pico*4b_b0p6.h5" '
    
    #cmd = trainJOB+' -u -m '+FvTModelComb["Nominal"]+' -c FvT --cuda '+CUDA+' --updatePostFix _Nominal_comb '
    #cmd += ' -d '+dataFiles3b
    #cmd += ' --data4b '+dataFiles4b
    #cmd += ' -t '+ttFile3b
    #cmd += ' --ttbar4b '+ttFile4b
    #
    #cmds.append(cmd)
    #logs.append(outputDir+"/log_Add_FvT_Nominal_comb_b0p6")


    #for s in subSamples:
    for s in ["0"]:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '        

        for e in FvTModelStudies[s]:
            cmd = trainJOB+' -u  -m '+FvTModelStudies[s][e]+' -c FvT  --cuda '+CUDA  + ' --updatePostFix _'+mixedName+'_v'+s+'_comb_'+e
            cmd += ' -d '+dataFiles3b
            cmd += ' --data4b '+dataFiles4bMix
            cmd += ' --ttbar4b '+ttFile4bMix
            cmd += ' -t '+ttFile3b
            cmds.append(cmd)
            logs.append(outputDir+"/log_Add_FvT_studies"+mixedName+"_b0p6_v"+s)
    

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] add FvTStudies  Done" | sendmail '+o.email,doRun)




#
#  Make Closure Plots
#
if o.makeClosurePlots:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*_b0p6/pico*3b*JCM_b0p6.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_b0p6/pico*4b_b0p6.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*_b0p6/pico*3b_wJCM_b0p6.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_b0p6/pico*4b_b0p6.h5" '

    #weightPostFix = ""
    weightPostFix = "_comb"

    cmd = makeClosurePlots+"  --weightName mcPseudoTagWeight_Nominal"+weightPostFix+"  --FvTName FvT_Nominal"+weightPostFix+" -o "+outputDir+"/PlotsNominal_b0p6"+weightPostFix
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)
    logs.append(outputDir+"/log_Train_FvT_3bTo4b_b0p6"+weightPostFix)

    for s in subSamples:

        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*_b0p6_v'+s+'/picoAOD_'+mixedName+'*_b0p6_v'+s+'.h5" '        

        cmd = makeClosurePlots+ "  --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+weightPostFix+"  --FvTName FvT_"+mixedName+"_v"+s+weightPostFix+"  -o "+outputDir+"/Plots_"+mixedName+"_b0p6_v"+s+weightPostFix
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Train_FvT_3bMix4b_b0p6_v"+s+weightPostFix)

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
