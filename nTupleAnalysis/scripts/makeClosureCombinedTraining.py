import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4", help="Year or comma separated list of subsamples")
parser.add_option('--doTrain', action="store_true",      help="Should be obvious")
parser.add_option('--plotFvTFits', action="store_true",      help="Should be obvious")
parser.add_option('--addSvB', action="store_true",      help="Should be obvious")
parser.add_option('--addFvT', action="store_true",      help="Should be obvious")
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
SvBModel="ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1713_lr0.008_epochs40_stdscale_epoch40_loss0.2138.pkl"
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

    dataFiles3b = '"'+outputDir+'/*data201*/pico*3b*JCM.h5" ' 
    dataFiles4b = '"'+outputDirNom+'/*data201*/pico*4b.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*/pico*3b_wJCM.h5" '
    ttFile4b    = '"'+outputDirNom+'/*TT*201*/pico*4b.h5" '

    outName = "3bTo4bITER3"
    cmd = trainJOB+ " -c FvT -e 40 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight_Nominal " 
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    #HACK PUT Back cmds.append(cmd)
    #HACK PUT Back logs.append(outputDir+"/log_Train_FvT_3bTo4b")

    for s in subSamples:

        outName = (mixedName+"_v"+s+"ITER2").replace("_","")
        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*v'+s+'/picoAOD_'+mixedName+'*v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*v'+s+'/picoAOD_'+mixedName+'*v'+s+'.h5" '        

        cmd = trainJOB+ " -c FvT -e 50 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight_"+mixedName+"_v"+s 
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Train_FvT_3bMix4b_v"+s)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] FvT Training  Done" | sendmail '+o.email,doRun)


#modelDetails=FvT_ResNet+multijetAttention_9_9_9_np2070_lr0.008_epochs40_stdscale.log
#modelDetails="ITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale.log"


#
#  plotFvTFits
#
if o.plotFvTFits:
    cmds = []
    logs = []

    modelsLogFiles  = modelDir+"/3bTo4bITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale.log"
    
    modelsLogFiles += ","+modelDir+"/3bMix4brWbW2v0ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"/3bMix4brWbW2v1ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"/3bMix4brWbW2v2ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"/3bMix4brWbW2v3ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale.log"
    modelsLogFiles += ","+modelDir+"/3bMix4brWbW2v4ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale.log"

    
    #modelNames = "Nominal,3bMix4bV0,3bMix4bV1,3bMix4bV2,3bMix4bV3,3bMix4bV4"
    modelNames = "Nominal,3bMix4bV0,3bMix4brWbW2V1,3bMix4bV2,3bMix4bV3,3bMix4bV4"

    cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_FvTFits_"+mixedName+"_ITER2"
    cmd += " -i "+modelsLogFiles+" --names "+modelNames

    cmds.append(cmd)
    logs.append(outputDir+"/log_plotFvT_"+mixedName)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] FvT Training  Done" | sendmail '+o.email,doRun)


#
# Add SvB
#
if o.addSvB:
    cmds = []
    logs = []

    cmd = trainJOB+' -u -m '+SvBModel+' -c SvB --cuda '+CUDA  
    cmd += ' -d "'+outputDir+'/*data201*/pico*3b*h5"' 
    cmd += ' --data4b "'+outputDirNom+'/*data201*/pico*4b.h5"'  
    cmd += ' -t "'+outputDir+'/*TT*201*/pico*3b_wJCM*h5"' 
    cmd += ' --ttbar4b "'+outputDirNom+'/*TT*201*/pico*4b.h5"' 

    cmds.append(cmd)
    logs.append(outputDir+"/log_Add_SvB_Nominal")


    for s in subSamples:

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
        cmd += ' -d "'+outputDir3bMix4b+'/*data201*v'+s+'/picoAOD_'+mixedName+'*v'+s+'.h5"'   
        cmd += ' -t "'+outputDir3bMix4b+'/*TT*201*v'+s+'/picoAOD_'+mixedName+'*v'+s+'.h5"' 

        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_SvB_"+mixedName+"_v"+s)
        

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] FvT Training  Done" | sendmail '+o.email,doRun)


#
# Add FvT
#
FvTModel = {} 
FvTModel["Nominal"]=modelDir+"/3bTo4bITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch40_loss0.1553.pkl"
#FvTModel["0"]=modelDir+"/3bMix4brWbW2v0FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch20_loss0.1425.pkl"
#FvTModel["1"]=modelDir+"/3bMix4brWbW2v1FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch29_loss0.1420.pkl"
#FvTModel["2"]=modelDir+"/3bMix4brWbW2v2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch18_loss0.1426.pkl"
#FvTModel["3"]=modelDir+"/3bMix4brWbW2v3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch28_loss0.1416.pkl"
#FvTModel["4"]=modelDir+"/3bMix4brWbW2v4FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch22_loss0.1420.pkl"


FvTModel["0"]=modelDir+"/3bMix4brWbW2v0ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale_epoch33_loss0.1426.pkl"
FvTModel["1"]=modelDir+"/3bMix4brWbW2v1ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale_epoch50_loss0.1423.pkl"
FvTModel["2"]=modelDir+"/3bMix4brWbW2v2ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale_epoch35_loss0.1431.pkl"
FvTModel["3"]=modelDir+"/3bMix4brWbW2v3ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale_epoch50_loss0.1418.pkl"
FvTModel["4"]=modelDir+"/3bMix4brWbW2v4ITER2FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.002_epochs50_stdscale_epoch50_loss0.1421.pkl"




#
# Add FvT
#
if o.addFvT:
    cmds = []
    logs = []
    
    cmd = trainJOB+' -u -m '+FvTModel["Nominal"]+' -c FvT --cuda '+CUDA+' --updatePostFix _Nominal '
    cmd += ' -d "'+outputDir+'/*data201*/pico*3b*h5"' 
    cmd += ' --data4b "'+outputDirNom+'/*data201*/pico*4b.h5"'  
    cmd += ' -t "'+outputDir+'/*TT*201*/pico*3b_wJCM*h5"' 
    cmd += ' --ttbar4b "'+outputDirNom+'/*TT*201*/pico*4b.h5"' 

    cmds.append(cmd)
    logs.append(outputDir+"/log_Add_FvT_Nominal")


    for s in subSamples:

        cmd = trainJOB+' -u  -m '+FvTModel[s]+' -c FvT  --cuda '+CUDA  + ' --updatePostFix _'+mixedName+'_v'+s
        cmd += ' -d "'+outputDir+'/*data201*/pico*3b*h5"' 
        cmd += ' --data4b "'+outputDir3bMix4b+'/*data201*v'+s+'/picoAOD_'+mixedName+'*v'+s+'.h5"'   
        cmd += ' --ttbar4b "'+outputDir3bMix4b+'/*TT*201*v'+s+'/picoAOD_'+mixedName+'*v'+s+'.h5"' 
        cmd += ' -t "'+outputDir+'/*TT*201*/pico*3b_wJCM*h5"' 
        cmds.append(cmd)
        logs.append(outputDir+"/log_Add_FvT_"+mixedName+"_v"+s)


    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] add FvT  Done" | sendmail '+o.email,doRun)


#
#  Make Closure Plots
#
if o.makeClosurePlots:
    cmds = []
    logs = []

    dataFiles3b = '"'+outputDir+'/*data201*/pico*3b*JCM.h5" ' 
    dataFiles4b = '"'+outputDirNom+'/*data201*/pico*4b.h5" '
    ttFile3b    = '"'+outputDir+'/*TT*201*/pico*3b_wJCM.h5" '
    ttFile4b    = '"'+outputDirNom+'/*TT*201*/pico*4b.h5" '

    cmd = makeClosurePlots+"  --weightName mcPseudoTagWeight_Nominal  --FvTName FvT_Nominal -o "+outputDir+"/PlotsNominal" 
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    #HACK PUT Back cmds.append(cmd)
    #HACK PUT Back logs.append(outputDir+"/log_Train_FvT_3bTo4b")

    for s in subSamples:

        outName = mixedName+"_v"+s
        dataFiles4bMix = '"'+outputDir3bMix4b+'/*data201*v'+s+'/picoAOD_'+mixedName+'*v'+s+'.h5" '
        ttFile4bMix    = '"'+outputDir3bMix4b+'/*TT*201*v'+s+'/picoAOD_'+mixedName+'*v'+s+'.h5" '        

        cmd = makeClosurePlots+ "  --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+"  --FvTName FvT_"+mixedName+"_v"+s+"  -o "+outputDir+"/Plots_"+mixedName+"_v"+s
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4bMix

        cmds.append(cmd)
        logs.append(outputDir+"/log_Train_FvT_3bMix4b_v"+s)

    babySit(cmds, doRun, logFiles=logs)
    if o.email: execute('echo "Subject: [makeClosureCombinedTraining] FvT Training  Done" | sendmail '+o.email,doRun)


#
#  Make Closure Plots
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
