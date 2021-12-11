import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('--doTrainDataVsTT', action="store_true",      help="Should be obvious")
parser.add_option('--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_option('--trainOffset', default="1", help='training offset.')
parser.add_option('--plotDvT', action="store_true",      help="Should be obvious")
parser.add_option('--writeOutDvT', action="store_true",      help="Should be obvious")
parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--addvAllWeights', action="store_true",      help="Should be obvious")
parser.add_option('--doTrainFvT', action="store_true",      help="Should be obvious")
parser.add_option('--doFineTuneFvT', action="store_true",      help="Should be obvious")
parser.add_option('--addSvB', action="store_true",      help="Should be obvious")
parser.add_option('--addFvT', action="store_true",      help="Should be obvious")
parser.add_option('--addFvTOneOffset', action="store_true",      help="Should be obvious")
parser.add_option('--addFvTAllOffsets', action="store_true",      help="Should be obvious")
parser.add_option('--addSvB_MA', action="store_true",      help="Should be obvious")
parser.add_option('--addSvBAllMixedSamples', action="store_true",      help="Should be obvious")
parser.add_option('--addSvBSignalMixData', action="store_true",      help="Should be obvious")
parser.add_option('--addSvBMixedSignalAndData', action="store_true",      help="Should be obvious")
parser.add_option('--addSvBMixed4bSignal', action="store_true",      help="Should be obvious")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9", help="Year or comma separated list of subsamples")
parser.add_option('--makeClosurePlots', action="store_true",      help="Should be obvious")
parser.add_option('--convertH5ToH5', action="store_true",      help="Should be obvious")
parser.add_option('--convertH5ToH5AllMixedSamples', action="store_true",      help="Should be obvious")
parser.add_option('--convertH5ToH5SignalMixData', action="store_true",      help="Should be obvious")
parser.add_option('--convertH5ToH5MixedSignalAndData', action="store_true",      help="Should be obvious")
parser.add_option('--convertH5ToH5Mixed4bSignal', action="store_true",      help="Should be obvious")

parser.add_option('--plotFvTFits', action="store_true",      help="Should be obvious")
parser.add_option('--debugFvT', action="store_true",      help="Should be obvious")

o, a = parser.parse_args()

doRun = o.execute

CUDA=str(o.cuda)

subSamples = o.subSamples.split(",")
mixedName = o.mixedName

outputDir="closureTests/ULTrig/"

signalSamples = ["ZZ4b","ZH4b","ggZH4b"]


trainJOB='python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py '
plotDvT='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py'
makeClosurePlots='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py'
convertH5ToH5 ='python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py'
copyColumn="python  ZZ4b/nTupleAnalysis/scripts/copyColumn.py "
#    python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT4        -i  "closureTests/UL//*201*/picoAOD_4b.h5"  --var DvT4,DvT4_pt4


#
# Train
#   (with GPU enviorment)
if o.doTrainDataVsTT:
    cmds = []


    dataFiles3b = '"'+outputDir+'/data201*/picoAOD_3b.h5" '
    ttFiles3b   = '"'+outputDir+'/*TT*201*/picoAOD_3b.h5" '

    outName = "3b"
    cmd = trainJOB+ " -c DvT3 -e 20 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight"+"  --trainOffset "+o.trainOffset+" --train   "#--updatePostFix _3b "
    cmd += " -d "+dataFiles3b + " -t " + ttFiles3b 

    cmds.append(cmd)

    dataFiles4b = '"'+outputDir+'/data201*/picoAOD_4b.h5" '
    ttFiles4b   = '"'+outputDir+'/*TT*201*/picoAOD_4b.h5" '

    outName = "4b"
    cmd = trainJOB+ " -c DvT4 -e 20 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight"+"  --trainOffset "+o.trainOffset+" --train  "
    cmd += " -d "+dataFiles4b + " -t " + ttFiles4b 

    cmds.append(cmd)

    babySit(cmds, doRun)



#
# Plot
#   (with GPU enviorment)
if o.plotDvT:
    cmds = []


    dataFiles3b = '"'+outputDir+'/data201*/picoAOD_3b.h5" '
    ttFiles3b   = '"'+outputDir+'/*TT*201*/picoAOD_3b.h5" '

    cmd = plotDvT+ "  -o "+outputDir+"/plots_DvT3" + "  --weightName mcPseudoTagWeight  --DvTName DvT3 "
    cmd += " -d "+dataFiles3b + " -t " + ttFiles3b 

    cmds.append(cmd)


    dataFiles4b = '"'+outputDir+'/data201*/picoAOD_4b.h5" '
    ttFiles4b   = '"'+outputDir+'/*TT*201*/picoAOD_4b.h5" '

    cmd = plotDvT+ "  -o "+outputDir+"/plots_DvT4" + "  --weightName mcPseudoTagWeight  --DvTName DvT4 "
    cmd += " -d "+dataFiles4b + " -t " + ttFiles4b 

    cmds.append(cmd)

    babySit(cmds, doRun)



#
# Write Out FvT
#   (with GPU enviorment)
if o.writeOutDvT:
    cmds = []

    modelDir = "ZZ4b/nTupleAnalysis/pytorchModels/"

    dataFiles3b = '"'+outputDir+'/data201*/picoAOD_3b.h5" '
    ttFiles3b   = '"'+outputDir+'/*TT*201*/picoAOD_3b.h5" '

    DvT3Models =      modelDir+"3bDvT3_HCR+attention_14_np2684_lr0.01_epochs20_offset0_epoch20.pkl"
    DvT3Models += ","+modelDir+"3bDvT3_HCR+attention_14_np2684_lr0.01_epochs20_offset1_epoch20.pkl"
    DvT3Models += ","+modelDir+"3bDvT3_HCR+attention_14_np2684_lr0.01_epochs20_offset2_epoch20.pkl"
    
    cmd = trainJOB+ " -c DvT3   --update   -m "+DvT3Models
    cmd += " -d "+dataFiles3b +  " -t " + ttFiles3b
    cmd += ' --writeWeightFile '
    cmd += ' --weightFilePostFix DvT3 '
    cmds.append(cmd)

    dataFiles4b = '"'+outputDir+'/data201*/picoAOD_4b.h5" '
    ttFiles4b   = '"'+outputDir+'/*TT*201*/picoAOD_4b.h5" '

    DvT4Models =      modelDir+"4bDvT4_HCR+attention_14_np2684_lr0.01_epochs20_offset0_epoch20.pkl"
    DvT4Models += ","+modelDir+"4bDvT4_HCR+attention_14_np2684_lr0.01_epochs20_offset1_epoch20.pkl"
    DvT4Models += ","+modelDir+"4bDvT4_HCR+attention_14_np2684_lr0.01_epochs20_offset2_epoch20.pkl"

    cmd = trainJOB+ " -c DvT4   --update   -m "+DvT4Models
    cmd += " -d "+dataFiles4b +  " -t " + ttFiles4b
    cmd += ' --writeWeightFile '
    cmd += ' --weightFilePostFix DvT4 '
    cmds.append(cmd)

    babySit(cmds, doRun)



#
# Add mcPseudoTagWeights for vAll Training
#   (with GPU enviorment)
if o.addvAllWeights:
    cmds = []


    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b_wTrigW/picoAOD_3b_wJCM.h5" '
    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData_wTrigW/picoAOD_4b_wJCM.h5" '

    for fileList in [dataFiles3b,ttFile3b,ttFile4b_noPS]:
        cmd = copyColumn + ' -i  '+fileList +'     --target mcPseudoTagWeight_'+mixedName+'_v4 --dest mcPseudoTagWeight_'+mixedName+'_vAll   '
        cmds.append(cmd)

        cmd = copyColumn + ' -i  '+fileList +'     --target pseudoTagWeight_'+mixedName+'_v4   --dest pseudoTagWeight_'+mixedName+'_vAll     '
        cmds.append(cmd)

    for s in subSamples:

        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '
        cmd = copyColumn + ' -i  '+dataFiles4bMix +'     --target mcPseudoTagWeight_'+mixedName+'_v'+s+' --dest mcPseudoTagWeight_'+mixedName+'_vAll   '
        cmds.append(cmd)

        cmd = copyColumn + ' -i  '+dataFiles4bMix +'     --target pseudoTagWeight_'+mixedName+'_v'+s+'   --dest pseudoTagWeight_'+mixedName+'_vAll     '
        cmds.append(cmd)



    babySit(cmds, doRun)




#
# Train
#   (with GPU enviorment)
if o.doTrainFvT:
    cmds = []

    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b_wTrigW/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b_wTrigW/picoAOD_4b_wJCM.h5" '

    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData_wTrigW/picoAOD_4b_wJCM.h5" '
    
    outNamePostFix = ".nf12"

    outName = "3bTo4b"+outNamePostFix
    cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_Nominal"+"  --trainOffset "+str(o.trainOffset)+" --train  "
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b
    # --updatePostFix _Nominal

    cmds.append(cmd)

    for s in subSamples:

        outName = (mixedName+"_v"+s+outNamePostFix).replace("_",".")
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+"  --trainOffset "+str(o.trainOffset)+" --train " #  --updatePostFix _"+mixedName+"_v"+s
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS

        cmds.append(cmd)


    outName = (mixedName+"_vAll"+outNamePostFix).replace("_",".")
    dataFiles4bMixAll = '"'+outputDir+'/*mixed201*_'+mixedName+'_v*/picoAOD_'+mixedName+'*_v?.h5" '

    cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_"+mixedName+"_vAll  --trainOffset "+str(o.trainOffset)+" --train " #--update  --updatePostFix _"+mixedName+"_vAll"
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMixAll + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS
    cmd += " --data4bWeightOverwrite  0.1"
    cmds.append(cmd)


    babySit(cmds, doRun)




#
# Train
#   (with GPU enviorment)
if o.doFineTuneFvT:
    cmds = []

    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b/picoAOD_4b_wJCM.h5" '

    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData/picoAOD_4b_wJCM.h5" '

    modelDir = "ZZ4b/nTupleAnalysis/pytorchModels/"

    FvTModel = modelDir+"3bTo4b.FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(o.trainOffset)+"_epoch20.pkl"
    
    outName = "3bTo4b."
    cmd = trainJOB+ " -c FvT -e 0 --finetune -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_Nominal"+"  --trainOffset "+str(o.trainOffset)+" --train   --updatePostFix _Nominal "
    cmd += " -m "+FvTModel
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)

    for s in subSamples:

        outName = (mixedName+"_v"+s).replace("_",".")
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        FvTModel =      modelDir+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(o.trainOffset)+"_epoch20.pkl"

        cmd = trainJOB+ " -c FvT -e 0 --finetune -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+"  --trainOffset "+str(o.trainOffset)+" --train   --updatePostFix _"+mixedName+"_v"+s
        cmd += " -m "+FvTModel
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS

        cmds.append(cmd)


    outName = (mixedName+"_vAll").replace("_",".")
    dataFiles4bMixAll = '"'+outputDir+'/*mixed201*_'+mixedName+'_v*/picoAOD_'+mixedName+'*_v?.h5" '
    FvTModel =      modelDir+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(o.trainOffset)+"_epoch20.pkl"

    cmd = trainJOB+ " -c FvT -e 0 --finetune -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_"+mixedName+"_vAll  --trainOffset "+str(o.trainOffset)+" --train   --updatePostFix _"+mixedName+"_vAll"
    cmd += " -m "+FvTModel
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMixAll + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS
    cmd += " --data4bWeightOverwrite  0.1"
    cmds.append(cmd)


    babySit(cmds, doRun)





#
# Add SvB
#
if o.addSvB:
    cmds = []


    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b_wTrigW/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b_wTrigW/picoAOD_4b_wJCM.h5" '

    SvBModel = "ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_14_np2013_lr0.01_epochs20_offset0_epoch20.pkl"


    cmd = trainJOB+' -u -m '+SvBModel+' -c SvB --cuda '+CUDA  
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b
    cmd += ' --writeWeightFile '
    cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
    #cmd += ' --weightFilePostFix FvTWeights '
    cmd += ' --weightFilePreFix /home/scratch/jalison/ '

    cmds.append(cmd)


    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData_wTrigW/picoAOD_4b_wJCM.h5" '

    cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
    cmd += ' -t '+ttFile4b_noPS
    cmd += ' --writeWeightFile '
    cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
    #cmd += ' --weightFilePostFix FvTWeights '
    cmd += ' --weightFilePreFix /home/scratch/jalison/ '

    cmds.append(cmd)


    for s in subSamples:
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
        cmd += ' -d '+dataFiles4bMix
        cmd += ' --writeWeightFile '
        #cmd += ' --weightFilePostFix FvTWeights '
        cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
        cmd += ' --weightFilePreFix /home/scratch/jalison/ '

        cmds.append(cmd)
        

    babySit(cmds, doRun)



#
# Add SvB
#
if o.addSvB_MA:
    cmds = []


    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b_wTrigW/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b_wTrigW/picoAOD_4b_wJCM.h5" '

    SvBModel = "ZZ4b/nTupleAnalysis/pytorchModels/SvB_MA_HCR+attention_14_np2714_lr0.01_epochs20_offset0_epoch20.pkl "


    cmd = trainJOB+' -u -m '+SvBModel+' -c SvB_MA --cuda '+CUDA  
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b
    cmd += ' --writeWeightFile '
    #cmd += ' --weightFilePostFix FvTWeights '
    cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
    cmd += ' --weightFilePreFix /home/scratch/jalison/ '
    cmds.append(cmd)


    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData_wTrigW/picoAOD_4b_wJCM.h5" '

    cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB_MA  --cuda '+CUDA  
    cmd += ' -t '+ttFile4b_noPS
    cmd += ' --writeWeightFile '
    #cmd += ' --weightFilePostFix FvTWeights '
    cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
    cmd += ' --weightFilePreFix /home/scratch/jalison/ '
    cmds.append(cmd)


    for s in subSamples:
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB_MA  --cuda '+CUDA  
        cmd += ' -d '+dataFiles4bMix
        cmd += ' --writeWeightFile '
        #cmd += ' --weightFilePostFix FvTWeights '
        cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
        cmd += ' --weightFilePreFix /home/scratch/jalison/ '
        cmds.append(cmd)
        

    babySit(cmds, doRun)



#
# Write Out FvT
#   (with GPU enviorment)
if o.addFvT:
    cmds = []

    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b_wTrigW/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b_wTrigW/picoAOD_4b_wJCM.h5" '

    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData_wTrigW/picoAOD_4b_wJCM.h5" '

    modelDir = "ZZ4b/nTupleAnalysis/pytorchModels/"
    FvTModels =      modelDir+"3bTo4b.nf12FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset0_epoch20.pkl"
    FvTModels += ","+modelDir+"3bTo4b.nf12FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset1_epoch20.pkl"
    FvTModels += ","+modelDir+"3bTo4b.nf12FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset2_epoch20.pkl"
    
    cmd = trainJOB+ " -c FvT   --update  --updatePostFix _Nominal  -m "+FvTModels
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b
    cmd += ' --writeWeightFile '
    cmd += ' --weightFilePostFix weights_nf12 '
    cmd += ' --weightFilePreFix /home/scratch/jalison/ '

    cmds.append(cmd)

    for s in subSamples:

        outName = (mixedName+"_v"+s+".nf12").replace("_",".")        

        FvTModels =      modelDir+outName+"FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset0_epoch20.pkl"
        FvTModels += ","+modelDir+outName+"FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset1_epoch20.pkl"
        FvTModels += ","+modelDir+outName+"FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset2_epoch20.pkl"

        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = trainJOB+ " -c FvT  --update  --updatePostFix _"+mixedName+"_v"+s + " -m "+FvTModels
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS
        cmd += ' --writeWeightFile '
        cmd += ' --weightFilePostFix weights_nf12 '
        cmd += ' --weightFilePreFix /home/scratch/jalison/ '
        cmds.append(cmd)


    outName = (mixedName+"_vAll.nf12").replace("_",".")
    FvTModels =      modelDir+outName+"FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset0_epoch20.pkl"
    FvTModels += ","+modelDir+outName+"FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset1_epoch20.pkl"
    FvTModels += ","+modelDir+outName+"FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset2_epoch20.pkl"

    dataFiles4bMixAll = '"'+outputDir+'/*mixed201*_'+mixedName+'_v*/picoAOD_'+mixedName+'*_v?.h5" '

    cmd = trainJOB+ " -c FvT  --update  --updatePostFix _"+mixedName+"_vAll"+ " -m "+FvTModels
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMixAll + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS
    cmd += ' --writeWeightFile '
    cmd += ' --weightFilePostFix weights_nf12 '
    cmd += ' --weightFilePreFix /home/scratch/jalison/ '
    cmds.append(cmd)


    babySit(cmds, doRun)



#
# Write Out FvT
#   (with GPU enviorment)
if o.addFvTOneOffset:
    cmds = []

    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b_wTrigW/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b_wTrigW/picoAOD_4b_wJCM.h5" '

    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData_wTrigW/picoAOD_4b_wJCM.h5" '

    modelDir = "ZZ4b/nTupleAnalysis/pytorchModels/"
    
    #FvTModels =      modelDir+"3bTo4b.noSRFvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+o.trainOffset+"_epoch20.pkl"
    FvTModels =      modelDir+"3bTo4b.nf12FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset"+o.trainOffset+"_epoch20.pkl"
    
    cmd = trainJOB+ " -c FvT   --update  --updatePostFix _Nominal  -m "+FvTModels+ "  --cuda "+CUDA  
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b
    cmd += ' --writeWeightFile '
    cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
    cmd += ' --weightFilePreFix /home/scratch/jalison/ '
    
    cmds.append(cmd)

    for s in subSamples:

        outName = (mixedName+"_v"+s+".nf12").replace("_",".")

        #FvTModels =      modelDir+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+o.trainOffset+"_epoch20.pkl"
        FvTModels =      modelDir+outName+"FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset"+o.trainOffset+"_epoch20.pkl"

        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = trainJOB+ " -c FvT  --update  --updatePostFix _"+mixedName+"_v"+s + " -m "+FvTModels+ "  --cuda "+CUDA  
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS
        cmd += ' --writeWeightFile '
        cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
        cmd += ' --weightFilePreFix /home/scratch/jalison/ '
        cmds.append(cmd)

    outName = (mixedName+"_vAll.nf12").replace("_",".")
    #FvTModels =      modelDir+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+o.trainOffset+"_epoch20.pkl"
    FvTModels =      modelDir+outName+"FvT_HCR+attention_12_np2076_lr0.01_epochs20_offset"+o.trainOffset+"_epoch20.pkl"

    dataFiles4bMixAll = '"'+outputDir+'/*mixed201*_'+mixedName+'_v*/picoAOD_'+mixedName+'*_v?.h5" '

    cmd = trainJOB+ " -c FvT  --update  --updatePostFix _"+mixedName+"_vAll"+ " -m "+FvTModels+ "  --cuda "+CUDA  
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMixAll + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS
    cmd += ' --writeWeightFile '
    cmd += ' --weightFilePostFix weights_nf12_offset'+o.trainOffset
    cmd += ' --weightFilePreFix /home/scratch/jalison/ '
    cmds.append(cmd)




    babySit(cmds, doRun)



#
# Write Out FvT
#   (with GPU enviorment)
if o.addFvTAllOffsets:
    cmds = []

    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b/picoAOD_4b_wJCM.h5" '

    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData/picoAOD_4b_wJCM.h5" '

    modelDir = "ZZ4b/nTupleAnalysis/pytorchModels/"
    
    #FvTModels =      modelDir+"3bTo4b.FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+o.trainOffset+"_epoch20.pkl"
    #
    #cmd = trainJOB+ " -c FvT   --update  --updatePostFix _Nominal  -m "+FvTModels
    #cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b
    #cmd += ' --writeWeightFile '
    #cmd += ' --weightFilePostFix weights_offset'+o.trainOffset
    #cmd += ' --weightFile '
    #
    #cmds.append(cmd)

    for s in subSamples:

        outName = (mixedName+"_v"+s).replace("_",".")
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        FvTModels =      modelDir+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset0_epoch20.pkl"
        FvTModels += ","+modelDir+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset1_epoch20.pkl"
        FvTModels += ","+modelDir+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset2_epoch20.pkl"

        cmd = trainJOB+ " -c FvT  --update  --updatePostFix _"+mixedName+"_v"+s + " -m "+FvTModels
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS
        cmd += ' --writeWeightFile '
        cmd += ' --weightFilePostFix weights_All'
        cmds.append(cmd)


        for os in ["0","1","2"]:
            FvTModels =      modelDir+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+o.trainOffset+"_epoch20.pkl"

            cmd = trainJOB+ " -c FvT  --update  --updatePostFix _"+mixedName+"_v"+s+"_os"+os + " -m "+FvTModels
            cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS
            cmd += ' --writeWeightFile '
            cmd += ' --weightFilePostFix weights_All'
            cmds.append(cmd)


    babySit(cmds, doRun)




#
# Add SvB All mixed samples
#
if o.addSvBAllMixedSamples:

    cmds = []

    SvBModel = "ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_14_np2160_lr0.01_epochs20_offset1_epoch20.pkl "

    for mixedName in ["3bMix4b","3bDvTMix4b",
                      "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:


        for s in subSamples:
            dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

            cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
            cmd += ' -d '+dataFiles4bMix

            cmds.append(cmd)
        

    babySit(cmds, doRun)


#
# Add SvB
#
if o.addSvBSignalMixData:

    cmds = []

    SvBModel = "ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_14_np2160_lr0.01_epochs20_offset1_epoch20.pkl "

    for sig in signalSamples:
        sigFiles = '"'+outputDir+'/*'+sig+'*_3bSubSampled/picoAOD_'+mixedName+'.h5" '

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
        cmd += ' -d '+sigFiles

        cmds.append(cmd)
        

    babySit(cmds, doRun)


#
# Add SvB
#
if o.addSvBMixedSignalAndData:

    cmds = []

    SvBModel = "ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_14_np2160_lr0.01_epochs20_offset1_epoch20.pkl "

    for sig in signalSamples:
        sigFiles = '"'+outputDir+'/*'+sig+'*_3bSubSampled/picoAOD_'+mixedName+'.h5" '

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
        cmd += ' -d '+sigFiles

        cmds.append(cmd)


    dataFiles = '"'+outputDir+'/data*_v*/picoAOD_'+mixedName+'_v*.h5" '
    cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
    cmd += ' -d '+dataFiles

    cmds.append(cmd)
        

    babySit(cmds, doRun)


#
# Add SvB
#
if o.addSvBMixed4bSignal:

    cmds = []

    SvBModel = "ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_14_np2160_lr0.01_epochs20_offset1_epoch20.pkl "

    for sig in signalSamples:
        sigFiles = '"'+outputDir+'/*'+sig+'201?/picoAOD_'+mixedName+'.h5" '

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
        cmd += ' -d '+sigFiles

        cmds.append(cmd)

        

    babySit(cmds, doRun)




#
#  Make Closure Plots
#
if o.makeClosurePlots:
    cmds = []


    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b/picoAOD_4b_wJCM.h5" '

    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData/picoAOD_4b_wJCM.h5" '


    cmd = makeClosurePlots+"  --weightName mcPseudoTagWeight_Nominal --FvTName FvT_Nominal   -o "+outputDir+"/PlotsExtended_Nominal"
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)
    
    for s in subSamples:
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = makeClosurePlots+ "  --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+"  --FvTName FvT_"+mixedName+"_v"+s+"  -o "+outputDir+"/PlotsExtended_"+mixedName+"_v"+s
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS

        cmds.append(cmd)

    babySit(cmds, doRun)





#
#  plotFvTFits
#
if o.plotFvTFits:
    cmds = []
    logs = []

    offset = o.trainOffset
    modelDir="ZZ4b/nTupleAnalysis/pytorchModels/"

    outName = "oldLR"
    #outName = ""
    modelsLogFiles = modelDir+"3bTo4b."+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames = "Nominal"

    modelsLogFiles += ","+modelDir+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames += ",PBRFit"

    if outName: outName = "."+outName

    modelsLogFiles += ","+modelDir+mixedName+".vAll"+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames     += ",vAll"


    for s in subSamples:
        #modelsLogFiles += ","+modelDir+mixedName+".v"+s+"FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
        modelsLogFiles += ","+modelDir+mixedName+".v"+s+outName+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
        modelNames     += ",v"+s

    #modelNames = "Nominal,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9"
    #modelNames = "Nominal,3bMix4bv0,3bMix4br,3bMix4bV2,3bMix4bV3,3bMix4bV4"

    cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_FvTFits_e20"+outName.replace(".","_")+"_offset"+str(offset)+"_"+mixedName
    cmd += " -i "+modelsLogFiles+" --names "+modelNames

    #modelsLogFiles =  modelDir+"FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
    #modelsLogFiles += ","+modelDir+"3bTo4b.FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
    #modelNames = "PBR,Auton"
    #
    #cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_Debug_offset"+str(offset)
    #cmd += " -i "+modelsLogFiles+" --names "+modelNames






    cmds.append(cmd)

    babySit(cmds, doRun)




#
#  plotFvTFits
#
if o.debugFvT:
    cmds = []
    logs = []

    offset = o.trainOffset
    modelDir="ZZ4b/nTupleAnalysis/pytorchModels/"

    outName = "oldLR"
    #outName = ""
    modelsLogFiles = modelDir+"3bTo4b.FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames = "Nominal"

    modelsLogFiles += ","+modelDir+"3bTo4b.oldLRFvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames += ",Nominal.oldLR"



    modelsLogFiles += ","+modelDir+mixedName+".vAllFvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames     += ",vAll"

    modelsLogFiles += ","+modelDir+mixedName+".vAll.oldLRFvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames     += ",vAll.oldLR"



    for s in ["0","2"]:
        #modelsLogFiles += ","+modelDir+mixedName+".v"+s+"FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
        modelsLogFiles += ","+modelDir+mixedName+".v"+s+"FvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
        modelNames     += ",v"+s

        modelsLogFiles += ","+modelDir+mixedName+".v"+s+".oldLRFvT_HCR+attention_14_np2714_lr0.01_epochs20_offset"+str(offset)+".log"
        modelNames     += ",v"+s+".oldLR"


    #modelNames = "Nominal,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9"
    #modelNames = "Nominal,3bMix4bv0,3bMix4br,3bMix4bV2,3bMix4bV3,3bMix4bV4"

    cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_DebugFvTFits_offset"+str(offset)+"_"+mixedName
    cmd += " -i "+modelsLogFiles+" --names "+modelNames

    #modelsLogFiles =  modelDir+"FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
    #modelsLogFiles += ","+modelDir+"3bTo4b.FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
    #modelNames = "PBR,Auton"
    #
    #cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_Debug_offset"+str(offset)
    #cmd += " -i "+modelsLogFiles+" --names "+modelNames

    cmds.append(cmd)

    babySit(cmds, doRun)
