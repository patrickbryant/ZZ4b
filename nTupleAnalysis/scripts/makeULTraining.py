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
parser.add_option('--mixedName',                        default="3bMix4b", help="Year or comma separated list of subsamples")
parser.add_option('--doTrainFvT', action="store_true",      help="Should be obvious")
parser.add_option('--addSvB', action="store_true",      help="Should be obvious")
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

o, a = parser.parse_args()

doRun = o.execute

CUDA=str(o.cuda)

subSamples = o.subSamples.split(",")
mixedName = o.mixedName

outputDir="closureTests/UL/"

signalSamples = ["ZZ4b","ZH4b","ggZH4b"]


trainJOB='python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py '
plotDvT='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py'
makeClosurePlots='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py'
convertH5ToH5 ='python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py'
#    python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT4        -i  "closureTests/UL//*201*/picoAOD_4b.h5"  --var DvT4,DvT4_pt4

#
# Train
#   (with GPU enviorment)
if o.doTrainDataVsTT:
    cmds = []


    dataFiles3b = '"'+outputDir+'/data201*/picoAOD_3b.h5" '
    ttFiles3b   = '"'+outputDir+'/*TT*201*/picoAOD_3b.h5" '

    outName = "3b"
    cmd = trainJOB+ " -c DvT3 -e 20 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight"+"  --trainOffset "+o.trainOffset+" --train --update  "#--updatePostFix _3b "
    cmd += " -d "+dataFiles3b + " -t " + ttFiles3b 

    cmds.append(cmd)

    dataFiles4b = '"'+outputDir+'/data201*/picoAOD_4b.h5" '
    ttFiles4b   = '"'+outputDir+'/*TT*201*/picoAOD_4b.h5" '

    outName = "4b"
    cmd = trainJOB+ " -c DvT4 -e 20 -o "+outName+" --cuda "+CUDA+" --weightName mcPseudoTagWeight"+"  --trainOffset "+o.trainOffset+" --train --update "
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
# Train
#   (with GPU enviorment)
if o.doTrainFvT:
    cmds = []

    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b/picoAOD_4b_wJCM.h5" '

    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData/picoAOD_4b_wJCM.h5" '


    outName = "3bTo4b.v2."
    cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_Nominal"+"  --trainOffset "+str(o.trainOffset)+" --train --update  --updatePostFix _Nominal "
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)

    for s in subSamples:

        outName = (mixedName+"_v"+s).replace("_",".")
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+"  --trainOffset "+str(o.trainOffset)+" --train --update  --updatePostFix _"+mixedName+"_v"+s
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS

        cmds.append(cmd)


    outName = (mixedName+"_vAll").replace("_",".")
    dataFiles4bMixAll = '"'+outputDir+'/*mixed201*_'+mixedName+'_v*/picoAOD_'+mixedName+'*_v*.h5" '

    cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_"+mixedName+"_v4  --trainOffset "+str(o.trainOffset)+" --train --update  --updatePostFix _"+mixedName+"_vAll"
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
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b/picoAOD_4b_wJCM.h5" '

    SvBModel = "ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_14_np2160_lr0.01_epochs20_offset1_epoch20.pkl "

    cmd = trainJOB+' -u -m '+SvBModel+' -c SvB --cuda '+CUDA  
    cmd += ' -d '+dataFiles3b
    cmd += ' --data4b '+dataFiles4b
    cmd += ' -t '+ttFile3b
    cmd += ' --ttbar4b '+ttFile4b

    cmds.append(cmd)


    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData/picoAOD_4b_wJCM.h5" '

    cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
    cmd += ' -t '+ttFile4b_noPS

    cmds.append(cmd)


    for s in subSamples:
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = trainJOB+' -u  -m '+SvBModel+' -c SvB  --cuda '+CUDA  
        cmd += ' -d '+dataFiles4bMix

        cmds.append(cmd)
        

    babySit(cmds, doRun)



#
# Add SvB
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


    cmd = makeClosurePlots+"  --weightName mcPseudoTagWeight_Nominal --FvTName FvT_Nominal   -o "+outputDir+"/PlotsNominal"
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)
    
    for s in subSamples:
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = makeClosurePlots+ "  --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+"  --FvTName FvT_"+mixedName+"_v"+s+"  -o "+outputDir+"/Plots_Umixed_"+mixedName+"_v"+s
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS

        cmds.append(cmd)

    babySit(cmds, doRun)



if o.convertH5ToH5:

    cmds = []

    dataFiles3b = '"'+outputDir+'/*data201*_3b/picoAOD_3b_wJCM.h5" ' 
    dataFiles4b = '"'+outputDir+'/*data201*_4b/picoAOD_4b_wJCM.h5" ' 
    ttFile3b    = '"'+outputDir+'/*TT*201*_3b/picoAOD_3b_wJCM.h5" '
    ttFile4b    = '"'+outputDir+'/*TT*201*_4b/picoAOD_4b_wJCM.h5" '

    ttFile4b_noPS    = '"'+outputDir+'/*TT*201*_4b_noPSData/picoAOD_4b_wJCM.h5" '

    def getFvTList(fvtName):
        if fvtName:
            return ["FvT"+fvtName,
                    "FvT"+fvtName+"_pd3",
                    "FvT"+fvtName+"_pt4",
                    "FvT"+fvtName+"_pt3"
                    ]

        else:
            return ["FvT"+fvtName,
                    "FvT"+fvtName+"_pd4",
                    "FvT"+fvtName+"_pd3",
                    "FvT"+fvtName+"_pt4",
                    "FvT"+fvtName+"_pt3",
                    "FvT"+fvtName+"_q_1234",
                    "FvT"+fvtName+"_q_1324",
                    "FvT"+fvtName+"_q_1423"]



    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
        #"SvB_MA_ps",
        #"SvB_MA_pzz",
        #"SvB_MA_pzh",
        #"SvB_MA_ptt",
        #"SvB_MA_q_1234",
        #"SvB_MA_q_1324",
        #"SvB_MA_q_1423",
        ]





    #
    # 3b
    #
    varList3b = list(varListSvB) + getFvTList("_Nominal")

    for s in subSamples:
        fvtName = "_"+mixedName+"_v"+s
        varList3b += getFvTList(fvtName)

    varList3b += getFvTList("_"+mixedName+"_vAll")

    cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+dataFiles3b + " --var "+",".join(varList3b))
    cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+ttFile3b    + " --var "+",".join(varList3b))

    #
    #  4b
    #
    varList4b = list(varListSvB) + getFvTList("_Nominal")
    cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+dataFiles4b + " --var "+",".join(varList4b))
    cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+ttFile4b    + " --var "+",".join(varList4b))


    #
    #  Mixed
    #
    varList4b_noPS = list(varListSvB )

    for s in subSamples:
        fvtName = "_"+mixedName+"_v"+s
        varList4b_noPS += getFvTList(fvtName)
    varList4b_noPS += getFvTList("_"+mixedName+"_vAll")
    cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+ttFile4b_noPS    + " --var "+",".join(varList4b_noPS))


    for s in subSamples:
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        varListMixed = list(varListSvB)

        fvtName = "_"+mixedName+"_v"+s
        varListMixed += getFvTList(fvtName)
        varListMixed += getFvTList("_"+mixedName+"_vAll")
        cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+dataFiles4bMix    + " --var "+",".join(varListMixed))



    babySit(cmds, doRun)


if o.convertH5ToH5AllMixedSamples:

    cmds = []


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
        #"SvB_MA_ps",
        #"SvB_MA_pzz",
        #"SvB_MA_pzh",
        #"SvB_MA_ptt",
        #"SvB_MA_q_1234",
        #"SvB_MA_q_1324",
        #"SvB_MA_q_1423",
        ]


    for mixedName in ["3bMix4b","3bDvTMix4b",
                      "3bMix3b","3bDvTMix3b","3bDvTMix3bDvT"]:


        for s in subSamples:
            dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

            varListMixed = list(varListSvB)
    
            cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+dataFiles4bMix    + " --var "+",".join(varListMixed))
    


    babySit(cmds, doRun)



if o.convertH5ToH5SignalMixData:

    cmds = []


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
        #"SvB_MA_ps",
        #"SvB_MA_pzz",
        #"SvB_MA_pzh",
        #"SvB_MA_ptt",
        #"SvB_MA_q_1234",
        #"SvB_MA_q_1324",
        #"SvB_MA_q_1423",
        ]


    for sig in signalSamples:
        sigFiles = '"'+outputDir+'/*'+sig+'*_3bSubSampled/picoAOD_'+mixedName+'.h5" '

        varListMixed = list(varListSvB)
        
        cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+sigFiles    + " --var "+",".join(varListMixed))
    


    babySit(cmds, doRun)



if o.convertH5ToH5MixedSignalAndData:

    cmds = []


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
        #"SvB_MA_ps",
        #"SvB_MA_pzz",
        #"SvB_MA_pzh",
        #"SvB_MA_ptt",
        #"SvB_MA_q_1234",
        #"SvB_MA_q_1324",
        #"SvB_MA_q_1423",
        ]


    for sig in signalSamples:
        sigFiles = '"'+outputDir+'/*'+sig+'*_3bSubSampled/picoAOD_'+mixedName+'.h5" '

        varListMixed = list(varListSvB)
        
        cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+sigFiles    + " --var "+",".join(varListMixed))


    dataFiles = '"'+outputDir+'/data*_v*/picoAOD_'+mixedName+'_v*.h5" '
    varListMixed = list(varListSvB)
    cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+dataFiles    + " --var "+",".join(varListMixed))
    


    babySit(cmds, doRun)



if o.convertH5ToH5Mixed4bSignal:

    cmds = []


    varListSvB = [
        "SvB_ps",
        "SvB_pzz",
        "SvB_pzh",
        "SvB_ptt",
        "SvB_q_1234",
        "SvB_q_1324",
        "SvB_q_1423",
        #"SvB_MA_ps",
        #"SvB_MA_pzz",
        #"SvB_MA_pzh",
        #"SvB_MA_ptt",
        #"SvB_MA_q_1234",
        #"SvB_MA_q_1324",
        #"SvB_MA_q_1423",
        ]


    for sig in signalSamples:
        sigFiles = '"'+outputDir+'/*'+sig+'201?/picoAOD_'+mixedName+'.h5" '

        varListMixed = list(varListSvB)
        
        cmds.append(convertH5ToH5 + " -o SvB_FvT  -i "+sigFiles    + " --var "+",".join(varListMixed))


    babySit(cmds, doRun)







#
#  plotFvTFits
#
if o.plotFvTFits:
    cmds = []
    logs = []

    offset = o.trainOffset
    modelDir="ZZ4b/nTupleAnalysis/pytorchModels/"

    modelsLogFiles = modelDir+"3bTo4b.FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames = "Nominal"

    modelsLogFiles += ","+modelDir+"FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
    modelNames += ",PBRFit"
    for s in subSamples:
        modelsLogFiles += ","+modelDir+mixedName+".v"+s+"FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
        modelNames     += ",v"+s

    #modelNames = "Nominal,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9"
    #modelNames = "Nominal,3bMix4bv0,3bMix4br,3bMix4bV2,3bMix4bV3,3bMix4bV4"

    cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_FvTFits_e20_offset"+str(offset)+"_"+mixedName
    cmd += " -i "+modelsLogFiles+" --names "+modelNames

    #modelsLogFiles =  modelDir+"FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
    #modelsLogFiles += ","+modelDir+"3bTo4b.FvT_HCR+attention_14_np2980_lr0.01_epochs20_offset"+str(offset)+".log"
    #modelNames = "PBR,Auton"
    #
    #cmd = "python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o "+outputDir+"/Plot_Debug_offset"+str(offset)
    #cmd += " -i "+modelsLogFiles+" --names "+modelNames






    cmds.append(cmd)

    babySit(cmds, doRun)
