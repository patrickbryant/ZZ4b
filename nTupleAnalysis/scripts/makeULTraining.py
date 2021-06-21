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
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6,7,8,9", help="Year or comma separated list of subsamples")
parser.add_option('--makeClosurePlots', action="store_true",      help="Should be obvious")
o, a = parser.parse_args()

doRun = o.execute

CUDA=str(o.cuda)

subSamples = o.subSamples.split(",")
mixedName = o.mixedName

outputDir="closureTests/UL/"

trainJOB='python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py '
plotDvT='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py'
makeClosurePlots='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py'

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


    outName = "3bTo4b."
    cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_Nominal"+"  --trainOffset "+str(o.trainOffset)+" --train --update  --updatePostFix _Nominal "
    cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4b + " -t " + ttFile3b + " --ttbar4b " + ttFile4b

    cmds.append(cmd)

    for s in subSamples:

        outName = (mixedName+"_v"+s).replace("_",".")
        dataFiles4bMix = '"'+outputDir+'/*mixed201*_'+mixedName+'_v'+s+'/picoAOD_'+mixedName+'*_v'+s+'.h5" '

        cmd = trainJOB+ " -c FvT -e 20 -o "+outName+" --cuda "+str(CUDA)+" --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+"  --trainOffset "+str(o.trainOffset)+" --train --update  --updatePostFix _"+mixedName+"_v"+s
        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b_noPS

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
    
#    #ttFile4bMix    = '"'+outputDir+'/*TT*201*_'+tagID+'_vAll/picoAOD_'+mixedName+'*_'+tagID+'_vAll.h5" '        
#
#    for s in subSamples:
#
#        #dataFiles4bMix = '"'+outputDir+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5" '
#        dataFiles4bMix = '"'+outputDir+'/*data201*_'+tagID+'_v'+s+'/picoAOD_'+mixedName+'*_'+tagID+'_v'+s+'.h5,'+outputDir+'/*TT*201*_PSData_'+tagID+'/pico*4b_PSData_'+tagID+'.h5" '
#        cmd = makeClosurePlots+ "  --weightName mcPseudoTagWeight_"+mixedName+"_v"+s+weightPostFix+"  --FvTName FvT_"+mixedName+"_v"+s+""+weightPostFix+"  -o "+outputDir+"/Plots_Umixed_"+mixedName+"_"+tagID+"_v"+s+weightPostFix+""
#        cmd += " -d "+dataFiles3b + " --data4b " + dataFiles4bMix + " -t " + ttFile3b + " --ttbar4b " + ttFile4b
#
#        cmds.append(cmd)

    babySit(cmds, doRun)
