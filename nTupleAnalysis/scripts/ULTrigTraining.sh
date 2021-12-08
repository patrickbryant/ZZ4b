#
# DvT Training 
#

#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainDataVsTT
> py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainDataVsTT --trainOffset 0,1,2
#  whic Gives:

#  0
python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT3 -e 20 -o 3b --cuda 1 --weightName mcPseudoTagWeight  --trainOffset 0,1,2 --train    -d "closureTests/ULTrig//data201*/picoAOD_3b.h5"  -t "closureTests/ULTrig//*TT*201*/picoAOD_3b.h5" 
#  1
python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT4 -e 20 -o 4b --cuda 1 --weightName mcPseudoTagWeight  --trainOffset 0,1,2 --train   -d "closureTests/ULTrig//data201*/picoAOD_4b.h5"  -t "closureTests/ULTrig//*TT*201*/picoAOD_4b.h5" 


# Plotting
> py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --plotDvT

#  0
python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py  -o closureTests/ULTrig//plots_DvT3  --weightName mcPseudoTagWeight  --DvTName DvT3  -d "closureTests/ULTrig//data201*/picoAOD_3b.h5"  -t "closureTests/ULTrig//*TT*201*/picoAOD_3b.h5" 
#  1
python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py  -o closureTests/ULTrig//plots_DvT4  --weightName mcPseudoTagWeight  --DvTName DvT4  -d "closureTests/ULTrig//data201*/picoAOD_4b.h5"  -t "closureTests/ULTrig//*TT*201*/picoAOD_4b.h5" 
#

## Write out h4

>py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --writeOutDvT
#  0
python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT3   --update   -m ZZ4b/nTupleAnalysis/pytorchModels/3bDvT3_HCR+attention_14_np2684_lr0.01_epochs20_offset0_epoch20.pkl,ZZ4b/nTupleAnalysis/pytorchModels/3bDvT3_HCR+attention_14_np2684_lr0.01_epochs20_offset1_epoch20.pkl,ZZ4b/nTupleAnalysis/pytorchModels/3bDvT3_HCR+attention_14_np2684_lr0.01_epochs20_offset2_epoch20.pkl -d "closureTests/ULTrig//data201*/picoAOD_3b.h5"  -t "closureTests/ULTrig//*TT*201*/picoAOD_3b.h5"  --writeWeightFile  --weightFilePostFix DvT3 
#  1
python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT4   --update   -m ZZ4b/nTupleAnalysis/pytorchModels/4bDvT4_HCR+attention_14_np2684_lr0.01_epochs20_offset0_epoch20.pkl,ZZ4b/nTupleAnalysis/pytorchModels/4bDvT4_HCR+attention_14_np2684_lr0.01_epochs20_offset1_epoch20.pkl,ZZ4b/nTupleAnalysis/pytorchModels/4bDvT4_HCR+attention_14_np2684_lr0.01_epochs20_offset2_epoch20.pkl -d "closureTests/ULTrig//data201*/picoAOD_4b.h5"  -t "closureTests/ULTrig//*TT*201*/picoAOD_4b.h5"  --writeWeightFile  --weightFilePostFix DvT4 



#
# FvT Training 
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addvAllWeights  --mixedName 3bDvTMix4bDvT  > runAddvAllWeights.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainFvT --trainOffset 0,1,2 --mixedName 3bDvTMix4bDvT  > runULFvTTraining.sh


py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addFvT  --mixedName 3bDvTMix4bDvT  > runAddFvT.sh



#
#  Add SvB
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvB  --mixedName 3bDvTMix4bDvT  > runAddSvB.sh

#
# FvT plots
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --plotFvT  --mixedName 3bDvTMix4bDvT 


#
##
## Plots
##
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --makeClosurePlots --mixedName 3bDvTMix4bDvT  > runUL_makeClosurePlots.sh
##py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --makeClosurePlots --mixedName 3bDvTMix4bDvT 
#
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5 --mixedName 3bDvTMix4bDvT > runUL_convertH5ToH5.sh
#
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5 --mixedName 3bDvTMix4bDvT
#
#
#
## Mixed Study
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBAllMixedSamples > runUL_addSvBAllMixedSamples.sh
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5AllMixedSamples  > runUL_convertH5ToH5AllMixedSamples.sh
#
## Signal Injection
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBSignalMixData  --mixedName 3bDvTMix4bDvT > runUL_addSvBSignalMixData.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5SignalMixData --mixedName 3bDvTMix4bDvT  > runUL_convertH5ToH5SignalMixData.sh
#
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBSignalMixData  --mixedName 3bDvTMix4bSignal > runUL_addSvBSignalMixSignal.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5SignalMixData --mixedName 3bDvTMix4bSignal  > runUL_convertH5ToH5SignalMix4bSignal.sh
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu0 > runUL_addSvBSignalMixSignal.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu0  > runUL_convertH5ToH5SignalMix4bSignal.sh
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu1 > runUL_addSvBSignalMixSignal.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu1  > runUL_convertH5ToH5SignalMix4bSignal.sh
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu3 > runUL_addSvBSignalMixSignal.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu3  > runUL_convertH5ToH5SignalMix4bSignal.sh
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu10 > runUL_addSvBSignalMixSignal.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu10  > runUL_convertH5ToH5SignalMix4bSignal.sh
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu30 > runUL_addSvBSignalMixSignal.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu30  > runUL_convertH5ToH5SignalMix4bSignal.sh
#
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixed4bSignal  --mixedName 4bMix4bSignalMu1 > runUL_addSvBSignalMixSignal.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5Mixed4bSignal --mixedName 4bMix4bSignalMu1  > runUL_convertH5ToH5SignalMix4bSignal.sh
#
#
#
#
#
