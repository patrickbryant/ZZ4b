
#
# FvT Training 
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addvAllWeights  --mixedName 3bDvTMix4bDvT  > runAddvAllWeights.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainFvT --trainOffset 0 --mixedName 3bDvTMix4bDvT 
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainFvT --trainOffset 0,1,2 --mixedName 3bDvTMix4bDvT  > runULFvTTrainingExtended.sh

# Had to do these is stages with one offset at a time....

#
#  Add SvB
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvB  --mixedName 3bDvTMix4bDvT   > runULExtended_addSvB.sh

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvB_MA  --mixedName 3bDvTMix4bDvT  >  runULExtended_addSvB_MA.sh


#
#  Add FvT
#

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addFvT  --mixedName 3bDvTMix4bDvT   > runULExtended_addFvT.sh


#
# Plots
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --makeClosurePlots --mixedName 3bDvTMix4bDvT  > runUL_makeClosurePlots.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --makeClosurePlots --mixedName 3bDvTMix4bDvT 


py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5 --mixedName 3bDvTMix4bDvT > runUL_convertH5ToH5.sh


py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5 --mixedName 3bDvTMix4bDvT

#
# FvT plots
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --plotFvT  --mixedName 3bDvTMix4bDvT 

###
#### Mixed Study
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBAllMixedSamples > runUL_addSvBAllMixedSamples.sh
###
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5AllMixedSamples  > runUL_convertH5ToH5AllMixedSamples.sh
###
#### Signal Injection
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBSignalMixData  --mixedName 3bDvTMix4bDvT > runUL_addSvBSignalMixData.sh
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5SignalMixData --mixedName 3bDvTMix4bDvT  > runUL_convertH5ToH5SignalMixData.sh
###
###
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBSignalMixData  --mixedName 3bDvTMix4bSignal > runUL_addSvBSignalMixSignal.sh
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5SignalMixData --mixedName 3bDvTMix4bSignal  > runUL_convertH5ToH5SignalMix4bSignal.sh
###
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu0 > runUL_addSvBSignalMixSignal.sh
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu0  > runUL_convertH5ToH5SignalMix4bSignal.sh
###
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu1 > runUL_addSvBSignalMixSignal.sh
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu1  > runUL_convertH5ToH5SignalMix4bSignal.sh
###
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu3 > runUL_addSvBSignalMixSignal.sh
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu3  > runUL_convertH5ToH5SignalMix4bSignal.sh
###
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu10 > runUL_addSvBSignalMixSignal.sh
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu10  > runUL_convertH5ToH5SignalMix4bSignal.sh
###
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu30 > runUL_addSvBSignalMixSignal.sh
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu30  > runUL_convertH5ToH5SignalMix4bSignal.sh
###
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixed4bSignal  --mixedName 4bMix4bSignalMu1 > runUL_addSvBSignalMixSignal.sh
###py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5Mixed4bSignal --mixedName 4bMix4bSignalMu1  > runUL_convertH5ToH5SignalMix4bSignal.sh
###




