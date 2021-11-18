#
# DvT Training 
#

#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainDataVsTT
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainDataVsTT --trainOffset 0,1,2
#  wihch Gives:


#  0
python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT3 -e 20 -o 3b --cuda 1 --weightName mcPseudoTagWeight  --trainOffset 0,1,2 --train --update   -d "closureTests/UL//data201*/picoAOD_3b.h5"  -t "closureTests/UL//*TT*201*/picoAOD_3b.h5" 
#  1
python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT4 -e 20 -o 4b --cuda 1 --weightName mcPseudoTagWeight  --trainOffset 0,1,2 --train --update  -d "closureTests/UL//data201*/picoAOD_4b.h5"  -t "closureTests/UL//*TT*201*/picoAOD_4b.h5" 


# Debug
python  ZZ4b/nTupleAnalysis/scripts/debugHDF5.py   --weightName mcPseudoTagWeight  --FvTName mcPseudoTagWeight  -o closureTests/UL/debug   -i  closureTests/UL/data2017/picoAOD_4b.h5

# Plotting
> py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --plotDvT
#Gives
python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py  -o closureTests/UL//plots_DvT3  --weightName mcPseudoTagWeight  -d "closureTests/UL//data201*/picoAOD_3b.h5"  -t "closureTests/UL//*TT*201*/picoAOD_3b.h5" 

python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py  -o closureTests/UL//plots_DvT4  --weightName mcPseudoTagWeight  -d "closureTests/UL//data201*/picoAOD_4b.h5"  -t "closureTests/UL//*TT*201*/picoAOD_4b.h5" 


python  ZZ4b/nTupleAnalysis/scripts/debugHDF5.py   --weightName mcPseudoTagWeight  --FvTName mcPseudoTagWeight  -o closureTests/UL/debug   -i  closureTests/UL/data2017/picoAOD_4b.h5


# Write out h4
python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT4        -i  "closureTests/UL//*201*/picoAOD_4b.h5"  --var DvT4,DvT4_pt4

python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT3        -i  "closureTests/UL//*201*/picoAOD_3b.h5"  --var DvT3,DvT3_pt3


#
# FvT Training 
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addvAllWeights  --mixedName 3bDvTMix4bDvT  > runAddvAllWeights.sh
#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainFvT --trainOffset 0 --mixedName 3bDvTMix4bDvT 
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainFvT --trainOffset 0,1,2 --mixedName 3bDvTMix4bDvT  > runULFvTTraining.sh

#
#  Add SvB
#
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvB  --mixedName 3bDvTMix4bDvT  > runUL_addSvB.sh

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


# Mixed Study
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBAllMixedSamples > runUL_addSvBAllMixedSamples.sh

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5AllMixedSamples  > runUL_convertH5ToH5AllMixedSamples.sh

# Signal Injection
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBSignalMixData  --mixedName 3bDvTMix4bDvT > runUL_addSvBSignalMixData.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5SignalMixData --mixedName 3bDvTMix4bDvT  > runUL_convertH5ToH5SignalMixData.sh


py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBSignalMixData  --mixedName 3bDvTMix4bSignal > runUL_addSvBSignalMixSignal.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5SignalMixData --mixedName 3bDvTMix4bSignal  > runUL_convertH5ToH5SignalMix4bSignal.sh

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu0 > runUL_addSvBSignalMixSignal.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu0  > runUL_convertH5ToH5SignalMix4bSignal.sh

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu1 > runUL_addSvBSignalMixSignal.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu1  > runUL_convertH5ToH5SignalMix4bSignal.sh

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu3 > runUL_addSvBSignalMixSignal.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu3  > runUL_convertH5ToH5SignalMix4bSignal.sh

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu10 > runUL_addSvBSignalMixSignal.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu10  > runUL_convertH5ToH5SignalMix4bSignal.sh

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu30 > runUL_addSvBSignalMixSignal.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5MixedSignalAndData --mixedName 3bDvTMix4bSignalMu30  > runUL_convertH5ToH5SignalMix4bSignal.sh

py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --addSvBMixed4bSignal  --mixedName 4bMix4bSignalMu1 > runUL_addSvBSignalMixSignal.sh
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --convertH5ToH5Mixed4bSignal --mixedName 4bMix4bSignalMu1  > runUL_convertH5ToH5SignalMix4bSignal.sh





