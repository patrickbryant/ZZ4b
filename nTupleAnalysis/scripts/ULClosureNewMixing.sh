# 
#  Make vAll datasets
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedData -e


#
#  vAll hists for Mixed Subsampling (Can probably just do a hadd here instead
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsForMixedSubSample -c --mixedName 3bDvTMix4bDvT -e


#
#  Make subSampleMixedQCD
#    (Check the overlap at the same time)
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --subSampleMixedQCD --mixedName 3bDvTMix4bDvT -e


#
#  Input Mixed data sets
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedDataNorm -e

#
#  Fit JCM
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsForJCM -c --mixedName 3bDvTMix4bDvT -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --doWeightsMixed -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --doWeightsNominal  -e
#py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --doWeightsMixed  -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --addJCM -c -e  


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyToAutonForFvTROOT   --mixedName 3bDvTMix4bDvT -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyFromAutonForFvTROOT   --mixedName 3bDvTMix4bDvT --gpuName gpu11 -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsWithFvT -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsWithFvT -c 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py  --histsNoFvT -c --histDetailStr "passPreSel.passTTCR.passTTCRe.passTTCRem.passMjjOth.HHSR.bdtStudy" -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsNoFvT -c -e

#
#
##


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --makeInputsForCombine -c 


