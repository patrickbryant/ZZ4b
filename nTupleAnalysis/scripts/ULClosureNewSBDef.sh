py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --inputsForDataVsTT -c  -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --doWeightsQCD  -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --subSample3bQCD  -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSubSampledQCD -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputsDvT3DvT4 -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --makeTTPseudoData  -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --makeTTPSDataFilesLists -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --checkPSData  -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedData -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsForJCM -c --mixedName 3bDvTMix4bDvT -e



py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeDvTFileLists -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --testDvTWeights -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --testDvTWeights -c -e --doDvTReweight

## Fit JCM
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --doWeightsMixed -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --doWeightsNominal  -e
#




# hists for JCM
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --addJCM -c -e  


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForFvT --mixedName 3bDvTMix4bDvT -e


# Copied to 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyToAutonForFvTROOT   --mixedName 3bDvTMix4bDvT -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyFromAutonForFvTROOT   --mixedName 3bDvTMix4bDvT --gpuName gpu13 --weightName FvTWeights -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSvBFvT --mixedName 3bDvTMix4bDvT -c --weightName testJCM  -e  
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsFriends --mixedName 3bDvTMix4bDvT  -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsWithFvT -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsWithFvT -c 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py  --histsNoFvT -c --histDetailStr "passPreSel.passTTCR.passTTCRe.passTTCRem.passMjjOth.HHSR.bdtStudy" -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --makeInputsForCombine -c 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsNoFvT -c -e
