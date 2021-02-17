py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --mixInputs -c -e


py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --makeInputFileLists -c -e

py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --makeTarball -e

py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --histsForJCM -c -e


py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 -w -c -e


py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --makeInputFileLists  -c -e

py closureTests/mixed/makeInputMixSamples.py --makeTarball -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --addJCM -c -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2  --convertROOT -c  -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --copyLocally -e


py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --makeAutonDirs --mixedName 3bMix4b_rWbW2 -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --copyToAuton --mixedName 3bMix4b_rWbW2  -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --copyToFrom --mixedName 3bMix4b_rWbW2  -e

### Train
# py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --doTrain --mixedName 3bMix4b_rWbW2 --cuda 2 --trainOffset 2 > runTraining_3bMix4b_rWbW2_v0.sh

# py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --makeClosurePlots --mixedName 3bMix4b_rWbW2  > runMakeClosurePlots_3bMix4b_rWbW2.sh 
##
### source these in a training session
##
### For wCRSR
##py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining_3bMix4b_4bTT.py --addFvT --mixedName 3bMix4b_rWbW2 --cuda 0  > runAddFvT_3bMix4b_rWbW2_wCRSR.sh
##
### Add SvB 
##py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining_3bMix4b_4bTT.py --addSvB --mixedName 3bMix4b_rWbW2 --cuda 0 > runAddSvB_3bMix4b_rWbW2.sh

# makeplots

# copy from auton
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --copyFromAuton --mixedName 3bMix4b_rWbW2 -e

# Copy to EOS
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --copyToEOS -e

# Convert H5 to ROOT
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2  --convertH5 -c  -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --makeOutputFileLists -e
py closureTests/mixed/makeInputMixSamples.py --makeTarball -e

py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --histsWithFvT -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --plotsWithFvT -c -e

py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --histsNoFvT -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --plotsNoFvT -c -e

### Combined Plots
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --haddSubSamples -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --scaleComb -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --plotsCombinedSamples -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --makeInputsForCombine -c 
