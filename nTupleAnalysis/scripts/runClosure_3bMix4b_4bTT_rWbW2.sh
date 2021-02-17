py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --mixInputs -c -e

py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --makeInputFileLists -c -e

py closureTests/mixed/makeInputMixSamples.py --makeTarball -e

py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --histsForJCM -c -e


py ZZ4b/nTupleAnalysis/scripts/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 -w -c -e




py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined_4bTT.py --mixedName 3bMix4b_4bTT_rWbW2 --makeInputFileLists  -c -e

py closureTests/mixed/makeInputMixSamples.py --makeTarball -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined_4bTT.py --mixedName 3bMix4b_4bTT_rWbW2 --addJCM -c -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined_4bTT.py --mixedName 3bMix4b_4bTT_rWbW2  --convertROOT -c  -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined_4bTT.py --mixedName 3bMix4b_4bTT_rWbW2 --copyLocally -e


# Train
#py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining_3bMix4b_4bTT.py --doTrain --mixedName 3bMix4b_4bTT_rWbW2 --cuda 0 > runTraining_3bMix4b_4bTT_rWbW2.sh
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining_3bMix4b_4bTT.py --doTrain --mixedName 3bMix4b_4bTT_rWbW2 --cuda 0 --trainOffset 0 > runTraining_3bMix4b_4bTT_rWbW2_wCRSR_fixPhi_0.sh 
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining_3bMix4b_4bTT.py --doTrain --mixedName 3bMix4b_4bTT_rWbW2 --cuda 0 --trainOffset 1 > runTraining_3bMix4b_4bTT_rWbW2_wCRSR_fixPhi_1.sh 
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining_3bMix4b_4bTT.py --doTrain --mixedName 3bMix4b_4bTT_rWbW2 --cuda 0 --trainOffset 2 > runTraining_3bMix4b_4bTT_rWbW2_wCRSR_fixPhi_2.sh 

# source these in a training session

# For wCRSR
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining_3bMix4b_4bTT.py --addFvT --mixedName 3bMix4b_4bTT_rWbW2 --cuda 0  > runAddFvT_3bMix4b_4bTT_rWbW2_wCRSR.sh

# Add SvB 
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining_3bMix4b_4bTT.py --addSvB --mixedName 3bMix4b_4bTT_rWbW2 --cuda 0 > runAddSvB_3bMix4b_4bTT_rWbW2.sh

# makeplots

# Copy to EOS
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined_4bTT.py --mixedName 3bMix4b_4bTT_rWbW2 --copyToEOS -e

# Convert H5 to ROOT
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined_4bTT.py --mixedName 3bMix4b_4bTT_rWbW2  --convertH5 -c  -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined_4bTT.py --mixedName 3bMix4b_4bTT_rWbW2 --makeOutputFileLists -e


py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --histsWithFvT -c -e
py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --plotsWithFvT -c -e

py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --histsNoFvT -c -e
py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --plotsNoFvT -c -e

# Combined Plots
py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --haddSubSamples -c -e
py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --scaleComb -c -e
py closureTests/3bMix4b_4bTT/make3bMix4b_4bTT_Closure.py --mixedName 3bMix4b_4bTT_rWbW2 --plotsCombinedSamples -c -e
