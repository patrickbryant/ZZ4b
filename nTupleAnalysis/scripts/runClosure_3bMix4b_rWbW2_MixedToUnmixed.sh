# Fit weights on Auton cluster

# Copy to LPC
  py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --copyFromAuton --mixedName 3bMix4b_rWbW2  -e

# Move to EOS
 py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --copyToEOS -e

# create weightFile in ROOT
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --writeOut  -c -e

# make input fileLists
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --makeMixedToUnMixedInputFiles  -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --makeInputFileListsTTPseudoData -e


# histograms
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --histsForJCM -c -e

# Hadd datasets for single FvT Fit
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --haddSubSamplesForOneFvT -c  -e

# fit JCM
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 -w -c  -e

# Add JCM (and Mixed->Unmixed weights) in picoAODs
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --addJCMMixedToUnmixed -c -e

# Convert to H5
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --convertROOTToH5MixedToUnmixed -c  -e

# Copy locally
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --copyLocally -c -e

# copy to AUTON
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --makeAutonDirs --mixedName 3bMix4b_rWbW2 -e
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --copyToAuton --mixedName 3bMix4b_rWbW2  -e

# Train and add FvT on Auton

# Closure

# Copy from AUTON
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombinedTraining.py --copyFromAuton --mixedName 3bMix4b_rWbW2 -e

# Copy to EOS
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --copyToEOS -e

# Write out weight file inputs
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2  --convertH5 -c  -e
py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --writeOutFvTWeights -c  -e

py ZZ4b/nTupleAnalysis/scripts/makeClosureCombined.py --mixedName 3bMix4b_rWbW2 --makeOutputFileLists -e


py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --histsWithFvT -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --plotsWithFvT -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --makeInputsForCombine -c 


py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --histsNoFvT -c -e

py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --haddSubSamples -c -e
py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosureMixedToUnMixed.py --mixedName 3bMix4b_rWbW2 --scaleCombSubSamples -c -e
