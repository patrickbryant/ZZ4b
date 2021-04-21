# Make Picoa and convert to h5
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --inputsForDataVsTT -c -e
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --inputsForDataVsTT -c --doTTbarPtReweight -e 


# Copy locally
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --copyLocally -c -e
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --copyLocally -c --doTTbarPtReweight -e

# SEtup auton
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --makeAutonDirs -e


# Copy to auton
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --copyToAuton -e
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --copyToAuton --doTTbarPtReweight -e


# Copy From auton
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --copyFromAuton -e
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --copyWeightsToEOS -e

py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --writeOutDvT3Weights -c -e
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --writeOutDvT4Weights -c -e

py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --makeDvT3FileLists -c -e
py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --makeDvT4FileLists -c -e


py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --testDvTWeights -c -e


py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py -c --subSample3bQCD  -e

py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --make4bHemisWithDvT --doTTbarPtReweight -e

py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py --makeInputFileListsSubSampledQCD -e


py ZZ4b/nTupleAnalysis/scripts/makeInputMixSamples.py -c  --mixInputsDvT  -s 0 -e
