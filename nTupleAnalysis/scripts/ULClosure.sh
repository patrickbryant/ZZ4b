#
#  Setup
#
py ZZ4b/nTupleAnalysis/scripts/getInputEventCounts.py
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copySkims > copySkims.sh 
source copySkims.sh 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeSkims -c -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --haddChunks -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileLists -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeTarball -e

#
#  Inputs for Data Vs TT
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --inputsForDataVsTT -c  -e

#
#  Copy to Auton
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirs -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyToAuton -e


#
#  Copy From Auton
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyFromAuton -e


#
# Write out weights
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutDvTWeights -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeDvTFileLists -c -e

#
#  Test DvT
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --testDvTWeights -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --testDvTWeights -c -e --doDvTReweight



#
#  Now to clossure
#

#
#  Subsample 3b 
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --doWeightsQCD  -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --doWeightsData  -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --subSample3bQCD  -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --subSample3bData  -e



#
# Make hemis
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --make4bHemisWithDvT   -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --make4bHemiTarballDvT -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSubSampledQCD -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputs  -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputsDvT3   -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputsDvT3DvT4 -e




#
#  Make TTbar PS Data
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --makeTTPseudoData  -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --makeTTPSDataFilesLists -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --checkPSData  -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --checkOverlap  -e




#
#  Make Combined Mixed Data sets
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedData -e

#
#  Fit JCM
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsForJCM -c --mixedName 3bDvTMix4bDvT -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --doWeightsMixed -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c --doWeightsNominal  -e

#
#  Add JCM / Convert
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --addJCM -c -e  # Does both pico.root and pico.h5
(py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --addJCM -c --onlyConvert -e) # Only dones root -> h5

#
# Copy
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForFvT --mixedName 3bDvTMix4bDvT -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyToAutonForFvT   --mixedName 3bDvTMix4bDvT -e



#
#  Train (on gpu nodes)
#
source ULTraining.sh  

#
# Copy back
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyFromAutonForFvT   --mixedName 3bDvTMix4bDvT -e

#
#  Write out FvT SvB File
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeights --mixedName 3bDvTMix4bDvT -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSvBFvT --mixedName 3bDvTMix4bDvT -c  -e 

#py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsWithFvT -c -e
#py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsWithFvT -c --histDetailStr "passMDRs.passMjjOth.HHSR" -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsWithFvT -c --histDetailStr "passMDRs.passMjjOth.HHSR.passSvB" -e





py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsWithFvT -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --makeInputsForCombine -c  -e
##py ZZ4b/nTupleAnalysis/scripts/make3bMix4bClosure.py --mixedName 3bMix4b_rWbW2 --makeInputsForCombine -c 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsWithFvTVHH -c -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsMixedVsNominal -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsNoFvT -c --histDetailStr "passMDRs.passMjjOth.HHSR.passSvB" -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsNoFvT -c -e

#
#  For Fun 
#


#
#  Mixed study
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --make3bHemisWithDvT   -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --make3bHemiTarballDvT -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputs3b  -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputs3bDvT3   -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputs3bDvT3DvT3 -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedData -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertMixedSamples -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForMixedSamples  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSamplesToAuton  -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSamplesFromAuton  -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsAllMixedSamples  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSvBFvTAllMixedSamples  -c  -e 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsWithFvTAllMixedSamples -c --histDetailStr "passMDRs.passMjjOth.HHSR" -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --plotsMixedVsNominalAllMixedSamples -c -e


#
#  Signal Injection
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSignal -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsSignal -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --subSample3bSignal -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSignalSubSampled -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsSignal3bSubSamples -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixSignalDataHemis --mixedName 3bDvTMix4bDvT -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertSignalMixData --mixedName 3bDvTMix4bDvT -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForSignalMixData --mixedName 3bDvTMix4bDvT  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copySignalMixDataToAuton --mixedName 3bDvTMix4bDvT   -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copySignalMixDataFromAuton --mixedName 3bDvTMix4bDvT  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSignalMixData --mixedName 3bDvTMix4bDvT -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsSignalMixData --mixedName 3bDvTMix4bDvT -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsSignalMixData  --mixedName 3bDvTMix4bDvT  -c -e

#
#  For Signal Hemis
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeSignalPseudoData   -c  -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeSignalPSFileLists   -c  -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --checkSignalPSData   -c  -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeHemisSignalOnly   -c  -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeHemiTarballSignal   -c  -e 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixSignalSignalHemis --mixedName 3bDvTMix4bSignal -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertSignalMixData --mixedName 3bDvTMix4bSignal -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copySignalMixDataToAuton --mixedName 3bDvTMix4bSignal   -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copySignalMixDataFromAuton --mixedName 3bDvTMix4bSignal  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsSignalMixData  --mixedName 3bDvTMix4bSignal  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSignalMixData --mixedName 3bDvTMix4bSignal -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsSignalMixData --mixedName 3bDvTMix4bSignal -c -e

#
#  For combined hemi
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeSignalPSFileLists   -c  -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeHemisSignalAndData   -c  -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeHemiTarballSignalAndData   -c  -e 

# 1./873.736267542  = 0.0011445  => mu = 1 
# 0.011445  => mu = 10

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixSignalAndData --mixedName 3bDvTMix4bSignalMu0  -c  --mcHemiWeight 0.00000000001   -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertMixedSignalAndData --mixedName 3bDvTMix4bSignalMu0 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataToAuton --mixedName 3bDvTMix4bSignalMu0   -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataFromAuton --mixedName 3bDvTMix4bSignalMu0  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu0  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu0 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu0 -c -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixSignalAndData --mixedName 3bDvTMix4bSignalMu1  -c  --mcHemiWeight 0.0011445   -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertMixedSignalAndData --mixedName 3bDvTMix4bSignalMu1 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForMixedSignalAndData    -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataToAuton --mixedName 3bDvTMix4bSignalMu1   -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataFromAuton --mixedName 3bDvTMix4bSignalMu1  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu1  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu1 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu1 -c -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixSignalAndData --mixedName 3bDvTMix4bSignalMu3  -c  --mcHemiWeight 0.0034335   -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertMixedSignalAndData --mixedName 3bDvTMix4bSignalMu3 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForMixedSignalAndData    -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataToAuton --mixedName 3bDvTMix4bSignalMu3   -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataFromAuton --mixedName 3bDvTMix4bSignalMu3  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu3  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu3 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu3 -c -e



py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixSignalAndData --mixedName 3bDvTMix4bSignalMu10 -c  --mcHemiWeight 0.011445    -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertMixedSignalAndData --mixedName 3bDvTMix4bSignalMu10 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataToAuton --mixedName 3bDvTMix4bSignalMu10   -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataFromAuton --mixedName 3bDvTMix4bSignalMu10  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu10  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu10 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu10 -c -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixSignalAndData --mixedName 3bDvTMix4bSignalMu30  -c  --mcHemiWeight 0.034335   -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertMixedSignalAndData --mixedName 3bDvTMix4bSignalMu30 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForMixedSignalAndData    -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataToAuton --mixedName 3bDvTMix4bSignalMu30   -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixedSignalAndDataFromAuton --mixedName 3bDvTMix4bSignalMu30  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsMixedSignalAndData  --mixedName 3bDvTMix4bSignalMu30  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu30 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsMixedSignalAndData --mixedName 3bDvTMix4bSignalMu30 -c -e


#
# 4b mixede signal
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mix4bSignal --mixedName 4bMix4bSignalMu1  -c  --mcHemiWeight 0.0011445   -e 
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --convertMixed4bSignal --mixedName 4bMix4bSignalMu1 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForMixed4bSignal    -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixed4bSignalToAuton --mixedName 4bMix4bSignalMu1   -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyMixed4bSignalFromAuton --mixedName 4bMix4bSignalMu1  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsMixed4bSignal  --mixedName 4bMix4bSignalMu1  -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsMixed4bSignal --mixedName 4bMix4bSignalMu1 -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --histsMixed4bSignal --mixedName 4bMix4bSignalMu1 -c -e

#
#  Skim Signal
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeSkimsSignal -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeSkimsSignalVHH -c  -e
