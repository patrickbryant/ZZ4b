#
#  Setup
#
# See ULClosure for initial setup
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeTarball -e


# Need to make the nominal hists for JCM
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --testDvTWeights -c -e

#
#  Need to make the 4b PS data hists
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --checkPSData  -e

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
#(py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --addJCM -c --onlyConvert -e) # Only dones root -> h5

#
# Copy
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeAutonDirsForFvT --mixedName 3bDvTMix4bDvT -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyToAutonForFvT   --mixedName 3bDvTMix4bDvT -e




#
#  Train (on gpu nodes)
#
#source ULExtendedTraining.sh

#
# Copy back
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copyFromAutonForFvT   --mixedName 3bDvTMix4bDvT -e

#
#  Write out FvT SvB File
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeights --mixedName 3bDvTMix4bDvT -c -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSvBFvT --mixedName 3bDvTMix4bDvT -c  -e 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsWithFvT -c -e
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
#  OFfset debugging
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --writeOutSvBFvTWeightsOneOffset --mixedName 3bDvTMix4bDvT -c -s 0 --weightName weights_offset012 -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsWithFvTOneOffset -c -s 0 --weightName weights_offset0  -e

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsWithFvTOneOffset -c -s 0 --weightName weights_offset012 -e
