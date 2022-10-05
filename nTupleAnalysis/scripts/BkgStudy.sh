
#py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --copyFromAutonForDvTROOT   --mixedName 3bDvTMix4bDvT --gpuName gpu14 --weightName DvTWeights -e


py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --makeInputFileListsFriendsRW --mixedName 3bDvTMix4bDvT  --weightName RW -e

py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithRW -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName RW 
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithRW -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName RW

#
#
#
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --makeInputFileListsFriendsOT --mixedName 3bDvTMix4bDvT  --weightName OT -e


py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithOT -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName OT
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithOT -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName OT

