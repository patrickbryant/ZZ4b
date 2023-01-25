
#py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --copyFromAutonForDvTROOT   --mixedName 3bDvTMix4bDvT --gpuName gpu14 --weightName DvTWeights -e


py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --makeInputFileListsFriendsRW --mixedName 3bDvTMix4bDvT  --weightName RW -e

py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithRW -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName RW 
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithRW -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName RW


#
#  No FvT
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py  --histsNoFvT -c --histDetailStr "passPreSel.pass4Jets.pass4AllJets.passMjjOth.HHSR.bdtStudy"  -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsNoFvT -c --histDetailStr "passPreSel,pass4Jets,pass4AllJets,passMjjOth,fourTag,SB,SR,HHSR" -s 0 -e 

#
#  With FvT
#
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --histsWithFvT -c --histDetailStr "passPreSel.pass4Jets.pass4AllJets.passMjjOth.HHSR.bdtStudy"  
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --mixedName 3bDvTMix4bDvT --plotsWithFvT -c --histDetailStr "passPreSel,pass4Jets,pass4AllJets,passMjjOth,fourTag,SB,SR"  -s 0 -e

#
#  OT v1
#
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --makeInputFileListsFriendsOT --mixedName 3bDvTMix4bDvT  --weightName OT -e


py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithOT -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName OT
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithOT -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName OT


#
#  OT PtAndM
#
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --makeInputFileListsFriendsOT --mixedName 3bDvTMix4bDvT  --weightName OT_PtAndM -e


py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithOT -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy.pass4Jets.pass4AllJets"  --weightName OT_PtAndM
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithOT -c --histDetailStr "passPreSel,pass4Jets,pass4AllJets,passMjjOth,fourTag,SB,SR"   --weightName OT_PtAndM



#
#  OT Random
#
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --makeInputFileListsFriendsOT --mixedName 3bDvTMix4bDvT  --weightName OT_Random -e


py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithOT -c --histDetailStr "passPreSel.pass4Jets.pass4AllJets.passMjjOth.HHSR.bdtStudy"  --weightName OT_Random
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithOT -c --histDetailStr "passPreSel,pass4Jets,pass4AllJets,passMjjOth,fourTag,SB,SR"  --weightName OT_Random


#
#  NN
#
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --makeInputFileListsFriendsNN --mixedName 3bDvTMix4bDvT  --weightName NN -e


py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithNN -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName NN
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithNN -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName NN

py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithNN -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName NN --doDvTReweight
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithNN -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName NN --doDvTReweight



#
#  NN
#
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --makeInputFileListsFriendsNN --mixedName 3bDvTMix4bDvT  --weightName NN_FvTClosure -e


py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --histsWithNN -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName NN_FvTClosure
py ZZ4b/nTupleAnalysis/scripts/makeBkgStudy.py --mixedName 3bDvTMix4bDvT --plotsWithNN -c --histDetailStr "passPreSel.passMjjOth.HHSR.bdtStudy"  --weightName NN_FvTClosure

