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
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --make4bHemisWithDvT -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --make4bHemiTarballDvT -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeInputFileListsSubSampledQCD -e


py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputs  -s 0 -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputsDvT3  -s 0 -e
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputsDvT3DvT4  -s 0 -e
#py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py -c  --mixInputsDvT   --useHemiWeights -e


#
#  Make TTbar PS Data
#


#
#  Make Combined Mixed Data sets
#

#
#  Fit JCM
#

#
#  Convert to H5
#

#
#  Train
#
