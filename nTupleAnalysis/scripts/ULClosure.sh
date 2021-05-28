#
#  Setup
#
py ZZ4b/nTupleAnalysis/scripts/getInputEventCounts.py
py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --copySkims > copySkims.sh 
source copySkims.sh 

py ZZ4b/nTupleAnalysis/scripts/makeULClosure.py --makeSkims -c

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

