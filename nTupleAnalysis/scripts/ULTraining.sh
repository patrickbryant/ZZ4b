#py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainDataVsTT
py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --doTrainDataVsTT --trainOffset 0,1,2
#  wihch Gives:


#  0
python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT3 -e 20 -o 3b --cuda 1 --weightName mcPseudoTagWeight  --trainOffset 0,1,2 --train --update   -d "closureTests/UL//data201*/picoAOD_3b.h5"  -t "closureTests/UL//*TT*201*/picoAOD_3b.h5" 
#  1
python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT4 -e 20 -o 4b --cuda 1 --weightName mcPseudoTagWeight  --trainOffset 0,1,2 --train --update  -d "closureTests/UL//data201*/picoAOD_4b.h5"  -t "closureTests/UL//*TT*201*/picoAOD_4b.h5" 


# Debug
python  ZZ4b/nTupleAnalysis/scripts/debugHDF5.py   --weightName mcPseudoTagWeight  --FvTName mcPseudoTagWeight  -o closureTests/UL/debug   -i  closureTests/UL/data2017/picoAOD_4b.h5

# Plotting
> py ZZ4b/nTupleAnalysis/scripts/makeULTraining.py --plotDvT
#Gives
python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py  -o closureTests/UL//plots_DvT3  --weightName mcPseudoTagWeight  -d "closureTests/UL//data201*/picoAOD_3b.h5"  -t "closureTests/UL//*TT*201*/picoAOD_3b.h5" 

python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsDvTHDF5.py  -o closureTests/UL//plots_DvT4  --weightName mcPseudoTagWeight  -d "closureTests/UL//data201*/picoAOD_4b.h5"  -t "closureTests/UL//*TT*201*/picoAOD_4b.h5" 


python  ZZ4b/nTupleAnalysis/scripts/debugHDF5.py   --weightName mcPseudoTagWeight  --FvTName mcPseudoTagWeight  -o closureTests/UL/debug   -i  closureTests/UL/data2017/picoAOD_4b.h5


# Write out h4
python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT4        -i  "closureTests/UL//*201*/picoAOD_4b.h5"  --var DvT4,DvT4_pt4

python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT3        -i  "closureTests/UL//*201*/picoAOD_3b.h5"  --var DvT3,DvT3_pt3




#python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT3        -i  'closureTests/inputs/*_b0p60p3/picoAOD_3b_b0p60p3.h5' --var DvT3_3b_pt3
#
#
#python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT3 -e 20 -o rwTT.3b.b0p60p3 --cuda 1 --weightName mcPseudoTagWeight  --trainOffset 1 --train --update  --updatePostFix _3b_rwTT  -d "closureTests/inputs//*data201*_b0p60p3/pico*3b_b0p60p3.h5"  -t "closureTests/inputs//*TT*201*_b0p60p3/pico*3b_b0p60p3_rwTT.h5" 2>&1 |tee log_DvT_3b_rwTT
#
#
#python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT3_rwTT        -i  'closureTests/inputs/*_b0p60p3/picoAOD_3b_b0p60p3_rwTT.h5' --var DvT3_3b_rwTT_pt3
#python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT3_rwTT        -i  'closureTests/inputs/data*_b0p60p3/picoAOD_3b_b0p60p3.h5' --var DvT3_3b_rwTT_pt3
#
#
#
#python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py  -c DvT4 -e 20 -o rwTT.4b.b0p60p3 --cuda 1 --weightName mcPseudoTagWeight  --trainOffset 1 --train --update  --updatePostFix _4b_rwTT  -d "closureTests/inputs//*data201*_b0p60p3/pico*4b_b0p60p3.h5"  -t "closureTests/inputs//*TT*201*_b0p60p3/pico*4b_b0p60p3_rwTT.h5"
#
#
#python  ZZ4b/nTupleAnalysis/scripts/convert_h52h5.py -o DvT4_rwTT        -i  'closureTests/inputs/data*_b0p60p3/picoAOD_3b_b0p60p3.h5' --var DvT4_3b_rwTT_pt4
