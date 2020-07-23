CUDA=0
baseDir=/zfsauton2/home/jalison/hh4b/
outputDir=$baseDir/closureTests/combined
outputDirNom=$baseDir/closureTests/nominal
outputDir3bMix4b=$baseDir/closureTests/3bMix4b


# Helpers
#SvBModel=ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1692_lr0.008_epochs40_stdscale_epoch40_loss0.2070.pkl
SvBModel=ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1713_lr0.008_epochs40_stdscale_epoch40_loss0.2138.pkl
trainJOB='python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py'
makeClosurePlots='python  ZZ4b/nTupleAnalysis/scripts/makeClosurePlotsHDF5.py'

#
# Train
#   (with GPU enviorment)
#$trainJOB -c FvT -d "${outputDir}/*data201*/pico*3b*h5" --data4b "${outputDirNom}/*data201*/pico*4b.h5" -t "${outputDir}/*TT*201*/pico*3b_wJCM*h5" --ttbar4b "${outputDirNom}/*TT*201*/pico*4b.h5" -e 40 -o 3bTo4bITER3 --cuda $CUDA --weightName mcPseudoTagWeight_Nominal 2>&1 |tee log ${outputDir}/log_Train_FvT_3bTo4b &
#
##for i in 0 1 2 3 4
#for i in 4
#do
#    $trainJOB -c FvT -d "${outputDir}/*data201*/pico*3b*h5" --data4b "${outputDir3bMix4b}/*data201*v${i}/pico*v${i}.h5"  -t "${outputDir}/*TT*201*/pico*3b_wJCM*h5" --ttbar4b "${outputDir3bMix4b}/*TT*201*v${i}/pico*v${i}.h5" -e 40 -o 3bMix4bv${i}ITER3 --cuda 1 --weightName mcPseudoTagWeight_3bMix4b_v${i} 2>&1 |tee log ${outputDir}/log_Train_FvT_3bMix4b_v${i} &
#done




modelDir=ZZ4b/nTupleAnalysis/pytorchModels/
#modelDetails=FvT_ResNet+multijetAttention_9_9_9_np2070_lr0.008_epochs40_stdscale.log
modelDetails=ITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale.log
#python ZZ4b/nTupleAnalysis/scripts/plotFvTFits.py -o ${outputDir}/Plot_FvTFitsITER3 -i ${modelDir}/3bTo4b${modelDetails},${modelDir}/3bMix4bv0${modelDetails},${modelDir}/3bMix4bv1${modelDetails},${modelDir}/3bMix4bv2${modelDetails},${modelDir}/3bMix4bv3${modelDetails},${modelDir}/3bMix4bv4${modelDetails} --names Nominal,3bMix4bV0,3bMix4bV1,3bMix4bV2,3bMix4bV3,3bMix4bV4


#
# Add SvB
#
#$trainJOB  -u  -d "${outputDir}/*data201*/pico*3b*h5" --data4b "${outputDirNom}/*data201*/pico*4b.h5"  -t "${outputDir}/*TT*201*/pico*3b_wJCM*h5" --ttbar4b "${outputDirNom}/*TT*201*/pico*4b.h5" -m $SvBModel -c SvB --cuda $CUDA  2>&1 |tee log ${outputDir}/log_Add_SvB_Nominal

#for i in 0 1 2 3 4
#do
#    $trainJOB  -u   -d "${outputDir3bMix4b}/*data201*v${i}/pico*v${i}.h5"   -t "${outputDir3bMix4b}/*TT*201*v${i}/pico*v${i}.h5" -m $SvBModel -c SvB  --cuda $CUDA  2>&1 |tee log ${outputDir}/log_Add_SvB_3bMix4b_v${i}
#done


#
# Add FvT
#
reweightModel_Nom=${modelDir}/3bTo4bITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch40_loss0.1553.pkl
reweightModel_v[0]=${modelDir}/3bMix4bv0ITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch31_loss0.1625.pkl
reweightModel_v[1]=${modelDir}/3bMix4bv1ITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch40_loss0.1621.pkl
reweightModel_v[2]=${modelDir}/3bMix4bv2ITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch38_loss0.1624.pkl
reweightModel_v[3]=${modelDir}/3bMix4bv3ITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch40_loss0.1615.pkl
reweightModel_v[4]=${modelDir}/3bMix4bv4ITER3FvT_ResNet+multijetAttention_9_9_9_np1881_lr0.008_epochs40_stdscale_epoch40_loss0.1617.pkl

#$trainJOB  -u  -d "${outputDir}/*data201*/pico*3b*h5" --data4b "${outputDirNom}/*data201*/pico*4b.h5"  -t "${outputDir}/*TT*201*/pico*3b_wJCM*h5" --ttbar4b "${outputDirNom}/*TT*201*/pico*4b.h5" -m $reweightModel_Nom -c FvT --updatePostFix _Nominal --cuda $CUDA  2>&1 |tee log ${outputDir}/log_Add_FvT_3bTo4b


#for i in 0 1 2 3 4
#do
#    $trainJOB  -u  -d "${outputDir}/*data201*/pico*3b*h5" --data4b "${outputDir3bMix4b}/*data201*v${i}/pico*v${i}.h5"  -t "${outputDir}/*TT*201*/pico*3b_wJCM*h5" --ttbar4b "${outputDir3bMix4b}/*TT*201*v${i}/pico*v${i}.h5" -m ${reweightModel_v[$i]} -c FvT --updatePostFix _3bMix4b_v${i} --cuda $CUDA  2>&1 |tee log ${outputDir}/log_Add_FvT_3bMix4b_v${i}
#done


#
##
##  Make Closure Plots
##
#$makeClosurePlots -d "${outputDir}/*data201*/pico*3b*h5" --data4b "${outputDirNom}/*data201*/pico*4b.h5"  -t "${outputDir}/*TT*201*/pico*3b_wJCM*h5" --ttbar4b "${outputDirNom}/*TT*201*/pico*4b.h5"  --weightName mcPseudoTagWeight_Nominal --FvTName FvT_Nominal  -o "${outputDir}/PlotsNominal" 
#
#for i in 0 1 2 3 4
#do
#    $makeClosurePlots -d "${outputDir}/*data201*/pico*3b*h5" --data4b "${outputDir3bMix4b}/*data201*v${i}/pico*v${i}.h5"  -t "${outputDir}/*TT*201*/pico*3b_wJCM*h5" --ttbar4b "${outputDir3bMix4b}/*TT*201*v${i}/pico*v${i}.h5"  --weightName mcPseudoTagWeight_3bMix4b_v${i}  --FvTName FvT_3bMix4b_v${i}  -o "${outputDir}/Plots_v${i}" 
#done
