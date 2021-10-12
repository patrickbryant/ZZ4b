import pandas as pd
thisFrame = pd.read_hdf("closureTests/ULExtended/mixed2016_3bDvTMix4bDvT_v9/picoAOD_3bDvTMix4bDvT_4b_wJCM_v9_weights.h5", key='df')

print thisFrame

print thisFrame.keys()
