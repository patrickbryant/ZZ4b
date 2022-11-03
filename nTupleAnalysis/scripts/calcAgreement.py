import uproot3
import numpy as np
import argparse

basePath="/hildafs/projects/phy210037p/alison/hh4b/closureTests/ULTrig/mixed2018_3bDvTMix4bDvT_v0/"
fileNameData = basePath+"picoAOD_3bDvTMix4bDvT_4b_wJCM_v0_newSBDef.root"
fileName1    = basePath+"FvT_3bDvTMix4bDvT_v0_newSBDef3bDvTMix4bDvT.v0.newSBDefFvT_HCR+attention_8_np1052_seed0_lr0.01_epochs20_offset0_epoch20.root"
fileName2    = basePath+"FvT_3bDvTMix4bDvT_v0_newSBDef3bDvTMix4bDvT.v0.newSBDefFvT_HCR+attention_8_np1052_seed3_lr0.01_epochs20_offset0_epoch20.root"


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file1', default=fileName1    )
parser.add_argument('--file2', default=fileName2    )
parser.add_argument('--fileData', default=fileNameData    )
parser.add_argument('--is4bInput', default=True    )
args = parser.parse_args()

#print("fileData is", args.fileData )


f  = uproot3.open(args.fileData)['Events']
#print(f.keys())
f1 = uproot3.open(args.file1)['Events']
f2 = uproot3.open(args.file2)['Events']
#print(f1.keys())

arrayNames = ["SR","SB", "mcPseudoTagWeight_3bDvTMix4bDvT_v0","nSelJets","event","run"]
data    = f.arrays(arrayNames)


# 
#  Weights
#
arrayNames = ['FvT_3bDvTMix4bDvT_v0_newSBDef_pd4', 'FvT_3bDvTMix4bDvT_v0_newSBDef_pd3', 'FvT_3bDvTMix4bDvT_v0_newSBDef_pt4', 'FvT_3bDvTMix4bDvT_v0_newSBDef_pt3']
f1Data    = f1.arrays(arrayNames)
f2Data    = f2.arrays(arrayNames)

#print(f1Data)
#print(f2Data)
#print(type(data))
#print(data[b'SB'])
#print(type(data[b'SB']))

#
#  Filter to say insample vs out of sample
#     SB = in-sample
#     SR = out-sample
#
SBFilter = data[b'SB'] == 1
SRFilter = data[b'SR'] == 1

#print(len(data[b'SB']))
#print(len(SBFilter))
#print(len(data[b'SB'][SBFilter]))
#print(f1Data[b'FvT_3bDvTMix4bDvT_v0_newSBDef_pd3'][SBFilter])

#
#  Get 4b predictions form input file and input sample
#
def getPredictions(inputData, inputFilter, debug = True):
    pd3 = inputData[b'FvT_3bDvTMix4bDvT_v0_newSBDef_pd3'][inputFilter]
    pt3 = inputData[b'FvT_3bDvTMix4bDvT_v0_newSBDef_pt3'][inputFilter]
    pd4 = inputData[b'FvT_3bDvTMix4bDvT_v0_newSBDef_pd4'][inputFilter]
    pt4 = inputData[b'FvT_3bDvTMix4bDvT_v0_newSBDef_pt4'][inputFilter]
    
    pm4   = pd4 - pt4
    pm3   = pd3 - pt3
    predict4 = pm4 > pm3
    return predict4

def getRegionNumbers(f1_predict4, f2_predict4, debug = False):

    nRegion = len(f1_predict4)
    
    #
    #  Get the accuracy for file 1
    #
    if args.is4bInput:
        f1_acc = f1_predict4.sum()/nRegion
    else:
        f1_acc = np.logical_not(f1_predict4).sum()/nRegion
    
    
    #
    #  Get the accuracy for file 2
    #
    
    if args.is4bInput:
        f2_acc = f2_predict4.sum()/nRegion
    else:
        f2_acc = np.logical_not(f2_predict4).sum()/nRegion
    
    if debug:
        print("SB accuracy f2",f2_acc)
        print("\tf2_predict4",f2_predict4)
    
    #
    #  Get the average accuracy
    #
    fAve_acc = 0.5*(f1_acc + f2_acc)
    
    
    #
    #  Get the agreement
    #
    
    f1_f2_agreement = np.logical_not(np.logical_xor(f1_predict4 , f2_predict4))
    f1_f2_agreement_total = f1_f2_agreement.sum()/nRegion
    if debug:
        print(f1_f2_agreement)
    
    return f1_acc, f2_acc, fAve_acc, f1_f2_agreement_total


    

f1_SB_predict4 = getPredictions(f1Data, SBFilter)
f2_SB_predict4 = getPredictions(f2Data, SBFilter)
f1_SB_acc, f2_SB_acc, fAve_SB_acc, f1_f2_SB_agreement = getRegionNumbers(f1_SB_predict4, f2_SB_predict4) 

print("SB accuracy",f1_SB_acc, f2_SB_acc, fAve_SB_acc, f1_f2_SB_agreement)
#print("\tf1_predict4",f1_predict4)


f1_SR_predict4 = getPredictions(f1Data, SRFilter)
f2_SR_predict4 = getPredictions(f2Data, SRFilter)
f1_SR_acc, f2_SR_acc, fAve_SR_acc, f1_f2_SR_agreement = getRegionNumbers(f1_SR_predict4, f2_SR_predict4) 
print("SR accuracy",f1_SR_acc, f2_SR_acc, fAve_SR_acc, f1_f2_SR_agreement)
#print(f1_SB_pm4)
#print(f1_SB_pm3)


