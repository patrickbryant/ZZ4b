import uproot3
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--sub_sample", default="0", help='Input File. Default: hists.pkl')
parser.add_argument("--input_dir", help='Input Sample')
parser.add_argument('--comb_FvT_3offsets', action="store_true", help="Write FvT")
parser.add_argument('--comb_FvT_1offset', action="store_true", help="Write FvT")
parser.add_argument("--n_kfolds", default=15)


args = parser.parse_args()


input_dir = args.input_dir
n_kfolds = int(args.n_kfolds)
sub_sample = args.sub_sample


if args.comb_FvT_3offsets:

    
    #
    # Read input files
    #
    inputFiles = []
    

    for seed in range(int(args.n_kfolds)):
        inputFileName = f"{args.input_dir}/FvT_3bDvTMix4bDvT_v{args.sub_sample}_newSBDefSeed{seed}.root"
        inputFiles.append(uproot3.open(inputFileName))
        

    #
    # Sanity check on sizes
    #
    size_Var0 = inputFiles[0]["Events"].array(f"FvT_3bDvTMix4bDvT_v{sub_sample}_newSBDefSeed0").size
    
    for seed in range(int(args.n_kfolds)):
        FvTName = f"FvT_3bDvTMix4bDvT_v{sub_sample}_newSBDefSeed{seed}"
    
        if not inputFiles[seed]["Events"].array(FvTName).size == size_Var0:
            print(f"ERROR sizes dont match! {seed}")

    
    nTot = size_Var0


    newFileName = f"{args.input_dir}/FvT_3bDvTMix4bDvT_v{args.sub_sample}_newSBDefSeedAll{args.n_kfolds}.root"


    
    print(f"Writting out {newFileName}")
    with uproot3.recreate(newFileName) as newFile:
    
        FvT_base_name = f"FvT_3bDvTMix4bDvT_v{sub_sample}_newSBDefSeed"
        
        branchDict = {f"{FvT_base_name}{seed}": np.float32 for seed in range(n_kfolds)} 
    
        ave_branch = f"{FvT_base_name}Ave"
        branchDict[ave_branch] = np.float32
    
        var_branch = f"{FvT_base_name}Var"
        branchDict[var_branch] = np.float32
    
        check_event_branch = f"{FvT_base_name}Ave_event"
        branchDict[check_event_branch] = int
    
        newFile['Events'] = uproot3.newtree(branchDict)
    
        branchData = {f"{FvT_base_name}{seed}": inputFiles[seed]["Events"].array(f"{FvT_base_name}{seed}") for seed in range(n_kfolds)}
        branchData.update( {f"{FvT_base_name}Ave":       np.average(np.array(list(branchData.values())), axis=0)} )
        branchData.update( {f"{FvT_base_name}Var":       np.var(np.array(list(branchData.values())), axis=0)} )
        branchData.update( {f"{FvT_base_name}Ave_event": inputFiles[0]["Events"].array(f"{FvT_base_name}0_event")} )
        newFile['Events'].extend(branchData)




if args.comb_FvT_1offset:

    #
    # Read input files
    #
    inputFiles = []

    
    for seed in range(int(args.n_kfolds)):
        for off_set in [0,1,2]:
            inputFileName = f"{args.input_dir}/FvT_3bDvTMix4bDvT_v{args.sub_sample}_newSBDefSeed{seed}OS{off_set}.root"        
            print(f"reading {inputFileName}")
            inputFiles.append(uproot3.open(inputFileName))



    #
    # Sanity check on sizes
    #
    size_Var0 = inputFiles[0]["Events"].array(f"FvT_3bDvTMix4bDvT_v{sub_sample}_newSBDefSeed0OS0").size
    print(f"size {size_Var0}")

    for seed in range(int(args.n_kfolds)):
        for off_set in [0,1,2]:
            FvTName = f"FvT_3bDvTMix4bDvT_v{sub_sample}_newSBDefSeed{seed}OS{off_set}"
            comb_idx = 3 * seed + off_set
            print(f"checking {FvTName} of comb_idx {comb_idx}" )
            if not inputFiles[comb_idx]["Events"].array(FvTName).size == size_Var0:
                print(f"ERROR sizes dont match! {seed}")


    newFileName = f"{args.input_dir}/FvT_3bDvTMix4bDvT_v{args.sub_sample}_newSBDefSeedAllSingleOffsets.root"
    
    print(f"Writting out {newFileName}")
    with uproot3.recreate(newFileName) as newFile:
    
        FvT_base_name = f"FvT_3bDvTMix4bDvT_v{sub_sample}_newSBDefSeed"

        branchDict = {}
        branchDict = branchDict | {f"{FvT_base_name}{seed}OS0": np.float32 for seed in range(n_kfolds)}
        branchDict = branchDict | {f"{FvT_base_name}{seed}OS1": np.float32 for seed in range(n_kfolds)}
        branchDict = branchDict | {f"{FvT_base_name}{seed}OS2": np.float32 for seed in range(n_kfolds)} 
    
        ave_branch = f"{FvT_base_name}Ave"
        branchDict[ave_branch] = np.float32
    
        var_branch = f"{FvT_base_name}Var"
        branchDict[var_branch] = np.float32
    
        check_event_branch = f"{FvT_base_name}Ave_event"
        branchDict[check_event_branch] = int
    
        newFile['Events'] = uproot3.newtree(branchDict)

        branchData = {}
        branchData = branchData | {f"{FvT_base_name}{seed}OS0": inputFiles[3 * seed + 0]["Events"].array(f"{FvT_base_name}{seed}OS0") for seed in range(n_kfolds)}
        branchData = branchData | {f"{FvT_base_name}{seed}OS1": inputFiles[3 * seed + 1]["Events"].array(f"{FvT_base_name}{seed}OS1") for seed in range(n_kfolds)}
        branchData = branchData | {f"{FvT_base_name}{seed}OS2": inputFiles[3 * seed + 2]["Events"].array(f"{FvT_base_name}{seed}OS2") for seed in range(n_kfolds)}
        
        branchData.update( {f"{FvT_base_name}Ave":       np.average(np.array(list(branchData.values())), axis=0)} )
        branchData.update( {f"{FvT_base_name}Var":       np.var(np.array(list(branchData.values())), axis=0)} )
        branchData.update( {f"{FvT_base_name}Ave_event": inputFiles[0]["Events"].array(f"{FvT_base_name}0OS0_event")} )
        newFile['Events'].extend(branchData)
    
