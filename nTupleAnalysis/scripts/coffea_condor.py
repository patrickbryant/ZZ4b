import work_queue as wq

from coffea.processor import Runner
from coffea.processor import WorkQueueExecutor

#####
# Using Conda on the LPC
#####
# https://github.com/conda-forge/miniforge
# Install miniforge:
# > unset PYTHONPATH
# > curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
# > bash Mambaforge-$(uname)-$(uname -m).sh # make sure to install in the nobackup area to avoid going over the quota of the nominal area: /uscms_data/d3/bryantp/mambaforge
# Setup a conda env: https://coffeateam.github.io/coffea/wq.html#intro-coffea-wq
# > conda create --name coffea-env
# > conda activate coffea-env
# Install Coffea and Work Queue into the environment https://cctools.readthedocs.io/en/latest/work_queue/
# > conda install python=3.8.3 six dill
# > conda install -c conda-forge coffea ndcctools conda-pack xrootd
# Pack the environment into a portable tarball.
# > conda-pack --name coffea-env --output coffea-env.tar.gz

###############################################################################
# Collect and display setup info.
###############################################################################

print("------------------------------------------------")
print("Example Coffea Analysis with Work Queue Executor")
print("------------------------------------------------")

import getpass
# import shutil
# import os.path

wq_env_tarball='coffea-env.tar.gz'
wq_manager_name = "coffea-wq-{}".format(getpass.getuser())
wq_port = 9123
# wq_wrapper_path=shutil.which('python_package_run')

print("Master Name: -M " + wq_manager_name)
print("Environment: "+wq_env_tarball)
# print("Wrapper Path: "+wq_wrapper_path)

print("------------------------------------------------")

###############################################################
# Configuration of the Work Queue Executor
###############################################################

work_queue_executor_args = {
    # Additional files needed by the processor, such as local code libraries.
    'extra_input_files' : [ 'ZZ4b/nTupleAnalysis/scripts/coffea_analysis.py', 'ZZ4b/nTupleAnalysis/scripts/MultiClassifierSchema.py', 'ZZ4b/nTupleAnalysis/scripts/networks.py',
                            'nTupleAnalysis/baseClasses/data/BTagSF2016/btagging_legacy16_deepJet_itFit.json.gz',
                            'nTupleAnalysis/baseClasses/data/BTagSF2017/btagging_legacy17_deepJet.json.gz',
                            'nTupleAnalysis/baseClasses/data/BTagSF2018/btagging_legacy18_deepJet.json.gz'],
    # Resources to allocate per task.
    "resources_mode": "auto",  # Adapt task resources to what's observed.
    "resource_monitor": True,  # Measure actual resource consumption
    # With resources set to auto, these are the max values for any task.
    "cores": 2,  # Cores needed per task.
    "memory": 500,  # Memory needed per task (MB)
    "disk": 1000,  # Disk needed per task (MB)
    "gpus": 0,  # GPUs needed per task.
    # Options to control how workers find this manager.
    "master_name": wq_manager_name,
    "port": wq_port,  # Port for manager to listen on: if zero, will choose automatically.
    # The named conda environment tarball will be transferred to each worker,
    # and activated. This is useful when coffea is not installed in the remote
    # machines.
    'environment_file': wq_env_tarball,
    # 'wrapper': wq_wrapper_path,
    'filepath': '/uscms_data/d3/bryantp/CMSSW_11_1_0_pre5/src/condor',
    # Debugging: Display output of task if not empty.
    "print_stdout": False,
    # Debugging: Display notes about each task submitted/complete.
    "verbose": True,
    # Debugging: Produce a lot at the manager side of things.
    "debug_log": "coffea-wq.log",
}

executor = WorkQueueExecutor(**work_queue_executor_args)

###############################################################################
# Run the analysis using local Work Queue workers
###############################################################################
if __name__ == '__main__':
    # import sys
    # sys.path.insert(0, 'ZZ4b/nTupleAnalysis/scripts')
    from coffea_analysis import *

    eos_base = 'root://cmseos.fnal.gov//store/user/pbryant/condor'
    nfs_base = '/uscms/home/bryantp/nobackup/ZZ4b'
    eos = False

    input_path  = f'{eos_base if eos else nfs_base}'
    output_path = f'{nfs_base}'

    metadata = {}
    fileset = {}
    years = ['2016', '2017', '2018']
    years = ['2018']
    for year in years:
        datasets = [f'HH4b{year}']
        # if year == '2016':
        #     datasets += [f'ZZ4b2016_preVFP',  f'ZH4b2016_preVFP',  f'ggZH4b2016_preVFP']
        #     datasets += [f'ZZ4b2016_postVFP', f'ZH4b2016_postVFP', f'ggZH4b2016_postVFP']
        # else:
        #     datasets += [f'ZZ4b{year}', f'ZH4b{year}', f'ggZH4b{year}']
        # datasets = [f'ZZ4b{year}']
        
        for dataset in datasets:
            VFP = '_'+dataset.split('_')[-1] if 'VFP' in dataset else ''

            metadata[dataset] = {'isMC'  : True,
                                 'xs'    : xsDictionary[dataset.replace(year+VFP,'')],
                                 'lumi'  : lumiDict[year+VFP],
                                 'year'  : year,
                                 'btagSF': btagSF_file(year+VFP, UL=False if 'HH4b' in dataset else True, conda_pack=True),
            }
            fileset[dataset] = {'files': [f'{input_path}/{dataset}/picoAOD.root',],
                                'metadata': metadata[dataset]}

    analysis_args = {'debug': False,
                     'JCM': 'ZZ4b/nTupleAnalysis/weights/dataRunII/jetCombinatoricModel_SB_00-00-02.txt',
                     'btagVariations': btagVariations(),
                     #'SvB': 'ZZ4b/nTupleAnalysis/pytorchModels/SvB_HCR_8_np753_seed0_lr0.01_epochs20_offset*_epoch20.pkl',
    }

    tstart = time.time()

    workers = wq.Factory(
        # local runs:
        # batch_type="local",
        # manager_host_port="localhost:{}".format(wq_port)
        # with a batch system, e.g., condor.
        # (If coffea not at the installation site, then a conda
        # environment_file should be defined in the work_queue_executor_args.)
        batch_type="condor", 
        manager_name=wq_manager_name
    )

    workers.max_workers = 4
    workers.min_workers = 1
    workers.cores = 2
    workers.memory = 1000  # MB.
    workers.disk = 2000  # MB

    with workers:
        # define the Runner instance
        run_fn = Runner(
            schema=processor.NanoAODSchema,
            executor=executor,
            chunksize=10_000,
            #maxchunks=4,  # change this to None for a large run
        )
        # execute the analysis on the given dataset
        output = run_fn(fileset, 
                        treename='Events', 
                        processor_instance=analysis(**analysis_args),
        )

    elapsed = time.time() - tstart
    nEvent = sum([output['nEvent'][dataset] for dataset in output['nEvent'].keys()])
    print(f'{nEvent/elapsed:,.0f} events/s total')

    with open(f'{output_path}/hists.pkl', 'wb') as hfile:
        pickle.dump(output, hfile)

