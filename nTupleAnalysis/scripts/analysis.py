import time
from copy import copy
import textwrap
import os, re
import sys
import subprocess
import shlex
from glob import glob
import optparse
from threading import Thread
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools 
from PlotTools import read_parameter_file
class nameTitle:
    def __init__(self, name, title):
        self.name  = name
        self.title = title

CMSSW = getCMSSW()
USER = getUSER()

EOSOUTDIR = 'root://cmseos.fnal.gov//store/user/'+USER+'/condor/'
CONDOROUTPUTBASE = '/store/user/'+USER+'/condor/'
TARBALL   = 'root://cmseos.fnal.gov//store/user/'+USER+'/condor/'+CMSSW+'.tgz'

parser = optparse.OptionParser()
parser.add_option('--makeFileList',action='store_true',                        default=False, help='Make file list with DAS queries')
parser.add_option('--makeFileListLeptons',action='store_true',                        default=False, help='Make file list with DAS queries')
parser.add_option('-e',            action='store_true', dest='execute',        default=False, help='Execute commands. Default is to just print them')
parser.add_option('-s',            action='store_true', dest='doSignal',       default=False, help='Run signal MC')
parser.add_option('-t',            action='store_true', dest='doTT',           default=False, help='Run ttbar MC')
parser.add_option('-a',            action='store_true', dest='doAccxEff',      default=False, help='Make Acceptance X Efficiency plots')
parser.add_option('-d',            action='store_true', dest='doData',         default=False, help='Run data')
parser.add_option('-q',            action='store_true', dest='doQCD',          default=False, help='Subtract ttbar MC from data to make QCD template')
parser.add_option('-y',                                 dest='year',           default='2016,2017,2018', help='Year or comma separated list of years')
parser.add_option('-o',                                 dest='outputBase',     default='/uscms/home/'+USER+'/nobackup/ZZ4b/', help='path to output')
parser.add_option('-w',            action='store_true', dest='doWeights',      default=False, help='Fit jetCombinatoricModel and nJetClassifier TSpline')
parser.add_option('--makeJECSyst', action='store_true', dest='makeJECSyst',    default=False, help='Make jet energy correction systematics friend TTrees')
parser.add_option('--doJECSyst',   action='store_true', dest='doJECSyst',      default=False, help='Run event loop for jet energy correction systematics')
parser.add_option('-j',            action='store_true', dest='useJetCombinatoricModel',       default=False, help='Use the jet combinatoric model')
parser.add_option('-r',            action='store_true', dest='reweight',       default=False, help='Do reweighting with nJetClassifier TSpline')
parser.add_option('--bTagSyst',    action='store_true', dest='bTagSyst',       default=False, help='run btagging systematics')
parser.add_option('--plot',        action='store_true', dest='doPlots',        default=False, help='Make Plots')
parser.add_option('-p', '--createPicoAOD',              dest='createPicoAOD',  type='string', help='Create picoAOD with given name')
parser.add_option(      '--subsample',                  dest='subsample',      default=False, action='store_true', help='Make picoAODs which are subsamples of threeTag to emulate fourTag')
parser.add_option(      '--root2h5',                    dest='root2h5',        default=False, action='store_true', help='convert picoAOD.h5 to .root')
parser.add_option(      '--xrdcph5',                    dest='xrdcph5',        default='', help='copy .h5 files to EOS if toEOS else download from EOS')
parser.add_option(      '--h52root',                    dest='h52root',        default=False, action='store_true', help='convert picoAOD.root to .h5')
parser.add_option('-f', '--fastSkim',                   dest='fastSkim',       action='store_true', default=False, help='Do fast picoAOD skim')
parser.add_option(      '--looseSkim',                  dest='looseSkim',      action='store_true', default=False, help='Relax preselection to make picoAODs for JEC Uncertainties which can vary jet pt by a few percent.')
parser.add_option('-n', '--nevents',                    dest='nevents',        default='-1', help='Number of events to process. Default -1 for no limit.')
parser.add_option(      '--detailLevel',                dest='detailLevel',  default='passMDRs,threeTag,fourTag', help='Histogramming detail level. ')
parser.add_option(      '--doTrigEmulation',                                   action='store_true', default=False, help='Emulate the trigger')
parser.add_option(      '--plotDetailLevel',            dest='plotDetailLevel',  default='passMDRs,fourTag,SB,CR,SRNoHH', help='Histogramming detail level. ')
parser.add_option('-c', '--doCombine',    action='store_true', dest='doCombine',      default=False, help='Make CombineTool input hists')
parser.add_option(   '--loadHemisphereLibrary',    action='store_true', default=False, help='load Hemisphere library')
parser.add_option(   '--noDiJetMassCutInPicoAOD',    action='store_true', default=False, help='create Output Hemisphere library')
parser.add_option(   '--createHemisphereLibrary',    action='store_true', default=False, help='create Output Hemisphere library')
parser.add_option(   '--maxNHemis',    default=10000, help='Max nHemis to load')
parser.add_option(   '--inputHLib3Tag', default='$PWD/data18/hemiSphereLib_3TagEvents_*root',          help='Base path for storing output histograms and picoAOD')
parser.add_option(   '--inputHLib4Tag', default='$PWD/data18/hemiSphereLib_4TagEvents_*root',           help='Base path for storing output histograms and picoAOD')
parser.add_option(   '--SvB_ONNX', action='store_true', default=False,           help='Run ONNX version of SvB model. Model path specified in analysis.py script')
parser.add_option(   '--condor',   action='store_true', default=False,           help='Run on condor')
o, a = parser.parse_args()

fromNANOAOD = (o.createPicoAOD == 'picoAOD.root' or o.createPicoAOD == 'none') 

#
# Analysis in several "easy" steps
#

### 1. Jet Combinatoric Model
# First run on data and ttbar MC
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -d -t -q -y 2016,2017,2018 -e
# Then make jet combinatoric model 
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -w -y 2016,2017,2018 -e
# Now run again and update the automatically generated picoAOD by making a temporary one which will then be copied over picoAOD.root
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -d -t -q -y 2016,2017,2018 -j -p tempPicoAOD.root -e

### 2. ThreeTag to FourTag reweighting
# Now convert the picoAOD to hdf5 to train the Four Vs Three tag classifier (FvT)
# > python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data201*/picoAOD.root /uscms/home/bryantp/nobackup/ZZ4b/TTTo*201*/picoAOD.root /uscms/home/bryantp/nobackup/ZZ4b/*Z*201*/picoAOD.root"
# Now train the classifier
# > python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py -c FvT -d "/uscms/home/bryantp/nobackup/ZZ4b/data201*/picoAOD.h5" -t "/uscms/home/bryantp/nobackup/ZZ4b/TTTo*201*/picoAOD.h5"
# Take the best result and update the hdf5 files with classifier output for each event
# > python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py -c FvT -d "/uscms/home/bryantp/nobackup/ZZ4b/data201*/picoAOD.h5" -t "/uscms/home/bryantp/nobackup/ZZ4b/TTTo*201*/picoAOD.h5" -s "/uscms/home/bryantp/nobackup/ZZ4b/*Z*201*/picoAOD.h5" -m <FvT_model.pkl> -u

### 3. Signal vs Background Classification
# Train the classifier
# > python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py -c SvB -d "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5" -t "/uscms/home/bryantp/nobackup/ZZ4b/TTTo*201*/picoAOD.h5" -s "/uscms/home/bryantp/nobackup/ZZ4b/*ZH2018/picoAOD.h5"
# Update the hdf5 files with the classifier output
# > python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py -c SvB -d "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5" -t "/uscms/home/bryantp/nobackup/ZZ4b/TTTo*201*/picoAOD.h5" -s "/uscms/home/bryantp/nobackup/ZZ4b/*ZH2018/picoAOD.h5" -m <SvB_model.pkl> -u
# Update the picoAOD.root with the results of the trained classifiers
# > python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data201*/picoAOD.h5 /uscms/home/bryantp/nobackup/ZZ4b/TTTo*201*/picoAOD.h5 /uscms/home/bryantp/nobackup/ZZ4b/*Z*201*/picoAOD.h5"
# Now run the Event Loop again to make the reweighted histograms with the classifier outputs
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py    -d -t -q -y 2016,2017,2018 -j    -e  (before reweighting)
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -s -d -t    -y 2016,2017,2018 -j -r -e   (after reweighting)

### 4. Make plots
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py --plot -y 2016,2017,2018,RunII -j -e    (before reweighting)
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py --plot -y 2016,2017,2018,RunII -j -r -e  (after reweighting)
# To make acceptance X efficiency plots first you need the cutflows without the loosened jet preselection needed for the JEC variations. -a will then make the accXEff plot input hists and make the nice .pdf's:
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -s  -y 2016,2017,2018 -p none -a -e

### 5. Jet Energy Correction Uncertainties!
# Make JEC variation friend TTrees with
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py --makeJECSyst -y 2018 -e
# Need to make onnx SvB model to run in CMSSW on JEC variations
# In mlenv5 on cmslpcgpu node run
# > python ZZ4b/nTupleAnalysis/scripts/multiClassifier.py -c SvB -m <model.pkl>
# Copy the onnx file to an sl7 CMSSW_11 area
# Specify the model.onnx above in the python variable SvB_ONNX
# Run signal samples with --SvB_ONNX --doJECSyst in sl7 and CMSSW_11


# Condor
# tar -zcvf CMSSW_11_1_0_pre5.tgz CMSSW_11_1_0_pre5 --exclude="*.pdf" --exclude=".git" --exclude="PlotTools" --exclude="madgraph" --exclude="*.pkl" --exclude="*.root" --exclude="tmp" --exclude="combine" --exclude-vcs --exclude-caches-all; ls -alh
# xrdfs root://cmseos.fnal.gov/ rm /store/user/bryantp/CMSSW_11_1_0_pre5.tgz
# xrdcp -f CMSSW_11_1_0_pre5.tgz root://cmseos.fnal.gov//store/user/bryantp/CMSSW_11_1_0_pre5.tgz

#
# Config
#
script     = 'ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py'
years      = o.year.split(',')
lumiDict   = {'2016':  '36.3e3',#35.8791
              '2016_preVFP': '19.5e3',
              '2016_postVFP': '16.5e3',
              '2017':  '36.7e3',#36.7338
              '2018':  '59.8e3',#59.9656
              '17+18': '96.7e3',
              'RunII':'132.8e3',
              }
bTagDict   = {'2016': '0.6',#'0.3093', #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation2016Legacy
              '2017': '0.6',#'0.3033', #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X
              '2018': '0.6',#'0.2770'} #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
              }
outputBase = o.outputBase
#outputBase = os.getcwd()+'/'
gitRepoBase= 'ZZ4b/nTupleAnalysis/weights/'

# File lists
periods = {'2016': 'BCDEFGH',
           '2017': 'CDEF',
           '2018': 'ABCD'}

JECSystList = ['_jerUp', '_jerDown',
               '_jesTotalUp', '_jesTotalDown']

def dataFiles(year):
    #return ['ZZ4b/fileLists/data'+year+period+'.txt' for period in periods[year]]
    files = []
    for period in periods[year]: 
        if fromNANOAOD:
            files += glob('ZZ4b/fileLists/data%s%s_chunk*.txt'%(year,period))
        else:
            files += glob('ZZ4b/fileLists/data%s%s.txt'%(year,period))
    return files

# Jet Combinatoric Model
JCMRegion = 'SB'
JCMVersion = '00-00-02'
JCMCut = 'passMDRs'
def jetCombinatoricModel(year):
    #return gitRepoBase+'data'+year+'/jetCombinatoricModel_'+JCMRegion+'_'+JCMVersion+'.txt'
    return gitRepoBase+'dataRunII/jetCombinatoricModel_'+JCMRegion+'_'+JCMVersion+'.txt'
#reweight = gitRepoBase+'data'+year+'/reweight_'+JCMRegion+'_'+JCMVersion+'.root'

SvB_ONNX = 'ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_8_8_8_np1391_lr0.01_epochs20_epoch20.onnx'


def mcFiles(year, kind='ttbar'):
    if kind=='ttbar':
        processes = ['TTToHadronic', 'TTToSemiLeptonic', 'TTTo2L2Nu']
    if kind=='signal':
        processes = ['ZZ4b', 'ZH4b', 'ggZH4b']
    files = []
    for process in processes:
        if fromNANOAOD:
            files += glob('ZZ4b/fileLists/%s%s*_chunk*.txt'%(process, year))
        else:
            if year == '2016': year = '2016_*VFP'
            files += glob('ZZ4b/fileLists/%s%s.txt'%(process, year))
    return files

def accxEffFiles(year):
    files = [outputBase+'ZZ4b'+year+'/histsFromNanoAOD.root',
             #outputBase+'ZH4b'+year+'/histsFromNanoAOD.root',
             #outputBase+'ggZH4b'+year+'/histsFromNanoAOD.root',
             outputBase+'bothZH4b'+year+'/histsFromNanoAOD.root',
             outputBase+'ZZandZH4b'+year+'/histsFromNanoAOD.root',
             ]
    return files

DAG = None
if o.condor:
    DAG = dag(fileName='analysis.dag')


def getFileListFile(dataset):
    fileList='ZZ4b/fileLists/'
    if '/BTagCSV/' in dataset or '/JetHT/' in dataset: # this is data
        idx = dataset.find('Run201')
        fileList = fileList+'data'+dataset[idx+3:idx+8]+'.txt'
    elif '/TTTo' in dataset: # this is a ttbar MC sample
        idx = dataset.find('_')
        fileList = fileList+dataset[1:idx]
        idx = dataset.find('20UL')
        fileList = fileList+'20'+dataset[idx+4:idx+6]+'.txt'
    elif '/MuonEG/' in dataset:
        idx = dataset.find('Run201')
        fileList = fileList+'MuonEgData'+dataset[idx+3:idx+8]+'.txt'

    elif '/SingleMuon/' in dataset:
        idx = dataset.find('Run201')
        fileList = fileList+'SingleMuonData'+dataset[idx+3:idx+8]+'.txt'


    if '/NANOAODSIM' in dataset and '20UL16' in dataset:
        if 'preVFP' in dataset: # 2016 MC split by pre/post VFP what ever that means. Has different lumi
            fileList = fileList.replace('.txt','_preVFP.txt')
        else:
            fileList = fileList.replace('.txt', '_postVFP.txt')
    return fileList

def makeFileList():
    #
    # Ultra Legacy (https://gitlab.cern.ch/cms-nanoAOD/nanoaod-doc/-/wikis/Releases/NanoAODv8) (https://twiki.cern.ch/twiki/bin/view/CMS/PdmVSummaryRun2DataProcessing)
    #

    # Data
    # dasgoclient -query="dataset=/BTagCSV/*UL*NanoAODv2*/NANOAOD" 
    # dasgoclient -query="dataset=/JetHT/*UL2018*NanoAODv2*/NANOAOD" 
    # ttbar
    # dasgoclient -query="dataset=/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/*20UL*NanoAOD*v2*/NANOAODSIM"
    # !!!!!! There is no 2017 SemiLeptonic sample with RunIISummer20UL !!!!!!
    # dasgoclient -query="dataset=/TTTo*_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAOD*/NANOAODSIM"
    datasets = ['/BTagCSV/Run2016B-ver1_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2016B-ver2_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2016C-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2016D-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2016E-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2016F-HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2016F-UL2016_MiniAODv1_NanoAODv2-v2/NANOAOD',
                '/BTagCSV/Run2016G-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2016H-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',

                #'/BTagCSV/Run2017B-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD', # HLT items were not running
                '/BTagCSV/Run2017C-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2017D-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/BTagCSV/Run2017E-UL2017_MiniAODv1_NanoAODv2-v2/NANOAOD',
                '/BTagCSV/Run2017F-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',

                '/JetHT/Run2018A-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/JetHT/Run2018B-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/JetHT/Run2018C-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
                '/JetHT/Run2018D-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',


                '/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM',
                '/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM',
                '/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM',
                '/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',

                '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM',
                '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM', 
                '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM',
                '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',

                '/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM',
                '/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM',
                '/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM',
                '/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM',
            ]
    

    removed = []
    fileLists = []
    for dataset in datasets:
        fileList = getFileListFile(dataset)
        if fileList not in removed:
            cmd = 'rm %s'%fileList
            execute(cmd, o.execute)
            removed.append(fileList)
        cmd = 'dasgoclient -query="file dataset=%s | grep file.name" | sort >> %s'%(dataset, fileList)
        execute(cmd, o.execute)
        if fileList not in fileLists:
            fileLists.append(fileList)

    for fileList in fileLists:
        cmd = "sed -i 's/\/store/root:\/\/cmsxrootd-site.fnal.gov\/\/store/g' %s"%fileList
        # cmd = "sed -i 's/\/store/root:\/\/cms-xrd-global.cern.ch\/\/store/g' %s"%fileList
        execute(cmd, o.execute)
        print 'made', fileList

    for fileList in fileLists:
        print fileList
        with open(fileList,'r') as f:
            files = [line for line in f.readlines()]
        nFiles = len(files)
        chunkSize = 10 if 'TT' in fileList else 20
        chunks = [files[i:i+chunkSize] for i in range(0, nFiles, chunkSize)]
        for c, chunk in enumerate(chunks):
            chunkName = fileList.replace('.txt','_chunk%02d.txt'%(c+1))
            with open(chunkName,'w') as f:
                f.writelines(chunk)
            print 'made', chunkName


def makeFileListLeptons():
    #
    # Ultra Legacy (https://gitlab.cern.ch/cms-nanoAOD/nanoaod-doc/-/wikis/Releases/NanoAODv8) (https://twiki.cern.ch/twiki/bin/view/CMS/PdmVSummaryRun2DataProcessing)
    #
    datasets = []
    datasets += [
        '/MuonEG/Run2016B-ver1_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2016B-ver2_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2016C-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2016D-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2016E-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2016F-HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2016F-UL2016_MiniAODv1_NanoAODv2-v2/NANOAOD',
        '/MuonEG/Run2016G-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2016H-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',

        '/MuonEG/Run2017B-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2017C-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2017D-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2017E-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2017F-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',

        '/MuonEG/Run2018A-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2018B-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2018C-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/MuonEG/Run2018D-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
    ]


    datasets += [
        '/SingleMuon/Run2016B-ver1_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2016B-ver2_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2016C-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2016D-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2016E-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2016F-HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2016F-UL2016_MiniAODv1_NanoAODv2-v4/NANOAOD',
        '/SingleMuon/Run2016G-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2016H-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD',

        '/SingleMuon/Run2017B-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2017C-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2017D-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2017E-UL2017_MiniAODv1_NanoAODv2-v2/NANOAOD',
        '/SingleMuon/Run2017F-UL2017_MiniAODv1_NanoAODv2-v2/NANOAOD',

        '/SingleMuon/Run2018A-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2018B-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2018C-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
        '/SingleMuon/Run2018D-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD',
    ]
        

    removed = []
    fileLists = []
    for dataset in datasets:
        fileList = getFileListFile(dataset)
        if fileList not in removed:
            cmd = 'rm %s'%fileList
            execute(cmd, o.execute)
            removed.append(fileList)
        cmd = 'dasgoclient -query="file dataset=%s | grep file.name" | sort >> %s'%(dataset, fileList)
        execute(cmd, o.execute)
        if fileList not in fileLists:
            fileLists.append(fileList)

    for fileList in fileLists:
        cmd = "sed -i 's/\/store/root:\/\/cmsxrootd-site.fnal.gov\/\/store/g' %s"%fileList
        # cmd = "sed -i 's/\/store/root:\/\/cms-xrd-global.cern.ch\/\/store/g' %s"%fileList
        execute(cmd, o.execute)
        print 'made', fileList

    for fileList in fileLists:
        print fileList
        with open(fileList,'r') as f:
            files = [line for line in f.readlines()]
        nFiles = len(files)
        chunkSize = 10 if 'TT' in fileList else 20
        chunks = [files[i:i+chunkSize] for i in range(0, nFiles, chunkSize)]
        for c, chunk in enumerate(chunks):
            chunkName = fileList.replace('.txt','_chunk%02d.txt'%(c+1))
            with open(chunkName,'w') as f:
                f.writelines(chunk)
            print 'made', chunkName



def makeTARBALL():
    base='/uscms/home/'+USER+'/nobackup/'
    if os.path.exists(base+CMSSW+'.tgz'):
        print '# TARBALL already exists, skip making it'
        return
    cmd  = 'tar -C '+base+' -zcvf '+base+CMSSW+'.tgz '+CMSSW
    cmd += ' --exclude="*.pdf" --exclude="*.jdl" --exclude="*.stdout" --exclude="*.stderr" --exclude="*.log"'
    cmd += ' --exclude=".git" --exclude="PlotTools" --exclude="madgraph" --exclude="*.pkl"'# --exclude="*.root"'#some root files needed for nano_postproc.py jetmetCorrector
    cmd += ' --exclude="toy4b"'
    cmd += ' --exclude="CombineHarvester"'
    cmd += ' --exclude="HiggsAnalysis"'
    cmd += ' --exclude="closureFits"'
    cmd += ' --exclude="higgsCombine*.root"'
    cmd += ' --exclude="tmp" --exclude="combine" --exclude-vcs --exclude-caches-all'
    execute(cmd, o.execute)
    cmd  = 'ls '+base+' -alh'
    execute(cmd, o.execute)
    cmd = 'xrdfs root://cmseos.fnal.gov/ mkdir /store/user/'+USER+'/condor'
    execute(cmd, o.execute)
    cmd = 'xrdcp -f '+base+CMSSW+'.tgz '+TARBALL
    execute(cmd, o.execute)
    

def makeJECSyst():
    basePath = EOSOUTDIR if o.condor else outputBase
    cmds=[]
    for year in years:
        for process in ['ZZ4b', 'ZH4b', 'ggZH4b']:
            cmd  = 'python PhysicsTools/NanoAODTools/scripts/nano_postproc.py '
            cmd += basePath+process+year+'/ '
            cmd += basePath+process+year+'/picoAOD.root '
            cmd += '--friend '
            cmd += '-I nTupleAnalysis.baseClasses.jetmetCorrectors jetmetCorrector'+year # modules are defined in https://github.com/patrickbryant/nTupleAnalysis/blob/master/baseClasses/python/jetmetCorrectors.py
            cmds.append(cmd)

    execute(cmds, o.execute, condor_dag=DAG)


def doSignal():
    basePath = EOSOUTDIR if o.condor else outputBase
    cp = 'xrdcp -f ' if o.condor else 'cp '

    mkdir(basePath, o.execute)

    cmds=[]
    JECSysts = ['']
    if o.doJECSyst: 
        JECSysts = JECSystList

    for JECSyst in JECSysts:
        histFile = 'hists'+JECSyst+'.root' #+('_j' if o.useJetCombinatoricModel else '')+('_r' if o.reweight else '')+'.root'
        if fromNANOAOD: histFile = 'histsFromNanoAOD'+JECSyst+'.root'

        for year in years:
            lumi = lumiDict[year]
            for fileList in mcFiles(year, 'signal'):
                cmd  = 'nTupleAnalysis '+script
                cmd += ' -i '+fileList
                cmd += ' -o '+basePath
                cmd += ' -y '+year
                if '2016' in fileList:
                    if 'preVFP' in fileList:
                        lumi = lumiDict['2016_preVFP']
                    else: 
                        lumi = lumiDict['2016_postVFP']
                cmd += ' -l '+lumi
                cmd += ' --histDetailLevel '+o.detailLevel
                cmd += ' --histFile '+histFile
                cmd += ' -j '+jetCombinatoricModel(year) if o.useJetCombinatoricModel else ''
                cmd += ' -r ' if o.reweight else ''
                cmd += ' -p '+o.createPicoAOD if o.createPicoAOD else ''
                #cmd += ' -f ' if o.fastSkim else ''
                cmd += ' --isMC'
                cmd += ' --bTag '+bTagDict[year]
                cmd += ' --bTagSF'
                cmd += ' --bTagSyst' if o.bTagSyst else ''
                cmd += ' --nevents '+o.nevents
                #cmd += ' --looseSkim' if o.looseSkim else ''
                cmd += ' --looseSkim' if (o.createPicoAOD or o.looseSkim) else '' # For signal samples we always want the picoAOD to be loose skim
                cmd += ' --SvB_ONNX '+SvB_ONNX if o.SvB_ONNX else ''
                cmd += ' --JECSyst '+JECSyst if JECSyst else ''
                if o.createPicoAOD and o.createPicoAOD != 'none':
                    if o.createPicoAOD != 'picoAOD.root':
                        sample = fileList.split('/')[-1].replace('.txt','')
                        cmd += '; '+cp+basePath+sample+'/'+o.createPicoAOD+' '+basePath+sample+'/picoAOD.root'

                cmds.append(cmd)

    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)

    cmds = []
    for year in years:

        for JECSyst in JECSysts:
            histFile = 'hists'+JECSyst+'.root' #+('_j' if o.useJetCombinatoricModel else '')+('_r' if o.reweight else '')+'.root'
            if fromNANOAOD: histFile = 'histsFromNanoAOD'+JECSyst+'.root'

            files = mcFiles(year, 'signal')
            if 'ZZ4b/fileLists/ZH4b'+year+'.txt' in files and 'ZZ4b/fileLists/ggZH4b'+year+'.txt' in files:
                mkdir(basePath+'bothZH4b'+year, o.execute)
                cmd = 'hadd -f '+basePath+'bothZH4b'+year+'/'+histFile+' '+basePath+'ZH4b'+year+'/'+histFile+' '+basePath+'ggZH4b'+year+'/'+histFile
                cmd += '' if o.condor else ' > hadd.log'
                cmds.append(cmd)

            if 'ZZ4b/fileLists/ZH4b'+year+'.txt' in files and 'ZZ4b/fileLists/ggZH4b'+year+'.txt' in files and 'ZZ4b/fileLists/ZZ4b'+year+'.txt' in files:
                mkdir(basePath+'ZZandZH4b'+year, o.execute)
                cmd = 'hadd -f '+basePath+'ZZandZH4b'+year+'/'+histFile+' '+basePath+'ZH4b'+year+'/'+histFile+' '+basePath+'ggZH4b'+year+'/'+histFile+' '+basePath+'ZZ4b'+year+'/'+histFile
                cmd += '' if o.condor else ' > hadd.log'
                cmds.append(cmd)

    if o.condor: 
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)

    cmds = []
    if '2016' in years and '2017' in years and '2018' in years:
        for JECSyst in JECSysts:
            histFile = 'hists'+JECSyst+'.root' #+('_j' if o.useJetCombinatoricModel else '')+('_r' if o.reweight else '')+'.root'

            if fromNANOAOD: histFile = 'histsFromNanoAOD'+JECSyst+'.root'

            for sample in ['ZZ4b', 'ZH4b', 'ggZH4b', 'bothZH4b', 'ZZandZH4b']:
                cmd  = 'hadd -f '+basePath+sample+'RunII/'+histFile+' '
                cmd += basePath+sample+'2016/'+histFile+' '
                cmd += basePath+sample+'2017/'+histFile+' '
                cmd += basePath+sample+'2018/'+histFile+' '
                cmd += '' if o.condor else ' > hadd.log'
                cmds.append(cmd)
                
    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)

      
def doAccxEff():   
    cmds = []

    plotYears = copy(years)
    if '2016' in years and '2017' in years and '2018' in years:
        plotYears += ['RunII']

    for year in plotYears:
        for signal in accxEffFiles(year):
            cmd = 'python ZZ4b/nTupleAnalysis/scripts/makeAccxEff.py -i '+signal
            cmds.append(cmd)
    babySit(cmds, o.execute)

def doDataTT():
    basePath = EOSOUTDIR if o.condor else outputBase
    cp = 'xrdcp -f ' if o.condor else 'cp '

    mkdir(basePath, o.execute)

    # run event loop
    cmds=[]
    histFile = 'hists'+('_j' if o.useJetCombinatoricModel else '')+('_r' if o.reweight else '')+'.root'
    if fromNANOAOD: histFile = 'histsFromNanoAOD.root'

    for year in years:
        lumi = lumiDict[year]
        files = []
        if o.doData: files += dataFiles(year)
        if o.doTT:   files += mcFiles(year)
        nFiles = len(files)
        if o.subsample: files = files*10
        for i, fileList in enumerate(files):
            cmd  = 'nTupleAnalysis '+script
            cmd += ' -i '+fileList
            cmd += ' -o '+basePath
            cmd += ' -y '+year
            cmd += ' --histDetailLevel '+o.detailLevel
            if o.subsample:
                vX = i//nFiles
                cmd += ' --histFile '+histFile.replace('.root','_subsample_v%d.root'%(vX))
            else:
                cmd += ' --histFile '+histFile
            cmd += ' -j '+jetCombinatoricModel(year) if o.useJetCombinatoricModel else ''
            cmd += ' -r ' if o.reweight else ''
            if o.subsample:
                cmd += ' -p picoAOD_subsample_v%d.root '%(vX)
                cmd += ' --emulate4bFrom3b --emulationOffset %d '%(vX)
            else:
                cmd += ' -p '+o.createPicoAOD if o.createPicoAOD else ''
            cmd += ' -f ' if o.fastSkim else ''
            cmd += ' --bTag '+bTagDict[year]
            cmd += ' --nevents '+o.nevents
            if fileList in mcFiles(year):
                if '2016' in fileList:
                    if 'preVFP' in fileList:
                        lumi = lumiDict['2016_preVFP']
                    else: 
                        lumi = lumiDict['2016_postVFP']
                cmd += ' -l '+lumi
                cmd += ' --bTagSF'
                #cmd += ' --bTagSyst' if o.bTagSyst else ''
                cmd += ' --doTrigEmulation' if o.doTrigEmulation else ''
                cmd += ' --isMC '
            if o.createHemisphereLibrary  and fileList not in ttbarFiles:
                cmd += ' --createHemisphereLibrary '
            if o.noDiJetMassCutInPicoAOD:
                cmd += ' --noDiJetMassCutInPicoAOD '
            if o.loadHemisphereLibrary:
                cmd += ' --loadHemisphereLibrary '
                cmd += ' --inputHLib3Tag '+o.inputHLib3Tag
                cmd += ' --inputHLib4Tag '+o.inputHLib4Tag
            cmd += ' --SvB_ONNX '+SvB_ONNX if o.SvB_ONNX else ''

            if o.createPicoAOD and o.createPicoAOD != 'none' and not o.subsample:
                if o.createPicoAOD != 'picoAOD.root':
                    sample = fileList.split('/')[-1].replace('.txt','')
                    cmd += '; '+cp+basePath+sample+'/'+o.createPicoAOD+' '+basePath+sample+'/picoAOD.root'

            cmds.append(cmd)

    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)

    if o.subsample:
        return

    # combine chunked picoAODs if we just skimmed them from the NANOAODs
    cmds = []
    if o.createPicoAOD == 'picoAOD.root':
        for year in years:
            if o.doData:
                for period in periods[year]:
                    cmd  = 'hadd -f %sdata%s%s/picoAOD.root'%(basePath, year, period)
                    nChunks = len(glob('ZZ4b/fileLists/data%s%s_chunk*.txt'%(year,period)))
                    for chunk in range(1,nChunks+1):
                        cmd += ' %sdata%s%s_chunk%02d/picoAOD.root '%(basePath, year, period, chunk)
                    cmds.append(cmd)

                    cmd  = 'hadd -f %sdata%s%s/histsFromNanoAOD.root'%(basePath, year, period)
                    for chunk in range(1,nChunks+1):
                        cmd += ' %sdata%s%s_chunk%02d/histsFromNanoAOD.root '%(basePath, year, period, chunk)
                    cmds.append(cmd)

            if o.doTT:
                processes = ['TTToHadronic'+year, 'TTToSemiLeptonic'+year, 'TTTo2L2Nu'+year]
                if year == '2016': 
                    processes = [p+'_preVFP' for p in processes] + [p+'_postVFP' for p in processes]
                for process in processes:
                    cmd  = 'hadd -f %s%s/picoAOD.root'%(basePath, process)
                    nChunks = len(glob('ZZ4b/fileLists/%s_chunk*.txt'%(process)))
                    for chunk in range(1,nChunks+1):
                        cmd += ' %s%s_chunk%02d/picoAOD.root'%(basePath, process, chunk)
                    cmds.append(cmd)

                    cmd  = 'hadd -f %s%s/histsFromNanoAOD.root'%(basePath, process)
                    for chunk in range(1,nChunks+1):
                        cmd += ' %s%s_chunk%02d/histsFromNanoAOD.root'%(basePath, process, chunk)
                    cmds.append(cmd)

        if o.condor:
            DAG.addGeneration()
        execute(cmds, o.execute, condor_dag=DAG)
        return
                    

    cmds = []
    if o.doTT and '2016' in years: # need to combine pre/postVFP hists
        for tt in ['TTToHadronic', 'TTToSemiLeptonic', 'TTTo2L2Nu']:
            mkdir(basePath+tt+'2016', o.execute)
            cmd = 'hadd -f '+basePath+tt+'2016/'+histFile+' '+basePath+tt+'2016_preVFP/'+histFile+' '+basePath+tt+'2016_postVFP/'+histFile
            cmd += '' if o.condor else ' > hadd.log'
            cmds.append(cmd)
        if o.condor:
            DAG.addGeneration()
        execute(cmds, o.execute, condor_dag=DAG)
        

    # make combined histograms for plotting purposes
    cmds = []
    for year in years:
        if o.doData:
            mkdir(basePath+'data'+year, o.execute)
            cmd = 'hadd -f '+basePath+'data'+year+'/'+histFile+' '+' '.join([basePath+'data'+year+period+'/'+histFile for period in periods[year]])
            cmd += '' if o.condor else ' > hadd.log'
            cmds.append(cmd)
    
        if o.doTT:
            # files = mcFiles(year)
            # if 'ZZ4b/fileLists/TTToHadronic'+year+'.txt' in files and 'ZZ4b/fileLists/TTToSemiLeptonic'+year+'.txt' in files and 'ZZ4b/fileLists/TTTo2L2Nu'+year+'.txt' in files:
            mkdir(basePath+'TT'+year, o.execute)
            cmd = 'hadd -f '+basePath+'TT'+year+'/'+histFile+' '+basePath+'TTToHadronic'+year+'/'+histFile+' '+basePath+'TTToSemiLeptonic'+year+'/'+histFile+' '+basePath+'TTTo2L2Nu'+year+'/'+histFile
            cmd += '' if o.condor else ' > hadd.log'
            cmds.append(cmd)

    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)

    cmds = []
    if '2016' in years and '2017' in years and '2018' in years:
        for sample in ['data', 'TT']:
            cmd  = 'hadd -f '+basePath+sample+'RunII/'+histFile+' '
            cmd += basePath+sample+'2016/'+histFile+' '
            cmd += basePath+sample+'2017/'+histFile+' '
            cmd += basePath+sample+'2018/'+histFile+' '
            cmd += '' if o.condor else ' > hadd.log'
            cmds.append(cmd)
                
    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)


def root2h5():
    basePath = EOSOUTDIR if o.condor else outputBase
    cmds = []
    for year in years:
        if not o.subsample:
            for process in ['ZZ4b', 'ggZH4b', 'ZH4b']:
                subdir = process+year
                cmd = 'python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'
                cmd += ' -i '+basePath+subdir+'/picoAOD.root'
                cmds.append( cmd )

        picoAODs = ['picoAOD']
        if o.subsample:
            picoAODs = ['picoAOD_subsample_v%d'%vX for vX in range(10)]
        
        for picoAOD in picoAODs:
            for period in periods[year]:
                subdir = 'data'+year+period
                cmd = 'python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'
                cmd += ' -i '+basePath+subdir+'/%s.root'%picoAOD
                cmds.append( cmd )                

            for process in ['TTToHadronic', 'TTToSemiLeptonic', 'TTTo2L2Nu']:
                subdir = process+year
                cmd = 'python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py'
                cmd += ' -i '+basePath+subdir+'/%s.root'%picoAOD
                cmds.append( cmd )

    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)


def xrdcph5(direction='toEOS'):
    cmds = []
    TO   = EOSOUTDIR  if direction=='toEOS' else outputBase
    FROM = outputBase if direction=='toEOS' else EOSOUTDIR
    for year in years:
        # for process in ['ZZ4b', 'ggZH4b', 'ZH4b']:
        #     cmd = 'xrdcp -f '+FROM+process+year+'/picoAOD.h5 '+TO+process+year+'/picoAOD.h5'
        #     cmds.append( cmd )

        picoAODs = ['picoAOD']
        if o.subsample:
            picoAODs = ['picoAOD_subsample_v%d'%vX for vX in range(10)]

        for picoAOD in picoAODs:
            for period in periods[year]:
                cmd = 'xrdcp -f %sdata%s%s/%s.h5 %sdata%s%s/%s.h5'%(FROM, year, period, picoAOD, TO, year, period, picoAOD)
                cmds.append( cmd )                

            for process in ['TTToHadronic', 'TTToSemiLeptonic', 'TTTo2L2Nu']:
                cmd = 'xrdcp -f %s%s%s/%s.h5 %s%s%s/%s.h5'%(FROM, process, year, picoAOD, TO, process, year, picoAOD)
                cmds.append( cmd )

    for cmd in cmds: execute(cmd, o.execute)    


def h52root():
    basePath = EOSOUTDIR if o.condor else outputBase
    cmds = []
    for year in years:
        for process in ['ZZ4b', 'ggZH4b', 'ZH4b']:
            subdir = process+year
            cmd = 'python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py'
            cmd += ' -i '+basePath+subdir+'/picoAOD.h5'
            cmds.append( cmd )

        for period in periods[year]:
            subdir = 'data'+year+period
            cmd = 'python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py'
            cmd += ' -i '+basePath+subdir+'/picoAOD.h5'
            cmds.append( cmd )                

        for process in ['TTToHadronic', 'TTToSemiLeptonic', 'TTTo2L2Nu']:
            subdir = process+year
            cmd = 'python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py'
            cmd += ' -i '+basePath+subdir+'/picoAOD.h5'
            cmds.append( cmd )

    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)
    


def subtractTT():
    basePath = EOSOUTDIR if o.condor else outputBase
    histFile = 'hists'+('_j' if o.useJetCombinatoricModel else '')+('_r' if o.reweight else '')+'.root'
    if o.createPicoAOD == 'picoAOD.root': histFile = 'histsFromNanoAOD.root'
    cmds=[]
    for year in years:
        mkdir(basePath+'qcd'+year, o.execute)
        cmd  = 'python ZZ4b/nTupleAnalysis/scripts/subtractTT.py'
        cmd += ' -d   '+ basePath+'data'+year+'/'+histFile
        cmd += ' --tt '+ basePath+  'TT'+year+'/'+histFile
        cmd += ' -q   '+ basePath+ 'qcd'+year+'/'+histFile
        cmds.append( cmd )

    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)        

    cmds = []
    if '2016' in years and '2017' in years and '2018' in years:
        mkdir(basePath+'qcdRunII', o.execute)
        cmd  = 'hadd -f '+basePath+'qcdRunII/'+histFile+' '
        cmd += basePath+'qcd2016/'+histFile+' '
        cmd += basePath+'qcd2017/'+histFile+' '
        cmd += basePath+'qcd2018/'+histFile+' '
        cmd += '' if o.condor else ' > hadd.log'
        cmds.append( cmd )

    if o.condor:
        DAG.addGeneration()
    execute(cmds, o.execute, condor_dag=DAG)        


def doWeights():
    basePath = EOSOUTDIR if o.condor else outputBase
    if '2016' in years and '2017' in years and '2018' in years:
        weightYears = ['RunII']
    else:
        weightYears = years
    for year in weightYears:
        mkdir(gitRepoBase+'data'+year, o.execute)
        histFile = 'hists'+('_j' if o.useJetCombinatoricModel else '')+('_r' if o.reweight else '')+'.root'
        cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeWeights.py'
        cmd += ' -d   '+basePath+'data'+year+'/'+histFile
        cmd += ' --tt '+basePath+  'TT'+year+'/'+histFile
        cmd += ' -c '+JCMCut
        cmd += ' -o '+gitRepoBase+'data'+year+'/ ' 
        cmd += ' -r '+JCMRegion
        cmd += ' -w '+JCMVersion
        cmd += ' -y '+year
        cmd += ' -l '+lumiDict[year]
        execute(cmd, o.execute)


def doPlots(extraPlotArgs=''):
    plotYears = copy(years)
    if '2016' in years and '2017' in years and '2018' in years and 'RunII' not in years:
        plotYears += ['RunII']

    samples = ['data', 'TT', 'ZZ4b', 'ZH4b', 'ggZH4b', 'bothZH4b', 'ZZandZH4b']
    if not o.reweight: samples += ['qcd']

    if o.condor: # download hists because repeated EOS access makes plotting about 25% slower
        for year in plotYears:
            for sample in samples:
                hists = 'hists.root'
                if sample in ['data', 'TT', 'qcd']:
                    hists = 'hists'+('_j' if o.useJetCombinatoricModel else '')+('_r' if o.reweight else '')+'.root'
                cmd = 'xrdcp -f '+EOSOUTDIR+sample+year+'/'+hists +' '+ outputBase+sample+year+'/'+hists
                execute(cmd, o.execute)

    basePath = EOSOUTDIR if o.condor else outputBase    
    plots = 'plots'+('_j' if o.useJetCombinatoricModel else '')+('_r' if o.reweight else '')
    cmds=[]
    for year in plotYears:
        lumi = lumiDict[year]
        cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makePlots.py'
        #cmd += ' -i '+basePath # you can uncomment this if you want to make plots directly from EOS
        cmd += ' -o '+outputBase
        cmd += ' -p '+plots+' -l '+lumi+' -y '+year
        cmd += ' -j' if o.useJetCombinatoricModel else ''
        cmd += ' -r' if o.reweight else ''
        cmd += ' --doJECSyst' if o.doJECSyst else ''
        cmd += ' --histDetailLevel '+o.plotDetailLevel
        cmd += ' '+extraPlotArgs+' '
        cmds.append(cmd)

    babySit(cmds, o.execute, maxJobs=4)
    cmd = 'tar -C '+outputBase+' -zcf '+outputBase+plots+'.tar '+plots
    execute(cmd, o.execute)

#
# ML Stuff
#

## in my_env with ROOT and Pandas
# time python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5

## in mlenv4 on cmslpcgpu1
# time python ZZ4b/nTupleAnalysis/scripts/nTagClassifier.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5 -l1e-3 -p 0.4 -e 50
## take best model
# time python ZZ4b/nTupleAnalysis/scripts/nTagClassifier.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5 -m [best model] -u

## in my_env with ROOT and Pandas
# time python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5 -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root

def impactPlots(workspace, expected=True):
    fitType = 'exp' if expected else 'obs'
    cmd = 'combineTool.py -M Impacts -d ZZ4b/nTupleAnalysis/combine/%s.root --doInitialFit --setParameterRanges rZZ=-10,10:rZH=-10,10 --setParameters rZZ=1,rZH=1 --robustFit 1 %s -m 125'%(workspace, '-t -1' if expected else '')
    execute(cmd, o.execute)
    cmd = 'combineTool.py -M Impacts -d ZZ4b/nTupleAnalysis/combine/%s.root --doFits       --setParameterRanges rZZ=-10,10:rZH=-10,10 --setParameters rZZ=1,rZH=1 --robustFit 1 %s -m 125'%(workspace, '-t -1' if expected else '')
    execute(cmd, o.execute)
    cmd = 'combineTool.py -M Impacts -d ZZ4b/nTupleAnalysis/combine/%s.root -o impacts_%s_%s.json -m 125'%(workspace, workspace, fitType)
    execute(cmd, o.execute)
    cmd = 'plotImpacts.py -i impacts_%s_%s.json -o impacts_%s_%s_ZZ --POI rZZ'%(workspace, fitType, workspace, fitType)
    execute(cmd, o.execute)
    cmd = 'plotImpacts.py -i impacts_%s_%s.json -o impacts_%s_%s_ZH --POI rZH'%(workspace, fitType, workspace, fitType)
    execute(cmd, o.execute)

def doCombine():

    region='ZZZHSR'
    cut = 'passMDRs'

    JECSysts = ['']
    if o.doJECSyst: 
        JECSysts += JECSystList

    outFileData = 'ZZ4b/nTupleAnalysis/combine/hists.root'
    execute('rm '+outFileData, o.execute)
    outFileMix  = 'ZZ4b/nTupleAnalysis/combine/hists_closure.root'
    execute('rm '+outFileMix, o.execute)
    mixFile = 'ZZ4b/nTupleAnalysis/combine/hists_closure_MixedToUnmixed_3bMix4b_rWbW2_b0p60p3_SRNoHH_e25_os012.root' #hists_closure_3bMix4b_rWbW2_b0p60p3_SRNoHH.root'
    mixFile = 'ZZ4b/nTupleAnalysis/combine/hists_closure_MixedToUnmixed_3bMix4b_rWbW2_b0p60p3_SRNoHH.root' #hists_closure_3bMix4b_rWbW2_b0p60p3_SRNoHH.root'
    mixName = '3bMix4b_rWbW2_v0'
    order = {'zz':2, 'zh':3}

    for year in years:

        #for channel in ['zz','zh','zh_0_75','zh_75_150','zh_150_250','zh_250_400','zh_400_inf','zz_0_75','zz_75_150','zz_150_250','zz_250_400','zz_400_inf']:
        for channel in ['zz','zh']:
            #rebin = ''%s''%str([0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
            rebin = '5'
            #if '0_75' in channel or '400_inf' in channel: rebin = '5'
            var = 'SvB_ps_'+channel
            for signal in [nameTitle('ZZ','ZZ4b'), nameTitle('ZH','bothZH4b')]:
                for JECSyst in JECSysts:

                    #Sigmal templates to data file
                    cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/'+USER+'/nobackup/ZZ4b/'+signal.title+year+'/hists'+JECSyst+'.root'
                    cmd += ' -o '+outFileData+' -r '+region+' --var '+var+' --channel '+channel+year+' -n '+signal.name+JECSyst+' --tag four  --cut '+cut+' --rebin '+rebin
                    execute(cmd, o.execute)

                    #Signal templates to mixed data file
                    cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/'+USER+'/nobackup/ZZ4b/'+signal.title+year+'/hists'+JECSyst+'.root'
                    cmd += ' -o '+outFileMix +' -r '+region+' --var '+var+' --channel '+channel+year+' -n '+signal.name+JECSyst+' --tag four  --cut '+cut+' --rebin '+rebin
                    execute(cmd, o.execute)

            #Multijet template to data file
            cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/'+USER+'/nobackup/ZZ4b/data'+year+'/hists_j_r.root'
            cmd += ' -o '+outFileData+' -r '+region+' --var '+var+' --channel '+channel+year+' -n multijet --tag three --cut '+cut+' --rebin '+rebin#+' --errorScale 1.414 '
            execute(cmd, o.execute)

            #Multijet template to mixed data file
            cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i '+mixFile
            cmd += ' -o '+outFileMix +' --TDirectory '+mixName+'/'+channel+year+' --channel '+channel+year+' --var multijet -n multijet --rebin '+rebin#+' --errorScale 1.414 '
            execute(cmd, o.execute)

            closureSysts = read_parameter_file('ZZ4b/nTupleAnalysis/combine/closureResults_%s_order%d.txt'%(channel, order[channel]))
            for name, variation in closureSysts.iteritems():
                if 'LP' in name:
                    #Multijet closure systematic templates to data file
                    cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/'+USER+'/nobackup/ZZ4b/data'+year+'/hists_j_r.root'
                    cmd += ' -o '+outFileData+' -r '+region+' --var '+var+' --channel '+channel+year+' -f ''+variation+'' -n '+name+' --tag three --cut '+cut+' --rebin '+rebin#+' --errorScale 1.414 '
                    execute(cmd, o.execute)

                    #Multijet closure systematic templates to mixed data file
                    cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i '+mixFile
                    cmd += ' -o '+outFileMix +' --TDirectory '+mixName+'/'+channel+year+' --channel '+channel+year+' -f ''+variation+'' --var multijet -n '+name+' --rebin '+rebin#+' --errorScale 1.414 '
                    execute(cmd, o.execute)
                if 'spurious' in name:
                    #Spurious Sigmal template to data file
                    cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/'+USER+'/nobackup/ZZ4b/data'+year+'/hists_j_r.root'
                    cmd += ' -o '+outFileData+' -r '+region+' --var '+var+' --channel '+channel+year+' -n '+name+' --tag three --cut '+cut+' --rebin '+rebin
                    cmd += ' --addHist /uscms/home/%s/nobackup/ZZ4b/ZZandZH4b%s/hists.root,%s/fourTag/mainView/%s/%s,%f'%(USER, year, cut, region, var, variation)
                    execute(cmd, o.execute)

                    #Spurious Signal template to mixed data file
                    cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i '+mixFile
                    cmd += ' -o '+outFileMix +' --TDirectory '+mixName+'/'+channel+year+' --channel '+channel+year+' --var multijet -n '+name+' --rebin '+rebin
                    cmd += ' --addHist /uscms/home/%s/nobackup/ZZ4b/ZZandZH4b%s/hists.root,%s/fourTag/mainView/%s/%s,%f'%(USER, year, cut, region, var, variation)
                    execute(cmd, o.execute)
                    

            #ttbar template to data file
            cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/'+USER+'/nobackup/ZZ4b/TT'+year+'/hists_j_r.root'
            cmd += ' -o '+outFileData+' -r '+region+' --var '+var+' --channel '+channel+year+' -n ttbar    --tag four  --cut '+cut+' --rebin '+rebin
            execute(cmd, o.execute)

            #ttbar template to mixed data file
            cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i '+mixFile
            cmd += ' -o '+outFileMix +' --TDirectory '+mixName+'/'+channel+year+' --channel '+channel+year+' --var ttbar -n ttbar --rebin '+rebin
            execute(cmd, o.execute)

            #data_obs to data file
            cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/'+USER+'/nobackup/ZZ4b/data'+year+'/hists_j_r.root'
            cmd += ' -o '+outFileData+' -r '+region+' --var '+var+' --channel '+channel+year+' -n data_obs --tag four  --cut '+cut+' --rebin '+rebin
            execute(cmd, o.execute)

            #mix data_obs to mixed data file
            cmd  = 'python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i '+mixFile
            cmd += ' -o '+outFileMix +' --TDirectory '+mixName+'/'+channel+year+' --channel '+channel+year+' --var data_obs -n data_obs --rebin '+rebin
            execute(cmd, o.execute)

    ### Using https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/
    ### and https://github.com/cms-analysis/CombineHarvester
    cmd = "text2workspace.py ZZ4b/nTupleAnalysis/combine/combine.txt         -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/ZZ:rZZ[1,0,10]' --PO 'map=.*/ZH:rZH[1,0,10]' -v 2"
    execute(cmd, o.execute)
    cmd = "text2workspace.py ZZ4b/nTupleAnalysis/combine/combine_closure.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/ZZ:rZZ[1,0,10]' --PO 'map=.*/ZH:rZH[1,0,10]' -v 2"
    execute(cmd, o.execute)

    impactPlots('combine_closure', expected=False)
    impactPlots('combine',         expected=True)

    ### Independent fit
    # combine -M MultiDimFit  ZZ4b/nTupleAnalysis/combine/combine.root  -t -1 --setParameterRanges rZZ=-4,6:rZH=-4,6 --setParameters rZZ=1,rZH=1 --algo=grid --points=2500 -n rZZ_rZH_scan_2d -v 1
    # python plot_scan_2d.py  
    ### Assuming SM
    # cmd = 'combine -M MultiDimFit  ZZ4b/nTupleAnalysis/combine/combine.root  -t -1 --setParameterRanges rZZ=-4,6:rZH=-4,6 --setParameters rZZ=1,rZH=1 --algo singles --cl=0.68'
    # execute(cmd, False)
    # combine -M Significance ZZ4b/nTupleAnalysis/combine/combine.txt   -t -1 --expectSignal=1
    # combine -M Significance ZZ4b/nTupleAnalysis/combine/combineZZ.txt -t -1 --expectSignal=1
    # combine -M Significance ZZ4b/nTupleAnalysis/combine/combineZH.txt -t -1 --expectSignal=1

    # PostFitShapesFromWorkspace -d ZZ4b/nTupleAnalysis/combine/combine_closure.txt -w ZZ4b/nTupleAnalysis/combine/combine_closure.root -f ZZ4b/nTupleAnalysis/combine/combine_closure.root:fit_b --output combine_closure_shapes.root --postfit --sampling --print --total-shapes

#
# Run analysis
#
if o.makeFileList:
    makeFileList()

if o.makeFileListLeptons:
    makeFileListLeptons()

if o.condor:
    makeTARBALL()

if o.h52root:
    h52root()

if o.makeJECSyst:
    makeJECSyst()

if o.doSignal or o.doData or o.doTT:
    startEventLoopGeneration = copy( DAG.iG )
if o.doSignal:
    doSignal()

if o.doData or o.doTT:
    DAG.setGeneration( startEventLoopGeneration )
    doDataTT()

if o.doWeights:
    doWeights()

if o.root2h5:
    root2h5()

if o.xrdcph5:
    xrdcph5(o.xrdcph5)

if o.doQCD:
    subtractTT()

if o.condor:
    DAG.submit(o.execute)
    print('# Use condor_monitor to watch jobs once they have finished submitting')
    # if o.execute and DAG.jobLines:
    #     print '# wait 10s for DAG jobs to start before starting condor_monitor'
    #     time.sleep(10)
    # if DAG.jobLines:
    #     cmd = 'python nTupleAnalysis/python/condor_monitor.py'
    #     execute(cmd, o.execute)

if o.doAccxEff:
    doAccxEff()
    doPlots('-a')

if o.doPlots:
    doPlots('-m')

if o.doCombine:
    doCombine()

