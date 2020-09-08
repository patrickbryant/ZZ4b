import time
import copy
import textwrap
import os, re
import sys
import subprocess
import shlex
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

def getCMSSW():
    return os.getenv('CMSSW_VERSION')

def getUSER():
    return os.getenv('USER')

CMSSW = getCMSSW()
USER = getUSER()
EOSOUTDIR = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"
TARBALL   = "root://cmseos.fnal.gov//store/user/"+USER+"/condor/"+CMSSW+".tgz"

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-s',            action="store_true", dest="doSignal",       default=False, help="Run signal MC")
parser.add_option('-t',            action="store_true", dest="doTT",       default=False, help="Run ttbar MC")
parser.add_option('-a',            action="store_true", dest="doAccxEff",      default=False, help="Make Acceptance X Efficiency plots")
parser.add_option('-d',            action="store_true", dest="doData",         default=False, help="Run data")
parser.add_option('-q',            action="store_true", dest="doQCD",          default=False, help="Subtract ttbar MC from data to make QCD template")
parser.add_option('-y',                                 dest="year",      default="2018", help="Year or comma separated list of years")
parser.add_option('-o',                                 dest="outputBase",      default="/uscms/home/"+USER+"/nobackup/ZZ4b/", help="path to output")
parser.add_option('-w',            action="store_true", dest="doWeights",      default=False, help="Fit jetCombinatoricModel and nJetClassifier TSpline")
parser.add_option('--makeJECSyst', action="store_true", dest="makeJECSyst",    default=False, help="Make jet energy correction systematics friend TTrees")
parser.add_option('--doJECSyst',   action="store_true", dest="doJECSyst",      default=False, help="Run event loop for jet energy correction systematics")
parser.add_option('-j',            action="store_true", dest="useJetCombinatoricModel",       default=False, help="Use the jet combinatoric model")
parser.add_option('-r',            action="store_true", dest="reweight",       default=False, help="Do reweighting with nJetClassifier TSpline")
parser.add_option('--bTagSyst',    action="store_true", dest="bTagSyst",       default=False, help="run btagging systematics")
parser.add_option('--plot',        action="store_true", dest="doPlots",        default=False, help="Make Plots")
parser.add_option('-p', '--createPicoAOD',              dest="createPicoAOD",  type="string", help="Create picoAOD with given name")
parser.add_option('-f', '--fastSkim',                   dest="fastSkim",       action="store_true", default=False, help="Do fast picoAOD skim")
parser.add_option(      '--looseSkim',                  dest="looseSkim",      action="store_true", default=False, help="Relax preselection to make picoAODs for JEC Uncertainties which can vary jet pt by a few percent.")
parser.add_option('-n', '--nevents',                    dest="nevents",        default="-1", help="Number of events to process. Default -1 for no limit.")
parser.add_option(      '--histogramming',              dest="histogramming",  default="1", help="Histogramming level. 0 to make no kinematic histograms. 1: only make histograms for full event selection, larger numbers add hists in reverse cutflow order.")
parser.add_option(      '--detailLevel',              dest="detailLevel",  default="1", help="Histogramming detail level. 3 makes allView hists, 5, 7 make ZZ, ZH specific region plots")
parser.add_option('-c', '--doCombine',    action="store_true", dest="doCombine",      default=False, help="Make CombineTool input hists")
parser.add_option(   '--loadHemisphereLibrary',    action="store_true", default=False, help="load Hemisphere library")
parser.add_option(   '--noDiJetMassCutInPicoAOD',    action="store_true", default=False, help="create Output Hemisphere library")
parser.add_option(   '--createHemisphereLibrary',    action="store_true", default=False, help="create Output Hemisphere library")
parser.add_option(   '--maxNHemis',    default=10000, help="Max nHemis to load")
parser.add_option(   '--inputHLib3Tag', default='$PWD/data18/hemiSphereLib_3TagEvents_*root',          help="Base path for storing output histograms and picoAOD")
parser.add_option(   '--inputHLib4Tag', default='$PWD/data18/hemiSphereLib_4TagEvents_*root',           help="Base path for storing output histograms and picoAOD")
parser.add_option(   '--SvB_ONNX', action="store_true", default=False,           help="Run ONNX version of SvB model. Model path specified in analysis.py script")
parser.add_option(   '--condor',   action="store_true", default=False,           help="Run on condor")
o, a = parser.parse_args()


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
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -s --histogramming 0 -y 2016,2017,2018 -p none -a -e

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
nWorkers   = 3
script     = "ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py"
years      = o.year.split(",")
lumiDict   = {"2016":  "35.9e3",#35.8791
              "2017":  "36.7e3",#36.7338
              "2018":  "60.0e3",#59.9656
              "17+18": "96.7e3",
              "RunII":"132.6e3",
              }
bTagDict   = {"2016": "0.6",#"0.3093", #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation2016Legacy
              "2017": "0.6",#"0.3033", #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X
              "2018": "0.6",#"0.2770"} #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
              }
outputBase = o.outputBase
#outputBase = os.getcwd()+"/"
gitRepoBase= 'ZZ4b/nTupleAnalysis/weights/'

# File lists
periods = {"2016": "BCDEFGH",
           "2017": "CDEF",
           "2018": "ABCD"}

JECSystList = ["_jerUp", "_jerDown",
               "_jesTotalUp", "_jesTotalDown"]

def dataFiles(year):
    return ["ZZ4b/fileLists/data"+year+period+".txt" for period in periods[year]]

# Jet Combinatoric Model
JCMRegion = "SB"
JCMVersion = "00-00-02"
JCMCut = "passMDRs"
def jetCombinatoricModel(year):
    #return gitRepoBase+"data"+year+"/jetCombinatoricModel_"+JCMRegion+"_"+JCMVersion+".txt"
    return gitRepoBase+"dataRunII/jetCombinatoricModel_"+JCMRegion+"_"+JCMVersion+".txt"
#reweight = gitRepoBase+"data"+year+"/reweight_"+JCMRegion+"_"+JCMVersion+".root"

SvB_ONNX = "ZZ4b/nTupleAnalysis/pytorchModels/SvB_ResNet_9_9_9_np1713_lr0.008_epochs40_stdscale_epoch40_loss0.2138.onnx"

def signalFiles(year):
    files = ["ZZ4b/fileLists/ggZH4b"+year+".txt",
             "ZZ4b/fileLists/ZH4b"+year+".txt",
             "ZZ4b/fileLists/ZZ4b"+year+".txt",
             ]
    return files

def ttbarFiles(year):
    files = ["ZZ4b/fileLists/TTToHadronic"+year+".txt",
             "ZZ4b/fileLists/TTToSemiLeptonic"+year+".txt",
             "ZZ4b/fileLists/TTTo2L2Nu"+year+".txt",
             #"ZZ4b/fileLists/TTJets"+year+".txt",
             ]
    return files

def accxEffFiles(year):
    files = [outputBase+"ZZ4b"+year+"/histsFromNanoAOD.root",
             #outputBase+"ZH4b"+year+"/histsFromNanoAOD.root",
             #outputBase+"ggZH4b"+year+"/histsFromNanoAOD.root",
             outputBase+"bothZH4b"+year+"/histsFromNanoAOD.root",
             outputBase+"ZZandZH4b"+year+"/histsFromNanoAOD.root",
             ]
    return files


def makeJECSyst():
    cmds=[]
    for year in years:
        for process in ['ZZ4b', 'ZH4b', 'ggZH4b']:
            cmd  = 'python PhysicsTools/NanoAODTools/scripts/nano_postproc.py '
            cmd += outputBase+process+year+'/ '
            cmd += outputBase+process+year+'/picoAOD.root '
            cmd += '--friend '
            cmd += '-I nTupleAnalysis.baseClasses.jetmetCorrectors jetmetCorrector'+year # modules are defined in https://github.com/patrickbryant/nTupleAnalysis/blob/master/baseClasses/python/jetmetCorrectors.py
            cmds.append(cmd)
    babySit(cmds, o.execute)

def makeTARBALL():
    base="/uscms/home/"+USER+"/nobackup/"
    if os.path.exists(base+CMSSW+".tgz"):
        print "TARBALL already exists, skip making it"
        return
    cmd  = 'tar -C '+base+' -zcvf '+base+CMSSW+'.tgz '+CMSSW
    cmd += ' --exclude="*.pdf" --exclude="*.jdl" --exclude="*.stdout" --exclude="*.stderr" --exclude="*.log"'
    cmd += ' --exclude=".git" --exclude="PlotTools" --exclude="madgraph" --exclude="*.pkl" --exclude="*.root"'
    cmd += ' --exclude="tmp" --exclude="combine" --exclude-vcs --exclude-caches-all'
    execute(cmd, o.execute)
    cmd  = 'ls '+base+' -alh'
    execute(cmd, o.execute)
    cmd = "xrdfs root://cmseos.fnal.gov/ mkdir /store/user/"+USER+"/condor"
    execute(cmd, o.execute)
    cmd = "xrdcp -f "+base+CMSSW+".tgz "+TARBALL
    execute(cmd, o.execute)
    

def doSignal():
    mkdir(outputBase, o.execute)

    cmds=[]
    JECSysts = [""]
    if o.doJECSyst: 
        JECSysts = JECSystList

    for JECSyst in JECSysts:
        histFile = "hists"+JECSyst+".root" #+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
        if o.createPicoAOD == "picoAOD.root" or o.createPicoAOD == "none": histFile = "histsFromNanoAOD"+JECSyst+".root"

        for year in years:
            #if year == "2016": continue
            lumi = lumiDict[year]
            for signal in signalFiles(year):
                cmd  = "nTupleAnalysis "+script
                cmd += " -i "+signal
                cmd += " -o "+outputBase
                cmd += " -y "+year
                cmd += " -l "+lumi
                cmd += " --histogramming "+o.histogramming
                cmd += " --histDetailLevel "+o.detailLevel
                cmd += " --histFile "+histFile
                cmd += " -j "+jetCombinatoricModel(year) if o.useJetCombinatoricModel else ""
                cmd += " -r " if o.reweight else ""
                cmd += " -p "+o.createPicoAOD if o.createPicoAOD else ""
                cmd += " -f " if o.fastSkim else ""
                cmd += " --isMC"
                cmd += " --bTag "+bTagDict[year]
                cmd += " --bTagSF"
                cmd += " --bTagSyst" if o.bTagSyst else ""
                cmd += " --nevents "+o.nevents
                cmd += " --looseSkim" if o.looseSkim else ""
                cmd += " --SvB_ONNX "+SvB_ONNX if o.SvB_ONNX else ""
                cmd += " --JECSyst "+JECSyst if JECSyst else ""

                if o.condor:
                    cmd += " --condor"
                    subdir = signal.split("/")[-1].replace(".txt","")
                    thisJDL = jdl(CMSSW=CMSSW, EOSOUTDIR=EOSOUTDIR+subdir, TARBALL=TARBALL, cmd=cmd)
                    thisJDL.make()
                    cmd = "condor_submit "+thisJDL.file

                cmds.append(cmd)

    # wait for jobs to finish
    if len(cmds)>1:
        if o.condor:
            for cmd in cmds: execute(cmd, o.execute)
        else:
            babySit(cmds, o.execute, maxJobs=nWorkers)
    else:
        execute(cmd, o.execute)

    if o.condor: 
        print "NEED A WAY TO WAIT UNTIL CONDOR IS DONE TO DO NEXT STEPS"
        return

    for year in years:

        for JECSyst in JECSysts:
            histFile = "hists"+JECSyst+".root" #+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
            if o.createPicoAOD == "picoAOD.root" or o.createPicoAOD == "none": histFile = "histsFromNanoAOD"+JECSyst+".root"

            if o.createPicoAOD and o.createPicoAOD != "none":
                if o.createPicoAOD != "picoAOD.root":
                    for sample in ["ZH4b", "ggZH4b", "ZZ4b"]:
                        cmd = "cp "+outputBase+sample+year+"/"+o.createPicoAOD+" "+outputBase+sample+year+"/picoAOD.root"
                        execute(cmd, o.execute)

            files = signalFiles(year)
            if "ZZ4b/fileLists/ZH4b"+year+".txt" in files and "ZZ4b/fileLists/ggZH4b"+year+".txt" in files:
                mkdir(outputBase+"bothZH4b"+year, o.execute)
                cmd = "hadd -f "+outputBase+"bothZH4b"+year+"/"+histFile+" "+outputBase+"ZH4b"+year+"/"+histFile+" "+outputBase+"ggZH4b"+year+"/"+histFile+" > hadd.log"
                execute(cmd, o.execute)

            if "ZZ4b/fileLists/ZH4b"+year+".txt" in files and "ZZ4b/fileLists/ggZH4b"+year+".txt" in files and "ZZ4b/fileLists/ZZ4b"+year+".txt" in files:
                mkdir(outputBase+"ZZandZH4b"+year, o.execute)
                cmd = "hadd -f "+outputBase+"ZZandZH4b"+year+"/"+histFile+" "+outputBase+"ZH4b"+year+"/"+histFile+" "+outputBase+"ggZH4b"+year+"/"+histFile+" "+outputBase+"ZZ4b"+year+"/"+histFile+" > hadd.log"
                execute(cmd, o.execute)

    if "2016" in years and "2017" in years and "2018" in years:
        cmds = []
        for JECSyst in JECSysts:
            histFile = "hists"+JECSyst+".root" #+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
            if o.createPicoAOD == "picoAOD.root" or o.createPicoAOD == "none": histFile = "histsFromNanoAOD"+JECSyst+".root"
            cmd  = "hadd -f "+outputBase+"ZZ4bRunII/"+histFile+" "
            cmd += outputBase+"ZZ4b2016/"+histFile+" "
            cmd += outputBase+"ZZ4b2017/"+histFile+" "
            cmd += outputBase+"ZZ4b2018/"+histFile+" "
            cmds.append(cmd)

            cmd  = "hadd -f "+outputBase+"bothZH4bRunII/"+histFile+" "
            cmd += outputBase+"bothZH4b2016/"+histFile+" "
            cmd += outputBase+"bothZH4b2017/"+histFile+" "
            cmd += outputBase+"bothZH4b2018/"+histFile+" "
            cmds.append(cmd)

            cmd  = "hadd -f "+outputBase+"ZZandZH4bRunII/"+histFile+" "
            cmd += outputBase+"ZZandZH4b2016/"+histFile+" "
            cmd += outputBase+"ZZandZH4b2017/"+histFile+" "
            cmd += outputBase+"ZZandZH4b2018/"+histFile+" "
            cmds.append(cmd)

        babySit(cmds, o.execute, maxJobs=nWorkers)

      
def doAccxEff():   
    cmds = []

    plotYears = copy.copy(years)
    if "2016" in years and "2017" in years and "2018" in years:
        plotYears += ["RunII"]

    for year in plotYears:
        for signal in accxEffFiles(year):
            cmd = "python ZZ4b/nTupleAnalysis/scripts/makeAccxEff.py -i "+signal
            cmds.append(cmd)
    babySit(cmds, o.execute)

def doDataTT():
    mkdir(outputBase, o.execute)

    # run event loop
    cmds=[]
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    if o.createPicoAOD == "picoAOD.root": histFile = "histsFromNanoAOD.root"

    for year in years:
        files = []
        if o.doData: files += dataFiles(year)
        if o.doTT:   files += ttbarFiles(year)
        lumi = lumiDict[year]
        for f in files:
            cmd  = "nTupleAnalysis "+script
            cmd += " -i "+f
            cmd += " -o "+outputBase
            cmd += " -y "+year
            cmd += " --histogramming "+o.histogramming
            cmd += " --histDetailLevel "+"1"
            cmd += " --histFile "+histFile
            cmd += " -j "+jetCombinatoricModel(year) if o.useJetCombinatoricModel else ""
            cmd += " -r " if o.reweight else ""
            cmd += " -p "+o.createPicoAOD if o.createPicoAOD else ""
            cmd += " -f " if o.fastSkim else ""
            cmd += " --bTag "+bTagDict[year]
            cmd += " --nevents "+o.nevents
            if f in ttbarFiles(year):
                cmd += " --bTagSF"
            
    #cmd += " --bTagSyst" if o.bTagSyst else ""
                cmd += " -l "+lumi
                cmd += " --isMC "
            if o.createHemisphereLibrary  and f not in ttbarFiles:
                cmd += " --createHemisphereLibrary "
            if o.noDiJetMassCutInPicoAOD:
                cmd += " --noDiJetMassCutInPicoAOD "
            if o.loadHemisphereLibrary:
                cmd += " --loadHemisphereLibrary "
                cmd += " --inputHLib3Tag "+o.inputHLib3Tag
                cmd += " --inputHLib4Tag "+o.inputHLib4Tag
            cmd += " --SvB_ONNX "+SvB_ONNX if o.SvB_ONNX else ""

            if o.condor:
                cmd += " --condor"
                subdir = f.split("/")[-1].replace(".txt","")
                thisJDL = jdl(CMSSW=CMSSW, EOSOUTDIR=EOSOUTDIR+subdir, TARBALL=TARBALL, cmd=cmd)
                thisJDL.make()
                cmd = "condor_submit "+thisJDL.file

            cmds.append(cmd)

    # wait for jobs to finish
    if len(cmds)>1:
        if o.condor:
            for cmd in cmds: execute(cmd, o.execute)
        else:
            babySit(cmds, o.execute, maxJobs=nWorkers)
    else:
        execute(cmd, o.execute)

    if o.condor: 
        print "NEED A WAY TO WAIT UNTIL CONDOR IS DONE TO DO NEXT STEPS"
        return

    cmds = []
    for year in years:
        # overwrite nominal picoAOD with newly created one
        if o.createPicoAOD and o.createPicoAOD != "picoAOD.root":
            if o.doData:
                for period in periods[year]:
                    cmd = "cp "+outputBase+"data"+year+period+"/"+o.createPicoAOD+" "+outputBase+"data"+year+period+"/picoAOD.root"
                    cmds.append(cmd)
            if o.doTT:
                for sample in ["TTToHadronic","TTToSemiLeptonic","TTTo2L2Nu"]: #,"TTJets"]
                    cmd = "cp "+outputBase+sample+year+"/"+o.createPicoAOD+" "+outputBase+sample+year+"/picoAOD.root"
                    cmds.append(cmd)
    babySit(cmds, o.execute)

    # make combined histograms for plotting purposes
    cmds = []
    for year in years:
        if o.doData:
            mkdir(outputBase+"data"+year, o.execute)
            cmd = "hadd -f "+outputBase+"data"+year+"/"+histFile+" "+" ".join([outputBase+"data"+year+period+"/"+histFile for period in periods[year]])#+" > hadd.log"
            cmds.append(cmd)
    
        if o.doTT:
            files = ttbarFiles(year)
            if "ZZ4b/fileLists/TTToHadronic"+year+".txt" in files and "ZZ4b/fileLists/TTToSemiLeptonic"+year+".txt" in files and "ZZ4b/fileLists/TTTo2L2Nu"+year+".txt" in files:
                mkdir(outputBase+"TT"+year, o.execute)
                cmd = "hadd -f "+outputBase+"TT"+year+"/"+histFile+" "+outputBase+"TTToHadronic"+year+"/"+histFile+" "+outputBase+"TTToSemiLeptonic"+year+"/"+histFile+" "+outputBase+"TTTo2L2Nu"+year+"/"+histFile#+" > hadd.log"
                cmds.append(cmd)
    babySit(cmds, o.execute)

    if "2016" in years and "2017" in years and "2018" in years:
        cmds = []
        cmd  = "hadd -f "+outputBase+"dataRunII/"+histFile+" "
        cmd += outputBase+"data2016/"+histFile+" "
        cmd += outputBase+"data2017/"+histFile+" "
        cmd += outputBase+"data2018/"+histFile+" "
        cmds.append(cmd)

        cmd  = "hadd -f "+outputBase+"TTRunII/"+histFile+" "
        cmd += outputBase+"TT2016/"+histFile+" "
        cmd += outputBase+"TT2017/"+histFile+" "
        cmd += outputBase+"TT2018/"+histFile+" "
        cmds.append(cmd)
        babySit(cmds, o.execute)


def subtractTT():
    cmds=[]
    for year in years:
        mkdir(outputBase+"qcd"+year, o.execute)
        histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
        if o.createPicoAOD == "picoAOD.root": histFile = "histsFromNanoAOD.root"
        cmd  = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py"
        cmd += " -d   "+ outputBase+"data"+year+"/"+histFile
        cmd += " --tt "+ outputBase+  "TT"+year+"/"+histFile
        cmd += " -q   "+ outputBase+ "qcd"+year+"/"+histFile
        cmds.append(cmd)
    babySit(cmds, o.execute)
    if "2016" in years and "2017" in years and "2018" in years:
        cmds = []
        cmd  = "hadd -f "+outputBase+"qcdRunII/"+histFile+" "
        cmd += outputBase+"qcd2016/"+histFile+" "
        cmd += outputBase+"qcd2017/"+histFile+" "
        cmd += outputBase+"qcd2018/"+histFile+" "
        cmds.append(cmd)
        babySit(cmds, o.execute)


def doWeights():
    for year in years:
        mkdir(gitRepoBase+"data"+year, o.execute)
        histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
        cmd  = "python ZZ4b/nTupleAnalysis/scripts/makeWeights.py"
        cmd += " -d   "+ outputBase+"data"+year+"/"+histFile
        cmd += " --tt "+ outputBase+  "TT"+year+"/"+histFile
        cmd += " -c "+JCMCut
        cmd += " -o "+gitRepoBase+"data"+year+"/ " 
        cmd += " -r "+JCMRegion
        cmd += " -w "+JCMVersion
        cmd += " -y "+year
        cmd += " -l "+lumiDict[year]
        execute(cmd, o.execute)


def doPlots(extraPlotArgs=""):
    plots = "plots"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")
    output = outputBase+plots
    cmds=[]
    
    plotYears = copy.copy(years)
    if "2016" in years and "2017" in years and "2018" in years:
        plotYears += ["RunII"]

    for year in plotYears:
        lumi = lumiDict[year]
        cmd  = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputBase+" -p "+plots+" -l "+lumi+" -y "+year
        cmd += " -j" if o.useJetCombinatoricModel else ""
        cmd += " -r" if o.reweight else ""
        cmd += " --doJECSyst" if o.doJECSyst else ""
        cmd += " "+extraPlotArgs+" "
        cmds.append(cmd)

    babySit(cmds, o.execute)
    cmd = "tar -C "+outputBase+" -zcf "+output+".tar "+plots
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

def doCombine():

    region="SR"
    cut = "passMDRs"

    JECSysts = [""]
    if o.doJECSyst: 
        JECSysts += JECSystList

    outFile = "ZZ4b/nTupleAnalysis/combine/hists.root"
    execute("rm "+outFile, o.execute)

    for year in years:

        #for channel in ['zz','zh','zh_0_75','zh_75_150','zh_150_250','zh_250_400','zh_400_inf','zz_0_75','zz_75_150','zz_150_250','zz_250_400','zz_400_inf']:
        for channel in ['zz','zh']:
            rebin = '4'
            if '0_75' in channel or '400_inf' in channel: rebin = '5'
            var = "SvB_ps_"+channel
            for signal in [nameTitle('ZZ','ZZ4b'), nameTitle('ZH','bothZH4b')]:
                for JECSyst in JECSysts:
                    cmd  = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/"+USER+"/nobackup/ZZ4b/"+signal.title+year+"/hists"+JECSyst+".root"
                    cmd += " -o "+outFile+" -r "+region+" --var "+var+" --channel "+channel+year+" -n "+signal.name+JECSyst+" --tag four  --cut "+cut+" --rebin "+rebin
                    execute(cmd, o.execute)
            cmd  = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/"+USER+"/nobackup/ZZ4b/data"+year+"/hists_j_r.root"
            cmd += " -o "+outFile+" -r "+region+" --var "+var+" --channel "+channel+year+" -n multijet --tag three --cut "+cut+" --rebin "+rebin
            execute(cmd, o.execute)

            closureSysts = read_parameter_file("ZZ4b/nTupleAnalysis/combine/closureResults_%s.txt"%channel)
            for name, variation in closureSysts.iteritems():
                cmd  = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/"+USER+"/nobackup/ZZ4b/data"+year+"/hists_j_r.root"
                cmd += " -o "+outFile+" -r "+region+" --var "+var+" --channel "+channel+year+" -f '"+variation+"' -n "+name+" --tag three --cut "+cut+" --rebin "+rebin
                execute(cmd, o.execute)

            cmd  = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/"+USER+"/nobackup/ZZ4b/TT"+year+"/hists_j_r.root"
            cmd += " -o "+outFile+" -r "+region+" --var "+var+" --channel "+channel+year+" -n ttbar    --tag four  --cut "+cut+" --rebin "+rebin
            execute(cmd, o.execute)
            cmd  = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/"+USER+"/nobackup/ZZ4b/data"+year+"/hists_j_r.root"
            cmd += " -o "+outFile+" -r "+region+" --var "+var+" --channel "+channel+year+" -n data_obs --tag four  --cut "+cut+" --rebin "+rebin
            execute(cmd, o.execute)

    ### Using https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/
    ### and https://github.com/cms-analysis/CombineHarvester
    # text2workspace.py ZZ4b/nTupleAnalysis/combine/combine.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/ZZ:rZZ[1,0,10]' --PO 'map=.*/ZH:rZH[1,0,10]' -v 2
    ### Independent fit
    # combine -M MultiDimFit  ZZ4b/nTupleAnalysis/combine/combine.root  -t -1 --setParameterRanges rZZ=-4,6:rZH=-4,6 --setParameters rZZ=1,rZH=1 --algo=grid --points=2500 -n rZZ_rZH_scan_2d -v 1
    # python plot_scan_2d.py  
    ### Assuming SM
    # combine -M MultiDimFit  ZZ4b/nTupleAnalysis/combine/combine.root  -t -1 --setParameterRanges rZZ=-4,6:rZH=-4,6 --setParameters rZZ=1,rZH=1 --algo singles --cl=0.68 
    # combine -M Significance ZZ4b/nTupleAnalysis/combine/combine.txt   -t -1 --expectSignal=1
    # combine -M Significance ZZ4b/nTupleAnalysis/combine/combineZZ.txt -t -1 --expectSignal=1
    # combine -M Significance ZZ4b/nTupleAnalysis/combine/combineZH.txt -t -1 --expectSignal=1
    ### Make Pull plot
    # combineTool.py -M Impacts -d ZZ4b/nTupleAnalysis/combine/combine.root --doInitialFit -t -1 --setParameterRanges rZZ=-4,6:rZH=-4,6 --setParameters rZZ=1,rZH=1 --robustFit 1 -m 125
    # combineTool.py -M Impacts -d ZZ4b/nTupleAnalysis/combine/combine.root --doFits       -t -1 --setParameterRanges rZZ=-4,6:rZH=-4,6 --setParameters rZZ=1,rZH=1 --robustFit 1 -m 125
    # combineTool.py -M Impacts -d ZZ4b/nTupleAnalysis/combine/combine.root -o impacts.json -m 125
    # plotImpacts.py -i impacts.json -o impacts

#
# Run analysis
#
if o.condor:
    makeTARBALL()

if o.makeJECSyst:
    makeJECSyst()

if o.doSignal:
    doSignal()

if o.doAccxEff:
    doAccxEff()
    doPlots("-a")

if o.doData or o.doTT:
    doDataTT()

if o.doQCD:
    subtractTT()

if o.doWeights:
    doWeights()

if o.doPlots:
    doPlots("-m")

if o.doCombine:
    doCombine()
