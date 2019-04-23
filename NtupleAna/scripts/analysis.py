import time
import os, re
import sys
import subprocess
import shlex
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-s',            action="store_true", dest="doSignal",       default=False, help="Run signal MC")
parser.add_option('-a',            action="store_true", dest="doAccxEff",      default=False, help="Make Acceptance X Efficiency plots")
parser.add_option('-d',            action="store_true", dest="doData",         default=False, help="Run data")
parser.add_option('-i',                                 dest="iteration",      default="", help="Reweight iteration")
parser.add_option('-w',            action="store_true", dest="doWeights",      default=False, help="Fit jetCombinatoricModel and nJetClassifier TSpline")
parser.add_option('-j',            action="store_true", dest="useJetCombinatoricModel",       default=False, help="Use the jet combinatoric model")
parser.add_option('-r',            action="store_true", dest="reweight",       default=False, help="Do reweighting with nJetClassifier TSpline")
parser.add_option('--plot',        action="store_true", dest="doPlots",        default=False, help="Make Plots")
parser.add_option('-p', '--createPicoAOD',              dest="createPicoAOD",  type="string", help="Create picoAOD with given name")
parser.add_option('-n', '--nevents',                    dest="nevents",        default="-1", help="Number of events to process. Default -1 for no limit.")
parser.add_option(      '--histogramming',              dest="histogramming",  default="3", help="Histogramming level. 0 to make no kinematic histograms. 1: only make histograms for full event selection, larger numbers add hists in reverse cutflow order.")
parser.add_option('-c',            action="store_true", dest="doCombine",      default=False, help="Make CombineTool input hists")
o, a = parser.parse_args()

#
# Config
#
script     = "ZZ4b/NtupleAna/scripts/procNtupleTest_cfg.py"
year       = "2018"
lumiDict   = {"2018": "14.0e3"}
lumi       = lumiDict[year]
outputBase = "/uscms/home/bryantp/nobackup/ZZ4b/"

# File lists
periods = {"2016": "BCDEFGH",
           "2017": "",
           "2018": "A"}
dataFiles = ["ZZ4b/fileLists/data"+year+period+".txt" for period in periods[year]]
# Jet Combinatoric Model
JCMRegion = "ZHSB"
JCMVersion = "00-00-02"
jetCombinatoricModel = outputBase+"data2018A/jetCombinatoricModel_"+JCMRegion+"_"+JCMVersion+".txt"
reweight = outputBase+"data"+year+"A/reweight_"+JCMRegion+"_"+JCMVersion+".root"

signalFiles = ["ZZ4b/fileLists/ggZH4b"+year+".txt",
               "ZZ4b/fileLists/ZH4b"+year+".txt",
               #"ZZ4b/fileLists/ZZ4b"+year+".txt",
               ]

accxEffFiles = [outputBase+"ZH4b"+year+"/histsFromNanoAOD.root",
                outputBase+"ggZH4b"+year+"/histsFromNanoAOD.root",
                ]


def execute(command): # use to run command like a normal line of bash
    print command
    if o.execute: os.system(command)

def watch(command): # use to run a command and keep track of the thread, ie to run something when it is done
    print command
    if o.execute: return (command, subprocess.Popen(shlex.split(command)))

def babySit(job):
    tries = 1
    code = job[1].wait()
    print "# Command: "
    print "# Exited with: ",code
    while code and tries < 3:
        tries += 1
        print "# -----------------------------------------"
        print "# RELAUNCH JOB (ATTEMPT #"+str(tries)+"):"
        code = watch(job[0])[1].wait()

def waitForJobs(jobs,failedJobs):
    for job in jobs:
        code = job[1].wait()
        if code: failedJobs.append(job)
    return failedJobs

def relaunchJobs(jobs):
    print "# --------------------------------------------"
    print "# RELAUNCHING JOBS"
    newJobs = []
    for job in jobs: newJobs.append(watch(job[0]))
    return newJobs

def mkdir(directory):
    if not os.path.isdir(directory):
        print "mkdir",directory
        if o.execute: os.mkdir(directory)
    else:
        print "#",directory,"already exists"

def rmdir(directory):
    if not o.execute: 
        print "rm -r",directory
        return
    if "*" in directory:
        execute("rm -r "+directory)
        return
    if os.path.isdir(directory):
        execute("rm -r "+directory)
    elif os.path.exists(directory):
        execute("rm "+directory)
    else:
        print "#",directory,"does not exist"



def doSignal():
    mkdir(outputBase)

    jobs = []
    for signal in signalFiles:
        cmd  = "procNtupleTest "+script
        cmd += " -i "+signal
        cmd += " -o "+outputBase
        cmd += " -y "+year
        cmd += " -l "+lumi
        cmd += " --histogramming "+o.histogramming
        cmd += " -j "+jetCombinatoricModel if o.useJetCombinatoricModel else ""
        cmd += " -r "+reweight if o.reweight else ""
        cmd += " -p "+o.createPicoAOD if o.createPicoAOD else ""
        cmd += " --isMC"
        jobs.append(watch(cmd))

    # wait for jobs to finish
    failedJobs = []
    if o.execute:
        failedJobs = waitForJobs(jobs, failedJobs)

      
def doAccxEff():   
    jobs = []
    for signal in accxEffFiles:
        cmd += "python ZZ4b/NtupleAna/scripts/makeAccxEff.py -i "+signal
        jobs.append(watch(cmd))

    # wait for jobs to finish
    failedJobs = []
    if o.execute:
        failedJobs = waitForJobs(jobs, failedJobs)


def doData():
    mkdir(outputBase)

    jobs = []
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    for data in dataFiles:
        cmd  = "procNtupleTest "+script
        cmd += " -i "+data
        cmd += " -o "+outputBase
        cmd += " -y "+year
        cmd += " --histogramming "+o.histogramming
        cmd += " --histFile "+histFile
        cmd += " -j "+jetCombinatoricModel if o.useJetCombinatoricModel else ""
        cmd += " -r "+reweight if o.reweight else ""
        cmd += " -p "+o.createPicoAOD if o.createPicoAOD else ""
        jobs.append(watch(cmd))

    # wait for jobs to finish
    failedJobs = []
    if o.execute:
        failedJobs = waitForJobs(jobs, failedJobs)


def doWeights():
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    cmd  = "python ZZ4b/NtupleAna/scripts/makeWeights.py -d /uscms/home/bryantp/nobackup/ZZ4b/data2018A/"+histFile+" -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/ -r "+JCMRegion+" -w "+JCMVersion
    execute(cmd)


def doPlots():
    plots = "plots"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")
    output = outputBase+plots
    cmd  = "python ZZ4b/NtupleAna/scripts/makePlots.py -o "+outputBase+" -p "+plots
    cmd += " -j" if o.useJetCombinatoricModel else ""
    cmd += " -r" if o.reweight else ""
    execute(cmd)
    cmd = "tar -C "+outputBase+" -zcf "+output+".tar "+plots
    execute(cmd)

#
# ML Stuff
#

## in my_env with ROOT and Pandas
# time python ZZ4b/NtupleAna/scripts/convert_root2h5.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5

## in mlenv4 on cmslpcgpu1
# time python ZZ4b/NtupleAna/scripts/nTagClassifier.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5 -l1e-3 -p 0.4 -e 50
## take best model
# time python ZZ4b/NtupleAna/scripts/nTagClassifier.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5 -m [best model] -u

## in my_env with ROOT and Pandas
# time python ZZ4b/NtupleAna/scripts/convert_h52root.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5 -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root

def doCombine():
    outFile = "combineTest.root"
    execute("rm "+outFile)

    region = "inclusive"
    #cut = "passDEtaBB"
    cut = "passMDRs"
    #var = "mZH"
    var = "ZHvsBackgroundClassifier"
    #var = "xZH"
    cmd = "python ZZ4b/NtupleAna/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/bothZH4b2018/hists.root -o "+outFile+" -r "+region+" --var "+var+" -n ZH --tag four --cut "+cut
    execute(cmd)
    cmd = "python ZZ4b/NtupleAna/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/hists1.root -o "+outFile+" -r "+region+" --var "+var+" -n multijet --tag three --cut "+cut
    execute(cmd)
    cmd = "python ZZ4b/NtupleAna/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/hists1.root -o "+outFile+" -r "+region+" --var "+var+" -n data_obs --tag four --cut "+cut
    execute(cmd)
    # cd /uscms/homes/b/bryantp/work/CMSSW_8_1_0/src
    # combine -M Significance ../../CMSSW_10_2_0/src/combineTest.txt -t -1 --expectSignal=1

#
# Run analysis
#
if o.doSignal:
    doSignal()

if o.doAccxEff:
    doAccxEff()

if o.doData:
    #make picoAOD0.root for nTagClassifier training with jetCombinatoricModel weights applied but not nTagClassifier reweight
    #procNtupleTest ZZ4b/NtupleAna/scripts/procNtupleTest_cfg.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root -o /uscms/home/bryantp/nobackup/ZZ4b/ -y 2018 --histogramming 0 --histFile hists.root -j /uscms/home/bryantp/nobackup/ZZ4b/data2018A/jetCombinatoricModel_ZHSB_00-00-02_iter0.txt -p picoAOD0.root
    #convert to .h5 for nTagClassifier training
    #py ZZ4b/NtupleAna/scripts/convert_root2h5.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.root -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.h5
    #update .h5 file after training
    #update .root file with nTagClassifier output
    #python ZZ4b/NtupleAna/scripts/convert_h52root.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.h5 -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.root
    #create hists for nTagClassifier reweight
    #procNtupleTest ZZ4b/NtupleAna/scripts/procNtupleTest_cfg.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.root -o /uscms/home/bryantp/nobackup/ZZ4b/ -y 2018 --histogramming 3 --histFile hists0.root -j /uscms/home/bryantp/nobackup/ZZ4b/data2018A/jetCombinatoricModel_ZHSB_00-00-02_iter0.txt
    #make reweighting spline
    #python ZZ4b/NtupleAna/scripts/makeWeights.py -d /uscms/home/bryantp/nobackup/ZZ4b/data2018A/hists0.root -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/ -r ZHSB -w 00-00-02 -i 0
    doData()

if o.doWeights:
    doWeights()

if o.doPlots:
    doPlots()

if o.doCombine:
    doCombine()
