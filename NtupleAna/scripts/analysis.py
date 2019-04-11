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
parser.add_option('-i',                                 dest="iteration",      default="0", help="Reweight iteration")
parser.add_option('-w',            action="store_true", dest="doWeights",      default=False, help="Calculate weights")
parser.add_option('-r', '--reweight',                   dest="reweight",       default="", help="Reweight file containing TSpline3 of nTagClassifier ratio")
parser.add_option('--plot',        action="store_true", dest="doPlots",        default=False, help="Make Plots")
parser.add_option('-p', '--createPicoAOD',              dest="createPicoAOD",  type="string", help="Create picoAOD with given name")
parser.add_option('-n', '--nevents',                    dest="nevents",        default="-1", help="Number of events to process. Default -1 for no limit.")
parser.add_option(      '--histogramming',              dest="histogramming",  default="1", help="Histogramming level. 0 to make no kinematic histograms. 1: only make histograms for full event selection, larger numbers add hists in reverse cutflow order.")
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
if o.iteration:
    dataFiles = [outputBase+"data"+year+period+"/picoAOD"+str(int(o.iteration)-1)+".root" for period in periods[year]]

signalFiles = ["ZZ4b/fileLists/ggZH4b"+year+".txt",
               "ZZ4b/fileLists/ZH4b"+year+".txt",
               #"ZZ4b/fileLists/ZZ4b"+year+".txt",
               ]

accxEffFiles = [outputBase+"ZH4b"+year+"/histsFromNanoAOD.root",
                outputBase+"ggZH4b"+year+"/histsFromNanoAOD.root",
                ]

# Jet Combinatoric Model
JCMRegion = "ZHSB"
JCMVersion = "00-00-01"
jetCombinatoricModel = outputBase+"data2018A/jetCombinatoricModel_"+JCMRegion+"_"+JCMVersion+"_iter"+o.iteration+".txt"
reweight = outputBase+"data2018A/reweight_"+JCMRegion+"_"+JCMVersion+"_iter"+o.iteration+".root"

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
        if jetCombinatoricModel: cmd += " -j "+jetCombinatoricModel
        if o.createPicoAOD: cmd += " -p "+o.createPicoAOD
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
    for data in dataFiles:
        cmd  = "procNtupleTest "+script
        cmd += " -i "+data
        cmd += " -o "+outputBase
        cmd += " -y "+year
        cmd += " --histogramming "+o.histogramming
        cmd += " --histFile hists"+o.iteration+".root"
        if jetCombinatoricModel: cmd += " -j "+jetCombinatoricModel
        if int(o.iteration): cmd += " -r "+reweight
        if o.createPicoAOD: cmd += " -p "+o.createPicoAOD
        jobs.append(watch(cmd))

    # wait for jobs to finish
    failedJobs = []
    if o.execute:
        failedJobs = waitForJobs(jobs, failedJobs)

def doWeights():
    cmd  = "python ZZ4b/NtupleAna/scripts/makeWeights.py -d /uscms/home/bryantp/nobackup/ZZ4b/data2018A/hists.root -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/ -r "+JCMRegion+" -w "+JCMVersion+" -i "+o.iteration
    execute(cmd)

def doPlots():
    plots = "plots"+o.iteration
    output = outputBase+plots
    cmd = "python ZZ4b/NtupleAna/scripts/makePlots.py -o "+outputBase+" -p "+plots
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

    region = "ZH"
    cut = "passDEtaBB"
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
    doData()

if o.doWeights:
    doWeights()

if o.doPlots:
    doPlots()

if o.doCombine:
    doCombine()
