import time
import textwrap
import os, re
import sys
import subprocess
import shlex
import optparse
from threading import Thread
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # python 2.x

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
        if queue.qsize()>1: queue.get_nowait()
    out.close()

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
parser.add_option(      '--histogramming',              dest="histogramming",  default="1", help="Histogramming level. 0 to make no kinematic histograms. 1: only make histograms for full event selection, larger numbers add hists in reverse cutflow order.")
parser.add_option('-c',            action="store_true", dest="doCombine",      default=False, help="Make CombineTool input hists")
o, a = parser.parse_args()

#
# Analysis in four "easy" steps
#

### 1. Jet Combinatoric Model
# First run on data
# > python ZZ4b/NtupleAna/scripts/analysis.py -d -e
# Then make jet combinatoric model 
# > python ZZ4b/NtupleAna/scripts/analysis.py -w -e
# Now run again and update the automatically generated picoAOD by making a temporary one which will then be copied over picoAOD.root
# > python ZZ4b/NtupleAna/scripts/analysis.py -d -j -p tempPicoAOD.root -e

### 2. ThreeTag to FourTag reweighting
# Now convert the picoAOD to hdf5 to train the Four Vs Three tag classifier (FvT)
# > python ZZ4b/NtupleAna/scripts/convert_root2h5.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.root"
# Now train the classifier
# > python ZZ4b/NtupleAna/scripts/signalBackgroundClassifier.py -b "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5"
# Take the best result and update the hdf5 files with classifier output for each event
# > py ZZ4b/NtupleAna/scripts/signalBackgroundClassifier.py -b "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5" -m <the best model> -u
# Update the picoAOD.root with the result
# > python ZZ4b/NtupleAna/scripts/convert_h52root.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5"
# Now run the data again so that you can compute the FvT reweighting 
# > python ZZ4b/NtupleAna/scripts/analysis.py -d -j -e
# And get the reweighting spline
# > python ZZ4b/NtupleAna/scripts/analysis.py -w -j -e

### 3. Signal vs Background Classification
# Now run the data again and update the picoAOD so that the background model can be used for signal vs background classification training
# > python ZZ4b/NtupleAna/scripts/analysis.py -d -j -r -p tempPicoAOD.root -e 
# Update the hdf5 so that it has the new model event weights
# > python ZZ4b/NtupleAna/scripts/convert_root2h5.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.root"
# Run the signal
# > python ZZ4b/NtupleAna/scripts/analysis.py -s -e [-j -r -p tempPicoAOD.root (if you want to estimate the signal contamination in the background model)] 
# Convert the signal to hdf5
# > python ZZ4b/NtupleAna/scripts/convert_root2h5.py -i "/uscms/home/bryantp/nobackup/ZZ4b/*ZH4b2018/picoAOD.root"
# Train the classifier
# > py ZZ4b/NtupleAna/scripts/signalBackgroundClassifier.py -b "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5" -s "/uscms/home/bryantp/nobackup/ZZ4b/*ZH2018/picoAOD.h5"
# Update the hdf5 files with the classifier output
# > py ZZ4b/NtupleAna/scripts/signalBackgroundClassifier.py -b "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5" -s "/uscms/home/bryantp/nobackup/ZZ4b/*ZH2018/picoAOD.h5" -m <best model> -u
# Update the picoAODs with the classifier output for each event
# > python ZZ4b/NtupleAna/scripts/convert_h52root.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5"
# > python ZZ4b/NtupleAna/scripts/convert_h52root.py -i "/uscms/home/bryantp/nobackup/ZZ4b/*ZH4b2018/picoAOD.h5"
# Run the data and signal again
# > python ZZ4b/NtupleAna/scripts/analysis.py -d -j -r -e
# > python ZZ4b/NtupleAna/scripts/analysis.py -s -e [-j -r -p tempPicoAOD.root (if you want to estimate the signal contamination in the background model)]

### 4. Profit
# Make plots
# > python ZZ4b/NtupleAna/scripts/analysis.py --plot -j -e    (before reweighting)
# > python ZZ4b/NtupleAna/scripts/analysis.py --plot -j -r -e  (after reweighting)

#
# Config
#
script     = "ZZ4b/NtupleAna/scripts/nTupleAnalysis_cfg.py"
year       = "2018"
lumiDict   = {"2018": "59.6e3"}
lumi       = lumiDict[year]
outputBase = "/uscms/home/bryantp/nobackup/ZZ4b/"
gitRepoBase= 'ZZ4b/NtupleAna/weights/'

# File lists
periods = {"2016": "BCDEFGH",
           "2017": "",
           "2018": "ABCD"}
dataFiles = ["ZZ4b/fileLists/data"+year+period+".txt" for period in periods[year]]
# Jet Combinatoric Model
JCMRegion = "ZHSB"
JCMVersion = "00-00-02"
jetCombinatoricModel = gitRepoBase+"data"+year+"/jetCombinatoricModel_"+JCMRegion+"_"+JCMVersion+".txt"
reweight = gitRepoBase+"data"+year+"/reweight_"+JCMRegion+"_"+JCMVersion+".root"

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

def watch(command, stdout=None): # use to run a command and keep track of the thread, ie to run something when it is done
    print command
    if o.execute:
        p = subprocess.Popen(shlex.split(command), stdout=stdout, universal_newlines=(True if stdout else False), bufsize=1, close_fds=ON_POSIX)
        if stdout:
            q = Queue()
            t = Thread(target=enqueue_output, args=(p.stdout, q))
            t.daemon = True
            t.start()
            return (command, p, q)
        return (command, p)

def placeCursor(L,C):
    print '\033['+str(L)+';'+str(C)+'H',
def moveCursorUp(N=''):
    print '\r\033['+str(N)+'A',
def moveCursorDown(N=''):
    print '\r\033['+str(N)+'B',
# \033[<L>;<C>H # Move the cursor to line L, column C
# \033[<N>A     # Move the cursor up N lines
# \033[<N>B     # Move the cursor down N lines
# \033[<N>C     # Move the cursor forward N columns
# \033[<N>D     # Move the cursor backward N columns
# \033[2J       # Clear the screen, move to (0,0)
# \033[K        # Erase to end of line

def babySit(commands, maxAttempts=3):
    attempts={}
    nJobs = len(commands)
    jobs=[]
    outs=[]
    for command in commands:
        attempts[command] = 1
        jobs.append(watch(command, stdout=subprocess.PIPE))
        outs.append("LAUNCHING")

    if not o.execute: return
    
    done=False
    while not done:
        done=True
        time.sleep(0.1)
        failure=False
        for j in range(nJobs):
            code = jobs[j][1].poll()
            if code == None: 
                done=False
            elif code:
                moveCursorDown(1000)
                outs[j] = "CRASHED AT <"+outs[j]+">"
                crash = ["",
                         "# "+"-"*200,
                         "# "+jobs[j][0],
                         "# "+outs[j],
                         "# Exited with: "+str(code),
                         "# Attempt: "+str(attempts[jobs[j][0]]),
                         "# "+"-"*200,
                         ""]
                time.sleep(1)
                for line in crash: print line
                if attempts[jobs[j][0]] > maxAttempts: continue
                attempts[jobs[j][0]] += 1
                time.sleep(10)
                done=False
                jobs[j] = watch(jobs[j][0], stdout=subprocess.PIPE)

        nLines = 3
        print "\033[K"
        for j in range(nJobs):
            nLines+=4
            lines=textwrap.wrap("Attempt "+str(attempts[jobs[j][0]])+": "+jobs[j][0], 200)
            nLines += len(lines)
            print "\033[K# "+"-"*200
            for line in lines: print "\033[K#", line
            
            try:          outs[j]=jobs[j][2].get_nowait().replace('\n','').replace('\r','')
            except Empty: outs[j]=outs[j]
            print "\033[K# "
            print "\033[K#    >>",outs[j]
            print "\033[K# "
        print "\033[K# "+"-"*200
        print "\033[K"
        moveCursorUp(nLines)
    moveCursorDown(1000)
            

def waitForJobs(jobs,failedJobs):
    for job in jobs:
        code = job[1].wait()
        if code: failedJobs.append(job)
    return failedJobs

def relaunchJobs(jobs):
    print "# "+"-"*200
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

    cmds=[]
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    for signal in signalFiles:
        cmd  = "nTupleAnalysis "+script
        cmd += " -i "+signal
        cmd += " -o "+outputBase
        cmd += " -y "+year
        cmd += " -l "+lumi
        cmd += " --histogramming "+o.histogramming
        cmd += " --histFile "+histFile
        #cmd += " -j "+jetCombinatoricModel if o.useJetCombinatoricModel else ""
        #cmd += " -r "+reweight if o.reweight else ""
        cmd += " -p "+o.createPicoAOD if o.createPicoAOD else ""
        cmd += " --isMC"
        cmds.append(cmd)

    # wait for jobs to finish
    babySit(cmds)

    if o.createPicoAOD:
        if o.createPicoAOD != "picoAOD.root":
            for sample in ["ZH4b", "ggZH4b"]:
                cmd = "cp "+outputBase+sample+year+"/"+o.createPicoAOD+" "+outputBase+sample+year+"/picoAOD.root"
                execute(cmd)

    #cmd = "hadd -f "+outputBase+"bothZH4b"+year+"/picoAOD.root "+outputBase+"ZH4b"+year+"/picoAOD.root "+outputBase+"ggZH4b"+year+"/picoAOD.root"
    #execute(cmd)
    cmd = "hadd -f "+outputBase+"bothZH4b"+year+"/"+histFile+" "+outputBase+"ZH4b"+year+"/"+histFile+" "+outputBase+"ggZH4b"+year+"/"+histFile+" > hadd.log"
    execute(cmd)

      
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

    cmds=[]
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    for data in dataFiles:
        cmd  = "nTupleAnalysis "+script
        cmd += " -i "+data
        cmd += " -o "+outputBase
        cmd += " -y "+year
        cmd += " --histogramming "+o.histogramming
        cmd += " --histFile "+histFile
        cmd += " -j "+jetCombinatoricModel if o.useJetCombinatoricModel else ""
        cmd += " -r "+reweight if o.reweight else ""
        cmd += " -p "+o.createPicoAOD if o.createPicoAOD else ""
        #jobs.append(watch(cmd))
        cmds.append(cmd)

    # wait for jobs to finish
    babySit(cmds)

    if o.createPicoAOD:
        if o.createPicoAOD != "picoAOD.root":
            for period in periods[year]:
                cmd = "cp "+outputBase+"data"+year+period+"/"+o.createPicoAOD+" "+outputBase+"data"+year+period+"/picoAOD.root"
                execute(cmd)

    mkdir(outputBase+"data"+year)
    #cmd = "hadd -f "+outputBase+"data"+year+"/picoAOD.root "+" ".join([outputBase+"data"+year+period+"/picoAOD.root" for period in periods[year]])
    #execute(cmd)
    cmd = "hadd -f "+outputBase+"data"+year+"/"+histFile+" "+" ".join([outputBase+"data"+year+period+"/"+histFile for period in periods[year]])+" > hadd.log"
    execute(cmd)
    


def doWeights():
    mkdir(gitRepoBase+"data"+year)
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    cmd  = "python ZZ4b/NtupleAna/scripts/makeWeights.py -d "+outputBase+"data"+year+"/"+histFile+" -o "+gitRepoBase+"data"+year+"/ -r "+JCMRegion+" -w "+JCMVersion
    execute(cmd)


def doPlots():
    plots = "plots"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")
    output = outputBase+plots
    cmd  = "python ZZ4b/NtupleAna/scripts/makePlots.py -o "+outputBase+" -p "+plots+" -l "+lumi
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
    var = "ZHvB"
    #var = "xZH"
    cmd = "python ZZ4b/NtupleAna/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/bothZH4b2018/hists.root -o "+outFile+" -r "+region+" --var "+var+" -n ZH --tag four --cut "+cut
    execute(cmd)
    cmd = "python ZZ4b/NtupleAna/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018/hists_j_r.root -o "+outFile+" -r "+region+" --var "+var+" -n multijet --tag three --cut "+cut
    execute(cmd)
    cmd = "python ZZ4b/NtupleAna/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018/hists_j_r.root -o "+outFile+" -r "+region+" --var "+var+" -n data_obs --tag four --cut "+cut
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
    #nTupleAnalysis ZZ4b/NtupleAna/scripts/nTupleAnalysis_cfg.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.root -o /uscms/home/bryantp/nobackup/ZZ4b/ -y 2018 --histogramming 0 --histFile hists.root -j /uscms/home/bryantp/nobackup/ZZ4b/data2018A/jetCombinatoricModel_ZHSB_00-00-02_iter0.txt -p picoAOD0.root
    #convert to .h5 for nTagClassifier training
    #py ZZ4b/NtupleAna/scripts/convert_root2h5.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.root -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.h5
    #update .h5 file after training
    #update .root file with nTagClassifier output
    #python ZZ4b/NtupleAna/scripts/convert_h52root.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.h5 -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.root
    #create hists for nTagClassifier reweight
    #nTupleAnalysis ZZ4b/NtupleAna/scripts/nTupleAnalysis_cfg.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD0.root -o /uscms/home/bryantp/nobackup/ZZ4b/ -y 2018 --histogramming 3 --histFile hists0.root -j /uscms/home/bryantp/nobackup/ZZ4b/data2018A/jetCombinatoricModel_ZHSB_00-00-02_iter0.txt
    #make reweighting spline
    #python ZZ4b/NtupleAna/scripts/makeWeights.py -d /uscms/home/bryantp/nobackup/ZZ4b/data2018A/hists0.root -o /uscms/home/bryantp/nobackup/ZZ4b/data2018A/ -r ZHSB -w 00-00-02 -i 0
    doData()

if o.doWeights:
    doWeights()

if o.doPlots:
    doPlots()

if o.doCombine:
    doCombine()
