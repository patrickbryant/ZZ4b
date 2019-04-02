import time
import os, re
import sys
import subprocess
import shlex
import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-s',            action="store_true", dest="doSignal",       default=False, help="Run signal MC")
parser.add_option('-d',            action="store_true", dest="doData",         default=False, help="Run data")
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
dataFiles = ["ZZ4b/fileLists/data"+year+"A.txt",
             ]
signalFiles = ["ZZ4b/fileLists/ggZH4b"+year+".txt",
               "ZZ4b/fileLists/ZH4b"+year+".txt",]
    

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
        cmd += " --histogramming 1"
        cmd += " --isMC"
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
        cmd += " --histogramming 1"
        jobs.append(watch(cmd))


#
# Run analysis
#
if o.doSignal:
    doSignal()

if o.doData:
    doData()
