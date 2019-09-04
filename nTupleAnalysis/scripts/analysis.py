import time
import textwrap
import os, re
import sys
import subprocess
import shlex
import optparse
from threading import Thread
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-s',            action="store_true", dest="doSignal",       default=False, help="Run signal MC")
parser.add_option('-t',            action="store_true", dest="doTT",       default=False, help="Run ttbar MC")
parser.add_option('-a',            action="store_true", dest="doAccxEff",      default=False, help="Make Acceptance X Efficiency plots")
parser.add_option('-d',            action="store_true", dest="doData",         default=False, help="Run data")
parser.add_option('-q',            action="store_true", dest="doQCD",          default=False, help="Subtract ttbar MC from data to make QCD template")
parser.add_option('-y',                                 dest="year",      default="2018", help="Year")
parser.add_option('-w',            action="store_true", dest="doWeights",      default=False, help="Fit jetCombinatoricModel and nJetClassifier TSpline")
parser.add_option('-j',            action="store_true", dest="useJetCombinatoricModel",       default=False, help="Use the jet combinatoric model")
parser.add_option('-r',            action="store_true", dest="reweight",       default=False, help="Do reweighting with nJetClassifier TSpline")
parser.add_option('--plot',        action="store_true", dest="doPlots",        default=False, help="Make Plots")
parser.add_option('-p', '--createPicoAOD',              dest="createPicoAOD",  type="string", help="Create picoAOD with given name")
parser.add_option('-f', '--fastSkim',                   dest="fastSkim",       action="store_true", default=False, help="Do fast picoAOD skim")
parser.add_option('-n', '--nevents',                    dest="nevents",        default="-1", help="Number of events to process. Default -1 for no limit.")
parser.add_option(      '--histogramming',              dest="histogramming",  default="1", help="Histogramming level. 0 to make no kinematic histograms. 1: only make histograms for full event selection, larger numbers add hists in reverse cutflow order.")
parser.add_option('-c',            action="store_true", dest="doCombine",      default=False, help="Make CombineTool input hists")
o, a = parser.parse_args()

#
# Analysis in four "easy" steps
#

### 1. Jet Combinatoric Model
# First run on data
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -d -e
# Then make jet combinatoric model 
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -w -e
# Now run again and update the automatically generated picoAOD by making a temporary one which will then be copied over picoAOD.root
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -d -j -p tempPicoAOD.root -e

### 2. ThreeTag to FourTag reweighting
# Now convert the picoAOD to hdf5 to train the Four Vs Three tag classifier (FvT)
# > python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.root"
# Now train the classifier
# > python ZZ4b/nTupleAnalysis/scripts/signalBackgroundClassifier.py -b "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5"
# Take the best result and update the hdf5 files with classifier output for each event
# > py ZZ4b/nTupleAnalysis/scripts/signalBackgroundClassifier.py -b "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5" -m <the best model> -u
# Update the picoAOD.root with the result
# > python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5"
# Now run the data again so that you can compute the FvT reweighting 
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -d -j -e
# And get the reweighting spline
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -w -j -e

### 3. Signal vs Background Classification
# Now run the data again and update the picoAOD so that the background model can be used for signal vs background classification training
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -d -j -r -p tempPicoAOD.root -e 
# Update the hdf5 so that it has the new model event weights
# > python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.root"
# Run the signal
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -s -e [-j -r -p tempPicoAOD.root (if you want to estimate the signal contamination in the background model)] 
# Convert the signal to hdf5
# > python ZZ4b/nTupleAnalysis/scripts/convert_root2h5.py -i "/uscms/home/bryantp/nobackup/ZZ4b/*ZH4b2018/picoAOD.root"
# Train the classifier
# > py ZZ4b/nTupleAnalysis/scripts/signalBackgroundClassifier.py -b "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5" -s "/uscms/home/bryantp/nobackup/ZZ4b/*ZH2018/picoAOD.h5"
# Update the hdf5 files with the classifier output
# > py ZZ4b/nTupleAnalysis/scripts/signalBackgroundClassifier.py -b "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5" -s "/uscms/home/bryantp/nobackup/ZZ4b/*ZH2018/picoAOD.h5" -m <best model> -u
# Update the picoAODs with the classifier output for each event
# > python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py -i "/uscms/home/bryantp/nobackup/ZZ4b/data2018*/picoAOD.h5"
# > python ZZ4b/nTupleAnalysis/scripts/convert_h52root.py -i "/uscms/home/bryantp/nobackup/ZZ4b/*ZH4b2018/picoAOD.h5"
# Run the data and signal again
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -d -j -r -e
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py -s -e [-j -r -p tempPicoAOD.root (if you want to estimate the signal contamination in the background model)]

### 4. Profit
# Make plots
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py --plot -j -e    (before reweighting)
# > python ZZ4b/nTupleAnalysis/scripts/analysis.py --plot -j -r -e  (after reweighting)

#
# Config
#
script     = "ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py"
year       = o.year
lumiDict   = {"2017": "1e3",
              "2018": "59.6e3"}
lumi       = lumiDict[year]
outputBase = "/uscms/home/bryantp/nobackup/ZZ4b/"
gitRepoBase= 'ZZ4b/nTupleAnalysis/weights/'

# File lists
periods = {"2016": "BCDEFGH",
           "2017": "BCDEF",
           "2018": "ABCD"}
dataFiles = ["ZZ4b/fileLists/data"+year+period+".txt" for period in periods[year]]
# Jet Combinatoric Model
JCMRegion = "SB"
JCMVersion = "00-00-02"
jetCombinatoricModel = gitRepoBase+"data"+year+"/jetCombinatoricModel_"+JCMRegion+"_"+JCMVersion+".txt"
#reweight = gitRepoBase+"data"+year+"/reweight_"+JCMRegion+"_"+JCMVersion+".root"

signalFiles = ["ZZ4b/fileLists/ggZH4b"+year+".txt",
               "ZZ4b/fileLists/ZH4b"+year+".txt",
               "ZZ4b/fileLists/ZZ4b"+year+".txt",
               ]

ttbarFiles = ["ZZ4b/fileLists/TTToHadronic"+year+".txt",
              "ZZ4b/fileLists/TTToSemiLeptonic"+year+".txt",
              "ZZ4b/fileLists/TTTo2L2Nu"+year+".txt",
              #"ZZ4b/fileLists/TTJets"+year+".txt",
              ]

accxEffFiles = [outputBase+"ZZ4b"+year+"/histsFromNanoAOD.root",
                outputBase+"ZH4b"+year+"/histsFromNanoAOD.root",
                outputBase+"ggZH4b"+year+"/histsFromNanoAOD.root",
                outputBase+"bothZH4b"+year+"/histsFromNanoAOD.root",
                ]


def doSignal():
    mkdir(outputBase, o.execute)

    cmds=[]
    histFile = "hists.root" #+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    if o.createPicoAOD == "picoAOD.root": histFile = "histsFromNanoAOD.root"
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
        cmd += " -f " if o.fastSkim else ""
        cmd += " --isMC"
        cmds.append(cmd)

    # wait for jobs to finish
    if len(cmds)>1:
        babySit(cmds, o.execute)
    else:
        execute(cmd, o.execute)

    if o.createPicoAOD:
        if o.createPicoAOD != "picoAOD.root":
            for sample in ["ZH4b", "ggZH4b", "ZZ4b"]:
                cmd = "cp "+outputBase+sample+year+"/"+o.createPicoAOD+" "+outputBase+sample+year+"/picoAOD.root"
                execute(cmd, o.execute)

    if "ZZ4b/fileLists/ZH4b"+year+".txt" in signalFiles and "ZZ4b/fileLists/ggZH4b"+year+".txt" in signalFiles:
        mkdir(outputBase+"bothZH4b"+year, o.execute)
        cmd = "hadd -f "+outputBase+"bothZH4b"+year+"/"+histFile+" "+outputBase+"ZH4b"+year+"/"+histFile+" "+outputBase+"ggZH4b"+year+"/"+histFile+" > hadd.log"
        execute(cmd, o.execute)

    if "ZZ4b/fileLists/ZH4b"+year+".txt" in signalFiles and "ZZ4b/fileLists/ggZH4b"+year+".txt" in signalFiles and "ZZ4b/fileLists/ZZ4b"+year+".txt" in signalFiles:
        mkdir(outputBase+"ZZandZH4b"+year, o.execute)
        cmd = "hadd -f "+outputBase+"ZZandZH4b"+year+"/"+histFile+" "+outputBase+"ZH4b"+year+"/"+histFile+" "+outputBase+"ggZH4b"+year+"/"+histFile+" "+outputBase+"ZZ4b"+year+"/"+histFile+" > hadd.log"
        execute(cmd, o.execute)

def doTT():
    mkdir(outputBase, o.execute)

    cmds=[]
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    if o.createPicoAOD == "picoAOD.root": histFile = "histsFromNanoAOD.root"
    for ttbar in ttbarFiles:
        cmd  = "nTupleAnalysis "+script
        cmd += " -i "+ttbar
        cmd += " -o "+outputBase
        cmd += " -y "+year
        cmd += " -l "+lumi
        cmd += " --histogramming "+o.histogramming
        cmd += " --histFile "+histFile
        cmd += " -j "+jetCombinatoricModel if o.useJetCombinatoricModel else ""
        cmd += " -r " if o.reweight else ""
        cmd += " -p "+o.createPicoAOD if o.createPicoAOD else ""
        cmd += " -f " if o.fastSkim else ""
        cmd += " --isMC"
        cmds.append(cmd)

    # wait for jobs to finish
    if len(cmds)>1:
        babySit(cmds, o.execute)
    else:
        execute(cmd, o.execute)

    if o.createPicoAOD:
        if o.createPicoAOD != "picoAOD.root":
            for sample in ["TTToHadronic","TTToSemiLeptonic"]:
                cmd = "cp "+outputBase+sample+year+"/"+o.createPicoAOD+" "+outputBase+sample+year+"/picoAOD.root"
                execute(cmd, o.execute)

    if "ZZ4b/fileLists/TTToHadronic"+year+".txt" in ttbarFiles and "ZZ4b/fileLists/TTToSemiLeptonic"+year+".txt" in ttbarFiles:
        mkdir(outputBase+"TT"+year, o.execute)
        cmd = "hadd -f "+outputBase+"TT"+year+"/"+histFile+" "+outputBase+"TTToHadronic"+year+"/"+histFile+" "+outputBase+"TTToSemiLeptonic"+year+"/"+histFile+" > hadd.log"
        execute(cmd, o.execute)

      
def doAccxEff():   
    cmds = []
    for signal in accxEffFiles:
        cmd = "python ZZ4b/nTupleAnalysis/scripts/makeAccxEff.py -i "+signal
        cmds.append(cmd)
    babySit(cmds, o.execute)

def doDataTT():
    mkdir(outputBase, o.execute)

    # run event loop
    cmds=[]
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    if o.createPicoAOD == "picoAOD.root": histFile = "histsFromNanoAOD.root"
    files = []
    if o.doData: files += dataFiles
    if o.doTT:   files += ttbarFiles
    for f in files:
        cmd  = "nTupleAnalysis "+script
        cmd += " -i "+f
        cmd += " -o "+outputBase
        cmd += " -y "+year
        cmd += " --histogramming "+o.histogramming
        cmd += " --histFile "+histFile
        cmd += " -j "+jetCombinatoricModel if o.useJetCombinatoricModel else ""
        cmd += " -r " if o.reweight else ""
        cmd += " -p "+o.createPicoAOD if o.createPicoAOD else ""
        cmd += " -f " if o.fastSkim else ""
        if f in ttbarFiles:
            cmd += " -l "+lumi
            cmd += " --isMC "
        cmds.append(cmd)

    # wait for jobs to finish
    if len(cmds)>1:
        babySit(cmds, o.execute)
    else:
        execute(cmd, o.execute)

    # overwrite nominal picoAOD with newly created one
    if o.createPicoAOD and o.createPicoAOD != "picoAOD.root":
        cmds = []
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
    if o.doData:
        mkdir(outputBase+"data"+year, o.execute)
        cmd = "hadd -f "+outputBase+"data"+year+"/"+histFile+" "+" ".join([outputBase+"data"+year+period+"/"+histFile for period in periods[year]])#+" > hadd.log"
        cmds.append(cmd)
    
    if o.doTT:
        if "ZZ4b/fileLists/TTToHadronic"+year+".txt" in ttbarFiles and "ZZ4b/fileLists/TTToSemiLeptonic"+year+".txt" in ttbarFiles and "ZZ4b/fileLists/TTTo2L2Nu"+year+".txt" in ttbarFiles:
            mkdir(outputBase+"TT"+year, o.execute)
            cmd = "hadd -f "+outputBase+"TT"+year+"/"+histFile+" "+outputBase+"TTToHadronic"+year+"/"+histFile+" "+outputBase+"TTToSemiLeptonic"+year+"/"+histFile+" "+outputBase+"TTTo2L2Nu"+year+"/"+histFile#+" > hadd.log"
            cmds.append(cmd)
    babySit(cmds, o.execute)


def subtractTT():
    mkdir(outputBase+"qcd"+year, o.execute)
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    if o.createPicoAOD == "picoAOD.root": histFile = "histsFromNanoAOD.root"
    cmd  = "python ZZ4b/nTupleAnalysis/scripts/subtractTT.py"
    cmd += " -d   "+ outputBase+"data"+year+"/"+histFile
    cmd += " --tt "+ outputBase+  "TT"+year+"/"+histFile
    cmd += " -q   "+ outputBase+ "qcd"+year+"/"+histFile
    babySit([cmd], o.execute)


def doWeights():
    mkdir(gitRepoBase+"data"+year, o.execute)
    histFile = "hists"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")+".root"
    cmd  = "python ZZ4b/nTupleAnalysis/scripts/makeWeights.py"
    cmd += " -d   "+ outputBase+"data"+year+"/"+histFile
    cmd += " --tt "+ outputBase+  "TT"+year+"/"+histFile
    cmd += " -o "+gitRepoBase+"data"+year+"/ " 
    cmd += " -r "+JCMRegion
    cmd += " -w "+JCMVersion
    execute(cmd, o.execute)


def doPlots(extraPlotArgs=""):
    plots = "plots"+("_j" if o.useJetCombinatoricModel else "")+("_r" if o.reweight else "")
    output = outputBase+plots
    cmd  = "python ZZ4b/nTupleAnalysis/scripts/makePlots.py -o "+outputBase+" -p "+plots+" -l "+lumi
    cmd += " -j" if o.useJetCombinatoricModel else ""
    cmd += " -r" if o.reweight else ""
    cmd += " "+extraPlotArgs+" "
    execute(cmd, o.execute)
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
    outFile = "combineZH.root"
    execute("rm "+outFile, o.execute)

    region = "ZH"
    #cut = "passDEtaBB"
    cut = "passMDRs"
    #var = "mZH"
    var = "ZHvB"
    cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/bothZH4b2018/hists.root -o "+outFile+" -r "+region+" --var "+var+" -n ZH --tag four --cut "+cut
    execute(cmd, o.execute)
    cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018/hists_j_r.root -o "+outFile+" -r "+region+" --var "+var+" -n multijet --tag three --cut "+cut
    execute(cmd, o.execute)
    cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/data2018/hists_j_r.root -o "+outFile+" -r "+region+" --var "+var+" -n data_obs --tag four --cut "+cut
    execute(cmd, o.execute)

    outFile = "combineZZ.root"
    execute("rm "+outFile, o.execute)

    region = "ZZ"
    var = "xZZ"
    cmd = "python ZZ4b/nTupleAnalysis/scripts/makeCombineHists.py -i /uscms/home/bryantp/nobackup/ZZ4b/ZZ4b2018/hists.root -o "+outFile+" -r "+region+" --var "+var+" -n ZZ --tag four --cut "+cut
    execute(cmd, o.execute)
    # cd /uscms/homes/b/bryantp/work/CMSSW_8_1_0/src
    # combine -M Significance ../../CMSSW_10_2_0/src/combineTest.txt -t -1 --expectSignal=1

#
# Run analysis
#
if o.doSignal:
    doSignal()

# if o.doTT:
#     doTT()

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
