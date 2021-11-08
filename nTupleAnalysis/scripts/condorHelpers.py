import os
from commandLineHelpers import *

def getCMSSW():
    return os.getenv('CMSSW_VERSION')

def getUSER():
    return os.getenv('USER')

def rmTARBALL(doRun):
    
    base="/uscms/home/"+getUSER()+"/nobackup/HH4b/"
    if os.path.exists(base+getCMSSW()+".tgz"):
        print "Removing tarball"
        cmd = "rm "+base+getCMSSW()+".tgz"
        execute(cmd,doRun)

def make_x509File(doRun):
    execute("voms-proxy-init -rfc -voms cms -valid 192:00 --out x509up_forCondor",doRun)


def rmTARBALL(doRun):
    base="/uscms/home/"+getUSER()+"/nobackup/HH4b/"
    localTarball = base+getCMSSW()+".tgz"
    cmd = "rm "+localTarball
    execute(cmd, doRun)    


def makeTARBALL(doRun, debug=False):
    base="/uscms/home/"+getUSER()+"/nobackup/HH4b/"
    TARBALL   = "root://cmseos.fnal.gov//store/user/"+getUSER()+"/condor/"+getCMSSW()+".tgz"

    if os.path.exists(base+getCMSSW()+".tgz"):
        print "TARBALL already exists, skip making it"
        return
    cmd  = 'tar -C '+base+' -zcf '+base+getCMSSW()+'.tgz '+getCMSSW()
    if debug:
        cmd  = 'tar -C '+base+' -zcvf '+base+getCMSSW()+'.tgz '+getCMSSW()
    cmd += ' --exclude="*.pdf" --exclude="*.jdl" --exclude="*.stdout" --exclude="*.stderr" --exclude="*.log"  --exclude="log_*" --exclude="*.stdout" --exclude="*.stderr"'
    cmd += ' --exclude=".git" --exclude="PlotTools" --exclude="madgraph" --exclude="*.pkl"   --exclude="*.h5"   --exclude=data*hemis*.tgz  --exclude=plotsWith*  --exclude=plotsNoFvT*  '
    cmd += ' --exclude="hemiSphereLib*.root" '
    cmd += ' --exclude="hists*.root" '
    cmd += ' --exclude="pico*.root" '
    cmd += " --exclude=Signal*hemis*.tgz "
    cmd += ' --exclude="closureTests/OLD" '
    #cmd += ' --exclude="closureTests/nominal" '
    cmd += ' --exclude=plotsRW* '
    cmd += ' --exclude="tmp" --exclude="combine" --exclude="genproductions" --exclude-vcs --exclude-caches-all'
    execute(cmd, doRun)
    cmd  = 'ls '+base+' -alh'
    execute(cmd, doRun)
    cmd = "xrdfs root://cmseos.fnal.gov/ mkdir /store/user/"+getUSER()+"/condor"
    execute(cmd, doRun)
    cmd = "xrdcp -f "+base+getCMSSW()+".tgz "+TARBALL
    execute(cmd, doRun)



def makeDAGFile(dag_file, dag_config, outputDir):
    fileName = outputDir+"/"+dag_file
    f=open(fileName,'w')

    dependencies = []

    if len(dag_config) > 25: 
        print "ERROR Too many subjobs ", len(dag_config)
        sys.exit(-1)
    
    from string import ascii_uppercase

    # Name the JOB and collect the JOB names
    for node_itr, node_list in enumerate(dag_config):
        dependencies.append([])
        for job_itr, job in enumerate(node_list):
            jobID = ascii_uppercase[node_itr]+str(job_itr)
            line = "JOB "+jobID+" "+job
            f.write(line+"\n")
            dependencies[-1].append(jobID)

    # derive the structure in terms of JOB names (assume each JOB at higher level depends on all jobs below it)
    for dep_itr in range(1,len(dependencies)):
        parents = dependencies[dep_itr-1]
        
        for child in dependencies[dep_itr]:
            line = "PARENT "
            for p in parents: line += p+" "
            line += " CHILD " + child
            f.write(line+"\n")

    f.close()
    return fileName

def makeCondorFile(cmd,eosOutDir,eosSubdir, outputDir, filePrefix):
    jdlFileName = filePrefix+eosSubdir
    TARBALL = "root://cmseos.fnal.gov//store/user/"+getUSER()+"/condor/"+getCMSSW()+".tgz"
    EOSOUTDIR = "None" if eosOutDir == "None" else eosOutDir+eosSubdir
    thisJDL = jdl(CMSSW=getCMSSW(), EOSOUTDIR=EOSOUTDIR, TARBALL=TARBALL, cmd=cmd, fileName=outputDir+jdlFileName+".jdl", logPath=outputDir, logName=jdlFileName)
    thisJDL.make()
    return thisJDL.fileName



class jdlHemiMixing:
    def __init__(self, CMSSW=None, EOSOUTDIR=None, TARBALL=None, HEMINAME=None, HEMITARBALL=None, cmd=None, fileName=None, logPath = "./", logName = "condor_$(Cluster)_$(Process)"):
        if fileName: 
            self.fileName = fileName+".jdl"
        else:
            self.fileName = str(np.random.uniform())[2:]+".jdl"

        self.CMSSW = CMSSW
        self.EOSOUTDIR = EOSOUTDIR
        self.TARBALL = TARBALL
        self.HEMINAME = HEMINAME
        self.HEMITARBALL = HEMITARBALL

        self.universe = "vanilla"
        self.use_x509userproxy = "true"
        self.Executable = "ZZ4b/nTupleAnalysis/scripts/condorHemiMixing.sh"
        #self.x509userproxy = "x509up_forCondor"
        self.should_transfer_files = "YES"
        self.when_to_transfer_output = "ON_EXIT"
        self.Output = logPath+logName+".stdout"
        self.Error = logPath+logName+".stderr"
        self.Log = logPath+logName+".log"
        self.Arguments = CMSSW+" "+EOSOUTDIR+" "+TARBALL+" "+HEMINAME+" "+HEMITARBALL+" "+cmd
        self.Queue = "1" # no equals sign in .jdl file

    def make(self):
        attributes=["universe",
                    "use_x509userproxy",
                    "Executable",
                    #"x509userproxy",
                    "should_transfer_files",
                    "when_to_transfer_output",
                    "Output",
                    "Error",
                    "Log",
                    "Arguments",
                ]
        f=open(self.fileName,'w')
        for attr in attributes:
            f.write(attr+" = "+str(getattr(self, attr))+"\n")

        f.write('+DesiredOS="SL7"\n')
        f.write("Queue "+str(self.Queue)+"\n")    
        f.close()



def makeCondorFileHemiMixing(cmd,eosOutDir,eosSubdir, outputDir, filePrefix, HEMINAME, HEMITARBALL):
    jdlFileName = filePrefix+eosSubdir
    TARBALL = "root://cmseos.fnal.gov//store/user/"+getUSER()+"/condor/"+getCMSSW()+".tgz"
    EOSOUTDIR = "None" if eosOutDir == "None" else eosOutDir+eosSubdir

    thisJDL = jdlHemiMixing(CMSSW=getCMSSW(), EOSOUTDIR=EOSOUTDIR, TARBALL=TARBALL, HEMINAME=HEMINAME, HEMITARBALL=HEMITARBALL, cmd=cmd, fileName=outputDir+jdlFileName, logPath=outputDir, logName=jdlFileName)
    thisJDL.make()
    return thisJDL.fileName
    
