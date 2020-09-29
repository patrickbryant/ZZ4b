#!/bin/bash
echo "Starting job on " `date` #Date/time of start of job
echo "Running on: `uname -a`" #Condor job is running on this node
echo "System software: `cat /etc/redhat-release`" #Operating System on that node
CMSSW=$1
EOSOUTDIR=$2
TARBALL=$3
HEMINAME=$4
HEMITARBALL=$5
CMD=(${@})
echo "Command was"
echo ${CMD[*]}
unset CMD[0]
unset CMD[1]
unset CMD[2]
unset CMD[3]
unset CMD[4]
echo "Command now"
echo ${CMD[*]}


## bring in the tarball you created before with caches and large files excluded:
echo "xrdcp -s ${TARBALL} ."
xrdcp -s ${TARBALL} .
echo "source /cvmfs/cms.cern.ch/cmsset_default.sh"
source /cvmfs/cms.cern.ch/cmsset_default.sh 
echo "tar -xf ${CMSSW}.tgz"
tar -xf ${CMSSW}.tgz
rm ${CMSSW}.tgz
echo "cd ${CMSSW}/src/"
cd ${CMSSW}/src/

# bring in the hemispheres 
echo "xrdcp -s ${HEMITARBALL} ."
xrdcp -s ${HEMITARBALL} .
echo "tar -xf ${HEMINAME}.tgz"
tar -xf ${HEMINAME}.tgz
rm ${HEMINAME}.tgz

scramv1 b ProjectRename # this handles linking the already compiled code - do NOT recompile
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers
echo $CMSSW_BASE "is the CMSSW we have on the local worker node"
#cd ${_CONDOR_SCRATCH_DIR}
#export X509_USER_PROXY=./x509up_forCondor
#echo "X509 user proxy"
#echo $X509_USER_PROXY 
pwd
echo "${CMD[*]}"
eval "${CMD[*]}"

if [ $EOSOUTDIR == "None" ]
then
    echo "Done"
else
    pwd
    echo "List all root files:"
    ls *.root
    echo "List all files:"
    ls -alh
    echo "*******************************************"
    echo "xrdcp output for condor to:"
    echo $EOSOUTDIR
    for FILE in *.root
    do
      echo "xrdcp -f ${FILE} ${EOSOUTDIR}/${FILE}"
      xrdcp -f ${FILE} ${EOSOUTDIR}/${FILE} 2>&1
      XRDEXIT=$?
      if [[ $XRDEXIT -ne 0 ]]; then
        rm *.root
        echo "exit code $XRDEXIT, failure in xrdcp"
        exit $XRDEXIT
      fi
      rm ${FILE}
    done
fi #EOSOUTDIR?=None
