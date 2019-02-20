# NtupleAna

Process NANOAOD files to create skims (picoAODs) and event loop histograms. 

From a CMSSW/src release area

>cmsenv

>scram b ZZ4b

>voms-proxy-init -voms cms

Run the following with --help to see what command line argument you can specify. Otherwise it runs with the default input files and default output directory. 

>procNtupleTest ZZ4b/NtupleAna/scripts/procNtupleTest_cfg.py


# Madgraph Studies

Generate ZZ->4b events in madgraph

>cd madgraph

>mg

>launch mg5_ZZ4b

Convert .lhe to .root (in command line from ZZ4b directory)

>gzip -d madgraph/mg5_ZZ4b/Events/run_01/unweighted_events.lhe.gz 

>ExRootLHEFConverter madgraph/mg5_ZZ4b/Events/run_01/unweighted_events.lhe madgraph/mg5_ZZ4b/Events/run_01/unweighted_events.root

Process events

>py3 python/run.py -i madgraph/mg5_ZZ4b/Events/run_01/unweighted_events.root -o hists/mg5_ZZ4b_run_01.root
