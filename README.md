# nTupleAnalysis

Process NANOAOD files to create skims (picoAODs) and event loop histograms. 

From a CMSSW/src release area

>cmsenv

Checkout the nTupleAnalysis base class repo

>git clone git@github.com:patrickbryant/nTupleAnalysis.git
>git clone git@github.com:johnalison/nTupleHelperTools.git
>git clone git@github.com:johnalison/TriggerEmulator.git

           
>scram b ZZ4b/nTupleAnalysis

>voms-proxy-init -voms cms

Run the following with --help to see what command line argument you can specify. Otherwise it runs with the default input files and default output directory. 

>nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py

# Luminosity Data

When running on data a live count of the integrated luminosity that has been processed can be displayed. The luminosity per lumiblock in a given lumiMask json file is calculated with brilcalc. 

First download the lumiMask:

>wget https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions18/13TeV/PromptReco/Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt

>mv Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt ZZ4b/lumiMasks/

Then setup the brilconda environment:

>export PATH=$HOME/.local/bin:/cvmfs/cms-bril.cern.ch/brilconda/bin:$PATH

And calculate the lumi contained in the json file for a given trigger. You must specify the trigger version, this can be done by including "_v*" at the end of the trigger name. 
The recommended normtag can be found at https://twiki.cern.ch/twiki/bin/viewauth/CMS/TWikiLUM.

>brilcalc lumi -c web --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json -u /pb -i lumiMasks/Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt --hltpath "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5_v*" --byls -o lumiMasks/brilcalc_2018_HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5.csv 

The output csv file can be quite large. For 2018 it is 30MB. This is because there is one line for every lumiblock and there are O(1M) lumiblocks in a dataset. The option --byls tells brilcalc to give the lumi split by lumiblock; the default is to only specify the lumi per run. 

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


Hemisphere Analysis

To create the hemisphere library

>nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt  -o $PWD -y 2018  --histogramming 1 --histFile hists.root  --nevents -1  --createHemisphereLibrary

To analyze the hemisphere data (may take significant amount of time)

> time hemisphereAnalysis ZZ4b/nTupleAnalysis/scripts/hemisphereAnalysis_cfg.py -i '$PWD/data18/hemiSphereLib_3TagEvents_*root' -o $PWD --histFile hists_3tag.root  -n -1

To load the library and create mixed eventsx

nTupleAnalysis ZZ4b/nTupleAnalysis/scripts/nTupleAnalysis_cfg.py -i ZZ4b/fileLists/data18.txt  -o $PWD -y 2018  --histogramming 1 --histFile hists.root  --nevents -1 --loadHemisphereLibrary --inputHLib3Tag '$PWD/data18/hemiSphereLib_3TagEvents_*root' --inputHLib4Tag '$PWD/data18/hemiSphereLib_4TagEvents_*root'