# ZZ4b

Generate ZZ->4b events in madgraph

>cd madgraph

>mg

>launch mg5_ZZ4b

Convert .lhe to .root (in command line from ZZ4b directory)

>gzip -d madgraph/mg5_ZZ4b/Events/run_01/unweighted_events.lhe.gz 

>ExRootLHEFConverter madgraph/mg5_ZZ4b/Events/run_01/unweighted_events.lhe madgraph/mg5_ZZ4b/Events/run_01/unweighted_events.root

Process events

>py3 python/analysis.py -i madgraph/mg5_ZZ4b/Events/run_01/unweighted_events.root -o hists/mg5_ZZ4b_run_01.root