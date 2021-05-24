import ROOT 
import sys

print "Dumping from ",sys.argv[1]
inFile = ROOT.TFile.Open(sys.argv[1])
inFile.ls()

eventTree = inFile.Get("Events")
if len(sys.argv) > 2:
    eventTree.Print(sys.argv[2])
else:
    eventTree.Print()
