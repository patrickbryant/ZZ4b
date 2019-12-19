import ROOT
from array import array

import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--inputDataFileName')
parser.add_option('-o', '--outputDataFileName')
o, a = parser.parse_args()


inFile = ROOT.TFile(o.inputDataFileName,"READ")
inFile.ls()

EventsTreeOLD  = inFile.Get("Events")
RunTreeOLD     = inFile.Get("Runs")
LBsTreeOLD     = inFile.Get("LuminosityBlocks")


outFile = ROOT.TFile(o.outputDataFileName,"RECREATE")
outFile.cd()
EventsTree = EventsTreeOLD.CloneTree()
RunTree    = RunTreeOLD.CloneTree()
LBsTree    = LBsTreeOLD.CloneTree()

genWeight = array( 'f', [ 0.0 ] )


b_genWeight = EventsTree.Branch("genWeight",genWeight,"genWeight/F")

nentries = EventsTree.GetEntries()
print "Have nentries",nentries
for i in xrange(nentries):
    EventsTree.GetEntry(i)
    genWeight[0] = -1e6
    b_genWeight.Fill()


RunTree.Write()
LBsTree.Write()
EventsTree.Write()
