import ROOT
from array import array

import optparse
parser = optparse.OptionParser()
parser.add_option('-1', '--inputFileName1')
parser.add_option('-2', '--inputFileName2')
o, a = parser.parse_args()



def compBranches(file1, file2, TreeName):

    tree1  = file1.Get(TreeName)
    branches1 = tree1.GetListOfBranches()
    branchNames1 = []
    for b in branches1:
        branchNames1.append(b.GetName())


    tree2  = file2.Get(TreeName)
    branches2 = tree2.GetListOfBranches()
    branchNames2 = []
    for b in branches2:
        branchNames2.append(b.GetName())


    print TreeName, "in",file1, "has",len(branchNames1),"branches"
    print TreeName, "in",file2, "has",len(branchNames2),"branches"
    
    b1Names_notInFile2 = []
    for b1Name in branchNames1:
        if b1Name not in branchNames2:
            b1Names_notInFile2.append(b1Name)

    print len(b1Names_notInFile2),"branches not in ",file2
    print b1Names_notInFile2

    b2Names_notInFile1 = []
    for b2Name in branchNames2:
        if b2Name not in branchNames1:
            b2Names_notInFile1.append(b2Name)
            #print b2Name,"not in",file1
    print len(b2Names_notInFile1),"branches not in ",file1
    print b2Names_notInFile1





inFileData = ROOT.TFile(o.inputFileName1,"READ")
inFileMC   = ROOT.TFile(o.inputFileName2,"READ")


compBranches(inFileData, inFileMC, "Events")
print "\n"*3
compBranches(inFileData, inFileMC, "Runs")
print "\n"*3
compBranches(inFileData, inFileMC, "LuminosityBlocks")

