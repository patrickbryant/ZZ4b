import sys
fileName = sys.argv[1]
print(fileName)
import ROOT

inFile = ROOT.TFile.Open(fileName)
print(inFile.Get("passPreSel/fourTag/mainView/SB/nSelJets").Integral())
