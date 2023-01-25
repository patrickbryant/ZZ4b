from __future__ import print_function
from ROOT import TFile
import sys

#classifier = sys.argv[1]

inputLumi = 132.8
inputFile = sys.argv[1] #'ZZ4b/nTupleAnalysis/combine/hists_%s.root'%classifier
no_rebin = True if 'no_rebin' in inputFile else False
print(inputFile,'add bin error to bin content for no rebin extrapolation' if no_rebin else '')


outputLumi = float(sys.argv[2])
outputFile = inputFile.replace('.root', '_%d.root'%outputLumi) #'ZZ4b/nTupleAnalysis/combine/hists_%s_%d.root'%(classifier, outputLumi)
print(outputFile)

scale = outputLumi / inputLumi

inputFile  = TFile(inputFile,  "READ")
outputFile = TFile(outputFile, "RECREATE")


def lumiScale(hist, scale): # scale bin error by sqrt(scale) rather than scale. This is what we want if we are assuming the stats of this hist will grow to match the lumi of future measurments.
    for bin in range(1,hist.GetNbinsX()+1):
        c, e = hist.GetBinContent(bin), hist.GetBinError(bin)
        # if no_rebin: c += e
        hist.SetBinContent(bin, c*scale)
        hist.SetBinError  (bin, e*scale**0.5)
        

for key_tdir in inputFile.GetListOfKeys():
    tdir = key_tdir.ReadObj()
    mj = inputFile.Get(tdir.GetName()+'/mj')
    lumiScale(mj, scale)
    outputFile.mkdir(tdir.GetName())
    # print(mj.GetNbinsX())

    for key_hist in tdir.GetListOfKeys():
        if key_hist.GetName() == 'mj': continue
        hist = key_hist.ReadObj()
        # print(hist.GetNbinsX())
        lumiScale(hist, scale)

        if '_vari_' in hist.GetName(): # want delta from variance to scale with sqrt(N) rather than N
            hist.Add(mj,-1) # get delta described by this variance term
            hist.Scale(scale**-0.5) # scale delta by 1/sqrt(lumi ratio) so that total scale factor of the delta is sqrt(lumi ratio)
            hist.Add(mj) # add back the nominal multijet template
            
        outputFile.cd(tdir.GetName())
        hist.Write()

    mj.Write()

