from __future__ import print_function
import ROOT
ROOT.gROOT.SetBatch(True)
import sys

channels = ['zz','zh','hh']
processes = ['ZZ', 'ZH', 'HH', 'tt', 'mj', 'data']
        
def addYears(f, directory):
    hists = []
    for ch in channels:
        for p in processes:
            try:
                f.Get('%s/%s/%s'%(directory,ch,p)).IsZombie()
                # hist already exists, don't need to make it
            except ReferenceError:
                # make sum of years
                if p != 'data':
                    hists.append( f.Get('%s/%s6/%s'%(directory,ch,p)) )
                    hists[-1].Add(f.Get('%s/%s7/%s'%(directory,ch,p)))
                    hists[-1].Add(f.Get('%s/%s8/%s'%(directory,ch,p)))
                else:
                    # special case of adding together TGraphAsymmErrors
                    g6 = f.Get('%s/%s6/%s'%(directory,ch,p))
                    g7 = f.Get('%s/%s7/%s'%(directory,ch,p))
                    g8 = f.Get('%s/%s8/%s'%(directory,ch,p))
                    n = g6.GetN()
                    hists.append( ROOT.TH1F('data', '', n, 0, n) )
                    y6, y7, y8 = g6.GetY(), g7.GetY(), g8.GetY()
                    for i in range(n):
                        hists[-1].SetBinContent(i+1, y6[i]+y7[i]+y8[i])

                try:
                    f.Get('%s/%s'%(directory,ch)).IsZombie()
                    # directory already exists
                except ReferenceError:
                    # make directory
                    f.mkdir('%s/%s'%(directory,ch))
                f.cd('%s/%s'%(directory,ch))
                hists[-1].Write()

fileName = sys.argv[1]
print(fileName)
f = ROOT.TFile(fileName, 'UPDATE')
addYears(f, 'shapes_fit_b')
#addYears(f, 'shapes_fit_s')
f.Close()
