from __future__ import print_function
import ROOT
ROOT.gROOT.SetBatch(True)
import sys

channels = ['zz','zh','hh']
processes = ['ZZ', 'ZH', 'HH', 'tt', 'mj', 'data_obs']
        
def addYears(f, fit):
    hists = []
    for ch in channels:
        for p in processes:
            try:
                f.Get('%s_%s/%s'%(ch,fit,p)).IsZombie()
                # hist already exists, don't need to make it
            except ReferenceError:
                # make sum of years
                hists.append( f.Get('%s6_%s/%s'%(ch,fit,p)) )
                hists[-1].Add(f.Get('%s7_%s/%s'%(ch,fit,p)))
                hists[-1].Add(f.Get('%s8_%s/%s'%(ch,fit,p)))

                try:
                    f.Get('%s_%s'%(ch,fit)).IsZombie()
                    # fit already exists
                except ReferenceError:
                    # make fit
                    f.mkdir('%s_%s'%(ch,fit))
                f.cd('%s_%s'%(ch,fit))
                hists[-1].Write()

fileName = sys.argv[1]
print(fileName)
f = ROOT.TFile(fileName, 'UPDATE')
addYears(f, 'prefit')
addYears(f, 'postfit')
f.Close()
