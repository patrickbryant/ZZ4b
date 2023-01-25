from __future__ import print_function
import sys
from ROOT import TFile

classifier = sys.argv[1]

channels = ['_zz','_zh','_hh','']
sig = {}
sig_stat = {}
lim = {}
lim_stat = {}
rXX = {}
rXX_stat = {}

for ch in channels:
    with open('combinePlots/%s/expected_significance%s.txt'%(classifier, ch) , 'r') as f:
        for line in f:
            line = line.split()
            if not line: continue
            if line[0] == 'Significance:': sig[ch] = float(line[1])
    with open('combinePlots/%s/expected_stat_only_significance%s.txt'%(classifier, ch) , 'r') as f:
        for line in f:
            line = line.split()
            if not line: continue
            if line[0] == 'Significance:': sig_stat[ch] = float(line[1])
    with open('combinePlots/%s/expected_limit%s.txt'%(classifier, ch) , 'r') as f:
        for line in f:
            line = line.split()
            if len(line)<5: continue
            if line[0:2] == ['Expected','97.5%:']: lim[ch] = float(line[4])            
    with open('combinePlots/%s/expected_stat_only_limit%s.txt'%(classifier, ch) , 'r') as f:
        for line in f:
            line = line.split()
            if len(line)<5: continue
            if line[0:2] == ['Expected','97.5%:']: lim_stat[ch] = float(line[4])

    if ch:
        XX = ch.replace('_','').upper()
        f = TFile('combinePlots/%s/higgsCombine.expected_r%s.root'%(classifier, XX))
        t = f.Get('limit')
        rXX[ch] = [None, None, None]
        t.GetEntry(0) # central value
        rXX[ch][0] = getattr(t, 'r%s'%XX)
        t.GetEntry(1) # down 1 sigma
        rXX[ch][1] = getattr(t, 'r%s'%XX) - rXX[ch][0]
        t.GetEntry(2) #   up 1 sigma
        rXX[ch][2] = getattr(t, 'r%s'%XX) - rXX[ch][0]
        f.Close()

        f = TFile('combinePlots/%s/higgsCombine.expected_stat_only_r%s.root'%(classifier, XX))
        t = f.Get('limit')
        rXX_stat[ch] = [None, None, None]
        t.GetEntry(0) # central value
        rXX_stat[ch][0] = getattr(t, 'r%s'%XX)
        t.GetEntry(1) # down 1 sigma
        rXX_stat[ch][1] = getattr(t, 'r%s'%XX) - rXX_stat[ch][0]
        t.GetEntry(2) #   up 1 sigma
        rXX_stat[ch][2] = getattr(t, 'r%s'%XX) - rXX_stat[ch][0]
        f.Close()

print(sig)
print(sig_stat)
print(lim)
print(lim_stat)
print(rXX)
print(rXX_stat)

table = [
    '\\begin{table}',
    '\\renewcommand{\\arraystretch}{1.2}',
    '\\centering',
    '\\begin{tabular}{c|cccc}',
    '%s                       & \\ZZ  & \\ZH  & \\HH  & Combined \\\\'%(classifier.replace('_',' ')),
    '\\hline',
    'Signal Strength (Stat. Only) & $%1.0f_{%3.1f}^{+%3.1f}$ ($%1.0f_{%3.1f}^{+%3.1f}$) & $%1.0f_{%3.1f}^{+%3.1f}$ ($%1.0f_{%3.1f}^{+%3.1f}$) & $%1.0f_{%3.1f}^{+%3.1f}$ ($%1.0f_{%3.1f}^{+%3.1f}$) & -- \\\\'%(rXX['_zz'][0], rXX['_zz'][1], rXX['_zz'][2], rXX_stat['_zz'][0], rXX_stat['_zz'][1], rXX_stat['_zz'][2], rXX['_zh'][0], rXX['_zh'][1], rXX['_zh'][2], rXX_stat['_zh'][0], rXX_stat['_zh'][1], rXX_stat['_zh'][2], rXX['_hh'][0], rXX['_hh'][1], rXX['_hh'][2], rXX_stat['_hh'][0], rXX_stat['_hh'][1], rXX_stat['_hh'][2]),
    'Significance (Stat. Only) $\\sigma$ & %0.2f (%0.2f) & %0.2f (%0.2f) & %0.2f (%0.2f) & %0.2f (%0.2f) \\\\'%(sig['_zz'], sig_stat['_zz'], sig['_zh'], sig_stat['_zh'], sig['_hh'], sig_stat['_hh'], sig[''], sig_stat['']),
    'Limit (Stat. Only) at $95\\%%$ CL & %3.1f (%3.1f) & %3.1f (%3.1f) & %3.1f (%3.1f) & %3.1f (%3.1f) \\\\'%(lim['_zz'], lim_stat['_zz'], lim['_zh'], lim_stat['_zh'], lim['_hh'], lim_stat['_hh'], lim[''], lim_stat['']),
    '\\end{tabular}',
    '\\caption{Expected sensitivity using the %s classifier.}\\label{tab:expected_%s}'%(classifier.replace('_',' '), classifier),
    '\\end{table}',
]

with open('combinePlots/%s/expected_sensitivity.tex'%classifier, 'w') as f:
    for line in table:
        print(line)
        f.write(line+'\n')
    
