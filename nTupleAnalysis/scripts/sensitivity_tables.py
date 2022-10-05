from __future__ import print_function
import sys

classifier = sys.argv[1]

channels = ['_zz','_zh','_hh','']
sig = {}
sig_stat = {}
lim = {}
lim_stat = {}

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

print(sig)
print(sig_stat)
print(lim)
print(lim_stat)

table = [
    '\\begin{table}',
    '\\centering',
    '\\begin{tabular}{c|cccc}',
    '%s                       & \\ZZ  & \\ZH  & \\HH  & Combined \\\\'%(classifier.replace('_',' ')),
    '\\hline',
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
    
