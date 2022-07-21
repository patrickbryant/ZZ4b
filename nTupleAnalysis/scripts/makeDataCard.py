from __future__ import print_function
import os, sys, pickle

SRs = ['zz', 'zh', 'hh']
eras = ['6', '7', '8']

outfile = sys.argv[1]
histsfile = sys.argv[2]
closurefile = sys.argv[3]
mcSystsfile = sys.argv[4]
classifier = 'SvB_MA' if 'SvB_MA' in histsfile else 'SvB'

print('Templates from', histsfile)

closureSysts = []
for SR in SRs:
    print('Closure systematics from', closurefile+SR+'.pkl')
    with open(closurefile+SR+'.pkl','rb') as cfile:
        closureSysts += sorted( pickle.load(cfile).keys() )
closureSysts = [s.replace('Up', '') for s in closureSysts if 'Up' in s]

print('btag, trigger, JEC systematics from', mcSystsfile, classifier)
btagSysts = []
with open(mcSystsfile,'rb') as sfile:
    mcSysts = pickle.load(sfile)[classifier]
    keys = []
    for systs in mcSysts.values():
        keys += systs.keys()
    mcSysts = sorted(set(keys))
    mcSysts = [s.replace('Up', '') for s in mcSysts if 'Up' in s]
    for s in mcSysts:
        if 'lf' in s: btagSysts.append(s)
        if 'hf' in s: btagSysts.append(s)
        if 'cf' in s: btagSysts.append(s)


class Channel:
    def __init__(self, SR, era, obs=-1):
        self.name = '%4s'%(SR+era)
        self.SR   = SR
        self.era  =    era
        self.obs  = '%4d'%obs


class Process:
    def __init__(self, name, index, rate=-1):
        self.name  = name
        self.title = '%4s'%name
        self.index = '%4d'%index
        self.rate  = '%4d'%rate


class Column:
    def __init__(self, channel, process):
        self.channel = channel
        self.process = process
        self.name = channel.name
        self.lumi = '%4s'%(str(1.03) if int(process.index)<1 else '-') # only signals have lumi uncertainty

        self.closureSysts = {}
        for nuisance in closureSysts:
            self.closureSysts[nuisance] = '%4s'%('1' if self.channel.SR in nuisance and self.process.name == 'mj' else '-')

        self.mcSysts = {}
        for nuisance in mcSysts:
            self.mcSysts[nuisance] = '%4s'%'-'
            if int(self.process.index)>0: continue
            if self.process.name == 'HH' and 'VFP'     in nuisance:                       continue # only apply 2016_*VFP systematics to ZZ and ZH
            if self.process.name != 'HH' and 'VFP' not in nuisance and   '6' in nuisance: continue # only apply 2016 systematics to HH
            if 'stats' in nuisance: # years are uncorrelated
                if self.channel.era not in nuisance: continue
            self.mcSysts[nuisance] = '%4s'%'1'
                

channels = []
for era in eras:
    for SR in SRs:
        channels.append( Channel(SR, era) )

processes = [Process('ZZ', -2), Process('ZH', -1), Process('HH', 0), Process('mj', 1), Process('tt', 2)]

columns = []
for channel in channels:
    for process in processes:
        columns.append( Column(channel, process) )

hline = len(columns)*(4+1)+30+5+1+1
hline = '-'*hline
lines = []
lines.append('imax %d number of channels'%(len(SRs)*len(eras)))
lines.append('jmax 4 number of processes minus one') # zz, zh, hh, mj, tt is five processes, so jmax is 4
lines.append('kmax * number of systematics')
lines.append(hline)
lines.append('shapes * * '+histsfile+' $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC')
lines.append(hline)
lines.append('%-30s %5s %s'%('bin',         '', ' '.join([channel.name for channel in channels])))
lines.append('%-30s %5s %s'%('observation', '', ' '.join([channel.obs  for channel in channels])))
lines.append(hline)
lines.append('%-30s %5s %s'%('bin',         '', ' '.join([column.name for column in columns])))
lines.append('%-30s %5s %s'%('process',     '', ' '.join([column.process.title for column in columns])))
lines.append('%-30s %5s %s'%('process',     '', ' '.join([column.process.index for column in columns])))
lines.append('%-30s %5s %s'%('rate',        '', ' '.join([column.process.rate  for column in columns])))
lines.append(hline)
lines.append('%-30s %5s %s'%('lumi',     'lnN', ' '.join([column.lumi for column in columns])))
# jer                 shape   1 1 1 - -  1 1 1 - -  1 1 1 - -    1 1 1 - -  1 1 1 - -  1 1 1 - -    1 1 1 - -  1 1 1 - -  1 1 1 - -
# jesTotal            shape   1 1 1 - -  1 1 1 - -  1 1 1 - -    1 1 1 - -  1 1 1 - -  1 1 1 - -    1 1 1 - -  1 1 1 - -  1 1 1 - -
for nuisance in closureSysts:
    lines.append('%-30s %5s %s'%(nuisance, 'shape', ' '.join([column.closureSysts[nuisance] for column in columns])))
for nuisance in mcSysts:
    lines.append('%-30s %5s %s'%(nuisance, 'shape', ' '.join([column.     mcSysts[nuisance] for column in columns])))
lines.append(hline)
lines.append('* autoMCStats 0 1 1')
lines.append(hline)
lines.append('multijet group = %s'%(' '.join(closureSysts)))
lines.append('btag     group = %s'%(' '.join(   btagSysts)))
lines.append('trig     group = trigger_emulation')
lines.append('lumi     group = lumi')


with open(outfile, 'w') as ofile:
    for line in lines:
        print(line)
        ofile.write(line+'\n')
    
