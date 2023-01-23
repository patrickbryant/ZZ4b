from __future__ import print_function
import os, sys, pickle
from copy import copy

SRs = ['zz', 'zh', 'hh']
eras = ['6', '7', '8']
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM?rev=167#LumiComb  # https://gitlab.cern.ch/hh/naming-conventions
uncert_lumi_corr = {'6': '1.006', '7': '1.009', '8': '1.020'}
uncert_lumi_1718 = {              '7': '1.006', '8': '1.002'}
uncert_lumi_2016 = {'6': '1.010'                            }
uncert_lumi_2017 = {              '7': '1.020'              }
uncert_lumi_2018 = {                            '8': '1.015'}
uncert_br = {'ZH': '1.013', 'HH': '1.025'} # https://gitlab.cern.ch/hh/naming-conventions https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR?rev=22#Higgs_2_fermions
uncert_pdf_HH = {'HH': '1.030'} #https://gitlab.cern.ch/hh/naming-conventions
uncert_pdf_ZH = {'ZH': '1.013'}
uncert_pdf_ZZ = {'ZZ': '1.001'} #https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV?rev=27
uncert_scale_ZZ = {'ZZ': '1.002'} #https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV?rev=27
uncert_scale_ZH = {'ZH': '0.97/1.038'} #https://gitlab.cern.ch/hh/naming-conventions
uncert_scale_HH = {'HH': '0.95/1.022'}
uncert_alpha_s  = {'ZH': '1.009'} #https://gitlab.cern.ch/hh/naming-conventions 
# all three signal processes have different production modes and so do not have shared pdf or scale nuisance parameters so they can be combined into a single parameter
uncert_xs = {'ZZ': '1.002', 'ZH': '0.966/1.041', 'HH': '0.942/1.037'}

outfile = sys.argv[1]
histsfile = sys.argv[2]
try:
    mcSystsfile = sys.argv[3]
except IndexError: 
    mcSystsfile = None
    print('No MC Systematics File')

try:
    closurefile = sys.argv[4]
except IndexError: 
    closurefile = None

stat_only = True if closurefile is None and mcSystsfile is None else False
classifier = 'SvB_MA' if 'SvB_MA' in histsfile else 'SvB'

print('Templates from', histsfile)

closureSysts = []
if closurefile:
    for SR in SRs:
        print('Closure systematics from', closurefile+SR+'.pkl')
        with open(closurefile+SR+'.pkl','rb') as cfile:
            closureSysts += sorted( pickle.load(cfile).keys() )
    closureSysts = [s.replace('Up', '') for s in closureSysts if 'Up' in s]

btagSysts = []
juncSysts = []
mcSysts = []
if mcSystsfile:
    print('btag, trigger, JEC systematics from', mcSystsfile, classifier)
    with open(mcSystsfile,'rb') as sfile:
        mcSysts = pickle.load(sfile)[classifier]
        keys = []
        for systs in mcSysts.values():
            keys += systs.keys()
        mcSysts = sorted(set(keys))
        mcSysts = [s.replace('Up', '') for s in mcSysts if 'Up' in s and 'Total' not in s]
        for s in mcSysts:
            if 'btag' in s: btagSysts.append(s)
            if 'junc' in s: juncSysts.append(s)


class Channel:
    def __init__(self, SR, era, obs=-1):
        self.name = '%3s'%(SR+era)
        self.SR   = SR
        self.era  =    era
        self.obs  = '%3d'%obs
    def space(self, s='  '):
        self.name = self.name+s
        self.obs  = self.obs +s


class Process:
    def __init__(self, name, index, rate=-1):
        self.name  = name
        self.title = '%3s'%name
        self.index = '%3d'%index
        self.rate  = '%3d'%rate
    def space(self, s='  '):
        self.title = self.title+s
        self.index = self.index+s
        self.rate  = self.rate +s


class Column:
    def __init__(self, channel, process):
        self.channel = copy(channel)
        self.process = copy(process)
        self.name = channel.name
        # self.lumi = '%s'%(str(uncert_lumi[self.channel.era]) if int(process.index)<1 else '-') # only signals have lumi uncertainty

        self.closureSysts = {}
        for nuisance in closureSysts:
            self.closureSysts[nuisance] = '%3s'%('1' if self.channel.SR in nuisance and self.process.name == 'mj' else '-')

        self.mcSysts = {}
        for nuisance in mcSysts:
            self.mcSysts[nuisance] = '%3s'%'-'
            if int(self.process.index)>0: continue
            if self.process.name == 'HH' and 'VFP'     in nuisance:                       continue # only apply 2016_*VFP systematics to ZZ and ZH
            if self.process.name != 'HH' and 'VFP' not in nuisance and   '6' in nuisance: continue # only apply 2016 systematics to HH
            if '201' in nuisance: # years are uncorrelated
                if self.channel.era not in nuisance: continue
            self.mcSysts[nuisance] = '%3s'%'1'

        self.br = uncert_br.get(self.process.name, '-')
        self.xs = uncert_xs.get(self.process.name, '-')
        self.lumi_corr = uncert_lumi_corr.get(self.channel.era, '-') if int(process.index)<1 else '-' # only signals have lumi uncertainty
        self.lumi_1718 = uncert_lumi_1718.get(self.channel.era, '-') if int(process.index)<1 else '-' 
        self.lumi_2016 = uncert_lumi_2016.get(self.channel.era, '-') if int(process.index)<1 else '-' 
        self.lumi_2017 = uncert_lumi_2017.get(self.channel.era, '-') if int(process.index)<1 else '-' 
        self.lumi_2018 = uncert_lumi_2018.get(self.channel.era, '-') if int(process.index)<1 else '-' 
                
        if self.process.name == 'tt': # add space for easier legibility
            self.space('  ')
            if self.channel.SR == 'hh': # add extra space for easier legibility
                self.space('  ')
                self.lumi_corr = self.lumi_corr + '    '
                self.lumi_1718 = self.lumi_1718 + '    '
                self.lumi_2016 = self.lumi_2016 + '    '
                self.lumi_2017 = self.lumi_2017 + '    '
                self.lumi_2018 = self.lumi_2018 + '    '
                self.br   = self.br   + '    '
                self.xs   = self.xs   + '    '

    def space(self, s='  '):
        self.channel.space(s)
        self.name = self.channel.name
        self.process.space(s)
        for nuisance in closureSysts:
            self.closureSysts[nuisance] = self.closureSysts[nuisance]+s
        for nuisance in mcSysts:
            self.mcSysts[nuisance] = self.mcSysts[nuisance]+s
        
        

channels = []
for era in eras:
    for SR in SRs:
        channels.append( Channel(SR, era) )

processes = [Process('ZZ', -2), Process('ZH', -1), Process('HH', 0), Process('mj', 1), Process('tt', 2)]

columns = []
for channel in channels:
    for process in processes:
        columns.append( Column(channel, process) )

hline = len(columns)*(4+1)+35+5+1+1
hline = '-'*hline
lines = []
lines.append('imax %d number of channels'%(len(SRs)*len(eras)))
lines.append('jmax 4 number of processes minus one') # zz, zh, hh, mj, tt is five processes, so jmax is 4
lines.append('kmax * number of systematics')
lines.append(hline)
lines.append('shapes * * '+histsfile+' $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC')
lines.append(hline)
lines.append('%-35s %5s %s'%('bin',         '', ' '.join([channel.name for channel in channels])))
lines.append('%-35s %5s %s'%('observation', '', ' '.join([channel.obs  for channel in channels])))
lines.append(hline)
lines.append('%-35s %5s %s'%('bin',         '', ' '.join([column.name for column in columns])))
lines.append('%-35s %5s %s'%('process',     '', ' '.join([column.process.title for column in columns])))
lines.append('%-35s %5s %s'%('process',     '', ' '.join([column.process.index for column in columns])))
lines.append('%-35s %5s %s'%('rate',        '', ' '.join([column.process.rate  for column in columns])))
lines.append(hline)
for nuisance in closureSysts:
    lines.append('%-35s %5s %s'%(nuisance, 'shape', ' '.join([column.closureSysts[nuisance] for column in columns])))
for nuisance in mcSysts:
    lines.append('%-35s %5s %s'%(nuisance, 'shape', ' '.join([column.     mcSysts[nuisance] for column in columns])))
lines.append('%-35s %5s %s'%('BR_hbb',   'lnN', ' '.join([column.br        for column in columns])))
lines.append('%-35s %5s %s'%('xs',       'lnN', ' '.join([column.xs        for column in columns])))
lines.append('%-35s %5s %s'%('lumi_corr','lnN', ' '.join([column.lumi_corr for column in columns])))
lines.append('%-35s %5s %s'%('lumi_1718','lnN', ' '.join([column.lumi_1718 for column in columns])))
lines.append('%-35s %5s %s'%('lumi_2016','lnN', ' '.join([column.lumi_2016 for column in columns])))
lines.append('%-35s %5s %s'%('lumi_2017','lnN', ' '.join([column.lumi_2017 for column in columns])))
lines.append('%-35s %5s %s'%('lumi_2018','lnN', ' '.join([column.lumi_2018 for column in columns])))
if not stat_only:
    lines.append(hline)
    lines.append('* autoMCStats 0 1 1')
lines.append(hline)
if closureSysts:
    lines.append('multijet group = %s'%(' '.join(closureSysts)))
if mcSysts:
    lines.append('btag     group = %s'%(' '.join(   btagSysts)))
    lines.append('junc     group = %s'%(' '.join(   juncSysts)))
    lines.append('trig     group = trigger_emulation')
lines.append('lumi     group = lumi_corr lumi_1718 lumi_2016 lumi_2017 lumi_2018')
lines.append('theory   group = BR_hbb xs')
if not stat_only:
    lines.append('others   group = trigger_emulation lumi_corr lumi_1718 lumi_2016 lumi_2017 lumi_2018 BR_hbb xs %s'%(' '.join(juncSysts)))


with open(outfile, 'w') as ofile:
    for line in lines:
        print(line)
        ofile.write(line+'\n')
    
