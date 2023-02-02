import ROOT
regionName = {'SB': 'Sideband',
              'CR': 'Control Region',
              'SR': 'Signal Region',
              'notSR': 'Sideband',
              'SRNoHH': 'Signal Region (Veto HH)',
          }
region='SR'
classifier = 'SvB'
inputfilename = f'ZZ4b/nTupleAnalysis/combine/hists_closure_3bDvTMix4bDvT_{region}_weights_newSBDef_vs_nJet.root'
outputfilename=inputfilename.replace('.root','_normalized.root')
mixname='3bDvTMix4bDvT_v'
years=['RunII','2016','2017','2018']
nmix = 15
channels = []
channels.append('zz')
channels.append('zh')
channels.append('hh')
njetbins = ['_eq4','_eq5','_eq6','_eq7','_ge8']

suffixName = {'': '',
              '_eq4': ' nJet = 4',
              '_eq5': ' nJet = 5',
              '_eq6': ' nJet = 6',
              '_eq7': ' nJet = 7',
              '_ge7': ' nJet #geq 7',
              '_ge8': ' nJet #geq 8',
              '_nJet': ''}


def compute_norm(mj, tt, dt):
    nmj, ntt, ndt = mj.Integral(), tt.Integral(), dt.Integral()
    print(f'     nmj = {nmj}')
    print(f'     ntt = {ntt}')
    print(f'     ndt = {ndt}')
    norm  = (ndt-ntt)/nmj
    print(f'    norm = {norm}')
    return norm

def write_hist(hist):
    hist.SetName(hist.GetName().split('/')[-1])
    hist.Write()

class Hist2d:
    def __init__(self, inputfile, path):
        self.h2d = inputfile.Get(path)
        self.h2d.SetName(f'{path}_h2d')
        self.path = path
        self.h1d = {}
        for njb in njetbins:
            self.h1d[njb] = self.h2d.ProjectionX(f'{self.path}{njb}', int(njb[-1])+1, int(njb[-1])+1 if 'eq' in njb else -1, 'e') # bin 5 is njet==4 because bin count starts at 1, not zero. Zero is underflow
            self.h1d[njb].SetDirectory(0)
        self.h1d['_nJet'] = self.h2d.ProjectionY(f'{self.path}_nJet', 1, -1, 'e')
        self.h1d['_nJet'].SetDirectory(0)
        if 'multijet' in self.path:
            self.h1d['_nJet_unnormalized'] = self.h1d['_nJet'].Clone()
            self.h1d['_nJet_unnormalized'].SetName(f'{self.path}_nJet_unnormalized')
            self.h1d['_nJet_unnormalized'].SetDirectory(0)

            self.h1d['_unnormalized'] = self.h1d[njetbins[0]].Clone()
            self.h1d['_unnormalized'].SetName(self.path+'_unnormalized')
            self.h1d['_unnormalized'].SetDirectory(0)
            for njb in njetbins:
                self.h1d[njb+'_unnormalized'] = self.h1d[njb].Clone()
                self.h1d[njb+'_unnormalized'].SetName(f'{self.path}{njb}_unnormalized')
                self.h1d[njb+'_unnormalized'].SetDirectory(0)
            for njb in njetbins[1:]:
                self.h1d['_unnormalized'].Add(self.h1d[njb+'_unnormalized'])

    def sum_njb(self):
        self.h1d[''] = self.h1d[njetbins[0]].Clone()
        self.h1d[''].SetName(self.path)
        self.h1d[''].SetDirectory(0)
        for njb in njetbins[1:]:
            self.h1d[''].Add(self.h1d[njb])


class Process:
    def __init__(self, inputfile, mix, ch, year):
        self.mix = mix
        self.ch = ch
        self.path = f'{mixname}{mix}/{ch}{year}'
        self.mj = Hist2d(inputfile, f'{self.path}/multijet')
        self.tt = Hist2d(inputfile, f'{self.path}/ttbar')
        self.dt = Hist2d(inputfile, f'{self.path}/data_obs')
        # self.norm()
        self.tt.sum_njb()
        self.dt.sum_njb()
        self.norm = {}

    def compute_norm(self):
        print(f'{self.ch} {mixname}{mix}')
        for njb in njetbins:
            print(f'    {njb}')
            self.norm[njb] = compute_norm(self.mj.h1d[njb+'_unnormalized'], self.tt.h1d[njb], self.dt.h1d[njb])
        
    def normalize(self, norm=None):
        if norm is None: norm = self.norm
        for njb in njetbins:
            self.mj.h1d[njb].Scale( norm[njb] )
            for bin in range(int(njb[-1])+1, int(njb[-1])+2 if 'eq' in njb else 17):
                c, e = self.mj.h1d['_nJet'].GetBinContent(bin), self.mj.h1d['_nJet'].GetBinError(bin)
                self.mj.h1d['_nJet'].SetBinContent(bin, c*norm[njb])
                self.mj.h1d['_nJet'].SetBinError  (bin, e*norm[njb])
                mj, tt, dt = self.mj.h1d['_nJet'].GetBinContent(bin), self.tt.h1d['_nJet'].GetBinContent(bin), self.dt.h1d['_nJet'].GetBinContent(bin)
                print(f'{njb}: mj+tt = {mj+tt} = {dt} = dt')
        self.mj.sum_njb()

    def write(self,outputfile):
        outputfile.cd(self.path)
        for njb in njetbins+['', '_nJet']:
            write_hist(self.mj.h1d[njb+'_unnormalized'])
            write_hist(self.mj.h1d[njb])
            write_hist(self.tt.h1d[njb])
            write_hist(self.dt.h1d[njb])


class MixSum:
    def __init__(self, mixes, year):
        self.ch = mixes[0].ch
        self.year = year
        self.path = f'{ch}{year}'
        self.mj = {}
        self.dt = {}
        self.tt = {}
        for njb in njetbins+['','_nJet']:
            self.mj[njb+'_unnormalized'] = mixes[0].mj.h1d[njb+'_unnormalized'].Clone()
            self.mj[njb+'_unnormalized'].SetName(f'{self.path}/multijet{njb}_unnormalized')
            self.mj[njb+'_unnormalized'].SetDirectory(0)
            self.mj[njb] = mixes[0].mj.h1d[njb].Clone()
            self.mj[njb].SetName(f'{self.path}/multijet{njb}')
            self.mj[njb].SetDirectory(0)
            self.dt[njb] = mixes[0].dt.h1d[njb].Clone()
            self.dt[njb].SetName(f'{self.path}/data_obs{njb}')
            self.dt[njb].SetDirectory(0)
            self.tt[njb] = mixes[0].tt.h1d[njb].Clone()
            self.tt[njb].SetName(f'{self.path}/ttbar{njb}')
            self.tt[njb].SetDirectory(0)

            for mix in mixes[1:]:
                self.mj[njb+'_unnormalized'].Add(mix.mj.h1d[njb+'_unnormalized'])
                self.mj[njb].Add(mix.mj.h1d[njb])
                self.dt[njb].Add(mix.dt.h1d[njb])
            self.mj[njb+'_unnormalized'].Scale(1.0/nmix)
            self.mj[njb].Scale(1.0/nmix)
            self.dt[njb].Scale(1.0/nmix)

    def write(self, outputfile):
        outputfile.cd(self.path)
        for njb in njetbins+['','_nJet']:
            write_hist(self.mj[njb+'_unnormalized'])
            write_hist(self.mj[njb])
            write_hist(self.dt[njb])
            write_hist(self.tt[njb])


        
inputfile=ROOT.TFile(inputfilename, 'READ')
processes = {}
for mix in range(nmix):
    processes[mix] = {}
    for ch in channels:
        processes[mix][ch] = {}
        for year in years:
            processes[mix][ch][year] = Process(inputfile, mix, ch, year)

for mix in range(nmix):
    for ch in channels:
        processes[mix][ch]['RunII'].compute_norm()
        for year in years:
            processes[mix][ch][year].normalize( processes[mix][ch]['RunII'].norm )

mixsums = {}
for ch in channels:
    mixsums[ch] = {}
    for year in years:
        mixsums[ch][year] = MixSum([processes[mix][ch][year] for mix in range(nmix)], year)
        

inputfile.Close()

outputfile=ROOT.TFile(outputfilename, 'RECREATE')
for mix in range(nmix):
    outputfile.mkdir(f'{mixname}{mix}')
    for ch in channels:
        for year in years:
            outputfile.mkdir(f'{mixname}{mix}/{ch}{year}')
            processes[mix][ch][year].write(outputfile)

for ch in channels:
    for year in years:
        outputfile.mkdir(f'{ch}{year}')
        mixsums[ch][year].write(outputfile)

# outputfile.Write()
outputfile.Close()

# f=ROOT.TFile(outputfilename, 'READ')
# f.cd(f'{mixname}0/zz2016')
# h=f.Get(f'{mixname}0/zz2016/multijet')
# for bin in range(1,h.GetSize()-2):
#     print(bin,h.GetBinContent(bin))
# f.ls()
# f.Close()
# exit()

import sys
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import collections
import PlotTools

lumiDict   = {"2016":  35.9,#35.8791
              "2017":  36.7,#36.7338
              "2018":  60.0,#59.9656
              "RunII":132.6,
              }


def plotMix(ch, year, mix='average', suffix='', rebin=1, norm=''):
    samples=collections.OrderedDict()
    samples[outputfilename] = collections.OrderedDict()
    if type(mix)==int:
        samples[outputfilename][f'{mixname}{mix}/{ch}{year}/data_obs{suffix}'] = {
            'label' : f'Mixed Data Set {mix}, {lumiDict[year]}/fb',
            'legend': 1,
            'isData' : True,
            #'drawOptions': 'P ex0',
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}
        samples[outputfilename][f'{mixname}{mix}/{ch}{year}/multijet{suffix}{norm}'] = {
            'label' : f'Multijet Model {mix}',
            'legend': 2,
            'stack' : 3,
            'ratio' : 'denom A',
            'color' : 'ROOT.kYellow'}
        samples[outputfilename][f'{mixname}{mix}/{ch}{year}/ttbar{suffix}'] = {
            'label' : '#lower[0.10]{t#bar{t}}',
            'legend': 3,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : 'ROOT.kAzure-9'}
    else:
        samples[outputfilename][f'{ch}{year}/data_obs{suffix}'] = {
            'label' : f'#LTMixed Data#GT {lumiDict[year]}/fb',
            'legend': 1,
            'isData' : True,
            #'drawOptions': 'P ex0',
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}
        samples[outputfilename][f'{ch}{year}/multijet{suffix}{norm}'] = {
            'label' : '#LTMultijet#GT',
            'legend': 2,
            'stack' : 3,
            'ratio' : 'denom A',
            'color' : 'ROOT.kYellow'}
        samples[outputfilename][f'{ch}{year}/ttbar{suffix}'] = {
            'label' : '#lower[0.10]{t#bar{t}}',
            'legend': 3,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : 'ROOT.kAzure-9'}

    xTitle = f'{classifier} P(Signal) #cbar P({ch.upper()}) is largest'
    if 'nJet' in suffix:
        xTitle = f'Number of Selected Jets #cbar P({ch.upper()}) is largest'

    parameters = {'titleLeft'   : '#bf{CMS} Internal',
                  'titleCenter' : regionName[region]+suffixName[suffix],
                  'titleRight'  : 'Pass #DeltaR(j,j)',
                  'maxDigits'   : 4,
                  'ratioErrors': True,
                  'ratio'     : True,
                  'rMin'      : 0.9,
                  'rMax'      : 1.1,
                  'rebin'     : rebin,
                  'rTitle'    : 'Data / Bkgd.',
                  'xTitle'    : xTitle,
                  'rPadxTitleOffset': 0.8,
                  'yTitle'    : 'Events',
                  'outputDir': f'closureFits/njet/{classifier}/{region}/{ch}/',
                  'outputName': f'mix_{mix}{suffix}{norm}'}

    print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
    PlotTools.plot(samples, parameters, debug=False)


def plotNormComparison(ch, year, mix='average', suffix='', rebin=1, norm=''):
    samples=collections.OrderedDict()
    samples[outputfilename] = collections.OrderedDict()
    if type(mix)==int:
        samples[outputfilename][f'{mixname}{mix}/{ch}{year}/multijet{suffix}_unnormalized'] = {
            'label' : f'Multijet Model {mix}',
            'legend': 1,
            'stack' : 3,
            'ratio' : 'denom A',
            'color' : 'ROOT.kYellow'}
        samples[outputfilename][f'{mixname}{mix}/{ch}{year}/multijet{suffix}'] = {
            'label' : f'Multijet Model {mix} nJet Normalized',
            'legend': 2,
            # 'isData': True,
            'ratioDrawOptions': 'HIST',
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlue'}
    else:
        samples[outputfilename][f'{ch}{year}/multijet{suffix}_unnormalized'] = {
            'label' : '#LTMultijet#GT',
            'legend': 1,
            'stack' : 3,
            'ratio' : 'denom A',
            'color' : 'ROOT.kYellow'}
        samples[outputfilename][f'{ch}{year}/multijet{suffix}'] = {
            'label' : '#LTMultijet#GT nJet Normalized',
            'legend': 2,
            'isData': True,
            'drawOptions': 'HIST',
            'ratioDrawOptions': 'HIST',
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlue'}

    xTitle = classifier+' P(Signal) #cbar P('+ch.upper()+') is largest'

    parameters = {'titleLeft'   : '#bf{CMS} Internal',
                  'titleCenter' : regionName[region]+suffixName[suffix],
                  'titleRight'  : 'Pass #DeltaR(j,j)',
                  'maxDigits'   : 4,
                  'ratioErrors': True,
                  'ratio'     : True,
                  'rMin'      : 0.95,
                  'rMax'      : 1.05,
                  'rebin'     : rebin,
                  'rTitle'    : 'Norm. / Nominal',
                  'rPadyTitleOffset': 1.4,
                  'xTitle'    : xTitle,
                  'rPadxTitleOffset': 0.8,
                  'yTitle'    : 'Events',
                  'outputDir': f'closureFits/njet/{classifier}/{region}/{ch}/',
                  'outputName': f'nJetNormComparison_{mix}{suffix}'}

    print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
    PlotTools.plot(samples, parameters, debug=False)

for ch in channels:
    plotNormComparison(ch, 'RunII')
    for njb in njetbins+['','_nJet']:
        plotMix(ch, 'RunII', suffix=njb)
        plotMix(ch, 'RunII', suffix=njb, norm='_unnormalized')
