from __future__ import print_function
import ROOT

eos_base = 'root://cmseos.fnal.gov//store/user/pbryant/condor/'

years = ['2016', '2017', '2018']

periods = {'2016': 'BCDEFGH',
           '2017': 'BCDEF',
           '2018': 'ABCD'}

ULMC = ['ZZ4b', 'ZH4b', 'ggZH4b',
        'TTToHadronic', 'TTToSemiLeptonic', 'TTTo2L2Nu']



datasets=[]
for year in years:
    datasets += ['HH4b'+year]
    if year == '2016':
        for process in ULMC:
            datasets += [process+'2016_preVFP', process+'2016_postVFP']
    else:
        for process in ULMC:
            datasets += [process+year]

    for period in periods[year]:
        datasets += ['data'+year+period]

print('Dataset                        | new/old |      new |      old')
for dataset in datasets:
    f = ROOT.TFile.Open(eos_base+dataset+'/picoAOD.root')
    t = f.Get('Events')
    n = t.GetEntries()
    f.Close()
    f = ROOT.TFile.Open(eos_base+dataset+'/picoAOD0.root')
    t = f.Get('Events')
    n0 = t.GetEntries()
    f.Close()
    if n!=n0:
        greater = 'WARNING' if n0>n else ''
        percent = '%3.1f%%'%(100.0*n/n0)
        number  = str(n) 
        number0 = str(n0)
        print(dataset.ljust(30), '|', percent.rjust(7), '|', number.rjust(8), '|', number0.rjust(8), greater)
        # greater = 'picoAOD is bigger than picoAOD0' if n>n0 else 'picoAOD0 is bigger than picoAOD'
        # print(dataset, '| picoAOD0-picoAOD:',n0-n, greater, '| picoAOD:', n, '| picoAOD0:', n0)
