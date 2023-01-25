import json

names = {'rZZ': '#mu_{ZZ}',
         'rZH': '#mu_{ZH}',
         'rHH': '#mu_{HH}'}

# "basis5_hh": "HH basis 5",
# "prop_binzz8_bin24": "2018 ZZ bin 24"
channels = ['zz','zh','hh']
for ch in channels:
    for b in range(5):
        names['basis%d_vari_%s'%(b,ch)] = 'variance: %s basis %d'%(ch.upper(), b)
        names['basis%d_bias_%s'%(b,ch)] =     'bias: %s basis %d'%(ch.upper(), b)


years = ['2016_preVFP', '2016_postVFP', '2016', '2017', '2018']
for ch in channels:
    for year in years:
        for bin in range(25):
            names['prop_bin%s%s_bin%d'%(ch,year[-1],bin)] = '%s %s bin %d'%(year, ch.upper(), bin)

names['btag_lf'] = 'b-tag light flavor'
names['btag_hf'] = 'b-tag heavy flavor'
names['trigger_emulation'] = 'trigger emulation'
names['junc_Absolute'] = 'JES Absolute'
names['junc_BBEC1'] = 'JES BBEC1'
names['junc_EC2'] = 'JES EC2'
names['junc_FlavorQCD'] = 'JES Flavor'
names['junc_HF'] = 'JES HF'
names['junc_RelativeBal'] = 'JES Balance'

names['BR_hbb'] = 'BR(H#rightarrowb#bar{b})'
names['xs'] = 'cross section'
names['lumi_corr'] = 'luminosity (correlated)'
names['lumi_1718'] = 'luminosity (2017, 2018)'
names['lumi'] = 'luminosity (uncorrelated)'

for year in years:
    names['btag_hfstats1_'+year] = year.replace('_',' ')+' b-tag heavy stats 1'
    names['btag_hfstats2_'+year] = year.replace('_',' ')+' b-tag heavy stats 2'
    names['btag_lfstats1_'+year] = year.replace('_',' ')+' b-tag light stats 1'
    names['btag_lfstats2_'+year] = year.replace('_',' ')+' b-tag light stats 2'
    names['junc_Absolute_'+year] = year.replace('_',' ')+' JES Absolute'
    names['junc_BBEC1_'+year] = year.replace('_',' ')+' JES BBEC1'
    names['junc_EC2_'+year] = year.replace('_',' ')+' JES EC2'
    names['junc_HF_'+year] = year.replace('_',' ')+' JES HF'
    names['junc_RelativeSample_'+year] = year.replace('_',' ')+' JES Sample'
    names['lumi_'+year] = year+' luminosity'
    

with open('ZZ4b/nTupleAnalysis/combine/nuisance_names.json', 'w') as f:
    json.dump(names, f)
