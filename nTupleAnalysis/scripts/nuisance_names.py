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


years = ['2016', '2017', '2018']
for ch in channels:
    for year in years:
        for bin in range(25):
            names['prop_bin%s%s_bin%d'%(ch,year[-1],bin)] = '%s %s bin %d'%(year, ch.upper(), bin)

names['lf'] = 'b-tag light flavor'
names['hf'] = 'b-tag heavy flavor'
names['trigger_emulation'] = 'trigger emulation'

for year in years:
    names['hfstats1_'+year] = year+' b-tag heavy stats 1'
    names['hfstats2_'+year] = year+' b-tag heavy stats 2'
    names['lfstats1_'+year] = year+' b-tag light stats 1'
    names['lfstats2_'+year] = year+' b-tag light stats 2'

with open('ZZ4b/nTupleAnalysis/combine/nuisance_names.json', 'w') as f:
    json.dump(names, f)
