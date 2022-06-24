import json

names = {'rZZ': '#mu_{ZZ}',
         'rZH': '#mu_{ZH}',
         'rHH': '#mu_{HH}'}

# "basis5_hh": "HH basis 5",
# "prop_binzz8_bin24": "2018 ZZ bin 24"
channels = ['zz','zh','hh']
for ch in channels:
    for b in range(5):
        names['basis%d_%s'%(b,ch)] = '%s basis %d'%(ch.upper(), b)


years = ['2016', '2017', '2018']
for ch in channels:
    for year in years:
        for bin in range(25):
            names['prop_bin%s%s_bin%d'%(ch,year[-1],bin)] = '%s %s bin %d'%(year, ch.upper(), bin)

with open('ZZ4b/nTupleAnalysis/combine/nuisance_names.json', 'w') as f:
    json.dump(names, f)
