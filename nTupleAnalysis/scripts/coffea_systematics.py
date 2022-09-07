# source /cvmfs/sft.cern.ch/lcg/views/LCG_102rc1/x86_64-centos7-gcc11-opt/setup.sh
# source /cvmfs/sft.cern.ch/lcg/nightlies/dev4/Wed/coffea/0.7.13/x86_64-centos7-gcc10-opt/coffea-env.sh 
import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import mkpath
import pickle, os, time
from copy import deepcopy
from coffea import hist, processor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from cycler import cycler
import numpy as np
ZZColor = 0.5+np.array([0.00, 0.39, 0.00])/2
ZHColor = 0.5+np.array([1.00, 0.00, 0.01])/2
HHColor = 0.5+np.array([0.02, 0.00, 0.80])/2
MCColor = '#000000'
markerMC = 0 # tick left
markerData = 1 # tick right
#DataColor = (0.75,0.75,0.75, 1.0)
#colors = [ZHColor, ZHColor, ZHColor, '#000000']

bbbb = r'$\bar{\mathrm{b}}\mathrm{b}\bar{\mathrm{b}}\mathrm{b}$'
ZZ4b = r'ZZ$\rightarrow$'+bbbb
ZH4b = r'ZH$\rightarrow$'+bbbb
HH4b = r'HH$\rightarrow$'+bbbb

order = {'zz' : [HH4b, ZH4b, ZZ4b],
         'zh' : [HH4b, ZZ4b, ZH4b],
         'hh' : [ZH4b, ZZ4b, HH4b],
         'all': [HH4b, ZH4b, ZZ4b]}

colors = {'zz' : [ZZColor, ZHColor, HHColor],
          'zh' : [ZHColor, ZZColor, HHColor],
          'hh' : [HHColor, ZZColor, ZHColor],
          'all': [ZZColor, ZHColor, HHColor]}

nbins = {'zz': 25, 'zh': 20, 'hh': 10, 'all': 25}
rebin = {'zz': 4, 'zh': 5, 'hh': 10, 'all': None}

fill_opts = {
    'edgecolor': (0,0,0, 0.5),
    #'alpha': 0.3,
}
stack_err_opts = {
    'label': 'Stat. Unc.',
    #'hatch': '////',
    #'facecolor': 'none',
    #'edgecolor': (0,0,0, 0.3),
    'color': (0,0,0, 0.3),
    'linewidth': 0,
    'zorder': 10,
}
data_err_opts = {
    'linestyle': 'none',
    'marker': '.',
    'markersize': 8.,
    'color': 'k',
    'elinewidth': 1,
}
guide_opts = {
    'linestyle': '-',
    'color': 'k',
    'linewidth': 1,
}


trigWeights = ['Bool', 'MC', 'Data']

JECSyst = ''
bTagSyst = True

btagVariations = ['central']
if 'jes' in JECSyst:
    if 'Down' in JECSyst:
        btagVariations = ['down'+JECSyst.replace('Down','')]
    if 'Up' in JECSyst:
        btagVariations = ['up'+JECSyst.replace('Up','')]
if bTagSyst:
    btagVariations += ['down_hfstats1', 'up_hfstats1']
    btagVariations += ['down_hfstats2', 'up_hfstats2']
    btagVariations += ['down_lfstats1', 'up_lfstats1']
    btagVariations += ['down_lfstats2', 'up_lfstats2']
    btagVariations += ['down_hf', 'up_hf']
    btagVariations += ['down_lf', 'up_lf']
    btagVariations += ['down_cferr1', 'up_cferr1']
    btagVariations += ['down_cferr2', 'up_cferr2']

downup = [(btagVariations[i], btagVariations[i+1]) for i in range(1, len(btagVariations), 2)]

group_zz = ['ZZ4b2016_preVFP', 'ZZ4b2016_postVFP', 'ZZ4b2017', 'ZZ4b2018']
group_zh = ['ZH4b2016_preVFP', 'ZH4b2016_postVFP', 'ZH4b2017', 'ZH4b2018', 'ggZH4b2016_preVFP', 'ggZH4b2016_postVFP', 'ggZH4b2017', 'ggZH4b2018']
group_hh = ['HH4b2016', 'HH4b2017', 'HH4b2018']
group_all = group_zz + group_zh + group_hh
groups = {ZZ4b: group_zz,
          ZH4b: group_zh,
          HH4b: group_hh}
groups_years = {'ZZ4b2016_preVFP': ['ZZ4b2016_preVFP'],
                'ZZ4b2016_postVFP': ['ZZ4b2016_postVFP'],
                'ZZ4b2017': ['ZZ4b2017'],
                'ZZ4b2018': ['ZZ4b2018'],
                'ZH4b2016_preVFP': ['ggZH4b2016_preVFP', 'ZH4b2016_preVFP'],
                'ZH4b2016_postVFP': ['ggZH4b2016_postVFP', 'ZH4b2016_postVFP'],
                'ZH4b2017': ['ggZH4b2017', 'ZH4b2017'],
                'ZH4b2018': ['ggZH4b2018', 'ZH4b2018'],
                'HH4b2016': ['HH4b2016'],
                'HH4b2017': ['HH4b2017'],
                'HH4b2018': ['HH4b2018'],
}
groups_years['ZZ4b2016'] = groups_years['ZZ4b2016_preVFP'] + groups_years['ZZ4b2016_postVFP']
groups_years['ZH4b2016'] = groups_years['ZH4b2016_preVFP'] + groups_years['ZH4b2016_postVFP']
groups_years['2016'] = groups_years['ZZ4b2016'] + groups_years['ZH4b2016'] + groups_years['HH4b2016']
groups_years['2017'] = groups_years['ZZ4b2017'] + groups_years['ZH4b2017'] + groups_years['HH4b2017']
groups_years['2018'] = groups_years['ZZ4b2018'] + groups_years['ZH4b2018'] + groups_years['HH4b2018']
groups_years['stack'] = {}
groups_years['stack']['2016'] = {ZZ4b: groups_years['ZZ4b2016'],
                                 ZH4b: groups_years['ZH4b2016'],
                                 HH4b: groups_years['HH4b2016']}
groups_years['stack']['2017'] = {ZZ4b: groups_years['ZZ4b2017'],
                                 ZH4b: groups_years['ZH4b2017'],
                                 HH4b: groups_years['HH4b2017']}
groups_years['stack']['2018'] = {ZZ4b: groups_years['ZZ4b2018'],
                                 ZH4b: groups_years['ZH4b2018'],
                                 HH4b: groups_years['HH4b2018']}

eras = {'ZZ4b': ['2016_preVFP', '2016_postVFP', '2016', '2017', '2018'],
        'ZH4b': ['2016_preVFP', '2016_postVFP', '2017', '2017', '2018'],
        'HH4b': ['2016', '2017', '2018'],
        ''    : ['2016', '2017', '2018']}


PLOTTIME = 0

def plot_systematic(nominal, variations, colors=None, order=None, name='test.pdf', rtitle='Variation/Nominal', xtitle=None, rebin=None):
    tstart = time.time()
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7,7),
        gridspec_kw={"height_ratios": (2, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)


    ratios = []
    for i, variation in enumerate(variations):
        numer = variation['hist'].sum('process')
        denom = nominal.sum('process')
        n_sumw = numer.values()[()]
        d_sumw = denom.values()[()]
        ratio = np.divide(n_sumw, d_sumw, out=np.ones(len(n_sumw)), where=d_sumw!=0)
        ratios.append(ratio)


    if rebin is not None:
        nominal = nominal.rebin('x', rebin)
        for variation in variations: variation['hist'] = variation['hist'].rebin('x', rebin)

    ax.set_prop_cycle(cycler(color=colors))
    hist.plot1d(nominal, 
                overlay='process',
                ax=ax,
                clear=False,
                stack=True,
                line_opts  = None,
                fill_opts  = fill_opts,
                error_opts = stack_err_opts,
                order = order,
    )

    for variation in variations:
        data_err_opts['marker'] = variation['marker']
        hist.plot1d(variation['hist'], 
                    overlay='process',
                    ax=ax,
                    clear=False,
                    line_opts  = None,
                    fill_opts  = None,
                    error_opts = data_err_opts,
        )
    ax.set_xlabel(None)

    ratios = []
    for i, variation in enumerate(variations):
        data_err_opts['marker'] = variation['marker']
        numer = variation['hist'].sum('process')
        denom = nominal.sum('process')
        n_sumw = numer.values()[()]
        d_sumw = denom.values()[()]
        ratio = np.divide(n_sumw, d_sumw, out=np.ones(len(n_sumw)), where=d_sumw!=0)
        ratios.append(ratio)
        hist.plotratio(num=numer,
                       denom=denom,
                       ax=rax,
                       clear=False,
                       error_opts = data_err_opts,
                       denom_fill_opts = stack_err_opts,
                       guide_opts = guide_opts,
                       unc='num')
    ax.autoscale(axis='y')
    ax.set_ylim(0, None)
    rax.set_ylabel(rtitle)
    rax.set_ylim(0.5, 1.5)
    if xtitle:
        rax.set_xlabel(xtitle)

    plt.tight_layout()
    print(name)

    fig.savefig(name)
    fig.clear()
    plt.close(fig)

    global PLOTTIME
    PLOTTIME += time.time()-tstart
    return ratios



if __name__ == '__main__':
    totalTime = 0
    tstart = time.time()
    systematics = {}
    
    eos_base = 'root://cmseos.fnal.gov//store/user/pbryant/condor'
    nfs_base = '/uscms/home/bryantp/nobackup/ZZ4b'

    output_path = f'{nfs_base}'

    classifiers = ['SvB', 'SvB_MA']

    with open(f'{output_path}/hists.pkl', 'rb') as hfile:
        hists = pickle.load(hfile)

        mkpath(f'{output_path}/plots_systematics/trigWeight/')
        for sf in btagVariations[1::2]:
            var = sf.split('_')[-1]
            mkpath(f'{output_path}/plots_systematics/btagSF_{var}/')

        h = {}
        for cl in classifiers:
            h[cl] = {}
            for bb in ['zz','zh','hh']:
                h[cl][bb] = {}
                for tr in trigWeights:
                    h[cl][bb][tr] = hists['hists']['passPreSel']['fourTag']['SR'][f'trigWeight_{tr}'][f'{cl}_ps_{bb}']
                for sf in btagVariations:
                    h[cl][bb][sf] = hists['hists']['passPreSel']['fourTag']['SR'][f'btagSF_{sf}']    [f'{cl}_ps_{bb}']

            h[cl]['all'] = {}
            for var in trigWeights+btagVariations:
                h[cl]['all'][var] = deepcopy( h[cl]['zz'][var] )
                h[cl]['all'][var].axis('x').label = h[cl]['all'][var].axis('x').label.split('$|$')[0]
                for bb in ['zh','hh']:
                    h[cl]['all'][var] += h[cl][bb][var]

            for bb in ['zz','zh','hh']:
                for var in trigWeights+btagVariations:
                    h[cl][bb][var] = h[cl][bb][var].rebin('x', rebin[bb])


    #
    # Systematics split by year
    #
    for cl in classifiers:
        systematics[cl] = {}
        for sample, color, name in zip(['', ZZ4b, ZH4b, HH4b], [colors['all'], [ZZColor], [ZHColor], [HHColor]], ['', 'ZZ4b', 'ZH4b', 'HH4b']):
            process = name[:2] if name else ''
            for era in eras[name]:
                year = era.split('_')[0]
                channel = process.lower()+year[-1]

                # trigger weights
                nominal = h[cl]['all']['Bool'].group('dataset', hist.Cat('process', ''), {f'{sample} ({year})': groups_years[f'{name}{year}']} if name else groups_years['stack'][year])

                h_MC    = h[cl]['all']['MC']  .group('dataset', hist.Cat('process', ''), {  f'MC Emulation {era.replace("_"," ")}': groups_years[f'{name}{era}']})
                h_Data  = h[cl]['all']['Data'].group('dataset', hist.Cat('process', ''), {f'Data Emulation {era.replace("_"," ")}': groups_years[f'{name}{era}']})

                if era != year:
                    other_era = f'{year}_postVFP' if 'preVFP' in era else f'{year}_preVFP'
                    h_MC   += h[cl]['all']['Bool'].group('dataset', hist.Cat('process', ''), {  f'MC Emulation {era.replace("_"," ")}': groups_years[f'{name}{other_era}']})
                    h_Data += h[cl]['all']['Bool'].group('dataset', hist.Cat('process', ''), {f'Data Emulation {era.replace("_"," ")}': groups_years[f'{name}{other_era}']})

                variations = []
                variations.append({'hist': h_MC,
                                   'marker': markerMC})
                variations.append({'hist': h_Data,
                                   'marker': markerData})

                ratios = plot_systematic(nominal, variations, 
                                         colors=color,
                                         name=f'{output_path}/plots_systematics/trigWeight/{name}{era}_{cl}_ps_all.pdf', 
                                         rtitle = f'Emulation/Simulation',
                                         rebin = 5,
                                         #rebin=rebin['zz'],
                )

                if channel not in systematics[cl].keys():
                    systematics[cl][channel] = {}
                systematics[cl][channel]['trigger_emulationDown'] =   ratios[0] # Down direction moves template along Sim->Emu direction
                systematics[cl][channel]['trigger_emulationUp']   = 1/ratios[0] #   Up direction moves template along Emu->Sim direction


                # btagging
                nominal = h[cl]['all']['central'].group('dataset', hist.Cat('process', ''), {f'{sample} ({year})': groups_years[f'{name}{year}']} if name else groups_years['stack'][year])

                for down, up in downup:
                    var = down.split('_')[-1]

                    h_down  = h[cl]['all'][down]     .group('dataset', hist.Cat('process', ''), {f'{var} {era.replace("_"," ")} Down': groups_years[f'{name}{era}']})
                    h_up    = h[cl]['all'][up]       .group('dataset', hist.Cat('process', ''), {f'{var} {era.replace("_"," ")} Up'  : groups_years[f'{name}{era}']})

                    if era != year:
                        other_era = f'{year}_postVFP' if 'preVFP' in era else f'{year}_preVFP'
                        h_down += h[cl]['all']['central'].group('dataset', hist.Cat('process', ''), {f'{var} {era.replace("_"," ")} Down': groups_years[f'{name}{other_era}']})
                        h_up   += h[cl]['all']['central'].group('dataset', hist.Cat('process', ''), {f'{var} {era.replace("_"," ")} Up'  : groups_years[f'{name}{other_era}']})

                    variations = []
                    variations.append({'hist': h_down,
                                       'marker': markerMC})
                    variations.append({'hist': h_up,
                                       'marker': markerData})

                    ratios = plot_systematic(nominal, variations, 
                                             colors=color, 
                                             name=f'{output_path}/plots_systematics/btagSF_{var}/{name}{era}_{cl}_ps_all.pdf', 
                                             rtitle = f'{var}/central',
                                             rebin = 5,
                                             #rebin=rebin['zz'],
                    )
                    nuisance = var
                    if 'stat' in var: # decorrelated, need different nuissance parameter name
                        nuisance += '_'+era
                    print(channel,nuisance)
                    systematics[cl][channel][f'{nuisance}Down'] = ratios[0]
                    systematics[cl][channel][f'{nuisance}Up']   = ratios[1]


    with open(f'{output_path}/systematics.pkl', 'wb') as sfile:
        pickle.dump(systematics, sfile, protocol=2)


    for cl in classifiers:
        for bb in ['all',  'zz','zh','hh']:
            # 
            # Trigger weights
            # 
            nominal = h[cl][bb]['Bool'].group('dataset', hist.Cat('process', ''), groups)
            MC      = h[cl][bb]['MC']  .group('dataset', hist.Cat('process', ''), {  'MC Emulation': group_all})
            Data    = h[cl][bb]['Data'].group('dataset', hist.Cat('process', ''), {'Data Emulation': group_all})
            variations = []
            variations.append({'hist': MC,
                               'marker': markerMC})
            variations.append({'hist': Data,
                               'marker': markerData})

            plot_systematic(nominal, variations, 
                            colors=colors[bb], 
                            order=order[bb], 
                            name=f'{output_path}/plots_systematics/trigWeight/{cl}_ps_{bb}.pdf', 
                            rtitle = 'Emulation/Simulation',
                            rebin = 5 if bb=='all' else None,
                            #rebin=rebin[bb],
            )

            for sample, color, name in zip([ZZ4b, ZH4b, HH4b], [ZZColor, ZHColor, HHColor], ['ZZ4b', 'ZH4b', 'HH4b']):
                h_MC   = h[cl][bb]['MC']  .group('dataset', hist.Cat('process', ''), {  'MC Emulation': groups[sample]})
                h_Data = h[cl][bb]['Data'].group('dataset', hist.Cat('process', ''), {'Data Emulation': groups[sample]})

                variations = []
                variations.append({'hist': h_MC,
                                   'marker': markerMC})
                variations.append({'hist': h_Data,
                                   'marker': markerData})
                plot_systematic(nominal[sample], variations, 
                                colors=[color], 
                                name=f'{output_path}/plots_systematics/trigWeight/{name}_{cl}_ps_{bb}.pdf', 
                                rtitle = 'Emulation/Simulation',
                                rebin = 5 if bb=='all' else None,
                                #rebin=rebin[bb],
                )

            #
            # btagging
            #
            nominal = h[cl][bb]['central'].group('dataset', hist.Cat('process', ''), groups)

            for down, up in downup:
                var = down.split('_')[-1]
                h_down = h[cl][bb][down].group('dataset', hist.Cat('process', ''), {f'{var} Down': group_all})
                h_up   = h[cl][bb][up]  .group('dataset', hist.Cat('process', ''), {f'{var} Up'  : group_all})

                variations = []
                variations.append({'hist': h_down,
                                   'marker': markerMC})
                variations.append({'hist': h_up,
                                   'marker': markerData})

                plot_systematic(nominal, variations, 
                                colors=colors[bb], 
                                order=order[bb], 
                                name=f'{output_path}/plots_systematics/btagSF_{var}/{cl}_ps_{bb}.pdf', 
                                rtitle = f'{var}/central',
                                rebin = 5 if bb=='all' else None,
                                #rebin=rebin[bb],
                )

                for sample, color, name in zip([ZZ4b, ZH4b, HH4b], [ZZColor, ZHColor, HHColor], ['ZZ4b', 'ZH4b', 'HH4b']):
                    h_down = h[cl][bb][down].group('dataset', hist.Cat('process', ''), {f'{var} Down': groups[sample]})
                    h_up   = h[cl][bb][up]  .group('dataset', hist.Cat('process', ''), {f'{var} Up'  : groups[sample]})

                    variations = []
                    variations.append({'hist': h_down,
                                       'marker': markerMC})
                    variations.append({'hist': h_up,
                                       'marker': markerData})
                    plot_systematic(nominal[sample], variations, 
                                    colors=[color], 
                                    name=f'{output_path}/plots_systematics/btagSF_{var}/{name}_{cl}_ps_{bb}.pdf', 
                                    rtitle = f'{var}/central',
                                    rebin = 5 if bb=='all' else None,
                                    #rebin=rebin[bb],
                    )


    totalTime = time.time()-tstart
    print(totalTime)
    print(PLOTTIME)
    print(PLOTTIME/totalTime)
