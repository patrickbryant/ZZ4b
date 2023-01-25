from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import numpy as np
from scipy import interpolate
import sys

classifier = sys.argv[1]

channels = ['zz','zh','hh']
colors = {'zz': '#016601', 'zh': '#FF0000', 'hh': '#0400CC'}
sig, sig_stat, sig_fine = {}, {}, {}
lim, lim_stat, lim_fine = {}, {}, {}

initial_lumi = 132.8

for ch in channels:
    sig[ch], sig_stat[ch], sig_fine[ch] = {}, {}, {}
    lim[ch], lim_stat[ch], lim_fine[ch] = {}, {}, {}
    for future in ['', '_200', '_300', '_500', '_1000', '_2000', '_3000']:
        lumi = float(future.replace('_','')) if future else initial_lumi
        with open('combinePlots/%s/future/expected_significance_%s%s.txt'%(classifier, ch, future) , 'r') as f:
            for line in f:
                line = line.split()
                if not line: continue
                if line[0] == 'Significance:': 
                    sig[ch][lumi] = float(line[1])
                    break

        with open('combinePlots/%s/future/expected_stat_only_significance_%s%s.txt'%(classifier, ch, future) , 'r') as f:
            for line in f:
                line = line.split()
                if not line: continue
                if line[0] == 'Significance:': 
                    sig_stat[ch][lumi] = float(line[1])
                    break
                
        with open('combinePlots/%s/future/expected_stat_only_no_rebin_significance_%s%s.txt'%(classifier, ch, future) , 'r') as f:
            for line in f:
                line = line.split()
                if not line: continue
                if line[0] == 'Significance:': 
                    sig_fine[ch][lumi] = float(line[1])
                    break
                
        with open('combinePlots/%s/future/expected_limit_%s%s.txt'%(classifier, ch, future) , 'r') as f:
            for line in f:
                line = line.split()
                if len(line)<5: continue
                if line[0:2] == ['Expected','97.5%:']: 
                    lim[ch][lumi] = float(line[4])            
                    break

        with open('combinePlots/%s/future/expected_stat_only_limit_%s%s.txt'%(classifier, ch, future) , 'r') as f:
            for line in f:
                line = line.split()
                if len(line)<5: continue
                if line[0:2] == ['Expected','97.5%:']: 
                    lim_stat[ch][lumi] = float(line[4])            
                    break

        with open('combinePlots/%s/future/expected_stat_only_no_rebin_limit_%s%s.txt'%(classifier, ch, future) , 'r') as f:
            for line in f:
                line = line.split()
                if len(line)<5: continue
                if line[0:2] == ['Expected','97.5%:']: 
                    lim_fine[ch][lumi] = float(line[4])            
                    break


    print(ch,sig_fine[ch])
    print(ch,lim_fine[ch])


def plotProjection(val, val_stat, val_fine, title, lines=[], ymax=5, scaling=0.5, logx=False, logy=False, spline_k=3, do_spline=True):
    xlim = [132.8, 3000]
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7,7),
        gridspec_kw={"height_ratios": (2, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)
    rlim = [0,1] if 'imit' in title else [1,2.5]
    rax.set_ylim(rlim[0], rlim[1])
    rax.set_ylabel(r'Ratio to Expected')
    rax.plot(xlim, [1,1], color='k', alpha=1.0, linestyle='-', linewidth=1.0)

    # fig, (ax) = plt.subplots(nrows=1)
    ax.set_xlim(xlim[0], xlim[1])
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(0.5,ymax)        
    else:
        ax.set_ylim(0,ymax)

    for line in lines:
        ax.plot(xlim, [line,line], color='k', alpha=1.0, linestyle='--', linewidth=0.5)
    #ax.set_xticks(x)
    ax.set_title('Projected '+title.split()[0])

    for ch in channels:
        sampled = np.array(sorted(val[ch].keys()))
        values = np.array([val[ch][lumi] for lumi in sampled])

        x = sampled
        y = values        
        if do_spline:
            spline = interpolate.make_interp_spline(sampled, values, bc_type='natural' if spline_k>2 else None, k=spline_k)
            x = np.linspace(sampled[0], sampled[-1], num=100, endpoint=True)
            y = spline(x)
            
        ax.plot(x, y, label=ch.upper(), color=colors[ch], linewidth=1)
        expected = y

        y = np.array( [(lumi/initial_lumi)**scaling*val[ch][initial_lumi] for lumi in x] )
        ax.plot(x, y, color=colors[ch], linewidth=1, linestyle='--')
        naive = y

        # values = np.array([val_stat[ch][lumi] for lumi in sampled])
        # y = values
        # if do_spline:
        #     spline = interpolate.make_interp_spline(sampled, values, bc_type='natural' if spline_k>2 else None, k=spline_k)
        #     y = spline(x)
        # ax.plot(x, y, color=colors[ch], linewidth=1, linestyle='-.')
        # stat=y

        values = np.array([val_fine[ch][lumi] for lumi in sampled])
        y = values
        if do_spline:
            spline = interpolate.make_interp_spline(sampled, values, bc_type='natural' if spline_k>2 else None, k=spline_k)
            y = spline(x)
        ax.plot(x, y, color=colors[ch], linewidth=1, linestyle='dotted')
        fine=y

        rax.plot(x,naive/expected, color=colors[ch], linewidth=1, linestyle='--')
        # rax.plot(x, stat/expected, color=colors[ch], linewidth=1, linestyle='-.')
        rax.plot(x, fine/expected, color=colors[ch], linewidth=1, linestyle='dotted')

    ax.plot([],[], label='Expected Scaling', color='k', alpha=0.5, linewidth=1)
    ax.plot([],[], label=r'Naive $\sqrt{L}$ Scaling' if scaling>0 else r'Naive $1/\sqrt{L}$ Scaling', color='k', alpha=0.5, linewidth=1, linestyle='--')
    # ax.plot([],[], label=r'Stat. Only', color='k', alpha=0.5, linewidth=1, linestyle='-.')
    ax.plot([],[], label=r'No Multijet Uncertainty', color='k', alpha=0.5, linewidth=1, linestyle='dotted')

    # ax.set_xlabel(None)
    rax.set_xlabel(r'Integrated Luminosity [fb$^{-1}$]')
    ax.set_ylabel(title)
    ax.legend(loc='best', fontsize='small')
    
    name = 'combinePlots/%s/future/projected_%s.pdf'%(classifier, title.split()[0].lower())
    print('fig.savefig( '+name+' )')
    plt.tight_layout()
    fig.savefig( name )
    plt.close(fig)

plotProjection(sig, sig_stat, sig_fine, 'Significance', lines=[3], do_spline=False)
plotProjection(lim, lim_stat, lim_fine, r'Limit ($95\%$ CL)', lines=[1], ymax=20, scaling=-0.5, logx=True, logy=True, do_spline=False)
