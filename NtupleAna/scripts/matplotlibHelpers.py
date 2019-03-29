#plotting macros
import numpy as np
import matplotlib.pyplot as plt

def binData(data, bins, weights=None, norm=None):
    data = np.array(data)
    n,  bins = np.histogram(data, bins=bins, weights=weights)

    #compute weights**2
    weights2=weights
    if type(weights) != type(None):
        weights  = np.array(weights)
        weights2 = weights**2

    #histogram weights**2 to get sum of squares of weights per bin
    w2, bins = np.histogram(data, bins=bins, weights=weights2)

    #yErr is sqrt of sum of squares of weights in each bin
    e = w2**0.5

    #normalize bins and errors
    if norm:
        nSum = n.sum()
        n = n/nSum*float(norm)
        e = e/nSum*float(norm)

    return n, e

def getRatio(ns, errs):
    r=[]
    rErr=[]
    for i in list(range(len(ns[0]))):
        n=ns[0][i]
        d=ns[1][i]
        ratio = n/d if d else 0
        r.append(ratio)
        dn = errs[0][i]
        dd = errs[1][i]
        dr = ( (dn/d)**2 + (dd*n/d**2)**2 )**0.5 if d else 0
        rErr.append(dr)

    r, rErr = np.array(r), np.array(rErr)
    return r, rErr

def plot(data,bins,xlabel,ylabel,norm=None,weights=[None,None],samples=['',''],drawStyle='steps-mid',alpha=0.3,ratio=False):
    bins = np.array(bins)
    ns    = []
    yErrs = []
    for i in list(range(len(data))):
        n, yErr = binData(data[i], bins, weights=weights[i], norm=norm)
        ns   .append(n)
        yErrs.append(yErr)
        
    binCenters=0.5*(bins[1:] + bins[:-1])

    if ratio:
        fig, (sub1, sub2) = plt.subplots(nrows=2, sharex=True, gridspec_kw = {'height_ratios':[2, 1]})
        plt.subplots_adjust(hspace = 0.05, left=0.11, top=0.95, right=0.95)
    else:
        fig, (sub1) = plt.subplots(nrows=1)
    sub1.errorbar(binCenters,
                  ns[0],
                  yerr=yErrs[0],
                  drawstyle=drawStyle,
                  label=samples[0],
    )
    sub1.errorbar(binCenters,
                  ns[1],
                  yerr=yErrs[1],
                  drawstyle=drawStyle,
                  label=samples[1],
    )
    sub1.legend()
    sub1.set_ylabel(ylabel)
    plt.xlim([bins[0],bins[-1]])

    if ratio:
        #sub1.set_xlabel('')
        #sub1.set_xticklabels([])
        r, rErr = getRatio(ns, yErrs)
        sub2.errorbar(binCenters,
                      r,
                      yerr=rErr,
                      drawstyle='steps-mid',
                      color='k',
        )
        plt.ylim([0,2])
        plt.xlim([bins[0],bins[-1]])
        plt.plot([bins[0], bins[-1]], [1, 1], color='k', linestyle=':', linewidth=1)
        sub2.set_xlabel(xlabel)
        sub2.set_ylabel(samples[0]+' / '+samples[1])

    else:
        sub1.set_xlabel(xlabel)

    return fig



class hist:
    def __init__(self, binContents, binErrors, bins):
        self.binContents = binContents
        self.binErrors = binErrors
        self.bins = bins
        self.nBins = len(bins)

    def findBin(self, x):
        for i in list(range(self.nBins-1)):
            if x >= self.bins[i] and x < self.bins[i+1]:
                return i

    def getBinContent(self, i):
        return self.binContents[i]

    def getBinError(self, i):
        return self.binError[i]

    def findBinContent(self, x):
        return self.getBinContent(self.findBin(x))

    def findBinError(self, x):
        return self.getBinError(self.findBin(x))
