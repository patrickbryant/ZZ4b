from __future__ import print_function
#from bTagSyst import getBTagSFName
from ROOT import TFile, TH1F, TF1
import numpy as np
import sys
import pickle
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
from PlotTools import do_variable_rebinning
from os import path
from optparse import OptionParser

parser = OptionParser()

parser.add_option('-i', '--in',       dest="inFile",  default="")
parser.add_option('-o', '--out',      dest="outFile", default="")
parser.add_option('-n', '--name',     dest="histName",default="")
#parser.add_option('--shape',          dest='shape',   default='')
parser.add_option('--var',            dest='var',     default='')
parser.add_option('--TDirectory',     dest='TDirectory', default='')
parser.add_option('--channel',        dest='channel', default='')
parser.add_option('--tag',            dest='tag',     default='')
parser.add_option('-c', '--cut',      dest="cut",     default="", help="")
parser.add_option('--rebin',          dest="rebin",   default='', help="")
parser.add_option('--scale',          dest="scale",   default=None, help="Scale factor for hist")
parser.add_option('--errorScale',     dest="errorScale", default="1.0", help="Scale factor for hist stat error")
# parser.add_option('-f', '--function', dest="function",default="", help="specified funtion will be used to scale the histogram along x dimension")
parser.add_option('-a', '--array',    dest="array",default="", help="specified array will be used to scale the histogram along x dimension")
parser.add_option('-r', '--region',   dest="region",  default="", help="")
parser.add_option('-b', '--bTagSyst', dest="bTagSyst",default=False,action="store_true", help="")
parser.add_option('-j', '--jetSyst',  dest="jetSyst", default=False,action="store_true", help="")
parser.add_option('-t', '--trigSyst', dest="trigSyst",default=False,action="store_true", help="")
parser.add_option(      '--debug',    dest="debug",   default=False,action="store_true", help="")
parser.add_option(      '--addHist',  dest="addHist", default=''   , help="path.root,path/to/hist,weight")
parser.add_option(      '--systematics',  dest="systematics", default=''   , help=".pkl file with arrays for all systematic variations")
parser.add_option(      '--beforeRebin',  dest="beforeRebin", default=False, action='store_true'   , help="apply variations from pkl file before rebining rather than after")
# parser.add_option(      '--systematics_sub_dict',  dest="systematics_sub_dict", default=''   , help="path in nested dictionary to dictionary of variations we want. use / separators")

o, a = parser.parse_args()

rebin = eval(o.rebin)

NPs = []
# NPs = [["Resolved_JET_GroupedNP_1__1up","Resolved_JET_GroupedNP_1__1down"],
#        ["Resolved_JET_GroupedNP_2__1up","Resolved_JET_GroupedNP_2__1down"],
#        ["Resolved_JET_GroupedNP_3__1up","Resolved_JET_GroupedNP_3__1down"],
#        ["Resolved_JET_EtaIntercalibration_NonClosure__1up","Resolved_JET_EtaIntercalibration_NonClosure__1down"],
#        ["Resolved_JET_JER_SINGLE_NP__1up"]]

#regions = [o.region]

def get(rootFile, path):
    obj = rootFile.Get(path)
    if obj == None:
        rootFile.ls()
        print()
        print("ERROR: Object not found -", rootFile, path)
        sys.exit()

    else: return obj
 
#remove negative bins
zero = 0.00000001
def makePositive(hist):
    for bin in range(1,hist.GetNbinsX()+1):
        x   = hist.GetXaxis().GetBinCenter(bin)
        y   = hist.GetBinContent(bin)
        err = hist.GetBinError(bin)
        hist.SetBinContent(bin, y if y > 0 else zero)
        hist.SetBinError(bin, err if y > 0 else zero/10)


addHist = None
if o.addHist:
    addHistInfo = o.addHist.split(',')
    f = TFile(addHistInfo[0], 'READ')
    addHist = f.Get(addHistInfo[1])
    if type(rebin) is list:
        addHist, _ = do_variable_rebinning(addHist, rebin, scaleByBinWidth=False)
    else:
        addHist.Rebin(rebin)
    addHist.Scale(float(addHistInfo[2]))
    addHist.SetName('addHist')
    addHist.SetDirectory(0)
    f.Close()


print("input file:", o.inFile)
f = TFile(o.inFile, "READ")
f_syst = {}
if o.jetSyst:
    for syst in NPs:
        for direction in syst:
            systFileName = o.inFile.replace("/hists.root","_"+direction+"/hists.root")
            print("input file:",systFileName)
            f_syst[direction] = TFile(systFileName, "READ")


if path.exists(o.outFile):
    if o.debug: print('UPDATE',o.outFile)
    out = TFile.Open(o.outFile, "UPDATE")
else:
    if o.debug: print('RECREATE',o.outFile)
    out = TFile.Open(o.outFile, "RECREATE")


if o.systematics:
    print('Read systematics pkl file:',o.systematics)
    with open(o.systematics, 'rb') as sfile:
        systematics = pickle.load(sfile)
        # sub_dict = o.systematics_sub_dict.split('/')
        # cl, process_year = sub_dict[0], sub_dict[1]
        # if o.systematics_sub_dict:
        #     for sub in o.systematics_sub_dict.split('/'):
        #         systematics = systematics[sub]
    


def scaleByArray(h, arr):
    if len(arr)!=h.GetNbinsX(): 
        print('ERROR: scaleByArray len(arr)!=h.GetNbinsX()')
        print(len(arr), h.GetNbinsX())
        return

    for bin in range(1,h.GetNbinsX()+1):
        s = 1
        # if function:
        #     l, u = h.GetXaxis().GetBinLowEdge(bin), h.GetXaxis().GetBinUpEdge(bin) #limits of integration
        #     w = h.GetBinWidth(bin) # divide intregral by bin width to get average of function over bin
        #     s = tf1.Integral(l,u)/w
        # try:
        s = arr[bin-1] # assume array at index i=bin-1 corresponds to bin 
        # except IndexError:
        #     print(len(arr),h.GetNbinsX())
            

        c, e = h.GetBinContent(bin), h.GetBinError(bin)
        h.SetBinContent(bin, c*s)
        h.SetBinError  (bin, e*s)


def getAndStore(var,channel,histName,suffix='',jetSyst=False, array=''):
    channel = channel.replace('201','')
    classifier = 'SvB_MA' if '_MA' in var else 'SvB' 
    sample = histName.lower()+channel[-1]
    #h={}
    #for region in regions:
    if not o.TDirectory:
        h = get(f, o.cut+"/"+o.tag+"Tag/mainView/"+o.region+"/"+var)
    else:
        h = get(f, o.TDirectory+"/"+var)

    histName = histName.replace('multijet','mj')
    histName = histName.replace('ttbar',   'tt')
    h.SetName(histName+suffix)

    h_syst = {}
    if o.systematics:

        # HACK to get pilup and prefire weights from coffea: replace fwlite histogram contents with coffea result
        if 'systematics.pkl' in o.systematics:
            print('>>>> HACK: replace fwlite histogram contents with coffea analysis result from',o.systematics, classifier, channel, histName)
            nominal = systematics[classifier][channel][histName]
            # print(nominal['contents'])
            og_contents = []
            for bin in range(1,h.GetNbinsX()+1):
                og_contents.append(h.GetBinContent(bin))
                h.SetBinContent(bin, nominal['contents'][bin-1])
                h.SetBinError  (bin, nominal[  'errors'][bin-1])
            og_contents = np.array(og_contents)
            coffea_to_root_norm = og_contents.sum() / nominal['contents'].sum()
            print('Normalized with root yield. root/coffea:',coffea_to_root_norm)
            h.Scale(coffea_to_root_norm)
            # print(nominal['contents']/og_contents)

        systematics_dict = systematics
        if 'systematics.pkl' in o.systematics:
            systematics_dict = systematics[classifier][sample]

        for nuisance, ratio in systematics_dict.items(): #[classifier][sample].items():
            if nuisance in ['ZZ', 'ZH', 'HH']: continue
            h_syst[nuisance] = h.Clone(histName+'_'+nuisance)
            if o.beforeRebin: 
                if type(ratio) is dict:
                    numer, denom = ratio['n_value'], ratio['d_value']
                    arr = np.divide(numer, denom, out=np.ones(len(numer)), where=denom!=0)
                else:
                    arr = ratio
                scaleByArray(h_syst[nuisance], arr)
            h_syst[nuisance].Rebin(int(o.rebin))
            if not o.beforeRebin:
                if type(ratio) is dict:
                    numer, denom = ratio['n_value'], ratio['d_value']
                    numer, denom = numer.reshape(rebin, len(numer)/rebin).sum(0), denom.reshape(rebin, len(denom)/rebin).sum(0)
                    arr = np.divide(numer, denom, out=np.ones(len(numer)), where=denom!=0)
                else:
                    arr = ratio
                scaleByArray(h_syst[nuisance], arr)
            makePositive(h_syst[nuisance])

    if rebin:
        if type(rebin) is list:
            h, _ = do_variable_rebinning(h, rebin, scaleByBinWidth=False)
        else:
            h.Rebin(rebin)

    if o.errorScale is not None:
        for bin in range(1,h.GetNbinsX()+1):
            h.SetBinError(bin, h.GetBinError(bin)*float(o.errorScale))

    if array:
        arr = np.array(eval(array))
        scaleByArray(h, np.array(eval(array)))

    makePositive(h)

    if o.scale is not None:
        h.Scale(float(o.scale))
    if addHist is not None:
        h.Add(addHist)

    out.cd()
    #for region in regions:
    try:
        directory = out.Get(channel)
        directory.IsZombie()
        # print('got dirctory',channel)
    except ReferenceError:        
        # print('make dirctory',channel)
        directory = out.mkdir(channel)

    # print('cd',channel)
    out.cd(channel)

    # print('write',h)
    h.Write()

    for nuisance, h in h_syst.items():
        out.cd(channel)
        # print('write',h,nuisance)
        h.Write()

    # if jetSyst:
    #     h_syst = {}
    #     for region in regions:
    #         h_syst[region] = {}
    #         for syst in NPs:
    #             h_syst[region][syst[0]] = get(f_syst[syst[0]], o.cut+"_"+o.tag+"Tag_"+region+"/"+var)
    #             h_syst[region][syst[0]].SetName((histName+"_"+syst[0]).replace("_hh","_"+names[region])+suffix)

    #             if len(syst) == 2: #has up and down variation
    #                 h_syst[region][syst[1]] = get(f_syst[syst[1]], o.cut+"_"+o.tag+"Tag_"+region+"/"+var)
    #                 h_syst[region][syst[1]].SetName((histName+"_"+syst[1]).replace("_hh","_"+names[region])+suffix)

    #                 makePositive(h_syst[region][syst[0]])
    #                 makePositive(h_syst[region][syst[1]])
    #                 out.Append(h_syst[region][syst[0]])
    #                 out.Append(h_syst[region][syst[1]])

    #             else: #one sided systematic
    #                 makePositive(h_syst[region][syst[0]])
    #                 out.Append(h_syst[region][syst[0]])

if o.debug: print('getAndStore()')
getAndStore(o.var,o.channel,o.histName,'',jetSyst=o.jetSyst, array=o.array)
if o.debug: print('got and stored')


#out.Write()
out.Close()
