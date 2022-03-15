#from bTagSyst import getBTagSFName
from ROOT import TFile, TH1F, TF1
import numpy as np
import sys
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
parser.add_option('-f', '--function', dest="function",default="", help="specified funtion will be used to scale the histogram along x dimension")
parser.add_option('-a', '--array',    dest="array",default="", help="specified array will be used to scale the histogram along x dimension")
parser.add_option('-r', '--region',   dest="region",  default="", help="")
parser.add_option('-b', '--bTagSyst', dest="bTagSyst",default=False,action="store_true", help="")
parser.add_option('-j', '--jetSyst',  dest="jetSyst", default=False,action="store_true", help="")
parser.add_option('-t', '--trigSyst', dest="trigSyst",default=False,action="store_true", help="")
parser.add_option(      '--debug',    dest="debug",   default=False,action="store_true", help="")
parser.add_option(      '--addHist',  dest="addHist", default=''   , help="path.root,path/to/hist,weight")

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
        print 
        print "ERROR: Object not found -", rootFile, path
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
        hist.SetBinError(bin, err if y > 0 else zero)


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


print "input file:", o.inFile
f = TFile(o.inFile, "READ")
f_syst = {}
if o.jetSyst:
    for syst in NPs:
        for direction in syst:
            systFileName = o.inFile.replace("/hists.root","_"+direction+"/hists.root")
            print "input file:",systFileName
            f_syst[direction] = TFile(systFileName, "READ")


if path.exists(o.outFile):
    if o.debug: print 'UPDATE',o.outFile
    out = TFile.Open(o.outFile, "UPDATE")
else:
    if o.debug: print 'RECREATE',o.outFile
    out = TFile.Open(o.outFile, "RECREATE")




def getAndStore(var,channel,histName,suffix='',jetSyst=False, function='', array=''):
    #h={}
    #for region in regions:
    if not o.TDirectory:
        h = get(f, o.cut+"/"+o.tag+"Tag/mainView/"+o.region+"/"+var)
    else:
        h = get(f, o.TDirectory+"/"+var)
    if rebin:
        if type(rebin) is list:
            h, _ = do_variable_rebinning(h, rebin, scaleByBinWidth=False)
        else:
            h.Rebin(int(o.rebin))

    histName = histName.replace('multijet','mj')
    histName = histName.replace('ttbar',   'tt')

    h.SetName(histName+suffix)

    if o.errorScale is not None:
        for bin in range(1,h.GetNbinsX()+1):
            h.SetBinError(bin, h.GetBinError(bin)*float(o.errorScale))

    if function or array:
        xmin, xmax = h.GetXaxis().GetXmax(), h.GetXaxis().GetXmin()
        tf1, arr=None, None
        if function:
            tf1 = TF1('function', function, xmin, xmax)
        if array:
            arr = np.array(eval(array))

        for bin in range(1,h.GetNbinsX()+1):
            s = 1
            if function:
                l, u = h.GetXaxis().GetBinLowEdge(bin), h.GetXaxis().GetBinUpEdge(bin) #limits of integration
                w = h.GetBinWidth(bin) # divide intregral by bin width to get average of function over bin
                s = tf1.Integral(l,u)/w
            if array:
                s = arr[bin-1] # assume array at index i=bin-1 corresponds to bin 

            c, e = h.GetBinContent(bin), h.GetBinError(bin)
            h.SetBinContent(bin, c*s)
            h.SetBinError  (bin, e*s)

    makePositive(h)

    if o.scale is not None:
        h.Scale(float(o.scale))
    if addHist is not None:
        h.Add(addHist)

    out.cd()
    #for region in regions:
    channel = channel.replace('201','')
    try:
        directory = out.Get(channel)
        directory.IsZombie()
        print 'got dirctory',channel
    except ReferenceError:        
        print 'make dirctory',channel
        directory = out.mkdir(channel)

    print 'cd',channel
    out.cd(channel)

    print 'write',h
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

if o.debug: print 'getAndStore()'
getAndStore(o.var,o.channel,o.histName,'',jetSyst=o.jetSyst, function=o.function, array=o.array)
if o.debug: print 'got and stored'


out.Write()
out.Close()
