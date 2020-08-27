#from bTagSyst import getBTagSFName
from ROOT import TFile, TH1F, TF1
import sys
from os import path
from optparse import OptionParser

parser = OptionParser()

parser.add_option('-i', '--in',       dest="inFile",  default="")
parser.add_option('-o', '--out',      dest="outFile", default="")
parser.add_option('-n', '--name',     dest="histName",default="")
#parser.add_option('--shape',          dest='shape',   default='')
parser.add_option('--var',            dest='var',     default='')
parser.add_option('--channel',        dest='channel', default='')
parser.add_option('--tag',            dest='tag',     default='')
parser.add_option('-c', '--cut',      dest="cut",     default="", help="")
parser.add_option('--rebin',          dest="rebin",   default="1", help="")
parser.add_option('-f', '--function', dest="function",default="", help="specified funtion will be used to scale the histogram along x dimension")
parser.add_option('-r', '--region',   dest="region",  default="", help="")
parser.add_option('-b', '--bTagSyst', dest="bTagSyst",default=False,action="store_true", help="")
parser.add_option('-j', '--jetSyst',  dest="jetSyst", default=False,action="store_true", help="")
parser.add_option('-t', '--trigSyst', dest="trigSyst",default=False,action="store_true", help="")

o, a = parser.parse_args()

NPs = []
# NPs = [["Resolved_JET_GroupedNP_1__1up","Resolved_JET_GroupedNP_1__1down"],
#        ["Resolved_JET_GroupedNP_2__1up","Resolved_JET_GroupedNP_2__1down"],
#        ["Resolved_JET_GroupedNP_3__1up","Resolved_JET_GroupedNP_3__1down"],
#        ["Resolved_JET_EtaIntercalibration_NonClosure__1up","Resolved_JET_EtaIntercalibration_NonClosure__1down"],
#        ["Resolved_JET_JER_SINGLE_NP__1up"]]

#regions = [o.region]

def get(rootFile, path):
    obj = rootFile.Get(path)
    if str(obj) == "<ROOT.TObject object at 0x(nil)>": 
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


f = TFile(o.inFile, "READ")
print "input file:", f
f_syst = {}
if o.jetSyst:
    for syst in NPs:
        for direction in syst:
            f_syst[direction] = TFile(o.inFile.replace("/hists.root","_"+direction+"/hists.root"), "READ")
            print "input file:",f_syst[direction]


if path.exists(o.outFile):
    out = TFile.Open(o.outFile, "UPDATE")
else:
    out = TFile.Open(o.outFile, "RECREATE")




def getAndStore(var,channel,histName,rebin=1,suffix='',jetSyst=False, function=''):
    #h={}
    #for region in regions:
    h = get(f, o.cut+"/"+o.tag+"Tag/mainView/"+o.region+"/"+var)
    h.SetName(histName+suffix)
    h.Rebin(int(rebin))

    tf1=None
    if function:
        xmin, xmax = h.GetXaxis().GetXmax(), h.GetXaxis().GetXmin()
        tf1 = TF1('function', function, xmin, xmax)
        for bin in range(1,h.GetNbinsX()+1):
            c, e, w = h.GetBinContent(bin), h.GetBinError(bin), h.GetBinWidth(bin)
            l, u = h.GetXaxis().GetBinLowEdge(bin), h.GetXaxis().GetBinUpEdge(bin) #limits of integration
            I = tf1.Integral(l,u)/w
            h.SetBinContent(bin, c*I)
            h.SetBinError  (bin, e*I)

    makePositive(h)

    out.cd()
    #for region in regions:
    print 'get dirctory',channel
    directory = out.Get(channel)
    print directory
    if 'nil' in str(directory):
        print 'mkdir',channel
        out.mkdir(channel)
    print 'cd',channel
    out.cd(channel)
    #out.Append(h)
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


getAndStore(o.var,o.channel,o.histName,o.rebin,'',jetSyst=o.jetSyst, function=o.function)


out.Write()
out.Close()
