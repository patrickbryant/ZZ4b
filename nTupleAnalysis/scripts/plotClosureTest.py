import sys
sys.path.insert(0, 'nTupleAnalysis/python/') #https://github.com/patrickbryant/nTupleAnalysis
from commandLineHelpers import *
import math

import ROOT 
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)


import optparse

parser = optparse.OptionParser()
parser.add_option('-e',            action="store_true", dest="execute",        default=False, help="Execute commands. Default is to just print them")
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4", help="Year or comma separated list of subsamples")
parser.add_option('-o',                                 dest="outDir"        )

o, a = parser.parse_args()

subSamples = o.subSamples.split(",")

mkdir(o.outDir)


def getBkgModel(hName,ttFile,d3File,rebin=1):
    ttHist = ttFile.Get(hName).Clone()
    ttHist.Rebin(rebin)
    d3Hist = d3File.Get(hName.replace("fourTag","threeTag")).Clone()
    d3Hist.Rebin(rebin)
    d3Hist.Add(ttHist)
    d3Hist.SetName(ttHist.GetName()+"_bkg")
    return d3Hist

def getRatio(hName,d4Hist,d3Hist,ttHist,rebin=1):
    hBkg  = getBkgModel(hName, ttHist, d3Hist,rebin=rebin)
    hData = d4Hist.Get(hName).Clone()
    hData.Rebin(rebin)
    hData.Divide(hBkg)
    hData.SetName(hData.GetName()+"_ratio")
    hData.SetMarkerStyle(20)
    hData.SetMarkerSize(0.7)
    hData.GetYaxis().SetRangeUser(0.7,1.3)
    hData.GetYaxis().SetTitle("Ratio (Data/Bkg)")
    return hData


def getPull(ratioIn):
    pull = ratioIn.Clone()
    pull.SetName(ratioIn.GetName()+"_pull")    
    pull.GetYaxis().SetRangeUser(-5,5)
    pull.GetYaxis().SetTitle("Pull (Data-Bkg)/#sigma")
    for iBin in range(ratioIn.GetNbinsX()+1):
        cv = ratioIn.GetBinContent(iBin)
        err = ratioIn.GetBinError(iBin)
        if cv == 0.0 and err == 0.0:
            pull.SetBinContent(iBin,0)
            pull.SetBinError(iBin,0)
        else:
            pval = (cv-1.0)/err if err else 1.
            pull.SetBinContent(iBin,pval)
            pull.SetBinError(iBin,0)

    return pull

def getMeanStd(hists,nSigma):
    meanStd = hists[0].Clone()
    meanStd.SetFillColor(ROOT.kYellow)

    N = len(hists)

    for iBin in range(meanStd.GetNbinsX()+1):
        sumY  = 0 
        sumY2 = 0

        for hItr, h in enumerate(hists):
            thisPull = h.GetBinContent(iBin)
            sumY  += thisPull
            sumY2 += (thisPull*thisPull)
            

        mean = sumY / N
        var  = sumY2 / N - mean*mean
        std = math.sqrt(var)

        meanStd.SetBinContent(iBin,mean)
        meanStd.SetBinError(iBin,nSigma*std)

    return meanStd



def lineAt(yVal,xMin, xMax):
    one = ROOT.TF1("one",str(yVal),xMin,xMax)
    one.SetLineStyle(ROOT.kDashed)
    one.SetLineColor(ROOT.kBlack)
    return one


def drawAll(name,histList,yLine=None,drawOpts="",underLays=None):
    for hItr, h in enumerate(histList):

        if hItr:
            h.Draw(drawOpts+" same")
        else:
            h.Draw(drawOpts)
            
            if not underLays is None:
                for u in underLays:
                    u.Draw("E2 same")
                h.Draw("same AXIS")

            if not yLine is None:
                one = lineAt(yLine,h.GetXaxis().GetXmin(),h.GetXaxis().GetXmax())
                one.Draw("same")
            h.Draw(drawOpts+" same")

    can.SaveAs(o.outDir+"/"+name+".pdf")

    

can = ROOT.TCanvas()
ttFiles  = []
d3Files  = []
d4Files  = []

colors = [ROOT.kBlack,ROOT.kGreen,ROOT.kBlue,ROOT.kRed,ROOT.kOrange]


#
# Read Files
#
for sItr, s in enumerate(subSamples):

    tt = "/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b/TTRunII/hists_4b_wFVT_3bMix4b_v"+s+".root"
    d3 = "/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b/dataRunII/hists_3b_wJCM_3bMix4b_v"+s+"_wFVT_3bMix4b_v"+s+".root "
    d4 = "/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b/dataRunII/hists_4b_wFVT_3bMix4b_v"+s+".root"

    ttFiles.append(ROOT.TFile(tt,"READ"))
    d3Files.append(ROOT.TFile(d3,"READ"))
    d4Files.append(ROOT.TFile(d4,"READ"))





def makePlots(hName,outName,rebin):
    ratios = []
    pulls  = []

    for sItr, s in enumerate(subSamples):

        ratios.append(getRatio(hName, d4Files[sItr], d3Files[sItr], ttFiles[sItr], rebin=rebin))
        ratios[sItr].SetLineColor(colors[sItr])
        ratios[sItr].SetMarkerColor(colors[sItr])


        pulls.append(getPull(ratios[sItr]))
        #pulls[sItr].SetLineColor(colors[sItr])
        #pulls[sItr].SetMarkerColor(colors[sItr])


        can.cd()
        ratios[sItr].Draw("")
        one = lineAt(1,ratios[sItr].GetXaxis().GetXmin(),ratios[sItr].GetXaxis().GetXmax())
        one.Draw("same")
        can.SaveAs(o.outDir+"/"+outName+"_ratio_v"+s+".pdf")


        can.cd()
        pulls[sItr].Draw("PE")
        zero = lineAt(0,ratios[sItr].GetXaxis().GetXmin(),ratios[sItr].GetXaxis().GetXmax())
        zero.Draw("same")
        can.SaveAs(o.outDir+"/"+outName+"_pull_v"+s+".pdf")



    can.cd()
    drawAll(outName+"_ratio",ratios,yLine=1)

    pullMean1sigma = getMeanStd(pulls,1)

    pullMean2sigma = getMeanStd(pulls,2)
    pullMean2sigma.SetFillColor(ROOT.kGreen)

    drawAll(outName+"_pull", pulls ,yLine=0, drawOpts="PE",underLays=[pullMean2sigma,pullMean1sigma])



hPath = "passXWt/fourTag/mainView"

for r in ["SR","SB","CR"]:

    variables = [
        ("SvB_ps_zz", 2),
        ("SvB_ps",    2),
        ("SvB_ps_zh", 2),
        ("SvB_q_score", 2),
        ("FvT", 2),
        ]

    for v in variables:
        makePlots(hName = hPath+"/"+r+"/"+v[0],  outName=r+"_"+v[0] ,rebin=v[1])
          
