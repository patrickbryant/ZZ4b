
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
parser.add_option('-s',                                 dest="subSamples",      default="0,1,2,3,4,5,6", help="Year or comma separated list of subsamples")
parser.add_option('-o',                                 dest="outDir"        )
parser.add_option('-m',                                 dest="mixedName"        )


o, a = parser.parse_args()

subSamples = o.subSamples.split(",")
mixedName  = o.mixedName

mkdir(o.outDir)




def getBkgModel(hName,ttFile,d3File,rebin=1):
    ttHist = ttFile.Get(hName).Clone()
    ttHist.Rebin(rebin)
    d3Hist = d3File.Get(hName.replace("fourTag","threeTag")).Clone()
    d3Hist.Rebin(rebin)
    d3Hist.Add(ttHist)
    d3Hist.SetName(ttHist.GetName()+"_bkg")
    return d3Hist



def getPullNoTTBar(hName,d4File,d3File,ttFile,rebin=1):

    ttHist = ttFile.Get(hName).Clone()
    ttHist.Rebin(rebin)
    
    hDataNoTT = d4File.Get(hName).Clone()
    hDataNoTT.Rebin(rebin)
    hDataNoTT.Add(ttHist,-1)

    # Normalize
    hBkgNoTT = d3File.Get(hName.replace("fourTag","threeTag")).Clone()
    hBkgNoTT.Rebin(rebin)
    hBkgNoTT.Scale(hDataNoTT.Integral()/hBkgNoTT.Integral())

    pull = hDataNoTT.Clone()
    pull.SetMarkerStyle(20)
    pull.SetMarkerSize(0.7)
    pull.SetName(hDataNoTT.GetName()+"_pull")    
    pull.GetYaxis().SetRangeUser(-5,5)
    pull.GetYaxis().SetTitle("Pull (Data-Bkg)/#sigma")

    for iBin in range(pull.GetNbinsX()+1):
        
        dataNoTT    = hDataNoTT.GetBinContent(iBin)
        bkgNoTT     = hBkgNoTT .GetBinContent(iBin)
        bkgNoTTErr  = hBkgNoTT .GetBinError  (iBin)

        if (bkgNoTT + bkgNoTTErr*bkgNoTTErr) > 0:
            bkgNoTTErrTot   = math.sqrt( bkgNoTT + bkgNoTTErr*bkgNoTTErr)
        else:
            bkgNoTTErrTot = 0
        #print "D = ",dataNoTT, "B = ", bkgNoTT, " +/- ", math.sqrt(bkgNoTT)," (stat) +/- ", bkgNoTTErr , " (sys) ", bkgNoTTErrTot
        
        #cv = ratioIn.GetBinContent(iBin)
        #err = ratioIn.GetBinError(iBin)
        if bkgNoTTErrTot == 0:
            pull.SetBinContent(iBin,0)
            pull.SetBinError(iBin,0)
        else:
            pval = (dataNoTT-bkgNoTT)/bkgNoTTErrTot
            pull.SetBinContent(iBin,pval)
            pull.SetBinError(iBin,0)

    return pull



def getRatio(hName,d4Hist,d3Hist,ttHist,rebin=1):
    hBkg  = getBkgModel(hName, ttHist, d3Hist,rebin=rebin)
    hData = d4Hist.Get(hName).Clone()
    hBkg.Scale(hData.Integral()/hBkg.Integral())
    hData.Rebin(rebin)
    hData.Divide(hBkg)
    hData.SetName(hData.GetName()+"_ratio")
    hData.SetMarkerStyle(20)
    hData.SetMarkerSize(0.7)
    hData.GetYaxis().SetRangeUser(0.7,1.3)
    hData.GetYaxis().SetTitle("Ratio (Data/Bkg)")
    return hData




def getPullFromRatio(ratioIn):
    pull = ratioIn.Clone()
    pull.SetName(ratioIn.GetName()+"_pullFromRatio")    
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
            h.Draw("same AXIS")
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
            h.Draw("same AXIS")

    can.SaveAs(o.outDir+"/"+name+".pdf")

    

can = ROOT.TCanvas()
ttFiles  = []
d3Files  = []
d4Files  = []

colors = [ROOT.kBlack,ROOT.kGray,ROOT.kBlue,ROOT.kRed,ROOT.kOrange,ROOT.kMagenta,ROOT.kCyan]


#
# Read Files
#
for sItr, s in enumerate(subSamples):

    tt = "/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b/TTRunII/hists_4b_wFVT_"+mixedName+"_v"+s+"_comb_b0p6.root"
    d3 = "/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b/dataRunII/hists_3b_wJCM_"+mixedName+"_v"+s+"_comb_wFVT_"+mixedName+"_v"+s+"_comb_b0p6.root "
    d4 = "/uscms/home/jda102/nobackup/HH4b/CMSSW_10_2_0/src/closureTests/3bMix4b/dataRunII/hists_4b_wFVT_"+mixedName+"_v"+s+"_comb_b0p6.root"

    ttFiles.append(ROOT.TFile(tt,"READ"))
    d3Files.append(ROOT.TFile(d3,"READ"))
    d4Files.append(ROOT.TFile(d4,"READ"))





def makePlots(outDir, hName,outName,rebin):

    pullsNoTTbar    = []
    ratios = []
    pullsFromRatio  = []


    for sItr, s in enumerate(subSamples):

        pullsNoTTbar.append(getPullNoTTBar(hName, d4Files[sItr], d3Files[sItr], ttFiles[sItr], rebin=rebin))
        pullsNoTTbar[sItr].SetLineColor(colors[sItr])
        pullsNoTTbar[sItr].SetMarkerColor(colors[sItr])

        ratios.append(getRatio(hName, d4Files[sItr], d3Files[sItr], ttFiles[sItr], rebin=rebin))
        ratios[sItr].SetLineColor(colors[sItr])
        ratios[sItr].SetMarkerColor(colors[sItr])

        pullsFromRatio.append(getPullFromRatio(ratios[sItr]))


        can.cd()
        pullsNoTTbar[sItr].Draw("PE")
        zero = lineAt(0,pullsNoTTbar[sItr].GetXaxis().GetXmin(),ratios[sItr].GetXaxis().GetXmax())
        zero.Draw("same")
        can.SaveAs(o.outDir+"/"+outName+"_pullNoTTbar_v"+s+".pdf")


        can.cd()
        ratios[sItr].Draw("")
        one = lineAt(1,ratios[sItr].GetXaxis().GetXmin(),ratios[sItr].GetXaxis().GetXmax())
        one.Draw("same")
        can.SaveAs(o.outDir+"/"+outName+"_ratio_v"+s+".pdf")


        can.cd()
        pullsFromRatio[sItr].Draw("PE")
        zero = lineAt(0,ratios[sItr].GetXaxis().GetXmin(),ratios[sItr].GetXaxis().GetXmax())
        zero.Draw("same")
        can.SaveAs(o.outDir+"/"+outName+"_pullFromRatio_v"+s+".pdf")





    can.cd()

    pullNoTTbarMean1sigma = getMeanStd(pullsNoTTbar,1)
    pullNoTTbarMean1sigma.SetFillColor(ROOT.kGreen)
    pullNoTTbarMean1sigma.SetMarkerColor(ROOT.kGreen)

    pullNoTTbarMean2sigma = getMeanStd(pullsNoTTbar,2)
    pullNoTTbarMean2sigma.SetFillColor(ROOT.kYellow)
    pullNoTTbarMean2sigma.SetMarkerColor(ROOT.kYellow)

    drawAll(outName+"_pullNoTTbar", pullsNoTTbar ,yLine=0, drawOpts="PE",underLays=[pullNoTTbarMean2sigma,pullNoTTbarMean1sigma])


    drawAll(outName+"_ratio",ratios,yLine=1)

    pullFromRatioMean1sigma = getMeanStd(pullsFromRatio,1)
    pullFromRatioMean1sigma.SetFillColor(ROOT.kGreen)

    pullFromRatioMean2sigma = getMeanStd(pullsFromRatio,2)
    pullFromRatioMean2sigma.SetFillColor(ROOT.kYellow)

    drawAll(outName+"_pullFromRatio", pullsFromRatio ,yLine=0, drawOpts="PE",underLays=[pullFromRatioMean2sigma,pullFromRatioMean1sigma])

    outFile.cd()
    pullNoTTbarMean1sigma.SetName(outName)
    pullNoTTbarMean1sigma.Write()


#hPath = "passXWt/fourTag/mainView"
hPath = "passMDRs/fourTag/mainView"

outFile = ROOT.TFile(o.outDir+"/plotClosureTest_"+o.mixedName+"_b0p6_comb.root","RECREATE")

for r in ["SR","SB","CR"]:

    variables = [
        ("SvB_ps_zz", 2),
        ("SvB_ps",    2),
        ("SvB_ps_zh", 2),
        #("SvB_q_score", 2),
        ("FvT", 2),
        ]

    for v in variables:
        makePlots(outFile, hName = hPath+"/"+r+"/"+v[0],  outName=r+"_"+v[0] ,rebin=v[1])
          
