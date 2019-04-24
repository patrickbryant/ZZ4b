import ROOT
ROOT.gROOT.SetBatch(True)
import operator as op
import sys
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
from PlotTools import read_parameter_file

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom #double slash means integer division or "floor" division

def getCombinatoricWeight(f,nj):#number of selected jets ie a 5jet event has nj=5 and w = 2f(1-f)+f^2.
    w = 0
    nb = 3 #number of required bTags
    nl = nj-nb #number of selected untagged jets ("light" jets)
    for i in range(1,nl + 1):#i is the number of pseudoTags in this combination
        #  (ways to choose i pseudoTags from nl light jets) * pseudoTagProb^i * (1-pseudoTagProb)^{nl-i}
        w += ncr(nl,i) * f**i * (1-f)**(nl-i)

    #fancier model looping over all possible pairs of "light" jets where each pair can have zero, one or two pseudoTags
    #for i in range(1, ncr(nl, 2)+1):
    #    w += 
    return w

import copy, sys
import optparse
import math
from array import array
import os

parser = optparse.OptionParser()

parser.add_option('--noFitWeight',dest='noFitWeight',default="")
parser.add_option('-w', '--weightSet',dest="weightSet",default="")
parser.add_option('-r',dest="weightRegion",default="")
parser.add_option('-d', '--data',dest="data",default="hists.root")
parser.add_option('-o', '--outputDir',dest='outputDir',default="")
parser.add_option('--injectFile',dest="injectFile",default="")

o, a = parser.parse_args()

if not os.path.isdir(o.outputDir):
    os.mkdir(o.outputDir)

inFile = ROOT.TFile(o.data,"READ")
print "Input file:",o.data
mu_qcd = {}
mu_qcd_err = {}

class modelParameter:
    def __init__(self, name=""):
        self.name = name
        self.value = None
        self.error = None

    def dump(self):
        print self.name,self.value,"+/-",self.error

class jetCombinatoricModel:
    def __init__(self):
        self.pseudoTagProb = modelParameter("pseudoTagProb")
        self.fourJetScale  = modelParameter("fourJetScale")
        self.moreJetScale  = modelParameter("moreJetScale")

    def dump(self):
        self.pseudoTagProb.dump()
        self.fourJetScale.dump()
        self.moreJetScale.dump()

jetCombinatoricModelNext = o.outputDir+"jetCombinatoricModel_"+o.weightRegion+"_"+o.weightSet+".txt"
print jetCombinatoricModelNext
jetCombinatoricModelFile =             open(jetCombinatoricModelNext, "w")
jetCombinatoricModels = {}

# variables = []
def get(rootFile, path):
    obj = rootFile.Get(path)
    if str(obj) == "<ROOT.TObject object at 0x(nil)>": 
        rootFile.ls()
        print 
        print "ERROR: Object not found -", rootFile, path
        sys.exit()

    else: return obj

def makePositive(hist):
    for bin in range(1,hist.GetNbinsX()+1):
        x   = hist.GetXaxis().GetBinCenter(bin)
        y   = hist.GetBinContent(bin)
        err = hist.GetBinError(bin)
        hist.SetBinContent(bin, y if y > 0 else 0.0)
        hist.SetBinError(bin, err if y>0 else 0.0)

def do_variable_rebinning(hist,bins,divide=True):
    a=hist.GetXaxis()
    newhist=ROOT.TH1F(hist.GetName()+"_variableBins",
                      hist.GetTitle()+";"+hist.GetXaxis().GetTitle()+";"+hist.GetYaxis().GetTitle(),
                      len(bins)-1,
                      array('d',bins))

    newhist.Sumw2()
    newa=newhist.GetXaxis()

    for b in range(1, hist.GetNbinsX()+1):
        newb             = newa.FindBin(a.GetBinCenter(b))

        # Get existing new content (if any)
        val              = newhist.GetBinContent(newb)
        err              = newhist.GetBinError(newb)

        # Get content to add
        ratio_bin_widths = newa.GetBinWidth(newb)/a.GetBinWidth(b) if divide else 1.0
        val              = val+hist.GetBinContent(b)/ratio_bin_widths
        err              = math.sqrt(err**2+(hist.GetBinError(b)/ratio_bin_widths)**2)
        newhist.SetBinContent(newb,val)
        newhist.SetBinError(newb,err)

    return newhist


def getHists(cut,region,var,plot=False):#allow for different cut for mu calculation
    baseName = cut+"_"+region+"_"+var#+("_use_mu" if mu_cut else "")
    data4b        = inFile       .Get(cut+"/fourTag/mainView/"+region+"/"+var)
    data4b        .SetName("data4b_"+baseName)
    data2b        = inFile       .Get(cut+"/threeTag/mainView/"+region+"/"+var)
    data2b        .SetName("data2b_"+baseName)

    #
    # Make qcd histogram
    #
    if "TH1" in str(data2b):
        qcd = ROOT.TH1F(data2b)
    elif "TH2" in str(data2b):
        qcd = ROOT.TH2F(data2b)
    qcd.SetName("qcd_"+baseName)

    if "TH1" in str(data2b):
        bkgd = ROOT.TH1F(qcd)
    elif "TH2" in str(data2b):
        bkgd = ROOT.TH2F(data2b)
    bkgd.SetName("bkgd_"+baseName)

    data4b.SetLineColor(ROOT.kBlack)
    qcd.SetFillColor(ROOT.kYellow)
    qcd.SetLineColor(ROOT.kBlack)
        
    if plot:
        c=ROOT.TCanvas(var+"_"+cut+"_postfit","Post-fit")
        data4b.Draw("P EX0")
        stack = ROOT.THStack("stack","stack")
        stack.Add(qcd,"hist")
        stack.Draw("HIST SAME")
        data4b.SetStats(0)
        data4b.SetMarkerStyle(20)
        data4b.SetMarkerSize(0.7)
        data4b.Draw("P EX0 SAME axis")
        data4b.Draw("P EX0 SAME")
        plotName = o.outputDir+"/"+var+"_"+cut+".pdf" 
        print plotName
        c.SaveAs(plotName)
        del stack

    return (data4b, data2b, qcd, bkgd)

cuts = ["passMDRs"]
for cut in cuts:
    #
    # Get scale factors for hists without pseudoTagWeight
    #
    (data4b, data2b, qcd, bkgd) = getHists(cut,o.weightRegion,"nSelJetsUnweighted")
    mu_qcd_no_nJetWeight    = data4b.Integral()/data2b.Integral()

    #jetCombinatoricModelFileLine = "mu_qcd_no_nJetWeight_"+cut+"       "+str(mu_qcd_no_nJetWeight)+"\n"
    #print jetCombinatoricModelFileLine,
    #jetCombinatoricModelFile.write(jetCombinatoricModelFileLine)    

    #
    # Get scale factors for hists with pseudoTagWeight
    #
    (data4b, data2b, qcd, bkgd) = getHists(cut,o.weightRegion,"nSelJets","passMDRs")
    mu_qcd[cut] = data4b.Integral()/data2b.Integral()

    #jetCombinatoricModelFileLine = "mu_qcd_"+cut+"       "+str(mu_qcd[cut])+"\n"
    #print jetCombinatoricModelFileLine,
    #jetCombinatoricModelFile.write(jetCombinatoricModelFileLine) 

    #post-fit plots
    #getHists(cut,o.weightRegion,"nSelJets","passMDRs")

    del data4b
    del data2b
    del qcd
    del bkgd



for cut in ["passMDRs"]:
    #
    # Compute correction for pseudoTagProb
    #
    (data4b, data2b, qcd, bkgd) = getHists(cut,o.weightRegion,"nSelJetsUnweighted")
    #pseudoTagProb = jetCombinatoricModelDict["pseudoTagProb_"+cut]
    #fourJetScale = jetCombinatoricModelDict["fourJetScale_"+cut]
    #moreJetScale = jetCombinatoricModelDict["moreJetScale_"+cut]

    def bkgd_func_njet(x,par):
        nj = int(x[0] + 0.5)
        if nj < 4: return 0
        modelRatio = getCombinatoricWeight(par[0],nj)#/getCombinatoricWeight(pseudoTagProb,nj)
        if nj == 4:
            modelRatio *= par[1]#/fourJetScale
        else:
            modelRatio *= par[2]#/moreJetScale
        b = qcd.GetXaxis().FindBin(x[0])
        return modelRatio*qcd.GetBinContent(b)

    # set to prefit scale factor
    tf1_bkgd_njet = ROOT.TF1("tf1_bkgd",bkgd_func_njet,3.5,9.5,3)
    tf1_bkgd_njet.SetParName(0,"pseudoTagProb")
    tf1_bkgd_njet.SetParameter(0,0.149242258546)
    #tf1_bkgd_njet.SetParameter(0,pseudoTagProb)
    tf1_bkgd_njet.SetParName(1,"fourJetScale")
    tf1_bkgd_njet.SetParameter(1,0.617844162274)
    #tf1_bkgd_njet.SetParameter(1,fourJetScale)
    #tf1_bkgd_njet.FixParameter(1,1)
    tf1_bkgd_njet.SetParName(2,"moreJetScale")
    tf1_bkgd_njet.SetParameter(2,0.524457189962)
    #tf1_bkgd_njet.SetParameter(2,moreJetScale)

    # perform fit
    data4b.Fit(tf1_bkgd_njet,"0R")
    chi2 = tf1_bkgd_njet.GetChisquare()
    prob = tf1_bkgd_njet.GetProb()
    print "chi^2 =",chi2,"| p-value =",prob

    jetCombinatoricModels[cut] = jetCombinatoricModel()
    jetCombinatoricModels[cut].pseudoTagProb.value = tf1_bkgd_njet.GetParameter(0)
    jetCombinatoricModels[cut].pseudoTagProb.error = tf1_bkgd_njet.GetParError(0)
    jetCombinatoricModels[cut].fourJetScale.value = tf1_bkgd_njet.GetParameter(1)
    jetCombinatoricModels[cut].fourJetScale.error = tf1_bkgd_njet.GetParError(1)
    jetCombinatoricModels[cut].moreJetScale.value = tf1_bkgd_njet.GetParameter(2)
    jetCombinatoricModels[cut].moreJetScale.error = tf1_bkgd_njet.GetParError(2)
    jetCombinatoricModels[cut].dump()
    jetCombinatoricModelFile.write("pseudoTagProb_"+cut+"       "+str(jetCombinatoricModels[cut].pseudoTagProb.value)+"\n")
    jetCombinatoricModelFile.write("pseudoTagProb_"+cut+"_err   "+str(jetCombinatoricModels[cut].pseudoTagProb.error)+"\n")
    jetCombinatoricModelFile.write("fourJetScale_"+cut+"       "+str(jetCombinatoricModels[cut].fourJetScale.value)+"\n")
    jetCombinatoricModelFile.write("fourJetScale_"+cut+"_err   "+str(jetCombinatoricModels[cut].fourJetScale.error)+"\n")
    jetCombinatoricModelFile.write("moreJetScale_"+cut+"       "+str(jetCombinatoricModels[cut].moreJetScale.value)+"\n")
    jetCombinatoricModelFile.write("moreJetScale_"+cut+"_err   "+str(jetCombinatoricModels[cut].moreJetScale.error)+"\n")

    #(data4b, data2b, qcd, bkgd) = getHists(cut,o.weightRegion,"nSelJets")
    c=ROOT.TCanvas("nJetOther_"+cut+"_postfit_tf1","Post-fit")
    #data4b.SetLineColor(ROOT.kBlack)
    data4b.Draw("PE")
    qcdDraw = ROOT.TH1F(qcd)
    qcdDraw.SetName(qcd.GetName()+"draw")

    stack = ROOT.THStack("stack","stack")
    stack.Add(qcdDraw,"hist")
    stack.Draw("HIST SAME")
    data4b.SetStats(0)
    data4b.SetMarkerStyle(20)
    data4b.SetMarkerSize(0.7)
    data4b.Draw("P EX0 SAME axis")
    data4b.Draw("P EX0 SAME")
    tf1_bkgd_njet.SetLineColor(ROOT.kRed)
    tf1_bkgd_njet.Draw("SAME")
    histName = o.outputDir+"/"+"nSelJets_"+cut+"_postfit_tf1.pdf"
    print histName
    c.SaveAs(histName)

jetCombinatoricModelFile.close()
jetCombinatoricModelFile.close()


#
# now compute reweighting functions
#
reweightFile = o.outputDir+"/reweight_"+o.weightRegion+"_"+o.weightSet+".root"
print reweightFile
outFile = ROOT.TFile(reweightFile,"RECREATE")
outFile.cd()

def getBins(data,bkgd,xMax=None):
    
    firstBin = 1
    for bin in range(1,bkgd.GetNbinsX()+1):
        if bkgd.GetBinContent(bin) > 0: 
            firstBin = bin
            break

    lastBin = 0
    for bin in range(bkgd.GetNbinsX(),0,-1):
        if bkgd.GetBinContent(bin) > 0:
            lastBin = bin
            break

    cMax = 0
    for bin in range(1,bkgd.GetNbinsX()+1):
        c = bkgd.GetBinContent(bin)
        if c > cMax: cMax = c

    bins = [lastBin+1]
    sMin = 50
    s=0
    b=0
    bkgd_err=0
    f=-1
    for bin in range(lastBin,firstBin-1,-1):
        if xMax:
            x  = bkgd.GetXaxis().GetBinLowEdge(bin)
            if x >= xMax: bins.append(bin)

        s += data.GetBinContent(bin)
        b += bkgd.GetBinContent(bin)
        bkgd_err += bkgd.GetBinError(bin)**2
        if not b:  continue
        if 1.0/b**0.5 > 0.05: continue
        bins.append(bin)
        f = s/b
        s = 0
        b = 0
        bkgd_err = 0

    if s<sMin: bins.pop()
    if firstBin not in bins: bins.append(firstBin)

    bins.sort()
    
    if len(bins)>20:
        newbins = []
        for i in range(len(bins)):
            if i == len(bins)-1 or i%2==0: newbins.append(bins[i])
        bins = newbins
    bins = range(1,bins[0]) + bins + range(bins[-1]+1,data.GetNbinsX()+1)

    binsX = []
    for bin in bins:
        x = int(data.GetXaxis().GetBinLowEdge(bin)*1e6)/1.0e6
        binsX.append(x)
    binsX.append(binsX[-1]+data.GetXaxis().GetBinWidth(bins[-1]))

    #compute x-mean of each bin
    meansX = []
    Nx = 0
    Ix = 0
    I0 = 0
    I1 = 0
    i  = 1
    for bin in range(1,bkgd.GetNbinsX()+1):
        c = bkgd.GetBinContent(bin)
        l = bkgd.GetXaxis().GetBinLowEdge(bin)
        w = bkgd.GetXaxis().GetBinWidth(bin)
        u = l+w
        x = bkgd.GetXaxis().GetBinCenter(bin)
        Nx += 1
        Ix += x
        I0 += c
        I1 += c*x
        if abs(u - binsX[i])<0.00001: 
            i+=1
            m = I1/I0 if I0>0 else Ix/Nx
            Nx = 0
            Ix = 0
            I0 = 0 
            I1 = 0
            meansX.append(m)
            
    return (binsX, meansX)



def calcWeights(var, cut, xMax=None):
    print "Make reweight spline for ",cut,var
    (data4b, data2b, qcd, bkgd) = getHists(cut, o.weightRegion, var)
    
    (rebin, mean) = getBins(data4b,qcd,xMax)
    print "rebin",rebin
    mean_double = array("d", mean)

    widths = [rebin[i+1]-rebin[i] for i in range(len(rebin)-1)]
    width = 1e6

    data4b        = do_variable_rebinning(data4b, rebin)
    qcd           = do_variable_rebinning(qcd,rebin)
    bkgd          = do_variable_rebinning(bkgd,   rebin)
    
    makePositive(bkgd)
    makePositive(data4b)
    makePositive(qcd)

    data4b.Write()
    qcd   .Write()
    bkgd  .Write()

    can = ROOT.TCanvas(data4b.GetName()+"_ratio",data4b.GetName()+"_ratio",800,400)
    can.SetTopMargin(0.05)
    can.SetBottomMargin(0.15)
    can.SetRightMargin(0.025)
    ratio = ROOT.TH1F(data4b)
    ratio.GetYaxis().SetRangeUser(0,2)
    ratio.SetName(data4b.GetName()+"_ratio")
    ratio.Divide(qcd)

    raw_ratio = ROOT.TH1F(ratio)
    raw_ratio.SetName(ratio.GetName()+"_raw")
    raw_ratio.SetLineColorAlpha(ROOT.kBlue,0)
    raw_ratio.SetMarkerColorAlpha(ROOT.kBlue,0)
    raw_ratio.SetTitle("")
    raw_ratio.SetStats(0)
    raw_ratio.SetMarkerStyle(20)
    raw_ratio.SetMarkerSize(0.7)
    raw_ratio.SetLineWidth(2)
    raw_ratio.GetXaxis().SetLabelFont(43)
    raw_ratio.GetXaxis().SetLabelSize(18)
    raw_ratio.GetXaxis().SetTitleFont(43)
    raw_ratio.GetXaxis().SetTitleSize(21)
    raw_ratio.GetXaxis().SetTitleOffset(1.1)
    raw_ratio.GetYaxis().SetTitle("Data / Background")
    raw_ratio.GetYaxis().SetLabelFont(43)
    raw_ratio.GetYaxis().SetLabelSize(18)
    raw_ratio.GetYaxis().SetLabelOffset(0.008)
    raw_ratio.GetYaxis().SetTitleFont(43)
    raw_ratio.GetYaxis().SetTitleSize(21)
    raw_ratio.GetYaxis().SetTitleOffset(0.85)
    raw_ratio.Draw("SAME PEX0")

    ratio.Write()

    ratio_TGraph  = ROOT.TGraphAsymmErrors()
    ratio_TGraph.SetName("ratio_TGraph"+var)

    #get first and last non-empty bin
    yf = ROOT.Double(1)
    yl = ROOT.Double(1)
    z  = ROOT.Double(0)
    found_first = False
    for bin in range(1,ratio.GetSize()-1):
        x = ROOT.Double(ratio.GetBinCenter(bin))
        c = ratio.GetBinContent(bin)
        if c > 0:
            if xMax:
                if x < xMax: yl = ROOT.Double(c)
            else:
                yl = ROOT.Double(c)
            if found_first: continue
            found_first = True
            yf = ROOT.Double(c)


    found_first = False
    found_last  = False
    p = 0
    done        = False
    errors=[]
    for bin in range(1,ratio.GetSize()-1):
        l = ROOT.Double(ratio.GetXaxis().GetBinLowEdge(bin))
        x = ROOT.Double(ratio.GetBinCenter(bin))
        u = l+ROOT.Double(ratio.GetXaxis().GetBinWidth(bin))
        m = ROOT.Double(mean[bin-1])
        ey  = ROOT.Double(ratio.GetBinError(bin))
        errors.append(ratio.GetBinError(bin))
        exl = m-l
        exh = u-m

        if x<l or x>u: print "ERROR: mean",m,"not between bin limits",l,u

        c = ratio.GetBinContent(bin)
        if c <= 0 and not found_first:
            y = yf
            ey = z
        elif c > 0:
            found_first = True
            y = ROOT.Double(c)
            if xMax:
                if x >= xMax: 
                    y = yl
                    found_last = True
        elif not found_first:
            y = yf
            ey = z
        else:
            y = yl
            ey = 0
            found_last = True


        if found_first and not found_last and widths[bin-1] < width: width = widths[bin-1]

        ratio_TGraph.SetPoint(p,m,y if "trigBit" not in var else ROOT.Double(c))
        ratio_TGraph.SetPointError(p,exl,exh,ey,ey)
        p+=1

    width = width*4

    ratio_TGraph.SetLineColor(ROOT.kBlack)
    ratio_TGraph.SetMarkerColor(ROOT.kBlack)
    ratio_TGraph.SetMarkerStyle(20)
    ratio_TGraph.SetMarkerSize(0.7)
    ratio_TGraph.Draw("SAME P")

    ratio_smoother = ROOT.TGraphSmooth("ratio_smoother")
    errors = array("d",errors)
    ratio_smooth = ratio_smoother.SmoothKern(ratio_TGraph,"normal",width,len(mean),mean_double)
    ratio_smooth.SetName("ratio_smooth")
    ratio_smooth.SetLineColor(ROOT.kGreen)
    ratio_smooth.Draw("SAME PE")
    ratio_TSpline = ROOT.TSpline3("spline_"+var, ratio_smooth)
    ratio_TSpline.SetName("spline_"+var)

    ratio_TSpline.Write()
    ratio_TSpline.SetLineColor(ROOT.kRed)
    ratio_TSpline.Draw("SAME")

    histName = o.outputDir+"/"+data4b.GetName()+"_ratio.pdf"
    print histName
    can.SaveAs(histName)

    del data4b
    del data2b
    del qcd
    del bkgd
    del ratio_TGraph
    return 


kinematicWeightsCut="passMDRs"
calcWeights("nTagClassifier",  kinematicWeightsCut)
