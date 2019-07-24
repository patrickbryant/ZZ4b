import ROOT
ROOT.gROOT.SetBatch(True)
import operator as op
import sys
from copy import copy
import optparse
import math
from array import array
import os
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
from PlotTools import read_parameter_file

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom #double slash means integer division or "floor" division

def getCombinatoricWeight(f,nj,pairEnhancement=0.0,pairEnhancementDecay=1.0,oddSuppression=0.0,oddSuppressionDecay=1.0):#number of selected jets ie a 5jet event has nj=5 and w = 2f(1-f)+f^2.
    #oddSuppression, oddSuppressionDecay = pairEnhancement, pairEnhancementDecay
    w = 0
    nb = 3 #number of required bTags
    nl = nj-nb #number of selected untagged jets ("light" jets)
    nPseudoTagProb = [0]
    for i in range(1,nl + 1):#i is the number of pseudoTags in this combination
        #  (ways to choose i pseudoTags from nl light jets) * pseudoTagProb^i * (1-pseudoTagProb)^{nl-i}
        wi  = ncr(nl,i) * f**i * (1-f)**(nl-i) #* ( 1 + (-1)**(i+1)*pairEnhancement/(nl**pairEnhancementDecay) )
        #wi *= 1 + (-1)**(i+1) * pairEnhancement/(nl**pairEnhancementDecay)
        if (i%2)==1: wi *= 1 + pairEnhancement/(nl**pairEnhancementDecay)
        #if(i%2)==0: wi *= (1 - oddSuppression /(nl**oddSuppressionDecay)
        w += wi
        nPseudoTagProb.append(wi)

    #fancier model looping over all possible pairs of "light" jets where each pair can have zero, one or two pseudoTags
    #for i in range(1, ncr(nl, 2)+1):
    #    w += 
    return w, nPseudoTagProb


parser = optparse.OptionParser()

parser.add_option('--noFitWeight',dest='noFitWeight',default="")
parser.add_option('-w', '--weightSet',dest="weightSet",default="")
parser.add_option('-r',dest="weightRegion",default="")
parser.add_option('-d', '--data',dest="data",default="hists.root")
parser.add_option('--tt',dest="tt",default="hists.root")#-t causes ROOT TH1::Fit to crash... weirdest bug I have ever seen.
parser.add_option('-o', '--outputDir',dest='outputDir',default="")
parser.add_option('--injectFile',dest="injectFile",default="")

o, a = parser.parse_args()

if not os.path.isdir(o.outputDir):
    os.mkdir(o.outputDir)

inFile = ROOT.TFile(o.data,"READ")
ttFile = ROOT.TFile(o.tt,"READ")
print "Input file:",o.data
print "tt file:",o.tt
#mu_qcd = {}
#mu_qcd_err = {}

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
        #self.fourJetScale  = modelParameter("fourJetScale")
        #self.moreJetScale  = modelParameter("moreJetScale")
        self.pairEnhancement=modelParameter("pairEnhancement")
        self.pairEnhancementDecay=modelParameter("pairEnhancementDecay")

    def dump(self):
        self.pseudoTagProb.dump()
        #self.fourJetScale.dump()
        #self.moreJetScale.dump()
        self.pairEnhancement.dump()
        self.pairEnhancementDecay.dump()

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
    data4b = inFile.Get(cut+"/fourTag/mainView/"+region+"/"+var)
    data4b.SetName("data4b_"+baseName)
    data3b = inFile.Get(cut+"/threeTag/mainView/"+region+"/"+var)
    data3b.SetName("data3b_"+baseName)
    tt4b = ttFile.Get(cut+"/fourTag/mainView/"+region+"/"+var)
    tt4b.SetName("tt4b_"+baseName)
    tt3b = ttFile.Get(cut+"/threeTag/mainView/"+region+"/"+var)
    tt3b.SetName("tt3b_"+baseName)

    #
    # Make qcd histograms
    #
    if "TH1" in str(data3b):
        qcd3b = ROOT.TH1F(data3b)
        qcd3b.SetName("qcd3b_"+baseName)
        qcd4b = ROOT.TH1F(data4b)
        qcd4b.SetName("qcd4b_"+baseName)
    elif "TH2" in str(data3b):
        qcd3b = ROOT.TH2F(data3b)
        qcd3b.SetName("qcd3b_"+baseName)
        qcd4b = ROOT.TH2F(data4b)
    qcd3b.Add(tt3b,-1)
    qcd4b.Add(tt4b,-1)

    if "TH1" in str(data3b):
        bkgd = ROOT.TH1F(qcd3b)
        bkgd.SetName("bkgd_"+baseName)
    elif "TH2" in str(data3b):
        bkgd = ROOT.TH2F(qcd3b)
        bkgd.SetName("bkgd_"+baseName)
    bkgd.Add(tt4b)

    data4b.SetLineColor(ROOT.kBlack)
    qcd3b.SetFillColor(ROOT.kYellow)
    qcd3b.SetLineColor(ROOT.kBlack)
    tt4b.SetLineColor(ROOT.kBlack)
    tt4b.SetFillColor(ROOT.kAzure-9)
        
    if plot:
        c=ROOT.TCanvas(var+"_"+cut+"_postfit","Post-fit")
        data4b.Draw("P EX0")
        stack = ROOT.THStack("stack","stack")
        stack.Add(tt4b,"hist")
        stack.Add(qcd3b,"hist")
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

    return (data4b, tt4b, qcd4b, data3b, tt3b, qcd3b)


cut="passMDRs"
getHists(cut,o.weightRegion,"FvT", plot=True)
getHists(cut,o.weightRegion,"FvTUnweighted", plot=True)
getHists(cut,o.weightRegion,"nPSTJets", plot=True)

for st in [""]:#, "_lowSt", "_midSt", "_highSt"]:
    getHists(cut,o.weightRegion,"nSelJets"+st, plot=True)
    #
    # Compute correction for pseudoTagProb
    #
    (data4b, tt4b, qcd4b, data3b, tt3b, qcd3b) = getHists(cut,o.weightRegion,"nSelJetsUnweighted"+st)
    print "nSelJetsUnweighted"+st, "data4b.Integral()", data4b.Integral(), "data3b.Integral()", data3b.Integral()

    (_, _, qcd4b_nTagJets, _, _, _) = getHists(cut,o.weightRegion,"nPSTJets"+st)
    n5b_true = qcd4b_nTagJets.GetBinContent(qcd4b_nTagJets.GetXaxis().FindBin(5))
    qcd4b.SetBinContent(qcd4b.GetXaxis().FindBin(0), qcd4b_nTagJets.GetBinContent(qcd4b_nTagJets.GetXaxis().FindBin(5)))
    qcd4b.SetBinContent(qcd4b.GetXaxis().FindBin(1), qcd4b_nTagJets.GetBinContent(qcd4b_nTagJets.GetXaxis().FindBin(6)))
    qcd4b.SetBinContent(qcd4b.GetXaxis().FindBin(2), qcd4b_nTagJets.GetBinContent(qcd4b_nTagJets.GetXaxis().FindBin(7)))

    def nTagPred(par,n):
        nPred = 0
        for bin in range(1,qcd3b.GetSize()-1):
            nj = int(qcd3b.GetBinCenter(bin))
            if nj < n: continue
            _, nPseudoTagProb = getCombinatoricWeight(par[0],nj,par[1],par[2])
            nPred += nPseudoTagProb[n-3] * qcd3b.GetBinContent(bin)
        return nPred

    def qcd3b_func_njet(x,par):
        nj = int(x[0] + 0.5)
        if nj == 0: # use first bin for nTagJets==5 model
            return nTagPred(par,5)
        if nj == 1:
            return nTagPred(par,6)
        if nj == 2:
            return nTagPred(par,7)

        if nj < 4: return 0
        w, _ = getCombinatoricWeight(par[0],nj,par[1],par[2])
        b = qcd3b.GetXaxis().FindBin(x[0])
        return w*qcd3b.GetBinContent(b)


    # set to prefit scale factor
    tf1_qcd3b_njet = ROOT.TF1("tf1_qcd3b",qcd3b_func_njet,-0.5,11.5,3)
    #tf1_qcd3b_njet = ROOT.TF1("tf1_qcd",qcd3b_func_njet,3.5,11.5,3)

    tf1_qcd3b_njet.SetParName(0,"pseudoTagProb")
    tf1_qcd3b_njet.SetParameter(0,0.149242258546)

    tf1_qcd3b_njet.SetParName(1,"pairEnhancement")
    tf1_qcd3b_njet.SetParLimits(1,0,1)
    tf1_qcd3b_njet.SetParameter(1,0)
    #tf1_qcd3b_njet.FixParameter(1,0.566741)
    #tf1_qcd3b_njet.FixParameter(1,0)

    tf1_qcd3b_njet.SetParName(2,"pairEnhancementDecay")
    tf1_qcd3b_njet.SetParameter(2,1.15)
    tf1_qcd3b_njet.SetParLimits(2,0.3,3)
    #tf1_qcd3b_njet.FixParameter(2,1.15127)
    #tf1_qcd3b_njet.FixParameter(2,1)

    # perform fit
    qcd4b.Fit(tf1_qcd3b_njet,"0R")
    chi2 = tf1_qcd3b_njet.GetChisquare()
    prob = tf1_qcd3b_njet.GetProb()
    print "chi^2 =",chi2,"| p-value =",prob

    jetCombinatoricModels[cut] = jetCombinatoricModel()
    jetCombinatoricModels[cut].pseudoTagProb.value = tf1_qcd3b_njet.GetParameter(0)
    jetCombinatoricModels[cut].pseudoTagProb.error = tf1_qcd3b_njet.GetParError(0)
    jetCombinatoricModels[cut].pairEnhancement.value = tf1_qcd3b_njet.GetParameter(1)
    jetCombinatoricModels[cut].pairEnhancement.error = tf1_qcd3b_njet.GetParError(1)
    jetCombinatoricModels[cut].pairEnhancementDecay.value = tf1_qcd3b_njet.GetParameter(2)
    jetCombinatoricModels[cut].pairEnhancementDecay.error = tf1_qcd3b_njet.GetParError(2)
    jetCombinatoricModels[cut].dump()
    jetCombinatoricModelFile.write("pseudoTagProb"+st+"_"+cut+"       "+str(jetCombinatoricModels[cut].pseudoTagProb.value)+"\n")
    jetCombinatoricModelFile.write("pseudoTagProb"+st+"_"+cut+"_err   "+str(jetCombinatoricModels[cut].pseudoTagProb.error)+"\n")
    jetCombinatoricModelFile.write("pairEnhancement"+st+"_"+cut+"       "+str(jetCombinatoricModels[cut].pairEnhancement.value)+"\n")
    jetCombinatoricModelFile.write("pairEnhancement"+st+"_"+cut+"_err   "+str(jetCombinatoricModels[cut].pairEnhancement.error)+"\n")
    jetCombinatoricModelFile.write("pairEnhancementDecay"+st+"_"+cut+"       "+str(jetCombinatoricModels[cut].pairEnhancementDecay.value)+"\n")
    jetCombinatoricModelFile.write("pairEnhancementDecay"+st+"_"+cut+"_err   "+str(jetCombinatoricModels[cut].pairEnhancementDecay.error)+"\n")

    n5b_pred = 0
    for bin in range(1,qcd3b.GetSize()-1):
        nj = int(qcd3b.GetBinCenter(bin))
        if nj < 5: continue
        w, nPseudoTagProb = getCombinatoricWeight(jetCombinatoricModels[cut].pseudoTagProb.value,nj,
                                                  jetCombinatoricModels[cut].pairEnhancement.value,
                                                  jetCombinatoricModels[cut].pairEnhancementDecay.value)
        n5b_pred += nPseudoTagProb[5-3] * qcd3b.GetBinContent(bin)
    print "Predicted number of 5b events:",n5b_pred
    print "   Actual number of 5b events:",n5b_true
        

    #(qcd4b, data3b, qcd, bkgd) = getHists(cut,o.weightRegion,"nSelJets")
    c=ROOT.TCanvas(cut+"_postfit_tf1","Post-fit")
    #data4b.SetLineColor(ROOT.kBlack)
    qcd4b.Draw("P EX0")
    #qcdDraw = ROOT.TH1F(qcd)
    #qcdDraw.SetName(qcd.GetName()+"draw")

    #stack = ROOT.THStack("stack","stack")
    #qcdDraw.Scale(data4b.Integral()/qcdDraw.Integral())
    #stack.Add(qcdDraw,"hist")
    #stack.Draw("HIST SAME")
    qcd3b.Draw("HIST SAME")
    qcd4b.SetStats(0)
    qcd4b.SetMarkerStyle(20)
    qcd4b.SetMarkerSize(0.7)
    qcd4b.Draw("P EX0 SAME axis")
    qcd4b.Draw("P EX0 SAME")
    tf1_qcd3b_njet.SetLineColor(ROOT.kRed)
    tf1_qcd3b_njet.Draw("SAME")
    histName = o.outputDir+"/"+"nJets"+st+"_"+cut+"_postfit_tf1.pdf"
    print histName
    c.SaveAs(histName)

jetCombinatoricModelFile.close()
jetCombinatoricModelFile.close()


# #
# # now compute reweighting functions
# #
# reweightFile = o.outputDir+"/reweight_"+o.weightRegion+"_"+o.weightSet+".root"
# print reweightFile
# outFile = ROOT.TFile(reweightFile,"RECREATE")
# outFile.cd()

# def getBins(data,bkgd,xMax=None):
    
#     firstBin = 1
#     for bin in range(1,bkgd.GetNbinsX()+1):
#         if bkgd.GetBinContent(bin) > 0: 
#             firstBin = bin
#             break

#     lastBin = 0
#     for bin in range(bkgd.GetNbinsX(),0,-1):
#         if bkgd.GetBinContent(bin) > 0:
#             lastBin = bin
#             break

#     cMax = 0
#     for bin in range(1,bkgd.GetNbinsX()+1):
#         c = bkgd.GetBinContent(bin)
#         if c > cMax: cMax = c

#     bins = [lastBin+1]
#     sMin = 50
#     s=0
#     b=0
#     bkgd_err=0
#     f=-1
#     minPrecision=0.03
#     for bin in range(lastBin,firstBin-1,-1):
#         if xMax:
#             x  = bkgd.GetXaxis().GetBinLowEdge(bin)
#             if x >= xMax: bins.append(bin)

#         s += data.GetBinContent(bin)
#         b += bkgd.GetBinContent(bin)
#         x = data.GetBinLowEdge(bin)
#         bkgd_err += bkgd.GetBinError(bin)**2
#         if not b:  continue
#         if 1.0/b**0.5 >= minPrecision or 1.0/s**0.5 >= minPrecision: continue
#         #print bin, x, s, b
#         bins.append(bin)
#         f = s/b
#         s = 0
#         b = 0
#         bkgd_err = 0

#     #if (b!=0 and 1.0/b**0.5 >= minPrecision): bins.pop()
#     if firstBin not in bins: bins.append(firstBin)

#     bins.sort()
    
#     # if len(bins)>20:
#     #     newbins = []
#     #     for i in range(len(bins)):
#     #         if i == len(bins)-1 or i%2==0: newbins.append(bins[i])
#     #     bins = newbins
#     bins = range(1,bins[0]) + bins + range(bins[-1]+1,data.GetNbinsX()+1)

#     binsX = []
#     for bin in bins:
#         x = int(data.GetXaxis().GetBinLowEdge(bin)*1e6)/1.0e6
#         binsX.append(x)
#     binsX.append(binsX[-1]+data.GetXaxis().GetBinWidth(bins[-1]))

#     #compute x-mean of each bin
#     meansX = []
#     Nx = 0
#     Ix = 0
#     I0 = 0
#     I1 = 0
#     i  = 1
#     for bin in range(1,bkgd.GetNbinsX()+1):
#         b = bkgd.GetBinContent(bin)
#         s = data.GetBinContent(bin)
#         c = b+s
#         l = bkgd.GetXaxis().GetBinLowEdge(bin)
#         w = bkgd.GetXaxis().GetBinWidth(bin)
#         u = l+w
#         x = bkgd.GetXaxis().GetBinCenter(bin)
#         Nx += 1
#         Ix += x
#         I0 += c
#         I1 += c*x
#         if abs(u - binsX[i])<0.00001: 
#             i+=1
#             m = I1/I0 if I0>0 else Ix/Nx
#             Nx = 0
#             Ix = 0
#             I0 = 0 
#             I1 = 0
#             meansX.append(m)
            
#     return (binsX, meansX)



# def calcWeights(var, cut, xMax=None):
#     print "Make reweight spline for ",cut,var
#     (data4b, data3b, qcd, bkgd) = getHists(cut, o.weightRegion, var)
    
#     (rebin, mean) = getBins(data4b,qcd,xMax)
#     print "rebin",rebin
#     mean_double = array("d", mean)

#     widths = [rebin[i+1]-rebin[i] for i in range(len(rebin)-1)]
#     width = 1e6

#     data4b        = do_variable_rebinning(data4b, rebin)
#     qcd           = do_variable_rebinning(qcd,rebin)
#     bkgd          = do_variable_rebinning(bkgd,   rebin)
    
#     makePositive(bkgd)
#     makePositive(data4b)
#     makePositive(qcd)

#     data4b.Write()
#     qcd   .Write()
#     bkgd  .Write()

#     can = ROOT.TCanvas(data4b.GetName()+"_ratio",data4b.GetName()+"_ratio",800,400)
#     can.SetTopMargin(0.05)
#     can.SetBottomMargin(0.15)
#     can.SetRightMargin(0.025)
#     ratio = ROOT.TH1F(data4b)
#     ratio.GetYaxis().SetRangeUser(0,2.5)
#     ratio.SetName(data4b.GetName()+"_ratio")
#     ratio.Divide(qcd)

#     raw_ratio = ROOT.TH1F(ratio)
#     raw_ratio.SetName(ratio.GetName()+"_raw")
#     raw_ratio.SetLineColorAlpha(ROOT.kBlue,0)
#     raw_ratio.SetMarkerColorAlpha(ROOT.kBlue,0)
#     raw_ratio.SetTitle("")
#     raw_ratio.SetStats(0)
#     raw_ratio.SetMarkerStyle(20)
#     raw_ratio.SetMarkerSize(0.7)
#     raw_ratio.SetLineWidth(2)
#     raw_ratio.GetXaxis().SetLabelFont(43)
#     raw_ratio.GetXaxis().SetLabelSize(18)
#     raw_ratio.GetXaxis().SetTitleFont(43)
#     raw_ratio.GetXaxis().SetTitleSize(21)
#     raw_ratio.GetXaxis().SetTitleOffset(1.1)
#     raw_ratio.GetYaxis().SetTitle("Data / Background")
#     raw_ratio.GetYaxis().SetLabelFont(43)
#     raw_ratio.GetYaxis().SetLabelSize(18)
#     raw_ratio.GetYaxis().SetLabelOffset(0.008)
#     raw_ratio.GetYaxis().SetTitleFont(43)
#     raw_ratio.GetYaxis().SetTitleSize(21)
#     raw_ratio.GetYaxis().SetTitleOffset(0.85)
#     raw_ratio.Draw("SAME PEX0")

#     ratio.Write()

#     ratio_TGraph  = ROOT.TGraphAsymmErrors()
#     ratio_TGraph.SetName("ratio_TGraph"+var)

#     #get first and last non-empty bin
#     yf = ROOT.Double(1)
#     yl = ROOT.Double(1)
#     z  = ROOT.Double(0)
#     found_first = False
#     for bin in range(1,ratio.GetSize()-1):
#         x = ROOT.Double(ratio.GetBinCenter(bin))
#         c = ratio.GetBinContent(bin)
#         if c > 0:
#             if xMax:
#                 if x < xMax: yl = ROOT.Double(c)
#             else:
#                 yl = ROOT.Double(c)
#             if found_first: continue
#             found_first = True
#             yf = ROOT.Double(c)


#     found_first = False
#     found_last  = False
#     p = 0
#     done        = False
#     errors=[]
#     for bin in range(1,ratio.GetSize()-1):
#         l = ROOT.Double(ratio.GetXaxis().GetBinLowEdge(bin))
#         x = ROOT.Double(ratio.GetBinCenter(bin))
#         u = l+ROOT.Double(ratio.GetXaxis().GetBinWidth(bin))
#         m = ROOT.Double(mean[bin-1])
#         ey  = ROOT.Double(ratio.GetBinError(bin))
#         errors.append(ratio.GetBinError(bin))
#         exl = m-l
#         exh = u-m

#         if x<l or x>u: print "ERROR: mean",m,"not between bin limits",l,u

#         c = ratio.GetBinContent(bin)
#         if c <= 0 and not found_first:
#             y = yf
#             ey = z
#         elif c > 0:
#             found_first = True
#             y = ROOT.Double(c)
#             if xMax:
#                 if x >= xMax: 
#                     y = yl
#                     found_last = True
#         elif not found_first:
#             y = yf
#             ey = z
#         else:
#             y = yl
#             ey = 0
#             found_last = True


#         if found_first and not found_last and widths[bin-1] < width: width = widths[bin-1]

#         ratio_TGraph.SetPoint(p,m,y)
#         ratio_TGraph.SetPointError(p,exl,exh,ey,ey)
#         p+=1

#     width = width*2

#     ratio_TGraph.SetLineColor(ROOT.kBlack)
#     ratio_TGraph.SetMarkerColor(ROOT.kBlack)
#     ratio_TGraph.SetMarkerStyle(20)
#     ratio_TGraph.SetMarkerSize(0.0)
#     ratio_TGraph.Draw("SAME P")

#     ratio_smoother = ROOT.TGraphSmooth("ratio_smoother")
#     errors = array("d",errors)
#     ratio_smooth = ratio_smoother.SmoothKern(ratio_TGraph,"normal",width,len(mean),mean_double)
#     ratio_smooth.SetName("ratio_smooth")
#     ratio_smooth.SetLineColor(ROOT.kGreen)
#     ratio_smooth.Draw("SAME PE")
#     ratio_TSpline = ROOT.TSpline3("spline_"+var, ratio_smooth)
#     ratio_TSpline.SetName("spline_beforeBoundaryConditions_"+var)

#     #find maximum of spline
#     sMin, sMax=ROOT.Double(1e6), ROOT.Double(-1e6)
#     xMin, xMax=0, 0
#     for bin in range(2,data3b.GetNbinsX()+1):
#         x=data3b.GetBinCenter(bin)
#         s=ratio_TSpline.Eval(x)
#         if s<sMin: xMin, sMin = x, ROOT.Double(s)
#         if s>sMax: xMax, sMax = x, ROOT.Double(s)
#     print "Spline Min", xMin, sMin
#     print "Spline Max", xMax, sMax
    
#     #set points after maximum equal to maximum to force spline boundary condition that spline stops at maximum
#     pMax=None
#     for p in range(ratio_smooth.GetN()):
#         x,y=ROOT.Double(),ROOT.Double()
#         ratio_smooth.GetPoint(p,x,y)
#         if x >= xMax:
#             ratio_smooth.SetPoint(p,x,sMax)
#             if not pMax: pMax=p
#     ratio_smooth.InsertPointBefore(pMax+1, ROOT.Double(xMax), sMax)
#     ratio_smooth.InsertPointBefore(pMax+1, ROOT.Double(xMax+width), sMax)
#     ratio_smooth.InsertPointBefore(pMax+1, ROOT.Double(xMax+2*width), sMax)

#     ratio_smooth.Sort()

#     #set points after minimum equal to minimum to force spline boundary condition that spline stops at minimum
#     pMin=None
#     for p in range(ratio_smooth.GetN(),0,-1):
#         x,y=ROOT.Double(),ROOT.Double()
#         ratio_smooth.GetPoint(p,x,y)
#         if x <= xMin:
#             if not pMin: pMin=p
#             ratio_smooth.SetPoint(p,x,sMin)
#     ratio_smooth.InsertPointBefore(pMin, ROOT.Double(xMin-2*width), sMin)
#     ratio_smooth.InsertPointBefore(pMin, ROOT.Double(xMin-width), sMin)
#     ratio_smooth.InsertPointBefore(pMin, ROOT.Double(xMin), sMin)

#     ratio_smooth.Sort()

#     ratio_TSpline = ROOT.TSpline3("spline_"+var, ratio_smooth)
#     ratio_TSpline.SetName("spline_"+var)

#     ratio_TSpline.Write()
#     ratio_TSpline.SetLineColor(ROOT.kRed)
#     ratio_TSpline.Draw("LSAME")

#     probRatio = ROOT.TF1("probRatio", "x/(1-x)", 0, 1)
#     probRatio.SetLineColor(ROOT.kBlue)
#     probRatio.Draw("SAME")

#     histName = o.outputDir+"/"+data4b.GetName()+"_ratio.pdf"
#     print histName
#     can.SaveAs(histName)

#     del data4b
#     del data3b
#     del qcd
#     del bkgd
#     del ratio_TGraph
#     return 


# kinematicWeightsCut="passMDRs"
# calcWeights("FvTUnweighted",  kinematicWeightsCut)
