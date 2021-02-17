import sys
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import PlotTools
import ROOT

ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetHistLineWidth(2)
ROOT.gStyle.SetLineStyleString(2,"[12 12]")


inputPath = "closureTests/3bMix4b_rWbW2/weights/dataRunII_3bMix4b_rWbW2_b0p60p3"
#fileName = "jetCombinatoricModel_SB_02-03-00.txt"
fileName = "jetCombinatoricModel_SB_03-00-00.txt"

nSubSamples = 11

inputData = {}

for v in range(nSubSamples):
    
    if v == 10: v="OneFvT"

    inputData[v] = {}

    thisFileName = inputPath+"_v"+str(v)+"/"+fileName
    thisFile = open(thisFileName,"r")
    for line in thisFile:
        words = line.split()
        #print words
        
        dataName  = words[0]
        dataValue = words[1]
        inputData[v][dataName] = dataValue







def makeVarPlot(varName,xMin,xMax):

    can = ROOT.TCanvas()
    
    histAxis = ROOT.TH1F("axis","axis;"+varName.replace("_passMDRs","")+";sub-sample",1,xMin,xMax)
    histAxis.GetYaxis().SetRangeUser(0,nSubSamples+1)
    can.cd()
    dataPlot = ROOT.TGraphErrors(nSubSamples)
    dataPlot.SetLineWidth(2)
    for i in range(nSubSamples):
        if i == 10: 
            v="OneFvT" 
        else:
            v=i

        print i,float(inputData[v][varName]), "+/-",float(inputData[v][varName+"_err"])
        dataPlot.SetPoint     (i,float(inputData[v][varName]),float(i+1))
        dataPlot.SetPointError(i,float(inputData[v][varName+"_err"]),0.0)
    histAxis.Draw()
    dataPlot.Draw("P")
    can.SaveAs(varName+".pdf")


# Mixed 
varsRanges = [("pseudoTagProb_passMDRs",0.03,0.04),
              ("pairEnhancement_passMDRs",1.0,2),
              ("pairEnhancementDecay_passMDRs", 1.0, 1.5)]

# Mixed with weights
varsRanges = [("pseudoTagProb_passMDRs",0.04,0.05),
              ("pairEnhancement_passMDRs",0.5,1.5),
              ("pairEnhancementDecay_passMDRs", 0.5, 1.0)]

for v in varsRanges:
    makeVarPlot(v[0],v[1],v[2])
