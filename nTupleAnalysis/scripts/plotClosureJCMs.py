import sys
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import PlotTools
import ROOT

ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetHistLineWidth(2)
ROOT.gStyle.SetLineStyleString(2,"[12 12]")


inputPath = "closureTests/3bMix4b/weights/dataRunII_3bMix4b_rWbW2_b0p6"
fileName = "jetCombinatoricModel_SB_00-00-07.txt"



inputData = {}

for v in range(7):
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
    histAxis.GetYaxis().SetRangeUser(0,8)
    can.cd()
    dataPlot = ROOT.TGraphErrors(7)
    dataPlot.SetLineWidth(2)
    for i in range(7):
        print i,float(inputData[i][varName]), "+/-",float(inputData[i][varName+"_err"])
        dataPlot.SetPoint     (i,float(inputData[i][varName]),float(i+1))
        dataPlot.SetPointError(i,float(inputData[i][varName+"_err"]),0.0)
    histAxis.Draw()
    dataPlot.Draw("P")
    can.SaveAs(varName+".pdf")
    

for v in [("pseudoTagProb_passMDRs",0.018,0.02),
          ("pairEnhancement_passMDRs",2.9,3.1),
          ("pairEnhancementDecay_passMDRs", 0.6,0.9)]:
    makeVarPlot(v[0],v[1],v[2])
