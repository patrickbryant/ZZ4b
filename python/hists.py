from ROOT import TH1F, TH2F

def makeTH1F(directory,name,title,bins,low,high):
    h = TH1F(name,title,bins,low,high)
    h.SetDirectory(directory)
    return h

def makeTH2F(directory,name,title,xBins,xLow,xHigh,yBins,yLow,yHigh):
    h = TH2F(name,title,xBins,xLow,xHigh,yBins,yLow,yHigh)
    h.SetDirectory(directory)
    return h
