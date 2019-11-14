import ROOT

TTH_FileName = "TTToHadronic2018_noDiJetMassCut/hists.root"
TTS_FileName = "TTToSemiLeptonic2018_noDiJetMassCut/hists.root"
TT2_FileName = "TTTo2L2Nu2018_noDiJetMassCut/hists.root"


TTH_File = ROOT.TFile(TTH_FileName,"READ")
TTS_File = ROOT.TFile(TTS_FileName,"READ")
TT2_File = ROOT.TFile(TT2_FileName,"READ")


BrlNu = 3./9
BrHad = 6./9

SF_2toH = (BrHad*BrHad)/(BrlNu*BrlNu)
SF_2toS = (2*BrlNu*BrHad)/(BrlNu*BrlNu)


def getCounts(inFile,tag,region):
    return inFile.Get("passMDRs/"+tag+"/mainView/"+region+"/nCanJets").GetBinContent(5)

def getIntegral(inFile,tag,region,minBin,maxBin):
    h = inFile.Get("passMDRs/"+tag+"/mainView/"+region+"/nTrueBJets")
    return h.Integral(h.FindBin(minBin),h.FindBin(maxBin))


def printTag(inFile_TTH, inFile_TTS, inFile_TT2,tag,region): 

    TTH_Counts = getCounts(TTH_File, tag, region)
    TTH_ttbb_Counts = getIntegral(TTH_File, tag, region,3,10)

    TTS_Counts = getCounts(TTS_File, tag, region)
    TTS_ttbb_Counts = getIntegral(TTS_File, tag, region,3,10)

    TT2_Counts = getCounts(TT2_File, tag, region)
    TT2_ttbb_Counts = getIntegral(TT2_File, tag, region,3,10)

    TTT_Counts = TTH_Counts + TTS_Counts + TT2_Counts
    TTT_ttbb_Counts = TTH_ttbb_Counts + TTS_ttbb_Counts + TT2_ttbb_Counts

    print "Total:",round(TTT_Counts,1)
    print "-"*20
    print "\t all had:",round(TTH_Counts,1),"(",round(TTH_Counts/TTT_Counts,3),")"
    print "\t \t ttbb ~",round(TTH_ttbb_Counts),"(",round(TTH_ttbb_Counts/TTT_Counts,3),")"
    print "\t semilep:",round(TTS_Counts,1),"(",round(TTS_Counts/TTT_Counts,3),")"
    print "\t \t ttbb ~",round(TTS_ttbb_Counts),"(",round(TTS_ttbb_Counts/TTT_Counts,3),")"
    print "\t di-lep:", round(TT2_Counts,1),"(",round(TT2_Counts/TTT_Counts,3),")"
    print "\t \t ttbb ~",round(TT2_ttbb_Counts),"(",round(TT2_ttbb_Counts/TTT_Counts,3),")"
    print "\t ttbb fraction ",round(TTT_ttbb_Counts/TTT_Counts,3)
    


def printRegion(inFile_TTH, inFile_TTS, inFile_TT2,region): 
    print "FourTag"
    printTag(inFile_TTH,inFile_TTS,inFile_TT2,"fourTag",region)

    print 
    print "ThreeTag"
    printTag(inFile_TTH,inFile_TTS,inFile_TT2,"threeTag",region)


print "Inclusive"
printRegion(TTH_File,TTS_File,TT2_File,"inclusive")

print   "\n"*3

print "SCSR"
printRegion(TTH_File,TTS_File,TT2_File,"SCSR")


print   "\n"*3

print "SR"
printRegion(TTH_File,TTS_File,TT2_File,"SR")




#nprint TTS_File
#print TT2_File
