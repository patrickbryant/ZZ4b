from hists import *
from array import array
from diJetHists import *
from truthHists import truthHists

class eventViewHists:
    def __init__(self, outFile, directory, truth = False):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.lead   = diJetHists(outFile, directory, "lead")
        self.subl   = diJetHists(outFile, directory, "subl")
        self.leadSt = diJetHists(outFile, directory, "leadSt")
        self.sublSt = diJetHists(outFile, directory, "sublSt")
        self.leadM  = diJetHists(outFile, directory, "leadM")
        self.sublM  = diJetHists(outFile, directory, "sublM")

        self.dEta   = makeTH1F(self.thisDir, "dEta", directory+"/dEta; #Delta#eta(diJet_{1}, diJet_{2}); Entries", 80, -4, 4)

        self.leadSt_m_vs_sublSt_m = makeTH2F(self.thisDir, "leadSt_m_vs_sublSt_m",
                                             directory+"/leadSt_m_vs_sublSt_m; leading(S_{T}) mass [GeV]; subleading(S_{T}) mass [GeV]; Entries",
                                             50,0,250, 50,0,250)
        self.xZZ = makeTH1F(self.thisDir, "xZZ", directory+"/xZZ; xZZ; Entries", 50, 0, 5)
        self.mZZ = makeTH1F(self.thisDir, "mZZ", directory+"/mZZ; m_{ZZ} [GeV]; Entries", 220, 100, 1200)

        binsZ = [150, 182, 191, 200, 210, 220, 231, 242, 254, 266, 279, 292, 306, 321, 337, 353, 370, 388, 407, 427, 448, 470, 493, 517, 542, 569, 597, 626, 657,
                 689, 723, 759, 796, 835, 876, 919, 964, 1012, 1062, 1115, 1170, 1228, 1289, 1353, 1420, 1491, 1565, 1643, 1725, 1811, 1901, 1996, 2095]
        self.m4j_cor_Z_v = makeTH1F(self.thisDir, "m4j_cor_Z_v", directory+"/m4j_cor_Z_v; m_{ZZ} [GeV]; Entries", 52, binList=binsZ)
        self.m4j_cor_Z_f = makeTH1F(self.thisDir, "m4j_cor_Z_f", directory+"/m4j_cor_Z_v; bin; Entries", 52, 0, 52)
        
        self.m4j_vs_leadStdR  = makeTH2F(self.thisDir, "m4j_vs_leadStdR",
                                         directory+"/m4j_vs_leadStdR; m_{4j} [GeV]; leading(S_{T}) diJet #DeltaR(j,j) [GeV]; Entries",
                                         110,100,1200, 20,0,4  )
        self.m4j_vs_sublStdR  = makeTH2F(self.thisDir, "m4j_vs_sublStdR",
                                         directory+"/m4j_vs_sublStdR; m_{4j} [GeV]; subleading(S_{T}) diJet #DeltaR(j,j) [GeV]; Entries",
                                         110,100,1200, 20,0,4  )
        self.m4j_vs_leadPt = makeTH2F(self.thisDir, "m4j_vs_leadPt",
                                      directory+"/m4j_vs_leadPt; m_{4j} [GeV]; leading(p_{T}) diJet p_{T} [GeV]; Entries",
                                      110,100,1200, 40,0,400  )
        self.m4j_vs_sublPt = makeTH2F(self.thisDir, "m4j_vs_sublPt",
                                      directory+"/m4j_vs_sublPt; m_{4j} [GeV]; subleading(p_{T}) diJet p_{T} [GeV]; Entries",
                                      110,100,1200, 40,0,400  )
        self.m4j_vs_nViews = makeTH2F(self.thisDir, "m4j_vs_nViews",
                                      directory+"/m4j_vs_nViews; m_{4j} [GeV]; # of event views; Entries",
                                      110,100,1200, 3,0.5,3.5  )
        #use eventData quantities
        self.xWt = makeTH1F(self.thisDir, "xWt", directory+"/xWt; x_{Wt}; Entries", 50, 0 , 5)

        self.truth = None
        if truth:
            outFile.mkdir(directory+"/truth")
            self.truth = truthHists(outFile, directory+"/truth")
        
    def Fill(self, view, weight=1, event=None):
        self.lead  .Fill(view.lead,   weight)
        self.subl  .Fill(view.subl,   weight)
        self.leadSt.Fill(view.leadSt, weight)
        self.sublSt.Fill(view.sublSt, weight)
        self.leadM .Fill(view.leadM,  weight)
        self.sublM .Fill(view.sublM,  weight)

        self.dEta  .Fill(view.dEta,   weight)

        self.leadSt_m_vs_sublSt_m.Fill(view.leadSt.m, view.sublSt.m, weight)
        self.xZZ.Fill(view.xZZ, weight)
        self.mZZ.Fill(view.mZZ, weight)

        self.m4j_cor_Z_v.Fill(view.mZZ, weight)
        self.m4j_cor_Z_f.Fill(self.m4j_cor_Z_v.GetXaxis().FindBin(view.mZZ), weight)
        
        self.m4j_vs_leadStdR.Fill(view.m4j, view.leadSt.dR, weight)
        self.m4j_vs_sublStdR.Fill(view.m4j, view.sublSt.dR, weight)
        self.m4j_vs_nViews.Fill(view.m4j, view.nViews, weight)

        if event:
            self.xWt.Fill(event.xWt, weight)

        if self.truth and event:
            self.truth.Fill(event, weight)


    def Write(self, outFile=None):
        self.thisDir.cd()
        self.leadSt.Write()
        self.sublSt.Write()
        self.lead  .Write()
        self.subl  .Write()
        self.dEta  .Write()
        self.leadSt_m_vs_sublSt_m.Write()
        self.xZZ.Write()        
        self.mZZ.Write()

        self.m4j_cor_Z_v.Write()
        self.m4j_cor_Z_f.Write()
        
        self.m4j_vs_leadStdR.Write()
        self.m4j_vs_sublStdR.Write()
        self.m4j_vs_nViews.Write()

        self.xWt.Write()

        if self.truth:
            self.truth.Write()

