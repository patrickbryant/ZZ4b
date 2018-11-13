from hists import *

class eventViewHists:
    def __init__(self, outFile, directory):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.leadSt_m = makeTH1F(self.thisDir, "leadSt_m", directory+"_leadSt_m;    leading(S_{T}) mass [GeV]; Entries", 250, 0, 250)
        self.sublSt_m = makeTH1F(self.thisDir, "sublSt_m", directory+"_sublSt_m; subleading(S_{T}) mass [GeV]; Entries", 250, 0, 250)
        self.leadSt_m_vs_sublSt_m = makeTH2F(self.thisDir, "leadSt_m_vs_sublSt_m",
                                             directory+"_leadSt_m_vs_sublSt_m; leading(S_{T}) mass [GeV]; subleading(S_{T}) mass [GeV]; Entries",
                                             50,0,250, 50,0,250)
        self.xZZ = makeTH1F(self.thisDir, "xZZ", directory+"_xZZ; xZZ; Entries", 50, 0, 5)
        self.mZZ = makeTH1F(self.thisDir, "mZZ", directory+"_mZZ; m_{ZZ} [GeV]; Entries", 220, 100, 1200)

        self.m4j_vs_leadStdR  = makeTH2F(self.thisDir, "m4j_vs_leadStdR",
                                         directory+"_m4j_vs_leadStdR; m_{4j} [GeV]; leading(S_{T}) mass [GeV]; Entries",
                                         110,100,1200, 20,0,4  )
        self.m4j_vs_sublStdR  = makeTH2F(self.thisDir, "m4j_vs_sublStdR",
                                         directory+"_m4j_vs_sublStdR; m_{4j} [GeV]; subleading(S_{T}) mass [GeV]; Entries",
                                         110,100,1200, 20,0,4  )

    def Fill(self, view, weight=1):
        self.leadSt_m.Fill(view.leadSt.m, weight)
        self.sublSt_m.Fill(view.sublSt.m, weight)
        self.leadSt_m_vs_sublSt_m.Fill(view.leadSt.m, view.sublSt.m, weight)
        self.xZZ.Fill(view.xZZ, weight)
        self.mZZ.Fill(view.mZZ, weight)
        self.m4j_vs_leadStdR.Fill(view.m4j, view.leadSt.dR, weight)
        self.m4j_vs_sublStdR.Fill(view.m4j, view.sublSt.dR, weight)

    def Write(self, outFile=None):
        self.thisDir.cd()
        self.leadSt_m.Write()
        self.sublSt_m.Write()
        self.leadSt_m_vs_sublSt_m.Write()
        self.xZZ.Write()        
        self.mZZ.Write()
        self.m4j_vs_leadStdR.Write()
        self.m4j_vs_sublStdR.Write()
