from hists import *

class eventViewHists:
    def __init__(self, outFile, directory):
        self.thisDir = outFile.mkdir(directory)

        self.leadSt_m = makeTH1F(self.thisDir, "leadSt_m", directory+"_leadSt_m;    leading(S_{T}) mass [GeV]; Entries", 250, 0, 250)
        self.sublSt_m = makeTH1F(self.thisDir, "sublSt_m", directory+"_sublSt_m; subleading(S_{T}) mass [GeV]; Entries", 250, 0, 250)
        self.leadSt_m_vs_sublSt_m = makeTH2F(self.thisDir, "leadSt_m_vs_sublSt_m",
                                             directory+"_leadSt_m_vs_sublSt_m; leading(S_{T}) mass [GeV]; subleading(S_{T}) mass [GeV];Entries",
                                             50,0,250, 50,0,250)
        self.xZZ = makeTH1F(self.thisDir, "xZZ", directory+"_xZZ; xZZ; Entries", 50, 0, 5)

    def Fill(self, view):
        self.leadSt_m.Fill(view.leadSt.m)
        self.sublSt_m.Fill(view.sublSt.m)
        self.leadSt_m_vs_sublSt_m.Fill(view.leadSt.m, view.sublSt.m)
        self.xZZ.Fill(view.xZZ)

    def Write(self, outFile=None):
        self.thisDir.cd()
        self.leadSt_m.Write()
        self.sublSt_m.Write()
        self.leadSt_m_vs_sublSt_m.Write()
        self.xZZ.Write()        

