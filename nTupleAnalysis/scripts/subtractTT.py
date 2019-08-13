import ROOT
ROOT.gROOT.SetBatch(True)
import optparse

parser = optparse.OptionParser()

parser.add_option('-d', '--data',dest="data",default="hists.root")
parser.add_option('--tt',dest="tt",default="hists.root")#-t causes ROOT TH1::Fit to crash... weirdest bug I have ever seen.
parser.add_option('-q', '--qcd',dest="qcd",default="hists.root")

o, a = parser.parse_args()

inFile = ROOT.TFile(o.data,"READ")
ttFile = ROOT.TFile(o.tt,"READ")
f_qcd  = ROOT.TFile(o.qcd,"RECREATE")

print "Subtracting ttbar MC from data to make qcd hists"
print " data:",inFile
print "   tt:",ttFile
print "  qcd:",f_qcd

def recursive_subtractTT(indir, inDirName=""):
    for key in indir.GetListOfKeys():
        name=key.GetName()
        obj=key.ReadObj()
        if obj.InheritsFrom(ROOT.TDirectoryFile.Class()):
            if "cutflow" in name:
                obj.Delete()
                continue
            thisDirName = inDirName+("/" if inDirName else "")+name
            print "mkdir",thisDirName
            f_qcd.mkdir(thisDirName)
            f_qcd.cd(thisDirName)
            recursive_subtractTT(obj, thisDirName)
        else:
            # print inDirName+"/"+name
            h_tt     = ttFile.Get(inDirName+"/"+name)
            f_qcd.cd(inDirName)
            if obj.InheritsFrom(ROOT.TH1F.Class()):
                h_qcd   = ROOT.TH1F(obj)
            if obj.InheritsFrom(ROOT.TH2F.Class()):
                h_qcd   = ROOT.TH2F(obj)
            h_qcd.Add(h_tt,-1)
            h_qcd.Write()
            h_tt.Delete()
            # if "nPV" in name:
            #     f_qcd.ls()
            #     raw_input()
        obj.Delete()
        

recursive_subtractTT(inFile)
f_qcd.Close()
