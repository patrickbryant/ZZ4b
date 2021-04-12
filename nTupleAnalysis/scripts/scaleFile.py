import ROOT
ROOT.gROOT.SetBatch(True)
import optparse

parser = optparse.OptionParser()

parser.add_option('-i', '--inputFile',default="hists.root")
parser.add_option('-s', '--scaleFactor',default=1.0)
#parser.add_option('-o', '--output',dest="data",default="hists_scaled.root")
#parser.add_option('--tt',dest="tt",default="hists.root")#-t causes ROOT TH1::Fit to crash... weirdest bug I have ever seen.
#parser.add_option('-q', '--qcd',dest="qcd",default="hists.root")

o, a = parser.parse_args()

inFile = ROOT.TFile.Open(o.inputFile)
output = o.inputFile
if "root://" in o.inputFile: output = o.inputFile.split("/")[-1]
output = output.replace(".root","_scaled.root")
f_output  = ROOT.TFile(output,"RECREATE")

print "Scaling:"
print " \tinputFile:",inFile
print " \toutputFile:",output
print " \tscale factor:",o.scaleFactor

def recursive_scale(indir, inDirName=""):
    for key in indir.GetListOfKeys():
        name=key.GetName()
        obj=key.ReadObj()
        if obj.InheritsFrom(ROOT.TDirectoryFile.Class()):
            if "cutflow" in name:
                obj.Delete()
                continue
            thisDirName = inDirName+("/" if inDirName else "")+name
            print "mkdir",thisDirName
            f_output.mkdir(thisDirName)
            f_output.cd(thisDirName)
            recursive_scale(obj, thisDirName)
        else:
            # print inDirName+"/"+name
            f_output.cd(inDirName)
            if obj.InheritsFrom(ROOT.TH1F.Class()):
                h_output   = ROOT.TH1F(obj)
                h_output.Sumw2()
            if obj.InheritsFrom(ROOT.TH2F.Class()):
                h_output   = ROOT.TH2F(obj)
            h_output.Scale(float(o.scaleFactor))
            h_output.Write()
            # if "nPV" in name:
            #     f_qcd.ls()
            #     raw_input()
        obj.Delete()
        

recursive_scale(inFile)
f_output.Close()
