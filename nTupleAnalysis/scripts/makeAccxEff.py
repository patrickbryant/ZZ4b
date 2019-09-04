import ROOT
import optparse

parser = optparse.OptionParser()
parser.add_option('-d', '--debug',                dest="debug",         action="store_true", default=False, help="debug")
parser.add_option('-y', '--year',                 dest="year",          default="2018", help="Year specifies trigger (and lumiMask for data)")
parser.add_option('-i', '--input',                dest="input",         default="/uscms/home/bryantp/nobackup/ZZ4b/ggZH4b2018/histsFromNanoAOD.root", help="input hists file containting cutflow")
o, a = parser.parse_args()

#make sure outputBase ends with /
output = "/".join(o.input.split("/")[:-1])+"/accxEff.root"
print output
f_in  = ROOT.TFile(o.input, "READ")
f_out = ROOT.TFile(output,  "RECREATE")

cuts=["all",
      "jetMultiplicity",
      "bTags",
      "DijetMass",
      "MDRs",
      "MDRs_SR",
      ]

triggers=["",
          "HLT"]

h2d = f_in.Get("cutflow/fourTag/truthM4b")
h2d.LabelsDeflate("Y")

for d in range(len(cuts)):
    denominator = cuts[d]
    b=h2d.GetYaxis().FindBin(denominator)
    hDenominator=h2d.ProjectionX(denominator, b, b)
    for n in range(d,len(cuts)):
        for trigger in triggers:
            numerator = cuts[n]+("_"+trigger if trigger else "")
            b=h2d.GetYaxis().FindBin(numerator)
            hNumerator=h2d.ProjectionX(numerator, b, b)
            hRatio=ROOT.TH1D(hNumerator)
            #hRatio.GetXaxis().SetAlphanumeric(0)
            hRatio.SetName(numerator+"_over_"+denominator)
            hRatio.Divide(hDenominator)
            hRatio.Write()

f_out.Close()
