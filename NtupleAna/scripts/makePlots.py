import sys
import collections
sys.path.insert(0, 'PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import PlotTools
import optparse

parser = optparse.OptionParser()
parser.add_option('-d', '--debug',                dest="debug",         action="store_true", default=False, help="debug")
parser.add_option('-y', '--year',                 dest="year",          default="2018", help="Year specifies trigger (and lumiMask for data)")
parser.add_option('-l', '--lumi',                 dest="lumi",          default="1",    help="Luminosity for MC normalization: units [pb]")
parser.add_option('-o', '--outputBase',           dest="outputBase",    default="/uscms/home/bryantp/nobackup/ZZ4b/", help="Base path for storing output histograms and picoAOD")
o, a = parser.parse_args()

#make sure outputBase ends with /
outputBase = o.outputBase + ("" if o.outputBase[-1] == "/" else "/")
print "Plot output:",outputBase+"plots"

files = {"data2018A": outputBase+"data2018A/hists.root",
         "ZH4b2018" : outputBase+"ZH4b2018/hists.root",
         }


class nameTitle:
    def __init__(self, name, title):
        self.name  = name
        self.title = title

cuts = [nameTitle("passMDCs", "Pass MDC's"), nameTitle("passDEtaBB", "|#Delta#eta| > 1.5")]
views = ["allViews","mainView"]
regions = [nameTitle("ZHSB", "ZH Sideband"), nameTitle("ZHCR", "ZH Control Region"), nameTitle("ZHSR", "ZH Signal Region")]

class variable:
    def __init__(self, name, xTitle, rebin = None, divideByBinWidth = False):
        self.name   = name
        self.xTitle = xTitle
        self.rebin  = rebin
        self.divideByBinWidth = divideByBinWidth

class standardPlot:
    def __init__(self, year, cut, view, region, var):
        self.samples=collections.OrderedDict()
        self.samples[files["data"+year+"A"]] = collections.OrderedDict()
        self.samples[files["ZH4b"+year    ]] = collections.OrderedDict()
        self.samples[files["data"+year+"A"]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label" : "Data 14.0/fb, 2018A",
                                                                                                   "legend": 1,
                                                                                                   "isData" : True,
                                                                                                   "ratio" : "numer A",
                                                                                                   "color" : "ROOT.kBlack"}
        self.samples[files["data"+year+"A"]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {"label" : "Multijet",
                                                                                                    "legend": 2,
                                                                                                    "stack" : 3,
                                                                                                    "ratio" : "denom A",
                                                                                                    "color" : "ROOT.kYellow"}
        self.samples[files["ZH4b"+year    ]][cut.name+"/threeTag/"+view+"/"+region.name+"/"+var.name] = {"label"    : "ZH#rightarrowb#bar{b}b#bar{b} (3-Tag)",
                                                                                                    "legend"   : 3,
                                                                                                    "stack"    : 4,
                                                                                                    "color"    : "ROOT.kRed"}
        self.samples[files["ZH4b"+year    ]][cut.name+"/fourTag/"+view+"/"+region.name+"/"+var.name] = {"label"    : "ZH#rightarrowb#bar{b}b#bar{b} (4-Tag)",
                                                                                                   "legend"   : 4,
                                                                                                   "color"    : "ROOT.kRed"}

        self.parameters = {"titleLeft"   : "#bf{CMS} Internal",
                           "titleCenter" : region.title,
                           "titleRight"  : cut.title,
                           "ratio"     : True,
                           "rTitle"    : "Data / Bkgd.",
                           "xTitle"    : var.xTitle,
                           "yTitle"    : "Events",
                           "outputDir" : outputBase+"plots/"+year+"/"+cut.name+"/"+view+"/"+region.name+"/",
                           "outputName": var.name}
        if var.divideByBinWidth: self.parameters["divideByBinWidth"] = True
        if var.rebin: self.parameters["rebin"] = var.rebin

    def plot(self):
        PlotTools.plot(self.samples, self.parameters, o.debug)


variables=[variable("nSelJets", "Number of Selected Jets"),
           variable("xZH", "x_{ZH}"),
           variable("mZH", "m_{ZH} [GeV]", divideByBinWidth = True),
           variable("dBB", "D_{BB} [GeV]"),
           variable("v4j/m_l", "m_{4j} [GeV]"),
           variable("v4j/pt_l", "p_{T,4j} [GeV]"),
           variable("leadSt/m", "Leading S_{T} Dijet Mass [GeV]"),
           variable("leadSt/dR", "Leading S_{T} Dijet #DeltaR(j,j)"),
           variable("leadSt/pt_m", "Leading S_{T} Dijet p_{T} [GeV]"),
           variable("leadSt/eta", "Leading S_{T} Dijet #eta"),
           variable("leadSt/phi", "Leading S_{T} Dijet #phi"),
           variable("sublSt/m", "Subleading S_{T} Dijet Mass [GeV]"),
           variable("sublSt/dR", "Subleading S_{T} Dijet #DeltaR(j,j)"),
           variable("sublSt/pt_m", "Subleading S_{T} Dijet p_{T} [GeV]"),
           variable("sublSt/eta", "Subleading S_{T} Dijet #eta"),
           variable("sublSt/phi", "Subleading S_{T} Dijet #phi"),
           variable("leadM/m", "Leading Mass Dijet Mass [GeV]"),
           variable("leadM/dR", "Leading Mass Dijet #DeltaR(j,j)"),
           variable("leadM/pt_m", "Leading Mass Dijet p_{T} [GeV]"),
           variable("leadM/eta", "Leading Mass Dijet #eta"),
           variable("leadM/phi", "Leading Mass Dijet #phi"),
           variable("sublM/m", "Subleading Mass Dijet Mass [GeV]"),
           variable("sublM/dR", "Subleading Mass Dijet #DeltaR(j,j)"),
           variable("sublM/pt_m", "Subleading Mass Dijet p_{T} [GeV]"),
           variable("sublM/eta", "Subleading Mass Dijet #eta"),
           variable("sublM/phi", "Subleading Mass Dijet #phi"),
           variable("lead/m", "Leading S_{T} Dijet Mass [GeV]"),
           variable("lead/dR", "Leading p_{T} Dijet #DeltaR(j,j)"),
           variable("lead/pt_m", "Leading p_{T} Dijet p_{T} [GeV]"),
           variable("lead/eta", "Leading p_{T} Dijet #eta"),
           variable("lead/phi", "Leading p_{T} Dijet #phi"),
           variable("subl/m", "Subleading p_{T} Dijet Mass [GeV]"),
           variable("subl/dR", "Subleading p_{T} Dijet #DeltaR(j,j)"),
           variable("subl/pt_m", "Subleading p_{T} Dijet p_{T} [GeV]"),
           variable("subl/eta", "Subleading p_{T} Dijet #eta"),
           variable("subl/phi", "Subleading p_{T} Dijet #phi"),
           ]

for cut in cuts:
    for view in views:
        for region in regions:
            for variable in variables:
                p=standardPlot(o.year, cut, view, region, variable)
                p.plot()

