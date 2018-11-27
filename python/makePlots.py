import sys
import collections
sys.path.insert(0, '../PlotTools/python/') #https://github.com/patrickbryant/PlotTools                                                                                                                                                     
import PlotTools

import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--inFileName',           dest="infileName",         default=None, help="Run in loop mode")
parser.add_option('-o', '--outFileName',          dest="outfileName",        default="../run/plots/", help="Run in loop mode")
parser.add_option('-d', '--debug',                dest="debug",    action="store_true",       default=False, help="Run in loop mode")
o, a = parser.parse_args()

zz="hists/mg5_ZZ4b_onshell_run_02.root"
zh="hists/mg5_ZH4b_onshell_run_02.root"
zz_zh="hists/mg5_ZZ+ZH4b_onshell_run_02.root"
atlas="resolved_4bSR_2016_PassHCdEta.root"


samples=collections.OrderedDict()
samples[zz] = collections.OrderedDict()
samples[zz]["allEvents/mbs"] = {"label"    : "all",
                                "ratio"    : "denom A",
                                "legend"   : 1,
                                "color"    : "ROOT.kBlack"}
samples[zz]["passPreSel/truth/mbs"] = {"label"    : "Preselection",
                                       "ratio"    : "numer A",
                                       "legend"   : 2,
                                       "color"    : "ROOT.kMagenta"}
samples[zz]["passMDRs/truth/mbs"] = {"label"    : "MDRs",
                                     "ratio"    : "numer A",
                                     "legend"   : 3,
                                     "color"    : "ROOT.kBlue"}
samples[zz]["passMDCs/truth/mbs"] = {"label"    : "MDCs",
                                     "ratio"    : "numer A",
                                     "legend"   : 4,
                                     "color"    : "ROOT.kGreen"}
samples[zz]["passHCdEta/truth/mbs"] = {"label"    : "|#Delta#eta|",
                                       "ratio"    : "numer A",
                                       "legend"   : 5,
                                       "color"    : "ROOT.kOrange"}
samples[zz]["passHCdEta/mainView/ZZ/truth/mbs"] = {"label"    : "X_{ZZ}",
                                                   "ratio"    : "numer A",
                                                   "legend"   : 6,
                                                   "color"    : "ROOT.kRed"}
parameters = {"ratio"     : True,
              #"maxDigits" : 4,
              "logY"      : True,
              "yMin"      : 2e-3,
              "yMax"      : 2e3,
              "rebin"     : [180, 220, 260, 320, 400, 500, 600, 800, 1200],
              "rTitle"    : "A#times#epsilon",
              "drawOptions": "HIST C SAME",
              "padDivide" : 0.55,
              "xTitleOffset": 1.6,
              "bMarginRatio": 0.16,
              "rMax"      : 0.04,
              "rMin"      : 0,
              "yTitle"    : "Events / Bin",
              "errors"    : False,
              "outputDir" : o.outfileName,
              "outputName": "ZZ_efficiency"}
PlotTools.plot(samples, parameters, o.debug)

samples=collections.OrderedDict()
samples[zh] = collections.OrderedDict()
samples[zh]["allEvents/mbs"] = {"label"    : "all",
                                "ratio"    : "denom A",
                                "legend"   : 1,
                                "color"    : "ROOT.kBlack"}
samples[zh]["passPreSel/truth/mbs"] = {"label"    : "Preselection",
                                       "ratio"    : "numer A",
                                       "legend"   : 2,
                                       "color"    : "ROOT.kMagenta"}
samples[zh]["passMDRs/truth/mbs"] = {"label"    : "MDRs",
                                     "ratio"    : "numer A",
                                     "legend"   : 3,
                                     "color"    : "ROOT.kBlue"}
samples[zh]["passMDCs/truth/mbs"] = {"label"    : "MDCs",
                                     "ratio"    : "numer A",
                                     "legend"   : 4,
                                     "color"    : "ROOT.kGreen"}
samples[zh]["passHCdEta/truth/mbs"] = {"label"    : "|#Delta#eta|",
                                       "ratio"    : "numer A",
                                       "legend"   : 5,
                                       "color"    : "ROOT.kOrange"}
parameters = {"ratio"     : True,
              #"maxDigits" : 4,
              "logY"      : True,
              "yMin"      : 5e-4,
              "yMax"      : 2e3,
              "rebin"     : [215, 260, 320, 400, 500, 600, 800, 1200],
              "rTitle"    : "A#times#epsilon",
              "drawOptions": "HIST C SAME",
              "padDivide" : 0.55,
              "xTitleOffset": 1.6,
              "bMarginRatio": 0.16,
              "rMax"      : 0.2,
              "rMin"      : 0,
              "yTitle"    : "Events / Bin",
              "errors"    : False,
              "outputDir" : o.outfileName,
              "outputName": "ZH_efficiency"}
PlotTools.plot(samples, parameters, o.debug)


samples=collections.OrderedDict()
samples[atlas] = collections.OrderedDict()
samples[zz] = collections.OrderedDict()
samples[zh] = collections.OrderedDict()
samples[zz_zh] = collections.OrderedDict()
samples[atlas]["data_LM"] = {"label" : "Data",
                             "legend": 1,
                             "isData" : True,
                             "ratio" : "numer A",
                             "color" : "ROOT.kBlack"}
samples[atlas]["nonallhad_LM"] = {"label" : "Semi-leptonic t#bar{t}",
                                  "legend": 4,
                                  "stack" : 1,
                                  "ratio" : "denom A",
                                  "color" : "ROOT.kAzure-4"}
samples[atlas]["allhad_LM"] = {"label" : "Hadronic t#bar{t}",
                               "legend": 3,
                               "stack" : 2,
                               "ratio" : "denom A",
                               "color" : "ROOT.kAzure-9"}
samples[atlas]["qcd_LM"] = {"label" : "Multijet",
                            "legend": 2,
                            "stack" : 3,
                            "ratio" : "denom A",
                            "color" : "ROOT.kYellow"}
samples[zz]["passHCdEta/mainView/ZZ/m4j_cor_Z_f"] = {"label"    : "ZZ",
                                                      "legend"   : 5,
                                                      "stack"    : 5,
                                                      "color"    : "ROOT.kGreen+2"}
samples[zh]["passHCdEta/mainView/ZZ/m4j_cor_Z_f"] = {"label"    : "ZH",
                                                      "legend"   : 6,
                                                      "stack"    : 4,
                                                      "color"    : "ROOT.kPink"}
samples[zz_zh]["passHCdEta/mainView/ZZ/m4j_cor_Z_f"] = {"label"    : "(ZZ+ZH) #times 10",
                                                        "weight"    : 10,
                                                        "legend"   : 7,
                                                        "color"    : "ROOT.kGreen+2"}
parameters = {"ratio"     : True,
              "rTitle"    : "Data / Bkgd",
              "xTitle"    : "Bin",
              "yTitle"    : "Events",
              "outputDir" : o.outfileName+"/results",
              "outputName": "mZZ_f"}
PlotTools.plot(samples, parameters, o.debug)

samples=collections.OrderedDict()
samples[atlas] = collections.OrderedDict()
samples[zz] = collections.OrderedDict()
samples[zh] = collections.OrderedDict()
samples[zz_zh] = collections.OrderedDict()
samples[atlas]["data_LM_v"] = {"label" : "Data",
                             "legend": 1,
                             "isData" : True,
                             "ratio" : "numer A",
                             "color" : "ROOT.kBlack"}
samples[atlas]["nonallhad_LM_v"] = {"label" : "Semi-leptonic t#bar{t}",
                                  "legend": 4,
                                  "stack" : 1,
                                  "ratio" : "denom A",
                                  "color" : "ROOT.kAzure-4"}
samples[atlas]["allhad_LM_v"] = {"label" : "Hadronic t#bar{t}",
                               "legend": 3,
                               "stack" : 2,
                               "ratio" : "denom A",
                               "color" : "ROOT.kAzure-9"}
samples[atlas]["qcd_LM_v"] = {"label" : "Multijet",
                            "legend": 2,
                            "stack" : 3,
                            "ratio" : "denom A",
                            "color" : "ROOT.kYellow"}
samples[zz]["passHCdEta/mainView/ZZ/m4j_cor_Z_v"] = {"label"    : "ZZ",
                                                      "legend"   : 5,
                                                      "stack"    : 5,
                                                      "color"    : "ROOT.kGreen+2"}
samples[zh]["passHCdEta/mainView/ZZ/m4j_cor_Z_v"] = {"label"    : "ZH",
                                                      "legend"   : 6,
                                                      "stack"    : 4,
                                                      "color"    : "ROOT.kPink"}
samples[zz_zh]["passHCdEta/mainView/ZZ/m4j_cor_Z_v"] = {"label"    : "(ZZ+ZH) #times 10",
                                                        "weight"    : 10,
                                                        "legend"   : 7,
                                                        "color"    : "ROOT.kGreen+2"}
parameters = {"ratio"     : True,
              "rTitle"    : "Data / Bkgd",
              "divideByBinWidth": True,
              "xTitle"    : "m_{ZZ} [GeV]",
              "yTitle"    : "Events",
              "xMax"      : 1115,
              "logX"      : True,
              "outputDir" : o.outfileName+"/results",
              "outputName": "mZZ_v"}
PlotTools.plot(samples, parameters, o.debug)


cuts=["passPreSel",
      "passMDRs",
      "passMDCs",
      "passHCdEta",
      "passTopVeto",
]
regions=["inclusive",
         "ZZ",
]
truthVars=["mbs",
           "bosons_m",
           "bosons_pt",
           "bosons_eta",
           "bosons_phi",
           "bosons_PID",
]
eventVars=["m4j",
           "xWt",
]
views=["allViews",
       "mainView",
]
viewVars=["xZZ",
          "mZZ",
          "m4j_cor_Z_v",
          "m4j_cor_Z_f",
          "dEta",
          "xWt",
]
diJets=["lead",   "subl",
        "leadSt", "sublSt"
]
diJetVars=["_m",
           "_dR",
           "_pt",
           "_st",
           "_eta",
           "_phi",
           "_lead_m",
           "_lead_pt",
           "_lead_eta",
           "_lead_phi",
           "_subl_m",
           "_subl_pt",
           "_subl_eta",
           "_subl_phi",
]
TH2ViewVars=["leadSt_m_vs_sublSt_m",
             "m4j_vs_leadStdR",
             "m4j_vs_sublStdR",
             "m4j_vs_nViews",
]
TH2EventVars=["m4j_vs_nViews",
]

for cut in cuts:
    #plot event vars
    for var in eventVars:
        samples=collections.OrderedDict()
        samples[zz] = collections.OrderedDict()
        samples[zh] = collections.OrderedDict()
        samples[zz_zh] = collections.OrderedDict()
        samples[zz_zh][cut+"/"+var] = {"label"    : "ZZ+ZH",
                                       "legend"   : 1,
                                       "isData"   : True,
                                       "ratio"    : "numer A",
                                       "color"    : "ROOT.kBlack"}    
        samples[zz][cut+"/"+var] = {"label"    : "ZZ",
                                    "ratio"    : "denom A",
                                    "stack"    : 2,
                                    "legend"   : 2,
                                    "color"    : "ROOT.kAzure+6"}
        samples[zh][cut+"/"+var] = {"label"    : "ZH",
                                    "ratio"    : "denom A",
                                    "stack"    : 1,
                                    "legend"   : 3,
                                    "color"    : "ROOT.kPink"}
        parameters = {"ratio"     : True,
                      #"maxDigits" : 4,
                      "rTitle"    : "ZZ+ZH / Stack",
                      "outputDir" : o.outfileName+"/"+cut,
                      "outputName": var}

        PlotTools.plot(samples, parameters, o.debug)


        #2D event hists
        for var in TH2EventVars:
            for sample in [zz, zh, zz_zh]:
                samples=collections.OrderedDict()
                samples[sample] = collections.OrderedDict()
                samples[sample][cut+"/"] = {"TObject":var,
                                            "drawOptions": "COLZ",
                }
                name = var+"_"
                if "ZZ+ZH" in sample: name=name+"ZZ+ZH"
                elif  "ZZ" in sample: name=name+"ZZ"
                elif  "ZH" in sample: name=name+"ZH"
                parameters = {"outputDir" : o.outfileName+"/"+cut+"/",
                              "lMargin" : 0.12,
                              "bMargin" : 0.10,
                              "canvasSize" : [600,560],
                              "xTitleOffset": 0.9,
                              "outputName": name}
                if "nViews" in var:
                    parameters["setNdivisionsY"] = 103
                    parameters["yMax"] = 3.5
                    parameters["yMin"] = 0.5
                PlotTools.plot(samples, parameters, o.debug)
        
        
    for var in truthVars:
        samples=collections.OrderedDict()
        samples[zz] = collections.OrderedDict()
        samples[zh] = collections.OrderedDict()
        samples[zz_zh] = collections.OrderedDict()
        samples[zz_zh][cut+"/truth/"+var] = {"label"    : "ZZ+ZH",
                                             "legend"   : 1,
                                             "isData"   : True,
                                             "ratio"    : "numer A",
                                             "color"    : "ROOT.kBlack"}    
        samples[zz][cut+"/truth/"+var] = {"label"    : "ZZ",
                                          "ratio"    : "denom A",
                                          "stack"    : 2,
                                          "legend"   : 2,
                                          "color"    : "ROOT.kAzure+6"}
        samples[zh][cut+"/truth/"+var] = {"label"    : "ZH",
                                          "ratio"    : "denom A",
                                          "stack"    : 1,
                                          "legend"   : 3,
                                          "color"    : "ROOT.kPink"}
        parameters = {"ratio"     : True,
                      #"maxDigits" : 4,
                      "rTitle"    : "ZZ+ZH / Stack",
                      "outputDir" : o.outfileName+"/"+cut+"/truth/",
                      "outputName": var}
        PlotTools.plot(samples, parameters, o.debug)

        
    for view in views:
        for region in regions:
            #plot view vars
            for var in viewVars:
                samples=collections.OrderedDict()
                samples[zz] = collections.OrderedDict()
                samples[zh] = collections.OrderedDict()
                samples[zz_zh] = collections.OrderedDict()
                samples[zz_zh][cut+"/"+view+"/"+region+"/"+var] = {"label"    : "ZZ+ZH",
                                                                   "legend"   : 1,
                                                                   "isData"   : True,
                                                                   "ratio"    : "numer A",
                                                                   "color"    : "ROOT.kBlack"}    
                samples[zz][cut+"/"+view+"/"+region+"/"+var] = {"label"    : "ZZ",
                                                                "ratio"    : "denom A",
                                                                "stack"    : 2,
                                                                "legend"   : 2,
                                                                "color"    : "ROOT.kAzure+6"}
                samples[zh][cut+"/"+view+"/"+region+"/"+var] = {"label"    : "ZH",
                                                                "ratio"    : "denom A",
                                                                "stack"    : 1,
                                                                "legend"   : 3,
                                                                "color"    : "ROOT.kPink"}
                parameters = {"ratio"     : True,
                              #"maxDigits" : 4,
                              "rTitle"    : "ZZ+ZH / Stack",
                              "outputDir" : o.outfileName+"/"+cut+"/"+view+"/"+region,
                              "outputName": var}

                PlotTools.plot(samples, parameters, o.debug)

            #truth vars in given view selection
            if region in ["ZZ"]:
                for var in truthVars:
                    samples=collections.OrderedDict()
                    samples[zz] = collections.OrderedDict()
                    samples[zh] = collections.OrderedDict()
                    samples[zz_zh] = collections.OrderedDict()
                    samples[zz_zh][cut+"/"+view+"/"+region+"/truth/"+var] = {"label"    : "ZZ+ZH",
                                                                             "legend"   : 1,
                                                                             "isData"   : True,
                                                                             "ratio"    : "numer A",
                                                                             "color"    : "ROOT.kBlack"}    
                    samples[zz][cut+"/"+view+"/"+region+"/truth/"+var] = {"label"    : "ZZ",
                                                                          "ratio"    : "denom A",
                                                                          "stack"    : 2,
                                                                          "legend"   : 2,
                                                                          "color"    : "ROOT.kAzure+6"}
                    samples[zh][cut+"/"+view+"/"+region+"/truth/"+var] = {"label"    : "ZH",
                                                                          "ratio"    : "denom A",
                                                                          "stack"    : 1,
                                                                          "legend"   : 3,
                                                                          "color"    : "ROOT.kPink"}
                    parameters = {"ratio"     : True,
                                  #"maxDigits" : 4,
                                  "rTitle"    : "ZZ+ZH / Stack",
                                  "outputDir" : o.outfileName+"/"+cut+"/"+view+"/"+region+"/truth",
                                  "outputName": var}

                    PlotTools.plot(samples, parameters, o.debug)
                
                
            #2D view hists
            for var in TH2ViewVars:
                for sample in [zz, zh, zz_zh]:
                    samples=collections.OrderedDict()
                    samples[sample] = collections.OrderedDict()
                    samples[sample][cut+"/"+view+"/"+region+"/"] = {"TObject":var,
                                                                    "drawOptions": "COLZ",
                    }
                    name = var+"_"
                    if "ZZ+ZH" in sample: name=name+"ZZ+ZH"
                    elif  "ZZ" in sample: name=name+"ZZ"
                    elif  "ZH" in sample: name=name+"ZH"
                    parameters = {"outputDir" : o.outfileName+"/"+cut+"/"+view+"/"+region,
                                  "lMargin" : 0.12,
                                  "bMargin" : 0.10,
                                  "canvasSize" : [600,560],
                                  "xTitleOffset": 0.9,
                                  "outputName": name}
                    if "_m_" in var:
                        parameters["setNdivisionsY"] = 505
                        parameters["yTitleOffset"]   = 1.15
                    if "m4j_vs_leadStdR" in var:
                        parameters["functions"] = [["(360/x-0.500 - y)",100,1200,0,4,[0],"ROOT.kRed",1],
                                                   ["(653/x+0.475 - y)",100,1200,0,4,[0],"ROOT.kRed",1]]
                    if "m4j_vs_sublStdR" in var:
                        parameters["functions"] = [["(235/x       - y)",100,1200,0,4,[0],"ROOT.kRed",1],
                                                   ["(875/x+0.350 - y)",100,1200,0,4,[0],"ROOT.kRed",1]]
                    if "nViews" in var:
                        parameters["setNdivisionsY"] = 103
                        parameters["yMax"] = 3.5
                        parameters["yMin"] = 0.5
                    PlotTools.plot(samples, parameters, o.debug)
                

            #plot diJet vars
            for diJet in diJets:
                for var in diJetVars:
                    samples=collections.OrderedDict()
                    samples[zz] = collections.OrderedDict()
                    samples[zh] = collections.OrderedDict()
                    samples[zz_zh] = collections.OrderedDict()
                    samples[zz_zh][cut+"/"+view+"/"+region+"/"+diJet+var] = {"label"    : "ZZ+ZH",
                                                                             "legend"   : 1,
                                                                             "isData"   : True,
                                                                             "ratio"    : "numer A",
                                                                             "color"    : "ROOT.kBlack"}    
                    samples[zz][cut+"/"+view+"/"+region+"/"+diJet+var] = {"label"    : "ZZ",
                                                                          "ratio"    : "denom A",
                                                                          "stack"    : 2,
                                                                          "legend"   : 2,
                                                                          "color"    : "ROOT.kAzure+6"}
                    samples[zh][cut+"/"+view+"/"+region+"/"+diJet+var] = {"label"    : "ZH",
                                                                          "ratio"    : "denom A",
                                                                          "stack"    : 1,
                                                                          "legend"   : 3,
                                                                          "color"    : "ROOT.kPink"}
                    parameters = {"ratio"     : True,
                                  #"maxDigits" : 4,
                                  "rTitle"    : "ZZ+ZH / Stack",
                                  "outputDir" : o.outfileName+"/"+cut+"/"+view+"/"+region+"/",
                                  "outputName": diJet+var}

                    PlotTools.plot(samples, parameters, o.debug)


                    
