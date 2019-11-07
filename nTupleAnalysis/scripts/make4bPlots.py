#from iPlotLoadPath import loadPath
#from iPlot import loadPath
#loadPath()

from iUtils import getPM, setBatch, plot
setBatch()


from iUtils import parseOpts as parseOpts

(o,a) = parseOpts()
pm = getPM(o)



diJetVars = [
    "dR",
    "e",
    "eta",
    "m",
    "m_l",
    "phi",
    "pt_l",
    "pt_m",
    "pt_s",
    "pz_l",
    ]


eventVars = [
    "hT",
    "stNotCan",
    "dRBB",
    "dBB",
    "xHH",
    "xWt",
    ]

eventVars_noRebin = [
    "nAllJets",
    "nPVs",
]

fourJetVars = [
    "pt_m",
    "eta",
    "phi",
    "m_l",
]

jetVars = [
    "pt_m",
    "eta",
    "phi",
    "m_s",
    "deepFlavB",
    "CSVv2_l",
]


def doCutPoint(cpName):
    
    for tag in ["threeTag","fourTag"]:

        for view in ["mainView","allViews"]:

            for reg in ["inclusive","SCSR","SB","CR","SR"]:
                
                #
                # Event Vars
                #
                for eV in eventVars:
                    plot(eV,cpName+"/"+tag+"/"+view+"/"+reg,logy=0,doratio=1,rMin=0.8,rMax=1.2,norm=0,rebin=2)

                for eV in eventVars_noRebin:
                    plot(eV,cpName+"/"+tag+"/"+view+"/"+reg,logy=0,doratio=1,rMin=0.8,rMax=1.2,norm=0)

                #
                # Four-Jet level Vars
                #
                for fourJV in fourJetVars:
                    try:
                        plot(fourJV,cpName+"/"+tag+"/"+view+"/"+reg+"/v4j",logy=0,doratio=1,rMin=0.8,rMax=1.2,norm=0,rebin=2)
                    except:
                        print "ERROR with",fourJV,cpName+"/"+tag+"/"+view+"/"+reg+"/v4j"
                    
                    
                #
                # di-Jet level Vars
                #
                for dJet in ["leadM","sublM","other","close"]:
                    for dv in diJetVars:
                        plot(dv,cpName+"/"+tag+"/"+view+"/"+reg+"/"+dJet,logy=0,doratio=1,rMin=0.8,rMax=1.2,norm=0,rebin=2)


                #
                # Jet-level Vars
                #
                for cJet in ["canJet0","canJet1","canJet2","canJet3","othJets","allNotCanJets","selJets"]:
                    for jv in jetVars:
                        plot(jv,cpName+"/"+tag+"/"+view+"/"+reg+"/"+cJet,logy=0,doratio=1,rMin=0.8,rMax=1.2,norm=0,rebin=2)



cutPoint = [
    "passMDRs",
    #"passDijetMass",
    ]

for cp in cutPoint:
    doCutPoint(cp)
