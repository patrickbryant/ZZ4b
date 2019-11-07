import optparse
parser = optparse.OptionParser()
parser.add_option('-n', '--name',              dest="name",            help="Run in loop mode")
parser.add_option('-p', '--prefix',            dest="prefix",          help="Run in loop mode")
#parser.add_option(      '--period',            dest="period",          help="Run in loop mode")
parser.add_option('-d', '--dirWithPdfs',       dest="dirWithPdfs",     help="Run in loop mode")
parser.add_option('-t', '--tag',       default="four",     help="Run in loop mode")
#parser.add_option('--rocPlot',                  dest="rocPlot",          help="Run in loop mode")
#parser.add_option('--flavPlotDir',              help="Run in loop mode")
#parser.add_option('--effDir',                   help="Run in loop mode")
#parser.add_option('--mcAlgoDir',                   help="Run in loop mode")
#parser.add_option('--dataAlgoDir',                   help="Run in loop mode")
#parser.add_option('--doCaloJets',     action="store_true",              help="Run in loop mode")
o, a = parser.parse_args()


def makeHeader(outFile):
    outFile.write("\documentclass{beamer} \n")
    outFile.write("\mode<presentation>\n")
    outFile.write("\setbeamertemplate{footline}[frame number]\n")
    outFile.write("\\addtobeamertemplate{frametitle}{\\vspace*{0.4cm}}{\\vspace*{-0.4cm}}\n")
    outFile.write("{ \usetheme{boxes} }\n")
    outFile.write("\usepackage{times}  % fonts are up to you\n")
    outFile.write("\usefonttheme{serif}  % fonts are up to you\n")
    outFile.write("\usepackage{graphicx}\n")
    outFile.write("\usepackage{tikz}\n")
    outFile.write("\usepackage{colortbl}\n")
    outFile.write("\setlength{\pdfpagewidth}{2\paperwidth}\n")
    outFile.write("\setlength{\pdfpageheight}{2\paperheight}\n")
    outFile.write("\\title{\huge \\textcolor{myblue}{{4b Study:\\\\ \\textcolor{myblack}{\\textit{Online vs Offline}} }}}\n")
    outFile.write("\\author{\\textcolor{cmured}{{\Large \\\\John Alison, Patrick Bryant\\\\}}\n")
    outFile.write("  \\textit{\Large Carnegie Mellon University}\n")
    outFile.write("}\n")
    outFile.write("\date{  } \n")
    outFile.write("\n")
    outFile.write("\logo{\n")
    outFile.write("\\begin{picture}(10,8) %university_of_chicago_logo\n")
    #outFile.write("\put(-2.5,7.6){\includegraphics[height=0.5in]{CMSlogo_outline_black_label_May2014.pdf}}\n")
    outFile.write("\put(-2.5,7.6){\includegraphics[height=0.5in]{logos/CMSlogo_outline_black_red_nolabel_May2014.pdf}}\n")
    outFile.write("\put(8.2,7.7){\includegraphics[height=0.45in]{logos/CMU_Logo_Stack_Red-eps-converted-to.pdf}}\n")
    outFile.write("\end{picture}\n")
    outFile.write("}\n")
    outFile.write("\n")
    outFile.write("\\beamertemplatenavigationsymbolsempty\n")
    outFile.write("\n")
    outFile.write("\unitlength=1cm\n")
    outFile.write("\definecolor{myblue}{RGB}{33,100,158}\n")
    outFile.write("\definecolor{myblack}{RGB}{0,0,0}\n")
    outFile.write("\definecolor{myred}{RGB}{168,56,39}\n")
    outFile.write("\definecolor{cmured}{RGB}{173,29,53}\n")
    outFile.write("\definecolor{UCred}{RGB}{154,52,38}\n")
    outFile.write("\definecolor{mygreen}{RGB}{0,204,0}\n")
    outFile.write("\\begin{document}\n")
    outFile.write("\n")
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    outFile.write("\\titlepage\n")
    outFile.write("\end{frame}\n")
    


def make1x2(outFile,title,files,text,xText,yText,addLeg=False):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    outFile.write("\\frametitle{\centerline{ \huge \\textcolor{myblack}{"+title+"}}}  \n")
    outFile.write("\\begin{picture}(10,8) \n")

    # for the fig
    width = 2.6
    xStart = -1.1
    xOffSet = 6.4
    yStart = 0.4
    outFile.write("  \put("+str(xStart)+","+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[0]+".pdf}}\n")
    if len(files) > 1:
        outFile.write("  \put("+str(xStart+xOffSet)+" ,"+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[1]+".pdf}}\n")

    # for the text
    textHeight = 6
    for tItr in range(len(text)):
        outFile.write("  \put("+str(xText[tItr])+","+str(yText[tItr])+"){\\textcolor{myred}{\large "+text[tItr]+"}}\n")
    #if len(text) > 1:
    #    outFile.write("  \put("+str(xStart+xOffSet+1)+","+str(textHeight)+"){\\textcolor{myred}{\large "+text[1]+" }}\n")


    if addLeg:
        legText =  [("\\textcolor{red}{Offline BJets}",                      1.5,0.3),
                    ("\\textcolor{myblack}{Offline Light-Flavour}",          5.5,0.3),
                    ("\\textcolor{red}{HLT BJets}",                          1.5,-0.2),
                    ("\\textcolor{myblack}{HLT Light-Flavour}",              5.5,-0.2),
                    ("\\tikz\draw[red,very thick] (0.25,0.4) -- (0.75,0.4);",0.8,0.415),
                    ("\\tikz\draw[red,fill=red] (0,0) circle (.55ex);",      0.975,-0.175),
                    ("\\tikz\draw[red,very thick] (0.25,0.4) -- (0.75,0.4);",0.8,-0.085),
                    ("\\tikz\draw[black,very thick] (0.25,0.4) -- (0.75,0.4);",4.8,0.415),
                    ("\\tikz\draw[black,fill=black] (0,0) circle (.55ex);",    4.975,-0.175),
                    ("\\tikz\draw[black,very thick] (0.25,0.4) -- (0.75,0.4);",4.8,-0.085),
                    ]


        for legData in legText:
            outFile.write("  \put("+str(legData[1])+","+str(legData[2])+"){\\textcolor{myred}{\large "+legData[0]+"}}\n")


    outFile.write("\end{picture}\n")
    outFile.write("\end{frame}\n")


def make2x2(outFile,title,files):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    outFile.write("\\frametitle{\centerline{ \huge \\textcolor{myblack}{"+title+"}}}  \n")
    outFile.write("\\begin{picture}(10,8) \n")

    # for the fig
    width = 2.3
    xStart = -0.6
    xOffSet = 5.8
    yOffSet = 4.0
    yStart = -0.7
    outFile.write("  \put("+str(xStart)+","+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[2]+".pdf}}\n")
    outFile.write("  \put("+str(xStart+xOffSet)+" ,"+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[3]+".pdf}}\n")
    outFile.write("  \put("+str(xStart)+" ,"+str(yStart+yOffSet)+"){\includegraphics[width="+str(width)+"in]{"+files[0]+".pdf}}\n")
    outFile.write("  \put("+str(xStart+xOffSet)+" ,"+str(yStart+yOffSet)+"){\includegraphics[width="+str(width)+"in]{"+files[1]+".pdf}}\n")


    outFile.write("\end{picture}\n")
    outFile.write("\end{frame}\n")


def make2x2_ratio(outFile,title,files):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    outFile.write("\\begin{picture}(10,8) \n")

    # for the fig
    width = 1.8
    xStart = 0.45
    xOffSet = 5.0
    yOffSet = 4.2
    yStart = -0.8
    if len(files) > 0:
        outFile.write("  \put("+str(xStart)+" ,"+str(yStart+yOffSet)+"){\includegraphics[width="+str(width)+"in]{"+files[0]+".pdf}}\n")
    if len(files) > 1:
        outFile.write("  \put("+str(xStart+xOffSet)+" ,"+str(yStart+yOffSet)+"){\includegraphics[width="+str(width)+"in]{"+files[1]+".pdf}}\n")        
    if len(files) > 2:
        outFile.write("  \put("+str(xStart)+","+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[2]+".pdf}}\n")
    if len(files) > 3:
        outFile.write("  \put("+str(xStart+xOffSet)+" ,"+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[3]+".pdf}}\n")




    outFile.write("\end{picture}\n")
    outFile.write("\\frametitle{\centerline{\\textcolor{myblack}{"+title+"}}}  \n")
    outFile.write("\end{frame}\n")



def make2x3(outFile,title,files):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    outFile.write("\\frametitle{\centerline{ \huge \\textcolor{myblack}{"+title+"}}}  \n")
    outFile.write("\\begin{picture}(10,8) \n")

    # for the fig
    width = 1.75
    xStart = -1.0
    xOffSet = 4.25
    yOffSet = 4.0
    yStart = -0.8

    outFile.write("  \put("+str(xStart)+" ,"+str(yStart+yOffSet)+"){\includegraphics[width="+str(width)+"in]{"+files[0]+".pdf}}\n")
    outFile.write("  \put("+str(xStart+xOffSet)+" ,"+str(yStart+yOffSet)+"){\includegraphics[width="+str(width)+"in]{"+files[1]+".pdf}}\n")
    outFile.write("  \put("+str(xStart+xOffSet+xOffSet)+" ,"+str(yStart+yOffSet)+"){\includegraphics[width="+str(width)+"in]{"+files[2]+".pdf}}\n")

    if len(files) > 3:
        outFile.write("  \put("+str(xStart)+" ,"+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[3]+".pdf}}\n")
        outFile.write("  \put("+str(xStart+xOffSet)+" ,"+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[4]+".pdf}}\n")
        outFile.write("  \put("+str(xStart+xOffSet+xOffSet)+" ,"+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[5]+".pdf}}\n")




    outFile.write("\end{picture}\n")
    outFile.write("\end{frame}\n")


def make1x1(outFile,title,file,text,xText,yText):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    outFile.write("\\frametitle{\centerline{ \huge \\textcolor{myblack}{"+title+"}}}  \n")
    outFile.write("\\begin{picture}(10,8) \n")

    # for the fig
    width = 4.5
    xStart = -0.5
    yStart = -0.2
    outFile.write("  \put("+str(xStart)+","+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+file+"}}\n")


    # for the text
    #textHeight = 7.
    outFile.write("  \put("+str(xText[0])+","+str(yText[0])+"){"+text[0]+"}\n")

    outFile.write("\end{picture}\n")
    outFile.write("\end{frame}\n")

def makeWholeSlide(outFile,inputPDF):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    #outFile.write("\\frametitle{\centerline{ \huge \\textcolor{myblack}{TEst}}}  \n")
    outFile.write("\\begin{picture}(10,8) \n")

    outFile.write("  \put("+str(-1)+","+str(-1)+"){\includegraphics[width="+str(5)+"in]{"+inputPDF+"}}\n")


    outFile.write("\end{picture}\n")
    outFile.write("\end{frame}\n")



def make1x3(outFile,title,files,text):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    outFile.write("\\frametitle{\centerline{ \huge \\textcolor{myblack}{"+title+"}}}  \n")
    outFile.write("\\begin{picture}(10,8) \n")

    # for the fig
    width = 1.75
    xStart = -1.
    xOffSet = 4.25
    yStart = 1
    outFile.write("  \put("+str(xStart)+","+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[0]+".pdf}}\n")
    if len(files) > 1:
        outFile.write("  \put("+str(xStart+xOffSet)+" ,"+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[1]+".pdf}}\n")
    if len(files) > 2:
        outFile.write("  \put("+str(xStart+2*xOffSet)+" ,"+str(yStart)+"){\includegraphics[width="+str(width)+"in]{"+files[2]+".pdf}}\n")

    # for the text
    textHeight = 5.2
    textStart = -0.25
    outFile.write("  \put("+str(textStart)+","+str(textHeight)+"){\\textcolor{myred}{\large "+text[0]+"}}\n")
    if len(text) > 1:
        outFile.write("  \put("+str(textStart+xOffSet)+","+str(textHeight)+"){\\textcolor{myred}{\large "+text[1]+" }}\n")
    if len(text) > 2:
        outFile.write("  \put("+str(textStart+(2*xOffSet))+","+str(textHeight)+"){\\textcolor{myred}{\large "+text[2]+" }}\n")

    outFile.write("\end{picture}\n")
    outFile.write("\end{frame}\n")


def makeTransition(outFile,text,doHuge=True):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    #outFile.write("\\frametitle{\centerline{ \huge \\textcolor{myblack}{"+title+"}}}  \n")
    outFile.write("\\begin{picture}(10,8) \n")

    # for the text
    textHeight = 4
    if doHuge:
        outFile.write("  \put("+str(0)+","+str(textHeight)+"){\\textcolor{myred}{\Huge \\textit{"+text+"}}}\n")
    else:
        outFile.write("  \put("+str(0)+","+str(textHeight)+"){\\textcolor{myred}{\Large \\textit{"+text+"}}}\n")
    outFile.write("\end{picture}\n")
    outFile.write("\end{frame}\n")


def makeText(outFile,text,title=""):
    outFile.write("\n")
    outFile.write("\\begin{frame}\n")
    outFile.write("\\frametitle{\centerline{ \huge \\textcolor{myblack}{"+title+"}}}  \n")
    for t in text:
        outFile.write(t+"\n")
    outFile.write("\end{frame}\n")





def makePresentation():

    outFile = open(o.name+".tex","w")
    makeHeader(outFile)

    pdfDir = o.dirWithPdfs
    prefix = o.prefix 

#    text = ["\\textcolor{cmured}{\large Looking at HLT Btagging in 2017 ttbar MC and Data 2018D}\\\\"]
#    if o.doCaloJets:
#        text+= ["\\textcolor{cmured}{\large Focus on Calo-Jets (w/'hltFastPixelBLifetimeL3Associator' tracks)}\\\\"]
#    else:
#        text+= ["\\textcolor{cmured}{\large Focus on PF-Jets (w/'hltParticleFlow' tracks)}\\\\"]
#
#    text += ["\\vspace*{0.2in}",
#             "\\textcolor{myblue}{\large Event Selection:}",
#             "\\begin{itemize}",
#             "\\item[\\textcolor{myblack}{-}]Require Electron+Muon Trigger\\\\"
#             "\\item[\\textcolor{myblack}{-}]Two Tight-leptons\\\\"
#             "\\end{itemize}",
#             "\\textcolor{myblue}{\large Jet Selection:}",
#             "\\begin{itemize}",
#             "\\item[\\textcolor{myblack}{-}]Overlap removal with tight leptons\\\\"
#             "\\item[\\textcolor{myblack}{-}]$|\eta| < 2.5 / P_T > 35$ GeV \\\\"
#             "\\item[\\textcolor{myblack}{-}]Study tracks in jets used for btagging:\\\\ \\textit{focus on offline/online differences}  \\\\"
#             "\\end{itemize}"]
    

    tag = o.tag
    
    #"inclusive"
    for reg in ["SCSR"]:#,"Inclusive"]:#,"SB","CR","SR"]:

        #
        #  Event Level Vars
        #

        for slideConfig in [("hT","nAllJets","dRBB","dBB"),
                            ("xHH","stNotCan","nPVs","xWt"),
                            ]:
            files = []
            for i in range(4):
                if slideConfig[i]:
                    files += [pdfDir+"/passMDRs_"+tag+"Tag_mainView_"+reg+"_"+slideConfig[i]]
            make2x2_ratio(outFile,reg+" / PassMDRs / Event-Level",
                          files = files,
                          )


        #
        #  Four-Jet Level Vars
        #
        for slideConfig in [("pt_m","eta","phi","m_l"),
                            ]:
            files = []
            for i in range(4):
                if slideConfig[i]:
                    files += [pdfDir+"/passMDRs_"+tag+"Tag_mainView_"+reg+"_v4j_"+slideConfig[i]]
            make2x2_ratio(outFile,reg+" / PassMDRs / 4-b system",
                          files = files,
                          )


        #
        #  di-Jet Level Vars
        #
        for dJ in ["leadM","sublM","close","other"]:
            
            for slideConfig in [("pt_m","eta","phi","m"),
                                ("dR","pz_l","",""),
                                ]:
                files = []
                for i in range(4):
                    if slideConfig[i]: files += [pdfDir+"/passMDRs_"+tag+"Tag_mainView_"+reg+"_"+dJ+"_"+slideConfig[i]]
                make2x2_ratio(outFile,reg+" / PassMDRs / "+dJ+"-jj",
                              files = files
                              )


        #
        #  Jet-Level Vars
        #
        for jV in ["canJet0","canJet1","canJet2","canJet3","selJets","othJets","allNotCanJets"]:
            
            for slideConfig in [("pt_m","eta","phi","m_s"),
                                ("deepFlavB","CSVv2_l","",""),
                                ]:
                files = []
                for i in range(4):
                    if slideConfig[i]: files += [pdfDir+"/passMDRs_"+tag+"Tag_mainView_"+reg+"_"+jV+"_"+slideConfig[i]]
                make2x2_ratio(outFile,reg+" / PassMDRs / "+jV,
                              files = files
                              )



                


    outFile.write("\n")
    outFile.write("\end{document}\n")



if __name__ == "__main__":
    makePresentation()
    import os
    os.system("pdflatex "+o.name+".tex")
    
    os.system("rm "+o.name+".out")
    os.system("rm "+o.name+".toc")
    os.system("rm "+o.name+".snm")
    os.system("rm "+o.name+".aux")
    os.system("rm "+o.name+".nav")
    os.system("rm "+o.name+".log")
