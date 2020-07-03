import optparse
parser = optparse.OptionParser()
parser.add_option('--d4',     help="")
parser.add_option('--d3',     help="")
parser.add_option('--t4',     help="")
parser.add_option('--t3',     help="")
parser.add_option('--t4_s',   default = None,help="")
parser.add_option('--t4_h',   default = None,help="")
parser.add_option('--t4_d',   default = None,help="")
parser.add_option('--t3_s',   default = None,help="")
parser.add_option('--t3_h',   default = None,help="")
parser.add_option('--t3_d',   default = None,help="")
parser.add_option('-r',            action="store_true", dest="reweight",       default=False, help="make cutflow after reweighting by FvTWeight")
parser.add_option('-d', '--debug', action="store_true",    help="")
parser.add_option('--makePDF', action="store_true",    help="")
parser.add_option('--name', default="table",    help="")
#parser.add_option('--cut',  default=""all","HLT","jetMultiplicity", "bTags", "bTags_HLT", "passPreSel", "passDijetMass", "passMDRs", "passXWt"",  help="")
parser.add_option('--cuts',  default="all,HLT,jetMultiplicity,bTags,bTags_HLT,passPreSel,passDijetMass,passMDRs,passXWt",  help="Comma separate list of cuts. Default is: \n"+"all,HLT,jetMultiplicity,bTags,bTags_HLT,passPreSel,passDijetMass,passMDRs,passXWt\n")
o, a = parser.parse_args()

reweight = o.reweight

import ROOT

def small(inputStr):
    return "{\\small "+inputStr+"}"

def bold(inputStr):
    return "\\textbf{"+inputStr+"}"

def cleanLatex(inputStr):
    return inputStr.replace("\\textbf{","").replace("{\small","").replace("\\textit{","").replace("\\bar{","").replace("}","").replace("$","")

def ttbarStr():
    return "$\\textit{t}\\bar{\\textit{t}}$"

def add_checkNone(in1,in2):
    if in1 is None:
        if in2 is None: return None
        else:           return in2
    else:
        if in2 is None: return in1

    return (in1+in2)

def writeLatexHeader(o):
    o.write("\documentclass[12pt]{article}\n")
    o.write("\usepackage{rotating}\n")
    o.write("\usepackage{nopageno}\n")
    o.write("\\begin{document}\n")


def writeLatexTrailer(o):
    o.write("\\end{document}\n")

    

class CutCounts:

    def __init__(self,name,reg,d4=0,d3=0,t4=0,t3=None, t4_s=None, t4_h=None, t4_d=None, t3_s=None, t3_h=None, t3_d=None):
        self.name = name
        self.reg = reg

        self.d4   = d4
        self.d3   = d3
        self.t4   = t4
        self.t3   = t3
        self.t4_s   = t4_s
        self.t4_h   = t4_h
        self.t4_d   = t4_d
        self.t3_s   = t3_s
        self.t3_h   = t3_h
        self.t3_d   = t3_d
        

        self.calc()



    def calc(self):
        self.ft4_s = "N/A"
        self.ft4_h = "N/A"
        self.ft4_d = "N/A"
        self.ft3_s = "N/A"
        self.ft3_h = "N/A"
        self.ft3_d = "N/A"
        self.ratio = "N/A"
        self.ft4 = "N/A"
        self.fmj = "N/A"
        self.ft3 = "N/A"

        self.do3tagTTDetails = bool(self.t3 is not None)
        

        # rewieight
        if reweight:
            self.multijet   = self.d3
        else:
            #if not self.do3tagTTDetails: print "Need to give 3-tag ttbar if not doing reweighting"
            if self.t3 is None: 
                self.multijet = self.d3
            else:
                self.multijet   = self.d3 - self.t3


        self.doTTDetails = False
        if self.t4_s is not None:
            if self.t4_h is None: print "ERROR t4_s defined but not t4_h"
            if self.t4_d is None: print "ERROR t4_s defined but not t4_d"
            self.doTTDetails = True
            if self.t4:
                self.ft4_s = int(100*self.t4_s/self.t4)
                self.ft4_h = int(100*self.t4_h/self.t4)
                self.ft4_d = int(100*self.t4_d/self.t4)

        if self.t3_s is not None:
            if self.t3_h is None: print "ERROR t4_s defined but not t4_h"
            if self.t3_d is None: print "ERROR t4_s defined but not t4_d"
            self.doTTDetails = True
            if self.t3:
                self.ft3_s = int(100*self.t3_s/self.t3)
                self.ft3_h = int(100*self.t3_h/self.t3)
                self.ft3_d = int(100*self.t3_d/self.t3)
                

        self.bkgTotal = self.multijet+self.t4

        if self.bkgTotal:
            self.ft4 = int(100*self.t4/self.bkgTotal)
            self.fmj = int(100*self.multijet/self.bkgTotal)
        if self.d3 and self.do3tagTTDetails:
            self.ft3 = int(100*self.t3/self.d3)


        if self.bkgTotal and self.d4: 
            self.ratio = round(self.d4/self.bkgTotal ,2)

        return

    def __add__(self,o):

        self.d4     += o.d4
        self.d3     += o.d3
        self.t4     += o.t4
        self.t3     = add_checkNone(self.t3,   o.t3)
        self.t4_s   = add_checkNone(self.t4_s, o.t4_s)
        self.t4_h   = add_checkNone(self.t4_h ,o.t4_h)
        self.t4_d   = add_checkNone(self.t4_d ,o.t4_d)
        self.t3_s   = add_checkNone(self.t3_s ,o.t3_s)
        self.t3_h   = add_checkNone(self.t3_h ,o.t3_h)
        self.t3_d   = add_checkNone(self.t3_d ,o.t3_d)

        self.calc()

        return self

    def getOutput(self):
        
        self.header =  [bold("Cut")]
        self.header += [bold("Region")]
        self.header += [bold("Multi-jet")]
        self.header += [""]
        self.header += [bold(ttbarStr())    ]
        self.header += [""]
        self.header += [bold("bkg Total")]
        self.header += [bold("4b data")  ]
        self.header += [bold("Ratio")    ]

        #
        # Latex tabs
        #
        self.tabular = "l"    # Cut
        self.tabular += "r"   # Region
        self.tabular += "||c" # M-J
        self.tabular += "r"   # M-J (details)
        self.tabular += "c"   # TT
        self.tabular += "r"   # TT (details
        self.tabular += "|c"  # Bkg Total
        self.tabular += "|c"  # 4b Data
        self.tabular += "|c"  # Ratio


        outputLines = []
        
        countLine = [""] * 9
        
        countLine[0] = self.name.replace("pass","")
        countLine[1] = self.reg
        countLine[2] = bold(str(self.multijet))  #M-J
        countLine[4] = bold(str(self.t4)) # TT
        countLine[6] = bold(str(self.bkgTotal))
        countLine[7] = bold(str(self.d4)      )
        countLine[8] = bold(str(self.ratio)   )

        #print countLine
        outputLines.append(countLine)


        fracLine = [""] * 9
        fracLine[2] = bold(str(self.fmj)+"%")
        fracLine[4] = bold(str(self.ft4)+"%")
        outputLines.append(fracLine)

        # 3b TTbar Line

        ttbar3bLine = [""]*9
        if self.do3tagTTDetails:
            ttbar3bLine[2] = small(str(self.t3)+" 3b-"+ttbarStr())
            ttbar3bLine[3] = small(str(self.ft3)+"% M-j")
        outputLines.append(ttbar3bLine)

        if self.doTTDetails:
            
            #
            # Had
            #
            ttbarHadLine = [""]*9
            # 3b 
            ttbarHadLine[2] = small(str(self.t3_h)+" tt-had")
            ttbarHadLine[3] = str(self.ft3_h)+"% 3b-"+ttbarStr()  

            # 4b 
            ttbarHadLine[4] = small(str(self.t4_h)+" tt-had")
            ttbarHadLine[5] = str(self.ft4_h)+"% 4b-"+ttbarStr()
            outputLines.append(ttbarHadLine)

            #
            # Semilep
            #
            ttbarSemiLine = [""]*9
            # 3b 
            ttbarSemiLine[2] = small(str(self.t3_s)+" tt-semi")
            ttbarSemiLine[3] = str(self.ft3_s)+"% 3b-"+ttbarStr()

            # 4b 
            ttbarSemiLine[4] = small(str(self.t4_s)+" tt-semi")
            ttbarSemiLine[5] = str(self.ft4_s)+"% 4b-"+ttbarStr()
            outputLines.append(ttbarSemiLine)


            #
            # DiLeplep
            #
            ttbarDiLepLine = [""]*9
            # 3b 
            ttbarDiLepLine[2] = small(str(self.t3_d)+" tt-2lep")
            ttbarDiLepLine[3] = str(self.ft3_d)+"% 3b-"+ttbarStr()

            # 4b 
            ttbarDiLepLine[4] = small(str(self.t4_d)+" tt-2lep")
            ttbarDiLepLine[5] = str(self.ft4_d)+"% 4b-"+ttbarStr()
            outputLines.append(ttbarDiLepLine)




        return outputLines

    def printLine(self, lineList,colWidth):
        for iw, word in enumerate(lineList): 
            thisWord = cleanLatex(str(word))
            while len(thisWord) < colWidth[iw]:
                thisWord = " "+thisWord
            print thisWord+"    ",
        print 

    def printHeader(self, colWidth):
        self.printLine(self.header,colWidth)




    def printOut(self, colWidth):

        lines = self.getOutput()

        #
        #  Region Sum 
        #
        for l in lines:
            self.printLine(l,colWidth)


        return 

    def printLatexLine(self,o,line):
        for iw, w in enumerate(line):
            if iw:
                o.write(" & ")
            o.write(str(w).replace("%","\%").replace("_","\_"))
        o.write("\\\\\n")

    def printLatexHeader(self,o):
        self.printLatexLine(o,self.header)
        o.write("\hline \n")
        o.write("\hline \n")

    def printLatex(self,o, regions):
        
        lines = self.getOutput()
        for l in lines:
            self.printLatexLine(o,l)
        return

    def writeLatexHeader(self,o):
        #o.write("\documentclass[12pt]{article}\n")
        #o.write("\usepackage{rotating}\n")
        #o.write("\\begin{document}\n")
        o.write("\\begin{sidewaystable}\n")
        o.write("\\begin{tabular}{"+self.tabular+"}\n")


class CutData:
    
    def __init__(self,name, reg, d4,d3,t4,t3,t4_s,t4_h,t4_d,t3_s,t3_h,t3_d):
        self.name = name
        self.counts = CutCounts(name,reg,d4,d3,t4,t3,t4_s,t4_h,t4_d,t3_s,t3_h,t3_d)
        self.countsPerRegion = {}

    def addRegion(self,reg,d4,d3,t4, t3, t4_s, t4_h, t4_d, t3_s, t3_h, t3_d ):
        self.countsPerRegion[reg] = CutCounts("",reg,d4,d3,t4,t3,t4_s,t4_h,t4_d,t3_s,t3_h,t3_d)

    def printHeader(self, colWidth):    
        self.counts.printHeader(colWidth)
        

    def getOutput(self, regions):

        output = []
        
        if not len(self.countsPerRegion):
        
            outputLines = self.counts.getOutput()
            for l in outputLines:
                output.append(l)
            output.append(self.counts.header)

        else:

            self.isSRBlind = not bool(self.countsPerRegion["SR"].d4)
            if self.isSRBlind:
                self.counts = CutCounts(self.name,"SB+CR")
                self.counts += self.countsPerRegion["SB"] 
                self.counts += self.countsPerRegion["CR"] 
            else:
                self.counts = CutCounts(self.name,"SB+CR+SR")
                self.counts += self.countsPerRegion["SB"] 
                self.counts += self.countsPerRegion["CR"] 
                self.counts += self.countsPerRegion["SR"] 

            outputLines = self.counts.getOutput()
            for l in outputLines:
                output.append(l)
            output.append(self.counts.header)


            if len(self.countsPerRegion):
                for r in regions:
                    outputLines = self.countsPerRegion[r].getOutput()
                    for l in outputLines:
                        output.append(l)                


        return output

    def printOut(self, colWidth, regions):
        
        self.counts.printOut(colWidth)

        #
        #  SubRegions Sum 
        #
        if len(self.countsPerRegion):
            for r in regions:
                self.countsPerRegion[r].printOut(colWidth)                

        print 

    def printLatex(self,o,regions):
        self.counts.printLatex(o, regions)

        #
        #  SubRegions Sum 
        #
        if len(self.countsPerRegion):
            for r in regions:
                self.countsPerRegion[r].printLatex(o,regions)                

        o.write("\\\\ \n") 
        o.write("\hline\n") 

        


    def writeLatexHeader(self,o):
        self.counts.writeLatexHeader(o)
        self.counts.printLatexHeader(o)

    def writeLatexTrailer(self,o):
        o.write("\\end{tabular}\n")
        o.write("\\end{sidewaystable}\n")
        
        #o.write("\\end{document}\n")



def getCounts(theFile,cutName,region,tag="fourTag"):
    #print theFile, cutName+"/"+tag+"/mainView/"+region+"/nCanJets"
    #return theFile.Get(cutName+"/"+tag+"/mainView/"+region+"/nCanJets").GetBinContent(5)
    if region[0:2] == "SR" and len(region) > 2:
        SvBcut = float(region.replace("SR",""))/100
        regionName = "SR"
        #print region,SvBcut
    else:
        regionName = region
        SvBcut = None
    

    haveSvB = bool(theFile.Get(cutName+"/"+tag+"/mainView/"+regionName+"/SvB_ps"))
    if not SvBcut is None and haveSvB:
        hist = theFile.Get(cutName+"/"+tag+"/mainView/"+regionName+"/SvB_ps")
        lowBin  = hist.GetXaxis().FindBin(SvBcut)
        highBin = hist.GetXaxis().FindBin(+1)
        return theFile.Get(cutName+"/"+tag+"/mainView/"+regionName+"/SvB_ps").Integral(lowBin,highBin)

    return theFile.Get(cutName+"/"+tag+"/mainView/"+regionName+"/nCanJets").GetBinContent(5)


def printValues(v1):
    return str(v1)

def makePreFix(cut,targetLen):
    outString = cut
    while(len(outString) < targetLen):
        outString += " "
    return outString

def getFileCounts(inFile,cuts, regions, tag, debug=False):
    counts = {}
    if inFile:
        in_cfHist = inFile.Get("cutflow/"+tag+"/weighted")
    else:
        in_cfHist = None

    for cut in cutFlow:

        if in_cfHist is None:
            counts[cut] = None
            for reg in regions:
                counts[cut+"_"+reg] = None
        else:

            if in_cfHist and in_cfHist.GetXaxis().FindFixBin(cut) > 0:
                d4Count = in_cfHist.GetBinContent(in_cfHist.GetXaxis().FindBin(cut))
                counts[cut] = round(d4Count,1)
                if debug: print makePreFix(cut,30),printValues(d4Count)
    
            else:
                #print inFile.ls()
                inCount = getCounts(inFile,cut,"SCSR",tag=tag)
                #inCount = getCounts(inFile,cut,"inclusive",tag=tag)
                counts[cut] = round(inCount,1)
                if debug: print makePreFix(cut,30),printValues(inCount)
                for reg in ["SB","CR","SR","SR95"]:
                    inCount = getCounts(inFile,cut,reg,tag=tag)
                    counts[cut+"_"+reg] = round(inCount,1)
                    if debug: print "\t",makePreFix(reg,30),printValues(inCount)
                    
    return counts


    
    

def doCutFlow(d4File, d3File, t4File, t3File, t4File_s, t4File_h, t4File_d, t3File_s, t3File_h, t3File_d, cuts, debug=False):

    regions = ["SB","CR","SR"]
    haveSvB = (bool(d4File.Get("passXWt/fourTag/mainView/SB/SvB_ps").GetEntries()) and bool(d3File.Get("passXWt/fourTag/mainView/SB/SvB_ps").GetEntries()))
    if haveSvB:
        regions += ["SR95"]

    d4Counts = getFileCounts(d4File, cuts, regions, tag="fourTag",  debug=debug)
    d3Counts = getFileCounts(d3File, cuts, regions, tag="threeTag", debug=debug)
    t4Counts = getFileCounts(t4File, cuts, regions, tag="fourTag" , debug=debug)
    t3Counts = getFileCounts(t3File, cuts, regions, tag="threeTag" ,debug=debug)
    
    t4Counts_s = getFileCounts(t4File_s, cuts, regions, tag="fourTag" ,debug=debug)
    t4Counts_h = getFileCounts(t4File_h, cuts, regions, tag="fourTag" ,debug=debug)
    t4Counts_d = getFileCounts(t4File_d, cuts, regions, tag="fourTag" ,debug=debug)

    t3Counts_s = getFileCounts(t3File_s, cuts, regions, tag="threeTag" ,debug=debug)
    t3Counts_h = getFileCounts(t3File_h, cuts, regions, tag="threeTag" ,debug=debug)
    t3Counts_d = getFileCounts(t3File_d, cuts, regions, tag="threeTag" ,debug=debug)


    #cutFlowData = []

    outFile = open(o.name+".tex","w")
    writeLatexHeader(outFile)
    
    for cut in cutFlow:
        if debug: print "Cut is",cut
        cutFlowData = CutData(cut,
                              "total",
                              d4Counts[cut],
                              d3Counts[cut],
                              t4Counts[cut],
                              t3Counts[cut],
                              t4Counts_s[cut],
                              t4Counts_h[cut],
                              t4Counts_d[cut],
                              t3Counts_s[cut],
                              t3Counts_h[cut],
                              t3Counts_d[cut],
                              )
                           

        
        if cut+"_SB" in d4Counts:

            for reg in regions:
                regCut = cut+"_"+reg
                if debug: print "Adding region is",regCut
                cutFlowData.addRegion(reg,
                                      d4Counts[regCut],
                                      d3Counts[regCut],
                                      t4Counts[regCut],
                                      t3Counts[regCut],
                                      t4Counts_s[regCut],
                                      t4Counts_h[regCut],
                                      t4Counts_d[regCut],
                                      t3Counts_s[regCut],
                                      t3Counts_h[regCut],
                                      t3Counts_d[regCut],
                                      )

        #
        # Calc col widths
        #
        colWidth = {}
    
        rawOutput = cutFlowData.getOutput(regions)
        for line in rawOutput:
            for iw, word in enumerate(line):
                if iw not in colWidth:  colWidth[iw] = 0
                
                thisLen = len(cleanLatex(str(word)))
                if thisLen > colWidth[iw]:  colWidth[iw] = thisLen

        if debug: print "Done calc col widths"

        #
        #  printout and make latex
        #
        cutFlowData.writeLatexHeader(outFile)
        cutFlowData.printLatex(outFile, regions)
        cutFlowData.writeLatexTrailer(outFile)

        #
        #  Print out
        #
        cutFlowData.printHeader(colWidth)
        print 
        cutFlowData.printOut(colWidth, regions)

    

    
    writeLatexTrailer(outFile)
    outFile.close()

    if o.makePDF:
        import os
        os.system("pdflatex "+o.name+".tex > "+o.name+"_pdflatex_build.log")
        
        # the .aux and .log files get made in PWD
        os.system("rm "+o.name.split("/")[-1]+".aux")
        os.system("rm "+o.name.split("/")[-1]+".log")





    

if __name__ == "__main__":
    cutFlow = o.cuts.split(",")

    d4File = ROOT.TFile(o.d4,"READ")
    d3File = ROOT.TFile(o.d3,"READ")
    t4File = ROOT.TFile(o.t4,"READ")

    t3File = None

    t4File_s = None
    t4File_h = None
    t4File_d = None

    t3File_s = None
    t3File_h = None
    t3File_d = None


    if o.t3:
        print "Loading a 3-tag ttbar"
        t3File = ROOT.TFile(o.t3,"READ")

        if o.t4_s: 
            t4File_s = ROOT.TFile(o.t4_s,"READ")
            t4File_h = ROOT.TFile(o.t4_h,"READ")
            t4File_d = ROOT.TFile(o.t4_d,"READ")
    
            t3File_s = ROOT.TFile(o.t3_s,"READ")
            t3File_h = ROOT.TFile(o.t3_h,"READ")
            t3File_d = ROOT.TFile(o.t3_d,"READ")


    
    doCutFlow(d4File,   d3File,   t4File, t3File, 
              t4File_s, t4File_h, t4File_d, 
              t3File_s, t3File_h, t3File_d, 
              cutFlow,  o.debug)



    
