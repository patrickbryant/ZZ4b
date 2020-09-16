import optparse
parser = optparse.OptionParser()
parser.add_option('--file1',     help="")
parser.add_option('--file2',     help="")
parser.add_option('--cuts',  default="all,HLT,jetMultiplicity,bTags,bTags_HLT,passPreSel,passDijetMass,passMDRs",  help="Comma separate list of cuts. Default is: \n"+"all,HLT,jetMultiplicity,bTags,bTags_HLT,passPreSel,passDijetMass,passMDRs,passXWt\n")
o, a = parser.parse_args()

import ROOT

from makeCutFlow import getFileCounts

def padStr(input,length):
    output = str(input)

    while len(output) < length:
        output = " "+output
    return output

def compCutFlow(file1, file2, cuts, debug=False):

    regions = ["SB","CR","SR"]
    #haveSvB = (bool(file1.Get("passMDRs/fourTag/mainView/SB/SvB_ps").GetEntries()) and bool(file2.Get("passMDRs/threeTag/mainView/SB/SvB_ps").GetEntries()))
    #if haveSvB:
    #    regions += ["SR95"]

    
    d4Counts_1 = getFileCounts(file1, cuts, regions, tag="threeTag",  cutFlowHistName="unitWeight", debug=debug)
    d4Counts_2 = getFileCounts(file2, cuts, regions, tag="threeTag",  cutFlowHistName="unitWeight", debug=debug)

    print o.file1,"vs",o.file2
    
    pad = 10
    maxLen = 0
    for c in cuts:
        if len(c) > maxLen: maxLen = len(c)

    totalLen = maxLen+pad

    for c in cuts:
        ratio = d4Counts_1[c]/d4Counts_2[c] if d4Counts_2[c] > 0 else 0
                

        print padStr(c,totalLen),padStr(d4Counts_1[c],totalLen),"vs",padStr(d4Counts_2[c],totalLen), "ratio",ratio



    

if __name__ == "__main__":
    cutFlow = o.cuts.split(",")

    file1 = ROOT.TFile(o.file1,"READ")
    file2 = ROOT.TFile(o.file2,"READ")


    compCutFlow(file1, file2, cutFlow)
