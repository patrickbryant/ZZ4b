import optparse
parser = optparse.OptionParser()
parser.add_option('--file1')
parser.add_option('--file2')
parser.add_option('--verbose', action="store_true")
parser.add_option('--txtFiles', action="store_true")
o, a = parser.parse_args()

import ROOT 

def getRunEventsText(fileName):
    
    #print "fileName is ",fileName
    file1 = open(fileName,"r")
    runList = {}
    nTotal = 0
    for line in file1:
        words = line.split()
        if len(words) < 2: continue
        
        if words[0] == "Run": continue

        run = int(words[0])
        event = int(words[1])
    
        if run not in runList:
            runList[run] = set()
    
        if event in runList[run]:
            print "ERROR event",event," already counted in run ", run

        runList[run].add(event)
        nTotal += 1

    return runList,nTotal


def getRunEvents(fileName):

    #print "fileName is ",fileName
    file1 = ROOT.TFile.Open(fileName)
    passed_events = file1.Get("passed_events")
    passed_runs   = file1.Get("passed_runs")

    nEntries = passed_events.size()

    runList = {}
    nTotal = 0
    for i in xrange(nEntries):

        run = passed_runs[i]
        event = passed_events[i]
    
        if run not in runList:
            runList[run] = set()
    
        if event in runList[run]:
            print "ERROR event",event," already counted in run ", run

        runList[run].add(event)
        nTotal += 1

    return runList,nTotal


def checkDuplicates(fileName):

    #print "fileName is ",fileName
    file1 = ROOT.TFile.Open(fileName)
    passed_events = file1.Get("passed_events")
    passed_runs   = file1.Get("passed_runs")
    passed_LBs   = file1.Get("passed_LBs")

    nEntries = passed_events.size()

    runList = {}
    #LBList = {}
    nTotal = 0
    nDups = 0
    for i in xrange(nEntries):

        run = passed_runs[i]
        event = passed_events[i]
        LB   = passed_LBs[i]
    
        if run not in runList:
            runList[run] = set()
            #LBList[run] = set()
    
        if (event, LB) in runList[run]:
            print "ERROR event",(event, LB)," already counted in run ", run
            nDups += 1

        runList[run].add((event,LB))
        #LBList[run].add(LB)
        nTotal += 1

    return runList,nTotal, nDups


def compEvents(runsEventsA, nameA, runsEventsB, nameB):


    nTotalAnotB = 0

    runListA = runsEventsA.keys()
    runListA.sort()

    for runA in runListA:

        if runA not in runsEventsB:
            if o.verbose: print runA,"not in ",nameA
            continue

        eventsA = runsEventsA[runA]
        eventsB = runsEventsB[runA]

        AnotB = eventsA.difference(eventsB)
        if len(AnotB): 

            if o.verbose: print runA,":\t",
            for e in AnotB:
                if o.verbose: print e,
                nTotalAnotB +=1
            if o.verbose: print 
    return nTotalAnotB


def getEventDiffs(file1, file2, isTextFile=False):
    if isTextFile:
        runEventsFile1, nEventsFile1 = getRunEventsText(file1)
        runEventsFile2, nEventsFile2 = getRunEventsText(file2)
    else:
        runEventsFile1, nEventsFile1 = getRunEvents(file1)
        runEventsFile2, nEventsFile2 = getRunEvents(file2)

    if o.verbose:
        print "\n"*3
        print "In ",o.file1,"not in",o.file2
    nEventsIn1not2 = compEvents(runEventsFile1,o.file1,runEventsFile2,o.file2)
    
    if o.verbose:
        print "\n"*3
        print "In ",o.file2,"not in",o.file1
    nEventsIn2not1 = compEvents(runEventsFile2,o.file2,runEventsFile1,o.file1)
    
    return nEventsFile1, nEventsFile2, nEventsIn1not2, nEventsIn2not1

    

def main(txtFiles):
    if txtFiles:
        nEventsFile1, nEventsFile2, nEventsIn1not2, nEventsIn2not1 = getEventDiffs(o.file1, o.file2, isTextFile=True)
    else:
        nEventsFile1, nEventsFile2, nEventsIn1not2, nEventsIn2not1 = getEventDiffs(o.file1, o.file2)
    
    print o.file1,"nEvents Total",nEventsFile1
    print "\t unique events",nEventsIn1not2
    print o.file2,"nEvents Total",nEventsFile2
    print "\t unique events",nEventsIn2not1
    
    print "% overlap:",round(float(nEventsFile1-nEventsIn1not2)/nEventsFile1,2),"or",round(float(nEventsFile2-nEventsIn2not1)/nEventsFile2,2)


if __name__ == "__main__":

    main(txtFiles = o.txtFiles)


