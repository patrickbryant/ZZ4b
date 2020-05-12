import optparse
parser = optparse.OptionParser()
parser.add_option('--file1')
parser.add_option('--file2')
o, a = parser.parse_args()


def getRunEvents(fileName):

    file1 = open(fileName,"r")
    runList = {}
    nTotal = 0
    for line in file1:
        words = line.split()
        if len(words) < 3: continue

        run = words[1]
        event = words[2]
    
        if run not in runList:
            runList[run] = set()
    
        if event in runList[run]:
            print "ERROR event",event," already counted ... "

        runList[run].add(event)
        nTotal += 1

    return runList,nTotal

def compEvents(runsEventsA, nameA, runsEventsB, nameB):


    nTotalAnotB = 0

    runListA = runsEventsA.keys()
    runListA.sort()

    for runA in runListA:

        if runA not in runsEventsB:
            print runA,"not in ",nameA
            continue

        eventsA = runsEventsA[runA]
        eventsB = runsEventsB[runA]

        AnotB = eventsA.difference(eventsB)
        if len(AnotB): 

            print runA,":\t",
            for e in AnotB:
                print e,
                nTotalAnotB +=1
            print 
    return nTotalAnotB

runEventsFile1, nEventsFile1 = getRunEvents(o.file1)
runEventsFile2, nEventsFile2 = getRunEvents(o.file2)

print "\n"*3
print "In ",o.file1,"not in",o.file2
nEventsIn1not2 = compEvents(runEventsFile1,o.file1,runEventsFile2,o.file2)

print "\n"*3
print "In ",o.file2,"not in",o.file1
nEventsIn2not1 = compEvents(runEventsFile2,o.file2,runEventsFile1,o.file1)


print o.file1,"nEvents Total",nEventsFile1
print "\t unique events ",nEventsIn1not2
print o.file2,"nEvents Total",nEventsFile2
print "\t unique events",nEventsIn2not1

