from compEventCounts import getEventDiffs



for y in ["2018","2017","2016"]:


    print 
    print y
    print

    rows = []

    for i in range(10):
        columns = []
        for j in range(10):
            if j > i: 
                nEventsFile1, nEventsFile2, nEventsIn1not2, nEventsIn2not1 = getEventDiffs("closureTests/mixed/data"+y+"_b0p6/events_3bSubSampled_b0p6_v"+str(i)+".txt",
                                                                                           "closureTests/mixed/data"+y+"_b0p6/events_3bSubSampled_b0p6_v"+str(j)+".txt")
    
                maxOverlap = max(float(nEventsFile1-nEventsIn1not2)/nEventsFile1,float(nEventsFile2-nEventsIn2not1)/nEventsFile2)
                columns.append(round(maxOverlap,2))
            else:
                if i == j:
                    columns.append("1.0")
                else:
                    columns.append(" - ")
        print "nEvents in ",i,nEventsFile1
        rows.append(columns)
    
    for rI, r in enumerate(rows):
        print "v"+str(rI)," & ",
        for c in r:
            print c," & ",
        print " \\\\ "
