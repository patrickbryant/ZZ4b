from compEventCounts import getEventDiffs

rows = []

for i in range(9):
    columns = []
    for j in range(9):
        if j > i: 
            nEventsFile1, nEventsFile2, nEventsIn1not2, nEventsIn2not1 = getEventDiffs("closureTests/mixed/data2018/events_3bSubSampled_b0p6_v"+str(i)+".txt",
                                                                                       "closureTests/mixed/data2018/events_3bSubSampled_b0p6_v"+str(j)+".txt")

            maxOverlap = max(float(nEventsFile1-nEventsIn1not2)/nEventsFile1,float(nEventsFile2-nEventsIn2not1)/nEventsFile2)
            columns.append(round(maxOverlap,2))
        else:
            columns.append("X")

    rows.append(columns)

for r in rows:
    print r
