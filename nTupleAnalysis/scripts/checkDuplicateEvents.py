from compEventCounts import checkDuplicates, getEventDiffs


eosPath = "root://cmseos.fnal.gov//store/user/johnda/condor/ZH4b/ULTrig/"
#root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig/data2018_v6/hists_3bSubSampled_v6.root
#data2016_b0p60p3_v6/hists_3bSubSampled_b0p60p3_v6.root  --file2 
#root://cmseos.fnal.gov//store/user/johnda/condor/mixed//data2016_b0p60p3_v0/hists_3bSubSampled_b0p60p3_v0.root 

#tagID = "b0p60p3"
do3bSubSampled = False


#for y in ["2018","2017","2016"]:
for y in ["2018"]:#,"2017","2016"]:


    print 
    print y
    print

    rows = []

    #
    #  3b subSampled
    #
    if do3bSubSampled:
        for i in range(15):
            columns = []
            for j in range(15):
                if j > i: 
                    nEventsFile1, nEventsFile2, nEventsIn1not2, nEventsIn2not1 = getEventDiffs(eosPath+"data"+y+"_v"+str(i)+"/hists_3bSubSampled_v"+str(i)+"_newSBDef.root",
                                                                                               eosPath+"data"+y+"_v"+str(j)+"/hists_3bSubSampled_v"+str(j)+"_newSBDef.root")
        
                    maxOverlap = max(float(nEventsFile1-nEventsIn1not2)/nEventsFile1,float(nEventsFile2-nEventsIn2not1)/nEventsFile2)
                    columns.append(round(maxOverlap,2))
                else:
                    if i == j:
                        columns.append("1.0")
                    else:
                        columns.append(" - ")
            print "nEvents in ",i,nEventsFile1
            rows.append(columns)
    
    else:
        for i in range(1):
            #data2017_3bDvTMix4bDvT_vAll/hists_3bDvTMix4bDvT_v0_newSBDef.root
            print eosPath+"data"+y+"_3bDvTMix4bDvT_vAll/hists_v"+str(i)+"_newSBDef.root"
            runList, nTotal, nDups = checkDuplicates(eosPath+"data"+y+"_3bDvTMix4bDvT_vAll/hists_v"+str(i)+"_newSBDef.root")
            print "nEvents in ",i,nTotal, nDups, float(nDups)/nTotal
            
            
#            for r in runList: 
#                print r, runList[r]
