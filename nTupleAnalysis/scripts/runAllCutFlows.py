import os
compCutFlow = "python ZZ4b/nTupleAnalysis/scripts/compCutFlow.py " 
patrickPath = "/uscms/home/bryantp/nobackup/ZZ4b/"
#myPath = "closureTests/nominal/"
myPath = "root://cmseos.fnal.gov//store/user/jda102/condor/nominal/"
#--file1 data2016D/histsFromNanoAOD.root  --file2 data2016D/histsFromNanoAOD.root 


for f in [
        "data2016B",
        "data2016C",
        "data2016D",
        "data2016E",
        "data2016F",
        "data2016G",
        "data2016H",
        "data2017C",
        "data2017D",
        "data2017E",
        "data2017F",
        "data2018A",
        "data2018B",
        "data2018C",
        "data2018D",
        "TTToHadronic2017",
        "TTToHadronic2016",
       "TTToHadronic2018",
       "TTToSemiLeptonic2017",
       "TTToSemiLeptonic2016",
       "TTToSemiLeptonic2018",
       "TTTo2L2Nu2017",
       "TTTo2L2Nu2016",
       "TTTo2L2Nu2018",
]:

    cmd = compCutFlow
    cmd += " --file1 "+patrickPath+"/"+f+"/histsFromNanoAOD.root "
    cmd += " --file2 "+myPath+"/"+f+"/histsFromNanoAOD.root "
    cmd += " --cuts all,lumiMask,all_HLT,HLT,jetMultiplicity,jetMultiplicity_SR,bTags,bTags_HLT"

    print f
    os.system(cmd)

