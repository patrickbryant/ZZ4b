import os
import ROOT

#
# From 
#   > source ZZ4b/nTupleAnalysis/scripts/getInputEventCounts.sh 
#
expectedCounts = {
    "TTTo2L2Nu2016":
        [{"file_size":121010029816,"nblocks":18,"nevents":65852400,"nfiles":74,"nlumis":329262,"num_block":18,"num_event":65852400,"num_file":74,"num_lumi":329262}],
    "TTToHadronic2017":
        [{"file_size":126744786549,"nblocks":17,"nevents":68518800,"nfiles":89,"nlumis":342594,"num_block":17,"num_event":68518800,"num_file":89,"num_lumi":342594}],
    "TTToSemiLeptonic2017":
        [{"file_size":200711942960,"nblocks":26,"nevents":107305100,"nfiles":127,"nlumis":643826,"num_block":26,"num_event":107305100,"num_file":127,"num_lumi":643826}],
    "TTTo2L2Nu2017":
        [{"file_size":17045146070,"nblocks":3,"nevents":9000000,"nfiles":11,"nlumis":9293,"num_block":3,"num_event":9000000,"num_file":11,"num_lumi":9293}],
    "TTToHadronic2017":
        [{"file_size":78888914727,"nblocks":2,"nevents":41729120,"nfiles":41,"nlumis":61848,"num_block":2,"num_event":41729120,"num_file":41,"num_lumi":61848}],
    "TTToSemiLeptonic2017":
        [{"file_size":84060179192,"nblocks":3,"nevents":43732445,"nfiles":46,"nlumis":49439,"num_block":3,"num_event":43732445,"num_file":46,"num_lumi":49439}],
    "TTToSemiLeptonic2018":
        [{"file_size":190788852043,"nblocks":4,"nevents":101550000,"nfiles":76,"nlumis":20310,"num_block":4,"num_event":101550000,"num_file":76,"num_lumi":20310}],
    "TTToHadronic2018":
        [{"file_size":248337796131,"nblocks":9,"nevents":133664000,"nfiles":125,"nlumis":16708,"num_block":9,"num_event":133664000,"num_file":125,"num_lumi":16708}],
    "TTTo2L2Nu2018":
        [{"file_size":119185835970,"nblocks":8,"nevents":64310000,"nfiles":57,"nlumis":12862,"num_block":8,"num_event":64310000,"num_file":57,"num_lumi":12862}],
    "Run2016B":
        [{"file_size":72727388725,"nblocks":2,"nevents":77890616,"nfiles":51,"nlumis":59642,"num_block":2,"num_event":77890616,"num_file":51,"num_lumi":59642}],
    "Run2016C":
        [{"file_size":29605199042,"nblocks":3,"nevents":30358567,"nfiles":27,"nlumis":18810,"num_block":3,"num_event":30358567,"num_file":27,"num_lumi":18810}],
    "Run2016D":
        [{"file_size":54581945953,"nblocks":2,"nevents":56527008,"nfiles":34,"nlumis":30295,"num_block":2,"num_event":56527008,"num_file":34,"num_lumi":30295}],
    "Run2016E":
        [{"file_size":60057006791,"nblocks":3,"nevents":60415444,"nfiles":39,"nlumis":27199,"num_block":3,"num_event":60415444,"num_file":39,"num_lumi":27199}],
    "Run2016F":
        [{"file_size":38005393269,"nblocks":3,"nevents":37608672,"nfiles":29,"nlumis":19584,"num_block":3,"num_event":37608672,"num_file":29,"num_lumi":19584}],
    "Run2016G":
        [{"file_size":101433788121,"nblocks":3,"nevents":100834056,"nfiles":60,"nlumis":46543,"num_block":3,"num_event":100834056,"num_file":60,"num_lumi":46543}],
    "Run2016H":
        [{"file_size":66031291979,"nblocks":4,"nevents":65242072,"nfiles":43,"nlumis":52754,"num_block":4,"num_event":65242072,"num_file":43,"num_lumi":52754}],
    "Run2017B":
        [{"file_size":2016436046,"nblocks":1,"nevents":1808836,"nfiles":4,"nlumis":26561,"num_block":1,"num_event":1808836,"num_file":4,"num_lumi":26561}],
    "Run2017C":
        [{"file_size":29960717875,"nblocks":1,"nevents":34491540,"nfiles":25,"nlumis":57761,"num_block":1,"num_event":34491540,"num_file":25,"num_lumi":57761}],
    "Run2017D":
        [{"file_size":8070824189,"nblocks":2,"nevents":7967055,"nfiles":9,"nlumis":28337,"num_block":2,"num_event":7967055,"num_file":9,"num_lumi":28337}],
    "Run2017E":
        [{"file_size":18373206156,"nblocks":1,"nevents":17185873,"nfiles":14,"nlumis":45465,"num_block":1,"num_event":17185873,"num_file":14,"num_lumi":45465}],
    "Run2017F":
        [{"file_size":88856779169,"nblocks":7,"nevents":75677461,"nfiles":60,"nlumis":61385,"num_block":7,"num_event":75677461,"num_file":60,"num_lumi":61385}],
    "Run2018A":
        [{"file_size":174325899194,"nblocks":8,"nevents":171502033,"nfiles":142,"nlumis":60958,"num_block":8,"num_event":171502033,"num_file":142,"num_lumi":60958}],
    "Run2018B":
        [{"file_size":80509657700,"nblocks":9,"nevents":78255208,"nfiles":68,"nlumis":29820,"num_block":9,"num_event":78255208,"num_file":68,"num_lumi":29820}],
    "Run2018C":
        [{"file_size":72209024389,"nblocks":5,"nevents":70027804,"nfiles":71,"nlumis":27664,"num_block":5,"num_event":70027804,"num_file":71,"num_lumi":27664}],
    "Run2018D":
        [{"file_size":378484528342,"nblocks":16,"nevents":356543782,"nfiles":162,"nlumis":136336,"num_block":16,"num_event":356543782,"num_file":162,"num_lumi":136336}],
    }


def getCounts(inFileName):
    inFile = ROOT.TFile(inFileName,"READ")

    try: 
        cfHist = inFile.Get("cutflow/fourTag/unitWeight")
        nEvents = cfHist.GetBinContent(1)
    except:
        print "Error cant find cutflow/fourTag/unitWeight in ", inFileName
        nEvents = 0

    return nEvents



runs = expectedCounts.keys()
runs.sort()

for run in runs:
    nExpected = expectedCounts[run][0]['nevents']

    skimedFile = "closureTests/nominal/"+run.replace("Run","data")+"/histsFromNanoAOD.root"
    #skimedFile = "/uscms/home/bryantp/nobackup/ZZ4b/"+run.replace("Run","data")+"/histsFromNanoAOD.root"
    if not os.path.isfile(skimedFile):
        print "Skipping ",run,skimedFile,"not found"
        continue
    
    nSeen = getCounts(skimedFile)

    print run,"Expected",nExpected,"Seen",nSeen,"Ratio",float(nSeen)/nExpected
