import os
import ROOT


#/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL16NanoAOD-106X_mcRun2_asymptotic_v13-v1/NANOAODSIM
#/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v4/NANOAODSIM
#/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v2/NANOAODSIM
#
#/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM
#
#/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v1/NANOAODSIM
#
#/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v2/NANOAODSIM
#
#/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM
#
#/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v1/NANOAODSIM
#
#/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v1/NANOAODSIM
#
#


#
# From 
#   > getInputEventCountsFromDAS.py
#
expectedCounts = {

    #"TTTo2L2Nu2016":
    #[{"file_size":132141877089,"nblocks":52,"nevents":67860400,"nfiles":119,"nlumis":339302,"num_block":52,"num_event":67860400,"num_file":119,"num_lumi":339302}],
    #[{"file_size":121010029816,"nblocks":18,"nevents":65852400,"nfiles":74,"nlumis":329262,"num_block":18,"num_event":65852400,"num_file":74,"num_lumi":329262}],

    "TTTo2L2Nu2016_preVFP":
    [{"file_size":84760500188,"nblocks":68,"nevents":41364000,"nfiles":102,"nlumis":41364,"num_block":68,"num_event":41364000,"num_file":102,"num_lumi":41364}],

    "TTTo2L2Nu2016_postVFP":
    [{"file_size":99088695178,"nblocks":68,"nevents":48232000,"nfiles":102,"nlumis":48232,"num_block":68,"num_event":48232000,"num_file":102,"num_lumi":48232}],

    #"TTToHadronic2016":
    #[{"file_size":213339433619,"nblocks":53,"nevents":117987400,"nfiles":135,"nlumis":1179874,"num_block":53,"num_event":117987400,"num_file":135,"num_lumi":1179874}],
    #[{"file_size":126744786549,"nblocks":17,"nevents":68518800,"nfiles":89,"nlumis":342594,"num_block":17,"num_event":68518800,"num_file":89,"num_lumi":342594}],

    "TTToHadronic2016_preVFP":
    [{"file_size":207293651683,"nblocks":24,"nevents":97823000,"nfiles":112,"nlumis":97823,"num_block":24,"num_event":97823000,"num_file":112,"num_lumi":97823}],

    "TTToHadronic2016_postVFP":
    [{"file_size":239344381462,"nblocks":77,"nevents":112592000,"nfiles":159,"nlumis":112592,"num_block":77,"num_event":112592000,"num_file":159,"num_lumi":112592}],

    #"TTToSemiLeptonic2016":
    #[{"file_size":214953563144,"nblocks":60,"nevents":107604800,"nfiles":178,"nlumis":645620,"num_block":60,"num_event":107604800,"num_file":178,"num_lumi":645620}],
    #[{"file_size":200711942960,"nblocks":26,"nevents":107305100,"nfiles":127,"nlumis":643826,"num_block":26,"num_event":107305100,"num_file":127,"num_lumi":643826}],

    "TTToSemiLeptonic2016_preVFP":
    [{"file_size":291206720794,"nblocks":108,"nevents":138169000,"nfiles":236,"nlumis":138169,"num_block":108,"num_event":138169000,"num_file":236,"num_lumi":138169}],

    "TTToSemiLeptonic2016_postVFP":
    [{"file_size":335213535307,"nblocks":134,"nevents":158594000,"nfiles":266,"nlumis":158594,"num_block":134,"num_event":158594000,"num_file":266,"num_lumi":158594}],

    "TTTo2L2Nu2017":
    #[{"file_size":126584480129,"nblocks":2,"nevents":66259900,"nfiles":51,"nlumis":662599,"num_block":2,"num_event":66259900,"num_file":51,"num_lumi":662599}],
    #[{"file_size":17045146070,"nblocks":3,"nevents":9000000,"nfiles":11,"nlumis":9293,"num_block":3,"num_event":9000000,"num_file":11,"num_lumi":9293}],
    [{"file_size":231155466864,"nblocks":3,"nevents":106978000,"nfiles":85,"nlumis":106978,"num_block":3,"num_event":106978000,"num_file":85,"num_lumi":106978}],

    "TTToHadronic2017":
    #[{"file_size":250743360993,"nblocks":10,"nevents":129706300,"nfiles":119,"nlumis":1297063,"num_block":10,"num_event":129706300,"num_file":119,"num_lumi":1297063}] ,
    #[{"file_size":78888914727,"nblocks":2,"nevents":41729120,"nfiles":41,"nlumis":61848,"num_block":2,"num_event":41729120,"num_file":41,"num_lumi":61848}],
    [{"file_size":557578980393,"nblocks":4,"nevents":249247999,"nfiles":203,"nlumis":249248,"num_block":4,"num_event":249247999,"num_file":203,"num_lumi":249248}],

    "TTToSemiLeptonic2017":
    #[{"file_size":221037125302,"nblocks":41,"nevents":114058500,"nfiles":123,"nlumis":1140585,"num_block":41,"num_event":114058500,"num_file":123,"num_lumi":1140585}],
    #[{"file_size":84060179192,"nblocks":3,"nevents":43732445,"nfiles":46,"nlumis":49439,"num_block":3,"num_event":43732445,"num_file":46,"num_lumi":49439}],
    [{"file_size":790805135484,"nblocks":5,"nevents":355826000,"nfiles":305,"nlumis":355826,"num_block":5,"num_event":355826000,"num_file":305,"num_lumi":355826}],


    "TTToSemiLeptonic2018":
    #[{"file_size":199389165956,"nblocks":35,"nevents":104660999,"nfiles":351,"nlumis":1046610,"num_block":35,"num_event":104660999,"num_file":351,"num_lumi":1046610}],
    #[{"file_size":190788852043,"nblocks":4,"nevents":101550000,"nfiles":76,"nlumis":20310,"num_block":4,"num_event":101550000,"num_file":76,"num_lumi":20310}],
    [{"file_size":1055121317049,"nblocks":75,"nevents":486770000,"nfiles":465,"nlumis":486770,"num_block":75,"num_event":486770000,"num_file":465,"num_lumi":486770}],

    "TTToHadronic2018":
    #[{"file_size":234480162776,"nblocks":200,"nevents":124497800,"nfiles":557,"nlumis":1244978,"num_block":200,"num_event":124497800,"num_file":557,"num_lumi":1244978}],
    #[{"file_size":248337796131,"nblocks":9,"nevents":133664000,"nfiles":125,"nlumis":16708,"num_block":9,"num_event":133664000,"num_file":125,"num_lumi":16708}],
    [{"file_size":751904872328,"nblocks":74,"nevents":344028000,"nfiles":343,"nlumis":344028,"num_block":74,"num_event":344028000,"num_file":343,"num_lumi":344028}],

    "TTTo2L2Nu2018":
    #[{"file_size":119185835970,"nblocks":8,"nevents":64310000,"nfiles":57,"nlumis":12862,"num_block":8,"num_event":64310000,"num_file":57,"num_lumi":12862}],
    [{"file_size":313342375408,"nblocks":59,"nevents":148470000,"nfiles":181,"nlumis":148470,"num_block":59,"num_event":148470000,"num_file":181,"num_lumi":148470}],

    "Run2016B":
   #[{"file_size":2115136169,"nblocks":1,  "nevents":1972666,"nfiles":2,"nlumis":6657,"num_block":1,"num_event":1972666,"num_file":2,"num_lumi":6657}]
    #[{"file_size":89870461906,"nblocks":1,"nevents":77890616,"nfiles":45,"nlumis":59642,"num_block":1,"num_event":77890616,"num_file":45,"num_lumi":59642}]
    [{"nevents":79863282}],
    # OLD[{"file_size":72727388725,"nblocks":2,"nevents":77890616,"nfiles":51,"nlumis":59642,"num_block":2,"num_event":77890616,"num_file":51,"num_lumi":59642}],

    "Run2016C":
        #[{"file_size":29605199042,"nblocks":3,"nevents":30358567,"nfiles":27,"nlumis":18810,"num_block":3,"num_event":30358567,"num_file":27,"num_lumi":18810}],
        [{"file_size":36417907276,"nblocks":6,"nevents":30358567,"nfiles":30,"nlumis":18810,"num_block":6,"num_event":30358567,"num_file":30,"num_lumi":18810}],

    "Run2016D":
        #[{"file_size":54581945953,"nblocks":2,"nevents":56527008,"nfiles":34,"nlumis":30295,"num_block":2,"num_event":56527008,"num_file":34,"num_lumi":30295}],
        [{"file_size":67367847737,"nblocks":6,"nevents":56527008,"nfiles":45,"nlumis":30295,"num_block":6,"num_event":56527008,"num_file":45,"num_lumi":30295}],

    "Run2016E":
    #[{"file_size":60057006791,"nblocks":3,"nevents":60415444,"nfiles":39,"nlumis":27199,"num_block":3,"num_event":60415444,"num_file":39,"num_lumi":27199}],
    [{"file_size":74223392787,"nblocks":19,"nevents":60415444,"nfiles":60,"nlumis":27807,"num_block":19,"num_event":60415444,"num_file":60,"num_lumi":27807}],

    "Run2016F":
    #[{"file_size":41086549720,"nblocks":10,"nevents":32756402,"nfiles":29,"nlumis":16981,"num_block":10,"num_event":32756402,"num_file":29,"num_lumi":16981}],
    #[{"file_size":6054559654,"nblocks":1,"nevents":4852270,"nfiles":4,"nlumis":2603,"num_block":1,"num_event":4852270,"num_file":4,"num_lumi":2603}],
    [{"nevents":37608672}],
    # OLD [{"file_size":38005393269,"nblocks":3,"nevents":37608672,"nfiles":29,"nlumis":19584,"num_block":3,"num_event":37608672,"num_file":29,"num_lumi":19584}],

    "Run2016G":
        #[{"file_size":101433788121,"nblocks":3,"nevents":100834056,"nfiles":60,"nlumis":46543,"num_block":3,"num_event":100834056,"num_file":60,"num_lumi":46543}],
        [{"file_size":126382415480,"nblocks":15,"nevents":100826567,"nfiles":84,"nlumis":46539,"num_block":15,"num_event":100826567,"num_file":84,"num_lumi":46539}],
    
    "Run2016H":
    #[{"file_size":66031291979,"nblocks":4,"nevents":65242072,"nfiles":43,"nlumis":52754,"num_block":4,"num_event":65242072,"num_file":43,"num_lumi":52754}],
    [{"file_size":82223946672,"nblocks":3,"nevents":65242072,"nfiles":51,"nlumis":52754,"num_block":3,"num_event":65242072,"num_file":51,"num_lumi":52754}],
    
    "Run2017B":
    #[{"file_size":2016436046,"nblocks":1,"nevents":1808836,"nfiles":4,"nlumis":26561,"num_block":1,"num_event":1808836,"num_file":4,"num_lumi":26561}],
    [{"file_size":2678332376,"nblocks":2,"nevents":1808836,"nfiles":4,"nlumis":26561,"num_block":2,"num_event":1808836,"num_file":4,"num_lumi":26561}],

    "Run2017C":
    #[{"file_size":29960717875,"nblocks":1,"nevents":34491540,"nfiles":25,"nlumis":57761,"num_block":1,"num_event":34491540,"num_file":25,"num_lumi":57761}],
    [{"file_size":36048430260,"nblocks":6,"nevents":34491540,"nfiles":54,"nlumis":57761,"num_block":6,"num_event":34491540,"num_file":54,"num_lumi":57761}],

    "Run2017D":
    #[{"file_size":8070824189,"nblocks":2,"nevents":7967055,"nfiles":9,"nlumis":28337,"num_block":2,"num_event":7967055,"num_file":9,"num_lumi":28337}],
    [{"file_size":9913201096,"nblocks":3,"nevents":7967055,"nfiles":12,"nlumis":28337,"num_block":3,"num_event":7967055,"num_file":12,"num_lumi":28337}],

    "Run2017E":
    #[{"file_size":18373206156,"nblocks":1,"nevents":17185873,"nfiles":14,"nlumis":45465,"num_block":1,"num_event":17185873,"num_file":14,"num_lumi":45465}],
    [{"file_size":22841632743,"nblocks":25,"nevents":17185184,"nfiles":41,"nlumis":45464,"num_block":25,"num_event":17185184,"num_file":41,"num_lumi":45464}],

    "Run2017F":
    #[{"file_size":88856779169,"nblocks":7,"nevents":75677461,"nfiles":60,"nlumis":61385,"num_block":7,"num_event":75677461,"num_file":60,"num_lumi":61385}],
    [{"file_size":110687271943,"nblocks":3,"nevents":75677461,"nfiles":72,"nlumis":61385,"num_block":3,"num_event":75677461,"num_file":72,"num_lumi":61385}],

    "Run2018A":
    #[{"file_size":174325899194,"nblocks":8,"nevents":171502033,"nfiles":142,"nlumis":60958,"num_block":8,"num_event":171502033,"num_file":142,"num_lumi":60958}],
     [{"file_size":221522548624,"nblocks":1,"nevents":171484635,"nfiles":124,"nlumis":60888,"num_block":1,"num_event":171484635,"num_file":124,"num_lumi":60888}],

    "Run2018B":
    #[{"file_size":80509657700,"nblocks":9,"nevents":78255208,"nfiles":68,"nlumis":29820,"num_block":9,"num_event":78255208,"num_file":68,"num_lumi":29820}],
    [{"file_size":102524214537,"nblocks":23,"nevents":78255208,"nfiles":97,"nlumis":29820,"num_block":23,"num_event":78255208,"num_file":97,"num_lumi":29820}],
    "Run2018C":
    #[{"file_size":72209024389,"nblocks":5,"nevents":70027804,"nfiles":71,"nlumis":27664,"num_block":5,"num_event":70027804,"num_file":71,"num_lumi":27664}],
    [{"file_size":92056011770,"nblocks":7,"nevents":70027804,"nfiles":58,"nlumis":27664,"num_block":7,"num_event":70027804,"num_file":58,"num_lumi":27664}],
    "Run2018D":
    #[{"file_size":378484528342,"nblocks":16,"nevents":356543782,"nfiles":162,"nlumis":136336,"num_block":16,"num_event":356543782,"num_file":162,"num_lumi":136336}],
    [{"file_size":472923351224,"nblocks":48,"nevents":356933632,"nfiles":295,"nlumis":135175,"num_block":48,"num_event":356933632,"num_file":295,"num_lumi":135175}],
}

expectedCountsTT = {

    "MuonEgRun2016B":
    [{"nevents": (225271+32727796) }],
    "MuonEgRun2016C":
    [{"nevents": 15405678 }],
    "MuonEgRun2016D":
    [{"nevents": 23482352 }],
    "MuonEgRun2016E":
    [{"nevents": 22519303 }],
    "MuonEgRun2016F":
    [{"nevents": (14100826+1901339) }],
    "MuonEgRun2016G":
    [{"nevents": 33854612 }],
    "MuonEgRun2016H":
    [{"nevents": 29236516 }],

    "MuonEgRun2017B":
    [{"nevents": 4453465 }],
    "MuonEgRun2017C":
    [{"nevents": 15595214 }],
    "MuonEgRun2017D":
    [{"nevents": 9164365 }],
    "MuonEgRun2017E":
    [{"nevents": 19043421 }],
    "MuonEgRun2017F":
    [{"nevents": 25776363 }],

    "MuonEgRun2018A":
    [{"nevents": 32958503 }],
    "MuonEgRun2018B":
    [{"nevents": 16211567 }],
    "MuonEgRun2018C":
    [{"nevents": 15652198 }],
    "MuonEgRun2018D":
    [{"nevents": 71965111 }],

    "SingleMuonRun2016B":
    [{"nevents": (2789243+158145722) }],
    "SingleMuonRun2016C":
    [{"nevents": 67441308 }],
    "SingleMuonRun2016D":
    [{"nevents": 98017996 }],
    "SingleMuonRun2016E":
    [{"nevents": 90984718 }],
    "SingleMuonRun2016F":
    [{"nevents": (57465359 +8024195) }],
    "SingleMuonRun2016G":
    [{"nevents": 149916849 }],
    "SingleMuonRun2016H":
    [{"nevents": 174035164 }],

    "SingleMuonRun2017B":
    [{"nevents": 136300266 }],
    "SingleMuonRun2017C":
    [{"nevents": 165652756 }],
    "SingleMuonRun2017D":
    [{"nevents": 70361660 }],
    "SingleMuonRun2017E":
    [{"nevents": 154626280 }],
    "SingleMuonRun2017F":
    [{"nevents": 242140980 }],

    "SingleMuonRun2018A":
    [{"nevents": 241119080 }],
    "SingleMuonRun2018B":
    [{"nevents": 119800414 }],
    "SingleMuonRun2018C":
    [{"nevents": 109940654 }],
    "SingleMuonRun2018D":
    [{"nevents": 513909894 }],


    "TTTo2L2Nu2016_preVFP":
    [{"file_size":84760500188,"nblocks":68,"nevents":41364000,"nfiles":102,"nlumis":41364,"num_block":68,"num_event":41364000,"num_file":102,"num_lumi":41364}],

    "TTTo2L2Nu2016_postVFP":
    [{"file_size":99088695178,"nblocks":68,"nevents":48232000,"nfiles":102,"nlumis":48232,"num_block":68,"num_event":48232000,"num_file":102,"num_lumi":48232}],


    "TTToHadronic2016_preVFP":
    [{"file_size":207293651683,"nblocks":24,"nevents":97823000,"nfiles":112,"nlumis":97823,"num_block":24,"num_event":97823000,"num_file":112,"num_lumi":97823}],

    "TTToHadronic2016_postVFP":
    [{"file_size":239344381462,"nblocks":77,"nevents":112592000,"nfiles":159,"nlumis":112592,"num_block":77,"num_event":112592000,"num_file":159,"num_lumi":112592}],


    "TTToSemiLeptonic2016_preVFP":
    [{"file_size":291206720794,"nblocks":108,"nevents":138169000,"nfiles":236,"nlumis":138169,"num_block":108,"num_event":138169000,"num_file":236,"num_lumi":138169}],

    "TTToSemiLeptonic2016_postVFP":
    [{"file_size":335213535307,"nblocks":134,"nevents":158594000,"nfiles":266,"nlumis":158594,"num_block":134,"num_event":158594000,"num_file":266,"num_lumi":158594}],

    "TTTo2L2Nu2017":
    [{"file_size":231155466864,"nblocks":3,"nevents":106978000,"nfiles":85,"nlumis":106978,"num_block":3,"num_event":106978000,"num_file":85,"num_lumi":106978}],

    "TTToHadronic2017":
    [{"file_size":557578980393,"nblocks":4,"nevents":249247999,"nfiles":203,"nlumis":249248,"num_block":4,"num_event":249247999,"num_file":203,"num_lumi":249248}],

    "TTToSemiLeptonic2017":
    [{"file_size":790805135484,"nblocks":5,"nevents":355826000,"nfiles":305,"nlumis":355826,"num_block":5,"num_event":355826000,"num_file":305,"num_lumi":355826}],

    "TTToSemiLeptonic2018":
    [{"file_size":1055121317049,"nblocks":75,"nevents":486770000,"nfiles":465,"nlumis":486770,"num_block":75,"num_event":486770000,"num_file":465,"num_lumi":486770}],

    "TTToHadronic2018":
    [{"file_size":751904872328,"nblocks":74,"nevents":344028000,"nfiles":343,"nlumis":344028,"num_block":74,"num_event":344028000,"num_file":343,"num_lumi":344028}],

    "TTTo2L2Nu2018":
    [{"file_size":313342375408,"nblocks":59,"nevents":148470000,"nfiles":181,"nlumis":148470,"num_block":59,"num_event":148470000,"num_file":181,"num_lumi":148470}],


}


    


def getCounts(inFileName):
    inFile = ROOT.TFile.Open(inFileName)

    try: 
        cfHist = inFile.Get("cutflow/fourTag/unitWeight")
        nEvents = cfHist.GetBinContent(1)
    except:
        try:
            cfHist = inFile.Get("cutflow/unitWeight")
            nEvents = cfHist.GetBinContent(1)
        except:
            print "Error cant find cutflow/fourTag/unitWeight in ", inFileName
            nEvents = 0

    return nEvents


def nominal(run, nExpected, skimedFile):
        #skimedFile = "closureTests/nominal/"+run.replace("Run","data")+"/histsFromNanoAOD.root"
    
        #skimedFile = "/uscms/home/bryantp/nobackup/ZZ4b/"+run.replace("Run","data")+"/histsFromNanoAOD.root"
        #if not os.path.isfile(skimedFile):
        #    print "Skipping ",run,skimedFile,"not found"
        #    continue

        
        nSeen = getCounts(skimedFile)
        print run,"Expected",nExpected,"Seen",nSeen,"Ratio",float(nSeen)/nExpected

def doChunks(run, nExpected):

    nChunks = {
        "TTTo2L2Nu2016_postVFP":11,
        "TTTo2L2Nu2016_preVFP":11,
        "TTTo2L2Nu2017":9,
        "TTTo2L2Nu2018":19,
        "TTToHadronic2016_postVFP":16,
        "TTToHadronic2016_preVFP":12,
        "TTToHadronic2017":21,
        "TTToHadronic2018":35,
        "TTToSemiLeptonic2016_postVFP":27,
        "TTToSemiLeptonic2016_preVFP":24,
        "TTToSemiLeptonic2017":31,
        "TTToSemiLeptonic2018":47,

        "Run2016E": 2,
        "Run2016G": 2,
        "Run2016H": 2,
        "Run2017C": 2,
        "Run2017F": 2,
        "Run2018B": 2,
        "Run2018C": 2,
        "Run2018D": 6,
        "Run2018A": 3,
        }

    nSeen = 0

    maxRange = 2
    if run in nChunks:
        maxRange = nChunks[run]+1

    for i in range(1,maxRange):
        if i < 10:
            skimedFile = "root://cmseos.fnal.gov//store/user/bryantp/condor/"+run.replace("Run","data")+"_chunk0"+str(i)+"/histsFromNanoAOD.root"            
        else:
            skimedFile = "root://cmseos.fnal.gov//store/user/bryantp/condor/"+run.replace("Run","data")+"_chunk"+str(i)+"/histsFromNanoAOD.root"            
        nSeen += getCounts(skimedFile)

    print run,"Expected",nExpected,"Seen",nSeen,"Ratio",float(nSeen)/nExpected


def compCountTT():
    runs = expectedCountsTT.keys()
    runs.sort()
    
    for run in runs:


        nExpected = expectedCountsTT[run][0]['nevents']

    
        nominal(run,nExpected,  skimedFile = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/TTStudy/"+run.replace("Run","Data")+"/histsFromNanoAOD.root")
        #doChunks(run, nExpected)


def compCount():
    runs = expectedCounts.keys()
    runs.sort()
    
    for run in runs:


        nExpected = expectedCounts[run][0]['nevents']

    
        nominal(run,nExpected,  skimedFile = "root://cmseos.fnal.gov//store/user/bryantp/condor/"+run.replace("Run","data")+"/histsFromNanoAOD.root")
        #doChunks(run, nExpected)






if __name__ == "__main__":
    
    #compCountTT()
    compCount()
