import os

def getDSNames():

    filesEx = [
        "/store/data/Run2016B/BTagCSV/NANOAOD/02Apr2020_ver2-v1/10000/05A98853-B664-684F-8745-1B581890C411.root" ,
        "/store/data/Run2016C/BTagCSV/NANOAOD/02Apr2020-v1/20000/3ACD774B-20B1-6046-BB19-427DB058085E.root"      ,
        "/store/data/Run2016D/BTagCSV/NANOAOD/02Apr2020-v1/60000/06707E38-2155-7C43-B7FC-9EF0CCC2E059.root"      ,
        "/store/data/Run2016E/BTagCSV/NANOAOD/02Apr2020-v1/50000/34B622A8-0C71-0A47-ADB3-C9E57F63995D.root"      ,
        "/store/data/Run2016F/BTagCSV/NANOAOD/02Apr2020-v1/240000/0788FD1F-8033-9549-96CB-8960A1F32CE8.root"     ,
        "/store/data/Run2016G/BTagCSV/NANOAOD/02Apr2020-v1/40000/C172D9D1-3400-1F45-A6A9-826578100141.root"  ,
        "/store/data/Run2016H/BTagCSV/NANOAOD/02Apr2020-v1/240000/E9EA66AB-371C-B740-9E80-11E11FE9B876.root" ,
        "/store/data/Run2017B/BTagCSV/NANOAOD/02Apr2020-v1/240000/2D9F3ABA-A595-C54B-81BB-159F3D1696A4.root" ,
        "/store/data/Run2017C/BTagCSV/NANOAOD/02Apr2020-v1/240000/01E46230-E7D9-8C4C-8AEE-87281504AE80.root" ,
        "/store/data/Run2017D/BTagCSV/NANOAOD/02Apr2020-v1/40000/011F3E7F-3791-354F-8F71-94E70AD9CC18.root"  ,
        "/store/data/Run2017E/BTagCSV/NANOAOD/02Apr2020-v1/50000/1D5D81E2-B72C-3141-B690-6867DE57B516.root"  ,
        "/store/data/Run2017F/BTagCSV/NANOAOD/02Apr2020-v1/50000/8E796A14-3F29-2B4E-BC07-D15209FCFD45.root"  ,
        "/store/data/Run2018A/JetHT/NANOAOD/02Apr2020-v1/40000/B5CBAF32-2E81-614C-8772-540FA7BA9313.root"   ,
        "/store/data/Run2018B/JetHT/NANOAOD/02Apr2020-v1/250000/E6DFB07D-7B0D-E74E-80AD-DDD1B5946A44.root" ,
        "/store/data/Run2018C/JetHT/NANOAOD/02Apr2020-v1/50000/C36E8F82-E066-9A49-9B41-5E50C6B25721.root" ,
        "/store/data/Run2018D/JetHT/NANOAOD/02Apr2020-v1/50000/EA87E04A-A9A1-0548-AB61-DA521E2625BA.root" ,
        "/store/mc/RunIISummer19UL16NanoAOD/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v13-v1/110000/0821A5F6-A0BF-9D4B-94FE-144CF55BC196.root",
        "/store/mc/RunIISummer19UL17NanoAOD/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v6-v4/100000/01A5544B-5290-3245-9FCA-5E25300AD920.root",
        "/store/mc/RunIISummer19UL18NanoAOD/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/100000/014F659E-D84A-FB46-92DE-BFA00AC6C27E.root",
        "/store/mc/RunIISummer16NanoAODv7/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/NANOAODSIM/PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/100000/1A511872-6E69-6945-8499-B2696EC437AD.root",
        "/store/mc/RunIISummer19UL17NanoAOD/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v6-v1/270000/11EB27F8-9C42-4840-BBE6-7DBD6744B74C.root",
        "/store/mc/RunIISummer19UL18NanoAOD/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/00000/378F62B4-DB98-514E-9A9B-337B401DB9A0.root",

        "/store/mc/RunIISummer16NanoAODv7/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/NANOAODSIM/PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/100000/1126BEFE-9516-224E-B35B-0285E466F4DA.root",
        "/store/mc/RunIISummer19UL17NanoAOD/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v6-v1/260000/0FC4E2C0-4D8D-3B48-8144-A9C737EFB46E.root",
        "/store/mc/RunIISummer19UL18NanoAOD/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v11_L1v1-v1/110000/0F1E4AEE-6AEC-C247-8C4D-ABC545CFBC4F.root",

    ]
     
    for f in filesEx:
     
        cmd = 'dasgoclient -query="dataset file='+f+'"'
        os.system(cmd)


def getCounts():

    dataSets =[
        "/BTagCSV/Run2016B-02Apr2020_ver2-v1/NANOAOD",
        "/BTagCSV/Run2016C-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2016D-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2016E-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2016F-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2016G-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2016H-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2017B-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2017C-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2017D-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2017E-02Apr2020-v1/NANOAOD",
        "/BTagCSV/Run2017F-02Apr2020-v1/NANOAOD",
        "/JetHT/Run2018A-02Apr2020-v1/NANOAOD",
        "/JetHT/Run2018B-02Apr2020-v1/NANOAOD",
        "/JetHT/Run2018C-02Apr2020-v1/NANOAOD",
        "/JetHT/Run2018D-02Apr2020-v1/NANOAOD",
        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL16NanoAOD-106X_mcRun2_asymptotic_v13-v1/NANOAODSIM",
        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v4/NANOAODSIM",
        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v2/NANOAODSIM",
        "/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM",
        "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v1/NANOAODSIM",
        "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v2/NANOAODSIM",
        "/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM",
        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v1/NANOAODSIM",
        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v1/NANOAODSIM",
    ]
    
    for d in dataSets:
        print d
        cmd = 'das_client  --query="summary dataset='+d+'"'
        os.system(cmd)



getDSNames()
getCounts()
