import os

def getDSNames():

    filesEx = [
        "/store/data/Run2016B/BTagCSV/NANOAOD/ver1_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/00000/22C056C4-AC83-4A47-84E4-4AC07D76A472.root",
        "/store/data/Run2016B/BTagCSV/NANOAOD/ver1_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/00000/22C056C4-AC83-4A47-84E4-4AC07D76A472.root",  
        "/store/data/Run2016C/BTagCSV/NANOAOD/UL2016_MiniAODv1_NanoAODv2-v1/270000/0221608C-FB52-934A-B2DC-DF52B25A5CF9.root",  
        "/store/data/Run2016D/BTagCSV/NANOAOD/UL2016_MiniAODv1_NanoAODv2-v1/280000/01C47DF8-17E7-9E4D-8005-14E76AE8F5D8.root",  
        "/store/data/Run2016E/BTagCSV/NANOAOD/UL2016_MiniAODv1_NanoAODv2-v1/270000/004332C0-B1BD-8D45-86D9-44435370CCA6.root",  
        "/store/data/Run2016F/BTagCSV/NANOAOD/HIPM_UL2016_MiniAODv1_NanoAODv2-v1/230000/04D9561D-09C3-F44D-8F65-FD9DC24D6A57.root",  
        "/store/data/Run2016G/BTagCSV/NANOAOD/UL2016_MiniAODv1_NanoAODv2-v1/270000/05107559-FB76-3046-9473-1B0FD4F537D2.root",  
        "/store/data/Run2016H/BTagCSV/NANOAOD/UL2016_MiniAODv1_NanoAODv2-v1/270000/1224690C-81DC-A645-BFDD-AE443BED4924.root",  
        "/store/data/Run2017B/BTagCSV/NANOAOD/UL2017_MiniAODv1_NanoAODv2-v1/270000/007E779F-8B0D-074C-A339-FA5E621130F4.root",  
        "/store/data/Run2017C/BTagCSV/NANOAOD/UL2017_MiniAODv1_NanoAODv2-v1/270000/0503B391-616F-D248-A574-3E1D3837E829.root",  
        "/store/data/Run2017D/BTagCSV/NANOAOD/UL2017_MiniAODv1_NanoAODv2-v1/270000/5DFA79CA-BBF3-3A46-A185-73DB21FB76FB.root",  
        "/store/data/Run2017E/BTagCSV/NANOAOD/UL2017_MiniAODv1_NanoAODv2-v1/230000/0A180BFF-8DFD-A74C-A7CF-C196AC8F4798.root",  
        "/store/data/Run2017F/BTagCSV/NANOAOD/UL2017_MiniAODv1_NanoAODv2-v1/270000/0E16D31F-DDD7-7549-BE77-8369553891E1.root",  
        "/store/data/Run2018A/JetHT/NANOAOD/UL2018_MiniAODv1_NanoAODv2-v1/280000/0032C024-245C-4040-9C24-FB821DDA145B.root",  
        "/store/data/Run2018B/JetHT/NANOAOD/UL2018_MiniAODv1_NanoAODv2-v1/270000/A5F6328D-7C54-4647-A440-67EC28686957.root",  
        "/store/data/Run2018C/JetHT/NANOAOD/UL2018_MiniAODv1_NanoAODv2-v1/270000/1216B275-8FC0-7648-A534-3B885886A9E8.root",  
        "/store/data/Run2018D/JetHT/NANOAOD/UL2018_MiniAODv1_NanoAODv2-v1/00000/004B322A-0422-E14F-A8CC-ED7179CFB756.root",  
        "/store/mc/RunIISummer20UL16NanoAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v15-v1/00000/02D3C041-82C6-BC49-BE63-FA1FC6FEA67F.root",  
        "/store/mc/RunIISummer20UL16NanoAODAPVv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v9-v1/40000/03091902-BE37-B943-9039-F2B2D1380BB2.root",  
        "/store/mc/RunIISummer20UL17NanoAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v8-v1/100000/02F83A6C-4B60-C24E-B9CC-983CD690FD0D.root",  
        "/store/mc/RunIISummer20UL18NanoAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/10000/01A4D532-6ABA-2940-BE3E-DD5D96EBBCA5.root",  
        "/store/mc/RunIISummer20UL16NanoAODv2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v15-v1/00000/04DB6487-69D2-8148-989A-2F184E3432B3.root",  
        "/store/mc/RunIISummer20UL16NanoAODAPVv2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v9-v1/10000/0468E489-CF32-4B49-BF63-A7D5C272FDC3.root",  
        "/store/mc/RunIISummer19UL17NanoAOD/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v6-v1/270000/11EB27F8-9C42-4840-BBE6-7DBD6744B74C.root",  
        "/store/mc/RunIISummer20UL18NanoAODv2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/20000/01A179D4-6A3E-BD4F-BDD8-A3B3E99D44F9.root",  
        "/store/mc/RunIISummer20UL16NanoAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v15-v1/00000/04004BA9-A496-774E-85EB-812B9777F78E.root",  
        "/store/mc/RunIISummer20UL16NanoAODAPVv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v9-v1/10000/04798F8C-5C5B-9148-A9F1-2F67E7073D63.root",  
        "/store/mc/RunIISummer20UL17NanoAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mc2017_realistic_v8-v1/00000/12B6E2C4-456F-7546-8395-EE557A14B441.root",  
        "/store/mc/RunIISummer20UL18NanoAODv2/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v15_L1v1-v1/270000/019426EE-3D50-1249-B266-F6DBA0AFE3B5.root",  
    ]
     
    for f in filesEx:
     
        cmd = 'dasgoclient -query="dataset file='+f+'"'
        os.system(cmd)
        print cmd

def getCounts():

    dataSets =[
        "/BTagCSV/Run2016B-ver1_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2016B-ver2_HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2016C-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2016D-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2016E-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2016F-HIPM_UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2016F-UL2016_MiniAODv1_NanoAODv2-v2/NANOAOD",
        "/BTagCSV/Run2016G-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2016H-UL2016_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2017B-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2017C-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2017D-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2017E-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/BTagCSV/Run2017F-UL2017_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/JetHT/Run2018A-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/JetHT/Run2018B-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/JetHT/Run2018C-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/JetHT/Run2018D-UL2018_MiniAODv1_NanoAODv2-v1/NANOAOD",
        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM",
        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM",
        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM",
        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM",
        "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM",
        "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM",
        "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM",
        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv2-106X_mcRun2_asymptotic_v15-v1/NANOAODSIM",
        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv2-106X_mcRun2_asymptotic_preVFP_v9-v1/NANOAODSIM",
        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv2-106X_mc2017_realistic_v8-v1/NANOAODSIM",
        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM",



#        "/BTagCSV/Run2016B-02Apr2020_ver2-v1/NANOAOD",
#        "/BTagCSV/Run2016C-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2016D-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2016E-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2016F-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2016G-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2016H-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2017B-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2017C-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2017D-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2017E-02Apr2020-v1/NANOAOD",
#        "/BTagCSV/Run2017F-02Apr2020-v1/NANOAOD",
#        "/JetHT/Run2018A-02Apr2020-v1/NANOAOD",
#        "/JetHT/Run2018B-02Apr2020-v1/NANOAOD",
#        "/JetHT/Run2018C-02Apr2020-v1/NANOAOD",
#        "/JetHT/Run2018D-02Apr2020-v1/NANOAOD",
#        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL16NanoAOD-106X_mcRun2_asymptotic_v13-v1/NANOAODSIM",
#        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v4/NANOAODSIM",
#        "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v2/NANOAODSIM",
#        "/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM",
#        "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v1/NANOAODSIM",
#        "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v2/NANOAODSIM",
#        "/TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIISummer16NanoAODv7-PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1/NANOAODSIM",
#        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL17NanoAOD-106X_mc2017_realistic_v6-v1/NANOAODSIM",
#        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer19UL18NanoAOD-106X_upgrade2018_realistic_v11_L1v1-v1/NANOAODSIM",
    ]
    
    for d in dataSets:
        print d
        cmd = 'das_client  --query="summary dataset='+d+'"'
        os.system(cmd)



#getDSNames()
getCounts()
