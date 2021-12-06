#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>
#include <signal.h>

#include "ZZ4b/nTupleAnalysis/interface/analysis.h"
#include "nTupleAnalysis/baseClasses/interface/helpers.h"

using std::cout;  using std::endl;

using namespace nTupleAnalysis;

analysis::analysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _isMC, bool _blind, std::string _year, std::string histDetailLevel, 
		   bool _doReweight, bool _debug, bool _fastSkim, bool doTrigEmulation, bool _calcTrigWeights, bool _useMCTurnOns, bool _isDataMCMix, bool usePreCalcBTagSFs,
		   std::string bjetSF, std::string btagVariations,
		   std::string JECSyst, std::string friendFile,
		   bool _looseSkim, std::string FvTName, std::string reweight4bName, std::string reweightDvTName,
       std::string bdtWeightFile, std::string bdtMethods){
  if(_debug) std::cout<<"In analysis constructor"<<std::endl;
  debug      = _debug;
  doReweight     = _doReweight;
  isMC       = _isMC;
  isDataMCMix = _isDataMCMix;
  blind      = _blind;
  year       = _year;
  events     = _events;
  looseSkim  = _looseSkim;
  calcTrigWeights = _calcTrigWeights;
  useMCTurnOns = _useMCTurnOns;
  events->SetBranchStatus("*", 0);

  //keep branches needed for JEC Uncertainties
  if(isMC){
    events->SetBranchStatus("nGenJet"  , 1);
    events->SetBranchStatus( "GenJet_*", 1);
  }
  events->SetBranchStatus(   "MET*", 1);
  events->SetBranchStatus("RawMET*", 1);
  events->SetBranchStatus("fixedGridRhoFastjetAll", 1);
  events->SetBranchStatus("Jet_rawFactor", 1);
  events->SetBranchStatus("Jet_area", 1);
  events->SetBranchStatus("Jet_neEmEF", 1);
  events->SetBranchStatus("Jet_chEmEF", 1);

  if(JECSyst!=""){
    std::cout << "events->AddFriend(\"Friends\", "<<friendFile<<")" << " for JEC Systematic " << JECSyst << std::endl;
    events->AddFriend("Friends", friendFile.c_str());
  }

  runs       = _runs;
  fastSkim = _fastSkim;
  

  //Calculate MC weight denominator
  if(isMC){
    if(debug) runs->Print();
    runs->SetBranchStatus("*", 0);
    Long64_t loadStatus = runs->LoadTree(0);
    if(loadStatus < 0){
      std::cout << "ERROR in loading tree for entry index: " << 0 << "; load status = " << loadStatus << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if(runs->FindBranch("genEventCount")){
      std::cout << "Runs has genEventCount" << std::endl;
      inputBranch(runs, "genEventCount", genEventCount);
      inputBranch(runs, "genEventSumw",  genEventSumw);
      inputBranch(runs, "genEventSumw2", genEventSumw2);
    }else{//for some presumably idiotic reason, NANOAODv6 added an underscore to these branch names...
      std::cout << "Runs has genEventCount_" << std::endl;
      inputBranch(runs, "genEventCount_", genEventCount);
      inputBranch(runs, "genEventSumw_",  genEventSumw);
      inputBranch(runs, "genEventSumw2_", genEventSumw2);      
    }
    for(int r = 0; r < runs->GetEntries(); r++){
      runs->GetEntry(r);
      mcEventCount += genEventCount;
      mcEventSumw  += genEventSumw;
      mcEventSumw2 += genEventSumw2;
    }
    cout << "mcEventCount " << mcEventCount << " | mcEventSumw " << mcEventSumw << endl;
  }

  bool doWeightStudy = nTupleAnalysis::findSubStr(histDetailLevel,"weightStudy");

  lumiBlocks = _lumiBlocks;
  event      = new eventData(events, isMC, year, debug, fastSkim, doTrigEmulation, calcTrigWeights, useMCTurnOns, isDataMCMix, doReweight, bjetSF, btagVariations, JECSyst, looseSkim, usePreCalcBTagSFs, FvTName, reweight4bName, reweightDvTName, doWeightStudy, bdtWeightFile, bdtMethods);  
  treeEvents = events->GetEntries();
  cutflow    = new tagCutflowHists("cutflow", fs, isMC, debug);
  if(isDataMCMix){
    cutflow->AddCut("mixedEventIsData_3plus4Tag");
    cutflow->AddCut("mixedEventIsMC_3plus4Tag");
    cutflow->AddCut("mixedEventIsData");
    cutflow->AddCut("mixedEventIsMC");
  }
  cutflow->AddCut("lumiMask");
  cutflow->AddCut("HLT");
  cutflow->AddCut("jetMultiplicity");
  cutflow->AddCut("bTags");
  cutflow->AddCut("DijetMass");
  cutflow->AddCut("MDRs");
  
  lumiCounts    = new lumiHists("lumiHists", fs, year, false, debug);

  if(nTupleAnalysis::findSubStr(histDetailLevel,"allEvents"))     allEvents     = new eventHists("allEvents",     fs, false, isMC, blind, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passPreSel"))    passPreSel    = new   tagHists("passPreSel",    fs, true,  isMC, blind, histDetailLevel, debug);
  //if(nTupleAnalysis::findSubStr(histDetailLevel,"passDijetMass")) passDijetMass = new   tagHists("passDijetMass", fs, true,  isMC, blind, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passMDRs"))      passMDRs      = new   tagHists("passMDRs",      fs, true,  isMC, blind, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passSvB"))       passSvB       = new   tagHists("passSvB",       fs, true,  isMC, blind, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passMjjOth"))    passMjjOth    = new   tagHists("passMjjOth",    fs, true,  isMC, blind, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"failrWbW2"))     failrWbW2     = new   tagHists("failrWbW2",     fs, true,  isMC, blind, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passMuon"))      passMuon      = new   tagHists("passMuon",      fs, true,  isMC, blind, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passDvT05"))     passDvT05     = new   tagHists("passDvT05",     fs, true,  isMC, blind, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passTTCR"))      passTTCR      = new   tagHists("passTTCR",      fs, true,  isMC, blind, histDetailLevel, debug);

  if(allEvents)     std::cout << "Turning on allEvents Hists" << std::endl; 
  if(passPreSel)    std::cout << "Turning on passPreSel Hists" << std::endl; 
  //if(passDijetMass) std::cout << "Turning on passDijetMass Hists" << std::endl; 
  if(passMDRs)      std::cout << "Turning on passMDRs Hists" << std::endl; 
  if(passSvB)       std::cout << "Turning on passSvB Hists" << std::endl; 
  if(passMjjOth)    std::cout << "Turning on passMjjOth Hists" << std::endl; 
  if(failrWbW2)     std::cout << "Turning on failrWbW2 Hists" << std::endl; 
  if(passMuon)      std::cout << "Turning on passMuon Hists" << std::endl; 
  if(passDvT05)     std::cout << "Turning on passDvT05 Hists" << std::endl; 
  if(passTTCR)      std::cout << "Turning on passTTCR Hists" << std::endl; 



  if(nTupleAnalysis::findSubStr(histDetailLevel,"trigStudy") && doTrigEmulation){
    std::cout << "Turning on Trigger Study Hists" << std::endl; 
    trigStudy     = new triggerStudy("passMDRs_",     fs, year, isMC, blind, histDetailLevel, debug);
    if(passMjjOth) trigStudyMjjOth  = new triggerStudy("passMjjOth_",     fs, year, isMC, blind, histDetailLevel, debug);
  }

  histFile = &fs.file();

} 




void analysis::createPicoAOD(std::string fileName, bool copyInputPicoAOD){
  writePicoAOD = true;
  picoAODFile = TFile::Open(fileName.c_str() , "RECREATE");
  if(copyInputPicoAOD){
    //We are making a skim so we can directly clone the input TTree
    picoAODEvents = events->CloneTree(0);
  }else{
    //We are making a derived TTree which changes some of the branches of the input TTree so start from scratch
    if(emulate4bFrom3b){
      picoAODEvents = new TTree("Events", "Events Emulated 4b from 3b");
    }else{
      picoAODEvents = new TTree("Events", "Events from Mixing");
    }
    createPicoAODBranches();
  }
  addDerivedQuantitiesToPicoAOD();

  if(isMC && calcTrigWeights){
    picoAODEvents->Branch("trigWeight_MC",     &event->trigWeight_MC      );
    picoAODEvents->Branch("trigWeight_Data",   &event->trigWeight_Data    );
  }

  picoAODRuns       = runs      ->CloneTree();
  picoAODLumiBlocks = lumiBlocks->CloneTree();
}



 void analysis::createPicoAODBranches(){
  cout << " analysis::createPicoAODBranches " << endl;

  //
  //  Initial Event Data
  //
  outputBranch(picoAODEvents, "run",               m_run, "i");
  outputBranch(picoAODEvents, "luminosityBlock",   m_lumiBlock,  "i");
  outputBranch(picoAODEvents, "event",             m_event,  "l");

  if(isMC){
    outputBranch(picoAODEvents, "genWeight",       m_genWeight,  "F");
    outputBranch(picoAODEvents, "bTagSF",          m_bTagSF,  "F");
  }

  m_mixed_jetData  = new nTupleAnalysis::jetData("Jet",picoAODEvents, false, "");
  m_mixed_muonData = new nTupleAnalysis::muonData("Muon",picoAODEvents, false );
  m_mixed_elecData = new nTupleAnalysis::elecData("Elec",picoAODEvents, false );

  if(isMC)
    m_mixed_truthParticle = new nTupleAnalysis::truthParticle("GenPart",picoAODEvents, false );
  
  outputBranch(picoAODEvents, "PV_npvs",         m_nPVs, "I");
  outputBranch(picoAODEvents, "PV_npvsGood",     m_nPVsGood, "I");
  outputBranch(picoAODEvents, "ttbarWeight",     m_ttbarWeight,  "F");

  //triggers
  for(auto &trigger: event-> L1_triggers) outputBranch(picoAODEvents, trigger.first, trigger.second, "O");
  for(auto &trigger: event->HLT_triggers) outputBranch(picoAODEvents, trigger.first, trigger.second, "O");
  //trigObjs = new trigData("TrigObj", tree);

  //
  //  Hemisphere Mixed branches
  //
  if(loadHSphereFile){
    if(debug) cout << " Making Hemisphere branches " << endl;
    //
    //  Hemisphere Event Data
    //
    outputBranch(picoAODEvents,     "h1_run"               ,   m_h1_run               ,         "i");
    outputBranch(picoAODEvents,     "h1_event"             ,   m_h1_event             ,         "l");
    outputBranch(picoAODEvents,     "h1_eventWeight"       ,   m_h1_eventWeight       ,         "F");
    outputBranch(picoAODEvents,     "h1_hemiSign"          ,   m_h1_hemiSign          ,         "O");
    outputBranch(picoAODEvents,     "h1_NJet"              ,   m_h1_NJet              ,         "i");     
    outputBranch(picoAODEvents,     "h1_NBJet"             ,   m_h1_NBJet             ,         "i");     
    outputBranch(picoAODEvents,     "h1_NNonSelJet"        ,   m_h1_NNonSelJet        ,         "i");     
    outputBranch(picoAODEvents,     "h1_matchCode"         ,   m_h1_matchCode         ,         "i");     
    outputBranch(picoAODEvents,     "h1_pz"                ,   m_h1_pz                ,         "F");
    outputBranch(picoAODEvents,     "h1_pz_sig"            ,   m_h1_pz_sig            ,         "F");
    outputBranch(picoAODEvents,     "h1_match_pz"          ,   m_h1_match_pz          ,         "F");
    outputBranch(picoAODEvents,     "h1_sumpt_t"           ,   m_h1_sumpt_t           ,         "F");
    outputBranch(picoAODEvents,     "h1_sumpt_t_sig"       ,   m_h1_sumpt_t_sig       ,         "F");
    outputBranch(picoAODEvents,     "h1_match_sumpt_t"     ,   m_h1_match_sumpt_t     ,         "F");
    outputBranch(picoAODEvents,     "h1_sumpt_ta"          ,   m_h1_sumpt_ta          ,         "F");
    outputBranch(picoAODEvents,     "h1_sumpt_ta_sig"      ,   m_h1_sumpt_ta_sig      ,         "F");
    outputBranch(picoAODEvents,     "h1_match_sumpt_ta"    ,   m_h1_match_sumpt_ta    ,         "F");
    outputBranch(picoAODEvents,     "h1_combinedMass"      ,   m_h1_combinedMass      ,         "F");
    outputBranch(picoAODEvents,     "h1_combinedMass_sig"  ,   m_h1_combinedMass_sig  ,         "F");
    outputBranch(picoAODEvents,     "h1_match_combinedMass",   m_h1_match_combinedMass,         "F");
    outputBranch(picoAODEvents,     "h1_match_dist"        ,   m_h1_match_dist        ,         "F");


    outputBranch(picoAODEvents,     "h2_run"               ,   m_h2_run               ,         "i");
    outputBranch(picoAODEvents,     "h2_event"             ,   m_h2_event             ,         "l");
    outputBranch(picoAODEvents,     "h2_eventWeight"       ,   m_h2_eventWeight       ,         "F");
    outputBranch(picoAODEvents,     "h2_hemiSign"          ,   m_h2_hemiSign          ,         "O");
    outputBranch(picoAODEvents,     "h2_NJet"              ,   m_h2_NJet              ,         "i");     
    outputBranch(picoAODEvents,     "h2_NBJet"             ,   m_h2_NBJet             ,         "i");     
    outputBranch(picoAODEvents,     "h2_NNonSelJet"        ,   m_h2_NNonSelJet        ,         "i");     
    outputBranch(picoAODEvents,     "h2_matchCode"         ,   m_h2_matchCode         ,         "i");     
    outputBranch(picoAODEvents,     "h2_pz"                ,   m_h2_pz                ,         "F");
    outputBranch(picoAODEvents,     "h2_pz_sig"            ,   m_h2_pz_sig            ,         "F");
    outputBranch(picoAODEvents,     "h2_match_pz"          ,   m_h2_match_pz          ,         "F");
    outputBranch(picoAODEvents,     "h2_sumpt_t"           ,   m_h2_sumpt_t           ,         "F");
    outputBranch(picoAODEvents,     "h2_sumpt_t_sig"       ,   m_h2_sumpt_t_sig       ,         "F");
    outputBranch(picoAODEvents,     "h2_match_sumpt_t"     ,   m_h2_match_sumpt_t     ,         "F");
    outputBranch(picoAODEvents,     "h2_sumpt_ta"          ,   m_h2_sumpt_ta          ,         "F");
    outputBranch(picoAODEvents,     "h2_sumpt_ta_sig"      ,   m_h2_sumpt_ta_sig      ,         "F");
    outputBranch(picoAODEvents,     "h2_match_sumpt_ta"    ,   m_h2_match_sumpt_ta    ,         "F");
    outputBranch(picoAODEvents,     "h2_combinedMass"      ,   m_h2_combinedMass      ,         "F");
    outputBranch(picoAODEvents,     "h2_combinedMass_sig"  ,   m_h2_combinedMass_sig  ,         "F");
    outputBranch(picoAODEvents,     "h2_match_combinedMass",   m_h2_match_combinedMass,         "F");
    outputBranch(picoAODEvents,     "h2_match_dist"        ,   m_h2_match_dist        ,         "F");

  }

}


void analysis::picoAODFillEvents(){
  if(debug) std::cout << "analysis::picoAODFillEvents()" << std::endl;
  if(alreadyFilled){
    if(debug) std::cout << "analysis::picoAODFillEvents() alreadyFilled" << std::endl;
    //std::cout << "ERROR: Filling picoAOD with same event twice" << std::endl;
    return;
  }
  alreadyFilled = true;
  //if(m4jPrevious == event->m4j) std::cout << "WARNING: previous event had identical m4j = " << m4jPrevious << std::endl;

  assert( !(event->ZZSR && event->ZZSB) );
  assert( !(event->ZZSR && event->ZZCR) );
  assert( !(event->ZZSB && event->ZZCR) );

  assert( !(event->ZHSR && event->ZHSB) );
  assert( !(event->ZHSR && event->ZHCR) );
  assert( !(event->ZHSB && event->ZHCR) );

  assert( !(event->SR && event->SB) );
  assert( !(event->SR && event->CR) );
  // assert( !(event->SB && event->CR) ); // Changed SB to contain CR

  if(loadHSphereFile || emulate4bFrom3b){
    //cout << "Loading " << endl;
    //cout << event->run <<  " " << event->event << endl;
    //cout << "Jets: " << endl;
    //for(const jetPtr& j: event->allJets){
    //  cout << "\t " << j->pt << " / " << j->eta << " / " << j->phi << endl;
    //}

    m_run       = event->run;
    m_lumiBlock = event->lumiBlock;
    m_event     = event->event;

    if(isMC){
      m_genWeight     = event->genWeight;
      m_bTagSF        = event->bTagSF;
    }

    //
    //  Undo the bjet reg corr if applied
    //
    std::vector<bool> reApplyBJetReg;
    for(const jetPtr &jet: event->allJets){
      if(jet->AppliedBRegression()) {
	jet->undo_bRegression();
	reApplyBJetReg.push_back(true);
      }else{
	reApplyBJetReg.push_back(false);
      }
    }
    m_mixed_jetData ->writeJets(event->allJets);

    for(unsigned int iJet = 0; iJet < event->allJets.size(); ++iJet){
      if(reApplyBJetReg.at(iJet)) event->allJets.at(iJet)->bRegression();
    }

    m_mixed_muonData->writeMuons(event->allMuons);
    m_mixed_elecData->writeElecs(event->allElecs);

    if(isMC)
      m_mixed_truthParticle->writeTruth(event->truth->truthParticles->getParticles());

    m_nPVs = event->nPVs;
    m_nPVsGood = event->nPVsGood;    
    m_ttbarWeight   = event->ttbarWeight;


    if(loadHSphereFile){
        hemisphereMixTool* thisHMixTool = nullptr;
        if(event->threeTag) thisHMixTool = hMixToolLoad3Tag;
        if(event->fourTag)  thisHMixTool = hMixToolLoad4Tag;
        assert(thisHMixTool);
    
        m_h1_run                = thisHMixTool->m_h1_run                ;
        m_h1_event              = thisHMixTool->m_h1_event              ;
        m_h1_eventWeight        = thisHMixTool->m_h1_eventWeight        ;
        m_h1_hemiSign           = thisHMixTool->m_h1_hemiSign           ;
        m_h1_NJet               = thisHMixTool->m_h1_NJet               ;
        m_h1_NBJet              = thisHMixTool->m_h1_NBJet              ;
        m_h1_NNonSelJet         = thisHMixTool->m_h1_NNonSelJet         ;
        m_h1_matchCode          = thisHMixTool->m_h1_matchCode          ;
        m_h1_pz                 = thisHMixTool->m_h1_pz                 ;
        m_h1_pz_sig             = thisHMixTool->m_h1_pz_sig             ;
        m_h1_match_pz           = thisHMixTool->m_h1_match_pz           ;
        m_h1_sumpt_t            = thisHMixTool->m_h1_sumpt_t            ;
        m_h1_sumpt_t_sig        = thisHMixTool->m_h1_sumpt_t_sig        ;
        m_h1_match_sumpt_t      = thisHMixTool->m_h1_match_sumpt_t      ;
        m_h1_sumpt_ta           = thisHMixTool->m_h1_sumpt_ta           ;
        m_h1_sumpt_ta_sig       = thisHMixTool->m_h1_sumpt_ta_sig       ;
        m_h1_match_sumpt_ta     = thisHMixTool->m_h1_match_sumpt_ta     ;
        m_h1_combinedMass       = thisHMixTool->m_h1_combinedMass       ;
        m_h1_combinedMass_sig   = thisHMixTool->m_h1_combinedMass_sig   ;
        m_h1_match_combinedMass = thisHMixTool->m_h1_match_combinedMass ;
        m_h1_match_dist         = thisHMixTool->m_h1_match_dist         ;
    
    
        m_h2_run                = thisHMixTool->m_h2_run                ;
        m_h2_event              = thisHMixTool->m_h2_event              ;
        m_h2_eventWeight        = thisHMixTool->m_h2_eventWeight        ;
        m_h2_hemiSign           = thisHMixTool->m_h2_hemiSign           ;
        m_h2_NJet               = thisHMixTool->m_h2_NJet               ;
        m_h2_NBJet              = thisHMixTool->m_h2_NBJet              ;
        m_h2_NNonSelJet         = thisHMixTool->m_h2_NNonSelJet         ;
        m_h2_matchCode          = thisHMixTool->m_h2_matchCode          ;
        m_h2_pz                 = thisHMixTool->m_h2_pz                 ;
        m_h2_pz_sig             = thisHMixTool->m_h2_pz_sig             ;
        m_h2_match_pz           = thisHMixTool->m_h2_match_pz           ;
        m_h2_sumpt_t            = thisHMixTool->m_h2_sumpt_t            ;
        m_h2_sumpt_t_sig        = thisHMixTool->m_h2_sumpt_t_sig        ;
        m_h2_match_sumpt_t      = thisHMixTool->m_h2_match_sumpt_t      ;
        m_h2_sumpt_ta           = thisHMixTool->m_h2_sumpt_ta           ;
        m_h2_sumpt_ta_sig       = thisHMixTool->m_h2_sumpt_ta_sig       ;
        m_h2_match_sumpt_ta     = thisHMixTool->m_h2_match_sumpt_ta     ;
        m_h2_combinedMass       = thisHMixTool->m_h2_combinedMass       ;
        m_h2_combinedMass_sig   = thisHMixTool->m_h2_combinedMass_sig   ;
        m_h2_match_combinedMass = thisHMixTool->m_h2_match_combinedMass ;
        m_h2_match_dist         = thisHMixTool->m_h2_match_dist         ;
    }    
  }//end if(loadHSphereFile || emulate4bFrom3b) clause

  if(debug) std::cout << "picoAODEvents->Fill()" << std::endl;
  picoAODEvents->Fill();  
  if(debug) std::cout << "analysis::picoAODFillEvents() done" << std::endl;
  return;
}

void analysis::createHemisphereLibrary(std::string fileName, fwlite::TFileService& fs){

  //
  // Hemisphere Mixing
  //
  hMixToolCreate3Tag = new hemisphereMixTool("3TagEvents", fileName, std::vector<std::string>(), true, fs, -1, debug, true, false, false, true);
  hMixToolCreate4Tag = new hemisphereMixTool("4TagEvents", fileName, std::vector<std::string>(), true, fs, -1, debug, true, false, false, true);
  writeHSphereFile = true;
  writePicoAODBeforeDiJetMass = true;
}


void analysis::loadHemisphereLibrary(std::vector<std::string> hLibs_3tag, std::vector<std::string> hLibs_4tag, fwlite::TFileService& fs, int maxNHemis, bool useHemiWeights, float mcHemiWeight){

  //
  // Load Hemisphere Mixing 
  //
  hMixToolLoad3Tag = new hemisphereMixTool("3TagEvents", "dummyName", hLibs_3tag, false, fs, maxNHemis, debug, true, false, false, true);
  hMixToolLoad3Tag->m_useHemiWeights = useHemiWeights;
  hMixToolLoad3Tag->m_mcHemiWeight   = mcHemiWeight;

  hMixToolLoad4Tag = new hemisphereMixTool("4TagEvents", "dummyName", hLibs_4tag, false, fs, maxNHemis, debug, true, false, false, true);
  hMixToolLoad4Tag->m_useHemiWeights = useHemiWeights;
  hMixToolLoad4Tag->m_mcHemiWeight   = mcHemiWeight;

  loadHSphereFile = true;
}


void analysis::addDerivedQuantitiesToPicoAOD(){
  cout << "analysis::addDerivedQuantitiesToPicoAOD()" << endl;
  if(fastSkim){
    cout<<"In fastSkim mode, skip adding derived quantities to picoAOD"<<endl;
    return;
  }
  picoAODEvents->Branch("pseudoTagWeight",   &event->pseudoTagWeight  );
  picoAODEvents->Branch("mcPseudoTagWeight", &event->mcPseudoTagWeight);

  for(const std::string& jcmName : event->jcmNames){
    picoAODEvents->Branch(("pseudoTagWeight_"+jcmName  ).c_str(), &event->pseudoTagWeightMap[jcmName]  );
    picoAODEvents->Branch(("mcPseudoTagWeight_"+jcmName).c_str(), &event->mcPseudoTagWeightMap[jcmName]);
  }

  picoAODEvents->Branch("weight", &event->weight);
  picoAODEvents->Branch("threeTag", &event->threeTag);
  picoAODEvents->Branch("fourTag", &event->fourTag);
  picoAODEvents->Branch("nPVsGood", &event->nPVsGood);
  picoAODEvents->Branch("canJet0_pt" , &event->canJet0_pt ); picoAODEvents->Branch("canJet1_pt" , &event->canJet1_pt ); picoAODEvents->Branch("canJet2_pt" , &event->canJet2_pt ); picoAODEvents->Branch("canJet3_pt" , &event->canJet3_pt );
  picoAODEvents->Branch("canJet0_eta", &event->canJet0_eta); picoAODEvents->Branch("canJet1_eta", &event->canJet1_eta); picoAODEvents->Branch("canJet2_eta", &event->canJet2_eta); picoAODEvents->Branch("canJet3_eta", &event->canJet3_eta);
  picoAODEvents->Branch("canJet0_phi", &event->canJet0_phi); picoAODEvents->Branch("canJet1_phi", &event->canJet1_phi); picoAODEvents->Branch("canJet2_phi", &event->canJet2_phi); picoAODEvents->Branch("canJet3_phi", &event->canJet3_phi);
  picoAODEvents->Branch("canJet0_m"  , &event->canJet0_m  ); picoAODEvents->Branch("canJet1_m"  , &event->canJet1_m  ); picoAODEvents->Branch("canJet2_m"  , &event->canJet2_m  ); picoAODEvents->Branch("canJet3_m"  , &event->canJet3_m  );
  picoAODEvents->Branch("d01TruthMatch", &event->d01TruthMatch);
  picoAODEvents->Branch("d23TruthMatch", &event->d23TruthMatch);
  picoAODEvents->Branch("d02TruthMatch", &event->d02TruthMatch);
  picoAODEvents->Branch("d13TruthMatch", &event->d13TruthMatch);
  picoAODEvents->Branch("d03TruthMatch", &event->d03TruthMatch);
  picoAODEvents->Branch("d12TruthMatch", &event->d12TruthMatch);
  picoAODEvents->Branch("truthMatch", &event->truthMatch);
  picoAODEvents->Branch("selectedViewTruthMatch", &event->selectedViewTruthMatch);
  picoAODEvents->Branch("dRjjClose", &event->dRjjClose);
  picoAODEvents->Branch("dRjjOther", &event->dRjjOther);
  picoAODEvents->Branch("aveAbsEta", &event->aveAbsEta);
  picoAODEvents->Branch("aveAbsEtaOth", &event->aveAbsEtaOth);
  picoAODEvents->Branch("nAllNotCanJets", &event->nAllNotCanJets);
  picoAODEvents->Branch("notCanJet_pt",  event->notCanJet_pt,  "notCanJet_pt[nAllNotCanJets]/F");
  picoAODEvents->Branch("notCanJet_eta", event->notCanJet_eta, "notCanJet_eta[nAllNotCanJets]/F");
  picoAODEvents->Branch("notCanJet_phi", event->notCanJet_phi, "notCanJet_phi[nAllNotCanJets]/F");
  picoAODEvents->Branch("notCanJet_m",   event->notCanJet_m,   "notCanJet_m[nAllNotCanJets]/F");
  picoAODEvents->Branch("HHSB", &event->HHSB); picoAODEvents->Branch("HHCR", &event->HHCR); picoAODEvents->Branch("HHSR", &event->HHSR);
  picoAODEvents->Branch("ZHSB", &event->ZHSB); picoAODEvents->Branch("ZHCR", &event->ZHCR); picoAODEvents->Branch("ZHSR", &event->ZHSR);
  picoAODEvents->Branch("ZZSB", &event->ZZSB); picoAODEvents->Branch("ZZCR", &event->ZZCR); picoAODEvents->Branch("ZZSR", &event->ZZSR);
  picoAODEvents->Branch("SB", &event->SB); picoAODEvents->Branch("CR", &event->CR); picoAODEvents->Branch("SR", &event->SR);
  picoAODEvents->Branch("leadStM", &event->leadStM); picoAODEvents->Branch("sublStM", &event->sublStM);
  picoAODEvents->Branch("st", &event->st);
  picoAODEvents->Branch("stNotCan", &event->stNotCan);
  picoAODEvents->Branch("m4j", &event->m4j);
  picoAODEvents->Branch("nSelJets", &event->nSelJets);
  picoAODEvents->Branch("nPSTJets", &event->nPSTJets);
  picoAODEvents->Branch("passHLT", &event->passHLT);
  picoAODEvents->Branch("passDijetMass", &event->passDijetMass);
  picoAODEvents->Branch("passMDRs", &event->passMDRs);
  picoAODEvents->Branch("passXWt", &event->passXWt);
  picoAODEvents->Branch("xW", &event->xW);
  picoAODEvents->Branch("xt", &event->xt);
  picoAODEvents->Branch("xWt", &event->xWt);
  picoAODEvents->Branch("xbW", &event->xbW);
  picoAODEvents->Branch("xWbW", &event->xWbW);
  picoAODEvents->Branch("nIsoMuons", &event->nIsoMuons);
  picoAODEvents->Branch("ttbarWeight", &event->ttbarWeight);
  cout << "analysis::addDerivedQuantitiesToPicoAOD() done" << endl;
  return;
}

void analysis::storePicoAOD(){
  picoAODFile->Write();
  picoAODFile->Close();
  return;
}

void analysis::storeHemiSphereFile(){
  hMixToolCreate3Tag->storeLibrary();
  hMixToolCreate4Tag->storeLibrary();
  return;
}


void analysis::monitor(long int e){
  //Monitor progress
  timeTotal = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  timeElapsed          = timeTotal - previousMonitorTime;
  eventsElapsed        =         e - previousMonitorEvent;
  if( timeElapsed < 1 && e+1!=nEvents) return;
  previousMonitorEvent = e;
  previousMonitorTime  = timeTotal;
  percent              = (e+1)*100/nEvents;
  eventRate            = eventRate ? 0.9*eventRate + 0.1*eventsElapsed/timeElapsed : eventsElapsed/timeElapsed; // Running average with 0.9 momentum
  timeRemaining        = (nEvents-e)/eventRate;
  //eventRate      = (e+1)/timeTotal;
  //timeRemaining  = (nEvents-e)/eventRate;
  hours   = static_cast<int>( timeRemaining/3600 );
  minutes = static_cast<int>( timeRemaining/60   )%60;
  seconds = static_cast<int>( timeRemaining      )%60;
  getrusage(who, &usage);
  usageMB = usage.ru_maxrss/1024;
  //print status and flush stdout so that status bar only uses one line
  if(isMC){
    fprintf(stdout, "\rProcessed: %9li of %9li ( %2li%% | %5.0f events/s | done in %02i:%02i:%02i | memory usage: %li MB)       ", 
	                          e+1, nEvents, percent,     eventRate,        hours, minutes, seconds,          usageMB);
  }else{
    fprintf(stdout, "\rProcessed: %9li of %9li ( %2li%% | %5.0f events/s | done in %02i:%02i:%02i | memory usage: %li MB | LumiBlocks %5i | Lumi %5.2f/fb [%3.0f%%, %3.0f%%: L1, HLT] )       ", 
	                          e+1, nEvents, percent,     eventRate,        hours, minutes, seconds,          usageMB,             nls,     intLumi/1000 , 100*intLumi_passL1/intLumi, 100*intLumi_passHLT/intLumi);    
  }
  fflush(stdout);
  return;
}

int analysis::eventLoop(int maxEvents, long int firstEvent){

  //Set Number of events to process. Take manual maxEvents if maxEvents is > 0 and less than the total number of events in the input files. 
  nEvents = (maxEvents > 0 && maxEvents < treeEvents) ? maxEvents : treeEvents;
  long int lastEvent = firstEvent + nEvents;
  
  cout << "\nProcess " << nEvents << " of " << treeEvents << " events.\n";
  if(firstEvent){
    cout << " \t... starting with  " <<  firstEvent << " \n";
    previousMonitorEvent = firstEvent;
  }

  bool mixedEventWasData = false;

  start = std::clock();
  for(long int e = firstEvent; e < lastEvent; e++){
    
    currentEvent = e;

    alreadyFilled = false;
    //m4jPrevious = event->m4j;

    event->update(e);    

    if(writeOutEventNumbers){
      passed_runs  .push_back(event->run);
      passed_events.push_back(event->event);
    }

    if(( event->mixedEventIsData & !mixedEventWasData) ||
       (!event->mixedEventIsData &  mixedEventWasData) ){
      cout << "Switching between Data and MC. Now isData: " << event->mixedEventIsData << " event is: " << e <<  " / " << nEvents << endl;
      mixedEventWasData = event->mixedEventIsData;
    }

    if(skip4b && event->fourTag)  continue;
    if(skip3b && event->threeTag) continue;

    //
    //  Get the Data/MC Mixing 
    //
    bool isMCEvent = (isMC || (isDataMCMix && !event->mixedEventIsData));
    bool passData = isMCEvent ? (event->passHLT) : (passLumiMask() && event->passHLT);

    if(emulate4bFrom3b){
      if(!passData)                 continue;
      if(!event->threeTag)          continue;
      if(!event->pass4bEmulation(emulationOffset)) continue;
      
      //
      // Correct weight so we are not double counting psudotag weight
      //   (Already factored into whether or not the event pass4bEmulation
      //event->weight /= event->pseudoTagWeight;
      event->weight = 1.0;

      //
      // Treat canJets as Tag jets
      //
      event->setLooseAndPSJetsAsTagJets();
      
    }


    //
    //  Write Hemishpere files
    //
    bool passNJets = (event->selJets.size() >= 4);
    if(writeHSphereFile && passData && passNJets ){
      if(event->threeTag) hMixToolCreate3Tag->addEvent(event);
      if(event->fourTag)  hMixToolCreate4Tag->addEvent(event);
    }

    if(loadHSphereFile && passData && passNJets ){

      //
      //  TTbar Veto on mixed event
      //
      //if(event->t->rWbW < 2){
      //	//if(!event->passXWt){
      //	//cout << "Mixing and vetoing on Xwt" << endl;
      //	continue;
      //}

      if(event->threeTag) hMixToolLoad3Tag->makeArtificialEvent(event);
      if(event->fourTag)  hMixToolLoad4Tag->makeArtificialEvent(event);
    }


    if(debug) cout << "processing event " << endl;    
    processEvent();
    if(debug) cout << "Done processing event " << endl;    
    if(debug) event->dump();
    if(debug) cout << "done " << endl;    

    //periodically update status
    monitor(e);
    if(debug) cout << "done loop " << endl;    
  }

  //std::cout<<"cutflow->labelsDeflate()"<<std::endl;
  //cutflow->labelsDeflate();

  lumiCounts->FillLumiBlock(intLumi - lumiLastWrite);

  cout << endl;
  if(!isMC){
    cout << "Runs " << firstRun << "-" << lastRun << endl;
    // float missingLumi = std::accumulate(std::begin(lumiData), std::end(lumiData), 0.0,
    // 					[](const float previous, const std::pair<edm::LuminosityBlockID, float>& p)
    // 					{ return previous + p.second; });
    // cout << "Missing Lumi = " << missingLumi << "/pb" << endl;
    // float runLumi = 0;
    // edm::RunNumber_t thisRun = 0;
    // for(auto &lumiBlock: lumiData){
    //   thisRun = lumiBlock.first.run();
    //   if(thisRun<firstRun) continue;
    //   if(thisRun>lastRun) continue;
    //   if(lumiBlock.second != 0){
    // 	if(thisRun != prevRun){
    // 	  if(runLumi > 10) cout << prevRun << " " << runLumi << "/pb" << endl;
    // 	  runLumi = 0;
    // 	  prevRun = lumiBlock.first.run();
    // 	}
    // 	runLumi += lumiBlock.second;
    // 	//cout << lumiBlock.first << " " << lumiBlock.second << endl;
    //   }
    // }
    // if(runLumi > 10) cout << thisRun << " " << runLumi << "/pb" << endl;
  }

  eventRate = (nEvents)/timeTotal;

  hours   = static_cast<int>( timeTotal/3600 );
  minutes = static_cast<int>( timeTotal/60   )%60;
  seconds = static_cast<int>( timeTotal      )%60;
                                 
  if(isMC){
    fprintf(stdout,"---------------------------\nProcessed  %9li events in %02i:%02i:%02i (%5.0f events/s)",            nEvents, hours, minutes, seconds, eventRate);
  }else{
    fprintf(stdout,"---------------------------\nProcessed  %9li events in %02i:%02i:%02i (%5.0f events/s | %5.2f/fb)", nEvents, hours, minutes, seconds, eventRate, intLumi/1000);
  }
  return 0;
}

int analysis::processEvent(){
  if(debug) cout << "processEvent start" << endl;

  if(isMC){
    event->mcWeight = event->genWeight * (lumi * xs * kFactor / mcEventSumw);
    if(event->nTrueBJets>=4) event->mcWeight *= fourbkfactor;
    event->mcPseudoTagWeight = event->mcWeight * event->bTagSF * event->pseudoTagWeight * event->ttbarWeight * event->trigWeight;
    event->weight *= event->mcWeight;
    event->weightNoTrigger *= event->mcWeight;

    //
    //  Sub-Sample the 4b dataset 
    //
    if(makePSDataFromMC){
      if(!event->passPSDataFilter(false)){
	return 0;
      }
    }

    //
    //  Make the inverse dataset 
    //
    if(removePSDataFromMC){
      if(!event->passPSDataFilter(true)){
	return 0;
      }

      //
      // Scale up MC weight to correct for events removed 
      //
      //cout << " Weight was " << event->mcWeight << "\t";
      float correctionFactor = 1./(1.0- event->weight);
      event->genWeight = (event->genWeight* correctionFactor);
      event->mcWeight = event->genWeight * (lumi * xs * kFactor / mcEventSumw);
      //cout << " Weight now " << event->mcWeight << endl;
    }

    if(debug){
    std::cout << "Event: " << event->event << " Run: " << event->run << std::endl;
    std::cout << "event->genWeight * (lumi * xs * kFactor / mcEventSumw) = " << std::endl;;
      std::cout<< event->genWeight << " * (" << lumi << " * " << xs << " * " << kFactor << " / " << mcEventSumw << ") = " << event->mcWeight << std::endl;
      std::cout<< "\tweight  " << event->weight << std::endl;
      std::cout<< "\tbTagSF  " << event->bTagSF << std::endl;
      std::cout<< "\tfourbkfactor " << fourbkfactor << std::endl;
      std::cout<< "\tnTrueBJets " << event->nTrueBJets << std::endl;
      std::cout<< "\tmcWeight " << event->mcWeight << std::endl;
      std::cout<< "\tmcPseudoTagWeight " << event->mcPseudoTagWeight << std::endl;
      std::cout<< "\tmcWeight " << event->mcWeight << std::endl;
      std::cout<< "\tpseudoTagWeight " << event->pseudoTagWeight << std::endl;
      std::cout<< "\treweight " << event->reweight << std::endl;
      }

    for(const std::string& jcmName : event->jcmNames){
      if(debug) cout << "event->mcPseudoTagWeightMap[" << jcmName << "]" << endl;
      event->mcPseudoTagWeightMap[jcmName] = event->mcWeight * event->bTagSF * event->pseudoTagWeightMap[jcmName] * event->ttbarWeight * event->trigWeight;
    }

    //
    //  If using unit MC weights
    //
    if(mcUnitWeight){
      event->mcWeight = 1.0;
      event->mcPseudoTagWeight = event->pseudoTagWeight;

      for(const std::string& jcmName : event->jcmNames){
	event->mcPseudoTagWeightMap[jcmName] = event->pseudoTagWeightMap[jcmName];
      }

      event->weight = 1.0;
      event->weightNoTrigger = 1.0;
    }

  }else{ //!isMC

    event->mcPseudoTagWeight = event->pseudoTagWeight;

    // The "*= event->reweight4b" is used to pass the Mixed->Unmixed FvT to the h5 files 
    event->mcPseudoTagWeight *= event->reweight4b;

    for(const std::string& jcmName : event->jcmNames){
      event->mcPseudoTagWeightMap[jcmName] = event->pseudoTagWeightMap[jcmName];
      event->mcPseudoTagWeightMap[jcmName] *= event->reweight4b;
    }

  }

  if(debug) cout << "cutflow->Fill(event, all, true)" << endl;
  cutflow->Fill(event, "all", true);


  lumiCounts->Fill(event);

  if(isDataMCMix){
    if(event->mixedEventIsData){
      cutflow->Fill(event, "mixedEventIsData_3plus4Tag", true);
      cutflow->Fill(event, "mixedEventIsData");
    }else{
      cutflow->Fill(event, "mixedEventIsMC_3plus4Tag", true);
      cutflow->Fill(event, "mixedEventIsMC");
    }
  }


  


  //
  //if we are processing data, first apply lumiMask and trigger
  //
  bool isMCEvent = (isMC || (isDataMCMix && !event->mixedEventIsData));
  if(!isMCEvent){
    if(!passLumiMask()){
      if(debug) cout << "Fail lumiMask" << endl;
      return 0;
    }
    cutflow->Fill(event, "lumiMask", true);

    //keep track of total lumi
    countLumi();

    if( (intLumi - lumiLastWrite) > 500){
      lumiCounts->FillLumiBlock(intLumi - lumiLastWrite);
      lumiLastWrite = intLumi;
    }

    if(!event->passHLT){
      if(debug) cout << "Fail HLT: data" << endl;
      return 0;
    }
    cutflow->Fill(event, "HLT", true);
  }else{
    if(currentEvent > 0 && (currentEvent % 10000) == 0) 
      lumiCounts->FillLumiBlock(1.0);
  }

  if(allEvents != NULL && event->passHLT) allEvents->Fill(event);



  //
  // Loose Pre-selection for use in JEC uncertainties
  //
  bool jetMultiplicity = (event->selJets.size() >= 4);
  bool bTags = (event->threeTag || event->fourTag);
  if(looseSkim){
    bool passPreSel = jetMultiplicity && bTags;
    bool passLoosePreSel = (event->selJetsLoosePt.size() >= 4) && (event->tagJetsLoosePt.size() >= 4);
    if(passLoosePreSel && !passPreSel){
      picoAODFillEvents();
    }    
  }


  //
  // Preselection
  // 
  if(!jetMultiplicity){
    if(debug) cout << "Fail Jet Multiplicity" << endl;
    return 0;
  }
  cutflow->Fill(event, "jetMultiplicity", true);
  cutflow->btagSF_norm(event);

  if(!bTags){
    if(debug) cout << "Fail b-tag " << endl;
    return 0;
  }
  cutflow->Fill(event, "bTags");

  if(passPreSel != NULL && event->passHLT) passPreSel->Fill(event, event->views);


  // Fill picoAOD
  if(writePicoAOD && (writePicoAODBeforeDiJetMass || looseSkim)){//if we are making picoAODs for hemisphere mixing, we need to write them out before the dijetMass cut
    // WARNING: Applying MDRs early will change apparent dijetMass cut efficiency.
    event->applyMDRs(); // computes some of the derived quantities added to the picoAOD. 
    picoAODFillEvents();
    if(fastSkim) return 0;
  }


  // // Dijet mass preselection. Require at least one view has leadM(sublM) dijets with masses between 45(45) and 190(190) GeV.
  // if(!event->passDijetMass){
  //   if(debug) cout << "Fail dijet mass cut" << endl;
  //   return 0;
  // }
  // cutflow->Fill(event, "DijetMass");

  // if(passDijetMass != NULL && event->passHLT) passDijetMass->Fill(event, event->views);

  
  //
  // Event View Requirements: Mass Dependent Requirements (MDRs) on event views
  //
  if(!event->appliedMDRs) event->applyMDRs();

  // Fill picoAOD
  if(writePicoAOD){//for regular picoAODs, keep them small by filling after dijetMass cut
    picoAODFillEvents();
    if(fastSkim) return 0;
  }

  if(!event->passMDRs){
    if(debug) cout << "Fail MDRs" << endl;
    return 0;
  }
  cutflow->Fill(event, "MDRs");

  //
  //  Do Trigger Study
  //
  if(trigStudy)
    trigStudy->Fill(event);


  if(passMDRs != NULL && event->passHLT){
    passMDRs->Fill(event, event->views_passMDRs);

    lumiCounts->FillMDRs(event);
  }

  if(passTTCR != NULL && event->passTTCR && event->passHLT){
    passTTCR->Fill(event, event->views_passMDRs);
  }

  if(passSvB != NULL &&  (event->SvB_ps > 0.9) && event->passHLT){ 
    passSvB->Fill(event, event->views_passMDRs);
  }    



  //
  //  For VHH Study
  //
  if(passMjjOth != NULL){
    if(event->othJets.size() > 1){
      float mjjOther = (event->othJets.at(0)->p + event->othJets.at(1)->p).M();
    
      if( (mjjOther > 60)  && (mjjOther < 110)){

	if(event->passHLT) passMjjOth->Fill(event, event->views_passMDRs);

	if(trigStudyMjjOth)
	  trigStudyMjjOth->Fill(event);
	
      }

    }
  }

  //
  // ttbar CRs
  //
  if(failrWbW2 != NULL && event->passHLT){
    if(event->t->rWbW < 2){
      failrWbW2->Fill(event, event->views_passMDRs);
    }
  }

  if(passMuon != NULL && event->passHLT && event->muons_isoMed25.size()>0){
    passMuon->Fill(event, event->views_passMDRs);
  }

  if(passDvT05 != NULL && event->passHLT){
    if(event->DvT_raw > 0.5){
      passDvT05->Fill(event, event->views_passMDRs);
    }
  }

   
  return 0;
}

bool analysis::passLumiMask(){
  // if the lumiMask is empty, then no JSON file was provided so all
  // events should pass
  if(lumiMask.empty()) return true;


  //make lumiID run:lumiBlock
  edm::LuminosityBlockID lumiID(event->run, event->lumiBlock);

  //define function that checks if a lumiID is contained in a lumiBlockRange
  bool (*funcPtr) (edm::LuminosityBlockRange const &, edm::LuminosityBlockID const &) = &edm::contains;

  //Loop over the lumiMask and use funcPtr to check for a match
  std::vector< edm::LuminosityBlockRange >::const_iterator iter = std::find_if (lumiMask.begin(), lumiMask.end(), boost::bind(funcPtr, _1, lumiID) );

  return lumiMask.end() != iter; 
}

void analysis::getLumiData(std::string fileName){
  cout << "Getting integrated luminosity estimate per lumiBlock from: " << fileName << endl;
  brilCSV brilFile(fileName);
  lumiData = brilFile.GetData();
}

void analysis::countLumi(){
  edm::LuminosityBlockID lumiID(event->run, event->lumiBlock);
  if(lumiID != prevLumiID){
  
    if(debug) std::cout << lumiID << " " << lumiData[lumiID] << " " << intLumi << " \n";

    // this is a new lumi block, count it
    lumiID_intLumi = lumiData[lumiID]; // units are /pb
    lumiData[lumiID] = 0;
    intLumi += lumiID_intLumi; // keep track of integrated luminosity
    nls     += 1;              // count number of lumi sections

    // set previous lumi block to this one
    prevLumiID = lumiID;

    // keep track of first and last run observed
    if(event->run < firstRun) firstRun = event->run;
    if(event->run >  lastRun)  lastRun = event->run;

    lumiID_passL1  = false;
    lumiID_passHLT = false;
  }
  if(!lumiID_passL1  && event->passL1 ){
    intLumi_passL1  += lumiID_intLumi; // keep track of integrated luminosity that passes trigger (expect ~100% of lumiblocks that pass lumi mask to have some events passing the trigger)
    lumiID_passL1  = true; // prevent counting this lumi block more than once
  }
  if(!lumiID_passHLT && event->passHLT){
    if(lumiID_intLumi == 0){
      if(event->passHLT) cout << endl << "WARNING: " << lumiID << " not in bril file but event passes trigger" << endl;
    }
    intLumi_passHLT += lumiID_intLumi; // keep track of integrated luminosity that passes trigger (expect ~100% of lumiblocks that pass lumi mask to have some events passing the trigger)
    lumiID_passHLT = true; // prevent counting this lumi block more than once
  }

  return;
}

void analysis::loadJetCombinatoricModel(std::string jcmName){
  cout << " Will use preloaded JCM with name " << jcmName << endl;
  event->loadJetCombinatoricModel(jcmName);
  return;
}

void analysis::storeJetCombinatoricModel(std::string fileName){
  if(fileName=="") return;
  cout << "Using jetCombinatoricModel: " << fileName << endl;
  std::ifstream jetCombinatoricModel(fileName);
  std::string parameter;
  float value;
  while(jetCombinatoricModel >> parameter >> value){
    if(parameter.find("_err")    != std::string::npos) continue;
    if(parameter.find("_pererr") != std::string::npos) continue;
    if(parameter.find("pseudoTagProb_pass")         == 0){ event->pseudoTagProb         = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancement_pass")       == 0){ event->pairEnhancement       = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancementDecay_pass")  == 0){ event->pairEnhancementDecay  = value; cout << parameter << " " << value << endl; }
    if(parameter.find("threeTightTagFraction_pass") == 0){ event->threeTightTagFraction = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pseudoTagProb_lowSt_pass")        == 0){ event->pseudoTagProb_lowSt        = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pairEnhancement_lowSt_pass")      == 0){ event->pairEnhancement_lowSt      = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pairEnhancementDecay_lowSt_pass") == 0){ event->pairEnhancementDecay_lowSt = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pseudoTagProb_midSt_pass")        == 0){ event->pseudoTagProb_midSt        = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pairEnhancement_midSt_pass")      == 0){ event->pairEnhancement_midSt      = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pairEnhancementDecay_midSt_pass") == 0){ event->pairEnhancementDecay_midSt = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pseudoTagProb_highSt_pass")        == 0){ event->pseudoTagProb_highSt        = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pairEnhancement_highSt_pass")      == 0){ event->pairEnhancement_highSt      = value; cout << parameter << " " << value << endl; }
    // if(parameter.find("pairEnhancementDecay_highSt_pass") == 0){ event->pairEnhancementDecay_highSt = value; cout << parameter << " " << value << endl; }
    event->useJetCombinatoricModel = true;
  }
  return;
}


void analysis::storeJetCombinatoricModel(std::string jcmName, std::string fileName){
  if(fileName=="") return;
  cout << "Storing weights from jetCombinatoricModel: " << fileName << " into " << jcmName << endl;
  std::ifstream jetCombinatoricModel(fileName);
  std::string parameter;
  float value;
  event->jcmNames.push_back(jcmName);
  event->pseudoTagWeightMap.insert( std::pair<std::string, float>(jcmName, 1.0));
  event->mcPseudoTagWeightMap.insert( std::pair<std::string, float>(jcmName, 1.0));
  while(jetCombinatoricModel >> parameter >> value){
    if(parameter.find("_err") != std::string::npos) continue;
    if(parameter.find("pseudoTagProb_pass")               == 0){ event->pseudoTagProbMap               .insert( std::pair<std::string, float>(jcmName, value)); cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancement_pass")             == 0){ event->pairEnhancementMap             .insert( std::pair<std::string, float>(jcmName, value)); cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancementDecay_pass")        == 0){ event->pairEnhancementDecayMap        .insert( std::pair<std::string, float>(jcmName, value)); cout << parameter << " " << value << endl; }
    if(parameter.find("threeTightTagFraction_pass")       == 0){ event->threeTightTagFractionMap       .insert( std::pair<std::string, float>(jcmName, value)); cout << parameter << " " << value << endl; }
  }
  return;
}


void analysis::storeReweight(std::string fileName){
  if(fileName=="") return;
  cout << "Using reweight: " << fileName << endl;
  TFile* weightsFile = new TFile(fileName.c_str(), "READ");
  spline = (TSpline3*) weightsFile->Get("spline_FvTUnweighted");
  weightsFile->Close();
  return;
}


analysis::~analysis(){
  if(writeOutEventNumbers){
    cout << "Writing out event Numbers" << endl;
    histFile->WriteObject(&passed_events, "passed_events"); 
    histFile->WriteObject(&passed_runs,   "passed_runs"); 
  }
} 

