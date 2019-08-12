#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>
#include <signal.h>

#include "ZZ4b/nTupleAnalysis/interface/analysis.h"


using namespace nTupleAnalysis;

analysis::analysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _isMC, bool _blind, std::string _year, int _histogramming, bool _debug){
  if(_debug) std::cout<<"In analysis constructor"<<std::endl;
  debug      = _debug;
  isMC       = _isMC;
  blind      = _blind;
  year       = _year;
  events     = _events;
  events->SetBranchStatus("*", 0);
  runs       = _runs;
  histogramming = _histogramming;
  

  //Calculate MC weight denominator
  if(isMC){
    runs->SetBranchStatus("*", 0);
    runs->LoadTree(0);
    inputBranch(runs, "genEventCount", genEventCount);
    inputBranch(runs, "genEventSumw",  genEventSumw);
    inputBranch(runs, "genEventSumw2", genEventSumw2);
    for(int r = 0; r < runs->GetEntries(); r++){
      runs->GetEntry(r);
      mcEventCount += genEventCount;
      mcEventSumw  += genEventSumw;
      mcEventSumw2 += genEventSumw2;
    }
    std::cout << "mcEventCount " << mcEventCount << " | mcEventSumw " << mcEventSumw << std::endl;
  }

  lumiBlocks = _lumiBlocks;
  event      = new eventData(events, isMC, year, debug);
  treeEvents = events->GetEntries();
  cutflow    = new tagCutflowHists("cutflow", fs, isMC);

  

  // hists
  if(histogramming >= 4) allEvents     = new eventHists("allEvents",     fs, false, isMC, blind, debug);
  if(histogramming >= 3) passPreSel    = new   tagHists("passPreSel",    fs, true,  isMC, blind, debug);
  if(histogramming >= 2) passDijetMass = new   tagHists("passDijetMass", fs, true,  isMC, blind, debug);
  if(histogramming >= 1) passMDRs      = new   tagHists("passMDRs",      fs, true,  isMC, blind, debug);
  //if(histogramming > 1        ) passMDCs     = new   tagHists("passMDCs",   fs,  true, isMC, blind, debug);
  //if(histogramming > 0        ) passDEtaBB   = new   tagHists("passDEtaBB", fs,  true, isMC, blind, debug);
  //if(histogramming > 0        ) passDEtaBBNoTrig   = new   tagHists("passDEtaBBNoTrig", fs, true, isMC, blind);
  //if(histogramming > 0        ) passDEtaBBNoTrigJetPts   = new   tagHists("passDEtaBBNoTrigJetPts", fs, true, isMC, blind);

} 

void analysis::createPicoAOD(std::string fileName, bool copyInputPicoAOD){
  writePicoAOD = true;
  picoAODFile = TFile::Open(fileName.c_str() , "RECREATE");
  if(copyInputPicoAOD){
    picoAODEvents     = events    ->CloneTree(0);
  }else{
    picoAODEvents     = new TTree("Events", "Events from Mixing");
  }
  picoAODRuns       = runs      ->CloneTree();
  picoAODLumiBlocks = lumiBlocks->CloneTree();
}



void analysis::createPicoAODBranches(){

  //
  //  Initial Event Data
  //
  outputBranch(picoAODEvents, "run",               m_run, "i");
  outputBranch(picoAODEvents, "luminosityBlock",   m_lumiBlock,  "i");
  outputBranch(picoAODEvents, "event",             m_event,  "l");

  
  m_mixed_jetData  = new nTupleAnalysis::jetData("Jet",picoAODEvents, false, "");
  m_mixed_muonData = new nTupleAnalysis::muonData("Muon",picoAODEvents, false );
  
  outputBranch(picoAODEvents, "PV_npvs",         m_nPVs, "I");
  outputBranch(picoAODEvents, "PV_npvsGood",     m_nPVsGood, "I");

  //triggers
  //trigObjs = new trigData("TrigObj", tree);
  if(year=="2016"){
    outputBranch(picoAODEvents, "HLT_QuadJet45_TripleBTagCSV_p087",            m_HLT_4j45_3b087, "O");
    outputBranch(picoAODEvents, "HLT_DoubleJet90_Double30_TripleBTagCSV_p087", m_HLT_2j90_2j30_3b087,"O");
  }
  if(year=="2018"){
    outputBranch(picoAODEvents, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", m_HLT_HT330_4j_75_60_45_40_3b,"O");
    outputBranch(picoAODEvents, "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",    m_HLT_4j_103_88_75_15_2b_VBF1,"O");
    outputBranch(picoAODEvents, "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",              m_HLT_4j_103_88_75_15_1b_VBF2,"O");
    outputBranch(picoAODEvents, "HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71",       m_HLT_2j116_dEta1p6_2b,"O");
    outputBranch(picoAODEvents, "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02",            m_HLT_J330_m30_2b,"O");
    outputBranch(picoAODEvents, "HLT_PFJet500",            m_HLT_j500,"O");
    outputBranch(picoAODEvents, "HLT_DiPFJetAve300_HFJEC", m_HLT_2j300ave,"O");
  }

  //
  //  Hemisphere Mixed branches
  //
  if(loadHSphereFile){

    //
    //  Hemisphere Event Data
    //
    outputBranch(picoAODEvents,     "h1_run"               ,   m_h1_run               ,         "i");
    outputBranch(picoAODEvents,     "h1_event"             ,   m_h1_event             ,         "l");
    outputBranch(picoAODEvents,     "h1_NJet"              ,   m_h1_NJet              ,         "i");     
    outputBranch(picoAODEvents,     "h1_NBJet"             ,   m_h1_NBJet             ,         "i");     
    outputBranch(picoAODEvents,     "h1_NNonSelJet"        ,   m_h1_NNonSelJet        ,         "i");     
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
    outputBranch(picoAODEvents,     "h2_NJet"              ,   m_h2_NJet              ,         "i");     
    outputBranch(picoAODEvents,     "h2_NBJet"             ,   m_h2_NBJet             ,         "i");     
    outputBranch(picoAODEvents,     "h2_NNonSelJet"        ,   m_h2_NNonSelJet        ,         "i");     
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
  if(loadHSphereFile){
    //std::cout << "Loading " << std::endl;
    //std::cout << event->run <<  " " << event->event << std::endl;
    //std::cout << "Jets: " << std::endl;
    //for(const jetPtr& j: event->allJets){
    //  std::cout << "\t " << j->pt << " / " << j->eta << " / " << j->phi << std::endl;
    //}

    m_run       = event->run;
    m_lumiBlock = event->lumiBlock;
    m_event     = event->event;

    //
    //  Undo the bjet reg corr if applied
    //
    for(const jetPtr &jet: event->allJets){
      if(jet->appliedBRegression) jet->scaleFourVector(1./jet->bRegCorr);
    }
    m_mixed_jetData ->writeJets(event->allJets);
    m_mixed_muonData->writeMuons(event->allMuons);

    m_nPVs = event->nPVs;
    m_nPVsGood = event->nPVsGood;    

    //2016
    if(year == "2016"){
      m_HLT_4j45_3b087       = event->HLT_4j45_3b087;
      m_HLT_2j90_2j30_3b087  = event->HLT_2j90_2j30_3b087;
    }
    //2018
    if(year == "2018"){
      m_HLT_HT330_4j_75_60_45_40_3b  = event->HLT_HT330_4j_75_60_45_40_3b;
      m_HLT_4j_103_88_75_15_2b_VBF1  = event->HLT_4j_103_88_75_15_2b_VBF1;
      m_HLT_4j_103_88_75_15_1b_VBF2  = event->HLT_4j_103_88_75_15_1b_VBF2;
      m_HLT_2j116_dEta1p6_2b         = event->HLT_2j116_dEta1p6_2b;
      m_HLT_J330_m30_2b              = event->HLT_J330_m30_2b;;
      m_HLT_j500                     = event->HLT_j500;
      m_HLT_2j300ave                 = event->HLT_2j300ave;
    }

    hemisphereMixTool* thisHMixTool = nullptr;
    if(event->threeTag) thisHMixTool = hMixToolLoad3Tag;
    if(event->fourTag)  thisHMixTool = hMixToolLoad4Tag;
    assert(thisHMixTool);

    m_h1_run                = thisHMixTool->m_h1_run                ;
    m_h1_event              = thisHMixTool->m_h1_event              ;
    m_h1_NJet               = thisHMixTool->m_h1_NJet               ;
    m_h1_NBJet              = thisHMixTool->m_h1_NBJet              ;
    m_h1_NNonSelJet         = thisHMixTool->m_h1_NNonSelJet         ;
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
    m_h2_NJet               = thisHMixTool->m_h2_NJet               ;
    m_h2_NBJet              = thisHMixTool->m_h2_NBJet              ;
    m_h2_NNonSelJet         = thisHMixTool->m_h2_NNonSelJet         ;
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

  picoAODEvents->Fill();  
}

void analysis::createHemisphereLibrary(std::string fileName, fwlite::TFileService& fs){

  //
  // Hemisphere Mixing 
  //
  hMixToolCreate3Tag = new hemisphereMixTool("3TagEvents", fileName, std::vector<std::string>(), true, fs, debug, true, false);
  hMixToolCreate4Tag = new hemisphereMixTool("4TagEvents", fileName, std::vector<std::string>(), true, fs, debug, true, false);
  writeHSphereFile = true;
}


void analysis::loadHemisphereLibrary(std::vector<std::string> hLibs_3tag, std::vector<std::string> hLibs_4tag, fwlite::TFileService& fs){

  //
  // Load Hemisphere Mixing 
  //
  hMixToolLoad3Tag = new hemisphereMixTool("3TagEvents", "dummyName", hLibs_3tag, false, fs, debug, true, false);
  hMixToolLoad4Tag = new hemisphereMixTool("4TagEvents", "dummyName", hLibs_4tag, false, fs, debug, true, false);
  loadHSphereFile = true;
}


void analysis::addDerivedQuantitiesToPicoAOD(){
  picoAODEvents->Branch("pseudoTagWeight", &event->pseudoTagWeight);
  picoAODEvents->Branch("FvTWeight", &event->FvTWeight);
  picoAODEvents->Branch("weight", &event->weight);
  picoAODEvents->Branch("threeTag", &event->threeTag);
  picoAODEvents->Branch("fourTag", &event->fourTag);
  picoAODEvents->Branch("nPVsGood", &event->nPVsGood);
  picoAODEvents->Branch("canJet0_pt" , &event->canJet0_pt ); picoAODEvents->Branch("canJet1_pt" , &event->canJet1_pt ); picoAODEvents->Branch("canJet2_pt" , &event->canJet2_pt ); picoAODEvents->Branch("canJet3_pt" , &event->canJet3_pt );
  picoAODEvents->Branch("canJet0_eta", &event->canJet0_eta); picoAODEvents->Branch("canJet1_eta", &event->canJet1_eta); picoAODEvents->Branch("canJet2_eta", &event->canJet2_eta); picoAODEvents->Branch("canJet3_eta", &event->canJet3_eta);
  picoAODEvents->Branch("canJet0_phi", &event->canJet0_phi); picoAODEvents->Branch("canJet1_phi", &event->canJet1_phi); picoAODEvents->Branch("canJet2_phi", &event->canJet2_phi); picoAODEvents->Branch("canJet3_phi", &event->canJet3_phi);
  picoAODEvents->Branch("canJet0_e"  , &event->canJet0_e  ); picoAODEvents->Branch("canJet1_e"  , &event->canJet1_e  ); picoAODEvents->Branch("canJet2_e"  , &event->canJet2_e  ); picoAODEvents->Branch("canJet3_e"  , &event->canJet3_e  );
  picoAODEvents->Branch("dRjjClose", &event->dRjjClose);
  picoAODEvents->Branch("dRjjOther", &event->dRjjOther);
  picoAODEvents->Branch("aveAbsEta", &event->aveAbsEta);
  picoAODEvents->Branch("aveAbsEtaOth", &event->aveAbsEtaOth);
  picoAODEvents->Branch("nOthJets", &event->nOthJets);
  picoAODEvents->Branch("othJet_pt",  event->othJet_pt,  "othJet_pt[nOthJets]/F");
  picoAODEvents->Branch("othJet_eta", event->othJet_eta, "othJet_eta[nOthJets]/F");
  picoAODEvents->Branch("othJet_phi", event->othJet_phi, "othJet_phi[nOthJets]/F");
  picoAODEvents->Branch("othJet_m",   event->othJet_m,   "othJet_m[nOthJets]/F");
  picoAODEvents->Branch("nAllNotCanJets", &event->nAllNotCanJets);
  picoAODEvents->Branch("notCanJet_pt",  event->notCanJet_pt,  "notCanJet_pt[nAllNotCanJets]/F");
  picoAODEvents->Branch("notCanJet_eta", event->notCanJet_eta, "notCanJet_eta[nAllNotCanJets]/F");
  picoAODEvents->Branch("notCanJet_phi", event->notCanJet_phi, "notCanJet_phi[nAllNotCanJets]/F");
  picoAODEvents->Branch("notCanJet_m",   event->notCanJet_m,   "notCanJet_m[nAllNotCanJets]/F");
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
  picoAODEvents->Branch("passDEtaBB", &event->passDEtaBB);
  picoAODEvents->Branch("xWt0", &event->xWt0);
  picoAODEvents->Branch("xWt1", &event->xWt1);
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
  percent        = (e+1)*100/nEvents;
  duration       = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  eventRate      = (e+1)/duration;
  timeRemaining  = (nEvents-e)/eventRate;
  minutes = static_cast<int>(timeRemaining/60);
  seconds = static_cast<int>(timeRemaining - minutes*60);
  getrusage(who, &usage);
  usageMB = usage.ru_maxrss/1024;
  //print status and flush stdout so that status bar only uses one line
  if(isMC){
    fprintf(stdout, "\rProcessed: %8li of %li ( %2li%% | %.0f events/s | done in %02i:%02i | memory usage: %li MB)       ", 
	                          e+1, nEvents, percent,   eventRate,    minutes, seconds,                usageMB);
  }else{
    fprintf(stdout, "\rProcessed: %8li of %li ( %2li%% | %.0f events/s | done in %02i:%02i | memory usage: %li MB | LumiBlocks %i | Est. Lumi %.2f/fb )       ", 
 	                          e+1, nEvents, percent,   eventRate,    minutes, seconds,                usageMB,            nls,         intLumi/1000 );    
  }
  fflush(stdout);
}

int analysis::eventLoop(int maxEvents, long int firstEvent){

  //Set Number of events to process. Take manual maxEvents if maxEvents is > 0 and less than the total number of events in the input files. 
  nEvents = (maxEvents > 0 && maxEvents < treeEvents) ? maxEvents : treeEvents;
  
  std::cout << "\nProcess " << (nEvents - firstEvent) << " of " << treeEvents << " events.\n";
  if(firstEvent)
    std::cout << " \t... starting with  " <<  firstEvent << " \n";

  start = std::clock();//2546000 //2546043
  for(long int e = firstEvent; e < nEvents; e++){

    event->update(e);    
        
    //
    //  Write Hemishpere files
    //
    bool passData = isMC ? (event->passHLT) : (passLumiMask() && event->passHLT);
    bool passNJets = (event->selJets.size() >= 4);
    if(writeHSphereFile && passData && passNJets ){
      if(event->threeTag) hMixToolCreate3Tag->addEvent(event);
      if(event->fourTag)  hMixToolCreate4Tag->addEvent(event);
    }

    if(loadHSphereFile && passData && passNJets ){
      if(event->threeTag) hMixToolLoad3Tag->makeArtificialEvent(event);
      if(event->fourTag)  hMixToolLoad4Tag->makeArtificialEvent(event);
    }

    processEvent();
    if(debug) event->dump();


    //periodically update status
    if( (e+1)%10000 == 0 || e+1==nEvents || debug) 
      monitor(e);

  }



  std::cout << std::endl;
  if(!isMC) std::cout << "Runs " << firstRun << "-" << lastRun << std::endl;

  minutes = static_cast<int>(duration/60);
  seconds = static_cast<int>(duration - minutes*60);
                                        
  if(isMC){
    fprintf(stdout,"---------------------------\nProcessed %li events in %02i:%02i", nEvents, minutes, seconds);
  }else{
    fprintf(stdout,"---------------------------\nProcessed %li events (%.2f/fb) in %02i:%02i", nEvents, intLumi/1000, minutes, seconds);
  }
  return 0;
}

int analysis::processEvent(){
  if(debug) std::cout << "processEvent start" << std::endl;
  if(isMC){
    event->weight = event->genWeight * (lumi * xs * kFactor / mcEventSumw);
    if(debug) std::cout << "event->genWeight * (lumi * xs * kFactor / mcEventSumw) = " << event->genWeight << " * (" << lumi << " * " << xs << " * " << kFactor << " / " << mcEventSumw << ") = " << event->weight << std::endl;
  }
  cutflow->Fill(event, "all", true);

  //
  //if we are processing data, first apply lumiMask and trigger
  //
  if(!isMC){
    if(!passLumiMask()){
      if(debug) std::cout << "Fail lumiMask" << std::endl;
      return 0;
    }
    cutflow->Fill(event, "lumiMask", true);

    //keep track of total lumi
    countLumi();

    if(!event->passHLT){
      if(debug) std::cout << "Fail HLT: data" << std::endl;
      return 0;
    }
    cutflow->Fill(event, "HLT", true);
  }
  if(allEvents != NULL && event->passHLT) allEvents->Fill(event);

  //
  // Preselection
  // 
  bool jetMultiplicity = (event->selJets.size() >= 4);
  //bool jetMultiplicity = (event->selJets.size() == 4);
  if(!jetMultiplicity){
    if(debug) std::cout << "Fail Jet Multiplicity" << std::endl;
    //event->dump();
    return 0;
  }
  cutflow->Fill(event, "jetMultiplicity", true);

  bool bTags = (event->threeTag || event->fourTag);
  if(!bTags){
    if(debug) std::cout << "Fail b-tag " << std::endl;
    return 0;
  }
  cutflow->Fill(event, "bTags");

  //Background model reweighting
  //if(spline != NULL && event->threeTag) applyReweight();
  if(doReweight && event->threeTag) applyReweight();

  if(passPreSel != NULL && event->passHLT) passPreSel->Fill(event, event->views);
  

  // Fill picoAOD
  event->applyMDRs();
  if(writePicoAOD) picoAODFillEvents();


  if(!event->passDijetMass){
    if(debug) std::cout << "Fail dijet mass cut" << std::endl;
    return 0;
  }
  cutflow->Fill(event, "DijetMass");

  if(passDijetMass != NULL && event->passHLT) passDijetMass->Fill(event, event->views);


  //
  // Event View Requirements: Mass Dependent Requirements (MDRs) on event views
  //
  if(!event->passMDRs){
    if(debug) std::cout << "Fail MDRs" << std::endl;
    return 0;
  }
  cutflow->Fill(event, "MDRs");

  if(passMDRs != NULL && event->passHLT) passMDRs->Fill(event, event->views);


  //
  // Event View Cuts: Mass Dependent Cuts (MDCs) on event view variables
  //
  if(!event->views[0]->passMDCs){
    if(debug) std::cout << "Fail MDCs" << std::endl;
    return 0;
  }
  cutflow->Fill(event, "MDCs");

  if(passMDCs != NULL && event->passHLT) passMDCs->Fill(event, event->views);





  if(!event->views[0]->passDEtaBB){
    if(debug) std::cout << "Fail dEtaBB" << std::endl;
    return 0;
  }
  cutflow->Fill(event, "dEtaBB");
  
  if(passDEtaBB != NULL && event->passHLT) passDEtaBB->Fill(event, event->views);
  //if(passDEtaBBNoTrig != NULL )            passDEtaBBNoTrig->Fill(event, event->views);
  //if(passDEtaBBNoTrigJetPts != NULL ){
  //  if (event->canJets[0]->pt > 75  && event->canJets[1]->pt > 60 && event->canJets[2]->pt > 45 && event->canJets[3]->pt > 40   ){
  //    passDEtaBBNoTrigJetPts->Fill(event, event->views);
  //  }
  //}
    

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
  std::cout << "Getting integrated luminosity estimate per lumiBlock from: " << fileName << std::endl;
  brilCSV brilFile(fileName);
  lumiData = brilFile.GetData();
}

void analysis::countLumi(){
  if(event->lumiBlock != prevLumiBlock || event->run != prevRun){
    if(event->run != prevRun){
      if(event->run < firstRun) firstRun = event->run;
      if(event->run >  lastRun)  lastRun = event->run;
    }
    prevLumiBlock = event->lumiBlock;
    prevRun       = event->run;
    edm::LuminosityBlockID lumiID(event->run, event->lumiBlock);
    intLumi += lumiData[lumiID];//convert units to /fb
    //std::cout << lumiID << " " << lumiData[lumiID] << " " << intLumi << " \n";
    nls   += 1;
    nruns += 1;
  }
  return;
}

void analysis::storeJetCombinatoricModel(std::string fileName){
  if(fileName=="") return;
  std::cout << "Using jetCombinatoricModel: " << fileName << std::endl;
  std::ifstream jetCombinatoricModel(fileName);
  std::string parameter;
  float value;
  while(jetCombinatoricModel >> parameter >> value){
    if(parameter.find("_err") != std::string::npos) continue;
    if(parameter.find("pseudoTagProb_pass")        == 0){ event->pseudoTagProb        = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pairEnhancement_pass")      == 0){ event->pairEnhancement      = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pairEnhancementDecay_pass") == 0){ event->pairEnhancementDecay = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pseudoTagProb_lowSt_pass")        == 0){ event->pseudoTagProb_lowSt        = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pairEnhancement_lowSt_pass")      == 0){ event->pairEnhancement_lowSt      = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pairEnhancementDecay_lowSt_pass") == 0){ event->pairEnhancementDecay_lowSt = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pseudoTagProb_midSt_pass")        == 0){ event->pseudoTagProb_midSt        = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pairEnhancement_midSt_pass")      == 0){ event->pairEnhancement_midSt      = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pairEnhancementDecay_midSt_pass") == 0){ event->pairEnhancementDecay_midSt = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pseudoTagProb_highSt_pass")        == 0){ event->pseudoTagProb_highSt        = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pairEnhancement_highSt_pass")      == 0){ event->pairEnhancement_highSt      = value; std::cout << parameter << " " << value << std::endl; }
    if(parameter.find("pairEnhancementDecay_highSt_pass") == 0){ event->pairEnhancementDecay_highSt = value; std::cout << parameter << " " << value << std::endl; }
    event->useJetCombinatoricModel = true;
  }
  return;
}

void analysis::storeReweight(std::string fileName){
  if(fileName=="") return;
  std::cout << "Using reweight: " << fileName << std::endl;
  TFile* weightsFile = new TFile(fileName.c_str(), "READ");
  spline = (TSpline3*) weightsFile->Get("spline_FvTUnweighted");
  weightsFile->Close();
  return;
}

void analysis::applyReweight(){
  if(debug) std::cout << "applyReweight: event->FvT = " << event->FvT << std::endl;
  //event->FvTWeight = spline->Eval(event->FvT);
  event->FvTWeight = event->FvT / (1-event->FvT);
  event->weight  *= event->FvTWeight;
  if(debug) std::cout << "applyReweight: event->FvTWeight = " << event->FvTWeight << std::endl;
  return;
}

analysis::~analysis(){} 

