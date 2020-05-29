#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>
#include <signal.h>

#include "ZZ4b/nTupleAnalysis/interface/analysis.h"

using std::cout;  using std::endl;

using namespace nTupleAnalysis;

analysis::analysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _isMC, bool _blind, std::string _year, int _histogramming, int _histDetailLevel, 
		   bool _doReweight, bool _debug, bool _fastSkim, bool _doTrigEmulation, bool _doTrigStudy, bool _mcUnitWeight, bool _isDataMCMix, bool _skip4b, bool _skip3b, bool _is3bMixed,
		   std::string bjetSF, std::string btagVariations,
		   std::string JECSyst, std::string friendFile,
		   bool _looseSkim){
  if(_debug) std::cout<<"In analysis constructor"<<std::endl;
  debug      = _debug;
  doReweight     = _doReweight;
  isMC       = _isMC;
  isDataMCMix = _isDataMCMix;
  skip4b = _skip4b;
  skip3b = _skip3b;
  is3bMixed = _is3bMixed;
  mcUnitWeight = _mcUnitWeight;
  blind      = _blind;
  year       = _year;
  events     = _events;
  looseSkim  = _looseSkim;
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
  histogramming = _histogramming;
  histDetailLevel = _histDetailLevel;
  fastSkim = _fastSkim;
  doTrigEmulation = _doTrigEmulation;
  doTrigStudy     = _doTrigStudy;
  

  //Calculate MC weight denominator
  if(isMC){
    if(debug) runs->Print();
    runs->SetBranchStatus("*", 0);
    runs->LoadTree(0);
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

  lumiBlocks = _lumiBlocks;
  event      = new eventData(events, isMC, year, debug, fastSkim, doTrigEmulation, isDataMCMix, doReweight, bjetSF, btagVariations, JECSyst, looseSkim, is3bMixed);
  treeEvents = events->GetEntries();
  cutflow    = new tagCutflowHists("cutflow", fs, isMC);
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
  cutflow->AddCut("xWt");
  cutflow->AddCut("MDCs");
  cutflow->AddCut("dEtaBB");
  cutflow->AddCut("all_ZHSR");
  cutflow->AddCut("lumiMask_ZHSR");
  cutflow->AddCut("HLT_ZHSR");
  cutflow->AddCut("jetMultiplicity_ZHSR");
  cutflow->AddCut("bTags_ZHSR");
  cutflow->AddCut("DijetMass_ZHSR");
  cutflow->AddCut("MDRs_ZHSR");
  cutflow->AddCut("xWt_ZHSR");
  cutflow->AddCut("MDCs_ZHSR");
  cutflow->AddCut("dEtaBB_ZHSR");
  
  if(histogramming >= 5) allEvents     = new eventHists("allEvents",     fs, false, isMC, blind, histDetailLevel, debug);
  if(histogramming >= 4) passPreSel    = new   tagHists("passPreSel",    fs, true,  isMC, blind, histDetailLevel, debug);
  if(histogramming >= 3) passDijetMass = new   tagHists("passDijetMass", fs, true,  isMC, blind, histDetailLevel, debug);
  if(histogramming >= 2) passMDRs      = new   tagHists("passMDRs",      fs, true,  isMC, blind, histDetailLevel, debug);
  if(histogramming >= 1) passXWt       = new   tagHists("passXWt",       fs, true,  isMC, blind, histDetailLevel, debug, event);
  //if(histogramming > 1        ) passMDCs     = new   tagHists("passMDCs",   fs,  true, isMC, blind, debug);
  //if(histogramming > 0        ) passDEtaBB   = new   tagHists("passDEtaBB", fs,  true, isMC, blind, debug);
  //if(histogramming > 0        ) passDEtaBBNoTrig   = new   tagHists("passDEtaBBNoTrig", fs, true, isMC, blind);
  //if(histogramming > 0        ) passDEtaBBNoTrigJetPts   = new   tagHists("passDEtaBBNoTrigJetPts", fs, true, isMC, blind);
  
  if(doTrigStudy)
    trigStudy     = new triggerStudy("trigStudy",     fs, debug);

} 

void analysis::createEventTextFile(std::string fileName){
  eventFile = new std::ofstream();
  cout << " Writing run and event numbers to " << fileName << endl;
  eventFile->open (fileName);
  (*eventFile) << "Run" << " " << "Event";
}



void analysis::createPicoAOD(std::string fileName, bool copyInputPicoAOD){
  writePicoAOD = true;
  picoAODFile = TFile::Open(fileName.c_str() , "RECREATE");
  if(copyInputPicoAOD){
    picoAODEvents     = events    ->CloneTree(0);
  }else{
    if(emulate4bFrom3b){
      picoAODEvents     = new TTree("Events", "Events Emulated 4b from 3b");
    }else{
      picoAODEvents     = new TTree("Events", "Events from Mixing");
    }
  }
  picoAODRuns       = runs      ->CloneTree();
  picoAODLumiBlocks = lumiBlocks->CloneTree();
  this->addDerivedQuantitiesToPicoAOD();
}



void analysis::createPicoAODBranches(){
  if(debug) cout << " analysis::createPicoAODBranches " << endl;

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

  if(isMC)
    m_mixed_truthParticle = new nTupleAnalysis::truthParticle("GenPart",picoAODEvents, false );
  
  outputBranch(picoAODEvents, "PV_npvs",         m_nPVs, "I");
  outputBranch(picoAODEvents, "PV_npvsGood",     m_nPVsGood, "I");

  //triggers
  //trigObjs = new trigData("TrigObj", tree);
  if(year=="2016"){
    outputBranch(picoAODEvents, "HLT_QuadJet45_TripleBTagCSV_p087",            m_HLT_4j45_3b087,     "O");
    outputBranch(picoAODEvents, "HLT_DoubleJet90_Double30_TripleBTagCSV_p087", m_HLT_2j90_2j30_3b087,"O");
    outputBranch(picoAODEvents, "L1_QuadJetC50",                               m_L1_QuadJetC50,"O");
    outputBranch(picoAODEvents, "L1_HTT300",                                   m_L1_HTT300,"O");
    outputBranch(picoAODEvents, "L1_TripleJet_88_72_56_VBF",                   m_L1_TripleJet_88_72_56_VBF,"O");
    outputBranch(picoAODEvents, "L1_DoubleJetC100",                            m_L1_DoubleJetC100,"O");
    outputBranch(picoAODEvents, "L1_SingleJet170",                             m_L1_SingleJet170,"O");            
  }
  
  if(year=="2017"){
    outputBranch(picoAODEvents, "HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0",      m_HLT_HT300_4j_75_60_45_40_3b ,  "O");
    outputBranch(picoAODEvents, "HLT_Mu12_DoublePFJets40MaxDeta1p6_DoubleCaloBTagCSV_p33",        m_HLT_mu12_2j40_dEta1p6_db,  "O");
    outputBranch(picoAODEvents, "HLT_PFJet500",                                                   m_HLT_j500,  "O");
    outputBranch(picoAODEvents, "HLT_AK8PFJet400_TrimMass30",                                     m_HLT_J400_m30,  "O");
    outputBranch(picoAODEvents, "L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6", m_L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6,  "O");
    outputBranch(picoAODEvents, "L1_HTT280er_QuadJet_70_55_40_35_er2p5",                          m_L1_HTT280er_QuadJet_70_55_40_35_er2p5,  "O");
    outputBranch(picoAODEvents, "L1_SingleJet170",                                                m_L1_SingleJet170,  "O");
    outputBranch(picoAODEvents, "L1_SingleJet180",                                                m_L1_SingleJet180,  "O");
    outputBranch(picoAODEvents, "L1_HTT300er",                                                    m_L1_HTT300er            ,  "O");                                          
  }

  if(year=="2018"){
    outputBranch(picoAODEvents, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", m_HLT_HT330_4j_75_60_45_40_3b,"O");
    outputBranch(picoAODEvents, "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",    m_HLT_4j_103_88_75_15_2b_VBF1,"O");
    outputBranch(picoAODEvents, "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",              m_HLT_4j_103_88_75_15_1b_VBF2,"O");
    outputBranch(picoAODEvents, "HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71",       m_HLT_2j116_dEta1p6_2b,"O");
    outputBranch(picoAODEvents, "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02",            m_HLT_J330_m30_2b,"O");
    outputBranch(picoAODEvents, "HLT_PFJet500",            m_HLT_j500,"O");
    outputBranch(picoAODEvents, "HLT_DiPFJetAve300_HFJEC", m_HLT_2j300ave,"O");
    outputBranch(picoAODEvents, "L1_HTT360er",                                                    m_L1_HTT360er				   ,"O");
    outputBranch(picoAODEvents, "L1_ETT2000",                                                     m_L1_ETT2000				   ,"O");
    outputBranch(picoAODEvents, "L1_HTT320er_QuadJet_70_55_40_40_er2p4",                          m_L1_HTT320er_QuadJet_70_55_40_40_er2p4	   ,"O");
    outputBranch(picoAODEvents, "L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5",                    m_L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5  ,"O");
    outputBranch(picoAODEvents, "L1_DoubleJet112er2p3_dEta_Max1p6",                               m_L1_DoubleJet112er2p3_dEta_Max1p6		   ,"O");
    outputBranch(picoAODEvents, "L1_DoubleJet150er2p5",                                           m_L1_DoubleJet150er2p5			   ,"O");
    outputBranch(picoAODEvents, "L1_SingleJet180",                                                m_L1_SingleJet180                              ,"O");

  }

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
  if(alreadyFilled) std::cout << "ERROR: Filling picoAOD with same event twice" << std::endl;
  alreadyFilled = true;
  if(m4jPrevious == event->m4j) std::cout << "WARNING: previous event had identical m4j = " << m4jPrevious << std::endl;

  assert( !(event->ZZSR && event->ZZSB) );
  assert( !(event->ZZSR && event->ZZCR) );
  assert( !(event->ZZSB && event->ZZCR) );

  assert( !(event->ZHSR && event->ZHSB) );
  assert( !(event->ZHSR && event->ZHCR) );
  assert( !(event->ZHSB && event->ZHCR) );

  assert( !(event->SR && event->SB) );
  assert( !(event->SR && event->CR) );
  assert( !(event->SB && event->CR) );

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

    if(isMC)
      m_mixed_truthParticle->writeTruth(event->truth->truthParticles->getParticles());

    m_nPVs = event->nPVs;
    m_nPVsGood = event->nPVsGood;    

    //2016
    if(year == "2016"){
      m_HLT_4j45_3b087               =   event->HLT_4j45_3b087	          ;
      m_HLT_2j90_2j30_3b087	     =   event->HLT_2j90_2j30_3b087	  ;    
      m_L1_QuadJetC50		     =   event->L1_QuadJetC50		  ;    
      m_L1_HTT300		     =   event->L1_HTT300		  ;    
      m_L1_TripleJet_88_72_56_VBF    =   event->L1_TripleJet_88_72_56_VBF ;  
      m_L1_DoubleJetC100	     =   event->L1_DoubleJetC100	  ;    
      m_L1_SingleJet170              =   event->L1_SingleJet170           ;  
    }

    //2017
    if(year == "2017"){
      m_HLT_HT300_4j_75_60_45_40_3b                                        = event->HLT_HT300_4j_75_60_45_40_3b                                      ;
      m_HLT_mu12_2j40_dEta1p6_db                                           = event->HLT_mu12_2j40_dEta1p6_db                                         ;
      m_HLT_j500                                                           = event->HLT_j500                                                         ;
      m_HLT_J400_m30                                                       = event->HLT_J400_m30                                                     ;
      m_L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6     = event->L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6   ;
      m_L1_HTT280er_QuadJet_70_55_40_35_er2p5                              = event->L1_HTT280er_QuadJet_70_55_40_35_er2p5                            ;
      m_L1_SingleJet170                                                    = event->L1_SingleJet170                                                  ;
      m_L1_SingleJet180                                                    = event->L1_SingleJet180                                                  ;
      m_L1_HTT300er                                                        = event->L1_HTT300er                                                      ;
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

      m_L1_HTT360er                                     =  event->L1_HTT360er                                   ;
      m_L1_ETT2000                                      =  event->L1_ETT2000                                    ;
      m_L1_HTT320er_QuadJet_70_55_40_40_er2p4           =  event->L1_HTT320er_QuadJet_70_55_40_40_er2p4         ;
      m_L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5     =  event->L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5   ;
      m_L1_DoubleJet112er2p3_dEta_Max1p6                =  event->L1_DoubleJet112er2p3_dEta_Max1p6              ;
      m_L1_DoubleJet150er2p5                            =  event->L1_DoubleJet150er2p5                          ;
      m_L1_SingleJet180                                 =  event->L1_SingleJet180                               ;
    }

    if(loadHSphereFile){
        hemisphereMixTool* thisHMixTool = nullptr;
        if(event->threeTag) thisHMixTool = hMixToolLoad3Tag;
        if(event->fourTag)  thisHMixTool = hMixToolLoad4Tag;
        assert(thisHMixTool);
    
        m_h1_run                = thisHMixTool->m_h1_run                ;
        m_h1_event              = thisHMixTool->m_h1_event              ;
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


    

  }

  picoAODEvents->Fill();  
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


void analysis::loadHemisphereLibrary(std::vector<std::string> hLibs_3tag, std::vector<std::string> hLibs_4tag, fwlite::TFileService& fs, int maxNHemis){

  //
  // Load Hemisphere Mixing 
  //
  hMixToolLoad3Tag = new hemisphereMixTool("3TagEvents", "dummyName", hLibs_3tag, false, fs, maxNHemis, debug, true, false, false, true);
  hMixToolLoad4Tag = new hemisphereMixTool("4TagEvents", "dummyName", hLibs_4tag, false, fs, maxNHemis, debug, true, false, false, true);
  loadHSphereFile = true;
}


void analysis::addDerivedQuantitiesToPicoAOD(){
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
  picoAODEvents->Branch("dRjjClose", &event->dRjjClose);
  picoAODEvents->Branch("dRjjOther", &event->dRjjOther);
  picoAODEvents->Branch("aveAbsEta", &event->aveAbsEta);
  picoAODEvents->Branch("aveAbsEtaOth", &event->aveAbsEtaOth);
  // picoAODEvents->Branch("nOthJets", &event->nOthJets);
  // picoAODEvents->Branch("othJet_pt",  event->othJet_pt,  "othJet_pt[nOthJets]/F");
  // picoAODEvents->Branch("othJet_eta", event->othJet_eta, "othJet_eta[nOthJets]/F");
  // picoAODEvents->Branch("othJet_phi", event->othJet_phi, "othJet_phi[nOthJets]/F");
  // picoAODEvents->Branch("othJet_m",   event->othJet_m,   "othJet_m[nOthJets]/F");
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
  picoAODEvents->Branch("passXWt", &event->passXWt);
  picoAODEvents->Branch("xW", &event->xW);
  picoAODEvents->Branch("xt", &event->xt);
  picoAODEvents->Branch("xWt", &event->xWt);
  picoAODEvents->Branch("xbW", &event->xbW);
  picoAODEvents->Branch("xWbW", &event->xWbW);
  //picoAODEvents->Branch("xWt0", &event->xWt0);
  //picoAODEvents->Branch("xWt1", &event->xWt1);
  //picoAODEvents->Branch("dRbW", &event->dRbW);
  picoAODEvents->Branch("nIsoMuons", &event->nIsoMuons);
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
    fprintf(stdout, "\rProcessed: %9li of %9li ( %2li%% | %5.0f events/s | done in %02i:%02i:%02i | memory usage: %li MB | LumiBlocks %5i | Est. Lumi %5.2f/fb )       ", 
	                          e+1, nEvents, percent,     eventRate,        hours, minutes, seconds,          usageMB,             nls,          intLumi/1000 );    
  }
  fflush(stdout);
}

int analysis::eventLoop(int maxEvents, long int firstEvent){

  //Set Number of events to process. Take manual maxEvents if maxEvents is > 0 and less than the total number of events in the input files. 
  nEvents = (maxEvents > 0 && maxEvents < treeEvents) ? maxEvents : treeEvents;
  
  cout << "\nProcess " << (nEvents - firstEvent) << " of " << treeEvents << " events.\n";
  if(firstEvent)
    cout << " \t... starting with  " <<  firstEvent << " \n";

  bool mixedEventWasData = false;

  start = std::clock();//2546000 //2546043
  for(long int e = firstEvent; e < nEvents; e++){

    alreadyFilled = false;
    m4jPrevious = event->m4j;

    event->update(e);    

    if(eventFile) (*eventFile) << event->run << " " << event->event << "\n";
      
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
      //
      event->weight /= event->pseudoTagWeight;

      //
      // Treat canJets as Tag jets
      //
      event->setPSJetsAsTagJets();
      
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
      //if(!event->passXWt){
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
    if( (e+1)%10000 == 0 || e+1==nEvents || debug) 
      monitor(e);
    if(debug) cout << "done loop " << endl;    
  }

  //std::cout<<"cutflow->labelsDeflate()"<<std::endl;
  //cutflow->labelsDeflate();

  cout << endl;
  if(!isMC) cout << "Runs " << firstRun << "-" << lastRun << endl;

  eventRate = (nEvents)/duration;

  hours   = static_cast<int>( duration/3600 );
  minutes = static_cast<int>( duration/60   )%60;
  seconds = static_cast<int>( duration      )%60;
                                 
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
    event->mcPseudoTagWeight = event->mcWeight * event->bTagSF * event->pseudoTagWeight;
    event->weight *= event->mcWeight;
    event->weightNoTrigger *= event->mcWeight;
    if(debug){
      std::cout << "event->weight * event->genWeight * (lumi * xs * kFactor / mcEventSumw) = ";
      std::cout<< event->weight <<" * "<< event->genWeight << " * (" << lumi << " * " << xs << " * " << kFactor << " / " << mcEventSumw << ") = " << event->weight << std::endl;
      std::cout<< "fourbkfactor " << fourbkfactor << std::endl;
    }

    for(const std::string& jcmName : event->jcmNames){
      event->mcPseudoTagWeightMap[jcmName] = event->mcWeight * event->bTagSF * event->pseudoTagWeightMap[jcmName];
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

  }else{
    event->mcPseudoTagWeight = event->pseudoTagWeight;

    for(const std::string& jcmName : event->jcmNames){
      event->mcPseudoTagWeightMap[jcmName] = event->pseudoTagWeightMap[jcmName];
    }

  }
  cutflow->Fill(event, "all", true);

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
  //  Do Trigger Study
  //
  if(doTrigStudy)
    trigStudy->Fill(event);
  


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

    if(!event->passHLT){
      if(debug) cout << "Fail HLT: data" << endl;
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
    if(debug) cout << "Fail Jet Multiplicity" << endl;
    //event->dump();
    return 0;
  }
  cutflow->Fill(event, "jetMultiplicity", true);

  bool bTags = (event->threeTag || event->fourTag);
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


  // Dijet mass preselection. Require at least one view has leadM(sublM) dijets with masses between 50(50) and 180(160) GeV.
  if(!event->passDijetMass){
    if(debug) cout << "Fail dijet mass cut" << endl;
    return 0;
  }
  cutflow->Fill(event, "DijetMass");

  if(passDijetMass != NULL && event->passHLT) passDijetMass->Fill(event, event->views);

  
  //
  // Event View Requirements: Mass Dependent Requirements (MDRs) on event views
  //
  if(!event->appliedMDRs) event->applyMDRs();

  // Fill picoAOD
  if(writePicoAOD && !writePicoAODBeforeDiJetMass && !looseSkim){//for regular picoAODs, keep them small by filling after dijetMass cut
    picoAODFillEvents();
    if(fastSkim) return 0;
  }

  if(!event->passMDRs){
    if(debug) cout << "Fail MDRs" << endl;
    return 0;
  }
  cutflow->Fill(event, "MDRs");

  if(passMDRs != NULL && event->passHLT) passMDRs->Fill(event, event->views);


  //
  // ttbar veto
  //
  if(fastSkim) return 0; // in fast skim mode, we do not construct top quark candidates. Return early.
  if(!event->passXWt){
    if(debug) cout << "Fail xWt" << endl;
    return 0;
  }
  cutflow->Fill(event, "xWt");

  if(passXWt != NULL && event->passHLT) passXWt->Fill(event, event->views);

  //
  // Don't need anything below here in cutflow for now.
  //
  return 0;



  //
  // Event View Cuts: Mass Dependent Cuts (MDCs) on event view variables
  //
  if(!event->views[0]->passMDCs){
    if(debug) cout << "Fail MDCs" << endl;
    return 0;
  }
  cutflow->Fill(event, "MDCs");

  if(passMDCs != NULL && event->passHLT) passMDCs->Fill(event, event->views);





  if(!event->views[0]->passDEtaBB){
    if(debug) cout << "Fail dEtaBB" << endl;
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
  cout << "Getting integrated luminosity estimate per lumiBlock from: " << fileName << endl;
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
    if(debug){
      std::cout << lumiID << " " << lumiData[lumiID] << " " << intLumi << " \n";
    }
    nls   += 1;
    nruns += 1;
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
    if(parameter.find("pseudoTagProb_pass")        == 0){ event->pseudoTagProb        = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancement_pass")      == 0){ event->pairEnhancement      = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancementDecay_pass") == 0){ event->pairEnhancementDecay = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pseudoTagProb_lowSt_pass")        == 0){ event->pseudoTagProb_lowSt        = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancement_lowSt_pass")      == 0){ event->pairEnhancement_lowSt      = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancementDecay_lowSt_pass") == 0){ event->pairEnhancementDecay_lowSt = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pseudoTagProb_midSt_pass")        == 0){ event->pseudoTagProb_midSt        = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancement_midSt_pass")      == 0){ event->pairEnhancement_midSt      = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancementDecay_midSt_pass") == 0){ event->pairEnhancementDecay_midSt = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pseudoTagProb_highSt_pass")        == 0){ event->pseudoTagProb_highSt        = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancement_highSt_pass")      == 0){ event->pairEnhancement_highSt      = value; cout << parameter << " " << value << endl; }
    if(parameter.find("pairEnhancementDecay_highSt_pass") == 0){ event->pairEnhancementDecay_highSt = value; cout << parameter << " " << value << endl; }
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
  if(eventFile) eventFile->close();
} 

