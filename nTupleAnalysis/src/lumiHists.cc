//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/lumiHists.h"

using namespace nTupleAnalysis;

lumiHists::lumiHists(std::string name, fwlite::TFileService& fs, bool loadLeptonTriggers, bool _debug) {
  m_debug = _debug;
  m_dir = fs.mkdir(name);

  if(loadLeptonTriggers){
    h_HLT_Mu23_Ele12 = new countsVsLumiHists("HLT_Mu23_Ele12",name,  m_dir);
    h_HLT_Mu12_Ele23 = new countsVsLumiHists("HLT_Mu12_Ele23",name,  m_dir);
    h_HLT_IsoMu24    = new countsVsLumiHists("HLT_IsoMu24",   name,  m_dir);
    h_HLT_IsoMu27    = new countsVsLumiHists("HLT_IsoMu27",   name,  m_dir);


    h_L1_Mu20_EG10        = new countsVsLumiHists("L1_Mu20_EG10"           , name, m_dir);
    h_L1_SingleMu22       = new countsVsLumiHists("L1_SingleMu22"          , name, m_dir);
    h_L1_SingleMu25       = new countsVsLumiHists("L1_SingleMu25"          , name, m_dir);
    h_L1_Mu5_EG23         = new countsVsLumiHists("L1_Mu5_EG23"            , name, m_dir);
    h_L1_Mu5_LooseIsoEG20 = new countsVsLumiHists("L1_Mu5_LooseIsoEG20"    , name, m_dir);
    h_L1_Mu7_EG23         = new countsVsLumiHists("L1_Mu7_EG23"            , name, m_dir);
    h_L1_Mu7_LooseIsoEG20 = new countsVsLumiHists("L1_Mu7_LooseIsoEG20"    , name, m_dir);
    h_L1_Mu7_LooseIsoEG23 = new countsVsLumiHists("L1_Mu7_LooseIsoEG23"    , name, m_dir);
    h_L1_Mu23_EG10        = new countsVsLumiHists("L1_Mu23_EG10"           , name, m_dir);
    h_L1_Mu20_EG17        = new countsVsLumiHists("L1_Mu20_EG17"           , name, m_dir);
    h_L1_SingleMu22er2p1  = new countsVsLumiHists("L1_SingleMu22er2p1"     , name, m_dir);
    h_L1_Mu20_EG10er2p5   = new countsVsLumiHists("L1_Mu20_EG10er2p5"      , name, m_dir);
  }

} 

void lumiHists::Fill(eventData* event){
  if(m_debug) std::cout << "lumiHists::Fill " << std::endl;

    
  return;
}

void lumiHists::Fill(tTbarEventData* event){
  if(m_debug) std::cout << "lumiHists::Fill " << std::endl;

  if(event->HLT_Mu23_Ele12)  h_HLT_Mu23_Ele12->Fill(event->weight);
  if(event->HLT_Mu12_Ele23)  h_HLT_Mu12_Ele23->Fill(event->weight);
  if(event->HLT_IsoMu24)     h_HLT_IsoMu24   ->Fill(event->weight);
  if(event->HLT_IsoMu27)     h_HLT_IsoMu27   ->Fill(event->weight);


  if(event->L1_Mu20_EG10       )     h_L1_Mu20_EG10        ->Fill(event->weight);
  if(event->L1_SingleMu22      )     h_L1_SingleMu22       ->Fill(event->weight);
  if(event->L1_SingleMu25      )     h_L1_SingleMu25       ->Fill(event->weight);
  if(event->L1_Mu5_EG23        )     h_L1_Mu5_EG23         ->Fill(event->weight);
  if(event->L1_Mu5_LooseIsoEG20)     h_L1_Mu5_LooseIsoEG20 ->Fill(event->weight);
  if(event->L1_Mu7_EG23        )     h_L1_Mu7_EG23         ->Fill(event->weight);
  if(event->L1_Mu7_LooseIsoEG20)     h_L1_Mu7_LooseIsoEG20 ->Fill(event->weight);
  if(event->L1_Mu7_LooseIsoEG23)     h_L1_Mu7_LooseIsoEG23 ->Fill(event->weight);
  if(event->L1_Mu23_EG10       )     h_L1_Mu23_EG10        ->Fill(event->weight);
  if(event->L1_Mu20_EG17       )     h_L1_Mu20_EG17        ->Fill(event->weight);
  if(event->L1_SingleMu22er2p1 )     h_L1_SingleMu22er2p1  ->Fill(event->weight);
  if(event->L1_Mu20_EG10er2p5  )     h_L1_Mu20_EG10er2p5   ->Fill(event->weight);


    
  return;
}


void lumiHists::FillLumiBlock(float lumiThisBlock){
  if(m_debug) std::cout << "lumiHists::FillLumiBlock " << std::endl;

  
  if(h_HLT_Mu23_Ele12){

    h_HLT_Mu23_Ele12->FillLumiBlock(lumiThisBlock);
    h_HLT_Mu12_Ele23->FillLumiBlock(lumiThisBlock);
    h_HLT_IsoMu24   ->FillLumiBlock(lumiThisBlock);
    h_HLT_IsoMu27   ->FillLumiBlock(lumiThisBlock);

    h_L1_Mu20_EG10        ->FillLumiBlock(lumiThisBlock);
    h_L1_SingleMu22       ->FillLumiBlock(lumiThisBlock);
    h_L1_SingleMu25       ->FillLumiBlock(lumiThisBlock);
    h_L1_Mu5_EG23         ->FillLumiBlock(lumiThisBlock);
    h_L1_Mu5_LooseIsoEG20 ->FillLumiBlock(lumiThisBlock);
    h_L1_Mu7_EG23         ->FillLumiBlock(lumiThisBlock);
    h_L1_Mu7_LooseIsoEG20 ->FillLumiBlock(lumiThisBlock);
    h_L1_Mu7_LooseIsoEG23 ->FillLumiBlock(lumiThisBlock);
    h_L1_Mu23_EG10        ->FillLumiBlock(lumiThisBlock);
    h_L1_Mu20_EG17        ->FillLumiBlock(lumiThisBlock);
    h_L1_SingleMu22er2p1  ->FillLumiBlock(lumiThisBlock);
    h_L1_Mu20_EG10er2p5   ->FillLumiBlock(lumiThisBlock);
  }

  return;
}


lumiHists::~lumiHists(){} 

