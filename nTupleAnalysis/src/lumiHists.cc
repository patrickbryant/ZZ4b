//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/lumiHists.h"

using namespace nTupleAnalysis;

lumiHists::lumiHists(std::string name, fwlite::TFileService& fs, bool loadLeptonTriggers, bool _debug) {
  m_debug = _debug;
  m_dir = fs.mkdir(name);

  m_lumiPerLB = m_dir.make<TH1F>("lumiPerLB", (name+"/lumiPerLB; ;Entries").c_str(),  1,1,2);
  m_lumiPerLB->SetCanExtend(1);


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

  }else{

    // 2016
    h_HLT_4j45_3b087                               = new countsVsLumiHists("HLT_QuadJet45_TripleBTagCSV_p087"                , name, m_dir);
    h_HLT_2j90_2j30_3b087         		   = new countsVsLumiHists("HLT_2j90_2j30_3b087"			     , name, m_dir);
    h_L1_QuadJetC50				   = new countsVsLumiHists("L1_QuadJetC50"				     , name, m_dir);
    h_L1_DoubleJetC100 				   = new countsVsLumiHists("L1_DoubleJetC100" 				     , name, m_dir);
    h_L1_SingleJet170				   = new countsVsLumiHists("L1_SingleJet170"				     , name, m_dir);
    h_L1_HTT300					   = new countsVsLumiHists("L1_HTT300"					     , name, m_dir);
    h_L1_HTT280					   = new countsVsLumiHists("L1_HTT280"					     , name, m_dir);

    // 2017
    h_HLT_HT300_4j_75_60_45_40_3b 		   = new countsVsLumiHists("HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0" , name, m_dir);
    h_L1_HTT380er				   = new countsVsLumiHists("L1_HTT380er"				     , name, m_dir);

    h_L1_HTT250er_QuadJet_70_55_40_35_er2p5  = new countsVsLumiHists("L1_HTT250er_QuadJet_70_55_40_35_er2p5"   ,  name, m_dir);
    h_L1_HTT280er_QuadJet_70_55_40_35_er2p5  = new countsVsLumiHists("L1_HTT280er_QuadJet_70_55_40_35_er2p5"   ,  name, m_dir);
    h_L1_HTT300er_QuadJet_70_55_40_35_er2p5  = new countsVsLumiHists("L1_HTT300er_QuadJet_70_55_40_35_er2p5"   ,  name, m_dir);
    h_L1_HTT320er_QuadJet_70_55_40_40_er2p4  = new countsVsLumiHists("L1_HTT320er_QuadJet_70_55_40_40_er2p4"   ,  name, m_dir);
    h_L1_HTT320er_QuadJet_70_55_40_40_er2p5  = new countsVsLumiHists("L1_HTT320er_QuadJet_70_55_40_40_er2p5"   ,  name, m_dir);
    h_L1_HTT320er_QuadJet_70_55_45_45_er2p5  = new countsVsLumiHists("L1_HTT320er_QuadJet_70_55_45_45_er2p5"   ,  name, m_dir);
    h_L1_HTT340er_QuadJet_70_55_40_40_er2p5  = new countsVsLumiHists("L1_HTT340er_QuadJet_70_55_40_40_er2p5"   ,  name, m_dir);
    h_L1_HTT340er_QuadJet_70_55_45_45_er2p5  = new countsVsLumiHists("L1_HTT340er_QuadJet_70_55_45_45_er2p5"   ,  name, m_dir);
    h_L1_HTT300er                            = new countsVsLumiHists("L1_HTT300er"                             ,  name, m_dir);
    h_L1_HTT320er                            = new countsVsLumiHists("L1_HTT320er"                             ,  name, m_dir);
    h_L1_HTT340er                            = new countsVsLumiHists("L1_HTT340er"                             ,  name, m_dir);
    h_L1_QuadJet50er2p7                      = new countsVsLumiHists("L1_QuadJet50er2p7"                       ,  name, m_dir);
    h_L1_QuadJet60er2p7                      = new countsVsLumiHists("L1_QuadJet60er2p7"                       ,  name, m_dir);

    // 2018
    h_HLT_HT330_4j_75_60_45_40_3b 		   = new countsVsLumiHists("HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5" 		     , name, m_dir);
    h_L1_HTT360er				   = new countsVsLumiHists("L1_HTT360er"				     , name, m_dir);
    h_L1_HTT320er_QuadJet_70_55_40_40_er2p4	   = new countsVsLumiHists("L1_HTT320er_QuadJet_70_55_40_40_er2p4"	     , name, m_dir);

    h_HLT_2j116_dEta1p6_2b        		   = new countsVsLumiHists("HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71"        		     , name, m_dir);    
    h_L1_DoubleJet112er2p3_dEta_Max1p6		   = new countsVsLumiHists("L1_DoubleJet112er2p3_dEta_Max1p6"		     , name, m_dir);

    h_L1_HTT280er                                  = new countsVsLumiHists("L1_HTT280er"                                  , name, m_dir);
    //h_L1_HTT320er                                  = new countsVsLumiHists("L1_HTT320er"                                  , name, m_dir);
    h_L1_HTT280er_QuadJet_70_55_40_35_er2p4        = new countsVsLumiHists("L1_HTT280er_QuadJet_70_55_40_35_er2p4"        , name, m_dir);
    //h_L1_HTT320er_QuadJet_70_55_40_40_er2p4        = new countsVsLumiHists("L1_HTT320er_QuadJet_70_55_40_40_er2p4"        , name, m_dir);
    h_L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3  = new countsVsLumiHists("L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3"  , name, m_dir);
    h_L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3  = new countsVsLumiHists("L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3"  , name, m_dir);

    // Regions
    h_SB_3b                                        = new countsVsLumiHists("SB_3b"                                           , name, m_dir);
    h_CR_3b                                        = new countsVsLumiHists("CR_3b"                                           , name, m_dir);
    h_SR_3b                                        = new countsVsLumiHists("SR_3b"                                           , name, m_dir);
    h_SB_4b                                        = new countsVsLumiHists("SB_4b"                                           , name, m_dir);
    h_CR_4b                                        = new countsVsLumiHists("CR_4b"                                           , name, m_dir);
    h_SR_4b                                        = new countsVsLumiHists("SR_4b"                                           , name, m_dir);

    h_passHLT                                      = new countsVsLumiHists("passHLT"                                          , name, m_dir);
    h_passL1                                       = new countsVsLumiHists("passL1"                                           , name, m_dir);


    // nPVs
    h_nPV       = new countsVsLumiHists("nPV"                                           , name, m_dir);
    h_nPVGood   = new countsVsLumiHists("nPVGood"                                       , name, m_dir);


  }

} 

void lumiHists::Fill(eventData* event){
  if(m_debug) std::cout << "lumiHists::Fill " << std::endl;

  // 2016
  if(event->HLT_triggers["HLT_QuadJet45_TripleBTagCSV_p087"]                 )  h_HLT_4j45_3b087                                  ->Fill(event->weight);
  if(event->HLT_triggers["HLT_DoubleJet90_Double30_TripleBTagCSV_p087"]      )  h_HLT_2j90_2j30_3b087                             ->Fill(event->weight);
  if(event-> L1_triggers["L1_HTT300"]                                        )  h_L1_HTT300                                       ->Fill(event->weight);           
  if(event-> L1_triggers["L1_DoubleJetC100"]                                 )  h_L1_DoubleJetC100                                ->Fill(event->weight);           
  if(event-> L1_triggers["L1_SingleJet170"]                                  )  h_L1_SingleJet170                                 ->Fill(event->weight);   

  if(event-> L1_triggers_mon["L1_HTT280"]                                    )  h_L1_HTT280                                       ->Fill(event->weight);           

  // 2017
  if(event->HLT_triggers["HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0"]                      )  h_HLT_HT300_4j_75_60_45_40_3b                     ->Fill(event->weight);
  if(event-> L1_triggers["L1_HTT380er"]                                      )  h_L1_HTT380er                                     ->Fill(event->weight);


  if(event-> L1_triggers_mon["L1_HTT250er_QuadJet_70_55_40_35_er2p5"])     h_L1_HTT250er_QuadJet_70_55_40_35_er2p5  ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT280er_QuadJet_70_55_40_35_er2p5"])     h_L1_HTT280er_QuadJet_70_55_40_35_er2p5  ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT300er_QuadJet_70_55_40_35_er2p5"])     h_L1_HTT300er_QuadJet_70_55_40_35_er2p5  ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT320er_QuadJet_70_55_40_40_er2p4"])     h_L1_HTT320er_QuadJet_70_55_40_40_er2p4  ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT320er_QuadJet_70_55_40_40_er2p5"])     h_L1_HTT320er_QuadJet_70_55_40_40_er2p5  ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT320er_QuadJet_70_55_45_45_er2p5"])     h_L1_HTT320er_QuadJet_70_55_45_45_er2p5  ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT340er_QuadJet_70_55_40_40_er2p5"])     h_L1_HTT340er_QuadJet_70_55_40_40_er2p5  ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT340er_QuadJet_70_55_45_45_er2p5"])     h_L1_HTT340er_QuadJet_70_55_45_45_er2p5  ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT300er"                          ])     h_L1_HTT300er                            ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT320er"                          ])     h_L1_HTT320er                            ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_HTT340er"                          ])     h_L1_HTT340er                            ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_QuadJet50er2p7"                    ])     h_L1_QuadJet50er2p7                      ->Fill(event->weight);
  if(event-> L1_triggers_mon["L1_QuadJet60er2p7"                    ])     h_L1_QuadJet60er2p7                      ->Fill(event->weight);


  // 2018
  if(event->HLT_triggers["HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5"]                      )  h_HLT_HT330_4j_75_60_45_40_3b                     ->Fill(event->weight);
  if(event-> L1_triggers["L1_HTT360er"]                                      )  h_L1_HTT360er                                     ->Fill(event->weight);
  if(event->HLT_triggers["HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71"]                             )  h_HLT_2j116_dEta1p6_2b                            ->Fill(event->weight);
  if(event-> L1_triggers["L1_DoubleJet112er2p3_dEta_Max1p6"]                 )  h_L1_DoubleJet112er2p3_dEta_Max1p6                ->Fill(event->weight);           


  if(event-> L1_triggers_mon["L1_HTT280er"                                 ])   h_L1_HTT280er                                  ->Fill(event->weight);;
  //if(event-> L1_triggers_mon["L1_HTT320er"                                 ])   h_L1_HTT320er                                  ->Fill(event->weight);;
  if(event-> L1_triggers_mon["L1_HTT280er_QuadJet_70_55_40_35_er2p4"       ])   h_L1_HTT280er_QuadJet_70_55_40_35_er2p4        ->Fill(event->weight);;
  //if(event-> L1_triggers_mon["L1_HTT320er_QuadJet_70_55_40_40_er2p4"       ])   h_L1_HTT320er_QuadJet_70_55_40_40_er2p4        ->Fill(event->weight);;
  if(event-> L1_triggers_mon["L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3" ])   h_L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3  ->Fill(event->weight);;
  if(event-> L1_triggers_mon["L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3" ])   h_L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3  ->Fill(event->weight);;




  if(event->passHLT) h_passHLT               ->Fill(event->weight);
  if(event->passL1 ) h_passL1                ->Fill(event->weight);
    
  return;
}


void lumiHists::FillMDRs(eventData* event){
  if(m_debug) std::cout << "lumiHists::Fill " << std::endl;

  h_nPV                    ->Fill(event->weight*event->nPVs);
  h_nPVGood                ->Fill(event->weight*event->nPVsGood);


  if(event->threeTag){
    if(event->SB) h_SB_3b                    ->Fill(event->weight);
    if(event->CR) h_CR_3b                    ->Fill(event->weight);
    if(event->SR) h_SR_3b                    ->Fill(event->weight);
  }

  if(event->fourTag){
    if(event->SB) h_SB_4b                    ->Fill(event->weight);
    if(event->CR) h_CR_4b                    ->Fill(event->weight);
    if(event->SR) h_SR_4b                    ->Fill(event->weight);
  }


    
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

    m_lumiPerLB ->GetXaxis()->FindBin(h_HLT_Mu23_Ele12->m_currentLBStr.c_str());
    m_lumiPerLB ->Fill(h_HLT_Mu23_Ele12->m_currentLBStr.c_str(), lumiThisBlock);

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

  if(h_HLT_4j45_3b087){                                    

    m_lumiPerLB ->GetXaxis()->FindBin(h_HLT_4j45_3b087->m_currentLBStr.c_str());
    m_lumiPerLB ->Fill(h_HLT_4j45_3b087->m_currentLBStr.c_str(), lumiThisBlock);

    // 2016 
    h_HLT_4j45_3b087                                 ->FillLumiBlock(lumiThisBlock);
    h_HLT_2j90_2j30_3b087                            ->FillLumiBlock(lumiThisBlock);
    h_L1_QuadJetC50                                  ->FillLumiBlock(lumiThisBlock);
    h_L1_DoubleJetC100                               ->FillLumiBlock(lumiThisBlock);         
    h_L1_SingleJet170                                ->FillLumiBlock(lumiThisBlock);         
    h_L1_HTT300                                      ->FillLumiBlock(lumiThisBlock);         
    h_L1_HTT280                                      ->FillLumiBlock(lumiThisBlock);         

    // 2017
    h_HLT_HT300_4j_75_60_45_40_3b                    ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT380er                                    ->FillLumiBlock(lumiThisBlock);


    h_L1_HTT250er_QuadJet_70_55_40_35_er2p5  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT280er_QuadJet_70_55_40_35_er2p5  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT300er_QuadJet_70_55_40_35_er2p5  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT320er_QuadJet_70_55_40_40_er2p4  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT320er_QuadJet_70_55_40_40_er2p5  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT320er_QuadJet_70_55_45_45_er2p5  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT340er_QuadJet_70_55_40_40_er2p5  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT340er_QuadJet_70_55_45_45_er2p5  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT300er                            ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT320er                            ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT340er                            ->FillLumiBlock(lumiThisBlock);
    h_L1_QuadJet50er2p7                      ->FillLumiBlock(lumiThisBlock);
    h_L1_QuadJet60er2p7                      ->FillLumiBlock(lumiThisBlock);


    // 2018
    h_HLT_HT330_4j_75_60_45_40_3b                    ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT360er                                    ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT320er_QuadJet_70_55_40_40_er2p4          ->FillLumiBlock(lumiThisBlock);

    h_HLT_2j116_dEta1p6_2b                           ->FillLumiBlock(lumiThisBlock);
    h_L1_DoubleJet112er2p3_dEta_Max1p6               ->FillLumiBlock(lumiThisBlock);         


    h_L1_HTT280er                                  ->FillLumiBlock(lumiThisBlock);
    //h_L1_HTT320er                                  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT280er_QuadJet_70_55_40_35_er2p4        ->FillLumiBlock(lumiThisBlock);
    //h_L1_HTT320er_QuadJet_70_55_40_40_er2p4      ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3  ->FillLumiBlock(lumiThisBlock);



    // other

    h_passHLT               ->FillLumiBlock(lumiThisBlock);
    h_passL1                ->FillLumiBlock(lumiThisBlock);


    h_SB_3b                    ->FillLumiBlock(lumiThisBlock);
    h_CR_3b                    ->FillLumiBlock(lumiThisBlock);
    h_SR_3b                    ->FillLumiBlock(lumiThisBlock);

    h_SB_4b                    ->FillLumiBlock(lumiThisBlock);
    h_CR_4b                    ->FillLumiBlock(lumiThisBlock);
    h_SR_4b                    ->FillLumiBlock(lumiThisBlock);

    h_nPV                    ->FillLumiBlock(lumiThisBlock);
    h_nPVGood                ->FillLumiBlock(lumiThisBlock);


  }

  return;
}


lumiHists::~lumiHists(){} 

