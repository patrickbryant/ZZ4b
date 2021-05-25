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
  }else{

    h_HLT_4j45_3b087                               = new countsVsLumiHists("HLT_4j45_3b087"              		     , name, m_dir);
    h_HLT_2j90_2j30_3b087         		   = new countsVsLumiHists("HLT_2j90_2j30_3b087"			     , name, m_dir);
    h_HLT_HT300_4j_75_60_45_40_3b 		   = new countsVsLumiHists("HLT_HT300_4j_75_60_45_40_3b" 		     , name, m_dir);
    h_HLT_2j100_dEta1p6_2b        		   = new countsVsLumiHists("HLT_2j100_dEta1p6_2b"        		     , name, m_dir);
    h_HLT_HT330_4j_75_60_45_40_3b 		   = new countsVsLumiHists("HLT_HT330_4j_75_60_45_40_3b" 		     , name, m_dir);
    h_HLT_2j116_dEta1p6_2b        		   = new countsVsLumiHists("HLT_2j116_dEta1p6_2b"        		     , name, m_dir);
    h_HLT_j500                     		   = new countsVsLumiHists("HLT_j500"                     		     , name, m_dir);
    h_HLT_2j300ave                		   = new countsVsLumiHists("HLT_2j300ave"                		     , name, m_dir);
    h_L1_DoubleJetC100 				   = new countsVsLumiHists("L1_DoubleJetC100" 				     , name, m_dir);
    h_L1_TripleJet_88_72_56_VBF			   = new countsVsLumiHists("L1_TripleJet_88_72_56_VBF"			     , name, m_dir);
    h_L1_QuadJetC50				   = new countsVsLumiHists("L1_QuadJetC50"				     , name, m_dir);
    h_L1_HTT300					   = new countsVsLumiHists("L1_HTT300"					     , name, m_dir);
    h_L1_HTT360er				   = new countsVsLumiHists("L1_HTT360er"				     , name, m_dir);
    h_L1_HTT380er				   = new countsVsLumiHists("L1_HTT380er"				     , name, m_dir);
    h_L1_ETT2000 				   = new countsVsLumiHists("L1_ETT2000" 				     , name, m_dir);
    h_L1_HTT320er_QuadJet_70_55_40_40_er2p4	   = new countsVsLumiHists("L1_HTT320er_QuadJet_70_55_40_40_er2p4"	     , name, m_dir);
    h_L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5  = new countsVsLumiHists("L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5"     , name, m_dir);
    h_L1_DoubleJet112er2p3_dEta_Max1p6		   = new countsVsLumiHists("L1_DoubleJet112er2p3_dEta_Max1p6"		     , name, m_dir);
    h_L1_DoubleJet100er2p3_dEta_Max1p6		   = new countsVsLumiHists("L1_DoubleJet100er2p3_dEta_Max1p6"		     , name, m_dir);
    h_L1_DoubleJet150er2p5			   = new countsVsLumiHists("L1_DoubleJet150er2p5"			     , name, m_dir);
    h_L1_SingleJet200				   = new countsVsLumiHists("L1_SingleJet200"				     , name, m_dir);
    h_L1_SingleJet180				   = new countsVsLumiHists("L1_SingleJet180"				     , name, m_dir);
    h_L1_SingleJet170				   = new countsVsLumiHists("L1_SingleJet170"				     , name, m_dir);
    h_L1_HTT280					   = new countsVsLumiHists("L1_HTT280"					     , name, m_dir);
    h_L1_HTT300er				   = new countsVsLumiHists("L1_HTT300er"				     , name, m_dir);
    h_L1_HTT280er_QuadJet_70_55_40_35_er2p5        = new countsVsLumiHists("L1_HTT280er_QuadJet_70_55_40_35_er2p5"           , name, m_dir);                



  }

} 

void lumiHists::Fill(eventData* event){
  if(m_debug) std::cout << "lumiHists::Fill " << std::endl;

  if(event->HLT_4j45_3b087                                   )  h_HLT_4j45_3b087                                  ->Fill(event->weight);
  if(event->HLT_2j90_2j30_3b087                              )  h_HLT_2j90_2j30_3b087                             ->Fill(event->weight);
  if(event->HLT_HT300_4j_75_60_45_40_3b                      )  h_HLT_HT300_4j_75_60_45_40_3b                     ->Fill(event->weight);
  if(event->HLT_2j100_dEta1p6_2b                             )  h_HLT_2j100_dEta1p6_2b                            ->Fill(event->weight);
  if(event->HLT_HT330_4j_75_60_45_40_3b                      )  h_HLT_HT330_4j_75_60_45_40_3b                     ->Fill(event->weight);
  if(event->HLT_2j116_dEta1p6_2b                             )  h_HLT_2j116_dEta1p6_2b                            ->Fill(event->weight);
  if(event->HLT_j500                                         )  h_HLT_j500                                        ->Fill(event->weight);
  if(event->HLT_2j300ave                                     )  h_HLT_2j300ave                                    ->Fill(event->weight);
  if(event->L1_DoubleJetC100                                 )  h_L1_DoubleJetC100                                ->Fill(event->weight);           
  if(event->L1_TripleJet_88_72_56_VBF                        )  h_L1_TripleJet_88_72_56_VBF                       ->Fill(event->weight);           
  if(event->L1_QuadJetC50                                    )  h_L1_QuadJetC50                                   ->Fill(event->weight);
  if(event->L1_HTT300                                        )  h_L1_HTT300                                       ->Fill(event->weight);           
  if(event->L1_HTT360er                                      )  h_L1_HTT360er                                     ->Fill(event->weight);
  if(event->L1_HTT380er                                      )  h_L1_HTT380er                                     ->Fill(event->weight);
  if(event->L1_ETT2000                                       )  h_L1_ETT2000                                      ->Fill(event->weight);
  if(event->L1_HTT320er_QuadJet_70_55_40_40_er2p4            )  h_L1_HTT320er_QuadJet_70_55_40_40_er2p4           ->Fill(event->weight);
  if(event->L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5      )  h_L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5     ->Fill(event->weight);
  if(event->L1_DoubleJet112er2p3_dEta_Max1p6                 )  h_L1_DoubleJet112er2p3_dEta_Max1p6                ->Fill(event->weight);           
  if(event->L1_DoubleJet100er2p3_dEta_Max1p6                 )  h_L1_DoubleJet100er2p3_dEta_Max1p6                ->Fill(event->weight);           
  if(event->L1_DoubleJet150er2p5                             )  h_L1_DoubleJet150er2p5                            ->Fill(event->weight);
  if(event->L1_SingleJet200                                  )  h_L1_SingleJet200                                 ->Fill(event->weight);   
  if(event->L1_SingleJet180                                  )  h_L1_SingleJet180                                 ->Fill(event->weight);   
  if(event->L1_SingleJet170                                  )  h_L1_SingleJet170                                 ->Fill(event->weight);   
  if(event->L1_HTT280                                        )  h_L1_HTT280                                       ->Fill(event->weight);           
  if(event->L1_HTT300er                                      )  h_L1_HTT300er                                     ->Fill(event->weight);
  if(event->L1_HTT280er_QuadJet_70_55_40_35_er2p5            )  h_L1_HTT280er_QuadJet_70_55_40_35_er2p5           ->Fill(event->weight);

    
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

  if(h_HLT_4j45_3b087){                                    
    h_HLT_4j45_3b087                                 ->FillLumiBlock(lumiThisBlock);
    h_HLT_2j90_2j30_3b087                            ->FillLumiBlock(lumiThisBlock);
    h_HLT_HT300_4j_75_60_45_40_3b                    ->FillLumiBlock(lumiThisBlock);
    h_HLT_2j100_dEta1p6_2b                           ->FillLumiBlock(lumiThisBlock);
    h_HLT_HT330_4j_75_60_45_40_3b                    ->FillLumiBlock(lumiThisBlock);
    h_HLT_2j116_dEta1p6_2b                           ->FillLumiBlock(lumiThisBlock);
    h_HLT_j500                                       ->FillLumiBlock(lumiThisBlock);
    h_HLT_2j300ave                                   ->FillLumiBlock(lumiThisBlock);
    h_L1_DoubleJetC100                               ->FillLumiBlock(lumiThisBlock);         
    h_L1_TripleJet_88_72_56_VBF                      ->FillLumiBlock(lumiThisBlock);         
    h_L1_QuadJetC50                                  ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT300                                      ->FillLumiBlock(lumiThisBlock);         
    h_L1_HTT360er                                    ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT380er                                    ->FillLumiBlock(lumiThisBlock);
    h_L1_ETT2000                                     ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT320er_QuadJet_70_55_40_40_er2p4          ->FillLumiBlock(lumiThisBlock);
    h_L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5    ->FillLumiBlock(lumiThisBlock);
    h_L1_DoubleJet112er2p3_dEta_Max1p6               ->FillLumiBlock(lumiThisBlock);         
    h_L1_DoubleJet100er2p3_dEta_Max1p6               ->FillLumiBlock(lumiThisBlock);         
    h_L1_DoubleJet150er2p5                           ->FillLumiBlock(lumiThisBlock);
    h_L1_SingleJet200                                ->FillLumiBlock(lumiThisBlock);         
    h_L1_SingleJet180                                ->FillLumiBlock(lumiThisBlock);         
    h_L1_SingleJet170                                ->FillLumiBlock(lumiThisBlock);         
    h_L1_HTT280                                      ->FillLumiBlock(lumiThisBlock);         
    h_L1_HTT300er                                    ->FillLumiBlock(lumiThisBlock);
    h_L1_HTT280er_QuadJet_70_55_40_35_er2p5          ->FillLumiBlock(lumiThisBlock);
  }

  return;
}


lumiHists::~lumiHists(){} 

