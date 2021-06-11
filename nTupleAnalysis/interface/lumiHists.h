// -*- C++ -*-
#if !defined(lumiHists_H)
#define lumiHists_H

#include <iostream>
#include <TH1F.h>
#include "ZZ4b/nTupleAnalysis/interface/countsVsLumiHists.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/tTbarEventData.h"

namespace nTupleAnalysis {

  class lumiHists {
  public:

    bool m_debug = false;
    TFileDirectory m_dir;
  
    TH1F* m_lumiPerLB = NULL;

    // ttbar triggers
    countsVsLumiHists*  h_HLT_Mu23_Ele12 = NULL;
    countsVsLumiHists*  h_HLT_Mu12_Ele23 = NULL;
    countsVsLumiHists*  h_HLT_IsoMu24    = NULL;
    countsVsLumiHists*  h_HLT_IsoMu27    = NULL;

    countsVsLumiHists*  h_L1_Mu20_EG10        = NULL;
    countsVsLumiHists*  h_L1_SingleMu22       = NULL;
    countsVsLumiHists*  h_L1_SingleMu25       = NULL;
    countsVsLumiHists*  h_L1_Mu5_EG23         = NULL;
    countsVsLumiHists*  h_L1_Mu5_LooseIsoEG20 = NULL;
    countsVsLumiHists*  h_L1_Mu7_EG23         = NULL;
    countsVsLumiHists*  h_L1_Mu7_LooseIsoEG20 = NULL;
    countsVsLumiHists*  h_L1_Mu7_LooseIsoEG23 = NULL;
    countsVsLumiHists*  h_L1_Mu23_EG10        = NULL;
    countsVsLumiHists*  h_L1_Mu20_EG17        = NULL;
    countsVsLumiHists*  h_L1_SingleMu22er2p1  = NULL;
    countsVsLumiHists*  h_L1_Mu20_EG10er2p5   = NULL;

    // b-jet triggers

    // 2016
    countsVsLumiHists* h_HLT_4j45_3b087                                     = NULL;
    countsVsLumiHists* h_HLT_2j90_2j30_3b087                                = NULL;
    countsVsLumiHists* h_L1_QuadJetC50                                      = NULL;
    countsVsLumiHists* h_L1_DoubleJetC100                                   = NULL;
    countsVsLumiHists* h_L1_SingleJet170                                    = NULL;
    countsVsLumiHists* h_L1_HTT300                                          = NULL;
    countsVsLumiHists* h_L1_HTT280                                          = NULL;

    // 2017 
    countsVsLumiHists* h_HLT_HT300_4j_75_60_45_40_3b                        = NULL;
    countsVsLumiHists* h_L1_HTT380er                                        = NULL;

    countsVsLumiHists* h_L1_HTT250er_QuadJet_70_55_40_35_er2p5  = NULL;
    countsVsLumiHists* h_L1_HTT280er_QuadJet_70_55_40_35_er2p5  = NULL;
    countsVsLumiHists* h_L1_HTT300er_QuadJet_70_55_40_35_er2p5  = NULL;
    countsVsLumiHists* h_L1_HTT320er_QuadJet_70_55_40_40_er2p4  = NULL;
    countsVsLumiHists* h_L1_HTT320er_QuadJet_70_55_40_40_er2p5  = NULL;
    countsVsLumiHists* h_L1_HTT320er_QuadJet_70_55_45_45_er2p5  = NULL;
    countsVsLumiHists* h_L1_HTT340er_QuadJet_70_55_40_40_er2p5  = NULL;
    countsVsLumiHists* h_L1_HTT340er_QuadJet_70_55_45_45_er2p5  = NULL;
    countsVsLumiHists* h_L1_HTT300er                            = NULL;
    countsVsLumiHists* h_L1_HTT320er                            = NULL;
    countsVsLumiHists* h_L1_HTT340er                            = NULL;
    countsVsLumiHists* h_L1_QuadJet50er2p7                      = NULL;
    countsVsLumiHists* h_L1_QuadJet60er2p7                      = NULL;


    // 2018
    countsVsLumiHists* h_HLT_HT330_4j_75_60_45_40_3b                        = NULL;
    countsVsLumiHists* h_HLT_2j116_dEta1p6_2b                               = NULL;
    countsVsLumiHists* h_L1_HTT360er                                        = NULL;
    countsVsLumiHists* h_L1_DoubleJet112er2p3_dEta_Max1p6                   = NULL;

    countsVsLumiHists* h_L1_HTT280er                                  = NULL;
    //countsVsLumiHists* h_L1_HTT320er                                  = NULL;
    countsVsLumiHists* h_L1_HTT280er_QuadJet_70_55_40_35_er2p4        = NULL;
    //countsVsLumiHists* h_L1_HTT320er_QuadJet_70_55_40_40_er2p4        = NULL;
    countsVsLumiHists* h_L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3  = NULL;
    countsVsLumiHists* h_L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3  = NULL;


    /// 
    countsVsLumiHists*     h_passHLT = NULL;
    countsVsLumiHists*     h_passL1  = NULL;

    countsVsLumiHists*     h_SB_3b = NULL;
    countsVsLumiHists*     h_CR_3b = NULL;
    countsVsLumiHists*     h_SR_3b = NULL;
    countsVsLumiHists*     h_SB_4b = NULL;
    countsVsLumiHists*     h_CR_4b = NULL;
    countsVsLumiHists*     h_SR_4b = NULL;

    countsVsLumiHists*     h_nPV       = NULL;
    countsVsLumiHists*     h_nPVGood   = NULL;



    lumiHists(std::string name, fwlite::TFileService& fs, std::string year, bool loadLeptonTriggers=false, bool _debug=false);
    
    void Fill(eventData* event);
    void FillMDRs(eventData* event);

    void Fill(tTbarEventData* event);

    void FillLumiBlock(float lumiThisBlock);

    ~lumiHists(); 

  };

}
#endif // lumiHists_H
