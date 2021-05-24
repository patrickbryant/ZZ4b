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



    lumiHists(std::string name, fwlite::TFileService& fs, bool loadLeptonTriggers=false, bool _debug=false);
    
    void Fill(eventData* event);
    void Fill(tTbarEventData* event);

    void FillLumiBlock(float lumiThisBlock);

    ~lumiHists(); 

  };

}
#endif // lumiHists_H
