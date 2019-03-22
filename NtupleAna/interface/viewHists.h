// -*- C++ -*-
#if !defined(viewHists_H)
#define viewHists_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"
#include "ZZ4b/NtupleAna/interface/vecHists.h"
#include "ZZ4b/NtupleAna/interface/jetHists.h"
#include "ZZ4b/NtupleAna/interface/muonHists.h"
#include "ZZ4b/NtupleAna/interface/dijetHists.h"

namespace NtupleAna {

  class viewHists {
  public:
    TFileDirectory dir;
    
    // Object Level
    TH1F*     nAllJets;
    TH1F*     nSelJets;
    TH1F*     nTagJets;
    TH1F*     nCanJets;
    jetHists*  allJets;
    jetHists*  selJets;
    jetHists*  tagJets;
    jetHists*  canJets;    

    TH1F* nAllMuons;
    TH1F* nIsoMuons;
    muonHists* allMuons;
    muonHists* isoMuons;

    dijetHists* lead;
    dijetHists* subl;
    TH2F* lead_m_vs_subl_m;

    dijetHists* leadSt;
    dijetHists* sublSt;
    TH2F* leadSt_m_vs_sublSt_m;

    dijetHists* leadM;
    dijetHists* sublM;
    TH2F* leadM_m_vs_sublM_m;

    dijetHists* close;
    dijetHists* other;
    TH2F* close_m_vs_other_m;

    // Event Level
    vecHists* v4j;
    TH1F* dBB;
    TH1F* xZZ;
    TH1F* mZZ;
    TH1F* xZH;
    TH1F* mZH;
    
    TH1F* truthM4b;
    TH2F* truthM4b_vs_mZH;

    viewHists(std::string, fwlite::TFileService&, bool isMC = false);
    void Fill(eventData*, std::unique_ptr<eventView>&);
    ~viewHists(); 

  };

}
#endif // viewHists_H
