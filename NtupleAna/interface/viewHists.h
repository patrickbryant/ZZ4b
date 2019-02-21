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
#include "ZZ4b/NtupleAna/interface/dijetHists.h"

namespace NtupleAna {

  class viewHists {
  public:
    TFileDirectory dir;
    
    // Object Level
    dijetHists* lead;
    dijetHists* subl;
    TH2F* lead_m_vs_subl_m;

    dijetHists* leadSt;
    dijetHists* sublSt;
    TH2F* leadSt_m_vs_sublSt_m;

    dijetHists* leadM;
    dijetHists* sublM;
    TH2F* leadM_m_vs_sublM_m;

    // Event Level
    vecHists* v4j;
    TH1F* dBB;
    TH1F* xZZ;
    TH1F* mZZ;
    TH1F* xZH;
    TH1F* mZH;

    viewHists(std::string, fwlite::TFileService&);
    void Fill(eventData*, std::unique_ptr<eventView>&);
    ~viewHists(); 

  };

}
#endif // viewHists_H
