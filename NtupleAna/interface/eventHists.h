// -*- C++ -*-
#if !defined(eventHists_H)
#define eventHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"
#include "ZZ4b/NtupleAna/interface/vecHists.h"
#include "ZZ4b/NtupleAna/interface/jetHists.h"
#include "ZZ4b/NtupleAna/interface/muonHists.h"
#include "ZZ4b/NtupleAna/interface/massRegionHists.h"

namespace NtupleAna {

  class eventHists {
  public:
    bool doViews;
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

    // Event Level
    vecHists* v4j;
    massRegionHists* allViews;
    massRegionHists* mainView;

    eventHists(std::string, fwlite::TFileService&, bool _doViews = false, bool blind = true);
    void Fill(eventData*);
    void Fill(eventData* event, std::vector<std::unique_ptr<eventView>> &views);
    ~eventHists(); 

  };

}
#endif // eventHists_H
