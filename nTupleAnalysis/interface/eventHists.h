// -*- C++ -*-
#if !defined(eventHists_H)
#define eventHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/eventView.h"
#include "nTupleAnalysis/baseClasses/interface/fourVectorHists.h"
#include "nTupleAnalysis/baseClasses/interface/jetHists.h"
#include "nTupleAnalysis/baseClasses/interface/muonHists.h"
#include "ZZ4b/nTupleAnalysis/interface/massRegionHists.h"

using namespace nTupleAnalysis;

namespace nTupleAnalysis {

  class eventHists {
  public:
    bool doViews;
    TFileDirectory dir;
    bool debug;
    
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
    fourVectorHists* v4j;
    massRegionHists* allViews;
    massRegionHists* mainView;

    eventHists(std::string, fwlite::TFileService&, bool _doViews = false, bool isMC = false, bool blind = true, bool _debug = false);
    void Fill(eventData*);
    void Fill(eventData* event, std::vector<std::unique_ptr<eventView>> &views);
    ~eventHists(); 

  };

}
#endif // eventHists_H
