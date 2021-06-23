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
#include "nTupleAnalysis/baseClasses/interface/elecHists.h"
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
    TH1F* nIsoMed25Muons;
    TH1F* nIsoMed40Muons;
    muonHists* allMuons;
    muonHists* muons_isoMed25;
    muonHists* muons_isoMed40;

    TH1F* nAllElecs;
    TH1F* nIsoMed25Elecs;
    TH1F* nIsoMed40Elecs;
    elecHists* allElecs;
    elecHists* elecs_isoMed25;
    elecHists* elecs_isoMed40;

    // Event Level
    fourVectorHists* v4j;
    massRegionHists* allViews = NULL;
    massRegionHists* mainView;

    eventHists(std::string, fwlite::TFileService&, bool _doViews = false, bool isMC = false, bool blind = true, std::string histDetailLevel = "", bool _debug = false, eventData* event=NULL);
    void Fill(eventData*);
    void Fill(eventData* event, std::vector<std::shared_ptr<eventView>> &views);
    ~eventHists(); 

  };

}
#endif // eventHists_H
