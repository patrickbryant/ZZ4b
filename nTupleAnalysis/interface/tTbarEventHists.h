// -*- C++ -*-
#if !defined(tTbarEventHists_H)
#define tTbarEventHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/tTbarEventData.h"
#include "nTupleAnalysis/baseClasses/interface/jetHists.h"
#include "nTupleAnalysis/baseClasses/interface/muonHists.h"
#include "nTupleAnalysis/baseClasses/interface/MeTHists.h"
#include "nTupleAnalysis/baseClasses/interface/elecHists.h"
#include "nTupleAnalysis/baseClasses/interface/dijetHists.h"
#include "nTupleAnalysis/baseClasses/interface/trijetHists.h"

using namespace nTupleAnalysis;

namespace nTupleAnalysis {

  class tTbarEventHists {
  public:

    TFileDirectory dir;
    bool debug;

    // Object Level
    TH1F*     nAllJets;
    TH1F*     nSelJets;
    TH1F*     nTagJets;
    jetHists*  allJets;
    jetHists*  selJets;
    jetHists*  tagJets;

    TH1F* nAllMuons;
    TH1F* nIsoMuons;
    TH1F* nIsoHighPtMuons;
    muonHists* allMuons;
    muonHists* muons_iso;
    muonHists* muons_isoHighPt;

    TH1F* nAllElecs;
    TH1F* nIsoElecs;
    TH1F* nIsoHighPtElecs;
    elecHists* allElecs;
    elecHists* elecs_iso;
    elecHists* elecs_isoHighPt;

    MeTHists* ChsMeT;
    MeTHists* MeT;
    MeTHists* TrkMeT;

    dijetHists* w;
    trijetHists* t;

    TH1F* nIsoLeps;

    tTbarEventHists(std::string name, fwlite::TFileService& fs, bool isMC = false, std::string histDetailLevel = "", bool _debug = false);
    void Fill(tTbarEventData* event);
    ~tTbarEventHists(); 

  };

}
#endif // tTbarEventHists_H
