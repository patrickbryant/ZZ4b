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

    MeTHists* ChsMeT;
    MeTHists* MeT;
    MeTHists* TrkMeT;

    TH1F* nIsoMed25Leps;

    tTbarEventHists(std::string name, fwlite::TFileService& fs, bool isMC = false, std::string histDetailLevel = "", bool _debug = false);
    void Fill(tTbarEventData* event);
    ~tTbarEventHists(); 

  };

}
#endif // tTbarEventHists_H
