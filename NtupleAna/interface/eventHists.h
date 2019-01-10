// -*- C++ -*-
#if !defined(eventHists_H)
#define eventHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/jetHists.h"
#include "ZZ4b/NtupleAna/interface/muonHists.h"

namespace NtupleAna {

  class eventHists {
  public:
    TFileDirectory dir;
    
    
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

    eventHists(std::string, fwlite::TFileService&);
    void Fill(eventData*);
    ~eventHists(); 

  };

}
#endif // eventHists_H
