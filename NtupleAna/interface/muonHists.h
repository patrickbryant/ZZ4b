// -*- C++ -*-
#if !defined(muonHists_H)
#define muonHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/muonData.h"
#include "ZZ4b/NtupleAna/interface/vecHists.h"

namespace NtupleAna {

  class muonHists {
  public:
    TFileDirectory dir;
    
    vecHists* v;
    TH1F* quality;
    TH1F* isolation;
    TH1F* dR;

    muonHists(std::string, fwlite::TFileService&, std::string title = "");
    void Fill(muon*, float);
    ~muonHists(); 

  };

}
#endif // muonHists_H
