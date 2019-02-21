// -*- C++ -*-
#if !defined(dijetHists_H)
#define dijetHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/dijet.h"
#include "ZZ4b/NtupleAna/interface/vecHists.h"

namespace NtupleAna {

  class dijetHists {
  public:
    TFileDirectory dir;
    
    vecHists* v;
    TH1F* dR;

    dijetHists(std::string, fwlite::TFileService&, std::string title = "");
    void Fill(std::shared_ptr<dijet>&, float);
    ~dijetHists(); 

  };

}
#endif // dijetHists_H
