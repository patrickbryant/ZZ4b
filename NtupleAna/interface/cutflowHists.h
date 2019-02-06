// -*- C++ -*-
#if !defined(cutflowHists_H)
#define cutflowHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"

namespace NtupleAna {

  class cutflowHists {
  public:
    TFileDirectory dir;
    
    TH1F* unitWeight;
    TH1F* weighted;

    cutflowHists(std::string, fwlite::TFileService&);
    void Fill(std::string, float);
    ~cutflowHists(); 

  };

}
#endif // cutflowHists_H
