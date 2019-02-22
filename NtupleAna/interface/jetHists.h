// -*- C++ -*-
#if !defined(jetHists_H)
#define jetHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/jetData.h"
#include "ZZ4b/NtupleAna/interface/vecHists.h"

namespace NtupleAna {

  class jetHists {
  public:
    TFileDirectory dir;
    
    vecHists* v;
    TH1F* deepB;
    TH1F* CSVv2;

    jetHists(std::string, fwlite::TFileService&, std::string title = "");
    void Fill(std::shared_ptr<jet>&, float);
    ~jetHists(); 

  };

}
#endif // jetHists_H
