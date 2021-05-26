// -*- C++ -*-
#if !defined(cutflowHists_H)
#define cutflowHists_H

#include <iostream>
#include <boost/range/numeric.hpp>
#include <boost/range/adaptor/map.hpp>
#include <TH1F.h>
#include <TH2F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

namespace nTupleAnalysis {

  class cutflowHists {
  public:
    bool debug = false;
    TFileDirectory dir;
    
    TH1I* unitWeight;
    TH1D* weighted;

    TH2F* truthM4b = NULL;

    cutflowHists(std::string, fwlite::TFileService&, bool, bool);
    void BasicFill(const std::string&, eventData*, bool doTriggers = false);
    void BasicFill(const std::string&, eventData*, float weight);
    void Fill(const std::string&, eventData*);

    void labelsDeflate();

    void AddCut(std::string cut);
    
    ~cutflowHists(); 

  };

}
#endif // cutflowHists_H
