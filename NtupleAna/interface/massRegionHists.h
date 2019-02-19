// -*- C++ -*-
#if !defined(massRegionHists_H)
#define massRegionHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"
#include "ZZ4b/NtupleAna/interface/viewHists.h"

namespace NtupleAna {

  class massRegionHists {
  public:
    TFileDirectory dir;
    bool blind;

    viewHists* inclusive;
    viewHists* ZZSR;
    viewHists* ZHSR;

    massRegionHists(std::string, fwlite::TFileService&, bool _blind = true);
    void Fill(eventData*, eventView*);
    ~massRegionHists(); 

  };

}
#endif // massRegionHists_H
