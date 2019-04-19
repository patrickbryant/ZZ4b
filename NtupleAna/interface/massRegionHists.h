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

    viewHists* ZZCR;
    viewHists* ZHCR;

    viewHists* ZZSB;
    viewHists* ZHSB;

    viewHists* ZH;
    viewHists* ZH_SvB_high;
    viewHists* ZH_SvB_low;

    massRegionHists(std::string, fwlite::TFileService&, bool isMC = false, bool _blind = true);
    void Fill(eventData*, std::unique_ptr<eventView>&);
    ~massRegionHists(); 

  };

}
#endif // massRegionHists_H
