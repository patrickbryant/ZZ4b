// -*- C++ -*-
#if !defined(massRegionHists_H)
#define massRegionHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/eventView.h"
#include "ZZ4b/nTupleAnalysis/interface/viewHists.h"

namespace nTupleAnalysis {

  class massRegionHists {
  public:
    TFileDirectory dir;
    bool blind;
    bool debug;

    viewHists* inclusive = NULL;
    viewHists* notSR = NULL;

    viewHists* ZHSR = NULL;
    viewHists* ZHCR = NULL;
    viewHists* ZHSB = NULL;

    viewHists* ZH = NULL;
    // viewHists* ZH_SvB_high;
    // viewHists* ZH_SvB_low;

    viewHists* ZZSR = NULL;
    viewHists* ZZCR = NULL;
    viewHists* ZZSB = NULL;
    viewHists* ZZ = NULL;

    viewHists* HHSR = NULL;
    viewHists* HHCR = NULL;
    viewHists* HHSB = NULL;
    viewHists* HH = NULL;


    viewHists* SR = NULL;
    viewHists* SRNoHH = NULL;
    viewHists* CR = NULL;
    viewHists* SB = NULL;
    viewHists* SCSR = NULL;

    massRegionHists(std::string, fwlite::TFileService&, bool isMC = false, bool _blind = true, std::string histDetailLevel = "", bool _debug = false, eventData* = NULL);
    void Fill(eventData*, std::shared_ptr<eventView>&);
    ~massRegionHists(); 

  };

}
#endif // massRegionHists_H
