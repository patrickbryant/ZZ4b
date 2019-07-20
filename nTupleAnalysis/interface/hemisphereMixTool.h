// -*- C++ -*-
#if !defined(hemisphereMixTool_H)
#define hemisphereMixTool_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include "TVector2.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

namespace nTupleAnalysis {

  class hemisphereMixTool {
  public:
    //TFileDirectory dir;
    bool m_debug;
    //TH1F* unitWeight;

    hemisphereMixTool(std::string, fwlite::TFileService&, bool);

    TVector2 getThrustAxis(eventData* event);

    void addEvent(eventData*);
    ~hemisphereMixTool(); 

  private:
    TVector2 calcThrust(const std::vector<TVector2>& jetPts);
    void calcT(const std::vector<TVector2>& momenta, double& t, TVector2& taxis);


  };

}
#endif // hemisphereMixTool_H
