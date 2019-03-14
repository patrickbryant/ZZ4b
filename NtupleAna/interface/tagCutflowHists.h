// -*- C++ -*-
#if !defined(tagCutflowHists_H)
#define tagCutflowHists_H

#include <iostream>
#include <TH1F.h>
#include "ZZ4b/NtupleAna/interface/cutflowHists.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"

namespace NtupleAna {

  class tagCutflowHists {
  public:
    TFileDirectory dir;
    
    cutflowHists* threeTag;
    cutflowHists*  fourTag;

    tagCutflowHists(std::string, fwlite::TFileService&, bool isMC = false);
    void Fill(eventData*, std::string, bool fillAll = false);
    ~tagCutflowHists(); 

  };

}
#endif // tagCutflowHists_H
