// -*- C++ -*-
#if !defined(triggerStudy_H)
#define triggerStudy_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/tagHists.h"

using namespace nTupleAnalysis;

namespace nTupleAnalysis {

  class triggerStudy {
  public:

    triggerStudy(std::string name, fwlite::TFileService& fs, std::string year, bool isMC, bool blind, std::string histDetailLevel, bool _debug = false);
    void Fill(eventData* event);
    ~triggerStudy(); 

  private:

    TFileDirectory dir;
    bool debug;

    //
    tagHists* hist_HLT_OR       = NULL;
    tagHists* hist_HLT_4j_3b    = NULL;
    tagHists* hist_HLT_2b       = NULL;
    tagHists* hist_HLT_2j_2j_3b = NULL;

    tagHists* hist_EMU_OR       = NULL;
    tagHists* hist_EMU_4j_3b    = NULL;
    tagHists* hist_EMU_2b       = NULL;
    tagHists* hist_EMU_2j_2j_3b = NULL;
  

  };

}
#endif // triggerStudy_H
