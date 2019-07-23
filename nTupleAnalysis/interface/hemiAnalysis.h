// -*- C++ -*-
#if !defined(hemiAnalysis_H)
#define hemiAnalysis_H

#include <ctime>
#include <sys/resource.h>

#include <TChain.h>
#include <TTree.h>
#include <TSpline.h>
#include "DataFormats/FWLite/interface/InputSource.h" //for edm::LuminosityBlockRange
#include "nTupleAnalysis/baseClasses/interface/brilCSV.h"
#include "nTupleAnalysis/baseClasses/interface/initBranch.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/cutflowHists.h"
#include "ZZ4b/nTupleAnalysis/interface/tagCutflowHists.h"
#include "ZZ4b/nTupleAnalysis/interface/eventHists.h"
#include "ZZ4b/nTupleAnalysis/interface/tagHists.h"
#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"
#include "nTupleAnalysis/baseClasses/interface/EventDisplayData.h"

namespace nTupleAnalysis {

  class hemiAnalysis {
  public:

    
    bool debug = false;
    int m_nTreeHemis;

    //
    // Hemisphere Mixing 
    //
    bool writeHSphereFile = false;
    hemisphereMixTool* hMixToolLoad = NULL;

    hemiAnalysis(std::string, fwlite::TFileService&, bool);
    //void createHemisphereLibrary(std::string, fwlite::TFileService& fs );
    //void storeHemiSphereFile();
    int hemiLoop(int maxHemi);
    ~hemiAnalysis();

  };

}
#endif // hemiAnalysis_H

