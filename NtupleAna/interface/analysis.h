// -*- C++ -*-
#if !defined(analysis_H)
#define analysis_H

#include <TChain.h>
#include <TTree.h>
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/cutflowHists.h"
#include "ZZ4b/NtupleAna/interface/eventHists.h"

namespace NtupleAna {

  class analysis {
  public:

    TChain* tree;
    bool debug = false;
    int treeEvents;
    eventData* event;
    cutflowHists* cutflow;

    eventHists* allEvents;
    eventHists* passPreSel;

    int nEvents   = 0;
    float lumi    = 1;
    float kFactor = 1;

    bool writePicoAOD = false;
    TTree* picoAODTree;

    analysis(TChain*, fwlite::TFileService&, bool);
    void createPicoAOD(fwlite::TFileService&);
    int eventLoop(int);
    int processEvent();
    ~analysis();

  };

}
#endif // analysis_H

