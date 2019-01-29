// -*- C++ -*-
#if !defined(analysis_H)
#define analysis_H

#include <TChain.h>
#include <TTree.h>
#include "ZZ4b/NtupleAna/interface/initBranch.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/cutflowHists.h"
#include "ZZ4b/NtupleAna/interface/eventHists.h"

namespace NtupleAna {

  class analysis {
  public:

    TChain* events;
    TChain* runs;
    ULong64_t genEventCount;
    double  genEventSumw;
    double  genEventSumw2;
    
    bool debug = false;
    bool isMC  = false;
    int treeEvents;
    eventData* event;
    cutflowHists* cutflow;

    eventHists* allEvents;
    eventHists* passPreSel;

    long int nEvents = 0;
    double lumi      = 1;
    double kFactor   = 1;

    bool writePicoAOD = false;
    TFile* picoAODFile;
    TTree* picoAODEvents;
    TTree* picoAODRuns;

    analysis(TChain*, TChain*, fwlite::TFileService&, bool, bool);
    void createPicoAOD(std::string);
    void storePicoAOD();
    int eventLoop(int);
    int processEvent();
    ~analysis();

  };

}
#endif // analysis_H

