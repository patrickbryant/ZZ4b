// -*- C++ -*-
#if !defined(analysis_H)
#define analysis_H

#include "DataFormats/FWLite/interface/InputSource.h" //for edm::LuminosityBlockRange

#include <TChain.h>
#include <TTree.h>
#include "ZZ4b/NtupleAna/interface/initBranch.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/cutflowHists.h"
#include "ZZ4b/NtupleAna/interface/tagCutflowHists.h"
#include "ZZ4b/NtupleAna/interface/eventHists.h"
#include "ZZ4b/NtupleAna/interface/tagHists.h"

namespace NtupleAna {

  class analysis {
  public:

    TChain* events;
    TChain* runs;
    Long64_t genEventCount;
    double_t genEventSumw;
    double_t genEventSumw2;
    
    bool debug = false;
    std::string year;
    bool isMC  = false;
    bool blind = true;
    int treeEvents;
    eventData* event;
    tagCutflowHists* cutflow;

    eventHists* allEvents;
    tagHists* passPreSel;

    long int nEvents = 0;
    double lumi      = 1;
    std::vector<edm::LuminosityBlockRange> lumiMask;
    double kFactor   = 1;

    bool writePicoAOD = false;
    TFile* picoAODFile;
    TTree* picoAODEvents;
    TTree* picoAODRuns;

    analysis(TChain*, TChain*, fwlite::TFileService&, bool, bool, std::string, bool);
    void createPicoAOD(std::string);
    void storePicoAOD();
    int eventLoop(int);
    int processEvent();
    bool passLumiMask();
    ~analysis();

  };

}
#endif // analysis_H

