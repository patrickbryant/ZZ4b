
// -*- C++ -*-
#if !defined(analysis_H)
#define analysis_H

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


namespace nTupleAnalysis {

  class analysis {
  public:

    TChain* events;
    TChain* runs;
    TChain* lumiBlocks;
    Long64_t genEventCount;
    double_t genEventSumw;
    double_t genEventSumw2;
    Long64_t mcEventCount = 0;
    double_t mcEventSumw  = 0;
    double_t mcEventSumw2 = 0;
    
    bool debug = false;
    std::string year;
    bool isMC  = false;
    bool blind = true;
    int histogramming = 1e6;
    int treeEvents;
    eventData* event;
    tagCutflowHists* cutflow;

    eventHists* allEvents   = NULL;
    tagHists* passPreSel    = NULL;
    tagHists* passDijetMass = NULL;
    tagHists* passMDRs      = NULL;
    tagHists* passMDCs      = NULL;
    tagHists* passDEtaBB    = NULL;
    //tagHists* passDEtaBBNoTrig  = NULL;
    //tagHists* passDEtaBBNoTrigJetPts  = NULL;

    long int nEvents = 0;
    double lumi      = 1;
    std::vector<edm::LuminosityBlockRange> lumiMask;
    UInt_t prevLumiBlock = 0;
    UInt_t firstRun      = 1e9;
    UInt_t lastRun       = 0;
    UInt_t prevRun       = 0;
    UInt_t nruns = 0;
    UInt_t nls   = 0;
    float  intLumi = 0;
    double kFactor = 1;
    double xs = 1;

    bool writePicoAOD = false;
    TFile* picoAODFile;
    TTree* picoAODEvents;
    TTree* picoAODRuns;
    TTree* picoAODLumiBlocks;

    //Monitoring Variables
    long int percent;
    std::clock_t start;
    double duration;
    double eventRate;
    double timeRemaining;
    int minutes;
    int seconds;
    int who = RUSAGE_SELF;
    struct rusage usage;
    long int usageMB;

    //reweight
    bool  doReweight = false;
    TSpline3* spline = NULL;

    //
    // Hemisphere Mixing 
    //
    bool writeHSphereFile = false;
    hemisphereMixTool* hMixToolCreate3Tag = NULL;
    hemisphereMixTool* hMixToolCreate4Tag = NULL;

    bool loadHSphereFile = false;
    hemisphereMixTool* hMixToolLoad3Tag = NULL;
    hemisphereMixTool* hMixToolLoad4Tag = NULL;
    

    analysis(TChain*, TChain*, TChain*, fwlite::TFileService&, bool, bool, std::string, int, bool);
    void createPicoAOD(std::string);
    void createHemisphereLibrary(std::string, fwlite::TFileService& fs );
    void loadHemisphereLibrary(std::vector<std::string> hLibs_3tag, std::vector<std::string> hLibs_4tag, fwlite::TFileService& fs);
    void addDerivedQuantitiesToPicoAOD();
    void storePicoAOD();
    void storeHemiSphereFile();
    void monitor(long int);
    int eventLoop(int maxEvents, long int firstEvent = 0);
    int processEvent();
    bool passLumiMask();
    std::map<edm::LuminosityBlockID, float> lumiData;
    void getLumiData(std::string);
    void countLumi();
    void storeJetCombinatoricModel(std::string);
    void storeReweight(std::string);
    void applyReweight();
    ~analysis();

  };

}
#endif // analysis_H

