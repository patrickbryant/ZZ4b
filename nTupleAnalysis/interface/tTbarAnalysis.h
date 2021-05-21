
// -*- C++ -*-
#if !defined(tTbarAnalysis_H)
#define tTbarAnalysis_H

#include <ctime>
#include <sys/resource.h>

#include <TChain.h>
#include <TTree.h>
#include <TSpline.h>
#include "DataFormats/FWLite/interface/InputSource.h" //for edm::LuminosityBlockRange
#include "nTupleAnalysis/baseClasses/interface/brilCSV.h"
#include "nTupleAnalysis/baseClasses/interface/initBranch.h"
#include "ZZ4b/nTupleAnalysis/interface/tTbarEventData.h"
#include "ZZ4b/nTupleAnalysis/interface/tTbarCutFlowHists.h"
#include "ZZ4b/nTupleAnalysis/interface/tTbarEventHists.h"
#include <fstream>


namespace nTupleAnalysis {

  class tTbarAnalysis {
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
    bool mcUnitWeight  = false;

    int treeEvents;
    tTbarEventData* event;
    tTbarCutFlowHists* cutflow;

    tTbarEventHists* allEvents   = NULL;
    tTbarEventHists* passPreSel  = NULL;
    tTbarEventHists* passEMuSel  = NULL;
    tTbarEventHists* passMuSel   = NULL;
    tTbarEventHists* passEMuSelAllMeT  = NULL;
    tTbarEventHists* passMuSelAllMeT   = NULL;

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
    double fourbkfactor = 1;

    bool writePicoAOD = false;
    TFile* picoAODFile;
    TTree* picoAODEvents;
    TTree* picoAODRuns;
    TTree* picoAODLumiBlocks;

    // debugging
    long int currentEvent = 0;

    //Monitoring Variables
    long int percent;
    std::clock_t start;
    double timeTotal;
    double previousMonitorTime = 0;
    double timeElapsed = 0;
    long int previousMonitorEvent = 0;
    long int eventsElapsed;
    double eventRate = 0;
    double timeRemaining;
    int hours;
    int minutes;
    int seconds;
    int who = RUSAGE_SELF;
    struct rusage usage;
    long int usageMB;
 
    tTbarAnalysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _isMC, std::string _year,
		  std::string histDetailLevel, bool _debug, 
		  std::string bjetSF = "", std::string btagVariations = "central",
		  std::string JECSyst = "", std::string friendFile = "");


    void createPicoAOD(std::string fileName, bool copyInputPicoAOD = true);

    bool alreadyFilled=false;
    float m4jPrevious=0;
    void picoAODFillEvents();

    // Write out all event and run numbers to histogram file
    bool writeOutEventNumbers = false;
    std::vector<UInt_t> passed_runs;
    std::vector<ULong64_t> passed_events;
    TFile* histFile = NULL;

    void addDerivedQuantitiesToPicoAOD();
    void storePicoAOD();
    void monitor(long int);
    int eventLoop(int maxEvents, long int firstEvent = 0);
    int processEvent();
    bool passLumiMask();
    std::map<edm::LuminosityBlockID, float> lumiData;
    void getLumiData(std::string);
    void countLumi();
    void storeJetCombinatoricModel(std::string fileName);
    void storeJetCombinatoricModel(std::string jcmName, std::string fileName);
    void loadJetCombinatoricModel(std::string jcmName);
    void storeReweight(std::string);

    ~tTbarAnalysis();

  };

}
#endif // analysis_H

