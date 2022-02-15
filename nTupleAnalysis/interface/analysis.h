
// -*- C++ -*-
#if !defined(analysis_H)
#define analysis_H

#include <ctime>
#include <chrono>
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
#include "ZZ4b/nTupleAnalysis/interface/triggerStudy.h"
#include "ZZ4b/nTupleAnalysis/interface/lumiHists.h"
#include "ZZ4b/nTupleAnalysis/interface/unsupervised.h"
#include <fstream>


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
    bool isDataMCMix  = false;
    bool skip4b  = false;
    bool skip3b  = false;
    bool mcUnitWeight  = false;
    bool makePSDataFromMC = false;
    bool removePSDataFromMC = false;
    bool blind = true;
    bool calcTrigWeights = false;


    int treeEvents;
    eventData* event;
    tagCutflowHists* cutflow;
    lumiHists* lumiCounts    = NULL;
    float  lumiLastWrite    = 0;

    eventHists* allEvents   = NULL;
    tagHists* passPreSel    = NULL;
    //tagHists* passDijetMass = NULL;
    tagHists* passMDRs      = NULL;
    tagHists* passSvB       = NULL;
    tagHists* passMjjOth    = NULL;
    tagHists* failrWbW2     = NULL;
    tagHists* passMuon      = NULL;
    tagHists* passDvT05     = NULL;
    tagHists* passSRvsSB1p  = NULL;
    tagHists* passSRvsSB10p = NULL;
    // tagHists* makeDijetPullTagged = NULL;
    std::map<float, std::map<float, tagHists*>> passSRvsSB_xp; // unsupervised
    tagHists* passTTCR      = NULL;

    triggerStudy* trigStudy  = NULL;
    triggerStudy* trigStudyMjjOth  = NULL;


    long int nEvents = 0;
    double lumi      = 1;
    std::vector<edm::LuminosityBlockRange> lumiMask;
    edm::LuminosityBlockID prevLumiID;
    //UInt_t prevLumiBlock = 0;
    UInt_t firstRun      = 1e9;
    UInt_t lastRun       = 0;
    edm::RunNumber_t prevRun;
    //UInt_t nruns = 0;
    UInt_t nls   = 0;
    float  lumiID_intLumi = 0;
    float  intLumi = 0;
    bool   lumiID_passL1  = false;
    bool   lumiID_passHLT = false;
    float  intLumi_passL1  = 0;
    float  intLumi_passHLT = 0;
    double kFactor = 1;
    double xs = 1;
    double fourbkfactor = 1;

    bool writePicoAOD = false;
    bool fastSkim = false;
    bool looseSkim = false;

    TFile* picoAODFile;
    TTree* picoAODEvents;
    TTree* picoAODRuns;
    TTree* picoAODLumiBlocks;

    // debugging
    long int currentEvent = 0;

    //Monitoring Variables
    long int percent;
    std::chrono::time_point<std::chrono::system_clock> start;
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

    //reweight
    bool  doReweight = false;
    TSpline3* spline = NULL;

    //
    // Hemisphere Mixing 
    //
    bool writeHSphereFile = false;
    bool writePicoAODBeforeDiJetMass = false;
    hemisphereMixTool* hMixToolCreate3Tag = NULL;
    hemisphereMixTool* hMixToolCreate4Tag = NULL;
    bool emulate4bFrom3b = false;
    unsigned int emulationOffset = 0;

    bool loadHSphereFile = false;
    hemisphereMixTool* hMixToolLoad3Tag = NULL;
    hemisphereMixTool* hMixToolLoad4Tag = NULL;

    
    //
    //  Output Data for the new PicoAOD when using Hemisphere Mixing
    //   (Move these to event data ?
    UInt_t    m_run       =  0;
    UInt_t    m_lumiBlock =  0;
    ULong64_t m_event     =  0;
    Float_t   m_genWeight =  0;
    Float_t   m_bTagSF    =  0;
    Float_t   m_ttbarWeight =  0;
    
    nTupleAnalysis::jetData*  m_mixed_jetData = NULL;
    nTupleAnalysis::muonData*  m_mixed_muonData = NULL;
    nTupleAnalysis::elecData*  m_mixed_elecData = NULL;
    nTupleAnalysis::truthParticle*  m_mixed_truthParticle = NULL;

    Int_t     m_nPVs;
    Int_t     m_nPVsGood;    

    UInt_t    m_h1_run       =  0;
    ULong64_t m_h1_event     =  0;
    Float_t   m_h1_eventWeight     =  0;
    Bool_t    m_h1_hemiSign  =  0;
    Int_t     m_h1_NJet       =  0;
    Int_t     m_h1_NBJet      =  0;
    Int_t     m_h1_NNonSelJet =  0;
    Int_t     m_h1_matchCode =  0;
    Float_t   m_h1_pz                 = 0;
    Float_t   m_h1_pz_sig             = 0;
    Float_t   m_h1_match_pz           = 0;
    Float_t   m_h1_sumpt_t            = 0;
    Float_t   m_h1_sumpt_t_sig        = 0;
    Float_t   m_h1_match_sumpt_t      = 0;
    Float_t   m_h1_sumpt_ta           = 0;
    Float_t   m_h1_sumpt_ta_sig       = 0;
    Float_t   m_h1_match_sumpt_ta     = 0;
    Float_t   m_h1_combinedMass       = 0;
    Float_t   m_h1_combinedMass_sig   = 0;
    Float_t   m_h1_match_combinedMass = 0;
    Float_t   m_h1_match_dist         = 0;

    UInt_t    m_h2_run       =  0;
    ULong64_t m_h2_event     =  0;
    Float_t   m_h2_eventWeight     =  0;
    Bool_t    m_h2_hemiSign  =  0;
    Int_t     m_h2_NJet       =  0;
    Int_t     m_h2_NBJet      =  0;
    Int_t     m_h2_NNonSelJet =  0;
    Int_t     m_h2_matchCode =  0;
    Float_t   m_h2_pz                 = 0;
    Float_t   m_h2_pz_sig             = 0;
    Float_t   m_h2_match_pz           = 0;
    Float_t   m_h2_sumpt_t            = 0;
    Float_t   m_h2_sumpt_t_sig        = 0;
    Float_t   m_h2_match_sumpt_t      = 0;
    Float_t   m_h2_sumpt_ta           = 0;
    Float_t   m_h2_sumpt_ta_sig       = 0;
    Float_t   m_h2_match_sumpt_ta     = 0;
    Float_t   m_h2_combinedMass       = 0;
    Float_t   m_h2_combinedMass_sig   = 0;
    Float_t   m_h2_match_combinedMass = 0;
    Float_t   m_h2_match_dist         = 0;

    TFile* pullHistFile = NULL; // unsupervised
    // TH2F* pull_hist = NULL ;

    analysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _isMC, bool _blind, std::string _year,
	     std::string histDetailLevel, bool _doReweight, bool _debug, bool _fastSkim = false, bool doTrigEmulation = false, bool _calcTrigWeights = false, bool useMCTurnOns=false, bool useUnitTurnOns=false, bool _isDataMCMix=false, bool usePreCalcBTagSFs=false,
	     std::string bjetSF = "", std::string btagVariations = "central",
	     std::string JECSyst = "", std::string friendFile = "",
	     bool looseSkim = false, std::string FvTName = "", std::string reweight4bName = "", std::string reweightDvTName = "",
       std::string bdtWeightFile = "", std::string bdtMethods = "");

    void createPicoAOD(std::string fileName, bool copyInputPicoAOD = true);

    bool alreadyFilled=false;
    float m4jPrevious=0;
    void picoAODFillEvents();

    //
    // only used when overwritting the input picoAOD info (eg: in hemisphere mixing)
    //
    void createPicoAODBranches();


    void createHemisphereLibrary(std::string, fwlite::TFileService& fs );
    void loadHemisphereLibrary(std::vector<std::string> hLibs_3tag, std::vector<std::string> hLibs_4tag, fwlite::TFileService& fs, int maxNHemis, bool useHemiWeights, float mcHemiWeight);

    // Write out all event and run numbers to histogram file
    bool writeOutEventNumbers = false;
    std::vector<UInt_t> passed_runs;
    std::vector<ULong64_t> passed_events;
    TFile* histFile = NULL;

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
    void storeJetCombinatoricModel(std::string fileName);
    void storeJetCombinatoricModel(std::string jcmName, std::string fileName);
    void loadJetCombinatoricModel(std::string jcmName);
    void storeReweight(std::string);
    bool eventPassedSRvsSB1p();
    bool eventPassedSRvsSB10p();
    bool eventPassedSRvsSBxP(int pull_ind = 0, int m4jLowBinIndex = 0); // unsupervised

    ~analysis();

  };

}
#endif // analysis_H

