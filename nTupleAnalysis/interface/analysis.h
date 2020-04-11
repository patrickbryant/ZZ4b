
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
#include "ZZ4b/nTupleAnalysis/interface/triggerStudy.h"


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
    bool blind = true;
    int histogramming = 1e6;  // The bigger the more points in the cutflow are plotted
    int histDetailLevel = 1e6;  // The bigger the more histograms are created
                                // < 10 turns off all views 
                                // < 5  turns off ZZ/ZH specific regions (Just keep inclusive SB/CR/SR)
    int treeEvents;
    eventData* event;
    tagCutflowHists* cutflow;

    eventHists* allEvents   = NULL;
    tagHists* passPreSel    = NULL;
    tagHists* passDijetMass = NULL;
    tagHists* passMDRs      = NULL;
    tagHists* passXWt       = NULL;
    tagHists* passMDCs      = NULL;
    tagHists* passDEtaBB    = NULL;
    //tagHists* passDEtaBBNoTrig  = NULL;
    //tagHists* passDEtaBBNoTrigJetPts  = NULL;

    triggerStudy* trigStudy  = NULL;


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
    bool fastSkim = false;
    bool looseSkim = false;
    bool doTrigEmulation = false;
    bool doTrigStudy = false;
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

    bool loadHSphereFile = false;
    hemisphereMixTool* hMixToolLoad3Tag = NULL;
    hemisphereMixTool* hMixToolLoad4Tag = NULL;


    //
    //  Output Data for the new PicoAOD when using Hemisphere Mixing
    //   (Move these to event data ?
    UInt_t    m_run       =  0;
    UInt_t    m_lumiBlock =  0;
    ULong64_t m_event     =  0;
    
    nTupleAnalysis::jetData*  m_mixed_jetData = NULL;
    nTupleAnalysis::muonData*  m_mixed_muonData = NULL;

    Int_t     m_nPVs;
    Int_t     m_nPVsGood;    

    //2016
    Bool_t m_HLT_4j45_3b087     ;
    Bool_t m_HLT_2j90_2j30_3b087;
    //2018
    Bool_t m_HLT_HT330_4j_75_60_45_40_3b;
    Bool_t m_HLT_4j_103_88_75_15_2b_VBF1;
    Bool_t m_HLT_4j_103_88_75_15_1b_VBF2;
    Bool_t m_HLT_2j116_dEta1p6_2b       ;
    Bool_t m_HLT_J330_m30_2b            ;
    Bool_t m_HLT_j500                   ;
    Bool_t m_HLT_2j300ave               ;


    UInt_t    m_h1_run       =  0;
    ULong64_t m_h1_event     =  0;
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


    analysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _isMC, bool _blind, std::string _year,
	     int _histogramming, int _histDetailLevel, bool _doReweight, bool _debug, bool _fastSkim = false, bool _doTrigEmulation = false, bool _doTrigStudy = false, bool _mcUnitWeight=false, bool _isDataMCMix=false,bool _skip4b=false, bool _skip3b=false,
	     std::string bjetSF = "", std::string btagVariations = "central",
	     std::string JECSyst = "", std::string friendFile = "",
	     bool looseSkim = false);

    void createPicoAOD(std::string fileName, bool copyInputPicoAOD = true);

    bool alreadyFilled=false;
    float m4jPrevious=0;
    void picoAODFillEvents();

    //
    // only used when overwritting the input picoAOD info (eg: in hemisphere mixing)
    //
    void createPicoAODBranches();


    void createHemisphereLibrary(std::string, fwlite::TFileService& fs );
    void loadHemisphereLibrary(std::vector<std::string> hLibs_3tag, std::vector<std::string> hLibs_4tag, fwlite::TFileService& fs, int maxNHemis);

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
    ~analysis();

  };

}
#endif // analysis_H

