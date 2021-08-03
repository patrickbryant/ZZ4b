// -*- C++ -*-

#if !defined(eventData_H)
#define eventData_H

#include <iostream>
#include <boost/range/numeric.hpp>
#include <boost/range/adaptor/map.hpp>
#include <TChain.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include "nTupleAnalysis/baseClasses/interface/initBranch.h"
#include "nTupleAnalysis/baseClasses/interface/truthData.h"
#include "nTupleAnalysis/baseClasses/interface/jetData.h"
#include "nTupleAnalysis/baseClasses/interface/muonData.h"
#include "nTupleAnalysis/baseClasses/interface/elecData.h"
#include "nTupleAnalysis/baseClasses/interface/dijet.h"
#include "nTupleAnalysis/baseClasses/interface/trijet.h"
#include "nTupleAnalysis/baseClasses/interface/trigData.h"
#include "ZZ4b/nTupleAnalysis/interface/eventView.h"
#include "TriggerEmulator/nTupleAnalysis/interface/TrigEmulatorTool.h"
#if SLC6 == 0 //Defined in ZZ4b/nTupleAnalysis/BuildFile.xml 
#include "ZZ4b/nTupleAnalysis/interface/multiClassifierONNX.h"
#endif

// for jet pseudoTag calculations
#include <TRandom3.h>
#include <numeric> 
#include <boost/math/special_functions/binomial.hpp> 

namespace nTupleAnalysis {

  class eventData {

  public:
    // Member variables
    TChain* tree;
    bool isMC;
    float year;
    bool debug;
    bool printCurrentFile = true;
    bool fastSkim = false;
    bool looseSkim = false;
    UInt_t    run       =  0;
    UInt_t    lumiBlock =  0;
    ULong64_t event     =  0;
    Int_t     nPVs = 0;
    Int_t     nPVsGood = 0;
    Float_t   reweight = 1.0;

    Float_t   FvT = 1.0;
    Float_t   FvT_pd4 = 1.0;
    Float_t   FvT_pd3 = 1.0;
    Float_t   FvT_pt4 = 1.0;
    Float_t   FvT_pt3 = 1.0;
    Float_t   FvT_pm4 = 1.0;
    Float_t   FvT_pm3 = 1.0;
    Float_t   FvT_pt  = 1.0;
    Float_t   FvT_q_1234 = -99.0;
    Float_t   FvT_q_1324 = -99.0;
    Float_t   FvT_q_1423 = -99.0;
    Float_t   FvT_q_score[3] = {-99.0};
    Float_t   SvB_ps  = -99.0;
    Float_t   SvB_pzz = -99.0;
    Float_t   SvB_pzh = -99.0;
    Float_t   SvB_ptt = -99.0;
    Float_t   SvB_q_1234 = -99.0;
    Float_t   SvB_q_1324 = -99.0;
    Float_t   SvB_q_1423 = -99.0;
    Float_t   SvB_q_score[3] = {-99.0};
    Float_t   SvB_MA_ps  = -99.0;
    Float_t   SvB_MA_pzz = -99.0;
    Float_t   SvB_MA_pzh = -99.0;
    Float_t   SvB_MA_ptt = -99.0;
    Float_t   SvB_MA_q_1234 = -99.0;
    Float_t   SvB_MA_q_1324 = -99.0;
    Float_t   SvB_MA_q_1423 = -99.0;
    Float_t   SvB_MA_q_score[3] = {-99.0};
    Float_t   reweight4b = 1.0;
    Float_t   DvT_raw = 0.0;
    Float_t   DvT_pt = 0.0;
    Float_t   DvT_pm = 1.0;
    Float_t   DvT_pd = 1.0;

    std::map<std::string, Float_t*> classifierVariables;

    Float_t   genWeight =  1;
    Float_t   weight    =  1;
    Float_t   weightNoTrigger    =  1;
    Float_t   trigWeight =  1;
    Float_t   mcWeight  =  1;
    Float_t   mcPseudoTagWeight = 1;
    Float_t   bTagSF = 1;
    int       nTrueBJets = 0;

    // used for hemisphere mixing
    Float_t   inputBTagSF = 0;

    nTupleAnalysis::truthData* truth = NULL;

    //Predefine btag sorting functions
    float       bTag    = 0.8484;//medium WP for CSVv2 https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80XReReco
    std::string bTagger = "CSVv2";
    bool (*sortTag)(std::shared_ptr<nTupleAnalysis::jet>&, std::shared_ptr<nTupleAnalysis::jet>&);

    //triggers
    std::map<std::string, bool> L1_triggers;
    std::map<std::string, bool> HLT_triggers;
    std::map<std::string, std::map<std::string, bool*>> HLT_L1_seeds;
    bool passL1              = false;
    bool passHLT             = false;

    std::map<std::string, bool*> L1_triggers_mon;

    //
    //  trigger Emulation
    //
  private:
    TriggerEmulator::TrigEmulatorTool* trigEmulator;

  public:
    bool doTrigEmulation = false;
    float GetTrigEmulationWeight();

    //
    // For signal injection study
    //
    bool isDataMCMix = false;
    bool mixedEventIsData = false;
    bool passMixedEvent = false;
    bool doReweight = false;
    bool doDvTReweight = false;
    bool doTTbarPtReweight = false;
    float ttbarWeight = 1.0;

    // For hemisphere mixing MC
    bool is3bMixed = false;


    //
    //  Ht Turnon study
    //
    bool doHtTurnOnStudy = true;

    // DEPRICATED. Need to update trigger study package to use maps
    bool HLT_HT330_4j_75_60_45_40 = false;
    bool HLT_HT330_4j_75_60_45_40_3b = false;
    bool L1_HTT320er_QuadJet_70_55_40_40_er2p4 = false;
    bool L1_HTT360er = false;
    bool L1_ETT2000 = false;

    float jetPtMin = 40;
    const float jetEtaMax= 2.4;
    const int puIdMin = 0b110;//7=tight, 6=medium, 4=loose working point
    const bool doJetCleaning=false;
     
    nTupleAnalysis::jetData* treeJets;
    std::vector<jetPtr> allJets;//all jets in nTuple
    std::vector<jetPtr> selJetsLoosePt;//jets passing loose pt/eta requirements
    std::vector<jetPtr> tagJetsLoosePt;//tagged jets passing loose pt/eta requirements
    std::vector<jetPtr> selJets;//jets passing pt/eta requirements
    std::vector<jetPtr> looseTagJets;//jets passing pt/eta and loose bTagging requirements
    std::vector<jetPtr> tagJets;//jets passing pt/eta and bTagging requirements
    std::vector<jetPtr> antiTag;//jets passing pt/eta and failing bTagging requirements
    std::vector<jetPtr> canJets;//jets used in Z/H boson candidates
    std::vector<jetPtr> topQuarkBJets;//jets considered as candidate b-quarks from top decay
    std::vector<jetPtr> topQuarkWJets;//jets considered as candidate udsc-quarks from top-W decay
    std::vector<jetPtr> othJets;//other selected jets
    std::vector<trigPtr> allTrigJets;//all jets in nTuple
    std::vector<trigPtr> selTrigJets;//sel jets in nTuple
    float ht, ht30, L1ht, L1ht30, HLTht, HLTht30, HLTht30Calo, HLTht30CaloAll, HLTht30Calo2p6;
    std::vector<jetPtr> allNotCanJets;//other jets pt>20
 
    uint nSelJets;
    uint nLooseTagJets;
    uint nTagJets;
    uint nAntiTag;
    uint nPSTJets;
    uint nOthJets;
    uint nAllNotCanJets;
    bool threeTag;
    bool fourTag;

    float st;
    TLorentzVector p4j;//combined 4-vector of the candidate jet system
    float m4j;
    float m123; float m013; float m023; float m012;
    float s4j;
    float canJet0_pt ; float canJet1_pt ; float canJet2_pt ; float canJet3_pt ;
    float canJet0_eta; float canJet1_eta; float canJet2_eta; float canJet3_eta;
    float canJet0_phi; float canJet1_phi; float canJet2_phi; float canJet3_phi;
    float canJet0_m  ; float canJet1_m  ; float canJet2_m  ; float canJet3_m  ;
    float aveAbsEta; float aveAbsEtaOth; float stNotCan;
    float dRjjClose;
    float dRjjOther;
    float dR0123; float dR0213; float dR0312;
    float othJet_pt[40]; float othJet_eta[40]; float othJet_phi[40]; float othJet_m[40];
    float notCanJet_pt[40]; float notCanJet_eta[40]; float notCanJet_phi[40]; float notCanJet_m[40];
    
    bool appliedMDRs;
    bool HHSB; bool HHCR; bool HHSR;
    bool ZHSB; bool ZHCR; bool ZHSR;
    bool ZZSB; bool ZZCR; bool ZZSR;
    bool SB; bool CR; bool SR;
    float leadStM; float sublStM;

    nTupleAnalysis::muonData* treeMuons;
    std::vector<muonPtr> allMuons;
    std::vector<muonPtr> muons_isoMed25;
    std::vector<muonPtr> muons_isoMed40;

    nTupleAnalysis::elecData* treeElecs;
    std::vector<elecPtr> allElecs;
    std::vector<elecPtr> elecs_isoMed25;
    std::vector<elecPtr> elecs_isoMed40;

    uint nIsoMuons;
    uint nIsoElecs;

    std::vector< std::shared_ptr<nTupleAnalysis::dijet> > dijets;
    std::shared_ptr<nTupleAnalysis::dijet> close;
    std::shared_ptr<nTupleAnalysis::dijet> other;
    std::vector< std::shared_ptr<eventView> > views;
    std::vector< std::shared_ptr<eventView> > views_passMDRs;
    std::shared_ptr<eventView> view_selected;
    std::shared_ptr<eventView> view_dR_min;
    std::shared_ptr<eventView> view_max_FvT_q_score;
    std::shared_ptr<eventView> view_max_SvB_q_score;

    bool passDijetMass;
    bool passMDRs;
    bool passXWt;
    //bool passDEtaBB;
    Int_t d01TruthMatch = 0;
    Int_t d23TruthMatch = 0;
    Int_t d02TruthMatch = 0;
    Int_t d13TruthMatch = 0;
    Int_t d03TruthMatch = 0;
    Int_t d12TruthMatch = 0;
    bool truthMatch = false;
    bool selectedViewTruthMatch = false;


    nTupleAnalysis::trigData* treeTrig = NULL;

    // Constructors and member functions
    eventData(TChain* t, bool mc, std::string y, bool d, bool _fastSkim = false, bool _doTrigEmulation = false, bool _isDataMCMix = false, bool _doReweight = false, std::string bjetSF = "", std::string btagVariations = "central",
	      std::string JECSyst = "", bool _looseSkim = false, bool is3bMixed = false, std::string FvTName="FvT", std::string reweight4bName="MixedToUnmixed", std::string reweightDvTName="weight_DvT3_3b_pt3", bool doWeightStudy = false); 
    void setTagger(std::string, float);
    void update(long int);
    void buildEvent();
    void resetEvent();

    // 
    //  Used to make new events with hemisphere mixing
    //
    int makeNewEvent(std::vector<nTupleAnalysis::jetPtr> new_allJets);

    //
    //  For signal Injection studies
    // 
    bool pass4bEmulation(unsigned int offset, bool passAll = false);
    void setPSJetsAsTagJets();
    void setLooseAndPSJetsAsTagJets(bool debug = false);
    bool passPSDataFilter(bool invertW = false);

    //jet combinatorics
    bool useJetCombinatoricModel = false;
    bool useLoadedJCM            = false;
    void loadJetCombinatoricModel(std::string jcmName);
    float inputPSTagWeight = -1;

    float pseudoTagProb = -1;
    float pairEnhancement = 0.0;
    float pairEnhancementDecay = 1.0;
    float threeTightTagFraction = 1.0;
    // float pseudoTagProb_lowSt = -1;
    // float pairEnhancement_lowSt = 0.0;
    // float pairEnhancementDecay_lowSt = 1.0;
    // float pseudoTagProb_midSt = -1;
    // float pairEnhancement_midSt = 0.0;
    // float pairEnhancementDecay_midSt = 1.0;
    // float pseudoTagProb_highSt = -1;
    // float pairEnhancement_highSt = 0.0;
    // float pairEnhancementDecay_highSt = 1.0;
    Float_t   pseudoTagWeight = 1;
    uint nPseudoTags = 0;
    TRandom3* random;
    void computePseudoTagWeight();
    void applyInputPseudoTagWeight();


    //jet combinatoric Lists
    std::vector<std::string> jcmNames;
    std::map<std::string, float> pseudoTagProbMap;
    std::map<std::string, float> pairEnhancementMap;
    std::map<std::string, float> pairEnhancementDecayMap;
    std::map<std::string, float> threeTightTagFractionMap;
    std::map<std::string, float>  pseudoTagWeightMap;
    std::map<std::string, float>  mcPseudoTagWeightMap;
    void computePseudoTagWeight(std::string jcmName);


    #if SLC6 == 0
    multiClassifierONNX* SvB_ONNX = NULL;
    void load_SvB_ONNX(std::string);
    void run_SvB_ONNX();
    #endif

    void chooseCanJets();
    void buildViews();
    void applyMDRs();
    
    std::shared_ptr<nTupleAnalysis::trijet> t;
    std::shared_ptr<nTupleAnalysis::trijet> t0;
    std::shared_ptr<nTupleAnalysis::trijet> t1;
    //std::shared_ptr<nTupleAnalysis::trijet> t2;
    float xWt0; float xWt1; float xWt; //float xWt2;
    float xWbW0; float xWbW1; float xWbW; //float xWbW2;
    float xW; float xt; float xbW;
    float dRbW;

    void buildTops();
    void dump();
    ~eventData(); 

    float ttbarSF(float pt);

    // 
    //  for the Unsupervised Analysis
    //
    float getSRvsSB_Pull (float m4j, float leadSt, float sublSt);
    TH2F* getSRvsSB_PullHist(float m4j);
    TFile* SRvsSB_pullFile4b = NULL;
    TFile* SRvsSB_pullFile3b = NULL;
    int   m4jBinIndex = -1;

    //float getSRvsSB_Ratio (float m4j, float leadSt, float sublSt);
    //TH2F* getSRvsSB_RatioHist(float m4j);
    //TFile* SRvsSB_ratioFile4b = NULL;
    //TFile* SRvsSB_ratioFile3b = NULL;


    std::string currentFile = "";


  };

}
#endif // eventData_H
