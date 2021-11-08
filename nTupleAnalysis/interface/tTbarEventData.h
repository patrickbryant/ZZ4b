// -*- C++ -*-

#if !defined(tTbarEventData_H)
#define tTbarEventData_H

#include <iostream>
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
#include "nTupleAnalysis/baseClasses/interface/MeTData.h"
#include "ZZ4b/nTupleAnalysis/interface/eventView.h"
#include "TriggerEmulator/nTupleAnalysis/interface/TrigEmulatorTool.h"

// for jet pseudoTag calculations
#include <TRandom3.h>
#include <numeric> 
#include <boost/math/special_functions/binomial.hpp> 

namespace nTupleAnalysis {

  class tTbarEventData {

  public:
    // Member variables
    TChain* tree;
    bool isMC;
    float year;
    bool debug;
    bool printCurrentFile = true;

    UInt_t    run       =  0;
    UInt_t    lumiBlock =  0;
    ULong64_t event     =  0;
    Int_t     nPVs = 0;
    Int_t     nPVsGood = 0;
    Float_t   reweight = 1.0;

    Float_t   genWeight =  1;
    Float_t   weight    =  1;
    Float_t   weightNoTrigger    =  1;
    Float_t   trigWeight =  1;
    Float_t   mcWeight  =  1;
    Float_t   bTagSF = 1;
    int       nTrueBJets = 0;

    // used for hemisphere mixing
    Float_t   inputBTagSF = 0;

    nTupleAnalysis::truthData* truth = NULL;

    //Predefine btag sorting functions
    float       bTag    = 0.8484;//medium WP for CSVv2 https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80XReReco
    std::string bTagger = "CSVv2";
    //bool (*sortTag)(std::shared_ptr<nTupleAnalysis::jet>&, std::shared_ptr<nTupleAnalysis::jet>&);

    //triggers
    bool passHLT             = false;
    bool passHLT_1L          = false;
    bool passHLT_2L          = false;

    bool HLT_4j_3b          = false;
    bool HLT_2j_2j_3b          = false;
    bool HLT_2b          = false;

    bool HLT_Mu23_Ele12 = false;
    bool HLT_IsoMu24    = false;
    bool HLT_IsoMu27    = false;
    bool HLT_Mu12_Ele23 = false;

    bool L1_Mu20_EG10 = false;
    bool L1_SingleMu22 = false;
    bool L1_SingleMu25 = false;
    bool L1_Mu5_EG23         = false;
    bool L1_Mu5_LooseIsoEG20 = false;
    bool L1_Mu7_EG23         = false;
    bool L1_Mu7_LooseIsoEG20 = false;
    bool L1_Mu7_LooseIsoEG23 = false;
    bool L1_Mu23_EG10        = false;
    bool L1_Mu20_EG17        = false;
    bool L1_SingleMu22er2p1  = false;
    bool L1_Mu20_EG10er2p5   = false;

    bool L1_QuadJetC50      = false;  
    bool L1_DoubleJetC100   = false;
    bool L1_SingleJet170    = false;
    bool L1_HTT300          = false;
    bool L1_SingleJet200    = false;

    bool L1_QuadJet60er2p7                      = false;           
    bool L1_HTT380er                            = false;
    bool L1_HTT280er_QuadJet_70_55_40_35_er2p5  = false;
    bool L1_QuadJet60er3p0                      = false;
    bool L1_DoubleJet100er2p3_dEta_Max1p6       = false;

    bool L1_ETT2000                             = false;
    bool L1_HTT360er                            = false;
    bool L1_HTT320er_QuadJet_70_55_40_40_er2p4  = false;
    bool L1_DoubleJet112er2p3_dEta_Max1p6       = false;



  public:

    float jetPtMin = 25; //40;
    const float jetEtaMax= 2.4;
    const int puIdMin = 0b110;//7=tight, 6=medium, 4=loose working point
    const bool doJetCleaning=false;
     
    nTupleAnalysis::jetData* treeJets;
    std::vector<jetPtr> allJets;//all jets in nTuple
    std::vector<jetPtr> selJets;//jets passing pt/eta requirements
    std::vector<jetPtr> looseTagJets;//jets passing pt/eta and loose bTagging requirements
    std::vector<jetPtr> tagJets;//jets passing pt/eta and bTagging requirements
    std::vector<jetPtr> antiTag;//jets passing pt/eta and failing bTagging requirements
    std::vector<jetPtr> topQuarkBJets;//jets considered as candidate b-quarks from top decay
    std::vector<jetPtr> topQuarkWJets;//jets considered as candidate udsc-quarks from top-W decay
    float ht, ht30, L1ht, L1ht30, HLTht, HLTht30, HLTht30Calo, HLTht30CaloAll, HLTht30Calo2p6;

 
    uint nSelJets;
    uint nTagJets;
    uint nLooseTagJets;
    uint nAntiTag;
    uint nOthJets;
    bool twoTag;

    float st;

    bool doTTbarPtReweight = false;
    float ttbarWeight = 1.0;
    
    nTupleAnalysis::muonData* treeMuons;
    std::vector<muonPtr> allMuons;
    std::vector<muonPtr> muons_iso;
    std::vector<muonPtr> muons_isoHighPt;

    nTupleAnalysis::elecData* treeElecs;
    std::vector<elecPtr> allElecs;
    std::vector<elecPtr> elecs_iso;
    std::vector<elecPtr> elecs_isoHighPt;

    uint nIsoMuons;
    uint nIsoElecs;
    uint nIsoLeps;

    nTupleAnalysis::MeTData*  treeCaloMET;
    nTupleAnalysis::MeTData*  treeChsMET   ;
    nTupleAnalysis::MeTData*  treeMET      ;
    nTupleAnalysis::MeTData*  treePuppiMET ;
    nTupleAnalysis::MeTData*  treeTrkMET   ;

    std::shared_ptr<nTupleAnalysis::dijet> w;
    std::shared_ptr<nTupleAnalysis::trijet> top;

    nTupleAnalysis::trigData* treeTrig = NULL;

    // Constructors and member functions
    tTbarEventData(TChain* t, bool mc, std::string y, bool d, bool _doTrigEmulation = false, std::string bjetSF = "", std::string btagVariations = "central", std::string JECSyst = ""); 
		   
    void setTagger(std::string, float);
    void update(long int);
    void buildEvent();
    void resetEvent();

    void buildTops();
    void dump();
    ~tTbarEventData(); 

    float ttbarSF(float pt);

    std::string currentFile = "";

    //
    //  trigger Emulation
    //

  public:

    TriggerEmulator::TrigEmulatorTool* trigEmulator;
    bool doTrigEmulation = false;
    float GetTrigEmulationWeight();



  };

}
#endif // tTbarEventData_H
