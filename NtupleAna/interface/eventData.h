// -*- C++ -*-

#if !defined(eventData_H)
#define eventData_H

#include <iostream>
#include <TChain.h>
#include <TFile.h>
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/initBranch.h"
#include "ZZ4b/NtupleAna/interface/truthData.h"
#include "ZZ4b/NtupleAna/interface/jetData.h"
#include "ZZ4b/NtupleAna/interface/muonData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"

// for jet pseudoTag calculations
#include <TRandom3.h>
#include <numeric> 
#include <boost/math/special_functions/binomial.hpp> 

namespace NtupleAna {

  class eventData {

  public:
    // Member variables
    TChain* tree;
    bool isMC;
    std::string year;
    bool debug;
    UInt_t    run       =  0;
    UInt_t    lumiBlock =  0;
    ULong64_t event     =  0;
    Float_t   nTagClassifier = -99;
    Float_t   ZHvsBackgroundClassifier = -99;
    Float_t   genWeight =  1;
    Float_t   weight    =  1;

    truthData* truth = NULL;

    //Predefine btag sorting functions
    float       bTag    = 0.8484;//medium WP for CSVv2 https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80XReReco
    std::string bTagger = "CSVv2";
    bool (*sortTag)(std::shared_ptr<jet>&, std::shared_ptr<jet>&);
    
    //triggers
    bool passHLT             = false;
    //2016
    bool HLT_4j45_3b087      = false;
    bool HLT_2j90_2j30_3b087 = false;
    //2018
    bool HLT_HT330_4j_75_60_45_40_3b = false;
    bool HLT_4j_103_88_75_15_2b_VBF1 = false;
    bool HLT_4j_103_88_75_15_1b_VBF2 = false;
    bool HLT_2j116_dEta1p6_2b        = false;
    bool HLT_J330_m30_2b             = false;
    bool HLT_j500                    = false;
    bool HLT_2j300ave                = false;

    jetData* treeJets;
    std::vector< std::shared_ptr<jet> > allJets;//all jets in nTuple
    std::vector< std::shared_ptr<jet> > selJets;//jets passing pt/eta requirements
    std::vector< std::shared_ptr<jet> > tagJets;//jets passing pt/eta and bTagging requirements
    std::vector< std::shared_ptr<jet> > antiTag;//jets passing pt/eta and failing bTagging requirements
    std::vector< std::shared_ptr<jet> > canJets;//jets used in Z/H boson candidates

    unsigned int nSelJets;
    unsigned int nTagJets;
    bool threeTag;
    bool fourTag;

    TLorentzVector p4j;//combined 4-vector of the candidate jet system
    float m4j;
    float canJet0_pt ; float canJet1_pt ; float canJet2_pt ; float canJet3_pt ;
    float canJet0_eta; float canJet1_eta; float canJet2_eta; float canJet3_eta;
    float canJet0_phi; float canJet1_phi; float canJet2_phi; float canJet3_phi;
    float canJet0_e  ; float canJet1_e  ; float canJet2_e  ; float canJet3_e  ;
    float aveAbsEta;
    float dRjjClose;
    float dRjjOther;
    
    bool ZHSB; bool ZHCR; bool ZHSR;
    float leadStM; float sublStM;

    muonData* treeMuons;
    std::vector< std::shared_ptr<muon> > allMuons;
    std::vector< std::shared_ptr<muon> > isoMuons;

    std::vector< std::shared_ptr<dijet> > dijets;
    std::shared_ptr<dijet> close;
    std::shared_ptr<dijet> other;
    std::vector< std::unique_ptr<eventView> > views;
    bool passMDRs;
    bool passDEtaBB;

    // Constructors and member functions
    eventData(TChain*, bool, std::string, bool); 
    void setTagger(std::string, float);
    void update(int);

    //jet combinatorics
    float pseudoTagProb = -1; // = 0.147442250963;
    float fourJetScale  = 0.640141122153;
    float moreJetScale  = 0.542014471066;
    Float_t   pseudoTagWeight = 1;
    unsigned int nPseudoTags = 0;
    TRandom3* random;
    void computePseudoTagWeight();
    float nTagClassifierWeight = 1;

    void chooseCanJets();
    void buildViews();
    void applyMDRs();
    float xWt0; float xWt1;
    void buildTops();
    void dump();
    ~eventData(); 

  };

}
#endif // eventData_H
