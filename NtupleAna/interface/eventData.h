// -*- C++ -*-

#if !defined(eventData_H)
#define eventData_H

#include <TChain.h>
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/initBranch.h"
#include "ZZ4b/NtupleAna/interface/truthData.h"
#include "ZZ4b/NtupleAna/interface/jetData.h"
#include "ZZ4b/NtupleAna/interface/muonData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"

// for jet pseudoTag calculations
#include <TRandom3.h>
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
    Float_t   genWeight =  1;
    Float_t   weight    =  1;

    truthData* truth;

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
    bool HLT_HT330_4j_75_60_45_40_3b4p5 = false;

    jetData* treeJets;
    std::vector< std::shared_ptr<jet> > allJets;//all jets in nTuple
    std::vector< std::shared_ptr<jet> > selJets;//jets passing pt/eta requirements
    std::vector< std::shared_ptr<jet> > tagJets;//jets passing pt/eta and bTagging requirements
    std::vector< std::shared_ptr<jet> > antiTag;//jets passing pt/eta and failing bTagging requirements
    std::vector< std::shared_ptr<jet> > canJets;//jets used in Z/H boson candidates

    unsigned int nTags;
    bool threeTag;
    bool fourTag;

    TLorentzVector p4j;//combined 4-vector of the candidate jet system

    muonData* treeMuons;
    std::vector< std::shared_ptr<muon> > allMuons;
    std::vector< std::shared_ptr<muon> > isoMuons;

    std::vector< std::shared_ptr<dijet> > dijets;
    std::shared_ptr<dijet> close;
    std::shared_ptr<dijet> other;
    std::vector< std::unique_ptr<eventView> > views;
    bool passMDRs;

    // Constructors and member functions
    eventData(TChain*, bool, std::string, bool); 
    void setTagger(std::string, float);
    void update(int);

    //jet combinatorics
    Float_t   pseudoTagWeight = 0;
    unsigned int nPseudoTags = 0;
    TRandom3* random;
    void computePseudoTagWeight();

    void chooseCanJets();
    void buildViews();
    void applyMDRs();
    void dump();
    ~eventData(); 

  };

}
#endif // eventData_H
