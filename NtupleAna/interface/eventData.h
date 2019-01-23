// -*- C++ -*-

#if !defined(eventData_H)
#define eventData_H

#include <TChain.h>
#include "ZZ4b/NtupleAna/interface/jetData.h"
#include "ZZ4b/NtupleAna/interface/muonData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"

namespace NtupleAna {

  class eventData {

  public:
    // Member variables
    TChain* tree;
    bool debug;
    UInt_t    run     =  0;
    ULong64_t event   =  0;
    float     weight  =  1;

    jetData* treeJets;
    std::vector<jet> allJets;//all jets in nTuple
    std::vector<jet> selJets;//jets passing pt/eta requirements
    std::vector<jet> tagJets;//jets passing pt/eta and bTagging requirements
    std::vector<jet> canJets;//jets used in Z/H boson candidates

    float m4j;

    muonData* treeMuons;
    std::vector<muon> allMuons;
    std::vector<muon> isoMuons;

    std::vector<eventView> views;

    // Constructors and member functions
    eventData(TChain*, bool); 
    void update(int);
    void chooseCanJets();
    void buildViews();
    void dump();
    ~eventData(); 

  };

}
#endif // eventData_H
