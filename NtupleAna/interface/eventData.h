// -*- C++ -*-

#if !defined(eventData_H)
#define eventData_H

#include "ZZ4b/NtupleAna/interface/jetData.h"

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

    // Constructors and member functions
    eventData(TChain*, bool); 
    void update(int);
    void dump();
    ~eventData(); 

  };

}
#endif // eventData_H
