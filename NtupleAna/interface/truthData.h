// -*- C++ -*-

#if !defined(truthData_H)
#define truthData_H

#include <TChain.h>
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/initBranch.h"
#include "ZZ4b/NtupleAna/interface/truthParticle.h"

// for jet pseudoTag calculations
#include <TRandom3.h>
#include <boost/math/special_functions/binomial.hpp> 

namespace NtupleAna {

  class truthData {

  public:
    // Member variables
    TChain* tree;
    bool isMC;
    std::string year;
    bool debug;

    truthParticle* truthParticles;
    std::vector< std::shared_ptr<particle> > bQuarksZ;//all bquarks from Z decays
    std::vector< std::shared_ptr<particle> > bQuarksH;//all bquarks from H decays
    float m4b;

    // Constructors and member functions
    truthData(TChain*, bool); 
    void update();
    void dump();
    ~truthData(); 

  };

}
#endif // truthData_H
