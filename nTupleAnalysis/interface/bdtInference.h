// -*- C++ -*-
#if !defined(bdtInference_H)
#define bdtInference_H

#include <string>
#include <vector>
#include <TLorentzVector.h>

#include "TMVA/Reader.h"

#include "ZZ4b/nTupleAnalysis/interface/eventView.h"

namespace nTupleAnalysis{

  class eventData;
  
  class bdtInference{
  public:

    bool debug;
    bool benchmark;

    std::string channel;

    std::unique_ptr<TMVA::Reader> model;
    std::vector<std::string> methods;

    Float_t V_pt = 0;
    Float_t H1_e = 0;
    Float_t H1_pt = 0;
    Float_t H1_eta = 0;
    Float_t H2_e = 0;
    Float_t H2_eta = 0;
    Float_t HH_e = 0;
    Float_t HH_m = 0;
    Float_t HH_eta = 0;
    Float_t dEta_H1_H2 = 0;
    Float_t dPhi_H1_H2 = 0;
    Float_t dPhi_V_H2 = 0;
    Float_t dR_H1_H2 = 0;
    Float_t pt_ratio = 0;

    bdtInference(std::string weightFile, std::string methodNames, bool debug = false, bool benchmark = false);
    
    bool setVariables(const TLorentzVector &H1_p, const TLorentzVector &H2_p, const TLorentzVector &V_p);
    std::map<std::string, Float_t> getBDTScore();
    std::vector<std::map<std::string, Float_t>> getBDTScore(eventData *event, bool mainViewOnly = false, bool useCorrectedMomentum = false);
  };

}

#endif // bdtInference_H