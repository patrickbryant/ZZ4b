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
    Float_t H1_m = 0;
    Float_t H1_e = 0;
    Float_t H1_pT = 0;
    Float_t H1_eta = 0;
    Float_t H2_m = 0;
    Float_t H2_e = 0;
    Float_t H2_pT = 0;
    Float_t H2_eta = 0;
    Float_t HH_e = 0;
    Float_t HH_m = 0;
    Float_t HH_eta = 0;
    Float_t HH_deta = 0;
    Float_t HH_dphi = 0;
    Float_t V_H2_dPhi = 0;
    Float_t HH_dR = 0;
    Float_t H2H1_pt_ratio = 0;

    bdtInference(std::string weightFile, std::string methodNames, bool debug = false, bool benchmark = false);
    
    bool setVariables(const TLorentzVector &H1_p, const TLorentzVector &H2_p, const TLorentzVector &V_p);
    std::map<std::string, Float_t> getBDTScore();
    std::map<std::string, Float_t> getBDTScore(eventData* event, std::shared_ptr<eventView> view);
  };

}

#endif // bdtInference_H