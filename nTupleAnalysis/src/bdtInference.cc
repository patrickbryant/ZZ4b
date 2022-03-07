#include <iostream>
#include <cmath>

#include "ZZ4b/nTupleAnalysis/interface/bdtInference.h"
#include "ZZ4b/nTupleAnalysis/interface/utils.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

using std::cout;
using std::endl;

namespace nTupleAnalysis{

  bdtInference::bdtInference(std::string weightFile, std::string methodNames, bool debug, bool benchmark)
    :debug(debug), benchmark(benchmark), methods(utils::splitString(methodNames, "."))
  {
    model = std::make_unique<TMVA::Reader>(debug? "!Color" : "Silent");

    model->AddVariable( "V_pt", &V_pt );
    model->AddVariable( "VHH_H1_m", &H1_m );
    model->AddVariable( "VHH_H1_e", &H1_e );
    model->AddVariable( "VHH_H1_pT", &H1_pT );
    model->AddVariable( "VHH_H1_eta", &H1_eta );
    model->AddVariable( "VHH_H2_m", &H2_m );
    model->AddVariable( "VHH_H2_e", &H2_e );
    model->AddVariable( "VHH_H2_pT", &H2_pT );
    model->AddVariable( "VHH_H2_eta", &H2_eta );
    model->AddVariable( "VHH_HH_e", &HH_e );
    model->AddVariable( "V", &HH_m );
    model->AddVariable( "VHH_HH_eta", &HH_eta );
    model->AddVariable( "VHH_HH_deta", &HH_deta );
    model->AddVariable( "VHH_HH_dphi", &HH_dphi );
    model->AddVariable( "VHH_V_H2_dPhi", &V_H2_dPhi );
    model->AddVariable( "VHH_HH_dR", &HH_dR );
    model->AddVariable( "VHH_H2H1_pt_ratio", &H2H1_pt_ratio );
    
    for(const auto &method : methods){
      auto weightFilePath = utils::fillString(weightFile, {{"method", method}});
      model->BookMVA(method + " method", weightFilePath);
      cout << method << " weight loaded from " << weightFilePath << endl;
    }
  }

  bool bdtInference::setVariables(const TLorentzVector &H1_p, const TLorentzVector &H2_p, const TLorentzVector &V_p){
    V_pt = V_p.Pt();
    H1_m = H1_p.M();
    H1_e = H1_p.E();
    H1_pT = H1_p.Pt();
    H1_eta = H1_p.Eta();
    H2_m = H2_p.M();
    H2_e = H2_p.E();
    H2_pT = H2_p.Pt();
    H2_eta = H2_p.Eta();
    auto HH_p = H1_p + H2_p;
    HH_e = HH_p.E();
    HH_m = HH_p.M();
    HH_eta = HH_p.Eta();
    HH_deta = std::abs(H1_p.Eta() - H2_p.Eta());
    HH_dphi = std::abs(H1_p.DeltaPhi(H2_p));
    V_H2_dPhi = std::abs(V_p.DeltaPhi(H2_p));
    HH_dR = H1_p.DeltaR(H2_p);
    H2H1_pt_ratio = H2_p.Pt()/H1_p.Pt();
    return true;
  }

  std::map<std::string, Float_t> bdtInference::getBDTScore(){
    std::map<std::string, Float_t> score;
    for(const auto &method : methods){
      score[method] = model->EvaluateMVA(method + " method");
    }
    return score;
  }

  std::map<std::string, Float_t> bdtInference::getBDTScore(eventData* event, std::shared_ptr<eventView> view){
    auto V_p = event->canVDijets[0]->p;
    auto H1_p = view->lead->p;
    auto H2_p = view->subl->p;
    setVariables(H1_p, H2_p, V_p);
    return getBDTScore();
  }
}