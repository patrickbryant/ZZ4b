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
    model->AddVariable( "H1_e", &H1_e );
    model->AddVariable( "H1_pt", &H1_pt );
    model->AddVariable( "H1_eta", &H1_eta );
    model->AddVariable( "H2_e", &H2_e );
    model->AddVariable( "H2_eta", &H2_eta );
    model->AddVariable( "HH_e", &HH_e );
    model->AddVariable( "HH_m", &HH_m );
    model->AddVariable( "HH_eta", &HH_eta );
    model->AddVariable( "dEta_H1_H2", &dEta_H1_H2 );
    model->AddVariable( "dPhi_H1_H2", &dPhi_H1_H2 );
    model->AddVariable( "dPhi_V_H2", &dPhi_V_H2 );
    model->AddVariable( "dR_H1_H2", &dR_H1_H2 );
    model->AddVariable( "pt_ratio", &pt_ratio );
    
    for(const auto &method : methods){
      auto weightFilePath = utils::fillString(weightFile, {{"method", method}});
      model->BookMVA(method + " method", weightFilePath);
      if(debug) cout << method << " weight loaded from " << weightFilePath << endl;
    }
  }

  bool bdtInference::setVariables(const TLorentzVector &H1_p, const TLorentzVector &H2_p, const TLorentzVector &V_p){
    V_pt = V_p.Pt();
    H1_e = H1_p.E();
    H1_pt = H1_p.Pt();
    H1_eta = H1_p.Eta();
    H2_e = H2_p.E();
    H2_eta = H2_p.Eta();
    auto HH_p = H1_p + H2_p;
    HH_e = HH_p.E();
    HH_m = HH_p.M();
    HH_eta = HH_p.Eta();
    dEta_H1_H2 = std::abs(H1_p.Eta() - H2_p.Eta());
    dPhi_H1_H2 = H1_p.DeltaPhi(H2_p);
    dR_H1_H2 = H1_p.DeltaR(H2_p);
    dPhi_V_H2 = V_p.DeltaPhi(H2_p);
    pt_ratio = H2_p.Pt()/H1_p.Pt();
    return true;
  }

  std::map<std::string, Float_t> bdtInference::getBDTScore(){
    std::map<std::string, Float_t> score;
    for(const auto &method : methods){
      score[method] = model->EvaluateMVA(method + " method");
    }
    return score;
  }

  std::vector<std::map<std::string, Float_t>> bdtInference::getBDTScore(eventData *event, bool mainViewOnly, bool useCorrectedMomentum){
    std::vector<std::map<std::string, Float_t>> scores;
    size_t nViews = mainViewOnly? 1 : event->views.size();
    auto V_p = useCorrectedMomentum ? (event->canVDijets[0]->pW + event->canVDijets[0]->pZ) * 0.5 : event->canVDijets[0]->p;
    for (size_t i = 0; i < nViews; i++){
      auto view = event->views[i];
      auto H1_p = useCorrectedMomentum ? view->lead->pH : view->lead->p;
      auto H2_p = useCorrectedMomentum ? view->subl->pH : view->subl->p;
      setVariables(H1_p, H2_p, V_p);
      scores.push_back(getBDTScore());
    }
    return scores;
  }
}