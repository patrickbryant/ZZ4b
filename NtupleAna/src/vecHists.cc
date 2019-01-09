#include "ZZ4b/NtupleAna/interface/vecHists.h"

using namespace NtupleAna;

vecHists::vecHists(std::string name, TFileDirectory& dir) {

    pt_s = dir.make<TH1F>("pt_s", (name+"_pt_s; p_T [GeV]; Entries").c_str(),  100,0, 100);
    pt_m = dir.make<TH1F>("pt_m", (name+"_pt_m; p_T [GeV]; Entries").c_str(),  100,0, 500);
    pt_l = dir.make<TH1F>("pt_l", (name+"_pt_l; p_T [GeV]; Entries").c_str(),  100,0,1000);

    eta = dir.make<TH1F>("eta", (name+"_eta; #eta; Entries").c_str(), 100, -5, 5);
    phi = dir.make<TH1F>("phi", (name+"_phi; #phi; Entries").c_str(), 64, -3.2, 3.2);

} 

void vecHists::Fill(TLorentzVector* p, float weight){

  pt_s->Fill(p->Pt(), weight);
  pt_m->Fill(p->Pt(), weight);
  pt_l->Fill(p->Pt(), weight);

  eta->Fill(p->Eta(), weight);
  phi->Fill(p->Phi(), weight);

  return;
}

vecHists::~vecHists(){} 
