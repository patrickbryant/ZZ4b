#include "ZZ4b/NtupleAna/interface/vecHists.h"

using namespace NtupleAna;

vecHists::vecHists(std::string name, TFileDirectory& dir, std::string title) {

    pt_s = dir.make<TH1F>("pt_s", (name+"_pt_s; "+title+" p_T [GeV]; Entries").c_str(),  100,0, 100);
    pt_m = dir.make<TH1F>("pt_m", (name+"_pt_m; "+title+" p_T [GeV]; Entries").c_str(),  100,0, 500);
    pt_l = dir.make<TH1F>("pt_l", (name+"_pt_l; "+title+" p_T [GeV]; Entries").c_str(),  100,0,1000);

    eta = dir.make<TH1F>("eta", (name+"_eta; "+title+" #eta; Entries").c_str(), 100, -5, 5);
    phi = dir.make<TH1F>("phi", (name+"_phi; "+title+" #phi; Entries").c_str(), 64, -3.2, 3.2);

    m = dir.make<TH1F>("m", (name+"_m; "+title+" Mass [GeV]; Entries").c_str(),  100,0, 500);
    e = dir.make<TH1F>("e", (name+"_e; "+title+" E [GeV]; Entries").c_str(),  100,0, 500);

} 

void vecHists::Fill(TLorentzVector& p, float weight){

  pt_s->Fill(p.Pt(), weight);
  pt_m->Fill(p.Pt(), weight);
  pt_l->Fill(p.Pt(), weight);

  eta->Fill(p.Eta(), weight);
  phi->Fill(p.Phi(), weight);

  m->Fill(p.M(), weight);
  e->Fill(p.E(), weight);

  return;
}

vecHists::~vecHists(){} 
