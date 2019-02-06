#include "ZZ4b/NtupleAna/interface/vecHists.h"

using namespace NtupleAna;

vecHists::vecHists(std::string name, fwlite::TFileService& fs, std::string title) {
  TFileDirectory dir = fs.mkdir(name);
  this->makeHists(name, dir, title);
}

vecHists::vecHists(std::string name, TFileDirectory& dir, std::string title) {
  this->makeHists(name, dir, title);
}

void vecHists::makeHists(std::string name, TFileDirectory& dir, std::string title) {

    pt_s = dir.make<TH1F>("pt_s", (name+"/pt_s; "+title+" p_T [GeV]; Entries").c_str(),  100,0, 100);
    pt_m = dir.make<TH1F>("pt_m", (name+"/pt_m; "+title+" p_T [GeV]; Entries").c_str(),  100,0, 500);
    pt_l = dir.make<TH1F>("pt_l", (name+"/pt_l; "+title+" p_T [GeV]; Entries").c_str(),  100,0,1000);

    eta = dir.make<TH1F>("eta", (name+"/eta; "+title+" #eta; Entries").c_str(), 100, -5, 5);
    phi = dir.make<TH1F>("phi", (name+"/phi; "+title+" #phi; Entries").c_str(), 64, -3.2, 3.2);

    m   = dir.make<TH1F>("m",   (name+"/m;   "+title+" Mass [GeV]; Entries").c_str(),  100,0, 500);
    m_l = dir.make<TH1F>("m_l", (name+"/m_l; "+title+" Mass [GeV]; Entries").c_str(),  130,100, 1400);
    e = dir.make<TH1F>("e", (name+"/e; "+title+" E [GeV]; Entries").c_str(),  100,0, 500);

    return;
} 

void vecHists::Fill(TLorentzVector& p, float weight){

  pt_s->Fill(p.Pt(), weight);
  pt_m->Fill(p.Pt(), weight);
  pt_l->Fill(p.Pt(), weight);

  eta->Fill(p.Eta(), weight);
  phi->Fill(p.Phi(), weight);

  m  ->Fill(p.M(), weight);
  m_l->Fill(p.M(), weight);
  e->Fill(p.E(), weight);

  return;
}

vecHists::~vecHists(){} 
