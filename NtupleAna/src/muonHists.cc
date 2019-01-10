#include "ZZ4b/NtupleAna/interface/muonHists.h"

using namespace NtupleAna;

muonHists::muonHists(std::string name, fwlite::TFileService& fs, std::string title) {

    dir = fs.mkdir(name);
    v = new vecHists(name, dir);

    quality   = dir.make<TH1F>("quality",   (name+"/quality;   "+title+" Quality;   Entries").c_str(), 3,-0.5,2.5);
    isolation = dir.make<TH1F>("isolation", (name+"/isolation; "+title+" Isolation; Entries").c_str(), 50,0,1);
    dR        = dir.make<TH1F>("dR",        (name+"/dR;        "+title+" Min #DeltaR(muon,jet); Entries").c_str(), 50, 0, 5);

} 

void muonHists::Fill(muon& muon, float weight){

  v->Fill(muon.p, weight);

  quality  ->Fill(muon.quality,   weight);
  isolation->Fill(muon.isolation, weight);
  dR       ->Fill(muon.dR,        weight);

  return;
}

muonHists::~muonHists(){} 
