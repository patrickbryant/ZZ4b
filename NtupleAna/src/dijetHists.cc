#include "ZZ4b/NtupleAna/interface/dijetHists.h"

using namespace NtupleAna;

dijetHists::dijetHists(std::string name, fwlite::TFileService& fs, std::string title) {

    dir = fs.mkdir(name);
    v = new vecHists(name, dir, title);

    dR = dir.make<TH1F>("dR", (name+"/dR; "+title+" #DeltaR(j,j); Entries").c_str(), 50,0,5);

} 

void dijetHists::Fill(std::shared_ptr<dijet> &dijet, float weight){

  v->Fill(dijet->p, weight);

  dR->Fill(dijet->dR, weight);

  return;
}

dijetHists::~dijetHists(){} 
