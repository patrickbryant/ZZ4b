#include "ZZ4b/NtupleAna/interface/jetHists.h"

using namespace NtupleAna;

jetHists::jetHists(std::string name, fwlite::TFileService& fs) {

    dir = fs.mkdir(name);
    v = new vecHists(name, dir);

    deepCSV = dir.make<TH1F>("deepCSV", (name+"/deepCSV; Deep CSV; Entries").c_str(), 100,0,1);

} 

void jetHists::Fill(jet& jet, float weight){

  v->Fill(jet.p, weight);

  deepCSV->Fill(jet.deepCSV, weight);

  return;
}

jetHists::~jetHists(){} 
