#include "ZZ4b/NtupleAna/interface/jetHists.h"

using namespace NtupleAna;

jetHists::jetHists(std::string name, fwlite::TFileService& fs, std::string title) {

    dir = fs.mkdir(name);
    v = new vecHists(name, dir, title);

    deepCSV = dir.make<TH1F>("deepCSV", (name+"/deepCSV; "+title+" Deep CSV; Entries").c_str(), 100,0,1);

} 

void jetHists::Fill(jet& jet, float weight){

  v->Fill(jet.p, weight);

  deepCSV->Fill(jet.deepCSV, weight);

  return;
}

jetHists::~jetHists(){} 
