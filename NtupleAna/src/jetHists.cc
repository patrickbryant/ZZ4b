#include "ZZ4b/NtupleAna/interface/jetHists.h"

using namespace NtupleAna;

jetHists::jetHists(std::string name, fwlite::TFileService& fs, std::string title) {

    dir = fs.mkdir(name);
    v = new vecHists(name, dir, title);

    deepB = dir.make<TH1F>("deepB", (name+"/deepB; "+title+" Deep B; Entries").c_str(), 100,0,1);
    CSVv2 = dir.make<TH1F>("CSVv2", (name+"/CSVv2; "+title+" CSV v2; Entries").c_str(), 100,0,1);

} 

void jetHists::Fill(std::shared_ptr<jet> &jet, float weight){

  v->Fill(jet->p, weight);

  deepB->Fill(jet->deepB, weight);
  CSVv2->Fill(jet->CSVv2, weight);

  return;
}

jetHists::~jetHists(){} 
