#include "ZZ4b/NtupleAna/interface/jetHists.h"

using namespace NtupleAna;

jetHists::jetHists(std::string name, fwlite::TFileService& fs, std::string title) {

    dir = fs.mkdir(name);
    v = new vecHists(name, dir, title);

    cleanmask = dir.make<TH1F>("cleanmask", (name+"/cleanmask; "+title+" Clean Mask; Entries").c_str(), 2,-0.5,1.5);

    deepB     = dir.make<TH1F>("deepB",     (name+"/deepB; "    +title+" Deep B; Entries").c_str(), 100,0,1);
    CSVv2     = dir.make<TH1F>("CSVv2",     (name+"/CSVv2; "    +title+" CSV v2; Entries").c_str(), 100,0,1);
    deepFlavB = dir.make<TH1F>("deepFlavB", (name+"/deepFlavB; "+title+" Deep (Jet) Flavour B; Entries").c_str(), 100,0,1);

} 

void jetHists::Fill(std::shared_ptr<jet> &jet, float weight){

  v->Fill(jet->p, weight);

  cleanmask->Fill(jet->cleanmask, weight);

  deepB    ->Fill(jet->deepB, weight);
  CSVv2    ->Fill(jet->CSVv2, weight);
  deepFlavB->Fill(jet->deepFlavB, weight);

  return;
}

jetHists::~jetHists(){} 
