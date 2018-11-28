//#include "TChain.h"
#include "ZZ4b/NtupleAna/interface/cutflowHists.h"

using namespace NtupleAna;

cutflowHists::cutflowHists(std::string name, fwlite::TFileService& fs) {

    dir = fs.mkdir(name);
    unitWeight = dir.make<TH1F>("unitWeight", (name+"/unitWeight; ;Entries").c_str(),  1,1,2);
    unitWeight->SetCanExtend(1);
    unitWeight->GetXaxis()->FindBin("all");

    weighted = dir.make<TH1F>("weighted", (name+"/weighted; ;Entries").c_str(),  1,1,2);
    weighted->SetCanExtend(1);
    weighted->GetXaxis()->FindBin("all");

} 

void cutflowHists::Fill(std::string cut, float weight){

  unitWeight->Fill(cut.c_str(), 1);
  weighted  ->Fill(cut.c_str(), weight);

  return;
}

cutflowHists::~cutflowHists() {} 

