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

  //Make a weighted cutflow as a function of the true m4b, xaxis is m4b, yaxis is cut name. 
  Double_t bins_m4b[] = {100, 110, 121, 133, 146, 160, 176, 193, 212, 233, 256, 281, 309, 339, 372, 409, 449, 493, 542, 596, 655, 720, 792, 871, 958, 1053, 1158, 1273, 1400, 1540, 1694, 2000};
  truthM4b = dir.make<TH2F>("truthM4b", (name+"/truthM4b; Truth m_{4b} [GeV]; ;Entries").c_str(), 31, bins_m4b, 1, 1, 2);
  truthM4b->SetCanExtend(1);
  truthM4b->GetXaxis()->FindBin("all");

} 

void cutflowHists::Fill(std::string cut, eventData* event){

  unitWeight->Fill(cut.c_str(), 1);
  weighted  ->Fill(cut.c_str(), event->weight);
  truthM4b  ->Fill(event->truth->m4b, cut.c_str(), event->weight);

  if(event->passHLT){
    unitWeight->Fill((cut+"_HLT").c_str(), 1);
    weighted  ->Fill((cut+"_HLT").c_str(), event->weight);
    truthM4b  ->Fill(event->truth->m4b, (cut+"_HLT").c_str(), event->weight);
  }

  return;
}

cutflowHists::~cutflowHists(){} 

