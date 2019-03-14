//#include "TChain.h"
#include "ZZ4b/NtupleAna/interface/cutflowHists.h"

using namespace NtupleAna;

cutflowHists::cutflowHists(std::string name, fwlite::TFileService& fs, bool isMC) {

  dir = fs.mkdir(name);
  unitWeight = dir.make<TH1F>("unitWeight", (name+"/unitWeight; ;Entries").c_str(),  1,1,2);
  unitWeight->SetCanExtend(1);
  unitWeight->GetXaxis()->FindBin("all");
  
  weighted = dir.make<TH1F>("weighted", (name+"/weighted; ;Entries").c_str(),  1,1,2);
  weighted->SetCanExtend(1);
  weighted->GetXaxis()->FindBin("all");

  if(isMC){
    //Make a weighted cutflow as a function of the true m4b, xaxis is m4b, yaxis is cut name. 
    Double_t bins_m4b[] = {100, 112, 126, 142, 160, 181, 205, 232, 263, 299, 340, 388, 443, 507, 582, 669, 770, 888, 1027, 1190, 1381, 1607, 2000};
    truthM4b = dir.make<TH2F>("truthM4b", (name+"/truthM4b; Truth m_{4b} [GeV]; ;Entries").c_str(), 22, bins_m4b, 1, 1, 2);
    truthM4b->SetCanExtend(TH1::kYaxis);
    truthM4b->GetYaxis()->FindBin("all");
    truthM4b->GetXaxis()->SetAlphanumeric(0);
  }
} 

void cutflowHists::BasicFill(std::string cut, eventData* event){
  unitWeight->Fill(cut.c_str(), 1);
  weighted  ->Fill(cut.c_str(), event->weight);
  if(truthM4b != NULL) 
    truthM4b->Fill(event->truth->m4b, cut.c_str(), event->weight);
  return;
}

void cutflowHists::Fill(std::string cut, eventData* event){

  BasicFill(cut, event);

  //Cut+Trigger
  if(event->passHLT) BasicFill(cut+"_HLT", event);

  if(event->views.size()>0){
    
    //Cut+SR
    if(event->views[0]->ZHSR){
      BasicFill(cut+"_ZHSR", event);

      //Cut+SR+Trigger
      if(event->passHLT) BasicFill(cut+"_ZHSR_HLT", event);

    }
  }

  return;
}

cutflowHists::~cutflowHists(){} 

