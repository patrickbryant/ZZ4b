//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/cutflowHists.h"

using namespace nTupleAnalysis;

cutflowHists::cutflowHists(std::string name, fwlite::TFileService& fs, bool isMC, bool _debug) {
  debug = _debug;
  dir = fs.mkdir(name);
  unitWeight = dir.make<TH1I>("unitWeight", (name+"/unitWeight; ;Entries").c_str(),  1,1,2);
  unitWeight->SetCanExtend(1);

  
  weighted = dir.make<TH1D>("weighted", (name+"/weighted; ;Entries").c_str(),  1,1,2);
  weighted->SetCanExtend(1);

  AddCut("all");

  if(isMC){
    //Make a weighted cutflow as a function of the true m4b, xaxis is m4b, yaxis is cut name. 
    //Double_t bins_m4b[] = {100, 112, 126, 142, 160, 181, 205, 232, 263, 299, 340, 388, 443, 507, 582, 669, 770, 888, 1027, 1190, 1381, 1607, 2000};
    Double_t bins_m4b[] =   {100,                160,              250,        340,      443,      582,      770,      1027,       1381,       2000};
    truthM4b = dir.make<TH2F>("truthM4b", (name+"/truthM4b; Truth m_{4b} [GeV]; ;Entries").c_str(), 9, bins_m4b, 1, 1, 2);
    truthM4b->SetCanExtend(TH1::kYaxis);
    truthM4b->GetYaxis()->FindBin("all");
    truthM4b->GetXaxis()->SetAlphanumeric(0);
  }
} 

void cutflowHists::AddCut(std::string cut){
  unitWeight->GetXaxis()->FindBin(cut.c_str());  
  weighted->GetXaxis()->FindBin(cut.c_str());
  unitWeight->GetXaxis()->FindBin((cut+"_SR").c_str());  
  weighted->GetXaxis()->FindBin((cut+"_SR").c_str());
  unitWeight->GetXaxis()->FindBin((cut+"_HLT").c_str());  
  weighted->GetXaxis()->FindBin((cut+"_HLT").c_str());
  unitWeight->GetXaxis()->FindBin((cut+"_SR_HLT").c_str());  
  weighted->GetXaxis()->FindBin((cut+"_SR_HLT").c_str());
  unitWeight->GetXaxis()->FindBin((cut+"_SR_HLT_VetoHH").c_str());  
  weighted->GetXaxis()->FindBin((cut+"_SR_HLT_VetoHH").c_str());
  if(truthM4b != NULL){
    truthM4b->GetYaxis()->FindBin(cut.c_str());
    truthM4b->GetYaxis()->FindBin((cut+"_SR").c_str());
    truthM4b->GetYaxis()->FindBin((cut+"_HLT").c_str());
    truthM4b->GetYaxis()->FindBin((cut+"_SR_HLT").c_str());
    truthM4b->GetYaxis()->FindBin((cut+"_SR_HLT_VetoHH").c_str());
  }
}

void cutflowHists::BasicFill(const std::string& cut, eventData* event, float weight){
  if(debug) std::cout << "cutflowHists::BasicFill(const std::string& cut, eventData* event, float weight) " << cut << std::endl; 
  unitWeight->Fill(cut.c_str(), 1);
  weighted  ->Fill(cut.c_str(), weight);
  if(truthM4b && event->truth){
    if(debug) std::cout << event->truth->m4b << " " << cut.c_str() << std::endl;
    truthM4b->Fill(event->truth->m4b, cut.c_str(), weight);
  }
  return;
}

void cutflowHists::BasicFill(const std::string& cut, eventData* event){
  BasicFill(cut, event, event->weight);
  return;
}

void cutflowHists::Fill(const std::string& cut, eventData* event){
  if(debug) std::cout << "cutflowHists::Fill(const std::string& cut, eventData* event) " << cut << std::endl;

  //Cut+Trigger
  if(event->doTrigEmulation){
    BasicFill(cut, event, event->weightNoTrigger);

    BasicFill(cut+"_HLT_HT330_4j_75_60_45_40_3b", event);
    BasicFill(cut+"_HLT", event);

  }else{
    BasicFill(cut, event);
    if(event->passL1)                      BasicFill(cut+"_L1", event);
    if(event->HLT_HT330_4j_75_60_45_40_3b) BasicFill(cut+"_HLT_HT330_4j_75_60_45_40_3b", event);
    if(event->HLT_4j_103_88_75_15_2b_VBF1) BasicFill(cut+"_HLT_4j_103_88_75_15_2b_VBF1", event);
    if(event->HLT_4j_103_88_75_15_1b_VBF2) BasicFill(cut+"_HLT_4j_103_88_75_15_1b_VBF2", event);
    if(event->HLT_2j116_dEta1p6_2b)        BasicFill(cut+"_HLT_2j116_dEta1p6_2b", event);
    if(event->HLT_J330_m30_2b)             BasicFill(cut+"_HLT_J330_m30_2b", event);
    if(event->HLT_j500)                    BasicFill(cut+"_HLT_j500", event);
    if(event->HLT_2j300ave)                BasicFill(cut+"_HLT_2j300ave", event);
    if(event->passHLT)                     BasicFill(cut+"_HLT", event);
  }

  if(event->views.size()>0){
    
    //Cut+SR
    if(event->views[0]->SR){
      if(event->doTrigEmulation){
	BasicFill(cut+"_SR", event, event->weightNoTrigger);
	BasicFill(cut+"_SR_HLT_HT330_4j_75_60_45_40_3b", event, event->weight);
	BasicFill(cut+"_SR_HLT", event, event->weight);
      }else{
	BasicFill(cut+"_SR", event);
	//Cut+SR+Trigger
	if(event->passL1)                      BasicFill(cut+"_SR_L1", event);
	if(event->HLT_HT330_4j_75_60_45_40_3b) BasicFill(cut+"_SR_HLT_HT330_4j_75_60_45_40_3b", event);
	if(event->HLT_4j_103_88_75_15_2b_VBF1) BasicFill(cut+"_SR_HLT_4j_103_88_75_15_2b_VBF1", event);
	if(event->HLT_4j_103_88_75_15_1b_VBF2) BasicFill(cut+"_SR_HLT_4j_103_88_75_15_1b_VBF2", event);
	if(event->HLT_2j116_dEta1p6_2b)        BasicFill(cut+"_SR_HLT_2j116_dEta1p6_2b", event);
	if(event->HLT_J330_m30_2b)             BasicFill(cut+"_SR_HLT_J330_m30_2b", event);
	if(event->HLT_j500)                    BasicFill(cut+"_SR_HLT_j500", event);
	if(event->HLT_2j300ave)                BasicFill(cut+"_SR_HLT_2j300ave", event);
	if(event->passHLT)                     BasicFill(cut+"_SR_HLT", event);
	if(event->passHLT & !event->HHSR)      BasicFill(cut+"_SR_HLT_VetoHH", event);
      }
    }
  }

  return;
}

void cutflowHists::labelsDeflate(){
  unitWeight->LabelsDeflate("X");
  //unitWeight->LabelsOption("a");
  weighted  ->LabelsDeflate("X");
  //weighted  ->LabelsOption("a");
  if(truthM4b != NULL){
    truthM4b->LabelsDeflate("Y");
    truthM4b->LabelsOption("a","Y");
  }
  return;  
}

cutflowHists::~cutflowHists(){} 

