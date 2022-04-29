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

    truthM4b_unitWeight = dir.make<TH2F>("truthM4b_unitWeight", (name+"/truthM4b_unitWeight; Truth m_{4b} [GeV]; ;Entries").c_str(), 9, bins_m4b, 1, 1, 2);
    truthM4b_unitWeight->SetCanExtend(TH1::kYaxis);
    truthM4b_unitWeight->GetYaxis()->FindBin("all");
    truthM4b_unitWeight->GetXaxis()->SetAlphanumeric(0);
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

    truthM4b_unitWeight->GetYaxis()->FindBin(cut.c_str());
    truthM4b_unitWeight->GetYaxis()->FindBin((cut+"_SR").c_str());
    truthM4b_unitWeight->GetYaxis()->FindBin((cut+"_HLT").c_str());
    truthM4b_unitWeight->GetYaxis()->FindBin((cut+"_SR_HLT").c_str());
    truthM4b_unitWeight->GetYaxis()->FindBin((cut+"_SR_HLT_VetoHH").c_str());
  }
}

void cutflowHists::BasicFill(const std::string& cut, eventData* event, float weight){
  if(debug) std::cout << "cutflowHists::BasicFill(const std::string& cut, eventData* event, float weight) " << cut << std::endl; 
  unitWeight->Fill(cut.c_str(), 1);
  weighted  ->Fill(cut.c_str(), weight);
  if(truthM4b && event->truth){
    if(debug) std::cout << event->truth->m4b << " " << cut.c_str() << std::endl;
    truthM4b->Fill(event->truth->m4b, cut.c_str(), weight);
    truthM4b_unitWeight->Fill(event->truth->m4b, cut.c_str(), 1);
  }
  return;
}

void cutflowHists::BasicFill(const std::string& cut, eventData* event, bool doTriggers){
  BasicFill(cut, event, event->weight);
  if(doTriggers){
    if(event->passL1) BasicFill(cut+"_L1", event, event->weight);
    for(auto &trigger: event->HLT_triggers){
      bool pass_seed = boost::accumulate(event->HLT_L1_seeds[trigger.first] | boost::adaptors::map_values, false, [](bool pass, bool *seed){return pass||*seed;});
      if(pass_seed && trigger.second) BasicFill(cut+"_"+trigger.first, event, event->weight);
    }
    if(event->passHLT) BasicFill(cut+"_HLT", event, event->weight);
  }
  return;
}

void cutflowHists::Fill(const std::string& cut, eventData* event){
  if(debug) std::cout << "cutflowHists::Fill(const std::string& cut, eventData* event) " << cut << std::endl;

  //Cut+Trigger
  if(event->doTrigEmulation){
    BasicFill(cut, event, event->weightNoTrigger);

    // BasicFill(cut+"_HLT_HT330_4j_75_60_45_40_3b", event);
    BasicFill(cut+"_HLT", event);

  }else{
    BasicFill(cut, event, true);
  }

  if(event->views.size()>0){
    
    //Cut+SR
    if(event->views[0]->SR){
      if(event->doTrigEmulation){
	BasicFill(cut+"_SR", event, event->weightNoTrigger);
	BasicFill(cut+"_SR_HLT", event, event->weight);
      }else{
	BasicFill(cut+"_SR", event, true);
	if(event->passHLT & !event->HHSR)      BasicFill(cut+"_SR_HLT_VetoHH", event);
      }
    }

    //Cut+SR+HHVeto
    if(event->views[0]->SR && !event->views[0]->HHSR){
      if(event->doTrigEmulation){
	BasicFill(cut+"_SRNoHH", event, event->weightNoTrigger);
	BasicFill(cut+"_SRNoHH_HLT", event, event->weight);
      }else{
	BasicFill(cut+"_SRNoHH", event, true);
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
    truthM4b_unitWeight->LabelsDeflate("Y");
    truthM4b_unitWeight->LabelsOption("a","Y");
  }
  return;  
}

cutflowHists::~cutflowHists(){} 

