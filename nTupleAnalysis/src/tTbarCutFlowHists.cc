//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/tTbarCutFlowHists.h"

using namespace nTupleAnalysis;

tTbarCutFlowHists::tTbarCutFlowHists(std::string name, fwlite::TFileService& fs, bool isMC, bool _debug) {
  debug = _debug;
  dir = fs.mkdir(name);
  unitWeight = dir.make<TH1I>("unitWeight", (name+"/unitWeight; ;Entries").c_str(),  1,1,2);
  unitWeight->SetCanExtend(1);

  
  weighted = dir.make<TH1D>("weighted", (name+"/weighted; ;Entries").c_str(),  1,1,2);
  weighted->SetCanExtend(1);

  AddCut("all");
} 

void tTbarCutFlowHists::AddCut(std::string cut){
  unitWeight->GetXaxis()->FindBin(cut.c_str());  
  unitWeight->GetXaxis()->FindBin((cut+"_HLT").c_str());  

  weighted->GetXaxis()->FindBin(cut.c_str());
  weighted->GetXaxis()->FindBin((cut+"_HLT").c_str());
}

void tTbarCutFlowHists::BasicFill(const std::string& cut, tTbarEventData* event, float weight){
  if(debug) std::cout << "tTbarCutFlowHists::BasicFill(const std::string& cut, tTbarEventData* event, float weight) " << cut << std::endl; 
  unitWeight->Fill(cut.c_str(), 1);
  weighted  ->Fill(cut.c_str(), weight);

  if(event->passHLT) {
    unitWeight->Fill((cut+"_HLT").c_str(), 1);
    weighted  ->Fill((cut+"_HLT").c_str(), weight);
  }

  return;
}

void tTbarCutFlowHists::BasicFill(const std::string& cut, tTbarEventData* event){
  BasicFill(cut, event, event->weight);
  return;
}

void tTbarCutFlowHists::Fill(const std::string& cut, tTbarEventData* event){
  if(debug) std::cout << "tTbarCutFlowHists::Fill(const std::string& cut, tTbarEventData* event) " << cut << std::endl;

  BasicFill(cut, event);

  return;
}

void tTbarCutFlowHists::labelsDeflate(){
  unitWeight->LabelsDeflate("X");
  //unitWeight->LabelsOption("a");
  weighted  ->LabelsDeflate("X");
  //weighted  ->LabelsOption("a");
  return;  
}

tTbarCutFlowHists::~tTbarCutFlowHists(){} 

