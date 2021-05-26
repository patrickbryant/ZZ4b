//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/tagCutflowHists.h"

using namespace nTupleAnalysis;

tagCutflowHists::tagCutflowHists(std::string name, fwlite::TFileService& fs, bool isMC, bool _debug) {
  debug = _debug;

  dir = fs.mkdir(name);
  threeTag = new cutflowHists(name+"/threeTag", fs, isMC, debug);
  fourTag  = new cutflowHists(name+"/fourTag",  fs, isMC, debug);

  if(isMC){
    btagSF_norm_mcWeight = dir.make<TH1D>("btagSF_norm_mcWeight", (name+"/btagSF_norm_mcWeight; nSelJets; Entries").c_str(), 16,-0.5,15.5);
    btagSF_norm_mcWithSF = dir.make<TH1D>("btagSF_norm_mcWithSF", (name+"/btagSF_norm_mcWithSF; nSelJets; Entries").c_str(), 16,-0.5,15.5);
  }

} 

void tagCutflowHists::AddCut(std::string cut){
  threeTag->AddCut(cut);
  fourTag ->AddCut(cut);
}


void tagCutflowHists::Fill(eventData* event, std::string cut, bool fillAll){

  if(fillAll || event->threeTag) threeTag->Fill(cut, event);
  if(fillAll || event-> fourTag)  fourTag->Fill(cut, event);

  return;
}

void tagCutflowHists::btagSF_norm(eventData* event){
  if(btagSF_norm_mcWeight) btagSF_norm_mcWeight->Fill(event->nSelJets, event->genWeight);
  if(btagSF_norm_mcWithSF) btagSF_norm_mcWithSF->Fill(event->nSelJets, event->genWeight * event->bTagSF);
}

void tagCutflowHists::labelsDeflate(){
  threeTag->labelsDeflate();
  fourTag ->labelsDeflate();
}

tagCutflowHists::~tagCutflowHists(){} 

