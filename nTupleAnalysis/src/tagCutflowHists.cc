//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/tagCutflowHists.h"

using namespace nTupleAnalysis;

tagCutflowHists::tagCutflowHists(std::string name, fwlite::TFileService& fs, bool isMC) {

  dir = fs.mkdir(name);
  threeTag = new cutflowHists(name+"/threeTag", fs, isMC);
  fourTag  = new cutflowHists(name+"/fourTag",  fs, isMC);

} 

void tagCutflowHists::Fill(eventData* event, std::string cut, bool fillAll){

  if(fillAll || event->threeTag) threeTag->Fill(cut, event);
  if(fillAll || event-> fourTag)  fourTag->Fill(cut, event);

  return;
}

tagCutflowHists::~tagCutflowHists(){} 

