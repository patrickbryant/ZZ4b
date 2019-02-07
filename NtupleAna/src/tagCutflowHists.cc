//#include "TChain.h"
#include "ZZ4b/NtupleAna/interface/tagCutflowHists.h"

using namespace NtupleAna;

tagCutflowHists::tagCutflowHists(std::string name, fwlite::TFileService& fs) {

  dir = fs.mkdir(name);
  threeTag = new cutflowHists(name+"/threeTag", fs);
  fourTag  = new cutflowHists(name+"/fourTag",  fs);

} 

void tagCutflowHists::Fill(eventData* event, std::string cut){
  
  if(event->threeTag) threeTag->Fill(cut, event->weight);
  if(event-> fourTag)  fourTag->Fill(cut, event->weight);

  return;
}

tagCutflowHists::~tagCutflowHists(){} 

