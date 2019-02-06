#include "ZZ4b/NtupleAna/interface/massRegionHists.h"

using namespace NtupleAna;

massRegionHists::massRegionHists(std::string name, fwlite::TFileService& fs) {
  dir = fs.mkdir(name);

  inclusive = new viewHists(name+"/inclusive", fs);
  ZZ        = new viewHists(name+"/ZZ",        fs);
  ZH        = new viewHists(name+"/ZH",        fs);

} 

void massRegionHists::Fill(eventData* event, eventView* view){
  inclusive->Fill(event, view);
  if(view->ZZ) ZZ->Fill(event, view);
  if(view->ZH) ZH->Fill(event, view);

  return;
}

massRegionHists::~massRegionHists(){} 


