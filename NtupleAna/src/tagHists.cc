#include "ZZ4b/NtupleAna/interface/tagHists.h"

using namespace NtupleAna;

tagHists::tagHists(std::string name, fwlite::TFileService& fs, bool doViews, bool blind) {
  dir = fs.mkdir(name);

  threeTag = new eventHists(name+"/threeTag", fs, doViews, false);
  fourTag  = new eventHists(name+"/fourTag",  fs, doViews, blind);

} 

void tagHists::Fill(eventData* event){

  if(event->threeTag) threeTag->Fill(event);
  if(event->fourTag)   fourTag->Fill(event);

  return;
}

void tagHists::Fill(eventData* event, std::vector<eventView*> &views){

  if(event->threeTag) threeTag->Fill(event, views);
  if(event->fourTag)   fourTag->Fill(event, views);

  return;
}

tagHists::~tagHists(){} 


