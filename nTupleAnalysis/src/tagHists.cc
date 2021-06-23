#include "ZZ4b/nTupleAnalysis/interface/tagHists.h"
#include "nTupleAnalysis/baseClasses/interface/helpers.h"

using namespace nTupleAnalysis;

tagHists::tagHists(std::string name, fwlite::TFileService& fs, bool doViews, bool isMC, bool blind, std::string histDetailLevel, bool _debug, eventData* event) {
  std::cout << "Initialize >>   tagHists: " << name << " with detail level: " << histDetailLevel << std::endl;
  dir = fs.mkdir(name);
  debug = _debug;

  if(nTupleAnalysis::findSubStr(histDetailLevel,"threeTag"))
    threeTag = new eventHists(name+"/threeTag", fs, doViews, isMC, false, histDetailLevel, debug, event);

  if(nTupleAnalysis::findSubStr(histDetailLevel,"fourTag"))
    fourTag  = new eventHists(name+"/fourTag",  fs, doViews, isMC, blind, histDetailLevel, debug, event);

  if(!threeTag) std::cout << "\t turning off threeTag Hists" << std::endl; 
  if(!fourTag)  std::cout << "\t turning off threeTag Hists" << std::endl; 

} 

void tagHists::Fill(eventData* event){

  if(threeTag && event->threeTag) threeTag->Fill(event);
  if(fourTag && event->fourTag)   fourTag->Fill(event);

  return;
}

void tagHists::Fill(eventData* event, std::vector<std::shared_ptr<eventView>> &views){

  if(threeTag && event->threeTag) threeTag->Fill(event, views);
  if(fourTag  && event->fourTag)   fourTag->Fill(event, views);

  return;
}

tagHists::~tagHists(){} 


