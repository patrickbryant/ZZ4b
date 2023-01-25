#include "ZZ4b/nTupleAnalysis/interface/massRegionHists.h"
#include "nTupleAnalysis/baseClasses/interface/helpers.h"

using namespace nTupleAnalysis;

massRegionHists::massRegionHists(std::string name, fwlite::TFileService& fs, bool isMC, bool _blind, std::string histDetailLevel, bool _debug, eventData* event) {
  if(_debug) std::cout << "massRegionHists::massRegionHists(std::string name, fwlite::TFileService& fs, bool isMC, bool _blind, std::string histDetailLevel, bool _debug, eventData* event)" << std::endl;
  dir = fs.mkdir(name);
  blind = _blind;
  debug = _debug;

  inclusive = new viewHists(name+"/inclusive", fs, isMC, debug, NULL, histDetailLevel);
  // notSR     = new viewHists(name+"/notSR", fs, isMC, debug, NULL, histDetailLevel);
  SR        = new viewHists(name+"/SR", fs, isMC, debug, event, histDetailLevel);
  // SRNoZZ    = new viewHists(name+"/SRNoZZ", fs, isMC, debug, event, histDetailLevel);
  // SRNoHH    = new viewHists(name+"/SRNoHH", fs, isMC, debug, event, histDetailLevel);
  // CR        = new viewHists(name+"/CR", fs, isMC, debug, NULL, histDetailLevel);
  SB        = new viewHists(name+"/SB", fs, isMC, debug, NULL, histDetailLevel);
  SBSR      = new viewHists(name+"/SBSR", fs, isMC, debug, NULL, histDetailLevel);
  outSB     = new viewHists(name+"/outSB", fs, isMC, debug, NULL, histDetailLevel);

  // if(nTupleAnalysis::findSubStr(histDetailLevel,"ZHRegions")){
  //   ZHSR      = new viewHists(name+"/ZHSR",      fs, isMC, debug, NULL, histDetailLevel );
  //   // ZHCR      = new viewHists(name+"/ZHCR",      fs, isMC, debug, NULL, histDetailLevel );
  //   // ZHSB      = new viewHists(name+"/ZHSB",      fs, isMC, debug, NULL, histDetailLevel );
  //   // ZH        = new viewHists(name+"/ZH",        fs, isMC, debug, NULL, histDetailLevel );
  // }
    
  // // ZH_SvB_high = new viewHists(name+"/ZH_SvB_high", fs, isMC, debug);
  // // ZH_SvB_low  = new viewHists(name+"/ZH_SvB_low",  fs, isMC, debug);
  // if(nTupleAnalysis::findSubStr(histDetailLevel,"ZZRegions")){
  //   ZZSR      = new viewHists(name+"/ZZSR",      fs, isMC, debug, NULL, histDetailLevel );
  //   // ZZCR      = new viewHists(name+"/ZZCR",      fs, isMC, debug, NULL, histDetailLevel );
  //   // ZZSB      = new viewHists(name+"/ZZSB",      fs, isMC, debug, NULL, histDetailLevel );
  //   // ZZ        = new viewHists(name+"/ZZ",        fs, isMC, debug, NULL, histDetailLevel );
  // }
    
  // if(nTupleAnalysis::findSubStr(histDetailLevel,"HHRegions")){
  //   HHSR      = new viewHists(name+"/HHSR",      fs, isMC, debug, NULL, histDetailLevel );
  //   // HHCR      = new viewHists(name+"/HHCR",      fs, isMC, debug, NULL, histDetailLevel );
  //   // HHSB      = new viewHists(name+"/HHSB",      fs, isMC, debug, NULL, histDetailLevel );
  //   // HH        = new viewHists(name+"/HH",        fs, isMC, debug, NULL, histDetailLevel );
  // }

  if(nTupleAnalysis::findSubStr(histDetailLevel,"HHSR")){
    HHSR      = new viewHists(name+"/HHSR",      fs, isMC, debug, NULL, histDetailLevel );
  }

  // if(!ZH) std::cout << "\t Turning off ZZ Regions " << std::endl;
  // if(!ZZ) std::cout << "\t Turning off ZH Regions " << std::endl;
  // if(!HH){ std::cout << "\t Turning off HH Regions " << std::endl;
  if(HHSR) std::cout << "\t\t Turning on HHSR " << std::endl;
  // }

} 

void massRegionHists::Fill(eventData* event, std::shared_ptr<eventView> &view){
  if(blind && (view->SR)) return;

  // int nViews    = 0; int this_random = (int) view->random;//event->views.size();
  // int nViews_10 = 0;
  // int nViews_11 = 0;
  // int nViews_12 = 0;
  // for(auto &v: event->views){ 
  //   if(this_random == (int)v->random) nViews += 1;
  //   if(v->random >= 10) nViews_10 += 1;
  //   if(v->random >= 11) nViews_11 += 1;
  //   if(v->random >= 12) nViews_12 += 1;
  // }
  inclusive->Fill(event, view);//, nViews, nViews_10, nViews_11, nViews_12);


  if(view->SR){
    // nViews = 0; nViews_10 = 0; nViews_11 = 0; nViews_12 = 0;
    // for(auto &v: event->views){ 
    //   if(!v->SR) continue;
    //   nViews += 1;
    //   if(v->random >= 10) nViews_10 += 1;
    //   if(v->random >= 11) nViews_11 += 1;
    //   if(v->random >= 12) nViews_12 += 1;
    // }
    SR->Fill(event, view);//, nViews, nViews_10, nViews_11, nViews_12);
  }


  if(view->SB){
    // nViews = 0; nViews_10 = 0; nViews_11 = 0; nViews_12 = 0;
    // for(auto &v: event->views){ 
    //   if(!v->SB) continue;
    //   nViews += 1;
    //   if(v->random >= 10) nViews_10 += 1;
    //   if(v->random >= 11) nViews_11 += 1;
    //   if(v->random >= 12) nViews_12 += 1;
    // }
    SB->Fill(event, view);//, nViews, nViews_10, nViews_11, nViews_12);
  }


  if(view->SB || view->SR){
    // nViews = 0; nViews_10 = 0; nViews_11 = 0; nViews_12 = 0;
    // for(auto &v: event->views){ 
    //   if(!(v->SB || v->SR)) continue;
    //   nViews += 1;
    //   if(v->random >= 10) nViews_10 += 1;
    //   if(v->random >= 11) nViews_11 += 1;
    //   if(v->random >= 12) nViews_12 += 1;
    // }
    SBSR->Fill(event, view);//, nViews, nViews_10, nViews_11, nViews_12);
  }

  if(!view->SB && !view->SR){
    outSB->Fill(event, view);
  }


  if(HHSR){
    if(view->HHSR){
      // nViews = 0; nViews_10 = 0; nViews_11 = 0; nViews_12 = 0;
      // for(auto &v: event->views){ 
      // 	if(!v->HHSR) continue;
      // 	nViews += 1;
      // 	if(v->random >= 10) nViews_10 += 1;
      // 	if(v->random >= 11) nViews_11 += 1;
      // 	if(v->random >= 12) nViews_12 += 1;
      // }
      HHSR->Fill(event, view);//, nViews, nViews_10, nViews_11, nViews_12);
    }
  }
  return;
}

massRegionHists::~massRegionHists(){} 


