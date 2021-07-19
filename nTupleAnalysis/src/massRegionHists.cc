#include "ZZ4b/nTupleAnalysis/interface/massRegionHists.h"
#include "nTupleAnalysis/baseClasses/interface/helpers.h"

using namespace nTupleAnalysis;

massRegionHists::massRegionHists(std::string name, fwlite::TFileService& fs, bool isMC, bool _blind, std::string histDetailLevel, bool _debug, eventData* event) {
  dir = fs.mkdir(name);
  blind = _blind;
  debug = _debug;

  inclusive = new viewHists(name+"/inclusive", fs, isMC, debug, NULL, histDetailLevel);
  SR        = new viewHists(name+"/SR", fs, isMC, debug, event, histDetailLevel);
  SRNoHH    = new viewHists(name+"/SRNoHH", fs, isMC, debug, event, histDetailLevel);
  CR        = new viewHists(name+"/CR", fs, isMC, debug, NULL, histDetailLevel);
  SB        = new viewHists(name+"/SB", fs, isMC, debug, NULL, histDetailLevel);
  SCSR      = new viewHists(name+"/SCSR", fs, isMC, debug, NULL, histDetailLevel);

  if(nTupleAnalysis::findSubStr(histDetailLevel,"ZHRegions")){
    ZHSR      = new viewHists(name+"/ZHSR",      fs, isMC, debug, NULL, histDetailLevel );
    ZHCR      = new viewHists(name+"/ZHCR",      fs, isMC, debug, NULL, histDetailLevel );
    ZHSB      = new viewHists(name+"/ZHSB",      fs, isMC, debug, NULL, histDetailLevel );
    ZH        = new viewHists(name+"/ZH",        fs, isMC, debug, NULL, histDetailLevel );
  }
    
  // ZH_SvB_high = new viewHists(name+"/ZH_SvB_high", fs, isMC, debug);
  // ZH_SvB_low  = new viewHists(name+"/ZH_SvB_low",  fs, isMC, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"ZZRegions")){
    ZZSR      = new viewHists(name+"/ZZSR",      fs, isMC, debug, NULL, histDetailLevel );
    ZZCR      = new viewHists(name+"/ZZCR",      fs, isMC, debug, NULL, histDetailLevel );
    ZZSB      = new viewHists(name+"/ZZSB",      fs, isMC, debug, NULL, histDetailLevel );
    ZZ        = new viewHists(name+"/ZZ",        fs, isMC, debug, NULL, histDetailLevel );
  }
    
  if(nTupleAnalysis::findSubStr(histDetailLevel,"HHRegions")){
    HHSR      = new viewHists(name+"/HHSR",      fs, isMC, debug, NULL, histDetailLevel );
    HHCR      = new viewHists(name+"/HHCR",      fs, isMC, debug, NULL, histDetailLevel );
    HHSB      = new viewHists(name+"/HHSB",      fs, isMC, debug, NULL, histDetailLevel );
    HH        = new viewHists(name+"/HH",        fs, isMC, debug, NULL, histDetailLevel );
  }

  if(nTupleAnalysis::findSubStr(histDetailLevel,"HHSR")){
    HHSR      = new viewHists(name+"/HHSR",      fs, isMC, debug, NULL, histDetailLevel );
  }

  if(!ZH) std::cout << "\t Turning off ZZ Regions " << std::endl;
  if(!ZZ) std::cout << "\t Turning off ZH Regions " << std::endl;
  if(!HH){ std::cout << "\t Turning off HH Regions " << std::endl;
    if(HHSR) std::cout << "\t\t Turning on HHSR " << std::endl;
  }

} 

void massRegionHists::Fill(eventData* event, std::shared_ptr<eventView> &view){
  if(blind && (view->ZZSR || view->ZHSR || view->HHSR)) return;
  
  inclusive->Fill(event, view);

  if(ZHSR && view->ZHSR) ZHSR->Fill(event, view);
  if(ZHCR && view->ZHCR) ZHCR->Fill(event, view);
  if(ZHSR && view->ZHSB) ZHSB->Fill(event, view);

  if(ZH && (view->ZHSB || view->ZHCR || view->ZHSR)){
    ZH->Fill(event, view);
    // if(event->ZHvB > 0.5)
    //   ZH_SvB_high->Fill(event, view);
    // else
    //   ZH_SvB_low ->Fill(event, view);
  }
  
  if(ZZSR && view->ZZSR) ZZSR->Fill(event, view);
  if(ZZCR && view->ZZCR) ZZCR->Fill(event, view);
  if(ZZSB && view->ZZSB) ZZSB->Fill(event, view);
  if(ZZ && (view->ZZSB || view->ZZCR || view->ZZSR)){
    ZZ->Fill(event, view);
  }


  if(HHSR && view->HHSR) HHSR->Fill(event, view);
  if(HHCR && view->HHCR) HHCR->Fill(event, view);
  if(HHSB && view->HHSB) HHSB->Fill(event, view);
  if(HH && (view->HHSB || view->HHCR || view->HHSR)){
    HH->Fill(event, view);
  }

  if(view->SR) SR->Fill(event, view);
  if(view->SR && !view->HHSR) SRNoHH->Fill(event, view);
  if(view->CR) CR->Fill(event, view);
  if(view->SB) SB->Fill(event, view);
  if(view->SB || view->CR || view->SR) SCSR->Fill(event, view);

  return;
}

massRegionHists::~massRegionHists(){} 


