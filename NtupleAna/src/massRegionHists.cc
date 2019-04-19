#include "ZZ4b/NtupleAna/interface/massRegionHists.h"

using namespace NtupleAna;

massRegionHists::massRegionHists(std::string name, fwlite::TFileService& fs, bool isMC, bool _blind) {
  dir = fs.mkdir(name);
  blind = _blind;

  inclusive = new viewHists(name+"/inclusive", fs, isMC);

  ZZSR      = new viewHists(name+"/ZZSR",      fs, isMC);
  ZHSR      = new viewHists(name+"/ZHSR",      fs, isMC);

  ZZCR      = new viewHists(name+"/ZZCR",      fs, isMC);
  ZHCR      = new viewHists(name+"/ZHCR",      fs, isMC);

  ZZSB      = new viewHists(name+"/ZZSB",      fs, isMC);
  ZHSB      = new viewHists(name+"/ZHSB",      fs, isMC);

  ZH        = new viewHists(name+"/ZH",        fs, isMC);

  ZH_SvB_high = new viewHists(name+"/ZH_SvB_high", fs, isMC);
  ZH_SvB_low  = new viewHists(name+"/ZH_SvB_low",  fs, isMC);

} 

void massRegionHists::Fill(eventData* event, std::unique_ptr<eventView> &view){
  if(blind && (view->ZZSR || view->ZHSR || view->HHSR)) return;
  
  inclusive->Fill(event, view);
  if(view->ZZSR) ZZSR->Fill(event, view);
  if(view->ZHSR) ZHSR->Fill(event, view);

  if(view->ZZCR) ZZCR->Fill(event, view);
  if(view->ZHCR) ZHCR->Fill(event, view);

  if(view->ZZSB) ZZSB->Fill(event, view);
  if(view->ZHSB) ZHSB->Fill(event, view);

  if(view->ZHSB || view->ZHCR || view->ZHSR){
    ZH->Fill(event, view);
    if(event->ZHvsBackgroundClassifier > 0.5)
      ZH_SvB_high->Fill(event, view);
    else
      ZH_SvB_low ->Fill(event, view);
  }

  return;
}

massRegionHists::~massRegionHists(){} 


