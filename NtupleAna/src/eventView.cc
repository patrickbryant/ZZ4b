#include "ZZ4b/NtupleAna/interface/eventView.h"

using namespace NtupleAna;

//eventView object
eventView::eventView(std::shared_ptr<dijet> &dijet1, std::shared_ptr<dijet> &dijet2){

  if(dijet1->pt > dijet2->pt){
    lead = dijet1;
    subl = dijet2;
  }else{
    lead = dijet2;
    subl = dijet1;
  }

  if(dijet1->st > dijet2->st){
    leadSt = dijet1;
    sublSt = dijet2;
  }else{
    leadSt = dijet2;
    sublSt = dijet1;
  }

  if(dijet1->m > dijet2->m){
    leadM = dijet1;
    sublM = dijet2;
  }else{
    leadM = dijet2;
    sublM = dijet1;
  }

  p   = dijet1->p + dijet2->p;
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();

  dBB = getDBB(leadSt->m, sublSt->m); //Distance from being equal mass boson candidates

  m4j = m;
  mZZ = (leadSt->pZ + sublSt->pZ).M();
  mZH = (leadM ->pH + sublM ->pZ).M();
  mHH = (leadSt->pH + sublSt->pH).M();

  //Signal Regions
  xZZ = getXZZ(leadSt->m, sublSt->m); //0 for perfect consistency with ZZ->4b
  xZH = getXZH(leadM ->m, sublM ->m); //0 for perfect consistency with ZH->4b
  xHH = getXHH(leadSt->m, sublSt->m); //0 for perfect consistency with HH->4b
  ZZSR = (xZZ < xMaxZZSR);
  ZHSR = (xZH < xMaxZHSR);
  HHSR = (xHH < xMaxHHSR);

  //Control Regions
  rZZCR = sqrt( pow(leadSt->m - leadZZ*sZZCR, 2) + pow(sublSt->m - sublZZ*sZZCR, 2) );
  rZHCR = sqrt( pow(leadM ->m - leadZH*sZHCR, 2) + pow(sublM ->m - sublZH*sZHCR, 2) );
  rHHCR = sqrt( pow(leadSt->m - leadHH*sHHCR, 2) + pow(sublSt->m - sublHH*sHHCR, 2) );
  ZZCR = (rZZCR < rMaxZZCR) && !ZZSR;
  ZHCR = (rZHCR < rMaxZHCR) && !ZHSR;
  HHCR = (rHHCR < rMaxHHCR) && !HHSR;

  //Sidebands
  rZZSB = sqrt( pow(leadSt->m - leadZZ*sZZSB, 2) + pow(sublSt->m - sublZZ*sZZSB, 2) );
  rZHSB = sqrt( pow(leadM ->m - leadZH*sZHSB, 2) + pow(sublM ->m - sublZH*sZHSB, 2) );
  rHHSB = sqrt( pow(leadSt->m - leadHH*sHHSB, 2) + pow(sublSt->m - sublHH*sHHSB, 2) );
  ZZSB = (rZZSB < rMaxZZSB) && !ZZSR && !ZZCR;
  ZHSB = (rZHSB < rMaxZHSB) && !ZHSR && !ZHCR;
  HHSB = (rHHSB < rMaxHHSB) && !HHSR && !HHCR;

  passLeadStMDR = (m4j < 1250) ? (360/m4j - 0.5 < leadSt->dR) && (leadSt->dR < 653/m4j + 0.475) : (leadSt->dR < 1);
  passSublStMDR = (m4j < 1250) ? (235/m4j       < sublSt->dR) && (sublSt->dR < 875/m4j + 0.350) : (sublSt->dR < 1);
  passMDRs = passLeadStMDR && passSublStMDR;

  passLeadMDC = lead->pt > m4j*0.51 - 103;
  passSublMDC = subl->pt > m4j*0.33 -  73;
  passMDCs = passLeadMDC && passSublMDC;

  dEtaBB = dijet1->eta - dijet2->eta;
  passDEtaBB = dEtaBB < 1.5;

}

eventView::~eventView(){}


