#include "ZZ4b/nTupleAnalysis/interface/eventView.h"

using namespace nTupleAnalysis;

//eventView object
eventView::eventView(std::shared_ptr<dijet> &dijet1, std::shared_ptr<dijet> &dijet2, float FvT_q_score_, float SvB_q_score_, float SvB_MA_q_score_){
  
  FvT_q_score = FvT_q_score_;
  SvB_q_score = SvB_q_score_;
  SvB_MA_q_score = SvB_MA_q_score_;

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

  if(dijet1->dR < dijet2->dR){
    close = dijet1;
    other = dijet2;
  }else{
    close = dijet2;
    other = dijet1;
  }

  truthMatch = (bool)dijet1->truthMatch && (bool)dijet2->truthMatch;

  p   = dijet1->p + dijet2->p;
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();

  m4j = m;
  mZZ = (leadSt->pZ + sublSt->pZ).M();
  mZH = (leadM ->pH + sublM ->pZ).M();
  mHH = (leadSt->pH + sublSt->pH).M();

  //Signal Regions
  xZZ = getXZZ(leadSt->m, sublSt->m); //0 for perfect consistency with ZZ->4b
  xZH = getXZH(leadSt->m, sublSt->m); //0 for perfect consistency with ZH->4b
  xHH = getXHH(leadSt->m, sublSt->m); //0 for perfect consistency with HH->4b
  ZZSR = (xZZ < xMaxZZSR);
  ZHSR = (xZH < xMaxZHSR);
  HHSR = (xHH < xMaxHHSR);
  SR = ZZSR || ZHSR || HHSR;

  // //Control Regions
  // rZZCR = sqrt( pow(leadSt->m - leadZ *sZZCR, 2) + pow(sublSt->m - sublZ *sZZCR, 2) );
  // rZHCR = sqrt( pow(leadSt->m - leadZH*sZHCR, 2) + pow(sublSt->m - sublZH*sZHCR, 2) );
  // rHHCR = sqrt( pow(leadSt->m - leadH *sHHCR, 2) + pow(sublSt->m - sublH *sHHCR, 2) );
  // // in outer radius but not in any SR
  // ZZCR = (rZZCR < rMaxZZCR) && !ZZSR && !ZHSR && !HHSR;
  // ZHCR = (rZHCR < rMaxZHCR) && !ZHSR && !ZZSR && !HHSR;
  // HHCR = (rHHCR < rMaxHHCR) && !HHSR && !ZZSR && !ZHSR;
  // CR = (ZZCR || ZHCR || HHCR) && !SR;

  // //Sidebands
  // rZZSB = sqrt( pow(leadSt->m - leadZ *sZZSB, 2) + pow(sublSt->m - sublZ *sZZSB, 2) );
  // rZHSB = sqrt( pow(leadSt->m - leadZH*sZHSB, 2) + pow(sublSt->m - sublZH*sZHSB, 2) );
  // rHHSB = sqrt( pow(leadSt->m - leadH *sHHSB, 2) + pow(sublSt->m - sublH *sHHSB, 2) );
  // ZZSB = (rZZSB < rMaxZZSB) && !ZZSR && !ZZCR && !ZHSR && !HHSR;
  // ZHSB = (rZHSB < rMaxZHSB) && !ZHSR && !ZHCR && !ZZSR && !HHSR;
  // HHSB = (rHHSB < rMaxHHSB) && !HHSR && !HHCR && !ZZSR && !ZHSR;
  // SB = (ZZSB || ZHSB || HHSB || CR) && !SR;// && !CR && !SR;
  passDijetMass = (leadSt->m>52) && (leadSt->m<180) && (sublSt->m>50) && (sublSt->m<173);
  SB = (!SR) && passDijetMass;

  dBB = getDBB(leadSt->m, sublSt->m); //Distance from being equal mass boson candidates

  passLeadStMDR = (360/m4j - 0.5 < leadSt->dR) && (leadSt->dR < std::max(650/m4j + 0.5, 1.5));
  passSublStMDR = (235/m4j       < sublSt->dR) && (sublSt->dR < std::max(650/m4j + 0.7, 1.5));
  //passLeadStMDR = (m4j < 1250) ? (360/m4j - 0.5 < leadSt->dR) && (leadSt->dR < 653/m4j + 0.977) : (leadSt->dR < 1.5);
  //passSublStMDR = (m4j < 1250) ? (235/m4j       < sublSt->dR) && (sublSt->dR < 875/m4j + 0.800) : (sublSt->dR < 1.5);
  passMDRs = passLeadStMDR && passSublStMDR;

  //passLeadMDC = lead->pt > m4j*0.51 - 103;
  //passSublMDC = subl->pt > m4j*0.33 -  73;
  //passMDCs = passLeadMDC && passSublMDC;

  dEtaBB = dijet1->eta - dijet2->eta;
  dRBB = dijet1->p.DeltaR(dijet2->p);
  dPhiBB = dijet1->p.DeltaPhi(dijet2->p);
  //passDEtaBB = fabs(dEtaBB) < 1.5;

}

eventView::~eventView(){}


