#include "ZZ4b/NtupleAna/interface/eventView.h"

using namespace NtupleAna;

//eventView object
eventView::eventView(dijet* dijet1, dijet* dijet2){

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

  dHH = getDHH(leadSt->m, sublSt->m);
  xZZ = getXZZ(leadSt->m, sublSt->m);
  mZZ = (dijet1->pZ + dijet2->pZ).M();
  mZH = (leadM->pH + sublM->pZ).M();

  ZZ = (xZZ < 1.6);


        // self.passLeadStMDR = (360/self.m4j - 0.5 < self.leadSt.dR) and (self.leadSt.dR < 653/self.m4j + 0.475) if self.m4j < 1250 else (self.leadSt.dR < 1)
        // self.passSublStMDR = (235/self.m4j       < self.sublSt.dR) and (self.sublSt.dR < 875/self.m4j + 0.350) if self.m4j < 1250 else (self.sublSt.dR < 1)
        // self.passMDRs = self.passLeadStMDR and self.passSublStMDR

        // self.passLeadMDC = self.lead.pt > self.m4j*0.51 - 103
        // self.passSublMDC = self.subl.pt > self.m4j*0.33 -  73
        // self.passMDCs = self.passLeadMDC and self.passSublMDC

        // self.dEta = self.leadSt.eta - self.sublSt.eta
        // self.passHCdEta = abs(self.dEta) < 1.5

}

eventView::~eventView(){}


