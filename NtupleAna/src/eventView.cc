#include "ZZ4b/NtupleAna/interface/eventView.h"

using namespace NtupleAna;

//eventView object
eventView::eventView(dijet dijet1, dijet dijet2){

  lead = (dijet1.pt > dijet2.pt) ? dijet1 : dijet2;
  subl = (dijet1.pt > dijet2.pt) ? dijet2 : dijet1;

  p = new TLorentzVector(dijet1.p + dijet2.p);
  pt  = p->Pt();
  eta = p->Eta();
  phi = p->Phi();
  m   = p->M();
  e   = p->E();

}

eventView::~eventView(){}


