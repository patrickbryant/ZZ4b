#include "ZZ4b/NtupleAna/interface/dijet.h"

using namespace NtupleAna;

//dijet object
dijet::dijet(){}
dijet::dijet(jet& jet1, jet& jet2){

  lead = (jet1.pt > jet2.pt) ? jet1 : jet2;
  subl = (jet1.pt > jet2.pt) ? jet2 : jet1;

  p   = jet1.p + jet2.p;
  dR  = jet1.p.DeltaR(jet2.p);
  st  = jet1.pt + jet2.pt;
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();

  pZ  = p*(mZ/m);
  pH  = p*(mH/m);

}

dijet::~dijet(){}


