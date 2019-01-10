#include "ZZ4b/NtupleAna/interface/dijet.h"

using namespace NtupleAna;

//dijet object
dijet::dijet(){}
dijet::dijet(jet& jet1, jet& jet2){

  lead = (jet1.pt > jet2.pt) ? jet1 : jet2;
  subl = (jet1.pt > jet2.pt) ? jet2 : jet1;

  p = jet1.p + jet2.p;
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();

}

dijet::~dijet(){}


