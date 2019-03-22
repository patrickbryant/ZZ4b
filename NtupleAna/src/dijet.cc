#include "ZZ4b/NtupleAna/interface/dijet.h"

using namespace NtupleAna;

//dijet object
//dijet::dijet(){}
dijet::dijet(std::shared_ptr<jet> &jet1, std::shared_ptr<jet> &jet2){

  if(jet1->pt > jet2->pt){
    lead = jet1;
    subl = jet2;
  }else{
    lead = jet2;
    subl = jet1;
  }

  dR  = jet1->p.DeltaR(jet2->p);
  st  = jet1->pt + jet2->pt;
  p   = jet1->p  + jet2->p;
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();

  pZ  = p*(mZ/m);
  pH  = p*(mH/m);

}

dijet::~dijet(){}


