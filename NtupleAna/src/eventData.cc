#include <iostream>
#include "TChain.h"

#include "ZZ4b/NtupleAna/interface/eventData.h"

using namespace NtupleAna;


eventData::eventData(TChain* t, bool d){
  tree  = t;
  debug = d;

  if(debug){
    std::cout<<"tree->Show(0)"<<std::endl;
    tree->Show(0);
  }

  tree->SetBranchAddress("run",       run_arr);
  tree->SetBranchAddress("event",     event_arr);
  tree->SetBranchAddress("genWeight", genWeight_arr);
} 

void eventData::update(int e){
  if(debug) std::cout<<"Get Entry "<<e<<std::endl;
  tree->GetEntry(e);

  run          = run_arr        [0];         
  event        = event_arr      [0];
  weight       = genWeight_arr  [0];

  return;
}


void eventData::dump(){

  std::cout << "   Run: " << run    << std::endl;
  std::cout << " Event: " << event  << std::endl;  
  std::cout << "Weight: " << weight << std::endl;

  return;
}

eventData::~eventData() {} 

