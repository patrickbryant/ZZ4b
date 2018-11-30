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

  tree->SetBranchAddress("run",       &run);
  tree->SetBranchAddress("event",     &event);
  tree->SetBranchAddress("genWeight", &weight);

  treeJets = new jetData("Jet", tree);
} 


void eventData::update(int e){
  if(debug) std::cout<<"Get Entry "<<e<<std::endl;
  tree->GetEntry(e);

  allJets = treeJets->getJets();
  selJets = treeJets->getJets(40, 2.5);
  tagJets = treeJets->getJets(40, 2.5, 0.4941);//medium WP 2017 from AN2018_073_v10
  //Hack to use leptons as bJets until we get real 4b samples

  return;
}


void eventData::dump(){

  std::cout << "   Run: " << run    << std::endl;
  std::cout << " Event: " << event  << std::endl;  
  std::cout << "Weight: " << weight << std::endl;

  return;
}

eventData::~eventData(){} 

