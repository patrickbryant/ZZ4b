#include <iostream>

#include "ZZ4b/NtupleAna/interface/truthData.h"

using namespace NtupleAna;

truthData::truthData(TChain* t, bool d){
  tree  = t;
  debug = d;

  truthParticles  = new truthParticle("GenPart",  tree);
} 

void truthData::update(){
  if(debug) std::cout<<"Reset truthData"<<std::endl;

  //Objects
  bQuarksZ = truthParticles->getParticles(5, 23);
  bQuarksH = truthParticles->getParticles(5, 25);
  
  m4b = -99;
  if(bQuarksZ.size() == 2 && bQuarksH.size() == 2){ //ZH
    m4b = (bQuarksZ[0]->p + bQuarksZ[1]->p + bQuarksH[0]->p + bQuarksH[1]->p).M();
  } 
  if(bQuarksZ.size() == 4){ //ZZ
    m4b = (bQuarksZ[0]->p + bQuarksZ[1]->p + bQuarksZ[2]->p + bQuarksZ[3]->p).M();
  } 
  if(bQuarksH.size() == 4){ //HH
    m4b = (bQuarksH[0]->p + bQuarksH[1]->p + bQuarksH[2]->p + bQuarksH[3]->p).M();
  } 

  if(debug) dump();

  if(debug) std::cout<<"truthData updated\n";
  return;
}

void truthData::dump(){
  std::cout << "truthData:\n";
  std::cout << "# bQuarksZ = " << bQuarksZ.size() << std::endl; 
  std::cout << "# bQuarksH = " << bQuarksH.size() << std::endl; 
  std::cout << "m4b = " << m4b << std::endl;
  return;
}

truthData::~truthData(){} 

