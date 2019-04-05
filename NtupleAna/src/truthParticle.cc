#include "TChain.h"

#include "ZZ4b/NtupleAna/interface/truthParticle.h"

using namespace NtupleAna;

//particle object
particle::particle(){}
particle::particle(UInt_t i, truthParticle* data){

  pt  = data->pt [i];
  eta = data->eta[i];
  phi = data->phi[i];
  m   = data->m  [i];
  p = TLorentzVector();
  p.SetPtEtaPhiM(pt, eta, phi, m);
  e = p.E();

  pdgId = data->pdgId[i];
  genPartIdxMother = data->genPartIdxMother[i];

}

particle::~particle(){}


//access tree
truthParticle::truthParticle(std::string name, TChain* tree){

  initBranch(tree, ("n"+name).c_str(), n );

  initBranch(tree, (name+"_pt"  ).c_str(), pt  );  
  initBranch(tree, (name+"_eta" ).c_str(), eta );  
  initBranch(tree, (name+"_phi" ).c_str(), phi );  
  initBranch(tree, (name+"_mass").c_str(), m   );  

  initBranch(tree, (name+"_genPartIdxMother").c_str(), genPartIdxMother );
  initBranch(tree, (name+"_pdgId").c_str(), pdgId );
  //initBranch(tree, (name+"_").c_str(),  );

}

std::vector<std::shared_ptr<particle>> truthParticle::getParticles(Int_t absPDG, Int_t absMomPDG){
  
  std::vector< std::shared_ptr<particle> > outputParticles;

  for(UInt_t i = 0; i < n; ++i){
    if(    absPDG != -1 && abs(pdgId[i])                   != absPDG    ) continue;
    if( absMomPDG != -1 && abs(pdgId[genPartIdxMother[i]]) != absMomPDG ) continue;
    outputParticles.push_back(std::make_shared<particle>(particle(i, this)));
  }

  return outputParticles;
}

truthParticle::~truthParticle(){}
