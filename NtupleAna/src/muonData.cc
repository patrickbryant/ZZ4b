#include "TChain.h"

#include "ZZ4b/NtupleAna/interface/muonData.h"

using namespace NtupleAna;

//muon object
muon::muon(UInt_t i, muonData* data){

  pt  = data->pt [i];
  eta = data->eta[i];
  phi = data->phi[i];
  m   = data->m  [i];
  p = new TLorentzVector();
  p->SetPtEtaPhiM(pt, eta, phi, m);

  deepCSV = data->deepCSV[i];

}

muon::~muon(){}


//access tree
muonData::muonData(std::string name, TChain* tree){

  tree->SetBranchAddress( ("n"+name).c_str(), &n );

  tree->SetBranchAddress( (name+"_pt"  ).c_str(), &pt );  
  tree->SetBranchAddress( (name+"_eta" ).c_str(), &eta );  
  tree->SetBranchAddress( (name+"_phi" ).c_str(), &phi );  
  tree->SetBranchAddress( (name+"_mass").c_str(), &m );  

  tree->SetBranchAddress( (name+"_softId").c_str(), &softId );
  tree->SetBranchAddress( (name+"_mediumId").c_str(), &mediumId );
  tree->SetBranchAddress( (name+"_tightId").c_str(), &tightId );
  //tree->SetBranchAddress( (name+"_").c_str(). & );

}

std::vector<muon> muonData::getMuons(float ptMin, float etaMax, float tagMin){

  std::vector<muon> outputMuons;
  for(UInt_t i = 0; i < n; ++i){
    if(      pt[i] < ptMin) continue;
    if(fabs(eta[i])>etaMax) continue;
    if( deepCSV[i] <tagMin) continue;
    outputMuons.push_back( muon(i, this) );
  }

  return outputMuons;
}

muonData::~muonData(){}
