#include "TChain.h"

#include "ZZ4b/NtupleAna/interface/muonData.h"

using namespace NtupleAna;

//muon object
muon::muon(UInt_t i, muonData* data){

  pt  = data->pt [i];
  eta = data->eta[i];
  phi = data->phi[i];
  m   = data->m  [i];
  p = TLorentzVector();
  p.SetPtEtaPhiM(pt, eta, phi, m);

  softId   = data->softId[i];
  highPtId = data->highPtId[i];

  mediumId = data->mediumId[i];
  tightId  = data->tightId[i];

  quality  = mediumId + tightId;

  jetIdx    = data->jetIdx[i];
  isolation = data->pfRelIso04_all[i];

}

muon::~muon(){}


//access tree
muonData::muonData(std::string name, TChain* tree){

  tree->SetBranchAddress( ("n"+name).c_str(), &n );

  tree->SetBranchAddress( (name+"_pt"  ).c_str(), &pt );  
  tree->SetBranchAddress( (name+"_eta" ).c_str(), &eta );  
  tree->SetBranchAddress( (name+"_phi" ).c_str(), &phi );  
  tree->SetBranchAddress( (name+"_mass").c_str(), &m );  

  tree->SetBranchAddress( (name+"_softId"  ).c_str(), &softId );
  tree->SetBranchAddress( (name+"_highPtId").c_str(), &highPtId );

  tree->SetBranchAddress( (name+"_mediumId").c_str(), &mediumId );
  tree->SetBranchAddress( (name+"_tightId" ).c_str(), &tightId );

  tree->SetBranchAddress( (name+"_jetIdx").c_str(), &jetIdx );
  tree->SetBranchAddress( (name+"_pfRelIso04_all").c_str(), &pfRelIso04_all );
  //tree->SetBranchAddress( (name+"_").c_str(), & );

}

std::vector<muon> muonData::getMuons(float ptMin, float etaMax, int tag, bool isolation){

  std::vector<muon> outputMuons;
  for(UInt_t i = 0; i < n; ++i){
    if(tag == 0 && softId[i]   == 0) continue;
    if(tag == 1 && highPtId[i] == 0) continue;
    if(tag == 2 && mediumId[i] == 0) continue;
    if(tag == 3 && tightId[i]  == 0) continue;
    if(isolation && pfRelIso04_all[i] > 0.20) continue; //working points here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2

    if(      pt[i] < ptMin) continue;
    if(fabs(eta[i])>etaMax) continue;

    outputMuons.push_back( muon(i, this) );
  }

  return outputMuons;
}

muonData::~muonData(){}
