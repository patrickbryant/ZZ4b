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

  initBranch(tree, ("n"+name).c_str(), &n );

  initBranch(tree, (name+"_pt"  ).c_str(), &pt );  
  initBranch(tree, (name+"_eta" ).c_str(), &eta );  
  initBranch(tree, (name+"_phi" ).c_str(), &phi );  
  initBranch(tree, (name+"_mass").c_str(), &m );  

  initBranch(tree, (name+"_softId"  ).c_str(), &softId );
  initBranch(tree, (name+"_highPtId").c_str(), &highPtId );

  initBranch(tree, (name+"_mediumId").c_str(), &mediumId );
  initBranch(tree, (name+"_tightId" ).c_str(), &tightId );

  initBranch(tree, (name+"_jetIdx").c_str(), &jetIdx );
  initBranch(tree, (name+"_pfRelIso04_all").c_str(), &pfRelIso04_all );
  //initBranch(tree, (name+"_").c_str(), & );

}

std::vector<muon*> muonData::getMuons(float ptMin, float etaMax, int tag, bool isolation){

  std::vector<muon*> outputMuons;
  for(UInt_t i = 0; i < n; ++i){
    if(tag == 0 && softId[i]   == 0) continue;
    if(tag == 1 && highPtId[i] == 0) continue;
    if(tag == 2 && mediumId[i] == 0) continue;
    if(tag == 3 && tightId[i]  == 0) continue;
    if(isolation && pfRelIso04_all[i] > 0.20) continue; //working points here: https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2

    if(      pt[i] < ptMin) continue;
    if(fabs(eta[i])>etaMax) continue;

    outputMuons.push_back(new muon(i, this));
  }

  return outputMuons;
}

muonData::~muonData(){}
