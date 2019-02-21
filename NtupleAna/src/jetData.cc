#include "TChain.h"

#include "ZZ4b/NtupleAna/interface/jetData.h"

using namespace NtupleAna;

//jet object
jet::jet(){}
jet::jet(UInt_t i, jetData* data){

  pt  = data->pt [i];
  eta = data->eta[i];
  phi = data->phi[i];
  m   = data->m  [i];
  p = TLorentzVector();
  p.SetPtEtaPhiM(pt, eta, phi, m);
  e = p.E();

  deepCSV = data->deepCSV[i];
  CSVv2   = data->CSVv2[i];

}

jet::jet(TLorentzVector& vec, float tag){

  p = TLorentzVector(vec);
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();

  deepCSV = tag;
  CSVv2   = tag;

}

jet::~jet(){}


//access tree
jetData::jetData(std::string name, TChain* tree){

  initBranch(tree, ("n"+name).c_str(), &n );

  initBranch(tree, (name+"_pt"  ).c_str(), &pt );  
  initBranch(tree, (name+"_eta" ).c_str(), &eta );  
  initBranch(tree, (name+"_phi" ).c_str(), &phi );  
  initBranch(tree, (name+"_mass").c_str(), &m );  

  initBranch(tree, (name+"_btagDeepB").c_str(), &deepCSV );
  initBranch(tree, (name+"_btagCSVV2").c_str(), &CSVv2   );
  //initBranch(tree, (name+"_").c_str(). & );

}

std::vector<std::shared_ptr<jet>> jetData::getJets(float ptMin, float etaMax, float tagMin, std::string tagger){

  std::vector<std::shared_ptr<jet>> outputJets;
  for(UInt_t i = 0; i < n; ++i){
    if(      pt[i] < ptMin) continue;
    if(fabs(eta[i])>etaMax) continue;
    if( deepCSV[i] <tagMin && tagger == "deepB") continue;
    if(   CSVv2[i] <tagMin && tagger == "CSVv2") continue;
    outputJets.push_back(std::make_shared<jet>(jet(i, this)));
  }

  return outputJets;
}

jetData::~jetData(){}
