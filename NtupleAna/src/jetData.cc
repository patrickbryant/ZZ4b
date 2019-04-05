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

  bRegCorr = data->bRegCorr[i];

  deepB     = data->deepB[i];
  CSVv2     = data->CSVv2[i];
  deepFlavB = data->deepFlavB[i];

}

jet::jet(TLorentzVector& vec, float tag){

  p = TLorentzVector(vec);
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();

  bRegCorr = pt;

  deepB = tag;
  CSVv2 = tag;
  deepFlavB = tag;

}

void jet::bRegression(){
  p  *= bRegCorr;
  pt  = p.Pt();
  eta = p.Eta();
  phi = p.Phi();
  m   = p.M();
  e   = p.E();
}

jet::~jet(){}


//access tree
jetData::jetData(std::string name, TChain* tree){

  initBranch(tree, ("n"+name).c_str(), n );

  initBranch(tree, (name+"_pt"  ).c_str(), pt  );  
  initBranch(tree, (name+"_eta" ).c_str(), eta );  
  initBranch(tree, (name+"_phi" ).c_str(), phi );  
  initBranch(tree, (name+"_mass").c_str(), m   );  

  initBranch(tree, (name+"_bRegCorr").c_str(), bRegCorr );  

  initBranch(tree, (name+"_btagDeepB"    ).c_str(), deepB     );
  initBranch(tree, (name+"_btagCSVV2"    ).c_str(), CSVv2     );
  initBranch(tree, (name+"_btagDeepFlavB").c_str(), deepFlavB );
  //initBranch(tree, (name+"_").c_str(),  );

}

std::vector<std::shared_ptr<jet>> jetData::getJets(float ptMin, float etaMax, float tagMin, std::string tagger, bool antiTag){
  
  std::vector< std::shared_ptr<jet> > outputJets;
  float *tag = CSVv2;
  if(tagger == "deepB")     tag = deepB;
  if(tagger == "deepFlavB") tag = deepFlavB;

  for(UInt_t i = 0; i < n; ++i){
    if(          pt[i]  <  ptMin ) continue;
    if(    fabs(eta[i]) > etaMax ) continue;
    if(antiTag^(tag[i]  < tagMin)) continue; // antiTag XOR (jet fails tagMin). This way antiTag inverts the tag criteria to select untagged jets
    //if( deepB[i] <tagMin && tagger == "deepB") continue;
    //if( CSVv2[i] <tagMin && tagger == "CSVv2") continue;
    outputJets.push_back(std::make_shared<jet>(jet(i, this)));
  }

  return outputJets;
}

jetData::~jetData(){}
