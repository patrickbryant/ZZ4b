#include "TChain.h"

#include "ZZ4b/NtupleAna/interface/trigData.h"

using namespace NtupleAna;

//trig object
trig::trig(){}
trig::trig(UInt_t i, trigData* data){

  pt  = data->pt [i];
  eta = data->eta[i];
  phi = data->phi[i];
  m   = 0;
  p = TLorentzVector();
  p.SetPtEtaPhiM(pt, eta, phi, m);
  e = p.E();

  l1pt = data->l1pt[i];
  l1pt_2 = data->l1pt_2[i];
  l2pt = data->l2pt[i];
  id = data->id[i];
  //l1iso = data->l1iso[i];
  //l1charge = data->l1charge[i];
  filterBits = data->filterBits[i];
}

trig::~trig(){}


//access tree
trigData::trigData(std::string name, TChain* tree){

  initBranch(tree, ("n"+name).c_str(), n );

  initBranch(tree, (name+"_pt"  ).c_str(), pt  );  
  initBranch(tree, (name+"_eta" ).c_str(), eta );  
  initBranch(tree, (name+"_phi" ).c_str(), phi );  

  initBranch(tree, (name+"_l1pt").c_str(), l1pt );
  initBranch(tree, (name+"_l1pt_2").c_str(), l1pt_2 );
  initBranch(tree, (name+"_l2pt").c_str(), l2pt );
  initBranch(tree, (name+"_id").c_str(), id );
  initBranch(tree, (name+"_filterBits").c_str(), filterBits );
  // *Br  330 :TrigObj_l1pt : Float_t pt of associated L1 seed                    *
  // *Br  331 :TrigObj_l1pt_2 : Float_t pt of associated secondary L1 seed        *
  // *Br  332 :TrigObj_l2pt :                                                     *
  // *         | Float_t pt of associated 'L2' seed (i.e. HLT before tracking/PF) *
  // *Br  333 :TrigObj_id : Int_t ID of the object:                               *
  // *         |  11 = Electron (PixelMatched e/gamma), 22 = Photon (PixelMatch-vetoed e/gamma), 13 = Muon, 15 = Tau, 1 = Jet, 6 = FatJet, 2 = MET, 3 = HT, 4 = MHT*
  // *Br  334 :TrigObj_l1iso : Int_t iso of associated L1 seed                    *
  // *Br  335 :TrigObj_l1charge : Int_t charge of associated L1 seed              *
  // *Br  336 :TrigObj_filterBits : Int_t extra bits of associated information:   *
  // *         |  1 = CaloIdL_TrackIdL_IsoVL, 2 = 1e (WPTight), 4 = 1e (WPLoose), 8 = OverlapFilter PFTau, 16 = 2e, 32 = 1e-1mu, 64 = 1e-1tau, 128 = 3e, 256 = 2e-1mu, 512 = 1e-2mu for Electron (PixelMatched e/gamma); 1 = TrkIsoVVL, 2 = Iso, 4 = OverlapFilter PFTau, 8 = 1mu, 16 = 2mu, 32 = 1mu-1e, 64 = 1mu-1tau, 128 = 3mu, 256 = 2mu-1e, 512 =1mu-2e for Muon; 1 = LooseChargedIso, 2 = MediumChargedIso, 4 = TightChargedIso, 8 = TightID OOSC photons, 16 = HPS, 32 = single-tau + tau+MET, 64 = di-tau, 128 = e-tau, 256 = mu-tau, 512 = VBF+di-tau for Tau; 1 = VBF cross-cleaned from loose iso PFTau for Jet; *

  //initBranch(tree, (name+"_").c_str(),  );

}

std::vector<std::shared_ptr<trig>> trigData::getTrigs(float ptMin, float etaMax, int objId){
  
  std::vector< std::shared_ptr<trig> > outputTrigs;

  for(UInt_t i = 0; i < n; ++i){
    if(             pt[i]  <  ptMin ) continue;
    if(       fabs(eta[i]) > etaMax ) continue;
    if(objId > 0 && id[i] == objId  ) continue;
    outputTrigs.push_back(std::make_shared<trig>(trig(i, this)));
  }

  return outputTrigs;
}

trigData::~trigData(){}
