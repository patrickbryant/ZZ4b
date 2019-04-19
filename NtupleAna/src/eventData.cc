#include "ZZ4b/NtupleAna/interface/eventData.h"

using namespace NtupleAna;

// Sorting functions
bool sortPt(       std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->pt        > rhs->pt   );     } // put largest  pt    first in list
bool sortdR(       std::shared_ptr<dijet>     &lhs, std::shared_ptr<dijet>     &rhs){ return (lhs->dR        < rhs->dR   );     } // 
bool sortDBB(      std::unique_ptr<eventView> &lhs, std::unique_ptr<eventView> &rhs){ return (lhs->dBB       < rhs->dBB  );     } // put smallest dBB   first in list
bool sortDeepB(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepB     > rhs->deepB);     } // put largest  deepB first in list
bool sortCSVv2(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->CSVv2     > rhs->CSVv2);     } // put largest  CSVv2 first in list
bool sortDeepFlavB(std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepFlavB > rhs->deepFlavB); } // put largest  deepB first in list

eventData::eventData(TChain* t, bool mc, std::string y, bool d){
  tree  = t;
  isMC  = mc;
  year  = y;
  debug = d;
  random = new TRandom3();

  tree->LoadTree(0);
  initBranch(tree, "run",             run);
  initBranch(tree, "luminosityBlock", lumiBlock);
  initBranch(tree, "event",           event);
  if(tree->FindBranch("nTagClassifier")){
    std::cout << "Tree has nTagClassifier" << std::endl;
    initBranch(tree, "nTagClassifier", nTagClassifier);
  }
  if(tree->FindBranch("ZHvsBackgroundClassifier")){
    std::cout << "Tree has ZHvsBackgroundClassifier" << std::endl;
    initBranch(tree, "ZHvsBackgroundClassifier", ZHvsBackgroundClassifier);
  }
  if(isMC){
    initBranch(tree, "genWeight", genWeight);
    truth = new truthData(tree, debug);
  }

  //triggers
  //trigObjs = new trigData("TrigObj", tree);
  if(year=="2016"){
    initBranch(tree, "HLT_QuadJet45_TripleBTagCSV_p087",            HLT_4j45_3b087);
    initBranch(tree, "HLT_DoubleJet90_Double30_TripleBTagCSV_p087", HLT_2j90_2j30_3b087);
  }
  if(year=="2018"){
    initBranch(tree, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", HLT_HT330_4j_75_60_45_40_3b);
    initBranch(tree, "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",    HLT_4j_103_88_75_15_2b_VBF1);
    initBranch(tree, "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",              HLT_4j_103_88_75_15_1b_VBF2);
    initBranch(tree, "HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71",       HLT_2j116_dEta1p6_2b);
    initBranch(tree, "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02",            HLT_J330_m30_2b);
    initBranch(tree, "HLT_PFJet500",            HLT_j500);
    initBranch(tree, "HLT_DiPFJetAve300_HFJEC", HLT_2j300ave);
    //                            HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1_v
    //                            HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1_v
    //                            HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2_v
    //                            HLT_QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2_v
    //                            HLT_QuadPFJet111_90_80_15_PFBTagDeepCSV_1p3_VBF2_v
    // HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v
    // HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v
    // HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02_v
  }

  treeJets  = new jetData( "Jet",  tree);
  treeMuons = new muonData("Muon", tree);
} 

//Set bTagging and sorting function
void eventData::setTagger(std::string tagger, float tag){
  bTagger = tagger;
  bTag    = tag;
  if(bTagger == "deepB")
    sortTag = sortDeepB;
  if(bTagger == "CSVv2")
    sortTag = sortCSVv2;
  if(bTagger == "deepFlavB")
    sortTag = sortDeepFlavB;
}

void eventData::update(int e){
  //if(e>2546040) debug = true;
  if(debug) std::cout<<"Reset eventData"<<std::endl;
  canJets.clear();
  dijets .clear();
  views  .clear();
  ZHSB = false; ZHCR = false; ZHSR = false;
  leadStM = -99; sublStM = -99;
  passDEtaBB = false;
  p4j    .SetPtEtaPhiM(0,0,0,0);
  canJet1_pt = -99;
  canJet3_pt = -99;
  aveAbsEta = -99;
  dRjjClose = -99;
  dRjjOther = -99;
  nPseudoTags = 0;
  pseudoTagWeight = 1;
  weight = 1;
  reweight = 1;
  xWt0 = 1e6; xWt1 = 1e6;

  if(debug){
    std::cout<<"Get Entry "<<e<<std::endl;
    std::cout<<tree->GetCurrentFile()->GetName()<<std::endl;
    tree->Show(e);
  }
  tree->GetEntry(e);
  if(debug) std::cout<<"Got Entry "<<e<<std::endl;

  if(isMC) truth->update();

  //Trigger
  if(year=="2016"){
    passHLT = HLT_4j45_3b087 || HLT_2j90_2j30_3b087;
  }
  if(year=="2018"){
    passHLT = HLT_HT330_4j_75_60_45_40_3b || HLT_4j_103_88_75_15_2b_VBF1 || HLT_4j_103_88_75_15_1b_VBF2 || HLT_2j90_2j30_3b087 || HLT_J330_m30_2b || HLT_j500 || HLT_2j300ave;
  }

  //Objects
  if(debug) std::cout << "Get Jets\n";
  allJets = treeJets->getJets();
  selJets = treeJets->getJets(40, 2.5);
  tagJets = treeJets->getJets(40, 2.5, bTag, bTagger);
  antiTag = treeJets->getJets(40, 2.5, bTag, bTagger, true); //boolean specifies antiTag=true, inverts tagging criteria
  nSelJets = selJets.size();
  
  if(debug) std::cout << "Get Muons\n";
  allMuons = treeMuons->getMuons();
  isoMuons = treeMuons->getMuons(40, 2.5, 2, true);

  //Hack to use leptons as bJets
  // for(auto &muon: isoMuons){
  //   selJets.push_back(new jet(muon->p, 1.0));
  //   tagJets.push_back(new jet(muon->p, 1.0));
  // }  

  nTagJets = tagJets.size();
  threeTag = (nTagJets == 3);
  fourTag  = (nTagJets >= 4);

  if(debug) std::cout<<"eventData updated\n";
  return;
}


void eventData::computePseudoTagWeight(){
  unsigned int nUntagged = antiTag.size();

  //First compute the probability to have n pseudoTags where n \in {0, ..., nUntagged Jets}
  float nPseudoTagProb[nUntagged+1];
  for(unsigned int i=0; i<=nUntagged; i++){
    float Cnk = boost::math::binomial_coefficient<float>(nUntagged, i);
    nPseudoTagProb[i] = Cnk * pow(pseudoTagProb, i) * pow( (1-pseudoTagProb), (nUntagged - i) ); //i pseudo tags and nUntagged-i pseudo untags
  }

  float nPseudoTagProbSum = std::accumulate(nPseudoTagProb, nPseudoTagProb+nUntagged+1, 0.0);
  if( fabs(nPseudoTagProbSum - 1.0) > 0.00001) std::cout << "Error: nPseudoTagProbSum - 1 = " << nPseudoTagProbSum - 1.0 << std::endl;

  pseudoTagWeight = std::accumulate(nPseudoTagProb+1, nPseudoTagProb+nUntagged+1, 0.0);
  // it seems a three parameter njet model is needed. 
  // Possibly a trigger effect? ttbar?
  if(selJets.size()==4){ 
    pseudoTagWeight *= fourJetScale;
  }else{
    pseudoTagWeight *= moreJetScale;
  }
  
  // Now pick nPseudoTags randomly by choosing a random number in the set (nPseudoTagProb[0], 1)
  nPseudoTags = 0;
  float cummulativeProb = 0;
  float randomProb = random->Uniform(nPseudoTagProb[0], 1.0);
  for(unsigned int i=0; i<nUntagged+1; i++){
    //keep track of the total pseudoTagProb for at least i pseudoTags
    cummulativeProb += nPseudoTagProb[i];

    //Wait until cummulativeProb >= randomProb
    if(cummulativeProb < randomProb) continue;
    //When cummulativeProb exceeds randomProb, we have found our pseudoTag selection

    //nPseudoTags+nTagJets should model the true number of b-tags in the fourTag data
    nPseudoTags = i;

    // update the event weight
    weight *= pseudoTagWeight;
    return;
  }
  
  std::cout << "Error: Did not find a valid pseudoTag assignment" << std::endl;
  return;
}


void eventData::chooseCanJets(){
  if(debug) std::cout<<"chooseCanJets()\n";
  if(threeTag){
    if(pseudoTagProb > 0) computePseudoTagWeight();
    // order by decreasing btag score
    std::sort(selJets.begin(), selJets.end(), sortTag);
    // take the four jets with highest btag score    
    for(int i = 0; i < 4; ++i) canJets.push_back(selJets[i]);
    // order by decreasing pt
    std::sort(selJets.begin(), selJets.end(), sortPt); 

  }else if(fourTag){

    // order by decreasing btag score
    std::sort(tagJets.begin(), tagJets.end(), sortTag);
    // take the four tagged jets with highest btag score
    for(int i = 0; i < 4; ++i) canJets.push_back(tagJets[i]);
    // order by decreasing pt
    std::sort(tagJets.begin(), tagJets.end(), sortPt); 

  }

  //apply bjet pt regression to candidate jets
  for(auto &jet: canJets) jet->bRegression();

  std::sort(canJets.begin(), canJets.end(), sortPt); // order by decreasing pt
  p4j = (canJets[0]->p + canJets[1]->p + canJets[2]->p + canJets[3]->p);
  m4j = p4j.M();

  //flat nTuple variables for neural network inputs
  aveAbsEta = (fabs(canJets[0]->eta) + fabs(canJets[1]->eta) + fabs(canJets[2]->eta) + fabs(canJets[3]->eta))/4;
  canJet0_pt  = canJets[0]->pt ; canJet1_pt  = canJets[1]->pt ; canJet2_pt  = canJets[2]->pt ; canJet3_pt  = canJets[3]->pt ;
  canJet0_eta = canJets[0]->eta; canJet1_eta = canJets[1]->eta; canJet2_eta = canJets[2]->eta; canJet3_eta = canJets[3]->eta;
  canJet0_phi = canJets[0]->phi; canJet1_phi = canJets[1]->phi; canJet2_phi = canJets[2]->phi; canJet3_phi = canJets[3]->phi;
  canJet0_e   = canJets[0]->e  ; canJet1_e   = canJets[1]->e  ; canJet2_e   = canJets[2]->e  ; canJet3_e   = canJets[3]->e  ;
  return;
}


void eventData::buildViews(){
  if(debug) std::cout<<"buildViews()\n";
  //construct all dijets from the four canJets. 
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[1])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[2], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[2])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[2])));

  //Find dijet with smallest dR and other dijet
  close = *std::min_element(dijets.begin(), dijets.end(), sortdR);
  int closeIdx = std::distance(dijets.begin(), std::find(dijets.begin(), dijets.end(), close));
  //Index of the dijet made from the other two jets is either the one before or one after because of how we constructed the dijets vector
  //if closeIdx is even, add one, if closeIdx is odd, subtract one.
  int otherIdx = (closeIdx%2)==0 ? closeIdx + 1 : closeIdx - 1; 
  other = dijets[otherIdx];

  //flat nTuple variables for neural network inputs
  dRjjClose = close->dR;
  dRjjOther = other->dR;

  views.push_back(std::make_unique<eventView>(eventView(dijets[0], dijets[1])));
  views.push_back(std::make_unique<eventView>(eventView(dijets[2], dijets[3])));
  views.push_back(std::make_unique<eventView>(eventView(dijets[4], dijets[5])));

  std::sort(views.begin(), views.end(), sortDBB);
  return;
}


bool failMDRs(std::unique_ptr<eventView> &view){ return !view->passMDRs; }

void eventData::applyMDRs(){
  views.erase(std::remove_if(views.begin(), views.end(), failMDRs), views.end());
  passMDRs = (views.size() > 0);
  if(passMDRs){
    ZHSB = views[0]->ZHSB; ZHCR = views[0]->ZHCR; ZHSR = views[0]->ZHSR;
    leadStM = views[0]->leadSt->m; sublStM = views[0]->sublSt->m;
    passDEtaBB = views[0]->passDEtaBB;
  }
  return;
}

void eventData::buildTops(){
  float mW; float mt; float xWt;
  for(auto &b: canJets){
    for(auto &j1: selJets){
      if(b->p.DeltaR(j1->p) < 0.3) continue; // ensure all three jets are different
      for(auto &j2: selJets){
	if(b ->p.DeltaR(j2->p) < 0.3) continue; // ensure all three jets are different
	if(j1->p.DeltaR(j2->p) < 0.3) continue; // ensure all three jets are different
	if(j1->pt < j2->pt) continue; // prevent double counting by only considering W pairs where j1 is leading jet
	mW  =        (j1->p + j2->p).M();
	mt  = (b->p + j1->p + j2->p).M();
	xWt = pow( pow((mW-80)/(0.1*mW),2)+pow((mt-173)/(0.1*mt),2) , 0.5);
	if(xWt < xWt0){
	  xWt1 = xWt0;
	  xWt0 = xWt;
	}
	else if(xWt < xWt1){
	  xWt1 = xWt;
	}
      }
    }
  }
  
}

void eventData::dump(){

  std::cout << "   Run: " << run    << std::endl;
  std::cout << " Event: " << event  << std::endl;  
  std::cout << "Weight: " << weight << std::endl;
  std::cout << " allJets: " << allJets .size() << " |  selJets: " << selJets .size() << " | tagJets: " << tagJets.size() << std::endl;
  std::cout << "allMuons: " << allMuons.size() << " | isoMuons: " << isoMuons.size() << std::endl;

  return;
}

eventData::~eventData(){} 

