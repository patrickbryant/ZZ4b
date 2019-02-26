#include <iostream>

#include "ZZ4b/NtupleAna/interface/eventData.h"

using namespace NtupleAna;

// Sorting functions
bool sortPt(   std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->pt    > rhs->pt   ); } // put largest  pt    first in list
bool sortDBB(  std::unique_ptr<eventView> &lhs, std::unique_ptr<eventView> &rhs){ return (lhs->dBB   < rhs->dBB  ); } // put smallest dBB   first in list
bool sortDeepB(std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepB > rhs->deepB); } // put largest  deepB first in list
bool sortCSVv2(std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->CSVv2 > rhs->CSVv2); } // put largest  CSVv2 first in list


eventData::eventData(TChain* t, bool mc, std::string y, bool d){
  tree  = t;
  isMC  = mc;
  year  = y;
  debug = d;

  if(debug){
    std::cout<<"tree->Show(0)"<<std::endl;
    tree->Show(0);
  }
  
  initBranch(tree, "run",             &run);
  initBranch(tree, "luminosityBlock", &lumiBlock);
  initBranch(tree, "event",           &event);
  if(isMC){
    initBranch(tree, "genWeight", &genWeight);
  }
  if(year=="2016"){
    initBranch(tree, "HLT_QuadJet45_TripleBTagCSV_p087",            &HLT_4j45_3b087);
    initBranch(tree, "HLT_DoubleJet90_Double30_TripleBTagCSV_p087", &HLT_2j90_2j30_3b087);
  }
  if(year=="2018"){
    initBranch(tree, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", &HLT_HT330_4j_75_60_45_40_3b4p5);
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
}

void eventData::update(int e){
  if(debug) std::cout<<"Reset eventData"<<std::endl;
  canJets.clear();
  dijets .clear();
  views  .clear();
  p4j    .SetPtEtaPhiM(0,0,0,0);
  weight = 1;

  if(debug) std::cout<<"Get Entry "<<e<<std::endl;
  tree->GetEntry(e);

  //Trigger
  if(year=="2016"){
    passHLT = HLT_4j45_3b087 || HLT_2j90_2j30_3b087;
  }
  if(year=="2018"){
    passHLT = HLT_HT330_4j_75_60_45_40_3b4p5;
  }

  //Objects
  allJets = treeJets->getJets();
  selJets = treeJets->getJets(40, 2.5);
  tagJets = treeJets->getJets(40, 2.5, bTag, bTagger);

  allMuons = treeMuons->getMuons();
  isoMuons = treeMuons->getMuons(40, 2.5, 2, true);

  //Hack to use leptons as bJets
  // for(auto &muon: isoMuons){
  //   selJets.push_back(new jet(muon->p, 1.0));
  //   tagJets.push_back(new jet(muon->p, 1.0));
  // }  

  nTags    = tagJets.size();
  threeTag = (nTags == 3);
  fourTag  = (nTags >= 4);

  return;
}


void eventData::chooseCanJets(){

  if(threeTag){

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

  std::sort(canJets.begin(), canJets.end(), sortPt); // order by decreasing pt
  p4j = (canJets[0]->p + canJets[1]->p + canJets[2]->p + canJets[3]->p);
  return;
}


void eventData::buildViews(){
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[1])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[2])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[2])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[2], canJets[3])));

  views.push_back(std::make_unique<eventView>(eventView(dijets[0], dijets[5])));
  views.push_back(std::make_unique<eventView>(eventView(dijets[1], dijets[4])));
  views.push_back(std::make_unique<eventView>(eventView(dijets[2], dijets[3])));

  std::sort(views.begin(), views.end(), sortDBB);
  return;
}


bool failMDRs(std::unique_ptr<eventView> &view){ return !view->passMDRs; }

void eventData::applyMDRs(){
  views.erase(std::remove_if(views.begin(), views.end(), failMDRs), views.end());
  passMDRs = (views.size() > 0);
  return;
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

