#include <iostream>

#include "ZZ4b/NtupleAna/interface/eventData.h"

using namespace NtupleAna;

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

  treeJets  = new jetData( "Jet",  tree);
  treeMuons = new muonData("Muon", tree);
} 

void eventData::update(int e){
  if(debug) std::cout<<"Reset eventData"<<std::endl;
  for(auto ptr: allJets) delete ptr;
  allJets.clear();
  for(auto ptr: selJets) delete ptr; 
  selJets.clear();
  for(auto ptr: tagJets) delete ptr; 
  tagJets.clear();
  for(auto ptr: selJets) delete ptr; 
  canJets.clear();

  for(auto ptr: dijets) delete ptr;
  dijets.clear();
  for(auto ptr: views) delete ptr;
  views.clear();

  for(auto ptr: allMuons) delete ptr;
  allMuons.clear();
  for(auto ptr: isoMuons) delete ptr;
  isoMuons.clear();

  p4j    .SetPtEtaPhiM(0,0,0,0);
  weight = 1;


  if(debug) std::cout<<"Get Entry "<<e<<std::endl;
  tree->GetEntry(e);

  //Trigger
  if(year=="2016"){
    passHLT = HLT_4j45_3b087 || HLT_2j90_2j30_3b087;
  }

  //Objects
  allJets = treeJets->getJets();
  selJets = treeJets->getJets(40, 2.5);
  //tagJets = treeJets->getJets(40, 2.5, 0.4941);//medium WP for DeepB 2017 from AN2018_073_v10
  tagJets = treeJets->getJets(40, 2.5, 0.8484);//medium WP for CSVv2 https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation80XReReco

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


bool sortDeepCSV(jet* lhs, jet* rhs){ return (lhs->deepCSV > rhs->deepCSV); }
bool sortPt(jet* lhs, jet* rhs){ return (lhs->pt > rhs->pt); }

void eventData::chooseCanJets(){

  if(threeTag){

    std::sort(selJets.begin(), selJets.end(), sortDeepCSV); // order by decreasing btag score
    for(int i = 0; i < 4; ++i) canJets.push_back(selJets[i]); // take the four jets with highest btag score    
    std::sort(selJets.begin(), selJets.end(), sortPt); // order by decreasing pt

  }else if(fourTag){

    std::sort(tagJets.begin(), tagJets.end(), sortDeepCSV); // order by decreasing btag score
    for(int i = 0; i < 4; ++i) canJets.push_back(tagJets[i]); // take the four tagged jets with highest btag score
    std::sort(tagJets.begin(), tagJets.end(), sortPt); // order by decreasing pt

  }

  std::sort(canJets.begin(), canJets.end(), sortPt); // order by decreasing pt
  p4j = (canJets[0]->p + canJets[1]->p + canJets[2]->p + canJets[3]->p);
  return;
}


void eventData::buildViews(){
  dijets.push_back(new dijet(canJets[0], canJets[1]));
  dijets.push_back(new dijet(canJets[0], canJets[2]));
  dijets.push_back(new dijet(canJets[0], canJets[3]));
  dijets.push_back(new dijet(canJets[1], canJets[2]));
  dijets.push_back(new dijet(canJets[1], canJets[3]));
  dijets.push_back(new dijet(canJets[2], canJets[3]));

  views.push_back(new eventView(dijets[0], dijets[5]));
  views.push_back(new eventView(dijets[1], dijets[4]));
  views.push_back(new eventView(dijets[2], dijets[3]));
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

