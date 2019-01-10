#include <iostream>

#include "ZZ4b/NtupleAna/interface/eventData.h"

using namespace NtupleAna;


eventData::eventData(TChain* t, bool d){
  tree  = t;
  debug = d;

  if(debug){
    std::cout<<"tree->Show(0)"<<std::endl;
    tree->Show(0);
  }

  tree->SetBranchAddress("run",       &run);
  tree->SetBranchAddress("event",     &event);
  tree->SetBranchAddress("genWeight", &weight);

  treeJets  = new jetData( "Jet",  tree);
  treeMuons = new muonData("Muon", tree);
} 


void eventData::update(int e){
  if(debug) std::cout<<"Reset eventData"<<std::endl;
  views  .clear();
  canJets.clear();

  if(debug) std::cout<<"Get Entry "<<e<<std::endl;
  tree->GetEntry(e);

  allJets = treeJets->getJets();
  selJets = treeJets->getJets(40, 2.5);
  tagJets = treeJets->getJets(40, 2.5, 0.4941);//medium WP 2017 from AN2018_073_v10

  allMuons = treeMuons->getMuons();
  isoMuons = treeMuons->getMuons(40, 2.5, 2, true);

  //Hack to use leptons as bJets until we get real 4b samples
  for(auto &muon: isoMuons){
    selJets.push_back(jet(muon.p, 1.0));
    tagJets.push_back(jet(muon.p, 1.0));
  }  

  return;
}


bool sortDeepCSV(jet& lhs, jet& rhs){ return (lhs.deepCSV > rhs.deepCSV); }
bool sortPt(jet& lhs, jet& rhs){ return (lhs.pt > rhs.pt); }

void eventData::chooseCanJets(){
  std::sort(tagJets.begin(), tagJets.end(), sortDeepCSV); // order by decreasing btag score
  for(int i = 0; i < 4; ++i) canJets.push_back(tagJets[i]); // take the four tagged jets with highest btag score
  std::sort(canJets.begin(), canJets.end(), sortPt); // order by decreasing pt
  std::sort(tagJets.begin(), tagJets.end(), sortPt); // order by decreasing pt
  return;
}


void eventData::buildViews(){
  views.push_back(eventView(dijet(canJets[0], canJets[1]), dijet(canJets[2], canJets[3])));
  views.push_back(eventView(dijet(canJets[0], canJets[2]), dijet(canJets[1], canJets[3])));
  views.push_back(eventView(dijet(canJets[0], canJets[3]), dijet(canJets[2], canJets[1])));
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

