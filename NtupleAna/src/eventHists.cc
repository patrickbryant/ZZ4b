#include "ZZ4b/NtupleAna/interface/eventHists.h"

using namespace NtupleAna;

eventHists::eventHists(std::string name, fwlite::TFileService& fs, bool _doViews, bool blind) {
  std::cout << "Initialize >> eventHists: " << name << std::endl;
  doViews = _doViews;
  dir = fs.mkdir(name);

  //
  // Object Level
  //
  if(!doViews){//these variables will be filled in the inclusive massplane hists if event views are defined
    nAllJets = dir.make<TH1F>("nAllJets", (name+"/nAllJets; Number of Jets (no selection); Entries").c_str(),  16,-0.5,15.5);
    nSelJets = dir.make<TH1F>("nSelJets", (name+"/nSelJets; Number of Selected Jets; Entries").c_str(),  16,-0.5,15.5);
    nTagJets = dir.make<TH1F>("nTagJets", (name+"/nTagJets; Number of Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
    nCanJets = dir.make<TH1F>("nCanJets", (name+"/nCanJets; Number of Boson Candidate Jets; Entries").c_str(),  16,-0.5,15.5);
    allJets = new jetHists(name+"/allJets", fs, "All Jets");
    selJets = new jetHists(name+"/selJets", fs, "Selected Jets");
    tagJets = new jetHists(name+"/tagJets", fs, "Tagged Jets");
    canJets = new jetHists(name+"/canJets", fs, "Boson Candidate Jets");
    
    nAllMuons = dir.make<TH1F>("nAllMuons", (name+"/nAllMuons; Number of Muons (no selection); Entries").c_str(),  6,-0.5,5.5);
    nIsoMuons = dir.make<TH1F>("nIsoMuons", (name+"/nIsoMuons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
    allMuons = new muonHists(name+"/allMuons", fs, "All Muons");
    isoMuons = new muonHists(name+"/isoMuons", fs, "Prompt Muons");

    v4j = new vecHists(name+"/v4j", fs, "4j");
  }

  //
  // Event  Level
  //
  if(doViews){
    allViews = new massRegionHists(name+"/allViews", fs, blind);
    mainView = new massRegionHists(name+"/mainView", fs, blind);
  }
} 

void eventHists::Fill(eventData* event){
  //
  // Object Level
  //
  if(!doViews){//these variables will be filled in the inclusive massplane hists if event views are defined
    nAllJets->Fill(event->allJets.size(), event->weight);
    nSelJets->Fill(event->selJets.size(), event->weight);
    nTagJets->Fill(event->tagJets.size(), event->weight);
    nCanJets->Fill(event->canJets.size(), event->weight);
    for(auto &jet: event->allJets) allJets->Fill(jet, event->weight);
    for(auto &jet: event->selJets) selJets->Fill(jet, event->weight);
    for(auto &jet: event->tagJets) tagJets->Fill(jet, event->weight);
    for(auto &jet: event->canJets) canJets->Fill(jet, event->weight);

    nAllMuons->Fill(event->allMuons.size(), event->weight);
    nIsoMuons->Fill(event->isoMuons.size(), event->weight);
    for(auto &muon: event->allMuons) allMuons->Fill(muon, event->weight);
    for(auto &muon: event->isoMuons) isoMuons->Fill(muon, event->weight);

    v4j->Fill(event->p4j, event->weight);
  }

  return;
}

void eventHists::Fill(eventData* event, std::vector<std::unique_ptr<eventView>> &views){
  // Basic Fill
  this->Fill(event);
    
  // View Fills
  mainView->Fill(event, views[0]);
  for(auto &view: views) 
    allViews->Fill(event, view);
  
  return;
}

eventHists::~eventHists(){} 

