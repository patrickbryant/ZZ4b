#include "ZZ4b/nTupleAnalysis/interface/tTbarEventHists.h"
#include "nTupleAnalysis/baseClasses/interface/helpers.h"

using namespace nTupleAnalysis;

tTbarEventHists::tTbarEventHists(std::string name, fwlite::TFileService& fs, bool isMC, std::string histDetailLevel, bool _debug) {
  std::cout << "Initialize >> tTbarEventHists: " << name << " with detail level: " << histDetailLevel << std::endl;

  dir = fs.mkdir(name);
  debug = _debug;

  //
  // Object Level
  //
  nAllJets = dir.make<TH1F>("nAllJets", (name+"/nAllJets; Number of Jets (no selection); Entries").c_str(),  16,-0.5,15.5);
  nSelJets = dir.make<TH1F>("nSelJets", (name+"/nSelJets; Number of Selected Jets; Entries").c_str(),  16,-0.5,15.5);
  nTagJets = dir.make<TH1F>("nTagJets", (name+"/nTagJets; Number of Tagged Jets; Entries").c_str(),  16,-0.5,15.5);

  allJets = new jetHists(name+"/allJets", fs, "All Jets");
  selJets = new jetHists(name+"/selJets", fs, "Selected Jets");
  tagJets = new jetHists(name+"/tagJets", fs, "Tagged Jets");

    
  nAllMuons = dir.make<TH1F>("nAllMuons", (name+"/nAllMuons; Number of Muons (no selection); Entries").c_str(),  6,-0.5,5.5);
  nIsoMuons = dir.make<TH1F>("nIsoMuons", (name+"/nIsoMuons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
  nIsoHighPtMuons = dir.make<TH1F>("nIsoHighPtMuons", (name+"/nIsoHighPtMuons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
  allMuons        = new muonHists(name+"/allMuons", fs, "All Muons");
  muons_iso  = new muonHists(name+"/muon_iso", fs, "iso Medium 25 Muons");
  muons_isoHighPt  = new muonHists(name+"/muon_isoHighPt", fs, "iso Medium 40 Muons");

  nAllElecs = dir.make<TH1F>("nAllElecs", (name+"/nAllElecs; Number of Elecs (no selection); Entries").c_str(),  16,-0.5,15.5);
  nIsoElecs = dir.make<TH1F>("nIsoElecs", (name+"/nIsoElecs; Number of Prompt Elecs; Entries").c_str(),  6,-0.5,5.5);
  nIsoHighPtElecs = dir.make<TH1F>("nIsoHighPtElecs", (name+"/nIsoHighPtElecs; Number of Prompt Elecs; Entries").c_str(),  6,-0.5,5.5);
  allElecs        = new elecHists(name+"/allElecs", fs, "All Elecs");
  elecs_iso  = new elecHists(name+"/elec_iso", fs, "iso Medium 25 Elecs");
  elecs_isoHighPt  = new elecHists(name+"/elec_isoHighPt", fs, "iso Medium 40 Elecs");

  nIsoLeps = dir.make<TH1F>("nIsoLeps", (name+"/nIsoLeps; Number of Prompt Leptons; Entries").c_str(),  6,-0.5,5.5);

  ChsMeT = new MeTHists(name+"/ChsMeT", fs, "Chs MeT");
  MeT    = new MeTHists(name+"/MeT",    fs, "MeT");
  TrkMeT = new MeTHists(name+"/TkMeT",  fs, "Tk MeT");


  w   = new dijetHists(name+"/w",   fs,    "W boson candidate");
  t = new trijetHists(name+"/t",  fs, "Top Candidate");

} 

void tTbarEventHists::Fill(tTbarEventData* event){

  //
  // Object Level
  //
  nAllJets->Fill(event->allJets.size(), event->weight);
  nSelJets->Fill(event->selJets.size(), event->weight);
  nTagJets->Fill(event->tagJets.size(), event->weight);

  for(auto &jet: event->allJets) allJets->Fill(jet, event->weight);
  for(auto &jet: event->selJets) selJets->Fill(jet, event->weight);
  for(auto &jet: event->tagJets) tagJets->Fill(jet, event->weight);


  nAllMuons->Fill(event->allMuons.size(), event->weight);
  nIsoMuons->Fill(event->muons_iso.size(), event->weight);
  nIsoHighPtMuons->Fill(event->muons_isoHighPt.size(), event->weight);
  for(auto &muon: event->allMuons) allMuons->Fill(muon, event->weight);
  for(auto &muon: event->muons_iso) muons_iso->Fill(muon, event->weight);
  for(auto &muon: event->muons_isoHighPt) muons_isoHighPt->Fill(muon, event->weight);

  nAllElecs->Fill(event->allElecs.size(), event->weight);
  nIsoElecs->Fill(event->elecs_iso.size(), event->weight);
  nIsoHighPtElecs->Fill(event->elecs_isoHighPt.size(), event->weight);
  for(auto &elec: event->allElecs)             allElecs->Fill(elec, event->weight);
  for(auto &elec: event->elecs_iso) elecs_iso->Fill(elec, event->weight);
  for(auto &elec: event->elecs_isoHighPt) elecs_isoHighPt->Fill(elec, event->weight);

  nIsoLeps ->Fill(event->elecs_iso.size() + event->muons_iso.size(), event->weight);

  ChsMeT -> Fill(*event->treeChsMET, event->weight);
  MeT    -> Fill(*event->treeMET,    event->weight);
  TrkMeT -> Fill(*event->treeTrkMET, event->weight);

  if(event->w)
    w -> Fill(event->w, event->weight);

  if(event->top)
    t ->Fill(event->top,  event->weight);

  return;
}


tTbarEventHists::~tTbarEventHists(){} 

