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
  nIsoMed25Muons = dir.make<TH1F>("nIsoMed25Muons", (name+"/nIsoMed25Muons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
  nIsoMed40Muons = dir.make<TH1F>("nIsoMed40Muons", (name+"/nIsoMed40Muons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
  allMuons        = new muonHists(name+"/allMuons", fs, "All Muons");
  muons_isoMed25  = new muonHists(name+"/isoMed25", fs, "iso Medium 25 Muons");
  muons_isoMed40  = new muonHists(name+"/isoMed40", fs, "iso Medium 40 Muons");

  nAllElecs = dir.make<TH1F>("nAllElecs", (name+"/nAllElecs; Number of Elecs (no selection); Entries").c_str(),  16,-0.5,15.5);
  nIsoMed25Elecs = dir.make<TH1F>("nIsoMed25Elecs", (name+"/nIsoMed25Elecs; Number of Prompt Elecs; Entries").c_str(),  6,-0.5,5.5);
  nIsoMed40Elecs = dir.make<TH1F>("nIsoMed40Elecs", (name+"/nIsoMed40Elecs; Number of Prompt Elecs; Entries").c_str(),  6,-0.5,5.5);
  allElecs        = new elecHists(name+"/allElecs", fs, "All Elecs");
  elecs_isoMed25  = new elecHists(name+"/elec_isoMed25", fs, "iso Medium 25 Elecs");
  elecs_isoMed40  = new elecHists(name+"/elec_isoMed40", fs, "iso Medium 40 Elecs");

  nIsoMed25Leps = dir.make<TH1F>("nIsoMed25Leps", (name+"/nIsoMed25Leps; Number of Prompt Leptons; Entries").c_str(),  6,-0.5,5.5);

  ChsMeT = new MeTHists(name+"/ChsMeT", fs, "Chs MeT");
  MeT    = new MeTHists(name+"/MeT",    fs, "MeT");
  TrkMeT = new MeTHists(name+"/TkMeT",  fs, "Tk MeT");


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
  nIsoMed25Muons->Fill(event->muons_isoMed25.size(), event->weight);
  nIsoMed40Muons->Fill(event->muons_isoMed40.size(), event->weight);
  for(auto &muon: event->allMuons) allMuons->Fill(muon, event->weight);
  for(auto &muon: event->muons_isoMed25) muons_isoMed25->Fill(muon, event->weight);
  for(auto &muon: event->muons_isoMed40) muons_isoMed40->Fill(muon, event->weight);

  nAllElecs->Fill(event->allElecs.size(), event->weight);
  nIsoMed25Elecs->Fill(event->elecs_isoMed25.size(), event->weight);
  nIsoMed40Elecs->Fill(event->elecs_isoMed40.size(), event->weight);
  for(auto &elec: event->allElecs)             allElecs->Fill(elec, event->weight);
  for(auto &elec: event->elecs_isoMed25) elecs_isoMed25->Fill(elec, event->weight);
  for(auto &elec: event->elecs_isoMed40) elecs_isoMed40->Fill(elec, event->weight);

  nIsoMed25Leps ->Fill(event->elecs_isoMed25.size() + event->muons_isoMed25.size(), event->weight);

  ChsMeT -> Fill(*event->treeChsMET, event->weight);
  MeT    -> Fill(*event->treeMET,    event->weight);
  TrkMeT -> Fill(*event->treeTrkMET, event->weight);
  return;
}


tTbarEventHists::~tTbarEventHists(){} 

