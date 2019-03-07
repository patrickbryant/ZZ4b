#include "ZZ4b/NtupleAna/interface/viewHists.h"

using namespace NtupleAna;

viewHists::viewHists(std::string name, fwlite::TFileService& fs) {
  dir = fs.mkdir(name);

  //
  // Object Level
  //
  //
  // Object Level
  //
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

  lead   = new dijetHists(name+"/lead",   fs,    "Leading p_{T} boson candidate");
  subl   = new dijetHists(name+"/subl",   fs, "Subleading p_{T} boson candidate");
  lead_m_vs_subl_m = dir.make<TH2F>("lead_m_vs_subl_m", (name+"/lead_m_vs_subl_m; p_{T} leading boson candidate Mass [GeV]; p_{T} subleading boson candidate Mass [GeV]; Entries").c_str(), 50,0,250, 50,0,250);

  leadSt = new dijetHists(name+"/leadSt", fs,    "Leading S_{T} boson candidate");
  sublSt = new dijetHists(name+"/sublSt", fs, "Subleading S_{T} boson candidate");
  leadSt_m_vs_sublSt_m = dir.make<TH2F>("leadSt_m_vs_sublSt_m", (name+"/leadSt_m_vs_sublSt_m; S_{T} leading boson candidate Mass [GeV]; S_{T} subleading boson candidate Mass [GeV]; Entries").c_str(), 50,0,250, 50,0,250);

  leadM  = new dijetHists(name+"/leadM",  fs,    "Leading mass boson candidate");
  sublM  = new dijetHists(name+"/sublM",  fs, "Subleading mass boson candidate");
  leadM_m_vs_sublM_m = dir.make<TH2F>("leadM_m_vs_sublM_m", (name+"/leadM_m_vs_sublM_m; mass leading boson candidate Mass [GeV]; mass subleading boson candidate Mass [GeV]; Entries").c_str(), 50,0,250, 50,0,250);
    
  //
  // Event  Level
  //
  v4j = new vecHists(name+"/v4j", fs, "4j");
  dBB = dir.make<TH1F>("dBB", (name+"/dBB; D_{BB}; Entries").c_str(), 40, 0, 200);
  xZZ = dir.make<TH1F>("xZZ", (name+"/xZZ; X_{ZZ}; Entries").c_str(), 100, 0, 10);
  Double_t bins_mZZ[] = {100, 182, 200, 220, 242, 266, 292, 321, 353, 388, 426, 468, 514, 565, 621, 683, 751, 826, 908, 998, 1097, 1206, 1326, 1500};
  mZZ = dir.make<TH1F>("mZZ", (name+"/mZZ; m_{ZZ} [GeV]; Entries").c_str(), 23, bins_mZZ);
  xZH = dir.make<TH1F>("xZH", (name+"/xZH; X_{ZH}; Entries").c_str(), 100, 0, 10);  
  Double_t bins_mZH[] = {100, 216, 237, 260, 286, 314, 345, 379, 416, 457, 502, 552, 607, 667, 733, 806, 886, 974, 1071, 1178, 1295, 1500};
  mZH = dir.make<TH1F>("mZH", (name+"/mZH; m_{ZH} [GeV]; Entries").c_str(), 21, bins_mZH);
} 

void viewHists::Fill(eventData* event, std::unique_ptr<eventView> &view){
  //
  // Object Level
  //
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

  lead  ->Fill(view->lead,   event->weight);
  subl  ->Fill(view->subl,   event->weight);
  lead_m_vs_subl_m->Fill(view->lead->m, view->subl->m, event->weight);

  leadSt->Fill(view->leadSt, event->weight);
  sublSt->Fill(view->sublSt, event->weight);
  leadSt_m_vs_sublSt_m->Fill(view->leadSt->m, view->sublSt->m, event->weight);

  leadM ->Fill(view->leadM,  event->weight);
  sublM ->Fill(view->sublM,  event->weight);
  leadM_m_vs_sublM_m->Fill(view->leadM->m, view->sublM->m, event->weight);

  //
  // Event Level
  //
  v4j->Fill(view->p, event->weight);
  dBB->Fill(view->dBB, event->weight);
  xZZ->Fill(view->xZZ, event->weight);
  mZZ->Fill(view->mZZ, event->weight);
  xZH->Fill(view->xZH, event->weight);
  mZH->Fill(view->mZH, event->weight);

  return;
}

viewHists::~viewHists(){} 

