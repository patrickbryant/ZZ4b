#include "ZZ4b/NtupleAna/interface/viewHists.h"

using namespace NtupleAna;

viewHists::viewHists(std::string name, fwlite::TFileService& fs, bool isMC) {
  dir = fs.mkdir(name);

  //
  // Object Level
  //
  nAllJets = dir.make<TH1F>("nAllJets", (name+"/nAllJets; Number of Jets (no selection); Entries").c_str(),  16,-0.5,15.5);
  nSelJets = dir.make<TH1F>("nSelJets", (name+"/nSelJets; Number of Selected Jets; Entries").c_str(),  16,-0.5,15.5);
  nSelJets_lowSt = dir.make<TH1F>("nSelJets_lowSt", (name+"/nSelJets_lowSt; Number of Selected Jets; Entries").c_str(),  16,-0.5,15.5);
  nSelJets_midSt = dir.make<TH1F>("nSelJets_midSt", (name+"/nSelJets_midSt; Number of Selected Jets; Entries").c_str(),  16,-0.5,15.5);
  nSelJets_highSt = dir.make<TH1F>("nSelJets_highSt", (name+"/nSelJets_highSt; Number of Selected Jets; Entries").c_str(),  16,-0.5,15.5);
  nSelJetsUnweighted = dir.make<TH1F>("nSelJetsUnweighted", (name+"/nSelJetsUnweighted; Number of Selected Jets (Unweighted); Entries").c_str(),  16,-0.5,15.5);
  nSelJetsUnweighted_lowSt = dir.make<TH1F>("nSelJetsUnweighted_lowSt", (name+"/nSelJetsUnweighted_lowSt; Number of Selected (Unweighted) Jets; Entries").c_str(),  16,-0.5,15.5);
  nSelJetsUnweighted_midSt = dir.make<TH1F>("nSelJetsUnweighted_midSt", (name+"/nSelJetsUnweighted_midSt; Number of Selected (Unweighted) Jets; Entries").c_str(),  16,-0.5,15.5);
  nSelJetsUnweighted_highSt = dir.make<TH1F>("nSelJetsUnweighted_highSt", (name+"/nSelJetsUnweighted_highSt; Number of Selected (Unweighted) Jets; Entries").c_str(),  16,-0.5,15.5);
  nTagJets = dir.make<TH1F>("nTagJets", (name+"/nTagJets; Number of Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
  nPSTJets = dir.make<TH1F>("nPSTJets", (name+"/nPSTJets; Number of Tagged + Pseudo-Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
  nPSTJets_lowSt = dir.make<TH1F>("nPSTJets_lowSt", (name+"/nPSTJets_lowSt; Number of Tagged + Pseudo-Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
  nPSTJets_midSt = dir.make<TH1F>("nPSTJets_midSt", (name+"/nPSTJets_midSt; Number of Tagged + Pseudo-Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
  nPSTJets_highSt = dir.make<TH1F>("nPSTJets_highSt", (name+"/nPSTJets_highSt; Number of Tagged + Pseudo-Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
  nCanJets = dir.make<TH1F>("nCanJets", (name+"/nCanJets; Number of Boson Candidate Jets; Entries").c_str(),  16,-0.5,15.5);
  allJets = new jetHists(name+"/allJets", fs, "All Jets");
  selJets = new jetHists(name+"/selJets", fs, "Selected Jets");
  tagJets = new jetHists(name+"/tagJets", fs, "Tagged Jets");
  canJets = new jetHists(name+"/canJets", fs, "Boson Candidate Jets");
  canJet0 = new jetHists(name+"/canJet0", fs, "Boson Candidate Jet_{0}");
  canJet1 = new jetHists(name+"/canJet1", fs, "Boson Candidate Jet_{1}");
  canJet2 = new jetHists(name+"/canJet2", fs, "Boson Candidate Jet_{2}");
  canJet3 = new jetHists(name+"/canJet3", fs, "Boson Candidate Jet_{3}");
  othJets = new jetHists(name+"/othJets", fs, "Other Selected Jets");
  aveAbsEta = dir.make<TH1F>("aveAbsEta", (name+"/aveAbsEta; <|#eta|>; Entries").c_str(), 25, 0 , 2.5);
  aveAbsEtaOth = dir.make<TH1F>("aveAbsEtaOth", (name+"/aveAbsEtaOth; Other Jets <|#eta|>; Entries").c_str(), 25, 0 , 2.5);
    
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

  close  = new dijetHists(name+"/close",  fs,               "Minimum #DeltaR(j,j) Dijet");
  other  = new dijetHists(name+"/other",  fs, "Complement of Minimum #DeltaR(j,j) Dijet");
  close_m_vs_other_m = dir.make<TH2F>("close_m_vs_other_m", (name+"/close_m_vs_other_m; Minimum #DeltaR(j,j) Dijet Mass [GeV]; Complement of Minimum #DeltaR(j,j) Dijet Mass [GeV]; Entries").c_str(), 50,0,250, 50,0,250);
    
  //
  // Event  Level
  //
  nPVs = dir.make<TH1F>("nPVs", (name+"/nPVs; Number of Primary Vertices; Entries").c_str(), 101, -0.5, 100.5);
  nPVsGood = dir.make<TH1F>("nPVsGood", (name+"/nPVs; Number of Good (!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2) Primary Vertices; Entries").c_str(), 101, -0.5, 100.5);
  st = dir.make<TH1F>("st", (name+"/st; Scalar sum of jet p_{T}'s [GeV]; Entries").c_str(), 130, 200, 1500);
  stNotCan = dir.make<TH1F>("stNotCan", (name+"/stNotCan; Scalar sum all other jet p_{T}'s [GeV]; Entries").c_str(), 150, 0, 1500);
  v4j = new vecHists(name+"/v4j", fs, "4j");
  s4j = dir.make<TH1F>("s4j", (name+"/s4j; Scalar sum of boson candidate jet p_{T}'s [GeV]; Entries").c_str(), 90, 100, 1000);
  r4j = dir.make<TH1F>("r4j", (name+"/r4j; Quadjet system p_{T} / s_{T}; Entries").c_str(), 50, 0, 1);
  dBB = dir.make<TH1F>("dBB", (name+"/dBB; D_{BB}; Entries").c_str(), 40, 0, 200);
  dEtaBB = dir.make<TH1F>("dEtaBB", (name+"/dEtaBB; #Delta#eta_{BB}; Entries").c_str(), 100, -5, 5);
  dRBB = dir.make<TH1F>("dRBB", (name+"/dRBB; #Delta#R_{BB}; Entries").c_str(), 50, 0, 5);
  xZZ = dir.make<TH1F>("xZZ", (name+"/xZZ; X_{ZZ}; Entries").c_str(), 100, 0, 10);
  Double_t bins_mZZ[] = {100, 182, 200, 220, 242, 266, 292, 321, 353, 388, 426, 468, 514, 565, 621, 683, 751, 826, 908, 998, 1097, 1206, 1326, 1500};
  mZZ = dir.make<TH1F>("mZZ", (name+"/mZZ; m_{ZZ} [GeV]; Entries").c_str(), 23, bins_mZZ);
  xZH = dir.make<TH1F>("xZH", (name+"/xZH; X_{ZH}; Entries").c_str(), 100, 0, 10);  
  Double_t bins_mZH[] = {100, 216, 237, 260, 286, 314, 345, 379, 416, 457, 502, 552, 607, 667, 733, 806, 886, 974, 1071, 1178, 1295, 1500};
  mZH = dir.make<TH1F>("mZH", (name+"/mZH; m_{ZH} [GeV]; Entries").c_str(), 21, bins_mZH);

  xWt0 = dir.make<TH1F>("xWt0", (name+"/xWt0; Minimum X_{Wt}; Entries").c_str(), 100, 0, 10);
  xWt1 = dir.make<TH1F>("xWt1", (name+"/xWt1; Next-to-minimum X_{Wt}; Entries").c_str(), 100, 0, 10);

  FvT = dir.make<TH1F>("FvT", (name+"/FvT; Four vs Three Tag Classifier Output; Entries").c_str(), 500, 0, 1);
  FvTUnweighted = dir.make<TH1F>("FvTUnweighted", (name+"/FvTUnweighted; Four vs Three Tag Classifier Output; Entries").c_str(), 500, 0, 1);
  ZHvB = dir.make<TH1F>("ZHvB", (name+"/ZHvB; ZH vs Background Classifier Output; Entries").c_str(), 100, 0, 1);

  if(isMC){
    Double_t bins_m4b[] = {100, 112, 126, 142, 160, 181, 205, 232, 263, 299, 340, 388, 443, 507, 582, 669, 770, 888, 1027, 1190, 1381, 1607, 2000};
    truthM4b = dir.make<TH1F>("truthM4b", (name+"/truthM4b; True m_{4b} [GeV]; Entries").c_str(), 21, bins_mZH);
    truthM4b_vs_mZH = dir.make<TH2F>("truthM4b_vs_mZH", (name+"/truthM4b_vs_mZH; True m_{4b} [GeV]; Reconstructed m_{ZH} [GeV];Entries").c_str(), 22, bins_m4b, 22, bins_m4b);
  }

} 

void viewHists::Fill(eventData* event, std::unique_ptr<eventView> &view){
  //
  // Object Level
  //
  nAllJets->Fill(event->allJets.size(), event->weight);
  st->Fill(event->st, event->weight);
  stNotCan->Fill(event->stNotCan, event->weight);
  nSelJets->Fill(event->nSelJets, event->weight);
  if     (event->s4j < 320) nSelJets_lowSt ->Fill(event->nSelJets, event->weight);
  else if(event->s4j < 450) nSelJets_midSt ->Fill(event->nSelJets, event->weight);
  else                      nSelJets_highSt->Fill(event->nSelJets, event->weight);
  nSelJetsUnweighted->Fill(event->selJets.size(), event->weight/event->pseudoTagWeight);
  if     (event->s4j < 320) nSelJetsUnweighted_lowSt ->Fill(event->nSelJets, event->weight/event->pseudoTagWeight);//these depend only on FvT classifier ratio spline
  else if(event->s4j < 450) nSelJetsUnweighted_midSt ->Fill(event->nSelJets, event->weight/event->pseudoTagWeight);
  else                      nSelJetsUnweighted_highSt->Fill(event->nSelJets, event->weight/event->pseudoTagWeight);
  nTagJets->Fill(event->nTagJets, event->weight);
  nPSTJets->Fill(event->nPSTJets, event->weight);
  if     (event->s4j < 320) nPSTJets_lowSt ->Fill(event->nPSTJets, event->weight);
  else if(event->s4j < 450) nPSTJets_midSt ->Fill(event->nPSTJets, event->weight);
  else                 nPSTJets_highSt->Fill(event->nPSTJets, event->weight);
  nCanJets->Fill(event->canJets.size(), event->weight);
  for(auto &jet: event->allJets) allJets->Fill(jet, event->weight);
  for(auto &jet: event->selJets) selJets->Fill(jet, event->weight);
  for(auto &jet: event->tagJets) tagJets->Fill(jet, event->weight);
  for(auto &jet: event->canJets) canJets->Fill(jet, event->weight);
  canJet0->Fill(event->canJets[0], event->weight);
  canJet1->Fill(event->canJets[1], event->weight);
  canJet2->Fill(event->canJets[2], event->weight);
  canJet3->Fill(event->canJets[3], event->weight);
  for(auto &jet: event->othJets) othJets->Fill(jet, event->weight);
  aveAbsEta->Fill(event->aveAbsEta, event->weight);
  aveAbsEtaOth->Fill(event->aveAbsEtaOth, event->weight);

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

  close ->Fill(event->close,  event->weight);
  other ->Fill(event->other,  event->weight);
  close_m_vs_other_m->Fill(event->close->m, event->other->m, event->weight);

  //
  // Event Level
  //
  nPVs->Fill(event->nPVs, event->weight);
  nPVsGood->Fill(event->nPVsGood, event->weight);
  v4j->Fill(view->p, event->weight);
  s4j->Fill(event->s4j, event->weight);
  r4j->Fill(view->pt/event->s4j, event->weight);
  dBB->Fill(view->dBB, event->weight);
  dEtaBB->Fill(view->dEtaBB, event->weight);
  dRBB->Fill(view->dRBB, event->weight);
  xZZ->Fill(view->xZZ, event->weight);
  mZZ->Fill(view->mZZ, event->weight);
  xZH->Fill(view->xZH, event->weight);
  mZH->Fill(view->mZH, event->weight);

  xWt0->Fill(event->xWt0, event->weight);
  xWt1->Fill(event->xWt1, event->weight);

  FvT->Fill(event->FvT, event->weight);
  FvTUnweighted->Fill(event->FvT, event->weight/event->FvTWeight); // depends only on jet combinatoric model
  ZHvB->Fill(event->ZHvB, event->weight);

  if(event->truth != NULL){
    truthM4b       ->Fill(event->truth->m4b,            event->weight);
    truthM4b_vs_mZH->Fill(event->truth->m4b, view->mZH, event->weight);
  }

  return;
}

viewHists::~viewHists(){} 

