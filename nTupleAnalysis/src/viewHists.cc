#include "ZZ4b/nTupleAnalysis/interface/viewHists.h"

using namespace nTupleAnalysis;

viewHists::viewHists(std::string name, fwlite::TFileService& fs, bool isMC, bool _debug, eventData* event) {
  dir = fs.mkdir(name);
  debug = _debug;

  //
  // Object Level
  //
  nAllJets = dir.make<TH1F>("nAllJets", (name+"/nAllJets; Number of Jets (pt>20); Entries").c_str(),  16,-0.5,15.5);
  nAllNotCanJets = dir.make<TH1F>("nAllNotCanJets", (name+"/nAllNotCanJets; Number of Jets excluding boson candidate jets (pt>20); Entries").c_str(),  16,-0.5,15.5);
  nSelJets = dir.make<TH1F>("nSelJets", (name+"/nSelJets; Number of Selected Jets; Entries").c_str(),  16,-0.5,15.5);
  nSelJets_noBTagSF = dir.make<TH1F>("nSelJets_noBTagSF", (name+"/nSelJets_noBTagSF; Number of Selected Jets; Entries").c_str(),  16,-0.5,15.5);
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
  nPSTJetsUnweighted = dir.make<TH1F>("nPSTJetsUnweighted", (name+"/nPSTJetsUnweighted; Number of Tagged + Pseudo-Tagged (Unweighted) Jets; Entries").c_str(),  16,-0.5,15.5);
  nCanJets = dir.make<TH1F>("nCanJets", (name+"/nCanJets; Number of Boson Candidate Jets; Entries").c_str(),  16,-0.5,15.5);
  //allJets = new jetHists(name+"/allJets", fs, "All Jets");
  allNotCanJets = new jetHists(name+"/allNotCanJets", fs, "All Jets Excluding Boson Candidate Jets");
  selJets = new jetHists(name+"/selJets", fs, "Selected Jets");
  tagJets = new jetHists(name+"/tagJets", fs, "Tagged Jets");
  canJets = new jetHists(name+"/canJets", fs, "Boson Candidate Jets");
  canJet0 = new jetHists(name+"/canJet0", fs, "Boson Candidate Jet_{0}");
  canJet1 = new jetHists(name+"/canJet1", fs, "Boson Candidate Jet_{1}");
  canJet2 = new jetHists(name+"/canJet2", fs, "Boson Candidate Jet_{2}");
  canJet3 = new jetHists(name+"/canJet3", fs, "Boson Candidate Jet_{3}");
  othJets = new jetHists(name+"/othJets", fs, "Other Selected Jets");
  aveAbsEta = dir.make<TH1F>("aveAbsEta", (name+"/aveAbsEta; <|#eta|>; Entries").c_str(), 25, 0 , 2.5);
  aveAbsEtaOth = dir.make<TH1F>("aveAbsEtaOth", (name+"/aveAbsEtaOth; Other Jets <|#eta|>; Entries").c_str(), 27, -0.2, 2.5);
  //allTrigJets = new trigHists(name+"/allTrigJets", fs, "All Trig Jets");
    
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
  m4j_vs_leadSt_dR = dir.make<TH2F>("m4j_vs_leadSt_dR", (name+"/m4j_vs_leadSt_dR; m_{4j} [GeV]; S_{T} leading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);
  m4j_vs_sublSt_dR = dir.make<TH2F>("m4j_vs_sublSt_dR", (name+"/m4j_vs_sublSt_dR; m_{4j} [GeV]; S_{T} subleading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);

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
  v4j = new fourVectorHists(name+"/v4j", fs, "4j");
  s4j = dir.make<TH1F>("s4j", (name+"/s4j; Scalar sum of boson candidate jet p_{T}'s [GeV]; Entries").c_str(), 90, 100, 1000);
  r4j = dir.make<TH1F>("r4j", (name+"/r4j; Quadjet system p_{T} / s_{T}; Entries").c_str(), 50, 0, 1);
  m123 = dir.make<TH1F>("m123", (name+"/m123; m_{1,2,3}; Entries").c_str(), 100, 0, 1000);
  m023 = dir.make<TH1F>("m023", (name+"/m023; m_{0,2,3}; Entries").c_str(), 100, 0, 1000);
  m013 = dir.make<TH1F>("m013", (name+"/m013; m_{0,1,3}; Entries").c_str(), 100, 0, 1000);
  m012 = dir.make<TH1F>("m012", (name+"/m012; m_{0,1,2}; Entries").c_str(), 100, 0, 1000);
  dBB = dir.make<TH1F>("dBB", (name+"/dBB; D_{BB}; Entries").c_str(), 40, 0, 200);
  dEtaBB = dir.make<TH1F>("dEtaBB", (name+"/dEtaBB; #Delta#eta_{BB}; Entries").c_str(), 100, -5, 5);
  dRBB = dir.make<TH1F>("dRBB", (name+"/dRBB; #Delta#R_{BB}; Entries").c_str(), 50, 0, 5);

  xZZ = dir.make<TH1F>("xZZ", (name+"/xZZ; X_{ZZ}; Entries").c_str(), 100, 0, 10);
  Double_t bins_mZZ[] = {100, 182, 200, 220, 242, 266, 292, 321, 353, 388, 426, 468, 514, 565, 621, 683, 751, 826, 908, 998, 1097, 1206, 1326, 1500};
  mZZ = dir.make<TH1F>("mZZ", (name+"/mZZ; m_{ZZ} [GeV]; Entries").c_str(), 23, bins_mZZ);

  xZH = dir.make<TH1F>("xZH", (name+"/xZH; X_{ZH}; Entries").c_str(), 100, 0, 10);  
  Double_t bins_mZH[] = {100, 216, 237, 260, 286, 314, 345, 379, 416, 457, 502, 552, 607, 667, 733, 806, 886, 974, 1071, 1178, 1295, 1500};
  mZH = dir.make<TH1F>("mZH", (name+"/mZH; m_{ZH} [GeV]; Entries").c_str(), 21, bins_mZH);

  xWt0 = dir.make<TH1F>("xWt0", (name+"/xWt0; X_{Wt,0}; Entries").c_str(), 60, 0, 12);
  xWt1 = dir.make<TH1F>("xWt1", (name+"/xWt1; X_{Wt,1}; Entries").c_str(), 60, 0, 12);
  //xWt2 = dir.make<TH1F>("xWt2", (name+"/xWt2; X_{Wt,2}; Entries").c_str(), 60, 0, 12);
  xWt  = dir.make<TH1F>("xWt",  (name+"/xWt;  X_{Wt};   Entries").c_str(), 60, 0, 12);
  t0 = new trijetHists(name+"/t0",  fs, "Top Candidate (#geq0 non-candidate jets)");
  t1 = new trijetHists(name+"/t1",  fs, "Top Candidate (#geq1 non-candidate jets)");
  //t2 = new trijetHists(name+"/t2",  fs, "Top Candidate (#geq2 non-candidate jets)");
  t = new trijetHists(name+"/t",  fs, "Top Candidate");

  FvT = dir.make<TH1F>("FvT", (name+"/FvT; Kinematic Reweight; Entries").c_str(), 100, 0, 5);
  FvTUnweighted = dir.make<TH1F>("FvTUnweighted", (name+"/FvTUnweighted; Kinematic Reweight; Entries").c_str(), 100, 0, 5);
  FvT_pd4 = dir.make<TH1F>("FvT_pd4", (name+"/FvT_pd4; FvT Regressed P(Four-tag Data) ; Entries").c_str(), 100, 0, 1);
  FvT_pd3 = dir.make<TH1F>("FvT_pd3", (name+"/FvT_pd3; FvT Regressed P(Three-tag Data) ; Entries").c_str(), 100, 0, 1);
  FvT_pt4 = dir.make<TH1F>("FvT_pt4", (name+"/FvT_pt4; FvT Regressed P(Four-tag t#bar{t}) ; Entries").c_str(), 100, 0, 1);
  FvT_pt3 = dir.make<TH1F>("FvT_pt3", (name+"/FvT_pt3; FvT Regressed P(Three-tag t#bar{t}) ; Entries").c_str(), 100, 0, 1);
  FvT_pm4 = dir.make<TH1F>("FvT_pm4", (name+"/FvT_pm4; FvT Regressed P(Four-tag Multijet) ; Entries").c_str(), 100, 0, 1);
  FvT_pm3 = dir.make<TH1F>("FvT_pm3", (name+"/FvT_pm3; FvT Regressed P(Three-tag Multijet) ; Entries").c_str(), 100, 0, 1);
  FvT_pt  = dir.make<TH1F>("FvT_pt",  (name+"/FvT_pt;  FvT Regressed P(t#bar{t}) ; Entries").c_str(), 100, 0, 1);
  SvB_ps  = dir.make<TH1F>("SvB_ps",  (name+"/SvB_ps;  SvB Regressed P(ZZ)+P(ZH); Entries").c_str(), 100, 0, 1);
  if(event){
    SvB_ps_bTagSysts = new systHists(SvB_ps, event->treeJets->m_btagVariations);
  }
  SvB_pzz = dir.make<TH1F>("SvB_pzz", (name+"/SvB_pzz; SvB Regressed P(ZZ); Entries").c_str(), 100, 0, 1);
  SvB_pzh = dir.make<TH1F>("SvB_pzh", (name+"/SvB_pzh; SvB Regressed P(ZH); Entries").c_str(), 100, 0, 1);
  SvB_ptt = dir.make<TH1F>("SvB_ptt", (name+"/SvB_ptt; SvB Regressed P(t#bar{t}); Entries").c_str(), 100, 0, 1);
  SvB_ps_zh = dir.make<TH1F>("SvB_ps_zh",  (name+"/SvB_ps_zh;  SvB Regressed P(ZZ)+P(ZH), P(ZH)$ #geq P(ZZ); Entries").c_str(), 100, 0, 1);
  SvB_ps_zz = dir.make<TH1F>("SvB_ps_zz",  (name+"/SvB_ps_zz;  SvB Regressed P(ZZ)+P(ZH), P(ZZ) > P(ZH); Entries").c_str(), 100, 0, 1);

  //Simplified template cross section binning https://cds.cern.ch/record/2669925/files/1906.02754.pdf
  SvB_ps_zh_0_75 = dir.make<TH1F>("SvB_ps_zh_0_75",  (name+"/SvB_ps_zh_0_75;  SvB Regressed P(ZZ)+P(ZH), P(ZH)$ #geq P(ZZ), 0<p_{T,Z}<75; Entries").c_str(), 100, 0, 1);
  SvB_ps_zh_75_150 = dir.make<TH1F>("SvB_ps_zh_75_150",  (name+"/SvB_ps_zh_75_150;  SvB Regressed P(ZZ)+P(ZH), P(ZH)$ #geq P(ZZ), 75<p_{T,Z}<150; Entries").c_str(), 100, 0, 1);
  SvB_ps_zh_150_250 = dir.make<TH1F>("SvB_ps_zh_150_250",  (name+"/SvB_ps_zh_150_250;  SvB Regressed P(ZZ)+P(ZH), P(ZH)$ #geq P(ZZ), 150<p_{T,Z}<250; Entries").c_str(), 100, 0, 1);
  SvB_ps_zh_250_400 = dir.make<TH1F>("SvB_ps_zh_250_400",  (name+"/SvB_ps_zh_250_400;  SvB Regressed P(ZZ)+P(ZH), P(ZH)$ #geq P(ZZ), 250<p_{T,Z}<400; Entries").c_str(), 100, 0, 1);
  SvB_ps_zh_400_inf = dir.make<TH1F>("SvB_ps_zh_400_inf",  (name+"/SvB_ps_zh_400_inf;  SvB Regressed P(ZZ)+P(ZH), P(ZH)$ #geq P(ZZ), 400<p_{T,Z}<inf; Entries").c_str(), 100, 0, 1);

  SvB_ps_zz_0_75 = dir.make<TH1F>("SvB_ps_zz_0_75",  (name+"/SvB_ps_zz_0_75;  SvB Regressed P(ZZ)+P(ZH), P(ZZ)$ > P(ZH), 0<p_{T,Z}<75; Entries").c_str(), 100, 0, 1);
  SvB_ps_zz_75_150 = dir.make<TH1F>("SvB_ps_zz_75_150",  (name+"/SvB_ps_zz_75_150;  SvB Regressed P(ZZ)+P(ZH), P(ZZ)$ > P(ZH), 75<p_{T,Z}<150; Entries").c_str(), 100, 0, 1);
  SvB_ps_zz_150_250 = dir.make<TH1F>("SvB_ps_zz_150_250",  (name+"/SvB_ps_zz_150_250;  SvB Regressed P(ZZ)+P(ZH), P(ZZ)$ > P(ZH), 150<p_{T,Z}<250; Entries").c_str(), 100, 0, 1);
  SvB_ps_zz_250_400 = dir.make<TH1F>("SvB_ps_zz_250_400",  (name+"/SvB_ps_zz_250_400;  SvB Regressed P(ZZ)+P(ZH), P(ZZ)$ > P(ZH), 250<p_{T,Z}<400; Entries").c_str(), 100, 0, 1);
  SvB_ps_zz_400_inf = dir.make<TH1F>("SvB_ps_zz_400_inf",  (name+"/SvB_ps_zz_400_inf;  SvB Regressed P(ZZ)+P(ZH), P(ZZ)$ > P(ZH), 400<p_{T,Z}<inf; Entries").c_str(), 100, 0, 1);


  xHH = dir.make<TH1F>("xHH", (name+"/xHH; X_{HH}; Entries").c_str(), 100, 0, 10);  
  Double_t bins_mHH[] = {100, 216, 237, 260, 286, 314, 345, 379, 416, 457, 502, 552, 607, 667, 733, 806, 886, 974, 1071, 1178, 1295, 1500};
  //mHH = dir.make<TH1F>("mHH", (name+"/mHH; m_{HH} [GeV]; Entries").c_str(), 100, 150,1500);
  mHH = dir.make<TH1F>("mHH", (name+"/mHH; m_{HH} [GeV]; Entries").c_str(), 21, bins_mHH);

  hT   = dir.make<TH1F>("hT", (name+"/hT; hT [GeV]; Entries").c_str(),  100,0,1000);
  hT30 = dir.make<TH1F>("hT30", (name+"/hT30; hT [GeV] (jet Pt > 30 GeV); Entries").c_str(),  100,0,1000);
  L1hT   = dir.make<TH1F>("L1hT", (name+"/L1hT; hT [GeV]; Entries").c_str(),  100,0,1000);
  L1hT30 = dir.make<TH1F>("L1hT30", (name+"/L1hT30; hT [GeV] (L1 jet Pt > 30 GeV); Entries").c_str(),  100,0,1000);
  HLThT   = dir.make<TH1F>("HLThT", (name+"/HLThT; hT [GeV]; Entries").c_str(),  100,0,1000);
  HLThT30 = dir.make<TH1F>("HLThT30", (name+"/HLThT30; hT [GeV] (HLT jet Pt > 30 GeV); Entries").c_str(),  100,0,1000);

  m4j_vs_nViews = dir.make<TH2F>("m4j_vs_nViews", (name+"/m4j_vs_nViews; m_{4j} [GeV]; Number of Event Views; Entries").c_str(), 40,100,1100, 3,0.5,3.5);

  if(isMC){
    Double_t bins_m4b[] = {100, 112, 126, 142, 160, 181, 205, 232, 263, 299, 340, 388, 443, 507, 582, 669, 770, 888, 1027, 1190, 1381, 1607, 2000};
    truthM4b = dir.make<TH1F>("truthM4b", (name+"/truthM4b; True m_{4b} [GeV]; Entries").c_str(), 21, bins_mZH);
    truthM4b_vs_mZH = dir.make<TH2F>("truthM4b_vs_mZH", (name+"/truthM4b_vs_mZH; True m_{4b} [GeV]; Reconstructed m_{ZH} [GeV];Entries").c_str(), 22, bins_m4b, 22, bins_m4b);
    nTrueBJets = dir.make<TH1F>("nTrueBJets", (name+"/nTrueBJets; Number of true b-jets; Entries").c_str(),  16,-0.5,15.5);
  }

} 

void viewHists::Fill(eventData* event, std::unique_ptr<eventView> &view){
  //
  // Object Level
  //
  nAllJets->Fill(event->allJets.size(), event->weight);
  nAllNotCanJets->Fill(event->nAllNotCanJets, event->weight);
  st->Fill(event->st, event->weight);
  stNotCan->Fill(event->stNotCan, event->weight);
  nSelJets->Fill(event->nSelJets, event->weight);
  nSelJets_noBTagSF->Fill(event->nSelJets, event->weight/event->bTagSF);
  if     (event->s4j < 320) nSelJets_lowSt ->Fill(event->nSelJets, event->weight);
  else if(event->s4j < 450) nSelJets_midSt ->Fill(event->nSelJets, event->weight);
  else                      nSelJets_highSt->Fill(event->nSelJets, event->weight);
  if(event->pseudoTagWeight < 1e-6) std::cout << "viewHists::Fill WARNING event->pseudoTagWeight " << event->pseudoTagWeight << std::endl;
  float weightDividedByPseudoTagWeight = (event->pseudoTagWeight > 0) ? event->weight/event->pseudoTagWeight : 0;
  if(debug) std::cout << "viewHists::Fill event->weight " << event->weight << " event->pseudoTagWeight " << event->pseudoTagWeight << " weightDividedByPseudoTagWeight " << weightDividedByPseudoTagWeight << std::endl;
  nSelJetsUnweighted->Fill(event->nSelJets, weightDividedByPseudoTagWeight);
  if     (event->s4j < 320) nSelJetsUnweighted_lowSt ->Fill(event->nSelJets, weightDividedByPseudoTagWeight);//these depend only on FvT classifier ratio spline
  else if(event->s4j < 450) nSelJetsUnweighted_midSt ->Fill(event->nSelJets, weightDividedByPseudoTagWeight);
  else                      nSelJetsUnweighted_highSt->Fill(event->nSelJets, weightDividedByPseudoTagWeight);
  nTagJets->Fill(event->nTagJets, event->weight);
  nPSTJets->Fill(event->nPSTJets, event->weight);
  if     (event->s4j < 320) nPSTJets_lowSt ->Fill(event->nPSTJets, event->weight);
  else if(event->s4j < 450) nPSTJets_midSt ->Fill(event->nPSTJets, event->weight);
  else                      nPSTJets_highSt->Fill(event->nPSTJets, event->weight);
  nPSTJetsUnweighted->Fill(event->nPSTJets, weightDividedByPseudoTagWeight);
  nCanJets->Fill(event->canJets.size(), event->weight);

  hT  ->Fill(event->ht,   event->weight);
  hT30->Fill(event->ht30, event->weight);

  for(auto &jet: event->selJets) selJets->Fill(jet, event->weight);
  for(auto &jet: event->tagJets) tagJets->Fill(jet, event->weight);
  for(auto &jet: event->canJets) canJets->Fill(jet, event->weight);
  for(auto &jet: event->allNotCanJets) allNotCanJets->Fill(jet, event->weight);
  canJet0->Fill(event->canJets[0], event->weight);
  canJet1->Fill(event->canJets[1], event->weight);
  canJet2->Fill(event->canJets[2], event->weight);
  canJet3->Fill(event->canJets[3], event->weight);
  for(auto &jet: event->othJets) othJets->Fill(jet, event->weight);
  aveAbsEta->Fill(event->aveAbsEta, event->weight);
  aveAbsEtaOth->Fill(event->aveAbsEtaOth, event->weight);

  if(allTrigJets){
    for(auto &trigjet: event->allTrigJets) allTrigJets->Fill(trigjet, event->weight);
    L1hT  ->Fill(event->L1ht,   event->weight);
    L1hT30->Fill(event->L1ht30, event->weight);

    HLThT  ->Fill(event->HLTht,   event->weight);
    HLThT30->Fill(event->HLTht30, event->weight);
  }

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
  m4j_vs_leadSt_dR->Fill(view->m4j, view->leadSt->dR, event->weight);
  m4j_vs_sublSt_dR->Fill(view->m4j, view->sublSt->dR, event->weight);

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
  m123->Fill(event->m123, event->weight);
  m023->Fill(event->m023, event->weight);
  m013->Fill(event->m013, event->weight);
  m012->Fill(event->m012, event->weight);
  dBB->Fill(view->dBB, event->weight);
  dEtaBB->Fill(view->dEtaBB, event->weight);
  dRBB->Fill(view->dRBB, event->weight);
  xZZ->Fill(view->xZZ, event->weight);
  mZZ->Fill(view->mZZ, event->weight);
  xZH->Fill(view->xZH, event->weight);
  mZH->Fill(view->mZH, event->weight);
  xHH->Fill(view->xHH, event->weight);
  mHH->Fill(view->mHH, event->weight);

  xWt0->Fill(event->xWt0, event->weight);
  xWt1->Fill(event->xWt1, event->weight);
  //xWt2->Fill(event->xWt2, event->weight);
  xWt ->Fill(event->xWt,  event->weight);
  t0->Fill(event->t0, event->weight);
  t1->Fill(event->t1, event->weight);
  //t2->Fill(event->t2, event->weight);
  t ->Fill(event->t,  event->weight);

  FvT->Fill(event->FvT, event->weight);
  FvTUnweighted->Fill(event->FvT, event->weight/event->reweight); // depends only on jet combinatoric model
  FvT_pd4->Fill(event->FvT_pd4, event->weight);
  FvT_pd3->Fill(event->FvT_pd3, event->weight);
  FvT_pt4->Fill(event->FvT_pt4, event->weight);
  FvT_pt3->Fill(event->FvT_pt3, event->weight);
  FvT_pm4->Fill(event->FvT_pm4, event->weight);
  FvT_pm3->Fill(event->FvT_pm3, event->weight);
  FvT_pt ->Fill(event->FvT_pt,  event->weight);
  SvB_ps ->Fill(event->SvB_ps , event->weight);
  if(SvB_ps_bTagSysts){
    SvB_ps_bTagSysts->Fill(event->SvB_ps, event->weight/event->bTagSF, event->treeJets->m_btagSFs);
  }
  SvB_pzz->Fill(event->SvB_pzz, event->weight);
  SvB_pzh->Fill(event->SvB_pzh, event->weight);
  SvB_ptt->Fill(event->SvB_ptt, event->weight);
  if(event->SvB_pzz<event->SvB_pzh){
    SvB_ps_zh->Fill(event->SvB_ps, event->weight);
    //Simplified template cross section binning https://cds.cern.ch/record/2669925/files/1906.02754.pdf
    if      (view->sublM->pt< 75){
      SvB_ps_zh_0_75   ->Fill(event->SvB_ps, event->weight);
    }else if(view->sublM->pt<150){
      SvB_ps_zh_75_150 ->Fill(event->SvB_ps, event->weight);
    }else if(view->sublM->pt<250){
      SvB_ps_zh_150_250->Fill(event->SvB_ps, event->weight);
    }else if(view->sublM->pt<400){
      SvB_ps_zh_250_400->Fill(event->SvB_ps, event->weight);
    }else{
      SvB_ps_zh_400_inf->Fill(event->SvB_ps, event->weight);
    }
  }else{
    SvB_ps_zz->Fill(event->SvB_ps, event->weight);
    //Simplified template cross section binning https://cds.cern.ch/record/2669925/files/1906.02754.pdf
    if      (view->sublM->pt< 75){
      SvB_ps_zz_0_75   ->Fill(event->SvB_ps, event->weight);
    }else if(view->sublM->pt<150){
      SvB_ps_zz_75_150 ->Fill(event->SvB_ps, event->weight);
    }else if(view->sublM->pt<250){
      SvB_ps_zz_150_250->Fill(event->SvB_ps, event->weight);
    }else if(view->sublM->pt<400){
      SvB_ps_zz_250_400->Fill(event->SvB_ps, event->weight);
    }else{
      SvB_ps_zz_400_inf->Fill(event->SvB_ps, event->weight);
    }
  }

  m4j_vs_nViews->Fill(view->m4j, event->views.size(), event->weight);

  if(event->truth != NULL){
    truthM4b       ->Fill(event->truth->m4b,            event->weight);
    truthM4b_vs_mZH->Fill(event->truth->m4b, view->mZH, event->weight);
    nTrueBJets->Fill(event->nTrueBJets, event->weight);
  }

  return;
}

viewHists::~viewHists(){} 

