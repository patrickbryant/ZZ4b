#include "ZZ4b/nTupleAnalysis/interface/triggerStudy.h"

using namespace nTupleAnalysis;

using std::cout;  using std::endl;
using std::vector;

triggerStudy::triggerStudy(std::string name, fwlite::TFileService& fs, bool _debug) {
  std::cout << "Initialize >> triggerStudy: " << name << std::endl;

  dir = fs.mkdir(name);
  debug = _debug;
  
  hT30_ht330            = new turnOnHist("ht330",            name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_L1HT360    = new turnOnHist("ht330_L1HT360",    name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_L1ETT2000  = new turnOnHist("ht330_L1ETT200",   name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_L1HT320_4j = new turnOnHist("ht330_L1HT320_4j", name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_L1OR       = new turnOnHist("ht330_L1OR",       name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);

  hT30_ht330_wrt_L1OR   = new turnOnHist("ht330_wrt_L1OR",   name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);

  hT30_ht330_3tag       = new turnOnHist("ht330_3tag",       name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_sel       = new turnOnHist("ht330_sel",       name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_sel_noSubSet   = new turnOnHist("ht330_sel_noSubSet",       name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_sel_3tag       = new turnOnHist("ht330_sel_3tag",       name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);

  hT30_L1HT360    = new turnOnHist("L1HT360",    name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_L1ETT2000  = new turnOnHist("L1ETT2000",  name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_L1HT320_4j = new turnOnHist("L1HT320_4j", name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_L1OR       = new turnOnHist("L1OR",       name, fs, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);

  hT30_all  = dir.make<TH1F>("hT30_all",  (name+"/hT30_all;  hT [GeV] (jet Pt > 30 GeV); Entries").c_str(),  200,0,2000);
  hT30_h330  = dir.make<TH1F>("hT30_h330",  (name+"/hT30_h330;  hT [GeV] (jet Pt > 30 GeV); Entries").c_str(),  200,0,2000);
  hT30_h330_l320  = dir.make<TH1F>("hT30_h330_l320",  (name+"/hT30_h330_l320;  hT [GeV] (jet Pt > 30 GeV); Entries").c_str(),  200,0,2000);
  hT30_h330_l320_j30  = dir.make<TH1F>("hT30_h330_l320_j30",  (name+"/hT30_h330_l320_j30;  hT [GeV] (jet Pt > 30 GeV); Entries").c_str(),  200,0,2000);



  //
  //  Turn-On Study In MC 
  //
  ht_4j           = new turnOnHist("ht_4j",       name, fs, "ht [GeV] ",200,0,2000);
  htcalo_4j       = new turnOnHist("htcalo_4j",       name, fs, "ht [GeV] ",200,0,2000);
  htcaloAll_4j    = new turnOnHist("htcaloAll_4j",       name, fs, "ht [GeV] ",200,0,2000);
  htcalo2p6_4j    = new turnOnHist("htcalo2p6_4j",       name, fs, "ht [GeV] ",200,0,2000);
  j0_4j           = new turnOnHist("j0_4j",       name, fs, "1st Jet PT [GeV] ",100,0,400);
  j1_4j           = new turnOnHist("j1_4j",       name, fs, "2nd Jet PT [GeV] ",100,0,200);
  j2_4j           = new turnOnHist("j2_4j",       name, fs, "3rd Jet PT [GeV] ",100,0,200);
  j3_4j           = new turnOnHist("j3_4j",       name, fs, "4th Jet PT [GeV] ",100,0,200);
  j3_4j_wrt_HT_3j = new turnOnHist("j3_4j_wrt_HT_3j",       name, fs, "4th Jet PT [GeV] ",100,0,200);

  //
  //  Turn-On Study wrt Emulation
  //
  ht_4j_em           = new turnOnHist("ht_4j_em",          name, fs, "ht [GeV] ",200,0,2000);
  ht_4j_3b_em        = new turnOnHist("ht_4j_3b_em",       name, fs, "ht [GeV] ",200,0,2000);
  ht_4j_l1_em        = new turnOnHist("ht_4j_l1_em",       name, fs, "ht [GeV] ",200,0,2000);
  ht_4j_l1_3b_em     = new turnOnHist("ht_4j_l1_3b_em",    name, fs, "ht [GeV] ",200,0,2000);
  ht_4j_ht_em           = new turnOnHist("ht_4j_ht_em",          name, fs, "ht [GeV] ",200,0,2000);
  ht_4j_3b_ht_em        = new turnOnHist("ht_4j_3b_ht_em",       name, fs, "ht [GeV] ",200,0,2000);

  //
  //  Jet-level Studies
  //
  hMatchedPt = dir.make<TH1F>("matchedPt",  (name+"/matchedPt;  matchedPt; Entries").c_str(),  100,0,200);
  hMatchedPt_jet = dir.make<TH1F>("matchedPt_jet",  (name+"/matchedPt_jet;  matchedPt; Entries").c_str(),  100,0,200);
  hAllPt     = dir.make<TH1F>("allPt",      (name+"/allPt;      allPt;     Entries").c_str(),  100,0,200);

  hMatchedEta = dir.make<TH1F>("matchedEta",  (name+"/matchedEta;  matchedEta; Entries").c_str(),  50,-3,3);
  hMatchedEta_jet = dir.make<TH1F>("matchedEta_jet",  (name+"/matchedEta_jet;  matchedEta; Entries").c_str(),  50,-3,3);
  hAllEta     = dir.make<TH1F>("allEta",      (name+"/allEta;      allEta;     Entries").c_str(),  50,-3,3);



  hMatched_dPt   = dir.make<TH1F>("matched_dPt",  (name+"/matched_dPt;  Delta Pt; Entries").c_str(),  50,-50,50);
  hMatched_dPtL1 = dir.make<TH1F>("matched_dPtL1",  (name+"/matched_dPtL1;  Delta Pt; Entries").c_str(),  50,-50,50);

  hMatched_dPt_l   = dir.make<TH1F>("matched_dPt_l",    (name+"/matched_dPt_l;    Delta Pt; Entries").c_str(),  50,-200,200);
  hMatched_dPtL1_l = dir.make<TH1F>("matched_dPtL1_l",  (name+"/matched_dPtL1_l;  Delta Pt; Entries").c_str(),  50,-200,200);

  hMatchedPt_h20     = dir.make<TH1F>("matchedPt_h20",      (name+"/matchedPt_h20; matchedPt HLT > 20; Entries").c_str(),                  100,0,200);

  hMatchedPt_h25     = dir.make<TH1F>("matchedPt_h25",      (name+"/matchedPt_h25; matchedPt HLT > 25; Entries").c_str(),                  100,0,200);
  hMatchedPt_h30     = dir.make<TH1F>("matchedPt_h30",      (name+"/matchedPt_h30; matchedPt HLT > 30; Entries").c_str(),                  100,0,200);
  hMatchedPt_h35     = dir.make<TH1F>("matchedPt_h35",      (name+"/matchedPt_h35; matchedPt HLT > 35; Entries").c_str(),                  100,0,200);

  hMatchedPt_h40     = dir.make<TH1F>("matchedPt_h40",      (name+"/matchedPt_h40; matchedPt HLT > 40; Entries").c_str(),                  100,0,200);
  hMatchedPt_h40_l40 = dir.make<TH1F>("matchedPt_h40_l40",  (name+"/matchedPt_h40_l40;  matchedPt HLT > 40 && L1 > 40; Entries").c_str(),  100,0,200);

  hMatchedPt_h45     = dir.make<TH1F>("matchedPt_h45",      (name+"/matchedPt_h45; matchedPt HLT > 45; Entries").c_str(),                  100,0,200);
  hMatchedPt_h45_l40 = dir.make<TH1F>("matchedPt_h45_l40",  (name+"/matchedPt_h45_l40;  matchedPt HLT > 45 && L1 > 40; Entries").c_str(),  100,0,200);

  hMatchedPt_h50     = dir.make<TH1F>("matchedPt_h50",      (name+"/matchedPt_h50; matchedPt HLT > 50; Entries").c_str(),                  100,0,200);

  hMatchedPt_h60     = dir.make<TH1F>("matchedPt_h60",      (name+"/matchedPt_h60; matchedPt HLT > 60; Entries").c_str(),                  100,0,200);
  hMatchedPt_h60_l55 = dir.make<TH1F>("matchedPt_h60_l55",  (name+"/matchedPt_h60_l55;  matchedPt HLT > 60 && L1 > 55; Entries").c_str(),  100,0,200);

  hMatchedPt_h70     = dir.make<TH1F>("matchedPt_h70",      (name+"/matchedPt_h70; matchedPt HLT > 70; Entries").c_str(),                  100,0,200);


  hMatchedPt_h75     = dir.make<TH1F>("matchedPt_h75",      (name+"/matchedPt_h75; matchedPt HLT > 75; Entries").c_str(),                  100,0,200);
  hMatchedPt_h75_l70 = dir.make<TH1F>("matchedPt_h75_l70",  (name+"/matchedPt_h75_l70;  matchedPt HLT > 75 && L1 > 70; Entries").c_str(),  100,0,200);

  hMatchedPt_h80     = dir.make<TH1F>("matchedPt_h80",      (name+"/matchedPt_h80; matchedPt HLT > 80; Entries").c_str(),                  100,0,200);
  hMatchedPt_h90     = dir.make<TH1F>("matchedPt_h90",      (name+"/matchedPt_h90; matchedPt HLT > 90; Entries").c_str(),                  100,0,200);
  hMatchedPt_h100    = dir.make<TH1F>("matchedPt_h100",     (name+"/matchedPt_h100; matchedPt HLT > 100; Entries").c_str(),                100,0,200);


  //hMatchedPt = dir.make<TH1F>("matchedPt",  (name+"/matchedPt;  matchedPt; Entries").c_str(),  100,0,100);
  //hAllPt     = dir.make<TH1F>("allPt",  (name+"/allPt;  allPt; Entries").c_str(),  100,0,100);

  hMinDr  = dir.make<TH1F>("minDr",  (name+"/minDr;  minDr(offline,HLT); Entries").c_str(),  100,-0.1,5);

  //
  // The Trig Emulator
  //
  int nToys = 100;
  trigEmulator = new TriggerEmulator::TrigEmulatorTool("trigEmulator", 1, nToys);
  trigEmulator->AddTrig("EMU_ht_4j_3Tag", "330ZH", {"75","60","45","40"}, {1,2,3,4},"2018",3);
  trigEmulator->AddTrig("EMU_ht_4j",      "330ZH", {"75","60","45","40"}, {1,2,3,4});

  trigEmulator->AddTrig("EMU_4j_3Tag", {"75","60","45","40"}, {1,2,3,4},"2018",3);
  trigEmulator->AddTrig("EMU_4j",      {"75","60","45","40"}, {1,2,3,4});
  
  trigEmulator->AddTrig("EMU_4j_3Tag_L1", {"75wL170","60wL155","45wL140","40wL140"}, {1,2,3,4},"2018",3);
  trigEmulator->AddTrig("EMU_4j_L1", {"75wL170","60wL155","45wL140","40wL140"}, {1,2,3,4});

  trigEmulator->AddTrig("EMU_3j_L1", {"75wL170","60wL155","45wL140"},      {1,2,3});
  trigEmulator->AddTrig("EMU_HT330_3j", "330", {"75wL170","60wL155","45wL140"},      {1,2,3});


} 

void triggerStudy::Fill(eventData* event){
  if(debug) cout << "In Fill " << endl;


  vector<float> selJet_pts;
  for(const jetPtr& sJet : event->selJets){
    selJet_pts.push_back(sJet->pt_wo_bRegCorr);
  }

  vector<float> tagJet_pts;
  unsigned int nTagJets = 0;
  for(const jetPtr& tJet : event->tagJets){
    if(nTagJets > 3) continue;
    ++nTagJets;
    tagJet_pts.push_back(tJet->pt_wo_bRegCorr);
  }

  trigEmulator->SetDecisions(selJet_pts, tagJet_pts, event->ht30);
  bool EMU_4j = trigEmulator->GetDecision("EMU_4j");
  bool EMU_4j_3Tag = trigEmulator->GetDecision("EMU_4j_3Tag");
  bool EMU_ht_4j = trigEmulator->GetDecision("EMU_ht_4j");
  bool EMU_ht_4j_3Tag = trigEmulator->GetDecision("EMU_ht_4j_3Tag");
  bool EMU_4j_L1 = trigEmulator->GetDecision("EMU_4j_L1");
  bool EMU_4j_3Tag_L1 = trigEmulator->GetDecision("EMU_4j_3Tag_L1");
  bool HLT_HT330_4j_75_60_45_40 = event->HLT_HT330_4j_75_60_45_40; // && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4;
  bool HLT_HT330_4j_75_60_45_40_3b = event->HLT_HT330_4j_75_60_45_40_3b; // && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4;


  if(debug) cout << "Filling ht30 " << endl;  

  bool L1_OR = (event->L1_HTT360er || event->L1_ETT2000 || event->L1_HTT320er_QuadJet_70_55_40_40_er2p4);

  //
  //  hT
  //
  float ht30 = event->ht30;
  float HLTht30 = event->HLTht30;
  float HLTht30Calo = event->HLTht30Calo;
  hT30_ht330            ->Fill(ht30, HLTht30,  HLT_HT330_4j_75_60_45_40,                           EMU_4j, event );
  hT30_ht330_L1HT360    ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40 && event->L1_HTT360er),    EMU_4j, event );
  hT30_ht330_L1ETT2000  ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40 && event->L1_ETT2000),     EMU_4j, event );
  hT30_ht330_L1HT320_4j ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4), EMU_4j, event );
  hT30_ht330_L1OR       ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40 && L1_OR), EMU_4j, event );

  hT30_ht330_wrt_L1OR  ->Fill(ht30, HLTht30,  (HLT_HT330_4j_75_60_45_40 && L1_OR), (L1_OR && EMU_4j), event );

  hT30_ht330_3tag       ->Fill(ht30, HLTht30,  HLT_HT330_4j_75_60_45_40,                           EMU_4j_3Tag, event );

  //cout<< " In trig study: " << (event->fourTag  || event->threeTag) << " " << event->passDijetMass << " " << event->passMDRs << endl;

  if(event->fourTag || event->threeTag){
    if(event->passDijetMass){
      event->applyMDRs(); 
      if(event->passMDRs){
	hT30_ht330_sel       ->Fill(ht30,  HLTht30, HLT_HT330_4j_75_60_45_40,                           EMU_4j,    event );
	hT30_ht330_sel_noSubSet   ->Fill(ht30,  HLTht30, HLT_HT330_4j_75_60_45_40,                           EMU_4j,    event, false );
	hT30_ht330_sel_3tag  ->Fill(ht30,  HLTht30, HLT_HT330_4j_75_60_45_40,                           EMU_4j_3Tag, event );
      }
    }
  }
  
  hT30_L1HT360    -> Fill(ht30, HLTht30, (event->L1_HTT360er), true, event );
  hT30_L1ETT2000  -> Fill(ht30, HLTht30, (event->L1_ETT2000), true, event );
  hT30_L1HT320_4j -> Fill(ht30, HLTht30, (event->L1_HTT320er_QuadJet_70_55_40_40_er2p4), true, event );
  hT30_L1OR       -> Fill(ht30, HLTht30, (L1_OR), true, event );

  //
  // hT 
  //
  hT30_all->Fill(event->ht30);
  if(HLTht30 > 330){
    hT30_h330->Fill(event->ht30);
    if(event->L1ht > 320){
      hT30_h330_l320->Fill(event->ht30);
    }

    if(event->L1ht30 > 320){
      hT30_h330_l320_j30->Fill(event->ht30);
    }

  }

  //
  // EMulation Study
  //
  ht_4j_em      ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40   ), EMU_4j   ,      event, false);
  ht_4j_3b_em   ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40_3b), EMU_4j_3Tag,    event, false);
  ht_4j_l1_em   ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40   ), EMU_4j_L1   ,   event, false);
  ht_4j_l1_3b_em->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40_3b), EMU_4j_3Tag_L1, event, false);

  ht_4j_ht_em    ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40   ), EMU_ht_4j   ,      event, false);
  ht_4j_3b_ht_em ->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40_3b), EMU_ht_4j_3Tag,    event, false);


  //
  //  MC Study
  //

  bool passHLTHt = (HLTht30 > 330);
  bool passHLTHtCalo = (HLTht30Calo > 320);
  unsigned int nJ40 = 0;
  unsigned int nJ45 = 0;
  unsigned int nJ60 = 0;
  unsigned int nJ75 = 0;
  for(const trigPtr& trig : event->selTrigJets){
    if(trig->pt > 40) ++nJ40;
    if(trig->pt > 45) ++nJ45;
    if(trig->pt > 60) ++nJ60;
    if(trig->pt > 75) ++nJ75;
  }

  unsigned int nJ30Calo = 0;
  for(const trigPtr& trig : event->allTrigJets){
    if(fabs(trig->eta) < 2.5){
      if(trig->l2pt > 30) ++nJ30Calo;
    }
  }





  bool pass4J30Calo  = (nJ30Calo > 3);
  bool pass4J40  = (nJ40 > 3);
  bool pass3J45  = (nJ45 > 2);
  bool pass2J60  = (nJ60 > 1);
  bool passJ75   = (nJ75 > 0);
  bool passAll = (passHLTHt && pass4J40 && pass3J45 && pass2J60 && passJ75) && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4 && passHLTHtCalo && pass4J30Calo;

  //if((HLT_HT330_4j_75_60_45_40 && passHLTHtCalo) != passAll){
  //  cout << "WARNING HLT= " << HLT_HT330_4j_75_60_45_40 << " trigObj = " << passAll << endl;
  //  if(HLT_HT330_4j_75_60_45_40){
  //    cout << "passHLTHt: " << passHLTHt << " pass4J40: " << pass4J40 << " pass3J45: " << pass3J45 << " pass2J60: " << pass2J60 << " passJ75: " << passJ75 << endl;
  //  }else{
  //    cout << "ht is " << HLTht30 << " htCalo is " << HLTht30Calo << " pass4J30Calo " << pass4J30Calo << endl;
  //  }
  //}
  //

  //
  //  HT
  //
  bool passAllButHT = (pass4J40 && pass3J45 && pass2J60 && passJ75 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4 && pass4J30Calo && passHLTHtCalo);
  ht_4j->Fill(ht30, HLTht30, (HLT_HT330_4j_75_60_45_40), passAllButHT, event);

  bool passAllButHTCalo = (pass4J40 && pass3J45 && pass2J60 && passJ75 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4 && pass4J30Calo && passHLTHt);
  htcalo_4j->Fill(ht30, HLTht30Calo, (HLT_HT330_4j_75_60_45_40), passAllButHTCalo, event);
  htcaloAll_4j->Fill(ht30, event->HLTht30CaloAll, (HLT_HT330_4j_75_60_45_40), passAllButHTCalo, event);
  htcalo2p6_4j->Fill(ht30, event->HLTht30Calo2p6, (HLT_HT330_4j_75_60_45_40), passAllButHTCalo, event);


  //
  //  lead Jet Pt
  //
  if(event->selJets.size() < 1) return;
  if(event->selTrigJets.size() < 1) return;
  float jetPt0 = event->selJets.at(0)->pt;
  float HLTjetPt0 = event->selTrigJets.at(0)->pt;
  //float passHt = (event->ht30 > 100);
  bool passAllButJ75 = (passHLTHtCalo && passHLTHt && pass4J40 && pass3J45 && pass2J60 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4 && pass4J30Calo);
  j0_4j->Fill(jetPt0, HLTjetPt0, (HLT_HT330_4j_75_60_45_40), passAllButJ75, event);

  //
  // Second Jet Pt
  //
  if(event->selJets.size() < 2) return;
  if(event->selTrigJets.size() < 2) return;
  float jetPt1 = event->selJets.at(1)->pt;
  float HLTjetPt1 = event->selTrigJets.at(1)->pt;
  bool passAllBut2J60 = (passHLTHtCalo && passHLTHt && pass4J40 && pass3J45 && passJ75 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4 && pass4J30Calo);
  j1_4j->Fill(jetPt1, HLTjetPt1, (HLT_HT330_4j_75_60_45_40), passAllBut2J60, event);


  //
  // Third Jet Pt
  //
  if(event->selJets.size() < 3) return;
  if(event->selTrigJets.size() < 3) return;
  float jetPt2 = event->selJets.at(2)->pt;
  float HLTjetPt2 = event->selTrigJets.at(2)->pt;
  bool passAllBut3J45 = (passHLTHtCalo && pass4J40 && pass2J60 && passJ75 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4 && pass4J30Calo);
  j2_4j->Fill(jetPt2, HLTjetPt2, (HLT_HT330_4j_75_60_45_40 && passHLTHtCalo), passAllBut3J45, event);


  //
  // Fourth Jet Pt
  //
  if(event->selJets.size() < 4) return;
  if(event->selTrigJets.size() < 4) return;
  //bool EMU_3j_75_60_45 = trigEmulator->GetDecision("EMU_3j");
  float jetPt3 = event->selJets.at(3)->pt;
  float HLTjetPt3 = event->selTrigJets.at(3)->pt;
  bool passAllBut4J45 = (passHLTHtCalo && passHLTHt && pass3J45 && pass2J60 && passJ75 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4 && pass4J30Calo);
  j3_4j->Fill(jetPt3, HLTjetPt3, HLT_HT330_4j_75_60_45_40, passAllBut4J45, event);
  //j3_4j_wrt_HT_3j->Fill(jetPt3, HLTjetPt3, (HLT_HT330_4j_75_60_45_40), trigEmulator->GetDecision("EMU_HT330_3j"), event);


  //
  //  Study jet-level 
  //
  for(const jetPtr& aJet : event->allJets){
    float minDr = 1e6;

    if(fabs(aJet->eta) > 2.4) continue;

    trigPtr matchedTrig;
    
    for(const trigPtr& tJet : event->allTrigJets){
      float thisDr = aJet->p.DeltaR(tJet->p);
      if(thisDr < minDr){
	minDr = thisDr;
	matchedTrig = tJet;
      }
    }
    
    hMinDr->Fill(minDr);

    hAllPt ->Fill(aJet->pt);
    hAllEta->Fill(aJet->eta);
    if(minDr < 0.4){
      hMatchedPt ->Fill(aJet->pt);
      hMatchedEta->Fill(aJet->eta);

      hMatchedPt_jet ->Fill(matchedTrig->pt);
      hMatchedEta_jet->Fill(matchedTrig->eta);

      hMatched_dPt ->Fill(aJet->pt - matchedTrig->pt);
      hMatched_dPt_l ->Fill(aJet->pt - matchedTrig->pt);

      hMatched_dPtL1 ->Fill(aJet->pt - matchedTrig->l1pt);
      hMatched_dPtL1_l ->Fill(aJet->pt - matchedTrig->l1pt);

      if(matchedTrig->pt > 20){
	hMatchedPt_h20->Fill(aJet->pt);
      }

      if(matchedTrig->pt > 25){
	hMatchedPt_h25->Fill(aJet->pt);
      }

      if(matchedTrig->pt > 30){
	hMatchedPt_h30->Fill(aJet->pt);
      }

      if(matchedTrig->pt > 35){
	hMatchedPt_h35->Fill(aJet->pt);
      }


      if(matchedTrig->pt > 40){
	hMatchedPt_h40->Fill(aJet->pt);
	if(matchedTrig->l1pt > 40){
	  hMatchedPt_h40_l40->Fill(aJet->pt);
	}
      }

      if(matchedTrig->pt > 45){
	hMatchedPt_h45->Fill(aJet->pt);
	if(matchedTrig->l1pt > 40){
	  hMatchedPt_h45_l40->Fill(aJet->pt);
	}
      }

      if(matchedTrig->pt > 50){
	hMatchedPt_h50->Fill(aJet->pt);
      }


      if(matchedTrig->pt > 60){
	hMatchedPt_h60->Fill(aJet->pt);
	if(matchedTrig->l1pt > 55){
	  hMatchedPt_h60_l55->Fill(aJet->pt);
	}
      }

      if(matchedTrig->pt > 70){
	hMatchedPt_h70->Fill(aJet->pt);
      }

      if(matchedTrig->pt > 75){
	hMatchedPt_h75->Fill(aJet->pt);
	if(matchedTrig->l1pt > 70){
	  hMatchedPt_h75_l70->Fill(aJet->pt);
	}
      }

      if(matchedTrig->pt > 80){
	hMatchedPt_h80->Fill(aJet->pt);
      }

      if(matchedTrig->pt > 90){
	hMatchedPt_h90->Fill(aJet->pt);
      }

      if(matchedTrig->pt > 100){
	hMatchedPt_h100->Fill(aJet->pt);
      }
      

    }


  }


  if(debug) cout << "Left Fill " << endl;  
  return;
}

triggerStudy::~triggerStudy(){} 

