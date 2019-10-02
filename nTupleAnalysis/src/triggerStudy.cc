#include "ZZ4b/nTupleAnalysis/interface/triggerStudy.h"

using namespace nTupleAnalysis;

using std::cout;  using std::endl;
using std::vector;

triggerStudy::triggerStudy(std::string name, fwlite::TFileService& fs, bool _debug) {
  std::cout << "Initialize >> triggerStudy: " << name << std::endl;

  dir = fs.mkdir(name);
  debug = _debug;
  
  hT30_ht330            = new turnOnHist("ht330",            name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_L1HT360    = new turnOnHist("ht330_L1HT360",    name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_L1ETT2000  = new turnOnHist("ht330_L1ETT200",   name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_L1HT320_4j = new turnOnHist("ht330_L1HT320_4j", name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_L1OR       = new turnOnHist("ht330_L1OR",       name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);

  hT30_ht330_wrt_L1OR   = new turnOnHist("ht330_wrt_L1OR",   name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);

  hT30_ht330_3tag       = new turnOnHist("ht330_3tag",       name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_sel       = new turnOnHist("ht330_sel",       name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_sel_noSubSet   = new turnOnHist("ht330_sel_noSubSet",       name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_ht330_sel_3tag       = new turnOnHist("ht330_sel_3tag",       name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);

  hT30_L1HT360    = new turnOnHist("L1HT360",    name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_L1ETT2000  = new turnOnHist("L1ETT2000",  name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_L1HT320_4j = new turnOnHist("L1HT320_4j", name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);
  hT30_L1OR       = new turnOnHist("L1OR",       name, dir, "hT [GeV] (jet Pt > 30 GeV)",200,0,2000);

  hT30_all  = dir.make<TH1F>("hT30_all",  (name+"/hT30_all;  hT [GeV] (jet Pt > 30 GeV); Entries").c_str(),  200,0,2000);

  //
  //  Jet Turn Ons
  //

  j0_4j           = new turnOnHist("j0_4j",       name, dir, "1st Jet PT [GeV] ",100,0,200);
  j1_4j           = new turnOnHist("j1_4j",       name, dir, "2nd Jet PT [GeV] ",100,0,200);
  j2_4j           = new turnOnHist("j2_4j",       name, dir, "3rd Jet PT [GeV] ",100,0,200);
  j3_4j           = new turnOnHist("j3_4j",       name, dir, "4th Jet PT [GeV] ",100,0,200);
  j3_4j_wrt_HT_3j = new turnOnHist("j3_4j_wrt_HT_3j",       name, dir, "4th Jet PT [GeV] ",100,0,200);

  //
  // The Trig Emulator
  //
  int nToys = 100;
  trigEmulator = new TriggerEmulator::TrigEmulatorTool("trigEmulator", 1, nToys);
  trigEmulator->AddTrig("EMU_4j_3Tag", {"75","60","45","40"}, {1,2,3,4},"2018",3);
  trigEmulator->AddTrig("EMU_4j", {"75","60","45","40"}, {1,2,3,4});
  trigEmulator->AddTrig("EMU_3j", {"75","60","45"},      {1,2,3});
  trigEmulator->AddTrig("EMU_HT330_3j", "330", {"75","60","45"},      {1,2,3});


} 

void triggerStudy::Fill(eventData* event){
  if(debug) cout << "In Fill " << endl;


  vector<float> allJet_pts;
  for(const jetPtr& aJet : event->allJets){
    allJet_pts.push_back(aJet->pt_wo_bRegCorr);
  }

  vector<float> tagJet_pts;
  unsigned int nTagJets = 0;
  for(const jetPtr& tJet : event->tagJets){
    if(nTagJets > 3) continue;
    ++nTagJets;
    tagJet_pts.push_back(tJet->pt_wo_bRegCorr);
  }

  trigEmulator->SetDecisions(allJet_pts, tagJet_pts, event->ht30);
  bool EMU_4j_75_60_45_40 = trigEmulator->GetDecision("EMU_4j");
  bool EMU_4j_75_60_45_40_3b = trigEmulator->GetDecision("EMU_4j_3Tag");
  bool HLT_HT330_4j_75_60_45_40 = event->HLT_HT330_4j_75_60_45_40;
  //bool HLT_HT330_4j_75_60_45_40_3b = event->HLT_HT330_4j_75_60_45_40_3b;

  hT30_all->Fill(event->ht30);

  if(debug) cout << "Filling ht30 " << endl;  

  bool L1_OR = (event->L1_HTT360er || event->L1_ETT2000 || event->L1_HTT320er_QuadJet_70_55_40_40_er2p4);

  //
  //  hT
  //
  float ht30 = event->ht30;
  hT30_ht330            ->Fill(ht30,  HLT_HT330_4j_75_60_45_40,                           EMU_4j_75_60_45_40, event );
  hT30_ht330_L1HT360    ->Fill(ht30, (HLT_HT330_4j_75_60_45_40 && event->L1_HTT360er),    EMU_4j_75_60_45_40, event );
  hT30_ht330_L1ETT2000  ->Fill(ht30, (HLT_HT330_4j_75_60_45_40 && event->L1_ETT2000),     EMU_4j_75_60_45_40, event );
  hT30_ht330_L1HT320_4j ->Fill(ht30, (HLT_HT330_4j_75_60_45_40 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4), EMU_4j_75_60_45_40, event );
  hT30_ht330_L1OR       ->Fill(ht30, (HLT_HT330_4j_75_60_45_40 && L1_OR), EMU_4j_75_60_45_40, event );

  hT30_ht330_wrt_L1OR  ->Fill(ht30, (HLT_HT330_4j_75_60_45_40 && L1_OR), (L1_OR && EMU_4j_75_60_45_40), event );

  hT30_ht330_3tag       ->Fill(ht30,  HLT_HT330_4j_75_60_45_40,                           EMU_4j_75_60_45_40_3b, event );

  //cout<< " In trig study: " << (event->fourTag  || event->threeTag) << " " << event->passDijetMass << " " << event->passMDRs << endl;

  if(event->fourTag || event->threeTag){
    if(event->passDijetMass){
      event->applyMDRs(); 
      if(event->passMDRs){
	hT30_ht330_sel       ->Fill(ht30,  HLT_HT330_4j_75_60_45_40,                           EMU_4j_75_60_45_40,    event );
	hT30_ht330_sel_noSubSet   ->Fill(ht30,  HLT_HT330_4j_75_60_45_40,                           EMU_4j_75_60_45_40,    event, false );
	hT30_ht330_sel_3tag  ->Fill(ht30,  HLT_HT330_4j_75_60_45_40,                           EMU_4j_75_60_45_40_3b, event );
      }
    }
  }
  
  hT30_L1HT360    -> Fill(ht30, (event->L1_HTT360er), true, event );
  hT30_L1ETT2000  -> Fill(ht30, (event->L1_ETT2000), true, event );
  hT30_L1HT320_4j -> Fill(ht30, (event->L1_HTT320er_QuadJet_70_55_40_40_er2p4), true, event );
  hT30_L1OR       -> Fill(ht30, (L1_OR), true, event );


  //
  //  lead Jet Pt
  //
  if(event->selJets.size() < 1) return;

  float jetPt0 = event->selJets.at(0)->pt;
  //float passHt = (event->ht30 > 100);

  j0_4j->Fill(jetPt0, HLT_HT330_4j_75_60_45_40, true, event);

  //
  // Second Jet Pt
  //
  if(event->selJets.size() < 2) return;

  float jetPt1 = event->selJets.at(1)->pt;
  j1_4j->Fill(jetPt1, HLT_HT330_4j_75_60_45_40, true, event);


  //
  // Third Jet Pt
  //
  if(event->selJets.size() < 3) return;

  float jetPt2 = event->selJets.at(2)->pt;
  j2_4j->Fill(jetPt2, HLT_HT330_4j_75_60_45_40, true, event);


  //
  // Fourth Jet Pt
  //
  if(event->selJets.size() < 4) return;
  bool EMU_3j_75_60_45 = trigEmulator->GetDecision("EMU_3j");
  float jetPt3 = event->selJets.at(3)->pt;
  j3_4j->Fill(jetPt3, HLT_HT330_4j_75_60_45_40, EMU_3j_75_60_45, event);
  j3_4j_wrt_HT_3j->Fill(jetPt3, HLT_HT330_4j_75_60_45_40, trigEmulator->GetDecision("EMU_HT330_3j"), event);

  if(debug) cout << "Left Fill " << endl;  
  return;
}

triggerStudy::~triggerStudy(){} 

