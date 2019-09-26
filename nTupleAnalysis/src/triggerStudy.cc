#include "ZZ4b/nTupleAnalysis/interface/triggerStudy.h"

using namespace nTupleAnalysis;

using std::cout;  using std::endl;

triggerStudy::triggerStudy(std::string name, fwlite::TFileService& fs, bool _debug) {
  std::cout << "Initialize >> triggerStudy: " << name << std::endl;

  dir = fs.mkdir(name);
  debug = _debug;

  hT30_ht330 = new turnOnHist("ht330", name, dir, "hT [GeV] (jet Pt > 30 GeV)");
  hT30_ht330_L1HT360 = new turnOnHist("ht330_L1HT360", name, dir, "hT [GeV] (jet Pt > 30 GeV)");
  hT30_ht330_L1ETT2000 = new turnOnHist("ht330_L1ETT200", name, dir, "hT [GeV] (jet Pt > 30 GeV)");
  hT30_ht330_L1HT320_4j = new turnOnHist("ht330_L1HT320_4j", name, dir, "hT [GeV] (jet Pt > 30 GeV)");
  hT30_ht330_L1OR       = new turnOnHist("ht330_L1OR", name, dir, "hT [GeV] (jet Pt > 30 GeV)");

  hT30_L1HT360    = new turnOnHist("L1HT360",    name, dir, "hT [GeV] (jet Pt > 30 GeV)");
  hT30_L1ETT2000  = new turnOnHist("L1ETT2000",  name, dir, "hT [GeV] (jet Pt > 30 GeV)");
  hT30_L1HT320_4j = new turnOnHist("L1HT320_4j", name, dir, "hT [GeV] (jet Pt > 30 GeV)");
  hT30_L1OR       = new turnOnHist("L1OR",       name, dir, "hT [GeV] (jet Pt > 30 GeV)");

  hT30_all  = dir.make<TH1F>("hT30_all",  (name+"/hT30_all;  hT [GeV] (jet Pt > 30 GeV); Entries").c_str(),  200,0,2000);
} 

void triggerStudy::Fill(eventData* event){
  if(debug) cout << "In Fill " << endl;
  hT30_all->Fill(event->ht30);

  if(debug) cout << "Filling ht30 " << endl;  

  bool L1_OR = (event->L1_HTT360er || event->L1_ETT2000 || event->L1_HTT320er_QuadJet_70_55_40_40_er2p4 || event->L1_HTT280er);

  hT30_ht330            ->Fill(event->ht30,  event->HLT_HT330_4j_75_60_45_40,                           event->EMU_4j_75_60_45_40, event );
  hT30_ht330_L1HT360    ->Fill(event->ht30, (event->HLT_HT330_4j_75_60_45_40 && event->L1_HTT360er),    event->EMU_4j_75_60_45_40, event );
  hT30_ht330_L1ETT2000  ->Fill(event->ht30, (event->HLT_HT330_4j_75_60_45_40 && event->L1_ETT2000),     event->EMU_4j_75_60_45_40, event );
  hT30_ht330_L1HT320_4j ->Fill(event->ht30, (event->HLT_HT330_4j_75_60_45_40 && event->L1_HTT320er_QuadJet_70_55_40_40_er2p4), event->EMU_4j_75_60_45_40, event );
  hT30_ht330_L1OR       ->Fill(event->ht30, (event->HLT_HT330_4j_75_60_45_40 && L1_OR), event->EMU_4j_75_60_45_40, event );
  
  hT30_L1HT360    -> Fill(event->ht30, (event->L1_HTT360er), true, event );
  hT30_L1ETT2000  -> Fill(event->ht30, (event->L1_ETT2000), true, event );
  hT30_L1HT320_4j -> Fill(event->ht30, (event->L1_HTT320er_QuadJet_70_55_40_40_er2p4), true, event );
  hT30_L1OR -> Fill(event->ht30, (L1_OR), true, event );




  if(debug) cout << "Left Fill " << endl;  
  return;
}

triggerStudy::~triggerStudy(){} 

