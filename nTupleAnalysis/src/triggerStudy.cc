#include "ZZ4b/nTupleAnalysis/interface/triggerStudy.h"

using namespace nTupleAnalysis;

using std::cout;  using std::endl;
using std::vector;

triggerStudy::triggerStudy(std::string name, fwlite::TFileService& fs, std::string year, bool isMC, bool blind, std::string histDetailLevel, bool _debug ) {
  std::cout << "Initialize >> triggerStudy: " << name << std::endl;

  debug = _debug;
  
  hist_HLT_OR       = new   tagHists(name+"HLT_OR",        fs, true,  isMC, blind, histDetailLevel, debug);
  hist_HLT_4j_3b    = new   tagHists(name+"HLT_4j_3b",     fs, true,  isMC, blind, histDetailLevel, debug);
  hist_HLT_2b       = new   tagHists(name+"HLT_2b",        fs, true,  isMC, blind, histDetailLevel, debug);
  if (year == "2016")
    hist_HLT_2j_2j_3b = new   tagHists(name+"HLT_2j_2j_3b",  fs, true,  isMC, blind, histDetailLevel, debug);


  hist_EMU_OR       = new   tagHists(name+"EMU_OR",        fs, true,  isMC, blind, histDetailLevel, debug);
  hist_EMU_4j_3b    = new   tagHists(name+"EMU_4j_3b",     fs, true,  isMC, blind, histDetailLevel, debug);
  hist_EMU_2b       = new   tagHists(name+"EMU_2b",        fs, true,  isMC, blind, histDetailLevel, debug);
  if (year == "2016")
    hist_EMU_2j_2j_3b = new   tagHists(name+"EMU_2j_2j_3b",  fs, true,  isMC, blind, histDetailLevel, debug);

} 

void triggerStudy::Fill(eventData* event){
  if(debug) cout << "In Fill " << endl;

  //
  //  triggerStudy is run with doTrigEmulation, remove the trigger weight first
  //
  if(debug) cout << "Weight was " << event->weight << " trig weight " << event->trigWeight << endl;
  float eventWeight_init = event->weight;
  float eventWeight_noTrig = event->trigWeight ? event->weight / event->trigWeight : 0;
  event->weight = eventWeight_noTrig;
  if(debug) cout << " \t now " << event->weight << endl;

  //
  // Set HLT decisions
  //
  bool HLT_4j_3b    = false;
  bool HLT_2b       = false;
  bool HLT_2j_2j_3b = false;

  if(event->year == 2016){
    HLT_4j_3b    = event->HLT_triggers["HLT_QuadJet45_TripleBTagCSV_p087"];
    HLT_2j_2j_3b = event->HLT_triggers["HLT_DoubleJet90_Double30_TripleBTagCSV_p087"];
    HLT_2b       = event->HLT_triggers["HLT_DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6"];
  }


  if(event->year == 2017){
    HLT_4j_3b = event->HLT_triggers["HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0"];
    HLT_2b    = event->HLT_triggers["HLT_DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33"];
  }

  if(event->year == 2018){
    HLT_4j_3b = event->HLT_triggers["HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5"];
    HLT_2b    = event->HLT_triggers["HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71"];
  }

  bool HLT_OR = HLT_4j_3b || HLT_2b || HLT_2j_2j_3b;
  

  //
  // Fill histograms
  //
  if(HLT_4j_3b)    hist_HLT_4j_3b    ->Fill(event, event->views_passMDRs);
  if(HLT_2b)       hist_HLT_2b       ->Fill(event, event->views_passMDRs);
  if(HLT_2j_2j_3b) hist_HLT_2j_2j_3b ->Fill(event, event->views_passMDRs);
  if(HLT_OR)       hist_HLT_OR       ->Fill(event, event->views_passMDRs);


  //
  // Set Emulation decisions
  //
  vector<float> selJet_pts;
  for(const jetPtr& sJet : event->selJets){
    selJet_pts.push_back(sJet->pt_wo_bRegCorr);
  }

  vector<float> tagJet_pts;
  unsigned int nTagJets = 0;
  for(const jetPtr& tJet : event->tagJets){
    tagJet_pts.push_back(tJet->pt_wo_bRegCorr);
  }

  event->trigEmulators.at(0)->SetWeights(selJet_pts, tagJet_pts, event->ht30);

  //
  // Fill histograms
  //
  event->weight = eventWeight_noTrig * event->trigEmulators.at(0)->GetWeight("EMU_4j_3b");
  hist_EMU_4j_3b    ->Fill(event, event->views_passMDRs);

  event->weight = eventWeight_noTrig * event->trigEmulators.at(0)->GetWeight("EMU_2b");
  hist_EMU_2b       ->Fill(event, event->views_passMDRs);

  if(hist_EMU_2j_2j_3b){
    event->weight = eventWeight_noTrig * event->trigEmulators.at(0)->GetWeight("EMU_2j_2j_3b");
    hist_EMU_2j_2j_3b ->Fill(event, event->views_passMDRs);
  }

  //
  //  Putting back the trigger weight
  //
  event->weight = eventWeight_init;
  hist_EMU_OR       ->Fill(event, event->views_passMDRs);

  if(debug) cout << " \t final " << event->weight << endl;
  if(debug) cout << "Left Fill " << endl;  
  return;
}

triggerStudy::~triggerStudy(){} 

