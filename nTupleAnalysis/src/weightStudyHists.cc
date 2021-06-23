#include "ZZ4b/nTupleAnalysis/interface/weightStudyHists.h"

using namespace nTupleAnalysis;
using std::cout; using std::endl; 


void weightStudyHists::hists::Fill(eventData* event, std::shared_ptr<eventView> &view, weightStudyHists* mother){

  float FvT1 = *(event->classifierVariables[mother->weightName1]);
  float FvT2 = *(event->classifierVariables[mother->weightName2]);
  float dFvT = FvT1-FvT2;
  float dFvTFrac = dFvT/FvT1;

  FvT_1->Fill(FvT1, event->weight);
  FvT_2->Fill(FvT2, event->weight);

  //float dFvT = *(event->classifierVariables["weight_FvT_3bMix4b_rWbW2_v0"]);
  deltaFvT  ->Fill(dFvT, event->weight);
  deltaFvT_l->Fill(dFvT, event->weight);

  deltaFvTfrac  ->Fill(dFvTFrac, event->weight);
  deltaFvTfrac_l->Fill(dFvTFrac, event->weight);

  FvT_vs_FvT->Fill(FvT1, FvT2, event->weight);
  FvT_vs_SvB    ->Fill(event->SvB_ps, event->FvT,     event->weight);

  dFvT_vs_FvT    ->Fill(event->FvT, dFvT,     event->weight);
  dFvTFrac_vs_FvT->Fill(event->FvT, dFvTFrac, event->weight);

  dFvT_vs_SvB    ->Fill(event->SvB_ps, dFvT,     event->weight);
  dFvTFrac_vs_SvB->Fill(event->SvB_ps, dFvTFrac, event->weight);

  if(event->SvB_pzz<event->SvB_pzh){
    dFvT_vs_SvB_zh->Fill(event->SvB_ps, dFvT, event->weight);
    dFvTFrac_vs_SvB_zh->Fill(event->SvB_ps, dFvTFrac, event->weight);
  }else{
    dFvT_vs_SvB_zz->Fill(event->SvB_ps, dFvT, event->weight);
    dFvTFrac_vs_SvB_zz->Fill(event->SvB_ps,dFvTFrac,  event->weight);
  }

  return;
}


weightStudyHists::weightStudyHists(std::string name, fwlite::TFileService& fs, std::string _weightName1, std::string _weightName2, bool _debug) {
  debug = _debug;

  weightName1 = _weightName1;
  weightName2 = _weightName2;

  hinclusive = new hists(name,fs);
  h0p98      = new hists(name+"_0p98",fs);
  h0p95      = new hists(name+"_0p95",fs);
  h0p90      = new hists(name+"_0p90",fs);

} 

void weightStudyHists::Fill(eventData* event, std::shared_ptr<eventView> &view){

  hinclusive->Fill(event, view, this);

  if(event->SvB_ps > 0.98)
    h0p98->Fill(event,view, this);

  if(event->SvB_ps > 0.95)
    h0p95->Fill(event,view, this);

  if(event->SvB_ps > 0.90)
    h0p90->Fill(event,view, this);


  if(debug) std::cout << "weightStudyHists::Fill done " << std::endl;
  return;
}

weightStudyHists::~weightStudyHists(){} 

