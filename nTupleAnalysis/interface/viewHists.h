// -*- C++ -*-
#if !defined(viewHists_H)
#define viewHists_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/eventView.h"
#include "nTupleAnalysis/baseClasses/interface/fourVectorHists.h"
#include "nTupleAnalysis/baseClasses/interface/jetHists.h"
#include "nTupleAnalysis/baseClasses/interface/muonHists.h"
#include "nTupleAnalysis/baseClasses/interface/dijetHists.h"
#include "nTupleAnalysis/baseClasses/interface/trijetHists.h"
#include "nTupleAnalysis/baseClasses/interface/trigHists.h"
#include "nTupleAnalysis/baseClasses/interface/systHists.h"

namespace nTupleAnalysis {

  class viewHists {
  public:
    TFileDirectory dir;
    bool debug;
    
    // Object Level
    TH1F*     nAllJets;
    TH1F*     nAllNotCanJets;
    TH1F*     st;
    TH1F*     nSelJets;
    TH1F*     nSelJets_noBTagSF;
    TH1F*     nSelJets_lowSt;
    TH1F*     nSelJets_midSt;
    TH1F*     nSelJets_highSt;
    TH1F*     nSelJetsUnweighted;
    TH1F*     nSelJetsUnweighted_lowSt;
    TH1F*     nSelJetsUnweighted_midSt;
    TH1F*     nSelJetsUnweighted_highSt;
    TH1F*     nTagJets;
    TH1F*     nPSTJets;
    TH1F*     nPSTJets_lowSt;
    TH1F*     nPSTJets_midSt;
    TH1F*     nPSTJets_highSt;
    TH1F*     nPSTJetsUnweighted;
    TH1F*     nCanJets;
    //jetHists*  allJets;
    jetHists*  selJets;
    jetHists*  tagJets;
    jetHists*  canJets;
    jetHists*  canJet0;
    jetHists*  canJet1;
    jetHists*  canJet2;
    jetHists*  canJet3;
    jetHists*  othJets;
    jetHists*  allNotCanJets;
    TH1F* aveAbsEta;
    TH1F* aveAbsEtaOth;
    TH1F* stNotCan;

    trigHists*  allTrigJets = NULL;

    TH1F* nAllMuons;
    TH1F* nIsoMuons;
    muonHists* allMuons;
    muonHists* isoMuons;

    dijetHists* lead;
    dijetHists* subl;
    TH2F* lead_m_vs_subl_m;

    dijetHists* leadSt;
    dijetHists* sublSt;
    TH2F* leadSt_m_vs_sublSt_m;
    TH2F* m4j_vs_leadSt_dR;
    TH2F* m4j_vs_sublSt_dR;

    dijetHists* leadM;
    dijetHists* sublM;
    TH2F* leadM_m_vs_sublM_m;

    dijetHists* close;
    dijetHists* other;
    TH2F* close_m_vs_other_m;

    // Event Level
    TH1F* nPVs;
    TH1F* nPVsGood;    
    fourVectorHists* v4j;
    TH1F* s4j;
    TH1F* r4j;
    TH1F* m123;
    TH1F* m023;
    TH1F* m013;
    TH1F* m012;
    TH1F* dBB;
    TH1F* dEtaBB;
    TH1F* dRBB;
    TH1F* xZZ;
    TH1F* mZZ;
    TH1F* xZH;
    TH1F* mZH;
    TH1F* xHH;
    TH1F* mHH;

    TH1F* hT;
    TH1F* hT30;
    TH1F* L1hT;
    TH1F* L1hT30;
    TH1F* HLThT;
    TH1F* HLThT30;

    TH1F* xWt0;
    TH1F* xWt1;
    //TH1F* xWt2;
    TH1F* xWt;
    trijetHists* t0;
    trijetHists* t1;
    //trijetHists* t2;
    trijetHists* t;

    TH1F* FvT;
    TH1F* FvTUnweighted;
    TH1F* FvT_pd4;
    TH1F* FvT_pd3;
    TH1F* FvT_pt4;
    TH1F* FvT_pt3;
    TH1F* FvT_pm4;
    TH1F* FvT_pm3;
    TH1F* FvT_pt;
    TH1F* SvB_ps;
    TH1F* SvB_pzz;
    TH1F* SvB_pzh;
    TH1F* SvB_ptt;
    TH1F* SvB_ps_zh;
    TH1F* SvB_ps_zz;
    systHists* SvB_ps_bTagSysts = NULL;

    //Simplified template cross section binning https://cds.cern.ch/record/2669925/files/1906.02754.pdf
    TH1F* SvB_ps_zh_0_75;
    TH1F* SvB_ps_zh_75_150;
    TH1F* SvB_ps_zh_150_250;
    TH1F* SvB_ps_zh_250_400;
    TH1F* SvB_ps_zh_400_inf;

    TH1F* SvB_ps_zz_0_75;
    TH1F* SvB_ps_zz_75_150;
    TH1F* SvB_ps_zz_150_250;
    TH1F* SvB_ps_zz_250_400;
    TH1F* SvB_ps_zz_400_inf;

    TH1F* FvT_q_score;
    TH1F* SvB_q_score;

    TH2F* m4j_vs_nViews;
    
    TH1F* truthM4b;
    TH2F* truthM4b_vs_mZH;
    TH1F* nTrueBJets;

    viewHists(std::string, fwlite::TFileService&, bool isMC = false, bool _debug = false, eventData* event = NULL);
    void Fill(eventData*, std::unique_ptr<eventView>&);
    ~viewHists(); 

  };

}
#endif // viewHists_H
