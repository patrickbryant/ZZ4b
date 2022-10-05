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
#include "nTupleAnalysis/baseClasses/interface/elecHists.h"
#include "nTupleAnalysis/baseClasses/interface/dijetHists.h"
#include "nTupleAnalysis/baseClasses/interface/trijetHists.h"
#include "nTupleAnalysis/baseClasses/interface/trigHists.h"
#include "nTupleAnalysis/baseClasses/interface/systHists.h"
#include "ZZ4b/nTupleAnalysis/interface/weightStudyHists.h"


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
    TH1F*     nTagJetsUnweighted;
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
    TH1F* mjjOther;
    TH1F* ptjjOther;

    trigHists*  allTrigJets = NULL;

    TH1F* nAllMuons;
    TH1F* nIsoMed25Muons;
    TH1F* nIsoMed40Muons;
    muonHists* allMuons;
    muonHists* muons_isoMed25;
    muonHists* muons_isoMed40;

    TH1F* nAllElecs;
    TH1F* nIsoMed25Elecs;
    TH1F* nIsoMed40Elecs;
    elecHists* allElecs;
    elecHists* elecs_isoMed25;
    elecHists* elecs_isoMed40;


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
    // TH1F* m123;
    // TH1F* m023;
    // TH1F* m013;
    // TH1F* m012;
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
    TH1F* FvT_std;
    TH1F* FvT_ferr;
    TH1F* SvB_ps;
    TH1F* SvB_pzz;
    TH1F* SvB_pzh;
    TH1F* SvB_phh;
    TH1F* SvB_ptt;
    TH1F* SvB_ps_hh;
    TH1F* SvB_ps_zh;
    TH1F* SvB_ps_zz;
    systHists* SvB_ps_hh_bTagSysts;
    systHists* SvB_ps_zh_bTagSysts;
    systHists* SvB_ps_zz_bTagSysts;
    TH1F* SvB_MA_ps;
    TH1F* SvB_MA_pzz;
    TH1F* SvB_MA_pzh;
    TH1F* SvB_MA_phh;
    TH1F* SvB_MA_ptt;
    TH1F* SvB_MA_ps_hh;
    TH1F* SvB_MA_ps_zh;
    TH1F* SvB_MA_ps_zz;
    bool bTagSysts = false;
    systHists* SvB_MA_ps_hh_bTagSysts;
    systHists* SvB_MA_ps_zh_bTagSysts;
    systHists* SvB_MA_ps_zz_bTagSysts;
    //TH1F* SvB_MA_VHH_pskl    = NULL;
    //TH1F* SvB_MA_VHH_plkl    = NULL;
    TH1F* SvB_MA_VHH_ps      = NULL;
    TH1F* SvB_MA_VHH_ps_sbdt = NULL;
    TH1F* SvB_MA_VHH_ps_lbdt = NULL;

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

    TH1F* otherWeight;

    TH1F* FvT_q_score;
    TH1F* FvT_q_score_dR_min;
    TH1F* FvT_q_score_SvB_q_score_max;
    TH1F* SvB_q_score;
    TH1F* SvB_q_score_FvT_q_score_max;
    TH1F* SvB_MA_q_score;

    TH1F* FvT_SvB_q_score_max_same;

    TH2F* m4j_vs_nViews_eq;
    TH2F* m4j_vs_nViews_00;
    TH2F* m4j_vs_nViews_01;
    TH2F* m4j_vs_nViews_02;
    TH2F* m4j_vs_nViews_10;
    TH2F* m4j_vs_nViews_11;
    TH2F* m4j_vs_nViews_12;
    
    TH1F* truthM4b;
    TH2F* truthM4b_vs_mZH;
    TH1F* nTrueBJets;

    weightStudyHists* weightStudy_v0v1  = NULL;
    weightStudyHists* weightStudy_v0v9  = NULL;
    weightStudyHists* weightStudy_os012 = NULL;
    weightStudyHists* weightStudy_e20   = NULL;

    TH1F* DvT_pt = NULL;
    TH1F* DvT_pt_l = NULL;

    TH1F* DvT_pm = NULL;
    TH1F* DvT_pm_l = NULL;

    TH1F* DvT_raw = NULL;

    TH1F* bdtScore = NULL;

    viewHists(std::string, fwlite::TFileService&, bool isMC = false, bool _debug = false, eventData* event = NULL, std::string histDetailLevel="");
    void Fill(eventData*, std::shared_ptr<eventView>&);//, int nViews=-1, int nViews_10=-1, int nViews_11=-1, int nViews_12=-1);
    ~viewHists(); 

  };

}
#endif // viewHists_H
