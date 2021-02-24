// -*- C++ -*-
#if !defined(weightStudyHists_H)
#define weightStudyHists_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/eventView.h"

namespace nTupleAnalysis {

  class weightStudyHists {
  public:

    bool debug;
    
    struct hists { 

      TH1F*     deltaFvT;
      TH1F*     deltaFvTfrac;
      TH1F*     deltaFvT_l;
      TH1F*     deltaFvTfrac_l;
    
      TH2F*     FvT_vs_FvT;
      TH2F*     FvT_vs_SvB;
      TH2F*     dFvT_vs_FvT;
      TH2F*     dFvT_vs_SvB;
      TH2F*     dFvT_vs_SvB_zz;
      TH2F*     dFvT_vs_SvB_zh;
    
      TH2F*     dFvTFrac_vs_FvT;
      TH2F*     dFvTFrac_vs_SvB;
      TH2F*     dFvTFrac_vs_SvB_zz;
      TH2F*     dFvTFrac_vs_SvB_zh;
      
      //hists(std::string name, std::string dirName, fwlite::TFileService& fs);

      hists(std::string dirName, fwlite::TFileService& fs) {
	TFileDirectory 	dir = fs.mkdir(dirName);

	deltaFvT       = dir.make<TH1F>("deltaFvT",       (dirName+"/deltaFvT; #Delta FvT; Entries").c_str(),  100,-3,3);
	deltaFvTfrac   = dir.make<TH1F>("deltaFvTfrac",   (dirName+"/deltaFvTfrac; #Delta FvT / FvT; Entries").c_str(),  100,-1,1);
	deltaFvT_l     = dir.make<TH1F>("deltaFvT_l",     (dirName+"/deltaFvT_l; #Delta FvT; Entries").c_str(),  100,-10,10);
	deltaFvTfrac_l = dir.make<TH1F>("deltaFvTfrac_l", (dirName+"/deltaFvTfrac_l; #Delta FvT; Entries").c_str(),  100,-5,5);
	FvT_vs_FvT     = dir.make<TH2F>("FvT_vs_FvT",     (dirName+"/FvT_vs_FvT; FvT_{1}; FvT_{2}; Entries").c_str(), 50,0,5, 50,0,5);
	FvT_vs_SvB     = dir.make<TH2F>("FvT_vs_SvB",     (dirName+"/FvT_vs_SvB; SvB; FvT; Entries").c_str(), 50,0,1, 50,0,3);
    
	dFvT_vs_FvT    = dir.make<TH2F>("dFvT_vs_FvT",    (dirName+"/dFvT_vs_FvT; FvT; #Delta FvT;  Entries"  ).c_str(), 50,0,3, 50,-1,1);
	dFvT_vs_SvB    = dir.make<TH2F>("dFvT_vs_SvB",    (dirName+"/dFvT_vs_SvB; SvB; #Delta FvT; Entries"   ).c_str(), 50,0,1, 50,-1,1);
	dFvT_vs_SvB_zz = dir.make<TH2F>("dFvT_vs_SvB_zz", (dirName+"/dFvT_vs_SvB_zz; SvB; #Delta FvT; Entries").c_str(), 50,0,1, 50,-1,1);
	dFvT_vs_SvB_zh = dir.make<TH2F>("dFvT_vs_SvB_zh", (dirName+"/dFvT_vs_SvB_zh; SvB; #Delta FvT; Entries").c_str(), 50,0,1, 50,-1,1);
    
	dFvTFrac_vs_FvT    = dir.make<TH2F>("dFvTFrac_vs_FvT",    (dirName+"/dFvTFrac_vs_FvT; FvT; #Delta FvT / FvT; Entries"   ).c_str(), 50,0,3, 50,-1,1);
	dFvTFrac_vs_SvB    = dir.make<TH2F>("dFvTFrac_vs_SvB",    (dirName+"/dFvTFrac_vs_SvB; SvB; #Delta FvT / FvT; Entries"   ).c_str(), 50,0,1, 50,-1,1);
	dFvTFrac_vs_SvB_zz = dir.make<TH2F>("dFvTFrac_vs_SvB_zz", (dirName+"/dFvTFrac_vs_SvB_zz; SvB; #Delta FvT / FvT; Entries").c_str(), 50,0,1, 50,-1,1);
	dFvTFrac_vs_SvB_zh = dir.make<TH2F>("dFvTFrac_vs_SvB_zh", (dirName+"/dFvTFrac_vs_SvB_zh; SvB; #Delta FvT / FvT; Entries").c_str(), 50,0,1, 50,-1,1);
      }

      void Fill(eventData* event, std::unique_ptr<eventView> &view, weightStudyHists* mother);

    };

    hists* hinclusive = NULL; 
    hists* h0p98 = NULL; 
    hists* h0p95 = NULL; 
    hists* h0p90 = NULL; 

    std::string weightName1; 
    std::string weightName2; 

    weightStudyHists(std::string, fwlite::TFileService&, std::string _weightName1, std::string _weightName2,  bool _debug = false);
    void Fill(eventData*, std::unique_ptr<eventView>&);
    ~weightStudyHists(); 

  };

}
#endif // weightStudyHists_H
