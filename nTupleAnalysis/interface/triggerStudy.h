// -*- C++ -*-
#if !defined(triggerStudy_H)
#define triggerStudy_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "TriggerEmulator/nTupleAnalysis/interface/TrigEmulatorTool.h"

using namespace nTupleAnalysis;

namespace nTupleAnalysis {

  class triggerStudy {
  public:

    triggerStudy(std::string, fwlite::TFileService&, bool _debug = false);
    void Fill(eventData* event);
    ~triggerStudy(); 

  private:

    TFileDirectory dir;
    bool debug;
    TriggerEmulator::TrigEmulatorTool* trigEmulator;

    //
    //  Histograms 
    //
    struct turnOnHist {
      std::string name = "";
      TH1F* hIncl = NULL;
      TH1F* hPass = NULL;
      TH1F* hIncl_hlt = NULL;
      TH1F* hPass_hlt = NULL;


      TH1F* hIncl_4b = NULL;
      TH1F* hPass_4b = NULL;
      TH1F* hIncl_4b_hlt = NULL;
      TH1F* hPass_4b_hlt = NULL;


      TH1F* hIncl_3b = NULL;
      TH1F* hPass_3b = NULL;
      TH1F* hIncl_3b_hlt = NULL;
      TH1F* hPass_3b_hlt = NULL;

      TFileDirectory dir;

      turnOnHist(std::string _name, std::string dirName, fwlite::TFileService& fs,   std::string xTitle, unsigned int nBins, float xMin, float xMax) : name(_name) 
      {
	dir = fs.mkdir(dirName+"/"+name);
	hIncl  = dir.make<TH1F>((name+"_incl").c_str(),  (dirName+"/"+name+"/all;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass  = dir.make<TH1F>((name+"_pass").c_str(), (dirName+"/"+name+"/incl;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);

	hIncl_4b  = dir.make<TH1F>((name+"_incl_4b").c_str(),  (dirName+"/"+name+"/all_4b;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass_4b  = dir.make<TH1F>((name+"_pass_4b").c_str(), (dirName+"/"+name+"/incl_4b;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);

	hIncl_3b  = dir.make<TH1F>((name+"_incl_3b").c_str(),  (dirName+"/"+name+"/all_3b;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass_3b  = dir.make<TH1F>((name+"_pass_3b").c_str(), (dirName+"/"+name+"/incl_3b;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);

	hIncl_hlt  = dir.make<TH1F>((name+"_incl_hlt").c_str(),  (dirName+"/"+name+"/all_hlt;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass_hlt  = dir.make<TH1F>((name+"_pass_hlt").c_str(), (dirName+"/"+name+"/incl_hlt;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);

	hIncl_4b_hlt  = dir.make<TH1F>((name+"_incl_4b_hlt").c_str(),  (dirName+"/"+name+"/all_4b_hlt;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass_4b_hlt  = dir.make<TH1F>((name+"_pass_4b_hlt").c_str(), (dirName+"/"+name+"/incl_4b_hlt;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);

	hIncl_3b_hlt  = dir.make<TH1F>((name+"_incl_3b_hlt").c_str(),  (dirName+"/"+name+"/all_3b_hlt;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass_3b_hlt  = dir.make<TH1F>((name+"_pass_3b_hlt").c_str(), (dirName+"/"+name+"/incl_3b_hlt;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);

      }

      void Fill(float var, float varHLT, bool passNumTrig, bool passDenTrig, eventData* event, bool subsets=true){
	if(passDenTrig){
	  hIncl->Fill(var);
	  if(event->fourTag)  hIncl_4b->Fill(var);
	  if(event->threeTag) hIncl_3b->Fill(var);

	  hIncl_hlt->Fill(varHLT);
	  if(event->fourTag)  hIncl_4b_hlt->Fill(varHLT);
	  if(event->threeTag) hIncl_3b_hlt->Fill(varHLT);
	}
	
	if(passNumTrig){
	  
	  if(passDenTrig || !subsets){
	    hPass->Fill(var);
	    if(event->fourTag)  hPass_4b->Fill(var);
	    if(event->threeTag) hPass_3b->Fill(var);

	    hPass_hlt->Fill(varHLT);
	    if(event->fourTag)  hPass_4b_hlt->Fill(varHLT);
	    if(event->threeTag) hPass_3b_hlt->Fill(varHLT);
	  }
	}

      }

    };

    turnOnHist* hT30_ht330 = NULL;
    turnOnHist* hT30_ht330_L1HT360    = NULL;
    turnOnHist* hT30_ht330_L1ETT2000  = NULL;
    turnOnHist* hT30_ht330_L1HT320_4j = NULL;
    turnOnHist* hT30_ht330_L1OR = NULL;

    turnOnHist* hT30_ht330_wrt_L1OR = NULL;
    turnOnHist* hT30_ht330_3tag = NULL;
    turnOnHist* hT30_ht330_sel = NULL;
    turnOnHist* hT30_ht330_sel_3tag = NULL;

    turnOnHist* hT30_ht330_sel_noSubSet = NULL;

    turnOnHist* hT30_L1HT360    = NULL;
    turnOnHist* hT30_L1ETT2000  = NULL;
    turnOnHist* hT30_L1HT320_4j = NULL;
    turnOnHist* hT30_L1OR = NULL;

    turnOnHist* ht_4j = NULL;
    turnOnHist* htcalo_4j = NULL;
    turnOnHist* htcaloAll_4j = NULL;
    turnOnHist* htcalo2p6_4j = NULL;
    turnOnHist* j0_4j = NULL;
    turnOnHist* j1_4j = NULL;
    turnOnHist* j2_4j = NULL;
    turnOnHist* j3_4j = NULL;

    turnOnHist* ht_4j_em = NULL;
    turnOnHist* ht_4j_3b_em = NULL;
    turnOnHist* ht_4j_l1_em = NULL;
    turnOnHist* ht_4j_l1_3b_em = NULL;
    turnOnHist* ht_4j_ht_em = NULL;
    turnOnHist* ht_4j_3b_ht_em = NULL;

    turnOnHist* j3_4j_wrt_HT_3j = NULL;

    TH1F* hT30_all = NULL;
    TH1F* hT30_h330 = NULL;
    TH1F* hT30_h330_l320 = NULL;
    TH1F* hT30_h330_l320_j30 = NULL;

    TH1F* hMinDr = NULL;
    TH1F* hMatchedPt = NULL;
    TH1F* hMatchedEta = NULL;
    TH1F* hMatched_dPt = NULL;
    TH1F* hMatched_dPtL1 = NULL;
    TH1F* hMatched_dPt_l = NULL;
    TH1F* hMatched_dPtL1_l = NULL;

    TH1F* hMatchedPt_h40 = NULL;
    TH1F* hMatchedPt_h40_l40 = NULL;

    TH1F* hMatchedPt_h45 = NULL;
    TH1F* hMatchedPt_h45_l40 = NULL;

    TH1F* hMatchedPt_h60 = NULL;
    TH1F* hMatchedPt_h60_l55 = NULL;

    TH1F* hMatchedPt_h75 = NULL;
    TH1F* hMatchedPt_h75_l70 = NULL;

    TH1F* hAllPt = NULL;
    TH1F* hAllEta     = NULL;

  };

}
#endif // triggerStudy_H
