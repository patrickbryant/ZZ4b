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

      TH1F* hIncl_4b = NULL;
      TH1F* hPass_4b = NULL;

      TH1F* hIncl_3b = NULL;
      TH1F* hPass_3b = NULL;


      turnOnHist(std::string _name, std::string dirName, TFileDirectory dir, std::string xTitle, unsigned int nBins, float xMin, float xMax) : name(_name) 
      {
	hIncl  = dir.make<TH1F>((name+"_incl").c_str(),  (dirName+"/"+name+"_all;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass  = dir.make<TH1F>((name+"_pass").c_str(), (dirName+"/"+name+"_incl;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);

	hIncl_4b  = dir.make<TH1F>((name+"_incl_4b").c_str(),  (dirName+"/"+name+"_all_4b;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass_4b  = dir.make<TH1F>((name+"_pass_4b").c_str(), (dirName+"/"+name+"_incl_4b;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);

	hIncl_3b  = dir.make<TH1F>((name+"_incl_3b").c_str(),  (dirName+"/"+name+"_all_3b;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
	hPass_3b  = dir.make<TH1F>((name+"_pass_3b").c_str(), (dirName+"/"+name+"_incl_3b;  "+xTitle+"; Entries").c_str(),  nBins,xMin,xMax);
      }

      void Fill(float var, bool passNumTrig, bool passDenTrig, eventData* event, bool subsets=true){
	if(passDenTrig){
	  hIncl->Fill(var);
	  if(event->fourTag)  hIncl_4b->Fill(var);
	  if(event->threeTag) hIncl_3b->Fill(var);
	}
	
	if(passNumTrig){
	  
	  if(passDenTrig || !subsets){
	    hPass->Fill(var);
	    if(event->fourTag)  hPass_4b->Fill(var);
	    if(event->threeTag) hPass_3b->Fill(var);
	  }
	}

      }

    };
    TH1F* hT30_all = NULL;

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


    turnOnHist* j0_4j = NULL;
    turnOnHist* j1_4j = NULL;
    turnOnHist* j2_4j = NULL;
    turnOnHist* j3_4j = NULL;

    turnOnHist* j3_4j_wrt_HT_3j = NULL;


  };

}
#endif // triggerStudy_H
