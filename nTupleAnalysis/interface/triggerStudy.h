// -*- C++ -*-
#if !defined(triggerStudy_H)
#define triggerStudy_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

using namespace nTupleAnalysis;

namespace nTupleAnalysis {

  class triggerStudy {
  public:

    TFileDirectory dir;
    bool debug;
    
    triggerStudy(std::string, fwlite::TFileService&, bool _debug = false);
    void Fill(eventData* event);
    ~triggerStudy(); 

  private:

    struct turnOnHist {
      std::string name = "";
      TH1F* hIncl = NULL;
      TH1F* hPass = NULL;

      TH1F* hIncl_4b = NULL;
      TH1F* hPass_4b = NULL;

      TH1F* hIncl_3b = NULL;
      TH1F* hPass_3b = NULL;


    turnOnHist(std::string _name, std::string dirName, TFileDirectory dir, std::string xTitle) : name(_name) 
      {
	hIncl  = dir.make<TH1F>((name+"_incl").c_str(),  (dirName+"/"+name+"_all;  "+xTitle+"; Entries").c_str(),  200,0,2000);
	hPass  = dir.make<TH1F>((name+"_pass").c_str(), (dirName+"/"+name+"_incl;  "+xTitle+"; Entries").c_str(),  200,0,2000);

	hIncl_4b  = dir.make<TH1F>((name+"_incl_4b").c_str(),  (dirName+"/"+name+"_all_4b;  "+xTitle+"; Entries").c_str(),  200,0,2000);
	hPass_4b  = dir.make<TH1F>((name+"_pass_4b").c_str(), (dirName+"/"+name+"_incl_4b;  "+xTitle+"; Entries").c_str(),  200,0,2000);

	hIncl_3b  = dir.make<TH1F>((name+"_incl_3b").c_str(),  (dirName+"/"+name+"_all_3b;  "+xTitle+"; Entries").c_str(),  200,0,2000);
	hPass_3b  = dir.make<TH1F>((name+"_pass_3b").c_str(), (dirName+"/"+name+"_incl_3b;  "+xTitle+"; Entries").c_str(),  200,0,2000);
      }

      void Fill(float var, bool passNumTrig, bool passDenTrig, eventData* event){
	if(!passDenTrig) return;
	hIncl->Fill(var);
	if(event->fourTag)  hIncl_4b->Fill(var);
	if(event->threeTag) hIncl_3b->Fill(var);

	if(passNumTrig){
	  hPass->Fill(var);
	  if(event->fourTag)  hPass_4b->Fill(var);
	  if(event->threeTag) hPass_3b->Fill(var);
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

    turnOnHist* hT30_L1HT360    = NULL;
    turnOnHist* hT30_L1ETT2000  = NULL;
    turnOnHist* hT30_L1HT320_4j = NULL;
    turnOnHist* hT30_L1OR = NULL;

//    TH1F* hT30_incl = NULL;
//    TH1F* hT30_pass = NULL;
//    TH1F* hT30_incl_3b = NULL;
//    TH1F* hT30_pass_3b = NULL;
//    TH1F* hT30_incl_4b = NULL;
//    TH1F* hT30_pass_4b = NULL;



  };

}
#endif // triggerStudy_H
