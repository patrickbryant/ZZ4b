// -*- C++ -*-
#if !defined(hemiAnalysis_H)
#define hemiAnalysis_H

#include <ctime>
#include <sys/resource.h>
#include <sstream>      // std::stringstream

#include <TChain.h>
#include <TTree.h>
#include <TSpline.h>
#include "DataFormats/FWLite/interface/InputSource.h" //for edm::LuminosityBlockRange
#include "nTupleAnalysis/baseClasses/interface/brilCSV.h"
#include "nTupleAnalysis/baseClasses/interface/initBranch.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/cutflowHists.h"
#include "ZZ4b/nTupleAnalysis/interface/tagCutflowHists.h"
#include "ZZ4b/nTupleAnalysis/interface/eventHists.h"
#include "ZZ4b/nTupleAnalysis/interface/tagHists.h"
#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"
#include "nTupleAnalysis/baseClasses/interface/EventDisplayData.h"

namespace nTupleAnalysis {

  struct hemiDiffHists {
    std::string m_name = "";
    unsigned int m_NJet = 0;
    unsigned int m_NBJet = 0;
    float m_varPz = 0;
    float m_varPt_T = 0;
    float m_varPt_Ta = 0;
    float m_varCombinedM = 0;

    TH1F*      hdelta_NJets   ;
    TH1F*      hdelta_NBJets  ;
    TH1F*      hdelta_Pz      ;
    TH1F*      hdelta_SumPt_T ;
    TH1F*      hdelta_SumPt_Ta;
    TH1F*      hdelta_CombMass;
    TH1F*      hdist;

    hemiDiffHists(std::string name, std::string diffName, TFileDirectory& thisDir, unsigned int NJet, unsigned int NBJet, 
		  float varPz, float varPt_T, float varPt_Ta, float combinedM) :  m_NJet(NJet), m_NBJet(NBJet), m_varPz(varPz), m_varPt_T(varPt_T), m_varPt_Ta(varPt_Ta), m_varCombinedM(combinedM)
    {
      std::stringstream ss;
      ss << NJet << "_" << NBJet;
      m_name = name + "_" + ss.str();


      hdelta_NJets    = thisDir.make<TH1F>(("hdel_"+diffName+"_NJets"   ).c_str(),  (m_name+"/del_"+diffName+"_NJets;  ;Entries"  ).c_str(),  19,-9.5,9.5);  
      hdelta_NBJets   = thisDir.make<TH1F>(("hdel_"+diffName+"_NBJets"  ).c_str(),  (m_name+"/del_"+diffName+"_NBJets; ;Entries"  ).c_str(),  19,-9.5,9.5);  
      hdelta_Pz       = thisDir.make<TH1F>(("hdel_"+diffName+"_Pz"      ).c_str(),  (m_name+"/del_"+diffName+"_Pz; ;Entries"      ).c_str(),  100,-500,500);  
      hdelta_SumPt_T  = thisDir.make<TH1F>(("hdel_"+diffName+"_SumPt_T" ).c_str(),  (m_name+"/del_"+diffName+"_SumPt_T; ;Entries" ).c_str(),     100,-300,300);  
      hdelta_SumPt_Ta = thisDir.make<TH1F>(("hdel_"+diffName+"_SumPt_Ta").c_str(),  (m_name+"/del_"+diffName+"_SumPt_Ta; ;Entries").c_str(),     100,-200,200);  
      hdelta_CombMass = thisDir.make<TH1F>(("hdel_"+diffName+"_CombMass").c_str(),  (m_name+"/del_"+diffName+"_CombMass; ;Entries").c_str(),     100,-300,300);  

      hdist           = thisDir.make<TH1F>(("dist_"+diffName).c_str(),     (m_name+"/dist_"+diffName+"; ;Entries").c_str(),     100,-0.1,5);  
    }

    void Fill(const hemisphere& hIn, const hemisphere& hMatch ){
      hdelta_NJets    ->Fill(hIn.NJets - hMatch.NJets);
      hdelta_NBJets   ->Fill(hIn.NBJets - hMatch.NBJets);

      float pzDiff =hIn.sumPz - hMatch.sumPz;
      float pzDiff_sig =(hIn.sumPz - hMatch.sumPz)/m_varPz;
      hdelta_Pz       ->Fill(pzDiff);

      float sumPt_T_diff = hIn.sumPt_T - hMatch.sumPt_T;
      float sumPt_T_diff_sig = (hIn.sumPt_T - hMatch.sumPt_T)/m_varPt_T;

      hdelta_SumPt_T  ->Fill(sumPt_T_diff);

      float sumPt_Ta_diff = hIn.sumPt_Ta - hMatch.sumPt_Ta;
      float sumPt_Ta_diff_sig = (hIn.sumPt_Ta - hMatch.sumPt_Ta)/m_varPt_Ta;
      hdelta_SumPt_Ta ->Fill(sumPt_Ta_diff);

      float combinedMass_diff = hIn.combinedMass - hMatch.combinedMass;
      float combinedMass_diff_sig = (hIn.combinedMass - hMatch.combinedMass)/m_varCombinedM;
      hdelta_CombMass ->Fill(combinedMass_diff);
      
      float dist = sqrt(pzDiff_sig*pzDiff_sig + sumPt_T_diff_sig*sumPt_T_diff_sig + sumPt_Ta_diff_sig*sumPt_Ta_diff_sig + combinedMass_diff_sig*combinedMass_diff_sig);
      hdist->Fill(dist);
    }

  };


  struct hemiHists {
    
    std::string m_name = "";
    unsigned int m_NJet = 0;
    unsigned int m_NBJet = 0;
    unsigned int m_NNonSelJet = 0;
    float m_varPz = 0;
    float m_varPt_T = 0;
    float m_varPt_Ta = 0;
    float m_varCombinedM = 0;
    
    TH1F*      hPz;
    TH1F*      hSumPt_T  ;  
    TH1F*      hSumPt_Ta ;  
    TH1F*      hCombMass ;  

    TH1F*      hPz_sig;
    TH1F*      hSumPt_T_sig  ;  
    TH1F*      hSumPt_Ta_sig ;  
    TH1F*      hCombMass_sig ;  

    hemiDiffHists* hDiffNN;
    hemiDiffHists* hDiffTopN;
    hemiDiffHists* hDiffRand;

    hemiHists(std::string name, TFileDirectory& dir, unsigned int NJet, unsigned int NBJet, unsigned int NNonSelJet, 
	      float varPz, float varPt_T, float varPt_Ta, float combinedM) :  m_NJet(NJet), m_NBJet(NBJet), m_NNonSelJet(NNonSelJet), m_varPz(varPz), m_varPt_T(varPt_T), m_varPt_Ta(varPt_Ta), m_varCombinedM(combinedM)
    {
      std::stringstream ss;
      ss << NJet << "_" << NBJet << "_" << NNonSelJet;
      m_name = name + "_" + ss.str();

      TFileDirectory thisDir = dir.mkdir(m_name);
      hPz     = thisDir.make<TH1F>("hPz",     (m_name+"/Pz; ;Entries").c_str(),     100,-1000,1000);  
      hSumPt_T = thisDir.make<TH1F>("hSumPt_T",     (m_name+"/SumPt_T; ;Entries").c_str(),     100,0,1000);  
      hSumPt_Ta = thisDir.make<TH1F>("hSumPt_Ta",     (m_name+"/SumPt_Ta; ;Entries").c_str(),     100,0,500);  
      hCombMass = thisDir.make<TH1F>("hCombMass",     (m_name+"/CombMass; ;Entries").c_str(),     100,0,500);  

      hPz_sig     = thisDir.make<TH1F>("hPz_sig",     (m_name+"/Pz_sig; ;Entries").c_str(),     100,-10,10);  
      hSumPt_T_sig = thisDir.make<TH1F>("hSumPt_T_sig",     (m_name+"/SumPt_T_sig; ;Entries").c_str(),     100,-10,10);  
      hSumPt_Ta_sig = thisDir.make<TH1F>("hSumPt_Ta_sig",     (m_name+"/SumPt_Ta_sig; ;Entries").c_str(),     100,-10,10);  
      hCombMass_sig = thisDir.make<TH1F>("hCombMass_sig",     (m_name+"/CombMass_sig; ;Entries").c_str(),     100,-10,10);  
      
      hDiffNN   = new hemiDiffHists(name, "NN",   thisDir, NJet, NBJet, varPz, varPt_T, varPt_Ta, combinedM);
      hDiffTopN = new hemiDiffHists(name, "TopN", thisDir, NJet, NBJet, varPz, varPt_T, varPt_Ta, combinedM);
      hDiffRand = new hemiDiffHists(name, "Rand", thisDir, NJet, NBJet, varPz, varPt_T, varPt_Ta, combinedM);
      // diff
    }
    
    void Fill(const hemisphere& hIn){
      hPz       ->Fill( hIn.sumPz);
      hSumPt_T  ->Fill( hIn.sumPt_T);
      hSumPt_Ta ->Fill( hIn.sumPt_Ta);
      hCombMass ->Fill( hIn.combinedMass);

      hPz_sig       ->Fill( hIn.sumPz       / m_varPz);
      hSumPt_T_sig  ->Fill( hIn.sumPt_T     / m_varPt_T);
      hSumPt_Ta_sig ->Fill( hIn.sumPt_Ta    / m_varPt_Ta);
      hCombMass_sig ->Fill( hIn.combinedMass/ m_varCombinedM);
	
      return;
    }


  };


  class hemiAnalysis {
  public:

    
    bool debug = false;

    //
    // Hemisphere Mixing 
    //
    bool writeHSphereFile = false;
    hemisphereMixTool* hMixToolLoad = NULL;

    hemiAnalysis(std::vector<std::string>, fwlite::TFileService&, bool);
    //void createHemisphereLibrary(std::string, fwlite::TFileService& fs );
    //void storeHemiSphereFile();
    int hemiLoop(int maxHemi);
    ~hemiAnalysis();

    std::map<hemisphereMixTool::EventID, hemiHists*> hists;

    //Monitoring Variables
    long int percent;
    std::clock_t start;
    double duration;
    double eventRate;
    double timeRemaining;
    int minutes;
    int seconds;
    int tot_minutes;
    int tot_seconds;
    int who = RUSAGE_SELF;
    struct rusage usage;
    long int usageMB;
    void monitor(long int e,long int nHemis);

  };

}
#endif // hemiAnalysis_H

