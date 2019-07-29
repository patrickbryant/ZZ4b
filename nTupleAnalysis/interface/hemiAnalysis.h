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

  struct hemiHists {
    
    std::string m_name = "";
    unsigned int m_NJet = 0;
    unsigned int m_NBJet = 0;
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


    hemiHists(std::string name, TFileDirectory& dir, unsigned int NJet, unsigned int NBJet, 
	      float varPz, float varPt_T, float varPt_Ta, float combinedM) :  m_NJet(NJet), m_NBJet(NBJet), m_varPz(varPz), m_varPt_T(varPt_T), m_varPt_Ta(varPt_Ta), m_varCombinedM(combinedM)
    {
      std::stringstream ss;
      ss << NJet << "_" << NBJet;
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

//  hdelta_NJets  = dir.make<TH1F>("hdelta_NJets",  (name+"/del_NJets;  ;Entries").c_str(),  19,-9.5,9.5);  
//  hdelta_NBJets = dir.make<TH1F>("hdelta_NBJets", (name+"/del_NBJets; ;Entries").c_str(),  19,-9.5,9.5);  
//  hdelta_Pz     = dir.make<TH1F>("hdeltaPz",      (name+"/del_Pz; ;Entries").c_str(),  100,-500,500);  
//  hdelta_SumPt_T = dir.make<TH1F>("hdeltaSumPt_T",     (name+"/del_SumPt_T; ;Entries").c_str(),     100,-300,300);  
//  hdelta_SumPt_Ta = dir.make<TH1F>("hdeltaSumPt_Ta",     (name+"/del_SumPt_Ta; ;Entries").c_str(),     100,-200,200);  
//  hdelta_CombMass = dir.make<TH1F>("hdeltaCombMass",     (name+"/del_CombMass; ;Entries").c_str(),     100,-300,300);  

      
    }
    
    void Fill(const hemisphere& hIn, const hemisphere& hMatch ){
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
    int m_nTreeHemis;

    //
    // Hemisphere Mixing 
    //
    bool writeHSphereFile = false;
    hemisphereMixTool* hMixToolLoad = NULL;

    hemiAnalysis(std::string, fwlite::TFileService&, bool);
    //void createHemisphereLibrary(std::string, fwlite::TFileService& fs );
    //void storeHemiSphereFile();
    int hemiLoop(int maxHemi);
    ~hemiAnalysis();

    std::map<hemisphereMixTool::EventID, hemiHists*> hists;

  };

}
#endif // hemiAnalysis_H

