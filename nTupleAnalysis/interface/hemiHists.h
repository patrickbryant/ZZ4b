// -*- C++ -*-
#if !defined(hemiHists_H)
#define hemiHists_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/hemisphere.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"

namespace nTupleAnalysis {

  struct hemiDiffHists {

    std::string m_name = "";

    TH1F*      hdelta_NJets   ;
    TH1F*      hdelta_NBJets  ;
    TH1F*      hdelta_Pz      ;
    TH1F*      hdelta_SumPt_T ;
    TH1F*      hdelta_SumPt_Ta;
    TH1F*      hdelta_CombMass;
    TH1F*      hdelta_CombDr;
    TH1F*      hdist;
    TH1F*      heventWeight;

    hemiDiffHists(std::string name, std::string diffName, TFileDirectory& thisDir, std::string postfix="");

    void Fill(const hemiPtr& hIn, const hemiPtr& hMatch, const hemiDataHandler* dataHandler);

  };


  class hemiHists {
  public:

    std::string m_name = "";
    
    TH1F*      heventWeight;

    TH1F*      hPz;
    TH1F*      hSumPt_T  ;  
    TH1F*      hSumPt_Ta ;  
    TH1F*      hCombMass ;  
    TH1F*      hCombDr ;  

    TH1F*      hPz_sig;
    TH1F*      hSumPt_T_sig  ;  
    TH1F*      hSumPt_Ta_sig ;  
    TH1F*      hCombMass_sig ;  
    TH1F*      hCombDr_sig ;  

    hemiDiffHists* hDiffNN;
    hemiDiffHists* hDiffTopN;
    hemiDiffHists* hDiffRand;


    hemiHists(std::string name, TFileDirectory& thisDir, std::string postFix= "",bool makeTopN=false, bool makeRand=false);

    void Fill(const hemiPtr& hIn, const hemiDataHandler* dataHandler);
    

    ~hemiHists(); 

  };

}
#endif // hemiHists_H
