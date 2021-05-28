// -*- C++ -*-
#if !defined(countsVsLumiHists_H)
#define countsVsLumiHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"

namespace nTupleAnalysis {

  class countsVsLumiHists {
  public:

    bool m_debug = false;
    
    TH1F* m_hist;
    TH1F* m_histUnit;

    float m_currentLB = 0;
    std::string m_currentLBStr;

    std::string getLumiName();

    countsVsLumiHists(std::string histName, std::string name, TFileDirectory& dir, bool _debug=false);
    
    void Fill(float weight);

    void FillLumiBlock(float lumiThisBlock);

    ~countsVsLumiHists(); 

  };

}
#endif // countsVsLumiHists_H
