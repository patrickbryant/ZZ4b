// -*- C++ -*-
#if !defined(vecHists_H)
#define vecHists_H

#include <iostream>
#include <TH1F.h>
#include <TLorentzVector.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"

namespace NtupleAna {

  class vecHists {
  public:
    
    TH1F* pt_s;
    TH1F* pt_m;
    TH1F* pt_l;

    TH1F* eta;
    TH1F* phi;

    TH1F* m;
    TH1F* m_l;
    TH1F* e;

    vecHists(std::string,       TFileDirectory&, std::string title = "");
    vecHists(std::string, fwlite::TFileService&, std::string title = "");

    void makeHists(std::string, TFileDirectory&, std::string title);
    void Fill(TLorentzVector&, float);

    ~vecHists(); 

  };

}
#endif // vecHists_H
