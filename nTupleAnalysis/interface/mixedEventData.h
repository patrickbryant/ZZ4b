// -*- C++ -*-

#if !defined(mixedEventData_H)
#define mixedEventData_H

#include <iostream>
#include <TChain.h>
#include <TFile.h>
#include "nTupleAnalysis/baseClasses/interface/initBranch.h"



namespace nTupleAnalysis {

  class mixedEventData {

  public:
    // Member variables
    TChain* tree;
    bool debug;

    UInt_t    run       =  0;
    UInt_t    lumiBlock =  0;
    ULong64_t event     =  0;

    bool fourTag = false;
    bool SB = false;
    bool CR = false;
    bool SR = false;

    UInt_t    h1_run       =  0;
    ULong64_t h1_event     =  0;
    Long64_t  h1_eventSigned =  0;
    Bool_t    h1_hemiSign  =  0;

    UInt_t    h2_run       =  0;
    ULong64_t h2_event     =  0;
    Long64_t  h2_eventSigned =  0;
    Bool_t    h2_hemiSign  =  0;

  public:

    // Constructors and member functions
    mixedEventData(TChain* t, bool d);

    void update(long int);

    void dump();
    ~mixedEventData(); 

    std::string currentFile = "";


  };

}
#endif // mixedEventData_H
