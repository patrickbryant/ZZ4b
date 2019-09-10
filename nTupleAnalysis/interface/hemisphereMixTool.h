// -*- C++ -*-
#if !defined(hemisphereMixTool_H)
#define hemisphereMixTool_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>
#include "TVector2.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "nTupleAnalysis/baseClasses/interface/EventDisplayData.h"
#include "ZZ4b/nTupleAnalysis/interface/hemisphere.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiHists.h"
#include <sstream>      // std::stringstream

namespace nTupleAnalysis {
  class hemiDataHandler;
}


namespace nTupleAnalysis {

  class hemisphereMixTool {

  public:

    hemisphereMixTool(std::string name, std::string outputFile, std::vector<std::string> inputFiles, bool fCreateLibrary, fwlite::TFileService& fs, int maxNHemis, bool debug, bool loadJetFourVecs, bool dualAccess);
    ~hemisphereMixTool(); 

    void addEvent(eventData*);
    int makeArtificialEvent(eventData*);

    void storeLibrary();
    bool m_debug;

    std::map<EventID, hemiDataHandler*>   m_dataHandleIndex;    
    hemiDataHandler* getDataHandler(EventID thisEventID, std::string inputFile = "");




  private:

    //TFileDirectory dir;
    std::string m_name;
    std::string m_outputFileName;
    std::vector<std::string> m_inputFileNames;
    bool m_loadJetFourVecs;
    bool m_dualAccess;

    bool m_createLibrary;
    int m_maxNHemis;

    TVector2 getThrustAxis(eventData* event);

    void makeIndexing();

    TFileDirectory dir;
    hemiHists* hHists;
    TH1F* hSameEventCheck;

    //
    // Event Displays
    //
    bool makeEventDisplays = false;
    nTupleAnalysis::EventDisplayData* eventDisplay = NULL;

  public:
    //
    //  To output to the PicoAODs
    //
    UInt_t    m_h1_run       =  0;
    ULong64_t m_h1_event     =  0;
    Int_t     m_h1_NJet       =  0;
    Int_t     m_h1_NBJet      =  0;
    Int_t     m_h1_NNonSelJet =  0;
    Int_t     m_h1_matchCode  =  0;
    Float_t   m_h1_pz                 = 0;
    Float_t   m_h1_pz_sig             = 0;
    Float_t   m_h1_match_pz           = 0;
    Float_t   m_h1_sumpt_t            = 0;
    Float_t   m_h1_sumpt_t_sig        = 0;
    Float_t   m_h1_match_sumpt_t      = 0;
    Float_t   m_h1_sumpt_ta           = 0;
    Float_t   m_h1_sumpt_ta_sig       = 0;
    Float_t   m_h1_match_sumpt_ta     = 0;
    Float_t   m_h1_combinedMass       = 0;
    Float_t   m_h1_combinedMass_sig   = 0;
    Float_t   m_h1_match_combinedMass = 0;
    Float_t   m_h1_match_dist         = 0;

    UInt_t    m_h2_run       =  0;
    ULong64_t m_h2_event     =  0;
    Int_t     m_h2_NJet       =  0;
    Int_t     m_h2_NBJet      =  0;
    Int_t     m_h2_NNonSelJet =  0;
    Int_t     m_h2_matchCode  =  0;
    Float_t   m_h2_pz                 = 0;
    Float_t   m_h2_pz_sig             = 0;
    Float_t   m_h2_match_pz           = 0;
    Float_t   m_h2_sumpt_t            = 0;
    Float_t   m_h2_sumpt_t_sig        = 0;
    Float_t   m_h2_match_sumpt_t      = 0;
    Float_t   m_h2_sumpt_ta           = 0;
    Float_t   m_h2_sumpt_ta_sig       = 0;
    Float_t   m_h2_match_sumpt_ta     = 0;
    Float_t   m_h2_combinedMass       = 0;
    Float_t   m_h2_combinedMass_sig   = 0;
    Float_t   m_h2_match_combinedMass = 0;
    Float_t   m_h2_match_dist         = 0;



  };



}
#endif // hemisphereMixTool_H
