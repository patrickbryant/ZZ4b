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
#include <sstream>      // std::stringstream

namespace nTupleAnalysis {
  class hemiDataHandler;
}


namespace nTupleAnalysis {

  class hemisphereMixTool {

  public:
    //TFileDirectory dir;
    std::string m_name;
    std::string m_outputFileName;
    std::vector<std::string> m_inputFileNames;
    bool m_debug;
    bool createLibrary;
    TVector2 m_thrustAxis;

    hemisphereMixTool(std::string name, std::string outputFile, std::vector<std::string> inputFiles, bool fCreateLibrary, fwlite::TFileService& fs, bool debug);

    TVector2 getThrustAxis(eventData* event);

    void addEvent(eventData*);
    ~hemisphereMixTool(); 
    void storeLibrary();


    typedef std::array<int, 2> EventID;
 
    std::map<EventID, hemiDataHandler*>   m_dataHandleIndex;    
    hemiDataHandler* getDataHandler(EventID thisEventID, std::string = "");

  private:

    void makeIndexing();

    TVector2 calcThrust(const std::vector<TVector2>& jetPts);
    void calcT(const std::vector<TVector2>& momenta, double& t, TVector2& taxis);

    void FillHists(const hemisphere& posH, const hemisphere& negH);
    void FillHists(const hemisphere& hIn);

    TFileDirectory dir;
    TH1F* hNJets;
    TH1F* hNBJets;
    TH1F* hPz       ;
    TH1F* hSumPt_T  ;
    TH1F* hSumPt_Ta ;
    TH1F* hCombMass ;


    TH1F* hdelta_NJets;
    TH1F* hdelta_NBJets;
    TH1F* hdelta_Pz      ;
    TH1F* hdelta_SumPt_T ;
    TH1F* hdelta_SumPt_Ta;
    TH1F* hdelta_CombMass;


    //
    // Event Displays
    //
    bool makeEventDisplays = false;
    nTupleAnalysis::EventDisplayData* eventDisplay = NULL;
    

  };



}
#endif // hemisphereMixTool_H
