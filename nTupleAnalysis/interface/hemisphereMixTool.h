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

    hemisphereMixTool(std::string name, std::string outputFile, std::vector<std::string> inputFiles, bool fCreateLibrary, fwlite::TFileService& fs, bool debug, bool loadJetFourVecs, bool dualAccess);
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

    TVector2 getThrustAxis(eventData* event);

    void makeIndexing();

    TVector2 calcThrust(const std::vector<TVector2>& jetPts);
    void calcT(const std::vector<TVector2>& momenta, double& t, TVector2& taxis);


    TFileDirectory dir;
    hemiHists* hHists;
    TH1F* hSameEventCheck;

    //
    // Event Displays
    //
    bool makeEventDisplays = false;
    nTupleAnalysis::EventDisplayData* eventDisplay = NULL;
    

  };



}
#endif // hemisphereMixTool_H
