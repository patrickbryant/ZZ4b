// -*- C++ -*-
#if !defined(mixedAnalysis_H)
#define mixedAnalysis_H

#include <ctime>
#include <sys/resource.h>
#include <sstream>      // std::stringstream
#include <map>
#include <set>

#include <TChain.h>
#include <TTree.h>
#include <TSpline.h>
#include "DataFormats/FWLite/interface/InputSource.h" //for edm::LuminosityBlockRange
#include "nTupleAnalysis/baseClasses/interface/brilCSV.h"
#include "nTupleAnalysis/baseClasses/interface/initBranch.h"
#include "ZZ4b/nTupleAnalysis/interface/mixedEventData.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"



namespace nTupleAnalysis {

  typedef  std::pair<UInt_t, Long64_t>             HemiEventID;
  typedef  std::pair<HemiEventID,    HemiEventID>  HemiPairEventID;
  typedef  std::map<HemiEventID,     unsigned int> HemiCount;
  typedef  std::map<HemiPairEventID, unsigned int> HemiPairCount;


  class mixedAnalysis {
  public:
    
    HemiCount allHemisCounts;
    HemiPairCount allHemiPairsCounts;
    HemiCount allHemisCounts_SR;
    HemiPairCount allHemiPairsCounts_SR;


    // details 
    
    TChain* events;
    TChain* runs;
    TChain* lumiBlocks;

    bool debug = false;

    TH1F* hHemiUsage;
    TH1F* hHemiPairUsage;

    TH1F* hHemiUsage_SR;
    TH1F* hHemiPairUsage_SR;


    int treeEvents;
    mixedEventData* event;
    
    long int nEvents = 0;
    double lumi      = 1;
    std::vector<edm::LuminosityBlockRange> lumiMask;
    UInt_t prevLumiBlock = 0;
    UInt_t firstRun      = 1e9;
    UInt_t lastRun       = 0;
    UInt_t prevRun       = 0;
    UInt_t nruns = 0;
    UInt_t nls   = 0;
    float  intLumi = 0;


    //Monitoring Variables
    long int percent;
    std::clock_t start;
    double duration;
    double eventRate;
    double timeRemaining;
    int hours;
    int minutes;
    int seconds;
    int tot_minutes;
    int tot_seconds;
    int who = RUSAGE_SELF;
    struct rusage usage;
    long int usageMB;


    mixedAnalysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _debug);

    //void invert(const std::map<EventID, unsigned int> & inputMap);
    template<class T>
    std::multimap<unsigned int, T> invert(const std::map<T, unsigned int> & inputMap);

    void monitor(long int);
    int eventLoop(int maxEvents, long int firstEvent = 0);
    ~mixedAnalysis();

    int processEvent();
    std::map<edm::LuminosityBlockID, float> lumiData;



  };


  template<class T>
  std::multimap<unsigned int, T> mixedAnalysis::invert(const std::map<T, unsigned int> & inputMap)
  {
    std::multimap<unsigned int, T> outputMap;
  
    for(const auto& mapItr: inputMap){
      outputMap.insert(make_pair(mapItr.second, mapItr.first));
    }

    return outputMap;
  }


}
#endif // mixedAnalysis_H

