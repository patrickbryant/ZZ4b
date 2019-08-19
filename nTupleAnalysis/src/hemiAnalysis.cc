#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>

#include "ZZ4b/nTupleAnalysis/interface/hemiAnalysis.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"

using namespace nTupleAnalysis;
using std::cout; using std::endl; 


hemiAnalysis::hemiAnalysis(std::vector<std::string>  _hemiFileNames, fwlite::TFileService& fs, bool _debug, bool _loadJetFourVecs){
  if(_debug) std::cout<<"In hemiAnalysis constructor"<<std::endl;
  debug      = _debug;

  unsigned int maxNHemis = 10000;
  m_loadJetFourVecs = _loadJetFourVecs;
  hMixToolLoad = new hemisphereMixTool("hToolAnalysis", "dummyName", _hemiFileNames, false, fs, maxNHemis, _debug, _loadJetFourVecs, true);

  TFileDirectory dir = fs.mkdir("hemiAnalysis");

  for (std::pair<EventID, hemiDataHandler*> element : hMixToolLoad->m_dataHandleIndex) {
    EventID& evID = element.first;
    hemiDataHandler* dataHandle = element.second;
    dataHandle->m_debug = debug;
    
    std::stringstream ss;
    ss << evID.at(0) << "_" << evID.at(1) << "_" << evID.at(2);

    TFileDirectory thisDir = dir.mkdir("hemiHist_"+ss.str());
    hists[evID] = new hemiHists("hemiHist", thisDir, ss.str(),true, true);

  }


}



int hemiAnalysis::hemiLoop(int maxHemi){

  for (std::pair<EventID, hemiDataHandler*> element : hMixToolLoad->m_dataHandleIndex) {
    
    EventID& evID = element.first;
    hemiDataHandler* dataHandle      = element.second;
    
    if(!dataHandle->m_isValid) continue;
    
    int nTreeHemis = dataHandle->hemiTree->GetEntries();    
        
    // skip the silly hemispheres iwth 0 Jets
    if(evID.at(0) == 0) continue;

    unsigned int nHemis = (maxHemi > 0 && maxHemi < nTreeHemis) ? maxHemi : nTreeHemis;
    std::cout << "\nProcess " << nHemis << " of " << nTreeHemis << " hemispheres in region " << evID.at(0) << " / " << evID.at(1) << " / " << evID.at(2) << ".\n";

    start = std::clock();//2546000 //2546043
    for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    
      if( (hemiIdx % 1000 == 0) ) 
	monitor(hemiIdx,nHemis);

      if(debug) cout << "Getting Hemi " << endl;
      hemiPtr thisHemi = dataHandle->getHemi(hemiIdx, m_loadJetFourVecs);

      if(debug) cout << thisHemi->Run << " " << thisHemi->Event << endl;

      hists[evID]->Fill(thisHemi,dataHandle);
			
      
      //UInt_t thisHemiPairIdx = thisHemi.pairIdx;
      if(debug) cout << "Getting NearNeig " << endl;
      // probably want to do this with pointers
      double bestMatchDist = 1e6;
      hemiPtr thisHemiBestMatch = dataHandle->getHemiNearNeig(hemiIdx, bestMatchDist);
      
      //if(debug) cout << "Filling Hists " << endl;

      hists[evID]->hDiffNN->Fill(thisHemi,thisHemiBestMatch,dataHandle);
      
      // probably want to do this with pointers
      std::vector<hemiPtr> thisHemiBestMatches = dataHandle->getHemiNearNeighbors(hemiIdx,5);
      for(const hemiPtr& hemiNN : thisHemiBestMatches){
	hists[evID]->hDiffTopN->Fill(thisHemi,hemiNN,dataHandle);
      }

      // Get Random hemi 
      // probably want to do this with pointers
      hemiPtr hemiRand = dataHandle->getHemiRandom();
      hists[evID]->hDiffRand->Fill(thisHemi,hemiRand,dataHandle);

    }//

  }// EventIDs


  return 0;
}


hemiAnalysis::~hemiAnalysis(){} 


void hemiAnalysis::monitor(long int e, long int nHemis){
  //Monitor progress
  percent        = (e+1)*100/nHemis;
  duration       = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  eventRate      = (e+1)/duration;
  timeRemaining  = (nHemis-e)/eventRate;
  minutes = static_cast<int>(timeRemaining/60);
  seconds = static_cast<int>(timeRemaining - minutes*60);
  tot_minutes = static_cast<int>(duration/60);
  tot_seconds = static_cast<int>(duration - tot_minutes*60);
  getrusage(who, &usage);
  usageMB = usage.ru_maxrss/1024;
  //print status and flush stdout so that status bar only uses one line
  fprintf(stdout, "\rProcessed: %8li of %li ( %2li%% | %.0f hemis/s | done in %02i:%02i | total time %02i:%02i memory usage: %li MB  )       ", 
	  e+1, nHemis, percent,   eventRate,    minutes, seconds,     tot_minutes, tot_seconds,           usageMB );    
  fflush(stdout);
}
