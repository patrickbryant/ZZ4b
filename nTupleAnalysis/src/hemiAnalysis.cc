#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>

#include "ZZ4b/nTupleAnalysis/interface/hemiAnalysis.h"


using namespace nTupleAnalysis;
using std::cout; using std::endl; 


hemiAnalysis::hemiAnalysis(std::string _hemiFileName, fwlite::TFileService& fs, bool _debug){
  if(_debug) std::cout<<"In hemiAnalysis constructor"<<std::endl;
  debug      = _debug;

  hMixToolLoad = new hemisphereMixTool("hToolAnalysis", _hemiFileName, false, fs, _debug);

  m_nTreeHemis = hMixToolLoad->hemiTree->GetEntries();
} 



int hemiAnalysis::hemiLoop(int maxHemi){

  //Set Number of hemis to process. Take manual maxHemi if maxEvents is > 0 and less than the total number of events in the input files. 
  int nHemis = (maxHemi > 0 && maxHemi < m_nTreeHemis) ? maxHemi : m_nTreeHemis;
  
  std::cout << "\nProcess " << nHemis << " of " << m_nTreeHemis << " events.\n";

  //start = std::clock();//2546000 //2546043
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    
    hemisphere thisHemi = hMixToolLoad->getHemi(hemiIdx);
    cout << thisHemi.Run << " " << thisHemi.Event << endl;
    
    UInt_t thisHemiPairIdx = thisHemi.pairIdx;

    float minDiff_sumPz         = 9999;
    float minDiff_sumPt_T       = 9999;
    float minDiff_sumPt_Ta      = 9999;
    float minDiff_combinedMass  = 9999;

    hemisphereMixTool::IndexVec otherHemiIdx = hMixToolLoad->getHemiSphereIndices(thisHemi);
    cout << "\t number to check: " << otherHemiIdx.size() << endl;
    //for(long int otherHemiIdx = 0; otherHemiIdx < nHemis; otherHemiIdx++){
    for(long int& otherHemiIdx : otherHemiIdx){
      if(hemiIdx == otherHemiIdx) continue;
      if(thisHemiPairIdx == otherHemiIdx) continue;
      hemisphere otherHemi = hMixToolLoad->getHemi(otherHemiIdx);
      if(thisHemi.NJets != otherHemi.NJets) continue;
      if(thisHemi.NBJets != otherHemi.NBJets) continue;
      
      //cout << "\t " << otherHemi.Run << " " << otherHemi.Event << "("<<thisHemi.Run << "," << thisHemi.Event << ")" << endl;
    }
  }




  return 0;
}


hemiAnalysis::~hemiAnalysis(){} 

