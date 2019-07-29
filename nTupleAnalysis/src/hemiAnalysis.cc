#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>

#include "ZZ4b/nTupleAnalysis/interface/hemiAnalysis.h"
#include "ZZ4b/nTupleAnalysis/interface/kdTree.h"


using namespace nTupleAnalysis;
using std::cout; using std::endl; 


hemiAnalysis::hemiAnalysis(std::string _hemiFileName, fwlite::TFileService& fs, bool _debug){
  if(_debug) std::cout<<"In hemiAnalysis constructor"<<std::endl;
  debug      = _debug;

  hMixToolLoad = new hemisphereMixTool("hToolAnalysis", _hemiFileName, false, fs, _debug);

  m_nTreeHemis = hMixToolLoad->hemiTree->GetEntries();

  TFileDirectory dir = fs.mkdir("hemiAnalysis");

  for (std::pair<hemisphereMixTool::EventID, hemisphereMixTool::hemiPoint> element : hMixToolLoad->m_varV) {
    hemisphereMixTool::EventID& evID = element.first;
    hists[evID] = new hemiHists("hemiHist", dir, evID.at(0), evID.at(1),
				hMixToolLoad->m_varV[evID].x[0],
				hMixToolLoad->m_varV[evID].x[1],
				hMixToolLoad->m_varV[evID].x[2],
				hMixToolLoad->m_varV[evID].x[3]
				);
  }


} 



int hemiAnalysis::hemiLoop(int maxHemi){

  int nHemis = (maxHemi > 0 && maxHemi < m_nTreeHemis) ? maxHemi : m_nTreeHemis;
  
  std::cout << "\nProcess " << nHemis << " of " << m_nTreeHemis << " events.\n";

  //start = std::clock();//2546000 //2546043
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    
    if( (hemiIdx % 10000 == 0) ) cout << "Process .... " << hemiIdx << "/" << nHemis << endl;

    if(debug) cout << "Getting Hemi " << endl;
    hemisphere thisHemi = hMixToolLoad->getHemi(hemiIdx);
    hemisphereMixTool::EventID thisEventID = { {int(thisHemi.NJets), int(thisHemi.NBJets) } };
    //if(thisHemi.regionIdx == 0) continue;
    if(debug) cout << thisHemi.Run << " " << thisHemi.Event << endl;
    
    
    //UInt_t thisHemiPairIdx = thisHemi.pairIdx;
    if(debug) cout << "Getting NearNeig " << endl;
    hemisphere thisHemiBestMatch = hMixToolLoad->getHemiNearNeig(thisHemi,hemiIdx);

    if(debug) cout << "Filling Hists " << endl;
    hists[thisEventID]->Fill(thisHemi,thisHemiBestMatch);

    //vector<hemisphere> thisHemiBestMatches = hMixToolLoad->getHemiNearNeighbors(thisHemi,hemiIdx,5);

    //      //cout << "\t" << tenNearestDists[i] << " ("<<tenNearestIdx[i] << ")";    

//
//    //
//    //  Get the nearest 5 for plotting
//    //
//    const unsigned int NN = 5;
//    int    multi_nearestIdx[NN];
//    double multi_nearestDists[NN];
//
//
//    thisKDTree->nnearest(localIndex,multi_nearestIdx,multi_nearestDists,NN);
//    double minDist = 999;
//    for(unsigned int i=0; i<NN; ++i){
//      if(multi_nearestDists[i] < minDist) minDist = multi_nearestDists[i];
//      //cout << "\t" << tenNearestDists[i] << " ("<<tenNearestIdx[i] << ")";
//    }
//    //cout << endl;
//
//    if(fabs(minDist - nearestDist[0]) > 0.01)
//      cout << "Diff " << minDist << " vs " << nearestDist[0] << endl;

  }//




  return 0;
}


hemiAnalysis::~hemiAnalysis(){} 

//    //
//    //  DIM 3
//    //
//    kdTree::Point<3> testPoint  = kdTree::Point<3>(1,2,3);
//    kdTree::Point<3> testPoint2 = kdTree::Point<3>(4,5,6);
//    kdTree::Point<3> testPoint3 = kdTree::Point<3>(7,8,6);
//    kdTree::Point<3> testPoint4 = kdTree::Point<3>(0,3,9);
//    kdTree::Point<3> testPoint5 = kdTree::Point<3>(4,5,6.5);
//    kdTree::Point<3> testPoint6 = kdTree::Point<3>(0,3.1,9);
//
//    cout << "Point " << testPoint.x[0] << " " << kdTree::dist<3>(testPoint, testPoint2) << endl;
//    // bestMatchDiff
//    kdTree::Box<3> testBox = kdTree::Box<3>(testPoint, testPoint2);
//    cout << "Box" << testBox.lo.x[0] << endl;
//    
//    std::vector< kdTree::Point<3> > points;
//    points.push_back(testPoint);
//    points.push_back(testPoint2);
//    points.push_back(testPoint3);
//    points.push_back(testPoint4);
//    points.push_back(testPoint5);
//    points.push_back(testPoint6);
//
//    kdTree::kdTree<3> testTree = kdTree::kdTree<3>(points);
//    cout << "Nearest 3D (2) is " << testTree.nearest(testPoint2) << endl; 
//    cout << " \t test to point2 " << endl;
//    for(auto& thisPt : points)
//      cout << kdTree::dist<3>(testPoint2, thisPt) << " ";
//    cout << endl;
//
//    cout << "Nearest 3D (6) is " << testTree.nearest(testPoint6) << endl; 
//    cout << " \t test to point6 " << endl;
//    for(auto& thisPt : points)
//      cout << kdTree::dist<3>(testPoint6, thisPt) << " ";
//    cout << endl;

    //
    //  DIM 4
    //
//    kdTree::Point<4> fd_testPoint  = kdTree::Point<4>(10,1,2,3);
//    kdTree::Point<4> fd_testPoint2 = kdTree::Point<4>(10,4,5,6);
//    kdTree::Point<4> fd_testPoint3 = kdTree::Point<4>(10,7,8,6);
//    kdTree::Point<4> fd_testPoint4 = kdTree::Point<4>(10,0,3,9);
//    kdTree::Point<4> fd_testPoint5 = kdTree::Point<4>(10,4,5,6.5);
//    kdTree::Point<4> fd_testPoint6 = kdTree::Point<4>(10,0,3.1,9);
//
//    cout << "Point " << fd_testPoint.x[0] << " " << kdTree::dist<4>(fd_testPoint, fd_testPoint2) << endl;
//    // bestMatchDiff
//    kdTree::Box<4> fd_testBox = kdTree::Box<4>(fd_testPoint, fd_testPoint2);
//    cout << "Box" << fd_testBox.lo.x[0] << endl;
//    
//    std::vector< kdTree::Point<4> > fd_points;
//    fd_points.push_back(fd_testPoint);
//    fd_points.push_back(fd_testPoint2);
//    fd_points.push_back(fd_testPoint3);
//    fd_points.push_back(fd_testPoint4);
//    fd_points.push_back(fd_testPoint5);
//    fd_points.push_back(fd_testPoint6);
//
//    kdTree::kdTree<4> fd_testTree = kdTree::kdTree<4>(fd_points);
//    cout << "Nearest 4D (2) is " << fd_testTree.nearest(fd_testPoint2) << endl; 
//    cout << " \t test to point2 " << endl;
//    for(auto& thisPt : fd_points)
//      cout << kdTree::dist<4>(fd_testPoint2, thisPt) << " ";
//    cout << endl;
//
//    cout << "Nearest 4D (6) is " << fd_testTree.nearest(fd_testPoint6) << endl; 
//    cout << " \t test to point6 " << endl;
//    for(auto& thisPt : fd_points)
//      cout << kdTree::dist<4>(fd_testPoint6, thisPt) << " ";
//    cout << endl;
    

    //cout << " \t test to point2 " << kdTree::dist<4>(testPoint2, testPoint5) << " " << 
