#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"

using namespace nTupleAnalysis;
using std::cout;  using std::endl; 
using std::vector;  

hemiDataHandler::hemiDataHandler(EventID thisEventID, bool createLibrary, std::string fileName, std::string name, int maxNHemis, bool loadJetFourVecs, bool dualAccess, bool debug ) : 
  NUMBER_MAX_HEMIS(maxNHemis), m_createLibrary(createLibrary), m_loadJetFourVecs(loadJetFourVecs), m_dualAccess(dualAccess), m_debug(debug)
{
  m_nJetBin = thisEventID.at(0);
  m_nBJetBin = thisEventID.at(1);
  m_nNonSelJetBin = thisEventID.at(2);

  std::stringstream ss;
  ss << m_nJetBin << "_" << m_nBJetBin << "_" << m_nNonSelJetBin;
  m_EventIDPostFix = ss.str();
  m_random = new TRandom3();

  m_isValid = true;

  if(m_createLibrary){
    hemiFile = TFile::Open((fileName+"_"+name+"_"+m_EventIDPostFix+".root").c_str() , "RECREATE");
    hemiTree = new TTree("hemiTree","Tree for hemishpere mixing");
    cout << " hemiDataHandler::created " << fileName+"_"+name+"_"+m_EventIDPostFix+".root" << " " << hemiFile << " " << hemiTree << endl;
  }else{
    hemiFile = TFile::Open((fileName).c_str() , "READ");
    //if(m_debug) cout << " hemisphereMixTool::Read file " << hemiFile << endl;
    hemiTree = (TTree*)hemiFile->Get("hemiTree");
    //if(m_debug) cout << " hemisphereMixTool::Got Tree " << hemiTree << endl;
    hemiTree->SetBranchStatus("*", 0);

    if(hemiTree->GetEntries() < NUMBER_MIN_HEMIS){
      m_isValid = false;
      hemiFile->Close();
      return;
    }

    hemiFileRandAccess = TFile::Open((fileName).c_str() , "READ");
    //if(m_debug) cout << " hemisphereMixTool::Read file " << hemiFile << endl;
    hemiTreeRandAccess = (TTree*)hemiFileRandAccess->Get("hemiTree");
    //if(m_debug) cout << " hemisphereMixTool::Got Tree " << hemiTree << endl;
    hemiTreeRandAccess->SetBranchStatus("*", 0);
  }

  bool readIn = !m_createLibrary;

  m_hemiData = new hemisphereData("hemiData", hemiTree, readIn, m_loadJetFourVecs);


  //
  //  For random access
  //
  if(!m_createLibrary && m_dualAccess){
    m_hemiData_randAccess = new hemisphereData("hemiDataRandAccess", hemiTreeRandAccess, readIn, m_loadJetFourVecs);
  }

}



hemiPtr hemiDataHandler::getHemi(unsigned int entry, bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemi  entry= " << entry << " loadJets= " << loadJets << endl ;
  if(m_debug) cout << "Hemi file is " << hemiFile->GetName() << " " << m_EventIDPostFix << endl;
  hemiTree->GetEntry(entry);
  return m_hemiData->getHemi(loadJets);
}

hemiPtr hemiDataHandler::getHemiRandom(bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemi " << endl;
  int randIndx = int(m_random->Rndm()*m_nTot);
  
  return getHemiRandAccess(randIndx, loadJets);
}



hemiPtr hemiDataHandler::getHemiRandAccess(unsigned int entry, bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemiRandAccess " << endl;
  hemiTreeRandAccess->GetEntry(entry);
  return m_hemiData_randAccess->getHemi(loadJets);
}
 
hemiPtr hemiDataHandler::getHemiNearNeig(unsigned int entry, double& matchDist, bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemiNearNeig " << endl;

  if(!m_kdTree){
    cout << "Warning no KDtree defined (" << m_nJetBin << " " << m_nBJetBin << " " << m_nNonSelJetBin << " ) " << endl;
    return std::make_shared<hemisphere>(hemisphere(0,0,0,0));
  }

  //
  //  Get the nearest
  //
  int    nearestIdx[1];
  double nearestDist[1];
  m_kdTree->nnearest(entry,nearestIdx,nearestDist,1);

  if(nearestDist[0] > 10){
    cout << "ERROR: closest distance is " << nearestDist[0] << " (" << m_nJetBin << " " << m_nBJetBin << " " << m_nNonSelJetBin << " ) " << entry << endl;
    //return std::make_shared<hemisphere>(hemisphere(0,0,0,0));
  }

  matchDist = nearestDist[0];

  if(m_debug) cout << " getHemiNearNeig " << loadJets << " " << m_loadJetFourVecs << endl;
  //return hemisphere(0,0,0,0);
  if(m_debug) cout << "Leave hemiDataHandler::getHemiNearNeig " << endl;
  if(m_dualAccess)
    return this->getHemiRandAccess(nearestIdx[0],loadJets);
  return this->getHemi(nearestIdx[0],loadJets);
}



hemiPtr hemiDataHandler::getHemiNearNeig(const hemiPtr& hIn, double& dist, bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemiNearNeig (hIn) " << endl;

  // First we get index that corrisponds to the hemisphere

  if(!m_kdTree){
    cout << "Warning no KDtree defined (" << hIn->NJets << " " << hIn->NBJets << " ) " << endl;
    return std::make_shared<hemisphere>(hemisphere(0,0,0,0));
  }

  hemiPoint hData = hemiPoint(hIn->sumPz       /m_varV.x[0], 
			      hIn->sumPt_T     /m_varV.x[1], 
			      hIn->sumPt_Ta    /m_varV.x[2], 
			      hIn->combinedMass/m_varV.x[3]);
  
  int indexThisHemi = m_kdTree->nearest(hData);

  //
  // Check to see if this is the intpu hemi or NOT
  // 


  //
  //  Now get the nearest neighbor
  //
  return getHemiNearNeig(indexThisHemi, dist, loadJets);
}

std::vector<hemiPtr> hemiDataHandler::getHemiNearNeighbors(unsigned int entry, unsigned int nNeighbors, bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemiNearNeighbors " << endl;

  std::vector<hemiPtr> outputHemis;

  if(!m_kdTree){
    cout << "Warning no KDtree defined (" << m_nJetBin << " " << m_nBJetBin << " ) " << endl;
    return outputHemis;
  }

  //
  //  Local index of this event
  //
  // (porbably want to precompute these local distances)
  // Might not be needed now....
  //int localIndex = std::distance(m_EventIndex.begin(), find(m_EventIndex.begin(), m_EventIndex.end(), entry));

  //
  //  Get the nearest
  //
  int    nearestIdx[nNeighbors];
  double nearestDist[nNeighbors];
  m_kdTree->nnearest(entry,nearestIdx,nearestDist,nNeighbors);

  //cout << "closest distance is " << nearestDist[0] << " (" << hIn.NJets << " " << hIn.NBJets << " ) " << localIndex << endl;

  for(unsigned int i = 0; i < nNeighbors; ++i){
    if(nearestDist[i] > 10){
      cout << "ERROR: closest distance is " << nearestDist[i] << " (" << m_nJetBin << " " << m_nBJetBin << " ) " << entry << endl;
      continue;
    }

    outputHemis.push_back(this->getHemiRandAccess(nearestIdx[i], loadJets));
  }

  //if(debug) cout << "Getting globalIdxOfMatch " << endl;  
  //long int globalIdxOfMatch = m_EventIndex.at(nearestIdx[0]);

  //return hemisphere(0,0,0,0);
  if(m_debug) cout << "Leave hemiDataHandler::getHemiNearNeig " << endl;
  return outputHemis;
}




void hemiDataHandler::buildData(){
  if(m_debug) cout << "hemiDataHandler::buildData In: " << endl;
  this->calcVariance();
  if(m_debug) cout << "got variances: " << endl;

  m_hemiPoints.clear();

  if(!m_isValid){
    if(m_debug) cout << "Not valid: " << endl;
    if(m_debug) cout << "hemiDataHandler::buildData Leave: " << endl;
    if(hemiFile) hemiFile->Close();
    if(hemiFileRandAccess) hemiFileRandAccess->Close();
    
    return;
  }

  //
  //  Sort by event index
  // 
  unsigned int nHemis = hemiTree->GetEntries();
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    hemiPtr thisHemi = this->getHemi(hemiIdx);

    if((NUMBER_MAX_HEMIS > 0) && (hemiIdx > NUMBER_MAX_HEMIS)) break;

    if(thisHemi->NJets != m_nJetBin)    cout << "ERROR hemiDataHandler::Sel jet counts dont match " << thisHemi->NJets << " vs " << m_nJetBin << endl;
    if(thisHemi->NBJets != m_nBJetBin)  cout << "ERROR hemiDataHandler::Tag jet counts dont match " << thisHemi->NBJets << " vs " << m_nBJetBin << endl;
    if(thisHemi->NNonSelJets != m_nNonSelJetBin) cout << "ERROR hemiDataHandler::NonSel jet counts dont match " << thisHemi->NNonSelJets << " vs " << m_nNonSelJetBin << endl;
    
    m_hemiPoints.push_back(hemiPoint(thisHemi->sumPz       /m_varV.x[0], 
				     thisHemi->sumPt_T     /m_varV.x[1], 
				     thisHemi->sumPt_Ta    /m_varV.x[2], 
				     thisHemi->combinedMass/m_varV.x[3])
			   );

  }

  //
  //  Make the kd-trees
  // 

  cout << " Making KD tree with " << m_hemiPoints.size() << " of " << nHemis << " entries " << endl;
  m_kdTree = new hemiKDtree(m_hemiPoints);
  if(m_debug) cout << "hemiDataHandler::buildData Leave " << endl;
}

void hemiDataHandler::calcVariance(){

  m_sumV  = hemiPoint(0,0,0,0);
  m_sumV2 = hemiPoint(0,0,0,0);
  m_nTot  = 0;
  m_varV  = hemiPoint(0,0,0,0);

  unsigned int nHemis = hemiTree->GetEntries();
  if(m_debug) cout << "nHemis is " << nHemis << endl;
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){

    if((NUMBER_MAX_HEMIS > 0) && (hemiIdx > NUMBER_MAX_HEMIS)) break;
    
    hemiPtr thisHemi = this->getHemi(hemiIdx);

    m_nTot += 1;
    
    m_sumV.x[0] += thisHemi->sumPz;
    m_sumV.x[1] += thisHemi->sumPt_T;
    m_sumV.x[2] += thisHemi->sumPt_Ta;
    m_sumV.x[3] += thisHemi->combinedMass;

    m_sumV2.x[0] += (thisHemi->sumPz        * thisHemi->sumPz);       
    m_sumV2.x[1] += (thisHemi->sumPt_T      * thisHemi->sumPt_T);     
    m_sumV2.x[2] += (thisHemi->sumPt_Ta     * thisHemi->sumPt_Ta);    
    m_sumV2.x[3] += (thisHemi->combinedMass * thisHemi->combinedMass);
  }

  
  if(m_isValid){
    m_varV.x[0] = sqrt((m_sumV2.x[0] - (m_sumV.x[0]*m_sumV.x[0])/m_nTot)/m_nTot);
    m_varV.x[1] = sqrt((m_sumV2.x[1] - (m_sumV.x[1]*m_sumV.x[1])/m_nTot)/m_nTot);
    m_varV.x[2] = sqrt((m_sumV2.x[2] - (m_sumV.x[2]*m_sumV.x[2])/m_nTot)/m_nTot);
    m_varV.x[3] = sqrt((m_sumV2.x[3] - (m_sumV.x[3]*m_sumV.x[3])/m_nTot)/m_nTot);
    cout << m_nTot << " nHemis for " << m_nJetBin << "/" << m_nBJetBin << " / " << m_nNonSelJetBin << " variances : " 
	 << m_varV.x[0] << " / " 
	 << m_varV.x[1] << " / " 
	 << m_varV.x[2] << " / " 
	 << m_varV.x[3] 
	 << endl;
  }

  return;
}
