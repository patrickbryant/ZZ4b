#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"

using namespace nTupleAnalysis;
using std::cout;  using std::endl; 
using std::vector;  

hemiDataHandler::hemiDataHandler(UInt_t nJetBin, UInt_t nBJetBin, bool createLibrary, std::string fileName, std::string name ) : 
  m_nJetBin(nJetBin), m_nBJetBin(nBJetBin), m_createLibrary(createLibrary)
{

  std::stringstream ss;
  ss << m_nJetBin << "_" << m_nBJetBin;
  m_EventIDPostFix = ss.str();


  if(m_createLibrary){
    hemiFile = TFile::Open((fileName+"_"+name+"_"+m_EventIDPostFix+".root").c_str() , "RECREATE");
    hemiTree = new TTree("hemiTree","Tree for hemishpere mixing");
  }else{
    hemiFile = TFile::Open((fileName).c_str() , "READ");
    //if(m_debug) cout << " hemisphereMixTool::Read file " << hemiFile << endl;
    hemiTree = (TTree*)hemiFile->Get("hemiTree");
    //if(m_debug) cout << " hemisphereMixTool::Got Tree " << hemiTree << endl;
  }

      
  initBranches();

  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "runNumber",   &m_Run, "i");
  connectBranch<ULong64_t>  (m_createLibrary, hemiTree, "evtNumber",   &m_Event, "l");
  connectBranch<float>      (m_createLibrary, hemiTree, "tAxis_x",     &m_tAxis_x, "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "tAxis_y",     &m_tAxis_y, "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "sumPz",       &m_sumPz         , "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "sumPt_T",     &m_sumPt_T      , "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "sumPt_Ta",    &m_sumPt_Ta    , "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "combinedMass",&m_combinedMass  , "F");
  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "NJets",       &m_NJets, "i");  
  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "NBJets",      &m_NBJets, "i");  
  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "pairIdx",     &m_pairIdx, "i");  


  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_pt",   &m_jet_pt);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_eta",  &m_jet_eta);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_phi",  &m_jet_phi);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_m",    &m_jet_m);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_e",    &m_jet_e);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_bRegCorr",    &m_jet_bRegCorr);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_deepB",    &m_jet_deepB);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_CSVv2",    &m_jet_CSVv2);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_deepFlavB",    &m_jet_deepFlavB);
  connectVecBranch<Bool_t>(m_createLibrary, hemiTree, "jet_isTag",    &m_jet_isTag);


}


void hemiDataHandler::initBranches() {
  m_jet_pt        = new vector<float>();
  m_jet_eta       = new vector<float>();  
  m_jet_phi       = new vector<float>();  
  m_jet_m         = new vector<float>();  
  m_jet_e         = new vector<float>();  
  m_jet_bRegCorr  = new vector<float>();  
  m_jet_deepB     = new vector<float>();  
  m_jet_CSVv2     = new vector<float>();  
  m_jet_deepFlavB = new vector<float>();  
  m_jet_isTag     = new vector<Bool_t>(); 
}

void hemiDataHandler::clearBranches() {
  m_jet_pt        ->clear();
  m_jet_eta       ->clear();
  m_jet_phi       ->clear();
  m_jet_m         ->clear();
  m_jet_e         ->clear();
  m_jet_bRegCorr  ->clear();
  m_jet_deepB     ->clear();
  m_jet_CSVv2     ->clear();
  m_jet_deepFlavB ->clear();
  m_jet_isTag     ->clear();
}



hemisphere hemiDataHandler::getHemi(unsigned int entry){
  if(debug) cout << "In hemiDataHandler::getHemi " << endl;
  hemiTree->GetEntry(entry);
  hemisphere outHemi = hemisphere(m_Run, m_Event, m_tAxis_x, m_tAxis_y);
  outHemi.sumPz = m_sumPz;
  outHemi.sumPt_T = m_sumPt_T;
  outHemi.sumPt_Ta = m_sumPt_Ta;
  outHemi.combinedMass = m_combinedMass;
  outHemi.NJets = m_NJets;
  outHemi.NBJets = m_NBJets;
  outHemi.pairIdx = m_pairIdx;
  if(debug) cout << "Leave hemiDataHandler::getHemi " << endl;
  return outHemi;
}

hemisphere hemiDataHandler::getHemiNearNeig(const hemisphere& hIn, unsigned int entry){
  if(debug) cout << "In hemiDataHandler::getHemiNearNeig " << endl;

  if(!m_kdTree){
    cout << "Warning no KDtree defined (" << hIn.NJets << " " << hIn.NBJets << " ) " << endl;
    return hemisphere(0,0,0,0);
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
  int    nearestIdx[1];
  double nearestDist[1];
  m_kdTree->nnearest(entry,nearestIdx,nearestDist,1);

  //cout << "closest distance is " << nearestDist[0] << " (" << hIn.NJets << " " << hIn.NBJets << " ) " << localIndex << endl;

  if(nearestDist[0] > 10){
    cout << "ERROR: closest distance is " << nearestDist[0] << " (" << hIn.NJets << " " << hIn.NBJets << " ) " << entry << endl;
    return hemisphere(0,0,0,0);
  }

  //if(debug) cout << "Getting globalIdxOfMatch " << endl;  
  //long int globalIdxOfMatch = m_EventIndex.at(nearestIdx[0]);

  //return hemisphere(0,0,0,0);
  if(debug) cout << "Leave hemiDataHandler::getHemiNearNeig " << endl;
  return this->getHemi(nearestIdx[0]);
}


void hemiDataHandler::buildData(){
  this->calcVariance();

  m_hemiPoints.clear();

  //
  //  Sort by event index
  // 
  unsigned int nHemis = hemiTree->GetEntries();
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    hemisphere thisHemi = this->getHemi(hemiIdx);
    
    m_hemiPoints.push_back(hemiPoint(thisHemi.sumPz       /m_varV.x[0], 
				     thisHemi.sumPt_T     /m_varV.x[1], 
				     thisHemi.sumPt_Ta    /m_varV.x[2], 
				     thisHemi.combinedMass/m_varV.x[3])
			   );

  }

  //
  //  Make the kd-trees
  // 
  if(m_hemiPoints.size() < 3) return;
  cout << " Making KD tree " << m_hemiPoints.size() << endl;
  m_kdTree = new hemiKDtree(m_hemiPoints);
  cout << " Done " << endl;
}

void hemiDataHandler::calcVariance(){

  m_sumV  = hemiPoint(0,0,0,0);
  m_sumV2 = hemiPoint(0,0,0,0);
  m_nTot  = 0;
  m_varV  = hemiPoint(0,0,0,0);

  unsigned int nHemis = hemiTree->GetEntries();
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    hemisphere thisHemi = this->getHemi(hemiIdx);
    
    m_nTot += 1;

    m_sumV.x[0] += thisHemi.sumPz;
    m_sumV.x[1] += thisHemi.sumPt_T;
    m_sumV.x[2] += thisHemi.sumPt_Ta;
    m_sumV.x[3] += thisHemi.combinedMass;

    m_sumV2.x[0] += (thisHemi.sumPz        * thisHemi.sumPz);       
    m_sumV2.x[1] += (thisHemi.sumPt_T      * thisHemi.sumPt_T);     
    m_sumV2.x[2] += (thisHemi.sumPt_Ta     * thisHemi.sumPt_Ta);    
    m_sumV2.x[3] += (thisHemi.combinedMass * thisHemi.combinedMass);
  }


  m_varV.x[0] = sqrt((m_sumV2.x[0] - (m_sumV.x[0]*m_sumV.x[0])/m_nTot)/m_nTot);
  m_varV.x[1] = sqrt((m_sumV2.x[1] - (m_sumV.x[1]*m_sumV.x[1])/m_nTot)/m_nTot);
  m_varV.x[2] = sqrt((m_sumV2.x[2] - (m_sumV.x[2]*m_sumV.x[2])/m_nTot)/m_nTot);
  m_varV.x[3] = sqrt((m_sumV2.x[3] - (m_sumV.x[3]*m_sumV.x[3])/m_nTot)/m_nTot);
  cout << " var for " << m_nJetBin << "/" << m_nBJetBin << ": " 
       << m_varV.x[0] << " / " 
       << m_varV.x[1] << " / " 
       << m_varV.x[2] << " / " 
       << m_varV.x[3] 
       << endl;
  

  return;
}
