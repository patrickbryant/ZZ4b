#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"

using namespace nTupleAnalysis;
using std::cout;  using std::endl; 
using std::vector;  

hemiDataHandler::hemiDataHandler(UInt_t nJetBin, UInt_t nBJetBin, UInt_t nNonSelJetBin, bool createLibrary, std::string fileName, std::string name, bool loadJetFourVecs, bool dualAccess, bool debug ) : 
  m_nJetBin(nJetBin), m_nBJetBin(nBJetBin), m_nNonSelJetBin(nNonSelJetBin), m_createLibrary(createLibrary), m_loadJetFourVecs(loadJetFourVecs), m_dualAccess(dualAccess), m_debug(debug)
{

  std::stringstream ss;
  ss << m_nJetBin << "_" << m_nBJetBin << "_" << m_nNonSelJetBin;
  m_EventIDPostFix = ss.str();
  m_random = new TRandom3();

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
    
    hemiFileRandAccess = TFile::Open((fileName).c_str() , "READ");
    //if(m_debug) cout << " hemisphereMixTool::Read file " << hemiFile << endl;
    hemiTreeRandAccess = (TTree*)hemiFileRandAccess->Get("hemiTree");
    //if(m_debug) cout << " hemisphereMixTool::Got Tree " << hemiTree << endl;
    hemiTreeRandAccess->SetBranchStatus("*", 0);
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
  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "NNonSelJets", &m_NNonSelJets, "i");  
  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "pairIdx",     &m_pairIdx, "i");  

  if(m_loadJetFourVecs){
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

  //
  //  For random access
  //
  if(!m_createLibrary && m_dualAccess){

    initBranchesRandAccess();

    connectBranch<UInt_t>     (m_createLibrary, hemiTreeRandAccess, "runNumber",   &m_rand_Run, "i");
    connectBranch<ULong64_t>  (m_createLibrary, hemiTreeRandAccess, "evtNumber",   &m_rand_Event, "l");
    connectBranch<float>      (m_createLibrary, hemiTreeRandAccess, "tAxis_x",     &m_rand_tAxis_x, "F");
    connectBranch<float>      (m_createLibrary, hemiTreeRandAccess, "tAxis_y",     &m_rand_tAxis_y, "F");
    connectBranch<float>      (m_createLibrary, hemiTreeRandAccess, "sumPz",       &m_rand_sumPz         , "F");
    connectBranch<float>      (m_createLibrary, hemiTreeRandAccess, "sumPt_T",     &m_rand_sumPt_T      , "F");
    connectBranch<float>      (m_createLibrary, hemiTreeRandAccess, "sumPt_Ta",    &m_rand_sumPt_Ta    , "F");
    connectBranch<float>      (m_createLibrary, hemiTreeRandAccess, "combinedMass",&m_rand_combinedMass  , "F");
    connectBranch<UInt_t>     (m_createLibrary, hemiTreeRandAccess, "NJets",       &m_rand_NJets, "i");  
    connectBranch<UInt_t>     (m_createLibrary, hemiTreeRandAccess, "NBJets",      &m_rand_NBJets, "i");  
    connectBranch<UInt_t>     (m_createLibrary, hemiTreeRandAccess, "NNonSelJets", &m_rand_NNonSelJets, "i");  
    connectBranch<UInt_t>     (m_createLibrary, hemiTreeRandAccess, "pairIdx",     &m_rand_pairIdx, "i");  

    if(m_loadJetFourVecs){
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_pt",   &m_rand_jet_pt);
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_eta",  &m_rand_jet_eta);
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_phi",  &m_rand_jet_phi);
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_m",    &m_rand_jet_m);
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_e",    &m_rand_jet_e);
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_bRegCorr",    &m_rand_jet_bRegCorr);
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_deepB",    &m_rand_jet_deepB);
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_CSVv2",    &m_rand_jet_CSVv2);
      connectVecBranch<float> (m_createLibrary, hemiTreeRandAccess,  "jet_deepFlavB",    &m_rand_jet_deepFlavB);
      connectVecBranch<Bool_t>(m_createLibrary, hemiTreeRandAccess, "jet_isTag",    &m_rand_jet_isTag);
    }
  }
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



void hemiDataHandler::initBranchesRandAccess() {
  m_rand_jet_pt        = new vector<float>();
  m_rand_jet_eta       = new vector<float>();  
  m_rand_jet_phi       = new vector<float>();  
  m_rand_jet_m         = new vector<float>();  
  m_rand_jet_e         = new vector<float>();  
  m_rand_jet_bRegCorr  = new vector<float>();  
  m_rand_jet_deepB     = new vector<float>();  
  m_rand_jet_CSVv2     = new vector<float>();  
  m_rand_jet_deepFlavB = new vector<float>();  
  m_rand_jet_isTag     = new vector<Bool_t>(); 
}

void hemiDataHandler::clearBranchesRandAccess() {
  m_rand_jet_pt        ->clear();
  m_rand_jet_eta       ->clear();
  m_rand_jet_phi       ->clear();
  m_rand_jet_m         ->clear();
  m_rand_jet_e         ->clear();
  m_rand_jet_bRegCorr  ->clear();
  m_rand_jet_deepB     ->clear();
  m_rand_jet_CSVv2     ->clear();
  m_rand_jet_deepFlavB ->clear();
  m_rand_jet_isTag     ->clear();
}



hemiPtr hemiDataHandler::getHemi(unsigned int entry, bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemi " << endl;
  hemiTree->GetEntry(entry);
  if(m_debug) cout << "got entry " << endl;
  hemiPtr outHemi = std::make_shared<hemisphere>(hemisphere(m_Run, m_Event, m_tAxis_x, m_tAxis_y));
  outHemi->sumPz = m_sumPz;
  outHemi->sumPt_T = m_sumPt_T;
  outHemi->sumPt_Ta = m_sumPt_Ta;
  outHemi->combinedMass = m_combinedMass;
  outHemi->NJets = m_NJets;
  outHemi->NBJets = m_NBJets;
  outHemi->NNonSelJets = m_NNonSelJets;
  outHemi->pairIdx = m_pairIdx;

  if(m_debug) cout << "Make hemi " << loadJets << " " << m_loadJetFourVecs << endl;

  if(loadJets && m_loadJetFourVecs){
    if(m_debug) cout << "load JetFourVecs " << endl;
    outHemi->allJets.clear();
      
    unsigned int nJets = m_jet_pt->size();
    for(unsigned int iJet = 0; iJet < nJets; ++iJet){
      outHemi->allJets.push_back(std::make_shared<jet>(jet()));
      outHemi->allJets.back()->pt  = m_jet_pt ->at(iJet);          
      outHemi->allJets.back()->eta = m_jet_eta->at(iJet);          
      outHemi->allJets.back()->phi = m_jet_phi->at(iJet);          
      outHemi->allJets.back()->m   = m_jet_m  ->at(iJet);          
      outHemi->allJets.back()->p   = TLorentzVector();
      outHemi->allJets.back()->p.SetPtEtaPhiM(m_jet_pt ->at(iJet), 
					     m_jet_eta->at(iJet), 
					     m_jet_phi->at(iJet), 
					     m_jet_m ->at(iJet));
      outHemi->allJets.back()->e = outHemi->allJets.back()->p.E();
      outHemi->allJets.back()->bRegCorr  = m_jet_bRegCorr->at(iJet);  
      outHemi->allJets.back()->deepB     = m_jet_deepB->at(iJet);        
      outHemi->allJets.back()->CSVv2     = m_jet_CSVv2->at(iJet);        
      outHemi->allJets.back()->deepFlavB = m_jet_deepFlavB->at(iJet); 
    }

  }


  if(m_debug) cout << "Leave hemiDataHandler::getHemi " << endl;
  return outHemi;
}

hemiPtr hemiDataHandler::getHemiRandom(bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemi " << endl;
  int randIndx = int(m_random->Rndm()*m_nTot);

  hemiTreeRandAccess->GetEntry(randIndx);
  hemiPtr outHemi = std::make_shared<hemisphere>(hemisphere(m_rand_Run, m_rand_Event, m_rand_tAxis_x, m_rand_tAxis_y));
  outHemi->sumPz = m_rand_sumPz;
  outHemi->sumPt_T = m_rand_sumPt_T;
  outHemi->sumPt_Ta = m_rand_sumPt_Ta;
  outHemi->combinedMass = m_rand_combinedMass;
  outHemi->NJets = m_rand_NJets;
  outHemi->NBJets = m_rand_NBJets;
  outHemi->NNonSelJets = m_rand_NNonSelJets;
  outHemi->pairIdx = m_rand_pairIdx;



  if(loadJets && m_loadJetFourVecs){
    if(m_debug) cout << "load JetFourVecs " << endl;
    outHemi->allJets.clear();
      
    unsigned int nJets = m_rand_jet_pt->size();
    for(unsigned int iJet = 0; iJet < nJets; ++iJet){
      outHemi->allJets.push_back(std::make_shared<jet>(jet()));
      outHemi->allJets.back()->pt  = m_rand_jet_pt ->at(iJet);          
      outHemi->allJets.back()->eta = m_rand_jet_eta->at(iJet);          
      outHemi->allJets.back()->phi = m_rand_jet_phi->at(iJet);          
      outHemi->allJets.back()->m   = m_rand_jet_m  ->at(iJet);          
      outHemi->allJets.back()->p   = TLorentzVector();
      outHemi->allJets.back()->p.SetPtEtaPhiM(m_rand_jet_pt ->at(iJet), 
					     m_rand_jet_eta->at(iJet), 
					     m_rand_jet_phi->at(iJet), 
					     m_rand_jet_m ->at(iJet));
      outHemi->allJets.back()->e = outHemi->allJets.back()->p.E();
      outHemi->allJets.back()->bRegCorr  = m_rand_jet_bRegCorr->at(iJet);  
      outHemi->allJets.back()->deepB     = m_rand_jet_deepB->at(iJet);        
      outHemi->allJets.back()->CSVv2     = m_rand_jet_CSVv2->at(iJet);        
      outHemi->allJets.back()->deepFlavB = m_rand_jet_deepFlavB->at(iJet); 
    }

  }

  return outHemi;
}



hemiPtr hemiDataHandler::getHemiRandAccess(unsigned int entry, bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemiRandAccess " << endl;
  hemiTreeRandAccess->GetEntry(entry);
  hemiPtr outHemi = std::make_shared<hemisphere>(hemisphere(m_rand_Run, m_rand_Event, m_rand_tAxis_x, m_rand_tAxis_y));
  outHemi->sumPz = m_rand_sumPz;
  outHemi->sumPt_T = m_rand_sumPt_T;
  outHemi->sumPt_Ta = m_rand_sumPt_Ta;
  outHemi->combinedMass = m_rand_combinedMass;
  outHemi->NJets = m_rand_NJets;
  outHemi->NBJets = m_rand_NBJets;
  outHemi->NNonSelJets = m_rand_NNonSelJets;
  outHemi->pairIdx = m_rand_pairIdx;

  if(loadJets && m_loadJetFourVecs){
    if(m_debug) cout << "load JetFourVecs " << endl;
    outHemi->allJets.clear();
      
    unsigned int nJets = m_rand_jet_pt->size();
    for(unsigned int iJet = 0; iJet < nJets; ++iJet){
      outHemi->allJets.push_back(std::make_shared<jet>(jet()));
      outHemi->allJets.back()->pt  = m_rand_jet_pt ->at(iJet);          
      outHemi->allJets.back()->eta = m_rand_jet_eta->at(iJet);          
      outHemi->allJets.back()->phi = m_rand_jet_phi->at(iJet);          
      outHemi->allJets.back()->m   = m_rand_jet_m  ->at(iJet);          
      outHemi->allJets.back()->p   = TLorentzVector();
      outHemi->allJets.back()->p.SetPtEtaPhiM(m_rand_jet_pt ->at(iJet), 
					     m_rand_jet_eta->at(iJet), 
					     m_rand_jet_phi->at(iJet), 
					     m_rand_jet_m ->at(iJet));
      outHemi->allJets.back()->e = outHemi->allJets.back()->p.E();
      outHemi->allJets.back()->bRegCorr  = m_rand_jet_bRegCorr->at(iJet);  
      outHemi->allJets.back()->deepB     = m_rand_jet_deepB->at(iJet);        
      outHemi->allJets.back()->CSVv2     = m_rand_jet_CSVv2->at(iJet);        
      outHemi->allJets.back()->deepFlavB = m_rand_jet_deepFlavB->at(iJet); 
    }

  }


  if(m_debug) cout << "Leave hemiDataHandler::getHemiRandAccess " << endl;
  return outHemi;
}
 
hemiPtr hemiDataHandler::getHemiNearNeig(unsigned int entry, bool loadJets){
  if(m_debug) cout << "In hemiDataHandler::getHemiNearNeig " << endl;

  if(!m_kdTree){
    cout << "Warning no KDtree defined (" << m_nJetBin << " " << m_nBJetBin << " ) " << endl;
    return std::make_shared<hemisphere>(hemisphere(0,0,0,0));
  }

  //
  //  Get the nearest
  //
  int    nearestIdx[1];
  double nearestDist[1];
  m_kdTree->nnearest(entry,nearestIdx,nearestDist,1);

  if(nearestDist[0] > 10){
    cout << "ERROR: closest distance is " << nearestDist[0] << " (" << m_nJetBin << " " << m_nBJetBin << " ) " << entry << endl;
    return std::make_shared<hemisphere>(hemisphere(0,0,0,0));
  }

  if(m_debug) cout << " getHemiNearNeig " << loadJets << " " << m_loadJetFourVecs << endl;
  //return hemisphere(0,0,0,0);
  if(m_debug) cout << "Leave hemiDataHandler::getHemiNearNeig " << endl;
  if(m_dualAccess)
    return this->getHemiRandAccess(nearestIdx[0],loadJets);
  return this->getHemi(nearestIdx[0],loadJets);
}



hemiPtr hemiDataHandler::getHemiNearNeig(const hemiPtr& hIn, bool loadJets){
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
  //  Now get the nearest neighbor
  //
  return getHemiNearNeig(indexThisHemi, loadJets);
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
    return;
  }

  //
  //  Sort by event index
  // 
  unsigned int nHemis = hemiTree->GetEntries();
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    hemiPtr thisHemi = this->getHemi(hemiIdx);
    
    m_hemiPoints.push_back(hemiPoint(thisHemi->sumPz       /m_varV.x[0], 
				     thisHemi->sumPt_T     /m_varV.x[1], 
				     thisHemi->sumPt_Ta    /m_varV.x[2], 
				     thisHemi->combinedMass/m_varV.x[3])
			   );

  }

  //
  //  Make the kd-trees
  // 

  cout << " Making KD tree with " << m_hemiPoints.size() << " entries." << endl;
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

  if(m_nTot > 100) m_isValid = true;
  
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
