

#include "ZZ4b/nTupleAnalysis/interface/hemisphere.h"
#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"
#include <cmath>

using namespace nTupleAnalysis;

using std::cout; using std::endl; 

void hemisphere::write(hemisphereMixTool* hMixTool, int localPairIndex){

  UInt_t nJets = (tagJets.size() + nonTagJets.size());
  UInt_t nBJets = tagJets.size();
  UInt_t nNonSelJets = nonSelJets.size();
  hemisphereMixTool::EventID thisEventID = { {int(nJets), int(nBJets), int(nNonSelJets)} };
  
  hemiDataHandler* dataHandler = hMixTool->getDataHandler(thisEventID);

  dataHandler->clearBranches();

  dataHandler->m_Run   = Run;
  dataHandler->m_Event = Event;
  dataHandler->m_tAxis_x = thrustAxis.X();
  dataHandler->m_tAxis_y = thrustAxis.Y();
  dataHandler->m_sumPz        = sumPz;
  dataHandler->m_sumPt_T 	   = sumPt_T;
  dataHandler->m_sumPt_Ta	   = sumPt_Ta;
  dataHandler->m_combinedMass = combinedMass;
  dataHandler->m_NJets = nJets;
  dataHandler->m_NBJets = tagJets.size();
  dataHandler->m_NNonSelJets = nonSelJets.size();
  
  if(hMixTool->m_debug) cout << " \t pairIdx " << dataHandler->hemiTree->GetEntries() + localPairIndex << endl;
  dataHandler->m_pairIdx = (dataHandler->hemiTree->GetEntries() + localPairIndex);
  

  for(const jetPtr& tagJet : tagJets){
    if(tagJet->appliedBRegression) tagJet->scaleFourVector(1./tagJet->bRegCorr);

    dataHandler->m_jet_pt        ->push_back(tagJet->pt);
    dataHandler->m_jet_eta       ->push_back(tagJet->eta);  
    dataHandler->m_jet_phi       ->push_back(tagJet->phi);  
    dataHandler->m_jet_m         ->push_back(tagJet->m);  
    dataHandler->m_jet_e         ->push_back(tagJet->e);  
    dataHandler->m_jet_bRegCorr  ->push_back(tagJet->bRegCorr);  
    dataHandler->m_jet_deepB     ->push_back(tagJet->deepB);  
    dataHandler->m_jet_CSVv2     ->push_back(tagJet->CSVv2);  
    dataHandler->m_jet_deepFlavB ->push_back(tagJet->deepFlavB);  
    dataHandler->m_jet_cleanmask ->push_back(tagJet->cleanmask);  
    dataHandler->m_jet_isTag     ->push_back(true);
    dataHandler->m_jet_isSel     ->push_back(true);
  }

  for(const jetPtr& nonTagJet : nonTagJets){
    if(nonTagJet->appliedBRegression) nonTagJet->scaleFourVector(1./nonTagJet->bRegCorr);

    dataHandler->m_jet_pt        ->push_back(nonTagJet->pt);
    dataHandler->m_jet_eta       ->push_back(nonTagJet->eta);  
    dataHandler->m_jet_phi       ->push_back(nonTagJet->phi);  
    dataHandler->m_jet_m         ->push_back(nonTagJet->m);  
    dataHandler->m_jet_e         ->push_back(nonTagJet->e);  
    dataHandler->m_jet_bRegCorr  ->push_back(nonTagJet->bRegCorr);  
    dataHandler->m_jet_deepB     ->push_back(nonTagJet->deepB);  
    dataHandler->m_jet_CSVv2     ->push_back(nonTagJet->CSVv2);  
    dataHandler->m_jet_deepFlavB ->push_back(nonTagJet->deepFlavB);  
    dataHandler->m_jet_cleanmask ->push_back(nonTagJet->cleanmask);  
    dataHandler->m_jet_isTag     ->push_back(false);
    dataHandler->m_jet_isSel     ->push_back(true);
  }

  for(const jetPtr& nonSelJet : nonSelJets){
    if(nonSelJet->appliedBRegression) nonSelJet->scaleFourVector(1./nonSelJet->bRegCorr);

    dataHandler->m_jet_pt        ->push_back(nonSelJet->pt);
    dataHandler->m_jet_eta       ->push_back(nonSelJet->eta);  
    dataHandler->m_jet_phi       ->push_back(nonSelJet->phi);  
    dataHandler->m_jet_m         ->push_back(nonSelJet->m);  
    dataHandler->m_jet_e         ->push_back(nonSelJet->e);  
    dataHandler->m_jet_bRegCorr  ->push_back(nonSelJet->bRegCorr);  
    dataHandler->m_jet_deepB     ->push_back(nonSelJet->deepB);  
    dataHandler->m_jet_CSVv2     ->push_back(nonSelJet->CSVv2);  
    dataHandler->m_jet_deepFlavB ->push_back(nonSelJet->deepFlavB);  
    dataHandler->m_jet_cleanmask ->push_back(nonSelJet->cleanmask);  
    dataHandler->m_jet_isTag     ->push_back(false);
    dataHandler->m_jet_isSel     ->push_back(false);
  }


  dataHandler->hemiTree->Fill();

}


void hemisphere::rotateTo(const TVector2& newTAxis, bool usePositiveHalf){

    
  bool  isPositiveHalf = (thrustAxis * TVector2(combinedVec.Px(), combinedVec.Py()) > 0);
  //cout << " \t\t thisThrust " << thrustAxis.Phi() << endl;
  //cout << " \t\t isPositiveHalf " << isPositiveHalf << endl;
  if( (!isPositiveHalf && usePositiveHalf ) || (isPositiveHalf && !usePositiveHalf )){
    //cout << "Rotating "
    thrustAxis = thrustAxis.Rotate(M_PI);
  }
  //cout << " \t\t thisThrust " << thrustAxis.Phi() << endl;
  float delta_phi = newTAxis.DeltaPhi(thrustAxis);
  //cout << " \t\t dPhi " << delta_phi << endl;  
  thrustAxis = thrustAxis.Rotate(delta_phi);
  
  combinedVec = TLorentzVector();
  for(nTupleAnalysis::jetPtr& jet: tagJets){
    jet->RotateZ(delta_phi);
    combinedVec += jet->p;
  }
  for(nTupleAnalysis::jetPtr& jet: nonTagJets){
    jet->RotateZ(delta_phi);
    combinedVec += jet->p;
  }

  for(nTupleAnalysis::jetPtr& jet: nonSelJets){
    jet->RotateZ(delta_phi);
    combinedVec += jet->p;
  }


}


void hemisphere::addJet(const jetPtr& thisJet, const std::vector<jetPtr>& selJetRef, const std::vector<jetPtr>& tagJetRef){
       
  combinedVec += thisJet->p;
  combinedMass = combinedVec.M();
      
  sumPz += thisJet->p.Pz();
  TVector2 thisJetPt = TVector2(thisJet->p.Px(), thisJet->p.Py());
            
  sumPt_T  += fabs(thisJetPt*thrustAxis);
  sumPt_Ta += fabs(thisJetPt*thrustAxisPerp);

  if(find(selJetRef.begin(), selJetRef.end(), thisJet) != selJetRef.end()){
    if(find(tagJetRef.begin(), tagJetRef.end(), thisJet) != tagJetRef.end()){
      tagJets.push_back(thisJet);
    }else{
      nonTagJets.push_back(thisJet);
    }
  }else{
    nonSelJets.push_back(thisJet);
  }

}

hemisphere::~hemisphere(){
  nonTagJets.clear();
  tagJets.clear();
  nonSelJets.clear();
  
} 
