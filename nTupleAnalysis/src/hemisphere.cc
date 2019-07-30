#include "ZZ4b/nTupleAnalysis/interface/hemisphere.h"
#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"

using namespace nTupleAnalysis;

using std::cout; using std::endl; 

void hemisphere::write(hemisphereMixTool* hMixTool, int localPairIndex){

  UInt_t nJets = (tagJets.size() + nonTagJets.size());
  UInt_t nBJets = tagJets.size();
  hemisphereMixTool::EventID thisEventID = { {int(nJets), int(nBJets)} };

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
  
  if(hMixTool->m_debug) cout << " \t pairIdx " << dataHandler->hemiTree->GetEntries() + localPairIndex << endl;
  dataHandler->m_pairIdx = (dataHandler->hemiTree->GetEntries() + localPairIndex);
  

  for(const jetPtr& tagJet : tagJets){
    dataHandler->m_jet_pt        ->push_back(tagJet->pt);
    dataHandler->m_jet_eta       ->push_back(tagJet->eta);  
    dataHandler->m_jet_phi       ->push_back(tagJet->phi);  
    dataHandler->m_jet_m         ->push_back(tagJet->m);  
    dataHandler->m_jet_e         ->push_back(tagJet->e);  
    dataHandler->m_jet_bRegCorr  ->push_back(tagJet->bRegCorr);  
    dataHandler->m_jet_deepB     ->push_back(tagJet->deepB);  
    dataHandler->m_jet_CSVv2     ->push_back(tagJet->CSVv2);  
    dataHandler->m_jet_deepFlavB ->push_back(tagJet->deepFlavB);  
    dataHandler->m_jet_isTag     ->push_back(true);

  }

  for(const jetPtr& nonTagJet : nonTagJets){
    dataHandler->m_jet_pt        ->push_back(nonTagJet->pt);
    dataHandler->m_jet_eta       ->push_back(nonTagJet->eta);  
    dataHandler->m_jet_phi       ->push_back(nonTagJet->phi);  
    dataHandler->m_jet_m         ->push_back(nonTagJet->m);  
    dataHandler->m_jet_e         ->push_back(nonTagJet->e);  
    dataHandler->m_jet_bRegCorr  ->push_back(nonTagJet->bRegCorr);  
    dataHandler->m_jet_deepB     ->push_back(nonTagJet->deepB);  
    dataHandler->m_jet_CSVv2     ->push_back(nonTagJet->CSVv2);  
    dataHandler->m_jet_deepFlavB ->push_back(nonTagJet->deepFlavB);  
    dataHandler->m_jet_isTag     ->push_back(false);
  }


  dataHandler->hemiTree->Fill();

}
