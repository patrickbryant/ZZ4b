

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
  EventID thisEventID = { {int(nJets), int(nBJets), int(nNonSelJets)} };
  
  hemiDataHandler* dataHandler = hMixTool->getDataHandler(thisEventID);
  hemisphereData* outputData = dataHandler->m_hemiData;
  
  outputData->m_Run   = Run;
  outputData->m_Event = Event;
  outputData->m_HemiSign = HemiSign;
  outputData->m_tAxis_x = thrustAxis.X();
  outputData->m_tAxis_y = thrustAxis.Y();
  outputData->m_sumPz        = sumPz;
  outputData->m_sumPt_T 	   = sumPt_T;
  outputData->m_sumPt_Ta	   = sumPt_Ta;
  outputData->m_combinedMass = combinedMass;
  outputData->m_combinedDr = combinedDr;
  outputData->m_NJets = nJets;
  outputData->m_NBJets = tagJets.size();
  outputData->m_NNonSelJets = nonSelJets.size();
  
  if(hMixTool->m_debug) cout << " \t pairIdx " << dataHandler->hemiTree->GetEntries() + localPairIndex << endl;
  outputData->m_pairIdx = (dataHandler->hemiTree->GetEntries() + localPairIndex);
  
  std::vector<jetPtr> outputJets;
  for(const jetPtr& tagJet : tagJets){
    if(tagJet->AppliedBRegression()) tagJet->undo_bRegression();
    //if(tagJet->pt < 40) cout << "ERROR tagJet pt " << tagJet->pt << endl;
    tagJet->isTag = true;
    tagJet->isSel = true;
    outputJets.push_back(tagJet);
  }

  for(const jetPtr& nonTagJet : nonTagJets){
    if(nonTagJet->AppliedBRegression()) nonTagJet->undo_bRegression();
    //if(nonTagJet->pt < 40) cout << "ERROR nonTagJet pt " << nonTagJet->pt << endl;
    nonTagJet->isTag = false;
    nonTagJet->isSel = true;
    outputJets.push_back(nonTagJet);

  }

  for(const jetPtr& nonSelJet : nonSelJets){
    if(nonSelJet->AppliedBRegression()) nonSelJet->undo_bRegression();

    nonSelJet->isTag = false;
    nonSelJet->isSel = false;
    outputJets.push_back(nonSelJet);

  }

  outputData->m_jetData->writeJets(outputJets);
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


void hemisphere::addJet(const jetPtr& thisJet, bool isSelJet, bool isTagJet, bool useCombinedMass){

  const TLorentzVector& thisJetP = thisJet->p;
  combinedVec += thisJetP;

  
  sumPz += thisJetP.Pz();
  TVector2 thisJetPt = TVector2(thisJetP.Px(), thisJetP.Py());
            
  sumPt_T  += fabs(thisJetPt*thrustAxis);
  sumPt_Ta += fabs(thisJetPt*thrustAxisPerp);

  if(useCombinedMass){
    combinedMass = combinedVec.M();
  }else{
    for(const nTupleAnalysis::jetPtr& jet: tagJets){
      combinedDr += jet->p.DeltaR(thisJetP);
    }
    for(const nTupleAnalysis::jetPtr& jet: nonTagJets){
      combinedDr += jet->p.DeltaR(thisJetP);
    }
    
    for(const nTupleAnalysis::jetPtr& jet: nonSelJets){
      combinedDr += jet->p.DeltaR(thisJetP);
    }
  }

  if(isSelJet){

    if(isTagJet){
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



hemisphereData::hemisphereData(std::string name, TTree* hemiTree, bool readIn, bool loadJetFourVecs)
{

  connectBranch(readIn, hemiTree, "runNumber",   m_Run, "i");
  connectBranch(readIn, hemiTree, "evtNumber",   m_Event, "l");
  connectBranch(readIn, hemiTree, "hemiSign",    m_HemiSign, "O");
  connectBranch(readIn, hemiTree, "tAxis_x",     m_tAxis_x, "F");
  connectBranch(readIn, hemiTree, "tAxis_y",     m_tAxis_y, "F");
  connectBranch(readIn, hemiTree, "sumPz",       m_sumPz         , "F");
  connectBranch(readIn, hemiTree, "sumPt_T",     m_sumPt_T      , "F");
  connectBranch(readIn, hemiTree, "sumPt_Ta",    m_sumPt_Ta    , "F");
  connectBranch(readIn, hemiTree, "combinedMass",m_combinedMass  , "F");
  connectBranch(readIn, hemiTree, "combinedDr",  m_combinedDr  , "F");
  connectBranch(readIn, hemiTree, "NJets",       m_NJets, "i");  
  connectBranch(readIn, hemiTree, "NBJets",      m_NBJets, "i");  
  connectBranch(readIn, hemiTree, "NNonSelJets", m_NNonSelJets, "i");  
  connectBranch(readIn, hemiTree, "pairIdx",     m_pairIdx, "i");  


  if(loadJetFourVecs){
    m_jetData  = new jetData( "Jet" , hemiTree, readIn, false, "");
  }
}


hemiPtr hemisphereData::getHemi(bool loadJets)
{
  hemiPtr outHemi = std::make_shared<hemisphere>(hemisphere(m_Run, m_Event, m_HemiSign, m_tAxis_x, m_tAxis_y));
  outHemi->sumPz = m_sumPz;
  outHemi->sumPt_T = m_sumPt_T;
  outHemi->sumPt_Ta = m_sumPt_Ta;
  outHemi->combinedMass = m_combinedMass;
  outHemi->combinedDr = m_combinedDr;
  outHemi->NJets = m_NJets;
  outHemi->NBJets = m_NBJets;
  outHemi->NNonSelJets = m_NNonSelJets;
  outHemi->pairIdx = m_pairIdx;

  //if(m_debug) cout << "Make hemi " << loadJets << " " << m_loadJetFourVecs << endl;

  if(loadJets && m_jetData){
    //if(m_debug) cout << "load JetFourVecs " << endl;
    outHemi->tagJets.clear();
    outHemi->nonTagJets.clear();
    outHemi->nonSelJets.clear();
    
    std::vector<jetPtr> inputJets = m_jetData->getJets();
    
    unsigned int nJets = inputJets.size();
    for(unsigned int iJet = 0; iJet < nJets; ++iJet){
      const jetPtr& thisJet = inputJets.at(iJet);

      outHemi->combinedVec += thisJet->p;

      if(thisJet->isSel){
	if(thisJet->isTag){
	  outHemi->tagJets.push_back(thisJet);
	}else{
	  outHemi->nonTagJets.push_back(thisJet);
	}
      }else{
	outHemi->nonSelJets.push_back(thisJet);
      }
      
    }
  
    assert(outHemi->NBJets == outHemi->tagJets.size());
    assert(outHemi->NJets  == (outHemi->tagJets.size()+outHemi->nonTagJets.size()));
    assert(outHemi->NNonSelJets  == outHemi->nonSelJets.size());
  }

  //if(m_debug) cout << "Leave hemiDataHandler::getHemi " << endl;
  return outHemi;
}

hemisphereData::~hemisphereData(){
}
