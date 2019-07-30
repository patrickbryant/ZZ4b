// -*- C++ -*-
#if !defined(hemisphere_H)
#define hemisphere_H

#include <TVector.h>
#include <TLorentzVector.h>
#include "nTupleAnalysis/baseClasses/interface/jetData.h"

namespace nTupleAnalysis {
  class hemisphereMixTool;
}

namespace nTupleAnalysis {

  class hemisphere {

  public:
    UInt_t Run;
    ULong64_t Event;
    TVector2 thrustAxis;
    TVector2 thrustAxisPerp;
    std::vector<jetPtr> nonTagJets;    
    std::vector<jetPtr> tagJets;
    float sumPz    = 0;
    float sumPt_T  = 0;
    float sumPt_Ta = 0;
    TLorentzVector combinedVec;
    float combinedMass = 0;
    UInt_t pairIdx;
    UInt_t NJets;
    UInt_t NBJets;

    hemisphere(UInt_t fRun, ULong64_t fEvent, float tAxis_x, float tAxis_y) : Run(fRun), Event(fEvent), thrustAxis(TVector2(tAxis_x, tAxis_y)) {
      thrustAxisPerp = TVector2(-1*thrustAxis.X(), thrustAxis.Y());
    }


    void addJet(const jetPtr& thisJet, const std::vector<jetPtr>& tagJetRef){
       
      combinedVec += thisJet->p;
      combinedMass = combinedVec.M();
      
      sumPz += thisJet->p.Pz();
      TVector2 thisJetPt = TVector2(thisJet->p.Px(), thisJet->p.Py());
            
      sumPt_T  += fabs(thisJetPt*thrustAxis);
      sumPt_Ta += fabs(thisJetPt*thrustAxisPerp);

      if(find(tagJetRef.begin(), tagJetRef.end(), thisJet) != tagJetRef.end()){
	tagJets.push_back(thisJet);
      }else{
	nonTagJets.push_back(thisJet);
      }

    }

    void write(hemisphereMixTool* hMixTool, int localPairIndex);

    
  }; // hemisphere

}
#endif // hemisphere_H
