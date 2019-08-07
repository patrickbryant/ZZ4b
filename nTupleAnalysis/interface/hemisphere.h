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

    // These combined make sel jets
    std::vector<jetPtr> nonTagJets;    
    std::vector<jetPtr> tagJets;

    // This with the above make alljets
    std::vector<jetPtr> nonSelJets;


    float sumPz    = 0;
    float sumPt_T  = 0;
    float sumPt_Ta = 0;
    TLorentzVector combinedVec;
    float combinedMass = 0;
    UInt_t pairIdx;
    UInt_t NJets;
    UInt_t NBJets;
    UInt_t NNonSelJets;

    hemisphere(UInt_t fRun, ULong64_t fEvent, float tAxis_x, float tAxis_y) : Run(fRun), Event(fEvent), thrustAxis(TVector2(tAxis_x, tAxis_y)) {
      thrustAxisPerp = TVector2(-1*thrustAxis.X(), thrustAxis.Y());
    }

    ~hemisphere(); 

    void rotateTo(const TVector2& newTAxis, bool usePositiveHalf);

    void addJet(const jetPtr& thisJet, const std::vector<jetPtr>& selJetRef, const std::vector<jetPtr>& tagJetRef);

    void write(hemisphereMixTool* hMixTool, int localPairIndex);

    
  }; // hemisphere

  typedef std::shared_ptr<hemisphere> hemiPtr;

}
#endif // hemisphere_H
