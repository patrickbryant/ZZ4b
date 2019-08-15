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

  //class for tree access
  class hemisphereData {

  public:

    bool m_loadJetFourVecs;

    UInt_t    m_Run;
    ULong64_t m_Event;
    float     m_tAxis_x;
    float     m_tAxis_y;
    float     m_sumPz;
    float     m_sumPt_T;
    float     m_sumPt_Ta;
    float     m_combinedMass;
    UInt_t    m_NJets;
    UInt_t    m_NBJets;
    UInt_t    m_NNonSelJets;
    UInt_t    m_pairIdx;

    nTupleAnalysis::jetData* m_jetData = nullptr;

    hemisphereData(std::string name, TTree* hemiTree, bool readIn = true, bool loadJetFourVecs = false); 
    
    ~hemisphereData(); 

    hemiPtr getHemi(bool loadJets);

  };




}
#endif // hemisphere_H



