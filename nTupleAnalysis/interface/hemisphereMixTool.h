// -*- C++ -*-
#if !defined(hemisphereMixTool_H)
#define hemisphereMixTool_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include <TFile.h>
#include "TVector2.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

namespace nTupleAnalysis {

  class hemisphereMixTool;

  struct hemisphere {

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

  hemisphere(UInt_t fRun, ULong64_t fEvent, TVector2 tAxis) : Run(fRun), Event(fEvent), thrustAxis(tAxis) {
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

    void write(hemisphereMixTool* hMixTool);

    
  };


  class hemisphereMixTool {
  public:
    //TFileDirectory dir;
    bool m_debug;
    bool createLibrary;
    //TH1F* unitWeight;

    hemisphereMixTool(std::string, fwlite::TFileService&, bool, bool);

    TVector2 getThrustAxis(eventData* event);

    void addEvent(eventData*);
    ~hemisphereMixTool(); 
    void storeLibrary();
    TTree* hemiTree;

  private:
    TVector2 calcThrust(const std::vector<TVector2>& jetPts);
    void calcT(const std::vector<TVector2>& momenta, double& t, TVector2& taxis);
    
    TFile* hemiFile;



  public:
    std::vector<float>* m_jet_pt;
    std::vector<float>* m_jet_eta;
    std::vector<float>* m_jet_phi;
    std::vector<float>* m_jet_m;
    std::vector<float>* m_jet_e;
    std::vector<float>* m_jet_bRegCorr;
    std::vector<float>* m_jet_deepB;
    std::vector<float>* m_jet_CSVv2;
    std::vector<float>* m_jet_deepFlavB;
    std::vector<Bool_t>* m_jet_isTag;
    

    template <typename T_BR> void connectBranch(TTree *tree, const std::string& branch, std::vector<T_BR> **variable)
      {
	
	if(createLibrary){
	  //template<typename T>
	  //void HelpTreeBase::setBranch(std::string prefix, std::string varName, std::vector<T>* localVectorPtr){
	  tree->Branch((branch).c_str(),        *variable);
	}else{
	  tree->SetBranchStatus  ((branch).c_str()  , 1);
	  tree->SetBranchAddress ((branch).c_str()  , variable);
	}
      }
    

  };

}
#endif // hemisphereMixTool_H
