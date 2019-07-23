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
#include "nTupleAnalysis/baseClasses/interface/EventDisplayData.h"

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
    UInt_t regionIdx;
    UInt_t pairIdx;
    UInt_t NJets;
    UInt_t NBJets;

    hemisphere(UInt_t fRun, ULong64_t fEvent, float tAxis_x, float tAxis_y, UInt_t fregionIdx) : Run(fRun), Event(fEvent), thrustAxis(TVector2(tAxis_x, tAxis_y)) , regionIdx(fregionIdx) {
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

    
  };


  class hemisphereMixTool {
  public:
    //TFileDirectory dir;
    std::string m_name;
    bool m_debug;
    bool createLibrary;
    TVector2 m_thrustAxis;

    hemisphereMixTool(std::string, std::string, bool, fwlite::TFileService&, bool);

    TVector2 getThrustAxis(eventData* event);

    void addEvent(eventData*);
    ~hemisphereMixTool(); 
    void storeLibrary();
    TTree* hemiTree;
    void clearBranches();
    hemisphere getHemi(unsigned int entry);

    typedef std::array<int, 2> EventID;
    typedef std::vector<long int>  IndexVec;
    std::map<EventID, IndexVec> m_EventIndex;
    void makeIndexing();
    IndexVec getHemiSphereIndices(const hemisphere& hIn);

  private:
    TVector2 calcThrust(const std::vector<TVector2>& jetPts);
    void calcT(const std::vector<TVector2>& momenta, double& t, TVector2& taxis);
    void initBranches();

    
    TFile* hemiFile;

    void FillHists(const hemisphere& posH, const hemisphere& negH);
    void FillHists(const hemisphere& hIn);

    TFileDirectory dir;
    TH1F* hNJets;
    TH1F* hNBJets;
    TH1F* hPz       ;
    TH1F* hSumPt_T  ;
    TH1F* hSumPt_Ta ;
    TH1F* hCombMass ;


    TH1F* hdelta_NJets;
    TH1F* hdelta_NBJets;
    TH1F* hdelta_Pz      ;
    TH1F* hdelta_SumPt_T ;
    TH1F* hdelta_SumPt_Ta;
    TH1F* hdelta_CombMass;



    //
    // Event Displays
    //
    bool makeEventDisplays = false;
    nTupleAnalysis::EventDisplayData* eventDisplay = NULL;


  public:
    
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
    UInt_t    m_pairIdx;
    UInt_t    m_regionIdx;

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
    

    template <typename T_BR> void connectVecBranch(TTree *tree, const std::string& branchName, std::vector<T_BR> **variable)
      {
	
	if(createLibrary){
	  //template<typename T>
	  //void HelpTreeBase::setBranch(std::string prefix, std::string varName, std::vector<T>* localVectorPtr){
	  tree->Branch((branchName).c_str(),        *variable);
	}else{
	  tree->SetBranchStatus  ((branchName).c_str()  , 1);
	  tree->SetBranchAddress ((branchName).c_str()  , variable);
	}
      }
    

    template <typename T_BR> void connectBranch(TTree *tree, const std::string& branchName, T_BR *variable, std::string type)
    {
      if(createLibrary){
	tree->Branch((branchName).c_str(),          variable,      (branchName+"/"+type).c_str());
      }else{
	tree->SetBranchStatus  (branchName.c_str()  , 1);
	tree->SetBranchAddress (branchName.c_str()  , variable);
      }
    }


  };

}
#endif // hemisphereMixTool_H
