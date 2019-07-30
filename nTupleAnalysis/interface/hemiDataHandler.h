// -*- C++ -*-
#if !defined(hemiDataHandler_H)
#define hemiDataHandler_H

#include <string>
#include <TFile.h>
#include <TTree.h>
#include "ZZ4b/nTupleAnalysis/interface/hemisphere.h"
#include "ZZ4b/nTupleAnalysis/interface/kdTree.h"


namespace nTupleAnalysis {

  // one of these per event Index
  class hemiDataHandler {

  public:
    bool debug = false;

    TTree* hemiTree;
    TFile* hemiFile;

    UInt_t m_nJetBin;
    UInt_t m_nBJetBin;
    bool m_createLibrary;

    std::string m_EventIDPostFix;

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

    // Variances and totals
    typedef kdTree::Point<4> hemiPoint;
    typedef std::vector<hemiPoint> hemiPointList;
    typedef kdTree::kdTree<4> hemiKDtree;
    hemiKDtree*   m_kdTree;
    hemiPointList m_hemiPoints;

    hemiPoint    m_varV;
    hemiPoint    m_sumV;
    hemiPoint    m_sumV2;
    unsigned int m_nTot;

    hemiDataHandler(UInt_t nJetBin, UInt_t nBJetBin, bool createLibrary, std::string fileName, std::string name );

    hemisphere getHemi(unsigned int entry);
    hemisphere getHemiNearNeig(const hemisphere& hIn, unsigned int entry);

    void initBranches();

    void buildData();
    void calcVariance();

    void clearBranches();
    

  }; // hemiDataHandler





  template <typename T_BR> void connectVecBranch(bool createLibrary, TTree *tree, const std::string& branchName, std::vector<T_BR> **variable)
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
    

  template <typename T_BR> void connectBranch(bool createLibrary, TTree *tree, const std::string& branchName, T_BR *variable, std::string type)
  {
    if(createLibrary){
      tree->Branch((branchName).c_str(),          variable,      (branchName+"/"+type).c_str());
    }else{
      tree->SetBranchStatus  (branchName.c_str()  , 1);
      tree->SetBranchAddress (branchName.c_str()  , variable);
    }
  }



}
#endif // hemiDataHandler_H
