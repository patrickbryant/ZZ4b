//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/countsVsLumiHists.h"

using namespace nTupleAnalysis;
using std::cout; using std::endl; 

countsVsLumiHists::countsVsLumiHists(std::string histName, std::string name, TFileDirectory& dir, bool _debug) {
  m_debug = _debug;

  m_hist = dir.make<TH1F>(histName.c_str(), (name+"/"+histName+"; ;Entries").c_str(),  1,1,2);
  m_hist->SetCanExtend(1);

  m_histUnit = dir.make<TH1F>((histName+"_unitWeight").c_str(), (name+"/"+histName+"_unitWeight; ;Entries").c_str(),  1,1,2);
  m_histUnit->SetCanExtend(1);
  
  m_currentLBStr = getLumiName();

  m_hist    ->GetXaxis()->FindBin(m_currentLBStr.c_str());
  m_histUnit->GetXaxis()->FindBin(m_currentLBStr.c_str());
} 

std::string countsVsLumiHists::getLumiName(){
  std::stringstream ss;
  ss << m_currentLB ;
  return "LB:"+ss.str();
}

void countsVsLumiHists::Fill(float weight){
  if(m_debug) cout << "countsVsLumiHists::Fill " << endl;

  m_hist    ->Fill(m_currentLBStr.c_str(), weight);
  m_histUnit->Fill(m_currentLBStr.c_str(), 1.0);
  return;
}

void countsVsLumiHists::FillLumiBlock(float /*lumiThisBlock*/){
  if(m_debug) cout << "countsVsLumiHists::FillLumiBlock " << endl;

  if(m_debug) cout << "currentLB " << m_currentLB << endl;

  //cout << "currentLB " << m_currentLB << endl;
  //cout << lumiThisBlock << endl; 
  //unsigned int iBin = m_hist    ->GetXaxis()->FindBin(m_currentLBStr.c_str());
  //
  // Can also scale by lumi in this block to be ultra percise
  //

  m_currentLB += 1;

  m_currentLBStr = getLumiName();
  m_hist    ->GetXaxis()->FindBin(m_currentLBStr.c_str());
  m_histUnit->GetXaxis()->FindBin(m_currentLBStr.c_str());


  return;
}


countsVsLumiHists::~countsVsLumiHists(){} 

