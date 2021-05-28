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
  
  //m_lumiPerLB = dir.make<TH1F>(histName.c_str(), (name+"/"+histName+"; ;Entries").c_str(),  1,1,2);
  //m_lumiPerLB->SetCanExtend(1);


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
  //if(m_debug) cout << "countsVsLumiHists::Fill " << endl;

  m_hist    ->Fill(m_currentLBStr.c_str(), weight);
  m_histUnit->Fill(m_currentLBStr.c_str(), 1.0);
  return;
}

void countsVsLumiHists::FillLumiBlock(float lumiThisBlock){
  if(m_debug) cout << "countsVsLumiHists::FillLumiBlock " << endl;

  if(m_debug) cout << "currentLB " << m_currentLB << endl;

  //cout << "currentLB " << m_currentLB << endl;
  if(m_debug) cout << lumiThisBlock << endl; 
  unsigned int iBin = m_hist    ->GetXaxis()->FindBin(m_currentLBStr.c_str());
  if(m_debug) cout << " \t bin " << iBin ;

  float counts = m_hist    ->GetBinContent(iBin);
  float error  = m_hist    ->GetBinError  (iBin);
  if(m_debug) cout << " \t counts " << counts << endl;

  float countsScaled = lumiThisBlock ? counts/lumiThisBlock : 0;
  float errorScaled  = lumiThisBlock ? error/lumiThisBlock : 0;

  if(m_debug) cout << " \t countsScaled " << countsScaled << endl;

  m_hist -> SetBinContent(iBin, countsScaled);
  m_hist -> SetBinError  (iBin, errorScaled );


  float counts_unit = m_histUnit    ->GetBinContent(iBin);
  float error_unit  = m_histUnit    ->GetBinError  (iBin);

  float countsScaled_unit = lumiThisBlock ? counts_unit/lumiThisBlock : 0;
  float errorScaled_unit  = lumiThisBlock ? error_unit/lumiThisBlock : 0;

  m_histUnit -> SetBinContent(iBin, countsScaled_unit);
  m_histUnit -> SetBinError  (iBin, errorScaled_unit );


  // Add new bin
  m_currentLB += 1;
  m_currentLBStr = getLumiName();
  m_hist    ->GetXaxis()->FindBin(m_currentLBStr.c_str());
  m_histUnit->GetXaxis()->FindBin(m_currentLBStr.c_str());


  return;
}


countsVsLumiHists::~countsVsLumiHists(){} 

