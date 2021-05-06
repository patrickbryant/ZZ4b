#include "ZZ4b/nTupleAnalysis/interface/hemiHists.h"

using namespace nTupleAnalysis;


hemiDiffHists::hemiDiffHists(std::string name, std::string diffName, TFileDirectory& thisDir, std::string postfix)
{
  m_name = name + "_" + postfix;


  hdelta_NJets    = thisDir.make<TH1F>(("hdel_"+diffName+"_NJets"   ).c_str(),  (m_name+"/del_"+diffName+"_NJets;  ;Entries"  ).c_str(),  19,-9.5,9.5);  
  hdelta_NBJets   = thisDir.make<TH1F>(("hdel_"+diffName+"_NBJets"  ).c_str(),  (m_name+"/del_"+diffName+"_NBJets; ;Entries"  ).c_str(),  19,-9.5,9.5);  
  hdelta_Pz       = thisDir.make<TH1F>(("hdel_"+diffName+"_Pz"      ).c_str(),  (m_name+"/del_"+diffName+"_Pz; ;Entries"      ).c_str(),  100,-500,500);  
  hdelta_SumPt_T  = thisDir.make<TH1F>(("hdel_"+diffName+"_SumPt_T" ).c_str(),  (m_name+"/del_"+diffName+"_SumPt_T; ;Entries" ).c_str(),     100,-300,300);  
  hdelta_SumPt_Ta = thisDir.make<TH1F>(("hdel_"+diffName+"_SumPt_Ta").c_str(),  (m_name+"/del_"+diffName+"_SumPt_Ta; ;Entries").c_str(),     100,-200,200);  
  hdelta_CombMass = thisDir.make<TH1F>(("hdel_"+diffName+"_CombMass").c_str(),  (m_name+"/del_"+diffName+"_CombMass; ;Entries").c_str(),     100,-300,300);  
  hdelta_CombDr   = thisDir.make<TH1F>(("hdel_"+diffName+"_CombDr").c_str(),    (m_name+"/del_"+diffName+"_CombDr; ;Entries").c_str(),     100,-0.1,10);  

  hdist           = thisDir.make<TH1F>(("dist_"+diffName).c_str(),     (m_name+"/dist_"+diffName+"; ;Entries").c_str(),     100,-0.1,5);  
  heventWeight    = thisDir.make<TH1F>(("eventWeight_"+diffName).c_str(),     (m_name+"/eventWeight_"+diffName+"; ;Entries").c_str(),     100,-2,2);  
}


void hemiDiffHists::Fill(const hemiPtr& hIn, const hemiPtr& hMatch, const hemiDataHandler* dataHandler)
{
  hdelta_NJets    ->Fill(hIn->NJets - hMatch->NJets);
  hdelta_NBJets   ->Fill(hIn->NBJets - hMatch->NBJets);

  float pzDiff =hIn->sumPz - hMatch->sumPz;

  hdelta_Pz       ->Fill(pzDiff);

  float sumPt_T_diff = hIn->sumPt_T - hMatch->sumPt_T;

  hdelta_SumPt_T  ->Fill(sumPt_T_diff);

  float sumPt_Ta_diff = hIn->sumPt_Ta - hMatch->sumPt_Ta;
  hdelta_SumPt_Ta ->Fill(sumPt_Ta_diff);

  float combinedMass_diff = hIn->combinedMass - hMatch->combinedMass;
  hdelta_CombMass ->Fill(combinedMass_diff);

  float combinedDr_diff = hIn->combinedDr - hMatch->combinedDr;
  hdelta_CombDr ->Fill(combinedDr_diff);

  float eventWeight_diff = hIn->eventWeight - hMatch->eventWeight;
  heventWeight ->Fill(eventWeight_diff);


  if(dataHandler){
    float pzDiff_sig        = pzDiff/dataHandler->m_varV.x[0];
    float sumPt_T_diff_sig  = sumPt_T_diff/dataHandler->m_varV.x[1];
    float sumPt_Ta_diff_sig = sumPt_Ta_diff/dataHandler->m_varV.x[2];

    float dist = -1;
    if(dataHandler->m_useCombinedMass){
      float combinedMass_diff_sig = combinedMass_diff/dataHandler->m_varV.x[3];
      dist = sqrt(pzDiff_sig*pzDiff_sig + sumPt_T_diff_sig*sumPt_T_diff_sig + sumPt_Ta_diff_sig*sumPt_Ta_diff_sig + combinedMass_diff_sig*combinedMass_diff_sig);
    }else{
      float combinedDr_diff_sig = combinedDr_diff/dataHandler->m_varV.x[3];
      dist = sqrt(pzDiff_sig*pzDiff_sig + sumPt_T_diff_sig*sumPt_T_diff_sig + sumPt_Ta_diff_sig*sumPt_Ta_diff_sig + combinedDr_diff_sig*combinedDr_diff_sig);
    }

    hdist->Fill(dist);
  }
}



hemiHists::hemiHists(std::string name, TFileDirectory& thisDir, std::string postFix, bool makeTopN, bool makeRand)
{
  m_name = name + "_" + postFix;
  
  
  hPz       = thisDir.make<TH1F>("Pz",           (m_name+"/Pz; ;Entries").c_str(),     100,-1000,1000);  
  hSumPt_T  = thisDir.make<TH1F>("SumPt_T",       (m_name+"/SumPt_T; ;Entries").c_str(),     100,0,1000);  
  hSumPt_Ta = thisDir.make<TH1F>("SumPt_Ta",     (m_name+"/SumPt_Ta; ;Entries").c_str(),     100,0,500);  
  hCombMass = thisDir.make<TH1F>("CombMass",     (m_name+"/CombMass; ;Entries").c_str(),     100,0,500);  
  hCombDr   = thisDir.make<TH1F>("CombDr",       (m_name+"/CombDr; ;Entries").c_str(),       100,-0.1,100);  

  hPz_sig       = thisDir.make<TH1F>("Pz_sig",     (m_name+"/Pz_sig; ;Entries").c_str(),     100,-10,10);  
  hSumPt_T_sig  = thisDir.make<TH1F>("SumPt_T_sig",     (m_name+"/SumPt_T_sig; ;Entries").c_str(),     100,-0.1,10);  
  hSumPt_Ta_sig = thisDir.make<TH1F>("SumPt_Ta_sig",     (m_name+"/SumPt_Ta_sig; ;Entries").c_str(),     100,-0.1,10);  
  hCombMass_sig = thisDir.make<TH1F>("CombMass_sig",     (m_name+"/CombMass_sig; ;Entries").c_str(),     100,-0.1,10);  
  hCombDr_sig = thisDir.make<TH1F>("CombDr_sig",     (m_name+"/CombDr_sig; ;Entries").c_str(),     100,-0.1,10);  
  heventWeight  = thisDir.make<TH1F>("eventWeight",     (m_name+"/eventWeight; ;Entries").c_str(),     100,-0.1,1.1);  
      
  hDiffNN   = new hemiDiffHists(name, "NN",   thisDir, postFix);
  if(makeTopN)
    hDiffTopN = new hemiDiffHists(name, "TopN", thisDir, postFix);
  if(makeRand)
    hDiffRand = new hemiDiffHists(name, "Rand", thisDir, postFix);
  // diff
}

    
void hemiHists::Fill(const hemiPtr& hIn, const hemiDataHandler* dataHandler){
  hPz       ->Fill( hIn->sumPz);
  hSumPt_T  ->Fill( hIn->sumPt_T);
  hSumPt_Ta ->Fill( hIn->sumPt_Ta);
  hCombMass ->Fill( hIn->combinedMass);
  hCombDr   ->Fill( hIn->combinedDr);
  heventWeight ->Fill( hIn->eventWeight);

  if(dataHandler){
    hPz_sig       ->Fill( hIn->sumPz       / dataHandler->m_varV.x[0]);
    hSumPt_T_sig  ->Fill( hIn->sumPt_T     / dataHandler->m_varV.x[1]);
    hSumPt_Ta_sig ->Fill( hIn->sumPt_Ta    / dataHandler->m_varV.x[2]);
    if(dataHandler->m_useCombinedMass){
      hCombMass_sig ->Fill( hIn->combinedMass/ dataHandler->m_varV.x[3]);
    }else{
      hCombDr_sig ->Fill( hIn->combinedDr/ dataHandler->m_varV.x[3]);
    }
  }
	
  return;
}


hemiHists::~hemiHists(){} 
