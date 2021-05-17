#include "ZZ4b/nTupleAnalysis/interface/tTbarEventData.h"

using namespace nTupleAnalysis;

using std::cout; using std::endl; 
using std::vector;

//// Sorting functions
//bool sortDeepB(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepB     > rhs->deepB);     } // put largest  deepB first in list
//bool sortCSVv2(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->CSVv2     > rhs->CSVv2);     } // put largest  CSVv2 first in list
//bool sortDeepFlavB(std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepFlavB > rhs->deepFlavB); } // put largest  deepB first in list

tTbarEventData::tTbarEventData(TChain* t, bool mc, std::string y, bool d, std::string bjetSF, std::string btagVariations, std::string JECSyst){
  cout << "tTbarEventData::tTbarEventData()" << endl;
  tree  = t;
  isMC  = mc;
  year  = ::atof(y.c_str());
  debug = d;

  //cout << "tTbarEventData::tTbarEventData() tree->Lookup(true)" << endl;
  //tree->Lookup(true);
  cout << "tTbarEventData::tTbarEventData() tree->LoadTree(0)" << endl;
  tree->LoadTree(0);
  inputBranch(tree, "run",             run);
  inputBranch(tree, "luminosityBlock", lumiBlock);
  inputBranch(tree, "event",           event);
  inputBranch(tree, "PV_npvs",         nPVs);
  inputBranch(tree, "PV_npvsGood",     nPVsGood);


  inputBranch(tree, "PV_npvsGood",     nPVsGood);
      

  


  if(isMC){
    inputBranch(tree, "genWeight", genWeight);
    if(tree->FindBranch("nGenPart")){
      truth = new truthData(tree, debug);
    }else{
      cout << "No GenPart (missing branch 'nGenPart'). Will ignore ..." << endl;
    }

    inputBranch(tree, "bTagSF", inputBTagSF);
  }


//  //triggers https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTPathsRunIIList
//  if(year==2016){
//    inputBranch(tree, "HLT_QuadJet45_TripleBTagCSV_p087",            HLT_4j45_3b087);//L1_QuadJetC50 L1_HTT300 L1_TripleJet_88_72_56_VBF
//    inputBranch(tree, "L1_QuadJetC50", L1_QuadJetC50);
//    inputBranch(tree, "L1_HTT300", L1_HTT300);
//
//    inputBranch(tree, "HLT_DoubleJet90_Double30_TripleBTagCSV_p087", HLT_2j90_2j30_3b087);//L1_TripleJet_88_72_56_VBF L1_HTT300 L1_SingleJet170 L1_DoubleJetC100
//    inputBranch(tree, "L1_DoubleJetC100", L1_DoubleJetC100);
//    inputBranch(tree, "L1_SingleJet170", L1_SingleJet170);
//
//
//    inputBranch(tree, "HLT_DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6", HLT_2j100_dEta1p6_2b);
//    inputBranch(tree, "L1_SingleJet200", L1_SingleJet200);
//
//
//  }
//
//  if(year==2017){
//    //https://cmswbm.cern.ch/cmsdb/servlet/TriggerMode?KEY=l1_hlt_collisions2017/v320
//    //https://cmsoms.cern.ch/cms/triggers/l1_rates?cms_run=306459
//
//    inputBranch(tree, "HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0", HLT_HT300_4j_75_60_45_40_3b);//L1_HTT280er_QuadJet_70_55_40_35_er2p5 L1_HTT300er
//    //inputBranch(tree, "L1_QuadJet60er2p7", L1_QuadJet60er2p7);//periods B,D and F have this
//    //inputBranch(tree, "L1_QuadJet60er3p0", L1_QuadJet60er3p0);//periods C and E have this
//    inputBranch(tree, "L1_HTT280er_QuadJet_70_55_40_35_er2p5", L1_HTT280er_QuadJet_70_55_40_35_er2p5);
//    inputBranch(tree, "L1_HTT380er", L1_HTT380er);
//
//    inputBranch(tree, "HLT_DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33", HLT_2j100_dEta1p6_2b);
//    inputBranch(tree, "L1_DoubleJet100er2p3_dEta_Max1p6", L1_DoubleJet100er2p3_dEta_Max1p6);
//
//
//    //inputBranch(tree, "HLT_Mu12_DoublePFJets40MaxDeta1p6_DoubleCaloBTagCSV_p33",   HLT_mu12_2j40_dEta1p6_db);//L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6
//    //inputBranch(tree, "L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6", L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6);
//    //inputBranch(tree, "HLT_Mu12_DoublePFJets350_CaloBTagCSV_p33",                  HLT_mu12_2j350_1b);//L1_Mu3_Jet120er2p7_dEta_Max0p4_dPhi_Max0p4
//    //inputBranch(tree, "L1_Mu3_Jet120er2p7_dEta_Max0p4_dPhi_Max0p4", L1_Mu3_Jet120er2p7_dEta_Max0p4_dPhi_Max0p4);//periods B,D and F have this
//    //inputBranch(tree, "L1_Mu3_JetC120_dEta_Max0p4_dPhi_Max0p4",     L1_Mu3_JetC120_dEta_Max0p4_dPhi_Max0p4);//periods C and E have this
//    //inputBranch(tree, "HLT_PFJet500",                                              HLT_j500);//L1_SingleJet170
//    //inputBranch(tree, "L1_SingleJet170", L1_SingleJet170);
//    //inputBranch(tree, "HLT_AK8PFJet400_TrimMass30",                                HLT_J400_m30);//L1_SingleJet180
//    //inputBranch(tree, "L1_SingleJet180", L1_SingleJet180);
//  }
//
//  if(year==2018){
//    inputBranch(tree, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", HLT_HT330_4j_75_60_45_40_3b);
//
//    //L1 seeds
//    inputBranch(tree, "L1_HTT360er", L1_HTT360er);
//    inputBranch(tree, "L1_ETT2000",  L1_ETT2000);
//    inputBranch(tree, "L1_HTT320er_QuadJet_70_55_40_40_er2p4",  L1_HTT320er_QuadJet_70_55_40_40_er2p4);
//
//    inputBranch(tree, "HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71",       HLT_2j116_dEta1p6_2b);
//    inputBranch(tree, "L1_DoubleJet112er2p3_dEta_Max1p6", L1_DoubleJet112er2p3_dEta_Max1p6);
//    inputBranch(tree, "L1_DoubleJet150er2p5", L1_DoubleJet150er2p5);
//
//    //inputBranch(tree, "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",    HLT_4j_103_88_75_15_2b_VBF1);
//    ////inputBranch(tree, "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",              HLT_4j_103_88_75_15_1b_VBF2);
//    ////L1 seeds
//    //inputBranch(tree, "L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5", L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5);
//    ////L1_SingleJet180
//    //
//    //
//    ////L1 seeds
//
//    //
//    //inputBranch(tree, "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02",            HLT_J330_m30_2b);
//    ////inputBranch(tree, "HLT_PFJet500",                                                  HLT_j500);
//    ////inputBranch(tree, "HLT_DiPFJetAve300_HFJEC",                                       HLT_2j300ave);
//    ////L1 seeds
//    //inputBranch(tree, "L1_SingleJet180", L1_SingleJet180);
//  }



  cout << "tTbarEventData::tTbarEventData() Initialize jets" << endl;
  treeJets  = new  jetData(    "Jet", tree, true, isMC, "", "", bjetSF, btagVariations, JECSyst);
  cout << "tTbarEventData::tTbarEventData() Initialize muons" << endl;
  treeMuons = new muonData(   "Muon", tree, true, isMC);
  cout << "tTbarEventData::tTbarEventData() Initialize elecs" << endl;
  treeElecs = new elecData(   "Electron", tree, true, isMC);
  cout << "tTbarEventData::tTbarEventData() Initialize MetData" << endl;

  treeCaloMET  = new MeTData(   "CaloMET",      tree, true, isMC);
  treeChsMET   = new MeTData(   "ChsMET",       tree, true, isMC);
  treeMET      = new MeTData(   "MET",          tree, true, isMC);
  treePuppiMET = new MeTData(   "PuppiMET",     tree, true, isMC);
  treeTrkMET   = new MeTData(   "TkMET",        tree, true, isMC);

} 

//Set bTagging and sorting function
void tTbarEventData::setTagger(std::string tagger, float tag){
  bTagger = tagger;
  bTag    = tag;
//  if(bTagger == "deepB")
//    sortTag = sortDeepB;
//  if(bTagger == "CSVv2")
//    sortTag = sortCSVv2;
//  if(bTagger == "deepFlavB" || bTagger == "deepjet")
//    sortTag = sortDeepFlavB;
}



void tTbarEventData::resetEvent(){
  if(debug) cout<<"Reset tTbarEventData"<<endl;

  //  topQuarkBJets.clear();
  //topQuarkWJets.clear();

  mcWeight = 1;
  weight = 1;
  bTagSF = 1;
  treeJets->resetSFs();
  nTrueBJets = 0;
  //  t.reset(); t0.reset(); t1.reset(); //t2.reset();
  
}

void tTbarEventData::update(long int e){
  if(debug){
    cout<<"Get Entry "<<e<<endl;
    cout<<tree->GetCurrentFile()->GetName()<<endl;
    tree->Show(e);
  }

  // if(printCurrentFile && tree->GetCurrentFile()->GetName() != currentFile){
  //   currentFile = tree->GetCurrentFile()->GetName();
  //   cout<< std::endl << "Loading: " << currentFile << endl;
  // }

  Long64_t loadStatus = tree->LoadTree(e);
  if(loadStatus<0){
   cout << "Error "<<loadStatus<<" getting event "<<e<<endl; 
   return;
  }

  tree->GetEntry(e);
  if(debug) cout<<"Got Entry "<<e<<endl;

  //
  // Reset the derived data
  //
  resetEvent();

  if(truth) truth->update();

  //
  //  TTbar Pt weighting
  //
  if(truth && doTTbarPtReweight){
    vector<particlePtr> tops = truth->truthParticles->getParticles(6,6);
    float minTopPt = 1e10;
    float minAntiTopPt = 1e10;
    for(const particlePtr& top :  tops){
      if(top->pdgId == 6 &&       top->pt < minTopPt)     minTopPt = top->pt;
      if(top->pdgId == -6 &&      top->pt < minAntiTopPt) minAntiTopPt = top->pt;
    }
    
    ttbarWeight = sqrt( ttbarSF(minTopPt) * ttbarSF(minAntiTopPt) );

    weight *= ttbarWeight;
    weightNoTrigger *= ttbarWeight;  

  }

  //Objects from ntuple
  if(debug) cout << "Get Jets\n";
  //getJets(float ptMin = -1e6, float ptMax = 1e6, float etaMax = 1e6, bool clean = false, float tagMin = -1e6, std::string tagger = "CSVv2", bool antiTag = false, int puIdMin = 0);
  allJets = treeJets->getJets(20, 1e6, 1e6, false, -1e6, bTagger, false, puIdMin);

  if(debug) cout << "Get Muons\n";
  allMuons         = treeMuons->getMuons();
  muons_isoMed25   = treeMuons->getMuons(25, 2.4, 2, true);
  muons_isoMed40   = treeMuons->getMuons(40, 2.4, 2, true);
  nIsoMuons = muons_isoMed25.size();

  allElecs         = treeElecs->getElecs();
  elecs_isoMed25   = treeElecs->getElecs(25, 2.4, true);
  elecs_isoMed40   = treeElecs->getElecs(40, 2.4, true);
  nIsoElecs = elecs_isoMed25.size();

  nIsoLeps = nIsoMuons + nIsoElecs;

  buildEvent();

  //cout << " nIsoLeps " << nIsoLeps 


  if(year==2016){

  //    HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v // in stream MuonEG
  //    HLT_IsoMu24_v // in SingleMuon
  //    HLT_IsoMu27_v // in SingleMuon
      
    passHLT = 
      (HLT_4j45_3b087       & (L1_QuadJetC50 || L1_HTT300) ) || 
      (HLT_2j90_2j30_3b087  & (L1_SingleJet170 || L1_DoubleJetC100 || L1_HTT300) ) || 
      (HLT_2j100_dEta1p6_2b & (L1_SingleJet200 || L1_DoubleJetC100));
  }

  if(year==2017){
 
  //    HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v // in MuonEG 
  //    HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v   // in MuonEG
  //    HLT_IsoMu24_v
  //    HLT_IsoMu27_v

    passHLT = 
      (HLT_HT300_4j_75_60_45_40_3b & (L1_HTT280er_QuadJet_70_55_40_35_er2p5 || L1_HTT300er)) || 
      (HLT_2j100_dEta1p6_2b & (L1_DoubleJet100er2p3_dEta_Max1p6));
  }

  if(year==2018){

    //HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_v // MuonEG
    //HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v
    //HLT_IsoMu24_v
    //HLT_IsoMu27_v

    passHLT = 
      (HLT_HT330_4j_75_60_45_40_3b & (L1_HTT360er || L1_ETT2000 || L1_HTT320er_QuadJet_70_55_40_40_er2p4)) || 
      //(HLT_4j_103_88_75_15_2b_VBF1 & (L1_SingleJet180 || L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5)) || 
      //(HLT_4j_103_88_75_15_1b_VBF2 & (L1_SingleJet180 || L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5)) || 
      (HLT_2j116_dEta1p6_2b        & (L1_DoubleJet112er2p3_dEta_Max1p6 || L1_DoubleJet150er2p5)) ;
    //(HLT_J330_m30_2b             & (L1_SingleJet180));// || 
    //(HLT_j500                    & (L1_SingleJet180)) || 
    //(HLT_2j300ave                & (L1_SingleJet180));
  }


  if(debug) cout<<"tTbarEventData updated\n";
  return;
}

void tTbarEventData::buildEvent(){

  //
  // Select Jets
  //
  selJets       = treeJets->getJets(     allJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning);
  looseTagJets  = treeJets->getJets(     selJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning, bTag/2, bTagger);
  tagJets       = treeJets->getJets(looseTagJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning, bTag,   bTagger);
  antiTag       = treeJets->getJets(     selJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning, bTag/2, bTagger, true); //boolean specifies antiTag=true, inverts tagging criteria
  nSelJets      =      selJets.size();
  nLooseTagJets = looseTagJets.size();
  nTagJets      =      tagJets.size();
  nAntiTag      =      antiTag.size();

  //btag SF
  if(isMC){
    //for(auto &jet: selJets) bTagSF *= treeJets->getSF(jet->eta, jet->pt, jet->deepFlavB, jet->hadronFlavour);
    for(auto &jet: selJets) treeJets->updateSFs(jet->eta, jet->pt, jet->deepFlavB, jet->hadronFlavour);
    bTagSF = treeJets->m_btagSFs["central"];
    weight *= bTagSF;
    weightNoTrigger *= bTagSF;
    for(auto &jet: allJets) nTrueBJets += jet->hadronFlavour == 5 ? 1 : 0;
  }
  
  st = 0;
  for(const auto &jet: allJets) st += jet->pt;

  twoTag = (nTagJets >= 2);

  //hack to get bTagSF normalization factor
  //fourTag = (nSelJets >= 4); threeTag = false;
  if(twoTag){
    // if event passes basic cuts start doing higher level constructions
    //buildTops();
  }

  ht = 0;
  ht30 = 0;
  for(const jetPtr& jet: allJets){

    if(fabs(jet->eta) < 2.5){
      ht += jet->pt_wo_bRegCorr;
      if(jet->pt_wo_bRegCorr > 30){
	ht30 += jet->pt_wo_bRegCorr;
      }
    }
  }

  if(debug) cout<<"tTbarEventData buildEvent\n";
  return;
}



//
//void tTbarEventData::buildTops(){
//  //All quadjet events will have well defined xWt0, a top candidate where all three jets are allowed to be candidate jets.
//  for(auto &b: topQuarkBJets){
//    for(auto &j: topQuarkWJets){
//      if(b.get()==j.get()) continue; //require they are different jets
//      if(b->deepFlavB < j->deepFlavB) continue; //don't consider W pairs where j is more b-like than b.
//      for(auto &l: topQuarkWJets){
//	if(b.get()==l.get()) continue; //require they are different jets
//	if(j.get()==l.get()) continue; //require they are different jets
//  	if(j->deepFlavB < l->deepFlavB) continue; //don't consider W pairs where l is more b-like than j.
//  	trijet* thisTop = new trijet(b,j,l);
//  	if(thisTop->xWbW < xWbW0){
//  	  xWt0 = thisTop->xWt;
//	  xWbW0= thisTop->xWbW;
//	  dRbW = thisTop->dRbW;
//	  t0.reset(thisTop);
//  	  xWt = xWt0; // define global xWt in this case
//	  xWbW= xWbW0;
//	  xW = thisTop->W->xW;
//	  xt = thisTop->xt;
//	  xbW = thisTop->xbW;
//	  t = t0;
//  	}else{delete thisTop;}
//      }
//    }
//  }
//  if(nSelJets<5) return; 
//
//  // for events with additional jets passing preselection criteria, make top candidates requiring at least one of the jets to be not a candidate jet. 
//  // This is a way to use b-tagging information without creating a bias in performance between the three and four tag data.
//  // This should be a higher quality top candidate because W bosons decays cannot produce b-quarks. 
//  for(auto &b: topQuarkBJets){
//    for(auto &j: topQuarkWJets){
//      if(b.get()==j.get()) continue; //require they are different jets
//      if(b->deepFlavB < j->deepFlavB) continue; //don't consider W pairs where j is more b-like than b.
//      for(auto &l: othJets){
//	if(b.get()==l.get()) continue; //require they are different jets
//	if(j.get()==l.get()) continue; //require they are different jets
//  	if(j->deepFlavB < l->deepFlavB) continue; //don't consider W pairs where l is more b-like than j.
//  	trijet* thisTop = new trijet(b,j,l);
//  	if(thisTop->xWbW < xWbW1){
//  	  xWt1 = thisTop->xWt;
//  	  xWbW1= thisTop->xWbW;
//	  dRbW = thisTop->dRbW;
//  	  t1.reset(thisTop);
//  	  xWt = xWt1; // overwrite global best top candidate
//  	  xWbW= xWbW1; // overwrite global best top candidate
//	  xW = thisTop->W->xW;
//	  xt = thisTop->xt;
//	  xbW = thisTop->xbW;
//  	  t = t1;
//  	}else{delete thisTop;}
//      }
//    }
//  }
//  // if(nSelJets<7) return;//need several extra jets for this to gt a good m_{b,W} peak at the top mass
//
//  // //try building top candidates where at least 2 jets are not candidate jets. This is ideal because it most naturally represents the typical hadronic top decay with one b-jet and two light jets
//  // for(auto &b: canJets){
//  //   for(auto &j: othJets){
//  //     for(auto &l: othJets){
//  // 	if(j->deepFlavB < l->deepFlavB) continue; //only consider W pairs where j is more b-like than l.
//  // 	if(j->p.DeltaR(l->p)<0.1) continue;
//  // 	trijet* thisTop = new trijet(b,j,l);
//  // 	if(thisTop->xWt < xWt2){
//  // 	  xWt2 = thisTop->xWt;
//  // 	  t2.reset(thisTop);
//  // 	  xWt = xWt2; // overwrite global best top candidate
//	  // xW = thisTop->W->xW;
//	  // xt = thisTop->xt;
//  // 	  t = t2;
//  // 	}else{delete thisTop;}
//  //     }
//  //   }
//  // }  
//
//  return;
//}

void tTbarEventData::dump(){

  cout << "   Run: " << run    << endl;
  cout << " Event: " << event  << endl;  
  cout << "Weight: " << weight << endl;
  cout << "Trigger Weight : " << trigWeight << endl;
  cout << "WeightNoTrig: " << weightNoTrigger << endl;
  cout << " allJets: " << allJets .size() << " |  selJets: " << selJets .size() << " | tagJets: " << tagJets.size() << endl;
  cout << "allMuons: " << allMuons.size() << " | isoMuons: " << muons_isoMed25.size() << endl;
  cout << "allElecs: " << allElecs.size() << " | isoElecs: " << muons_isoMed25.size() << endl;
  cout << "     MeT: " <<  treeCaloMET->pt << " " <<  treeChsMET   ->pt << " " << treeMET      ->pt << " " 
       << treePuppiMET ->pt << " " 
       << treeTrkMET   ->pt << endl;


  cout << "All Jets" << endl;
  for(auto& jet : allJets){
    cout << "\t " << jet->pt << " " << jet->eta << " " << jet->phi << " " << jet->deepB  << " " << jet->deepFlavB << " " << (jet->pt - 40) << endl;
  }

  cout << "Sel Jets" << endl;
  for(auto& jet : selJets){
    cout << "\t " << jet->pt << " " << jet->eta << " " << jet->phi << " " << jet->deepB  << " " << jet->deepFlavB << endl;
  }

  cout << "Tag Jets" << endl;
  for(auto& jet : tagJets){
    cout << "\t " << jet->pt << " " << jet->eta << " " << jet->phi << " " << jet->deepB  << " " << jet->deepFlavB << endl;
  }


  return;
}

tTbarEventData::~tTbarEventData(){} 


// https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting
//  SF  data/POWHEG+Pythia8
float tTbarEventData::ttbarSF(float pt){

  float inputPt = pt;
  if(pt > 500) inputPt = 500;
  
  return exp(0.0615 - 0.0005*inputPt);
}
