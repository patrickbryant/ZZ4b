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

  //triggers https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTPathsRunIIList
  if(year==2016){
    inputBranch(tree, "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", HLT_Mu23_Ele12); // in stream MuonEG  Lumi 36.47  30.06  L1_Mu20_EG10
    inputBranch(tree, "HLT_IsoMu24", HLT_IsoMu24); // in SingleMuon  36.47 36.47       L1_SingleMu22
    inputBranch(tree, "HLT_IsoMu27", HLT_IsoMu27); // in SingleMuon  36.47 36.47           L1_SingleMu22 OR L1_SingleMu25

    inputBranch(tree, "L1_Mu20_EG10",  L1_Mu20_EG10); 
    inputBranch(tree, "L1_SingleMu22", L1_SingleMu22); 
    inputBranch(tree, "L1_SingleMu25", L1_SingleMu25); 
  }

  if(year==2017){
    inputBranch(tree, "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", HLT_Mu23_Ele12); // in stream MuonEG  41.54  41.54   L1_Mu23_EG10 OR  L1_Mu20_EG17
    inputBranch(tree, "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", HLT_Mu12_Ele23); // in stream MuonEG  41.54  41.54   L1_Mu5_EG23 OR L1_Mu5_LooseIsoEG20 OR L1_Mu7_EG23 OR L1_Mu7_LooseIsoEG20 OR L1_Mu7_LooseIsoEG23
    inputBranch(tree, "HLT_IsoMu24", HLT_IsoMu24); // in SingleMuon  38.06  38.06  L1_SingleMu22er2p1
    inputBranch(tree, "HLT_IsoMu27", HLT_IsoMu27); // in SingleMuon  41.54  41.54  L1_SingleMu22 OR L1_SingleMu25

    inputBranch(tree, "L1_Mu5_EG23"          ,     L1_Mu5_EG23         );
    inputBranch(tree, "L1_Mu5_LooseIsoEG20"  ,     L1_Mu5_LooseIsoEG20 );
    inputBranch(tree, "L1_Mu7_EG23"          ,     L1_Mu7_EG23         );
    inputBranch(tree, "L1_Mu7_LooseIsoEG20"  ,     L1_Mu7_LooseIsoEG20 );
    inputBranch(tree, "L1_Mu7_LooseIsoEG23"  ,     L1_Mu7_LooseIsoEG23 );
    inputBranch(tree, "L1_Mu23_EG10"         ,     L1_Mu23_EG10        );
    inputBranch(tree, "L1_Mu20_EG17"         ,     L1_Mu20_EG17        );
    inputBranch(tree, "L1_SingleMu22er2p1"   ,     L1_SingleMu22er2p1  );
    inputBranch(tree, "L1_SingleMu22"        ,     L1_SingleMu22       );      
    inputBranch(tree, "L1_SingleMu25"        ,     L1_SingleMu25       );
  }

  if(year==2018){
    inputBranch(tree, "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", HLT_Mu23_Ele12); // in stream MuonEG 59.96 59.963  L1_Mu20_EG10er2p5 OR L1_SingleMu22
    inputBranch(tree, "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ", HLT_Mu12_Ele23); // in stream MuonEG 59.96 59.96   L1_Mu5_EG23 OR L1_Mu5_LooseIsoEG20 OR L1_Mu7_EG23 OR L1_Mu7_LooseIsoEG20 OR L1_Mu7_LooseIsoEG23
    inputBranch(tree, "HLT_IsoMu24", HLT_IsoMu24); // in SingleMuon  59.96 59.95  L1_SingleMu22
    inputBranch(tree, "HLT_IsoMu27", HLT_IsoMu27); // in SingleMuon  59.96 59.95  L1_SingleMu22 OR L1_SingleMu25

    inputBranch(tree, "L1_Mu5_EG23"            ,   L1_Mu5_EG23         );
    inputBranch(tree, "L1_Mu5_LooseIsoEG20"    ,   L1_Mu5_LooseIsoEG20 );
    inputBranch(tree, "L1_Mu7_EG23"            ,   L1_Mu7_EG23         );
    inputBranch(tree, "L1_Mu7_LooseIsoEG20"    ,   L1_Mu7_LooseIsoEG20 );
    inputBranch(tree, "L1_Mu7_LooseIsoEG23"    ,   L1_Mu7_LooseIsoEG23 );
    inputBranch(tree, "L1_Mu20_EG10er2p5"      ,   L1_Mu20_EG10er2p5   );
    inputBranch(tree, "L1_SingleMu22"          ,   L1_SingleMu22       );
    inputBranch(tree, "L1_SingleMu25"          ,   L1_SingleMu25       );
  }


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
  top.reset();
  w.reset();
  
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
  muons_iso        = treeMuons->getMuons(30, 2.4, 4, true);
  muons_isoHighPt  = treeMuons->getMuons(40, 2.4, 4, true);
  nIsoMuons = muons_iso.size();

  allElecs         = treeElecs->getElecs();
  elecs_iso        = treeElecs->getElecs(30, 2.4, true);
  elecs_isoHighPt  = treeElecs->getElecs(40, 2.4, true);
  nIsoElecs = elecs_iso.size();

  nIsoLeps = nIsoMuons + nIsoElecs;

  buildEvent();

  //cout << " nIsoLeps " << nIsoLeps 


  if(year==2016){
    passHLT_2L = (HLT_Mu23_Ele12);
    passHLT_1L = HLT_IsoMu24; // || HLT_IsoMu27;
  }

  if(year==2017){
    passHLT_2L = (HLT_Mu23_Ele12 || HLT_Mu12_Ele23);
    //passHLT_1L = (HLT_IsoMu24 || HLT_IsoMu27);
    passHLT_1L = (HLT_IsoMu27);
  }

  if(year==2018){
    passHLT_2L = (HLT_Mu23_Ele12 || HLT_Mu12_Ele23);
    passHLT_1L = (HLT_IsoMu24 || HLT_IsoMu27);
  }

  passHLT = passHLT_1L || passHLT_2L;


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
  if(twoTag && nSelJets >= 4){
    // if event passes basic cuts start doing higher level constructions
    buildTops();
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




void tTbarEventData::buildTops(){
  if(debug) cout << "nAntiTag " << nAntiTag << endl;
  //
  // Find best W from non-tagged jets
  //
  float min_xW = 1e6;
  for(unsigned int iAntiTag = 0; iAntiTag < nAntiTag; ++iAntiTag){
    
    for(unsigned int jAntiTag = iAntiTag; jAntiTag < nAntiTag; ++jAntiTag){
      if(iAntiTag == jAntiTag) continue;
      
      std::shared_ptr<nTupleAnalysis::dijet> wCand = std::make_shared<dijet>(dijet(antiTag[iAntiTag], antiTag[jAntiTag]));

      if(fabs(wCand->xW) < min_xW){
	w = wCand;
	min_xW = fabs(wCand->xW);
      }

    }
  }

  if(!w){
    if(debug) cout << "No W Cands. returning w/o building top" << endl;
    return;
  }

  //
  //  Find best top from Ws and btagged jets
  //
  float min_xt = 1e6;
  for(jetPtr& bTagJet: tagJets){
    std::shared_ptr<nTupleAnalysis::trijet> topCand = std::make_shared<trijet>(trijet(bTagJet, w->lead, w->subl));
    
    if(fabs(topCand->xt) < min_xt){
      top = topCand;
      min_xt = fabs(topCand->xt);
    }

  }
  return;
}

void tTbarEventData::dump(){

  cout << "   Run: " << run    << endl;
  cout << " Event: " << event  << endl;  
  cout << "Weight: " << weight << endl;
  cout << "Trigger Weight : " << trigWeight << endl;
  cout << "WeightNoTrig: " << weightNoTrigger << endl;
  cout << " allJets: " << allJets .size() << " |  selJets: " << selJets .size() << " | tagJets: " << tagJets.size() << endl;
  cout << "allMuons: " << allMuons.size() << " | isoMuons: " << muons_iso.size() << endl;
  cout << "allElecs: " << allElecs.size() << " | isoElecs: " << muons_iso.size() << endl;
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
