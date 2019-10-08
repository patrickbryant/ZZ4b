#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

using namespace nTupleAnalysis;

using std::cout; using std::endl; 
using std::vector;

// Sorting functions
bool sortPt(       std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->pt        > rhs->pt   );     } // put largest  pt    first in list
bool sortdR(       std::shared_ptr<dijet>     &lhs, std::shared_ptr<dijet>     &rhs){ return (lhs->dR        < rhs->dR   );     } // 
bool sortDBB(      std::unique_ptr<eventView> &lhs, std::unique_ptr<eventView> &rhs){ return (lhs->dBB       < rhs->dBB  );     } // put smallest dBB   first in list
bool sortDeepB(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepB     > rhs->deepB);     } // put largest  deepB first in list
bool sortCSVv2(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->CSVv2     > rhs->CSVv2);     } // put largest  CSVv2 first in list
bool sortDeepFlavB(std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepFlavB > rhs->deepFlavB); } // put largest  deepB first in list

eventData::eventData(TChain* t, bool mc, std::string y, bool d, bool _fastSkim, bool _doTrigEmulation){
  std::cout << "eventData::eventData()" << std::endl;
  tree  = t;
  isMC  = mc;
  year  = y;
  debug = d;
  fastSkim = _fastSkim;
  doTrigEmulation = _doTrigEmulation;
  random = new TRandom3();

  //std::cout << "eventData::eventData() tree->Lookup(true)" << std::endl;
  //tree->Lookup(true);
  std::cout << "eventData::eventData() tree->LoadTree(0)" << std::endl;
  tree->LoadTree(0);
  inputBranch(tree, "run",             run);
  inputBranch(tree, "luminosityBlock", lumiBlock);
  inputBranch(tree, "event",           event);
  inputBranch(tree, "PV_npvs",         nPVs);
  inputBranch(tree, "PV_npvsGood",     nPVsGood);
  if(tree->FindBranch("FvT")){
    std::cout << "Tree has FvT" << std::endl;
    inputBranch(tree, "FvT", FvT);
  }
  if(tree->FindBranch("ZHvB")){
    std::cout << "Tree has ZHvB" << std::endl;
    inputBranch(tree, "ZHvB", ZHvB);
  }
  if(tree->FindBranch("ZZvB")){
    std::cout << "Tree has ZZvB" << std::endl;
    inputBranch(tree, "ZZvB", ZZvB);
  }
  if(isMC){
    inputBranch(tree, "genWeight", genWeight);
    truth = new truthData(tree, debug);
  }

  //triggers https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTPathsRunIIList

  //
  //  Trigger Emulator
  //
  if(doTrigEmulation){
    int nToys = 100;
    trigEmulator = new TriggerEmulator::TrigEmulatorTool("trigEmulator", 1, nToys);
    
    if(year=="2018"){
      trigEmulator->AddTrig("EMU_HT330_4j_3Tag", "330ZH", {"75","60","45","40"}, {1,2,3,4},"2018",3);
    }
  }else{

    //triggers
    if(year=="2016"){
      inputBranch(tree, "HLT_QuadJet45_TripleBTagCSV_p087",            HLT_4j45_3b087);
      inputBranch(tree, "HLT_DoubleJet90_Double30_TripleBTagCSV_p087", HLT_2j90_2j30_3b087);
    }
    if(year=="2017"){
      inputBranch(tree, "HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0", HLT_HT300_4j_75_60_45_40_3b);
      inputBranch(tree, "HLT_Mu12_DoublePFJets40MaxDeta1p6_DoubleCaloBTagCSV_p33",   HLT_mu12_2j40_dEta1p6_db);
      inputBranch(tree, "HLT_Mu12_DoublePFJets350_CaloBTagCSV_p33",                  HLT_mu12_2j350_1b);
      inputBranch(tree, "HLT_PFJet500",                                              HLT_j500);
      inputBranch(tree, "HLT_AK8PFJet400_TrimMass30",                                HLT_J400_m30);
    }
    if(year=="2018"){
      inputBranch(tree, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", HLT_HT330_4j_75_60_45_40_3b);
      inputBranch(tree, "L1_HTT360er",                            L1_HTT360er);
      inputBranch(tree, "L1_ETT2000",                             L1_ETT2000);
      inputBranch(tree, "L1_HTT320er_QuadJet_70_55_40_40_er2p4",  L1_HTT320er_QuadJet_70_55_40_40_er2p4);

      //
      // for HT Turn-on Study
      //
      if(doHtTurnOnStudy){
	inputBranch(tree, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40", HLT_HT330_4j_75_60_45_40);
	inputBranch(tree, "L1_HTT280er", L1_HTT280er);
      }
      
      //inputBranch(tree, "HLT_PFHT330PT30_QuadPFJet_75_60_45_40_v"
      //inputBranch(tree, "HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1",    HLT_4j_103_88_75_15_2b_VBF1);
      //inputBranch(tree, "HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2",              HLT_4j_103_88_75_15_1b_VBF2);
      //inputBranch(tree, "HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71",       HLT_2j116_dEta1p6_2b);
      //inputBranch(tree, "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02",            HLT_J330_m30_2b);
      //inputBranch(tree, "HLT_PFJet500",                                                  HLT_j500);
      //inputBranch(tree, "HLT_DiPFJetAve300_HFJEC",                                       HLT_2j300ave);
      //                            HLT_QuadPFJet103_88_75_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1_v
      //                            HLT_QuadPFJet111_90_80_15_DoublePFBTagDeepCSV_1p3_7p7_VBF1_v
      //                            HLT_QuadPFJet103_88_75_15_PFBTagDeepCSV_1p3_VBF2_v
      //                            HLT_QuadPFJet105_88_76_15_PFBTagDeepCSV_1p3_VBF2_v
      //                            HLT_QuadPFJet111_90_80_15_PFBTagDeepCSV_1p3_VBF2_v
      // HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v
      // HLT_DoublePFJets128MaxDeta1p6_DoubleCaloBTagDeepCSV_p71_v
      // HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_p02_v
    }
  }

  std::string bjetSF = "";
  if(isMC && !fastSkim && year=="2018") {
    bjetSF = "deepjet2018";
    cout << " TURNING OFF BJET SF BY HAND !!!!!" << endl;
    bjetSF = "";
  }

  std::cout << "eventData::eventData() Initialize jets" << std::endl;
  treeJets  = new  jetData(    "Jet", tree, true, isMC, "", "", bjetSF);
  std::cout << "eventData::eventData() Initialize muons" << std::endl;
  treeMuons = new muonData(   "Muon", tree, true, isMC);
  std::cout << "eventData::eventData() Initialize TrigObj" << std::endl;
  treeTrig  = new trigData("TrigObj", tree);
} 

//Set bTagging and sorting function
void eventData::setTagger(std::string tagger, float tag){
  bTagger = tagger;
  bTag    = tag;
  if(bTagger == "deepB")
    sortTag = sortDeepB;
  if(bTagger == "CSVv2")
    sortTag = sortCSVv2;
  if(bTagger == "deepFlavB")
    sortTag = sortDeepFlavB;
}



void eventData::resetEvent(){
  if(debug) std::cout<<"Reset eventData"<<std::endl;
  canJets.clear();
  othJets.clear();
  allNotCanJets.clear();
  dijets .clear();
  views  .clear();
  ZHSB = false; ZHCR = false; ZHSR = false;
  SB = false; CR = false; SR = false;
  leadStM = -99; sublStM = -99;
  passDijetMass = false;
  passMDRs = false;
  passXWt = false;
  passDEtaBB = false;
  p4j    .SetPtEtaPhiM(0,0,0,0);
  canJet1_pt = -99;
  canJet3_pt = -99;
  aveAbsEta = -99; aveAbsEtaOth = -0.1; stNotCan = 0;
  dRjjClose = -99;
  dRjjOther = -99;
  nPseudoTags = 0;
  pseudoTagWeight = 1;
  mcWeight = 1;
  mcPseudoTagWeight = 1;
  FvTWeight = 1;
  weight = 1;
  weightNoTrigger = 1;
  trigWeight = 1;
  bTagSF = 1;
  nTrueBJets = 0;
  t.reset(); t0.reset(); t1.reset(); //t2.reset();
  xWt0 = 1e6; xWt1 = 1e6; xWt = 1e6; //xWt2=1e6;
}



void eventData::update(long int e){
  //if(e>2546040) debug = true;
  if(debug){
    std::cout<<"Get Entry "<<e<<std::endl;
    std::cout<<tree->GetCurrentFile()->GetName()<<std::endl;
    tree->Show(e);
  }
  //Long64_t loadStatus = tree->LoadTree(e);
  //if(loadStatus<0){
  //  std::cout << "Error "<<loadStatus<<" getting event "<<e<<std::endl; 
  //  return;
  //}
  if(printCurrentFile && tree->GetCurrentFile()->GetName() != currentFile){
    currentFile = tree->GetCurrentFile()->GetName();
    std::cout<<"Loading: " << currentFile<<std::endl;
  }
  tree->GetEntry(e);
  if(debug) std::cout<<"Got Entry "<<e<<std::endl;

  //
  // Reset the derived data
  //
  resetEvent();

  if(isMC) truth->update();

  
  //Objects from ntuple
  if(debug) std::cout << "Get Jets\n";
  allJets = treeJets->getJets(20);

  if(debug) std::cout << "Get Muons\n";
  allMuons = treeMuons->getMuons();
  isoMuons = treeMuons->getMuons(40, 2.4, 2, true);

  buildEvent();

  //
  // Trigger 
  //
  if(doTrigEmulation){

    SetTrigEmulation(true);

    passHLT = true;

    if(year == "2018"){
      trigWeight = trigEmulator->GetWeight("EMU_HT330_4j_3Tag");
      weight *= trigWeight;
    }

  }else{
  
    //Trigger
    if(year=="2016"){
      passHLT = HLT_4j45_3b087 || HLT_2j90_2j30_3b087;
    }
    if(year=="2018"){
      passL1 = (L1_HTT360er || L1_ETT2000 || L1_HTT320er_QuadJet_70_55_40_40_er2p4);
      passHLT = (HLT_HT330_4j_75_60_45_40_3b && passL1);
      //passHLT = HLT_HT330_4j_75_60_45_40_3b || HLT_4j_103_88_75_15_2b_VBF1 || HLT_4j_103_88_75_15_1b_VBF2 || HLT_2j90_2j30_3b087 || HLT_J330_m30_2b || HLT_j500 || HLT_2j300ave;
    }
  }



  if(debug) std::cout<<"eventData updated\n";
  return;
}

void eventData::buildEvent(){

  //
  // Select Jets
  //
  selJets = treeJets->getJets(allJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning);
  tagJets = treeJets->getJets(selJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning, bTag, bTagger);
  antiTag = treeJets->getJets(selJets, jetPtMin, 1e6, jetEtaMax, doJetCleaning, bTag, bTagger, true); //boolean specifies antiTag=true, inverts tagging criteria
  nSelJets = selJets.size();
  nAntiTag = antiTag.size();

  //btag SF
  if(isMC){
    for(auto &jet: selJets) bTagSF *= treeJets->getSF(jet->eta, jet->pt, jet->deepFlavB, jet->hadronFlavour);
    weight *= bTagSF;
    weightNoTrigger *= bTagSF;
    for(auto &jet: allJets) nTrueBJets += jet->hadronFlavour == 5 ? 1 : 0;
  }
  
  //passHLTEm = false;
  //selJets = treeJets->getJets(40, 2.5);  

  st = 0;
  for(const auto &jet: allJets) st += jet->pt;

  //Hack to use leptons as bJets
  // for(auto &muon: isoMuons){
  //   selJets.push_back(new jet(muon->p, 1.0));
  //   tagJets.push_back(new jet(muon->p, 1.0));
  // }  

  nTagJets = tagJets.size();
  threeTag = (nTagJets == 3 && nSelJets >= 4);
  fourTag  = (nTagJets >= 4);
  if(threeTag || fourTag){
    // if event passes basic cuts start doing higher level constructions
    chooseCanJets(); // need to do this before computePseudoTagWeight which uses s4j
    buildViews();
    if(fastSkim) return; // early exit when running fast skim to maximize event loop rate
    buildTops();
    passXWt = (xWt > 2);
  }
  if(threeTag && useJetCombinatoricModel) computePseudoTagWeight();
  nPSTJets = nTagJets + nPseudoTags;

  //allTrigJets = treeTrig->getTrigs(0,1e6,1);
  //std::cout << "L1 Jets size:: " << allTriggerJets.size() << std::endl;

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

  if(treeTrig) {
    //allTrigJets = treeTrig->getTrigs(0,1e6,1);

    L1ht = 0;
    L1ht30 = 0;
    HLTht = 0;
    HLTht30 = 0;
    //for(auto &trigjet: allTrigJets){
    //  if(fabs(trigjet->eta) < 2.5){
    //	L1ht += trigjet->l1pt;
    //	HLTht += trigjet->pt;
    //	if(trigjet->l1pt > 30){
    //	  L1ht30 += trigjet->l1pt;
    //	}
    //	if(trigjet->pt > 30){
    //	  HLTht30 += trigjet->pt;
    //	}
    //  }
    //}
  }


  if(debug) std::cout<<"eventData buildEvent\n";
  return;
}



int eventData::makeNewEvent(std::vector<nTupleAnalysis::jetPtr> new_allJets)
{

  
  bool threeTag_old = (nTagJets == 3 && nSelJets >= 4);
  bool fourTag_old  = (nTagJets >= 4);
  int nTagJet_old = nTagJets;
  int nSelJet_old = nSelJets;
  int nAllJet_old = allJets.size();

//  std::cout << "Old Event " << std::endl;
//  std::cout << run <<  " " << event << std::endl;
//  std::cout << "Jets: " << std::endl;
//  for(const jetPtr& j: allJets){
//    std::cout << "\t " << j->pt << " / " << j->eta << " / " << j->phi << std::endl;
//  }


  allJets.clear();
  selJets.clear();
  tagJets.clear();
  antiTag.clear();
  resetEvent();

  allJets = new_allJets;

  buildEvent();



  bool threeTag_new = (nTagJets == 3 && nSelJets >= 4);
  bool fourTag_new = (nTagJets >= 4);

  if(fourTag_old != fourTag_new) {
    std::cout << "ERROR : four tag_new " << fourTag_new << " vs " << fourTag_old 
	      << " nTag_new=" << nTagJets << " vs " << nTagJet_old 
	      << " nSel_new=" <<  nSelJets << " vs " << nSelJet_old 
	      << " nAll_new=" <<  allJets.size() << " vs " << nAllJet_old << std::endl;
    return -1;
  }
  

  if(threeTag_old != threeTag_new) {
    std::cout << "ERROR : three tag_new " << threeTag_new << " vs " << threeTag_old
	      << " nTag_new=" << nTagJets << " vs " << nTagJet_old 
	      << " nSel_new=" <<  nSelJets << " vs " << nSelJet_old 
	      << " nAll_new=" <<  allJets.size() << " vs " << nAllJet_old << std::endl;
    return -1;
  }


  //std::cout << "New Event " << std::endl;
  //std::cout << run <<  " " << event << std::endl;
  //std::cout << "Jets: " << std::endl;
  //for(const jetPtr& j: allJets){
  //  std::cout << "\t " << j->pt << " / " << j->eta << " / " << j->phi << std::endl;
  //}


  return 0;
}



void eventData::chooseCanJets(){
  if(debug) std::cout<<"chooseCanJets()\n";

  //std::vector< std::shared_ptr<jet> >* preCanJets;
  //if(fourTag) preCanJets = &tagJets;
  //else        preCanJets = &selJets;

  // order by decreasing btag score
  std::sort(selJets.begin(), selJets.end(), sortTag);
  // take the four jets with highest btag score    
  for(uint i = 0; i < 4;        ++i) canJets.push_back(selJets.at(i));
  for(uint i = 4; i < nSelJets; ++i) othJets.push_back(selJets.at(i));
  nOthJets = othJets.size();
  // order by decreasing pt
  std::sort(selJets.begin(), selJets.end(), sortPt); 


  //Build collections of other jets: othJets is all selected jets not in canJets
  //uint i = 0;
  for(auto &jet: othJets){
    //othJet_pt[i] = jet->pt; othJet_eta[i] = jet->eta; othJet_phi[i] = jet->phi; othJet_m[i] = jet->m; i+=1;
    aveAbsEtaOth += fabs(jet->eta)/nOthJets;
  }

  //allNotCanJets is all jets pt>20 not in canJets and not pileup vetoed 
  uint i = 0;
  for(auto &jet: allJets){
    if(fabs(jet->eta)>2.4 && jet->pt < 40) continue; //only keep forward jets above some threshold to reduce pileup contribution
    bool matched = false;
    for(auto &can: canJets){
      if(jet->p.DeltaR(can->p)<0.1){ matched = true; continue; }
    }
    if(matched) continue;
    allNotCanJets.push_back(jet);
    notCanJet_pt[i] = jet->pt; notCanJet_eta[i] = jet->eta; notCanJet_phi[i] = jet->phi; notCanJet_m[i] = jet->m; i+=1;
    stNotCan += jet->pt;
  }
  nAllNotCanJets = allNotCanJets.size();

  //apply bjet pt regression to candidate jets
  for(auto &jet: canJets) jet->bRegression();

  std::sort(canJets.begin(), canJets.end(), sortPt); // order by decreasing pt
  std::sort(othJets.begin(), othJets.end(), sortPt); // order by decreasing pt
  p4j = canJets[0]->p + canJets[1]->p + canJets[2]->p + canJets[3]->p;
  m4j = p4j.M();
  m123 = (canJets[1]->p + canJets[2]->p + canJets[3]->p).M();
  m023 = (canJets[0]->p + canJets[2]->p + canJets[3]->p).M();
  m013 = (canJets[0]->p + canJets[1]->p + canJets[3]->p).M();
  m012 = (canJets[0]->p + canJets[1]->p + canJets[2]->p).M();
  s4j = canJets[0]->pt + canJets[1]->pt + canJets[2]->pt + canJets[3]->pt;

  //flat nTuple variables for neural network inputs
  aveAbsEta = (fabs(canJets[0]->eta) + fabs(canJets[1]->eta) + fabs(canJets[2]->eta) + fabs(canJets[3]->eta))/4;
  canJet0_pt  = canJets[0]->pt ; canJet1_pt  = canJets[1]->pt ; canJet2_pt  = canJets[2]->pt ; canJet3_pt  = canJets[3]->pt ;
  canJet0_eta = canJets[0]->eta; canJet1_eta = canJets[1]->eta; canJet2_eta = canJets[2]->eta; canJet3_eta = canJets[3]->eta;
  canJet0_phi = canJets[0]->phi; canJet1_phi = canJets[1]->phi; canJet2_phi = canJets[2]->phi; canJet3_phi = canJets[3]->phi;
  canJet0_m   = canJets[0]->m  ; canJet1_m   = canJets[1]->m  ; canJet2_m   = canJets[2]->m  ; canJet3_m   = canJets[3]->m  ;
  //canJet0_e   = canJets[0]->e  ; canJet1_e   = canJets[1]->e  ; canJet2_e   = canJets[2]->e  ; canJet3_e   = canJets[3]->e  ;

  return;
}


void eventData::computePseudoTagWeight(){
  if(nAntiTag != (nSelJets-nTagJets)) std::cout << "eventData::computePseudoTagWeight WARNING nAntiTag = " << nAntiTag << " != " << (nSelJets-nTagJets) << " = (nSelJets-nTagJets)" << std::endl;

  float p; float e; float d;
  // if(s4j < 320){
  //   p = pseudoTagProb_lowSt;
  //   e = pairEnhancement_lowSt;
  //   d = pairEnhancementDecay_lowSt;
  // }else if(s4j < 450){
  //   p = pseudoTagProb_midSt;
  //   e = pairEnhancement_midSt;
  //   d = pairEnhancementDecay_midSt;
  // }else{
  //   p = pseudoTagProb_highSt;
  //   e = pairEnhancement_highSt;
  //   d = pairEnhancementDecay_highSt;
  // }

  p = pseudoTagProb;
  e = pairEnhancement;
  d = pairEnhancementDecay;

  //First compute the probability to have n pseudoTags where n \in {0, ..., nAntiTag Jets}
  //float nPseudoTagProb[nAntiTag+1];
  nPseudoTagProb.clear();
  float nPseudoTagProbSum = 0;
  for(uint i=0; i<=nAntiTag; i++){
    float Cnk = boost::math::binomial_coefficient<float>(nAntiTag, i);
    nPseudoTagProb.push_back( Cnk * pow(p, i) * pow((1-p), (nAntiTag - i)) ); //i pseudo tags and nAntiTag-i pseudo antiTags
    if((i%2)==1) nPseudoTagProb[i] *= 1 + e/pow(nAntiTag, d);//this helps fit but makes sum of prob != 1
    nPseudoTagProbSum += nPseudoTagProb[i];
  }

  //if( fabs(nPseudoTagProbSum - 1.0) > 0.00001) std::cout << "Error: nPseudoTagProbSum - 1 = " << nPseudoTagProbSum - 1.0 << std::endl;

  pseudoTagWeight = nPseudoTagProbSum - nPseudoTagProb[0];

  if(pseudoTagWeight < 1e-6) std::cout << "eventData::computePseudoTagWeight WARNING pseudoTagWeight " << pseudoTagWeight << " nAntiTag " << nAntiTag << " nPseudoTagProbSum " << nPseudoTagProbSum << std::endl;

  // it seems a three parameter njet model is needed. 
  // Possibly a trigger effect? ttbar? 
  //Actually seems to be well fit by the pairEnhancement model which posits that b-quarks should come in pairs and that the chance to have an even number of b-tags decays with the number of jets being considered for pseudo tags.
  // if(selJets.size()==4){ 
  //   pseudoTagWeight *= fourJetScale;
  // }else{
  //   pseudoTagWeight *= moreJetScale;
  // }

  // update the event weight
  if(debug) std::cout << "eventData::computePseudoTagWeight pseudoTagWeight " << pseudoTagWeight << std::endl;
  weight *= pseudoTagWeight;
  weightNoTrigger *= pseudoTagWeight;
  
  // Now pick nPseudoTags randomly by choosing a random number in the set (nPseudoTagProb[0], nPseudoTagProbSum)
  nPseudoTags = 0;
  float cummulativeProb = 0;
  random->SetSeed(event);
  float randomProb = random->Uniform(nPseudoTagProb[0], nPseudoTagProbSum);
  for(uint i=0; i<nAntiTag+1; i++){
    //keep track of the total pseudoTagProb for at least i pseudoTags
    cummulativeProb += nPseudoTagProb[i];

    //Wait until cummulativeProb >= randomProb
    if(cummulativeProb < randomProb) continue;
    //When cummulativeProb exceeds randomProb, we have found our pseudoTag selection

    //nPseudoTags+nTagJets should model the true number of b-tags in the fourTag data
    nPseudoTags = i;
    return;
  }
  
  std::cout << "Error: Did not find a valid pseudoTag assignment" << std::endl;
  return;
}


void eventData::buildViews(){
  if(debug) std::cout<<"buildViews()\n";
  //construct all dijets from the four canJets. 
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[1])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[2], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[2])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[3])));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[2])));

  //Find dijet with smallest dR and other dijet
  close = *std::min_element(dijets.begin(), dijets.end(), sortdR);
  int closeIdx = std::distance(dijets.begin(), std::find(dijets.begin(), dijets.end(), close));
  //Index of the dijet made from the other two jets is either the one before or one after because of how we constructed the dijets vector
  //if closeIdx is even, add one, if closeIdx is odd, subtract one.
  int otherIdx = (closeIdx%2)==0 ? closeIdx + 1 : closeIdx - 1; 
  other = dijets[otherIdx];

  //flat nTuple variables for neural network inputs
  dRjjClose = close->dR;
  dRjjOther = other->dR;

  views.push_back(std::make_unique<eventView>(eventView(dijets[0], dijets[1])));
  views.push_back(std::make_unique<eventView>(eventView(dijets[2], dijets[3])));
  views.push_back(std::make_unique<eventView>(eventView(dijets[4], dijets[5])));

  //Check that at least one view has two dijets above mass thresholds
  for(auto &view: views){
    passDijetMass = passDijetMass || ( (50 < view->leadM->m) && (view->leadM->m < 180) && (50 < view->sublM->m) && (view->sublM->m < 160) );
  }

  std::sort(views.begin(), views.end(), sortDBB);
  return;
}


bool failMDRs(std::unique_ptr<eventView> &view){ return !view->passMDRs; }

void eventData::applyMDRs(){
  views.erase(std::remove_if(views.begin(), views.end(), failMDRs), views.end());
  passMDRs = (views.size() > 0);
  if(passMDRs){
    ZHSB = views[0]->ZHSB; ZHCR = views[0]->ZHCR; ZHSR = views[0]->ZHSR;
    ZZSB = views[0]->ZZSB; ZZCR = views[0]->ZZCR; ZZSR = views[0]->ZZSR;
    SB = views[0]->SB; CR = views[0]->CR; SR = views[0]->SR;
    leadStM = views[0]->leadSt->m; sublStM = views[0]->sublSt->m;
    passDEtaBB = views[0]->passDEtaBB;
  }else{
    ZHSB = false; ZHCR = false; ZHSR=false;
    ZZSB = false; ZZCR = false; ZZSR=false;
    SB   = false;   CR = false;   SR=false;
    leadStM = 0;  sublStM = 0;
    passDEtaBB = false;
  }
  return;
}

void eventData::buildTops(){
  //All quadjet events will have well defined xWt0, a top candidate where all three jets are allowed to be candidate jets.
  for(auto &b: canJets){
    for(auto &j: selJets){
      if(b->deepFlavB < j->deepFlavB) continue; //only consider W pairs where b is more b-like than j
      if(b->p.DeltaR(j->p)<0.1) continue;
      for(auto &l: selJets){
  	if(j->deepFlavB < l->deepFlavB) continue; //only consider W pairs where j is more b-like than l
	if(j->p.DeltaR(l->p)<0.1) continue;
  	trijet* thisTop = new trijet(b,j,l);
  	if(thisTop->xWt < xWt0){
  	  xWt0 = thisTop->xWt;
	  t0.reset(thisTop);
  	  xWt = xWt0; // define global xWt in this case
	  t = t0;
  	}else{delete thisTop;}
      }
    }
  }
  if(nSelJets<5) return; 

  // for events with additional jets passing preselection criteria, make top candidates requiring at least one of the jets to be not a candidate jet. 
  // This is a way to use b-tagging information without creating a bias in performance between the three and four tag data.
  // This should be a higher quality top candidate because W bosons decays cannot produce b-quarks. 
  for(auto &b: canJets){
    for(auto &j: selJets){
      if(b->deepFlavB < j->deepFlavB) continue; //only consider W pairs where b is more b-like than j.
      if(b->p.DeltaR(j->p)<0.1) continue;
      for(auto &l: othJets){
  	if(j->deepFlavB < l->deepFlavB) continue; //only consider W pairs where j is more b-like than l.
	if(j->p.DeltaR(l->p)<0.1) continue;
  	trijet* thisTop = new trijet(b,j,l);
  	if(thisTop->xWt < xWt1){
  	  xWt1 = thisTop->xWt;
  	  t1.reset(thisTop);
  	  xWt = xWt1; // overwrite global best top candidate
  	  t = t1;
  	}else{delete thisTop;}
      }
    }
  }
  // if(nSelJets<7) return;//need several extra jets for this to gt a good m_{b,W} peak at the top mass

  // //try building top candidates where at least 2 jets are not candidate jets. This is ideal because it most naturally represents the typical hadronic top decay with one b-jet and two light jets
  // for(auto &b: canJets){
  //   for(auto &j: othJets){
  //     for(auto &l: othJets){
  // 	if(j->deepFlavB < l->deepFlavB) continue; //only consider W pairs where j is more b-like than l.
  // 	if(j->p.DeltaR(l->p)<0.1) continue;
  // 	trijet* thisTop = new trijet(b,j,l);
  // 	if(thisTop->xWt < xWt2){
  // 	  xWt2 = thisTop->xWt;
  // 	  t2.reset(thisTop);
  // 	  xWt = xWt2; // overwrite global best top candidate
  // 	  t = t2;
  // 	}else{delete thisTop;}
  //     }
  //   }
  // }  

  return;
}

void eventData::dump(){

  std::cout << "   Run: " << run    << std::endl;
  std::cout << " Event: " << event  << std::endl;  
  std::cout << "Weight: " << weight << std::endl;
  std::cout << "Trigger Weight : " << trigWeight << std::endl;
  std::cout << "WeightNoTrig: " << weightNoTrigger << std::endl;
  std::cout << " allJets: " << allJets .size() << " |  selJets: " << selJets .size() << " | tagJets: " << tagJets.size() << std::endl;
  std::cout << "allMuons: " << allMuons.size() << " | isoMuons: " << isoMuons.size() << std::endl;

  return;
}

eventData::~eventData(){} 


void eventData::SetTrigEmulation(bool doWeights){

  vector<float> allJet_pts;
  for(const jetPtr& aJet : allJets){
    allJet_pts.push_back(aJet->pt_wo_bRegCorr);
  }

  vector<float> tagJet_pts;
  unsigned int nTagJets = 0;
  for(const jetPtr& tJet : tagJets){
    if(nTagJets > 3) continue;
    ++nTagJets;
    tagJet_pts.push_back(tJet->pt_wo_bRegCorr);
  }

  if(doWeights){
    trigEmulator->SetWeights  (allJet_pts, tagJet_pts, ht30);
  }else{
    trigEmulator->SetDecisions(allJet_pts, tagJet_pts, ht30);
  }
  
}

bool eventData::PassTrigEmulationDecision(){
  if(year == "2018"){
    return trigEmulator->GetDecision("EMU_HT330_4j_3Tag");
  }

  return false;
}
