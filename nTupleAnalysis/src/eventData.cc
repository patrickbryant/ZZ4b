#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

using namespace nTupleAnalysis;

using std::cout; using std::endl; 
using std::vector; using std::string;

using TriggerEmulator::hTTurnOn;   using TriggerEmulator::jetTurnOn; using TriggerEmulator::bTagTurnOn;

// Sorting functions
bool sortPt(       std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->pt        > rhs->pt   );     } // put largest  pt    first in list
bool sortDijetPt(  std::shared_ptr<dijet>     &lhs, std::shared_ptr<dijet>     &rhs){ return (lhs->pt        > rhs->pt   );     } // put largest  pt    first in list
bool sortdR(       std::shared_ptr<dijet>     &lhs, std::shared_ptr<dijet>     &rhs){ return (lhs->dR        < rhs->dR   );     } // 
bool sortDBB(      std::shared_ptr<eventView> &lhs, std::shared_ptr<eventView> &rhs){ return (lhs->dBB       < rhs->dBB  );     } // put smallest dBB   first in list
bool sortRandom(   std::shared_ptr<eventView> &lhs, std::shared_ptr<eventView> &rhs){ return (lhs->random    > rhs->random);    } // random sorting, largest value first in list
bool sortDeepB(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepB     > rhs->deepB);     } // put largest  deepB first in list
bool sortCSVv2(    std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->CSVv2     > rhs->CSVv2);     } // put largest  CSVv2 first in list
bool sortDeepFlavB(std::shared_ptr<jet>       &lhs, std::shared_ptr<jet>       &rhs){ return (lhs->deepFlavB > rhs->deepFlavB); } // put largest  deepB first in list

// std::max/min_element uses "The value returned indicates whether the element passed as first argument is considered less than the second."
bool comp_FvT_q_score(std::shared_ptr<eventView> &first, std::shared_ptr<eventView> &second){ return (first->FvT_q_score < second->FvT_q_score); }
bool comp_SvB_q_score(std::shared_ptr<eventView> &first, std::shared_ptr<eventView> &second){ return (first->SvB_q_score < second->SvB_q_score); }
bool comp_dR_close(   std::shared_ptr<eventView> &first, std::shared_ptr<eventView> &second){ return (first->close->dR   < second->close->dR  ); }

eventData::eventData(TChain* t, bool mc, std::string y, bool d, bool _fastSkim, bool _doTrigEmulation, bool _calcTrigWeights, bool _useMCTurnOns, bool _useUnitTurnOns, bool _isDataMCMix, bool _doReweight, std::string bjetSF, std::string btagVariations, std::string JECSyst, bool _looseSkim, bool _usePreCalcBTagSFs, std::string FvTName, std::string reweight4bName, std::string reweightDvTName, vector<string> otherWeightsNames, bool doWeightStudy, std::string bdtWeightFile, std::string bdtMethods, bool _runKlBdt){
  std::cout << "eventData::eventData()" << std::endl;
  tree  = t;
  isMC  = mc;
  year  = ::atof(y.c_str());
  debug = d;
  useMCTurnOns = _useMCTurnOns;
  useUnitTurnOns = _useUnitTurnOns;
  fastSkim = _fastSkim;
  doTrigEmulation = _doTrigEmulation;
  calcTrigWeights = _calcTrigWeights;
  runKlBdt = _runKlBdt;
  if(!tree->FindBranch("trigWeight_Data") && doTrigEmulation && !calcTrigWeights){
    cout << "WARNING:: You are trying to use trigger emulation without precomputed weights and without computing weights. Falling back to MC trigger decisions." << endl;
    assert(!tree->FindBranch("trigWeight_Data") && doTrigEmulation && !calcTrigWeights); // for now lets just throw error to prevent this from going unnoticed. Comment this line to fall back to simulated triggers
    doTrigEmulation = false;
    calcTrigWeights = false;
  }
  doReweight = _doReweight;
  isDataMCMix = _isDataMCMix;
  usePreCalcBTagSFs = _usePreCalcBTagSFs;
  looseSkim = _looseSkim;
  if (bdtWeightFile != "" && bdtMethods != "" && runKlBdt)
    bdtModel = std::make_unique<bdtInference>(bdtWeightFile, bdtMethods, debug);
  // if(looseSkim) {
  //   std::cout << "Using loose pt cut. Needed to produce picoAODs for JEC uncertainties which can change jet pt by a few percent." << std::endl;
  //   jetPtMin = 35;
  // }
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

  // Testing
  //inputBranch(tree, "SBtest",     SBTest);
  //inputBranch(tree, "CRtest",     CRTest);

  // if(doTrigEmulation){
  inputBranch(tree, "trigWeight_MC",     trigWeight_MC);
  inputBranch(tree, "trigWeight_Data",   trigWeight_Data);
  // }
  
  std::cout << "eventData::eventData() using FvT name (\"" << FvTName << "\")" << std::endl;
  std::cout << "\t doReweight = " << doReweight  << std::endl;
  classifierVariables[FvTName    ] = &FvT;
  classifierVariables[FvTName+"_pd4"] = &FvT_pd4;
  classifierVariables[FvTName+"_pd3"] = &FvT_pd3;
  classifierVariables[FvTName+"_pt4"] = &FvT_pt4;
  classifierVariables[FvTName+"_pt3"] = &FvT_pt3;
  classifierVariables[FvTName+"_pm4"] = &FvT_pm4;
  classifierVariables[FvTName+"_pm3"] = &FvT_pm3;
  classifierVariables[FvTName+"_pt" ] = &FvT_pt;
  classifierVariables[FvTName+"_std" ] = &FvT_std;
  classifierVariables[FvTName+"_q_1234"] = &FvT_q_score[0]; //&FvT_q_1234;
  classifierVariables[FvTName+"_q_1324"] = &FvT_q_score[1]; //&FvT_q_1324;
  classifierVariables[FvTName+"_q_1423"] = &FvT_q_score[2]; //&FvT_q_1423;
  classifierVariables["weight_dRjjClose"] = &weight_dRjjClose;
  check_classifierVariables[FvTName+"_event"] = &FvT_event;

  classifierVariables["SvB_ps" ] = &SvB_ps;
  classifierVariables["SvB_pzz"] = &SvB_pzz;
  classifierVariables["SvB_pzh"] = &SvB_pzh;
  classifierVariables["SvB_phh"] = &SvB_phh;
  classifierVariables["SvB_ptt"] = &SvB_ptt;
  classifierVariables["SvB_q_1234"] = &SvB_q_score[0]; //&SvB_q_1234;
  classifierVariables["SvB_q_1324"] = &SvB_q_score[1]; //&SvB_q_1324;
  classifierVariables["SvB_q_1423"] = &SvB_q_score[2]; //&SvB_q_1423;
  check_classifierVariables["SvB_event"] = &SvB_event;

  classifierVariables["SvB_MA_ps" ] = &SvB_MA_ps;
  classifierVariables["SvB_MA_pzz"] = &SvB_MA_pzz;
  classifierVariables["SvB_MA_pzh"] = &SvB_MA_pzh;
  classifierVariables["SvB_MA_phh"] = &SvB_MA_phh;
  classifierVariables["SvB_MA_ptt"] = &SvB_MA_ptt;
  classifierVariables["SvB_MA_q_1234"] = &SvB_MA_q_score[0]; //&SvB_MA_q_1234;
  classifierVariables["SvB_MA_q_1324"] = &SvB_MA_q_score[1]; //&SvB_MA_q_1324;
  classifierVariables["SvB_MA_q_1423"] = &SvB_MA_q_score[2]; //&SvB_MA_q_1423;
  check_classifierVariables["SvB_MA_event"] = &SvB_MA_event;

  classifierVariables["SvB_MA_VHH_ps"] = &SvB_MA_VHH_ps;

  classifierVariables[reweight4bName    ] = &reweight4b;
  classifierVariables[reweightDvTName   ] = &DvT;

  classifierVariables["BDT_kl"] = &BDT_kl;

  for(string weightName : otherWeightsNames){
    if(debug) cout << " initializing other weightName " << weightName << endl;
    otherWeights.push_back(Float_t(-1));
    classifierVariables[weightName] = &otherWeights.back();
  }

  //
  //  Hack for weight Study
  //
  if(doWeightStudy){
    classifierVariables["weight_FvT_3bMix4b_rWbW2_v0_e25_os012"] = new Float_t(-1);
    classifierVariables["weight_FvT_3bMix4b_rWbW2_v1_e25_os012"] = new Float_t(-1);
    classifierVariables["weight_FvT_3bMix4b_rWbW2_v9_e25_os012"] = new Float_t(-1);
    classifierVariables["weight_FvT_3bMix4b_rWbW2_v0_os012"] = new Float_t(-1);
    classifierVariables["weight_FvT_3bMix4b_rWbW2_v0_e25"] = new Float_t(-1);
  }

  
  for(auto& variable: classifierVariables){
    if(tree->FindBranch(variable.first.c_str())){
      std::cout << "Tree has " << variable.first << std::endl;
      inputBranch(tree, variable.first.c_str(), *variable.second);
    }else{
      std::cout << "Tree does not have " << variable.first << std::endl;
    }
  }

  for(auto& variable: check_classifierVariables){
    if(tree->FindBranch(variable.first.c_str())){
      std::cout << "Tree has " << variable.first << std::endl;
      inputBranch(tree, variable.first.c_str(), *variable.second);
      if(variable.first == FvTName+"_event"){
	check_FvT_event = true;
      }
      if(variable.first == "SvB_event"){
	check_SvB_event = true;
      }
      if(variable.first == "SvB_MA_event"){
	check_SvB_MA_event = true;
      }
    }else{
      std::cout << "Tree does not have " << variable.first << std::endl;
    }
  }

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
    //L1_triggers["L1_QuadJetC50"] = false;
    //L1_triggers["L1_DoubleJetC100"] = false;
    //L1_triggers["L1_SingleJet170"] = false;
    //L1_triggers["L1_HTT300"] = false;
    // L1_QuadJetC50 OR L1_QuadJetC60 OR 
    // L1_HTT280 OR L1_HTT300 OR L1_HTT320 OR 
    // L1_TripleJet_84_68_48_VBF OR L1_TripleJet_88_72_56_VBF OR L1_TripleJet_92_76_64_VBF"

    //HLT_L1_seeds["HLT_QuadJet45_TripleBTagCSV_p087"] = {{"L1_QuadJetC50", &L1_triggers["L1_QuadJetC50"]},
    //							{"L1_HTT300",     &L1_triggers["L1_HTT300"]},
    //};
    // L1_TripleJet_84_68_48_VBF OR L1_TripleJet_88_72_56_VBF OR L1_TripleJet_92_76_64_VBF OR 
    // L1_HTT280 OR L1_HTT300 OR L1_HTT320 OR 
    // L1_SingleJet170 OR L1_SingleJet180 OR L1_SingleJet200 OR 
    // L1_DoubleJetC100 OR L1_DoubleJetC112 OR L1_DoubleJetC120"


    //HLT_L1_seeds["HLT_DoubleJet90_Double30_TripleBTagCSV_p087"] = {{"L1_DoubleJetC100", &L1_triggers["L1_DoubleJetC100"]},
    //								   {"L1_SingleJet170",  &L1_triggers["L1_SingleJet170"]},
    //								   {"L1_HTT300",        &L1_triggers["L1_HTT300"]},
    //};

    HLT_triggers["HLT_QuadJet45_TripleBTagCSV_p087"] = false; 
    HLT_triggers["HLT_DoubleJet90_Double30_TripleBTagCSV_p087"] = false;
    HLT_triggers["HLT_DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6"] = false;

  }

  if(year==2017){
    //L1_triggers["L1_HTT380er"] = false;
    HLT_triggers["HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0"] = false;
    HLT_triggers["HLT_DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33"] = false;

    //HLT_triggers["HLT_DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33"] = false;
//HLT_L1_seeds["HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0"] = {//{"L1_HTT250er_QuadJet_70_55_40_35_er2p5", false}, // not in 2017C
//										 //{"L1_HTT280er_QuadJet_70_55_40_35_er2p5", false}, // not in 2017C
//                                                                             //{"L1_HTT300er_QuadJet_70_55_40_35_er2p5", false}, // only partial in 2017F
//                                                                             //{"L1_HTT320er_QuadJet_70_55_40_40_er2p4", false}, // not in 2017C
//                                                                             //{"L1_HTT320er_QuadJet_70_55_40_40_er2p5", false}, // not in 2017C
//                                                                             //{"L1_HTT320er_QuadJet_70_55_45_45_er2p5", false}, // not in 2017C
//                                                                             //{"L1_HTT340er_QuadJet_70_55_40_40_er2p5", false}, // not in 2017C
//                                                                             //{"L1_HTT340er_QuadJet_70_55_45_45_er2p5", false}, // not in 2017C
//                                                                             //{"L1_HTT300er", false}, // not in 2017C
//                                                                             //{"L1_HTT320er", false}, // not in 2017C
//										 //{"L1_HTT340er", false}, // not in 2017C
//										 {"L1_HTT380er", &L1_triggers["L1_HTT380er"]},
//										 //{"L1_QuadJet50er2p7", false}, // not in 2017C
//										 //{"L1_QuadJet60er2p7", false}, // not in 2017C
//    };

  }

  if(year==2018){
    //L1_triggers["L1_HTT320er_QuadJet_70_55_40_40_er2p4"] = false;// missing in one period!
    //L1_triggers["L1_HTT360er"] = false;
    //L1_triggers["L1_DoubleJet112er2p3_dEta_Max1p6"] = false;
    //L1_triggers["L1_DoubleJet150er2p5"] = false;

    // L1_QuadJet60er2p5 OR 
    // L1_HTT280er OR L1_HTT320er OR L1_HTT360er OR L1_HTT400er OR L1_HTT450er OR 
    // L1_HTT280er_QuadJet_70_55_40_35_er2p4 OR L1_HTT320er_QuadJet_70_55_40_40_er2p4 OR L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3 OR L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3    

    //HLT_L1_seeds["HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5"] = {//{"L1_HTT320er_QuadJet_70_55_40_40_er2p4", &L1_triggers["L1_HTT320er_QuadJet_70_55_40_40_er2p4"]},
    //                                                                                 {"L1_HTT360er",                           &L1_triggers["L1_HTT360er"]},
    //										     //{"", &L1_triggers[""]},
    //};

    // L1_DoubleJet112er2p3_dEta_Max1p6

    //HLT_L1_seeds["HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71"] = {{"L1_DoubleJet112er2p3_dEta_Max1p6", &L1_triggers["L1_DoubleJet112er2p3_dEta_Max1p6"]},
    //									       //{"", &L1_triggers[""]},
    //};

    HLT_triggers["HLT_DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71"] = false;
    HLT_triggers["HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5"] = false;


  }

  for(auto &trigger:  L1_triggers)     inputBranch(tree, trigger.first, trigger.second);
  for(auto &trigger: HLT_triggers)     inputBranch(tree, trigger.first, trigger.second);
  //for(auto &trigger:  L1_triggers_mon){
  //  if(L1_triggers.find(trigger.first)!=L1_triggers.end()) continue; // don't initialize branch again!
  //  inputBranch(tree, trigger.first, trigger.second);
  //}

  //
  //  Trigger Emulator
  //
  if(calcTrigWeights){
    int nToys = 10;
    //int nToys = 100;

    
    if(year==2018){

      cout << "Loading the 2018 Trigger emulator" << endl;
      trigEmulators.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorData", nToys, "2018", debug, false) );
      trigEmulators.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorMC", nToys,   "2018", debug,  true) );

      for(TriggerEmulator::TrigEmulatorTool* tEmulator : trigEmulators){
	tEmulator->AddTrig("EMU_4j_3b",   
			   //{hTTurnOn::L1ORAll_Ht330_4j_3b,hTTurnOn::CaloHt320,hTTurnOn::PFHt330},     
			   //{hTTurnOn::CaloHt320,hTTurnOn::PFHt330},     
			   {hTTurnOn::L1ORAll_Ht330_4j_3b,hTTurnOn::PFHt330},     
			   //{hTTurnOn::PFHt330},     
			   {jetTurnOn::PF30BTag,jetTurnOn::PF75BTag,jetTurnOn::PF60BTag,jetTurnOn::PF45BTag,jetTurnOn::PF40BTag},{4,1,2,3,4},  // Calo 30 ?
			   {bTagTurnOn::CaloDeepCSV, bTagTurnOn::PFDeepCSV},{2, 3}
			   );

	tEmulator->AddTrig("EMU_2b",    
			   {jetTurnOn::L1112BTag, jetTurnOn::PF116BTag, jetTurnOn::PF116DrBTag}, {2, 2, 1}, 
			   {bTagTurnOn::Calo100BTag},{2} 
			   );
      }


      trigEmulators3b.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorData3b", nToys, "2018", debug, false) );
      trigEmulators3b.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorMC3b",   nToys, "2018", debug,  true) );

      for(TriggerEmulator::TrigEmulatorTool* tEmulator : trigEmulators3b){
	tEmulator->AddTrig("EMU_4j_3b",   
			   //{hTTurnOn::L1ORAll_Ht330_4j_3b,hTTurnOn::CaloHt320,hTTurnOn::PFHt330},     
			   //{hTTurnOn::CaloHt320,hTTurnOn::PFHt330},     
			   {hTTurnOn::L1ORAll_Ht330_4j_3b,hTTurnOn::PFHt330},     
			   //{hTTurnOn::PFHt330},     
			   {jetTurnOn::PF30BTag,jetTurnOn::PF75BTag,jetTurnOn::PF60BTag,jetTurnOn::PF45BTag,jetTurnOn::PF40BTag},{4,1,2,3,4},  // Calo 30 ?
			   {bTagTurnOn::CaloDeepCSVloose, bTagTurnOn::PFDeepCSVloose},{2, 3}
			   );

	tEmulator->AddTrig("EMU_2b",    
			   {jetTurnOn::L1112BTag, jetTurnOn::PF116BTag, jetTurnOn::PF116DrBTag}, {2, 2, 1}, 
			   {bTagTurnOn::Calo100BTagloose},{2} 
			   );
      }


    }else if(year==2017){

      cout << "Loading the 2017 Trigger emulator" << endl;
      trigEmulators.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorData", nToys, "2017", debug, false) );
      trigEmulators.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorMC",   nToys, "2017", debug, true ) );

      for(TriggerEmulator::TrigEmulatorTool* tEmulator : trigEmulators){
	tEmulator->AddTrig("EMU_4j_3b",   
			   //{hTTurnOn::L1ORAll_Ht300_4j_3b,hTTurnOn::CaloHt300,hTTurnOn::PFHt300},     
			   {hTTurnOn::L1ORAll_Ht300_4j_3b,hTTurnOn::PFHt300},     
			   {jetTurnOn::PF30BTag,jetTurnOn::PF75BTag,jetTurnOn::PF60BTag,jetTurnOn::PF45BTag,jetTurnOn::PF40BTag},{4,1,2,3,4},
			   {bTagTurnOn::CaloCSV, bTagTurnOn::PFCSV},{2,3}
			   );

	tEmulator->AddTrig("EMU_2b",   
			   {jetTurnOn::L1100BTag, jetTurnOn::PF100BTag, jetTurnOn::PF100DrBTag}, {2, 2, 1}, 
			   {bTagTurnOn::Calo100BTag},{2} // Should multiply these together...
			   );
      }


      trigEmulators3b.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorData3b", nToys, "2017", debug, false) );
      trigEmulators3b.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorMC3b",   nToys, "2017", debug, true ) );

      for(TriggerEmulator::TrigEmulatorTool* tEmulator : trigEmulators3b){
	tEmulator->AddTrig("EMU_4j_3b",   
			   //{hTTurnOn::L1ORAll_Ht300_4j_3b,hTTurnOn::CaloHt300,hTTurnOn::PFHt300},     
			   {hTTurnOn::L1ORAll_Ht300_4j_3b,hTTurnOn::PFHt300},     
			   {jetTurnOn::PF30BTag,jetTurnOn::PF75BTag,jetTurnOn::PF60BTag,jetTurnOn::PF45BTag,jetTurnOn::PF40BTag},{4,1,2,3,4},
			   {bTagTurnOn::CaloCSVloose, bTagTurnOn::PFCSVloose},{2,3}
			   );

	tEmulator->AddTrig("EMU_2b",   
			   {jetTurnOn::L1100BTag, jetTurnOn::PF100BTag, jetTurnOn::PF100DrBTag}, {2, 2, 1}, 
			   {bTagTurnOn::Calo100BTagloose},{2} 
			   );
      }


    }else if(year==2016){
      cout << "Loading the 2016 Trigger emulator" << endl;
      trigEmulators.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorData", nToys, "2016", debug, false) );
      trigEmulators.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorMC",   nToys, "2016", debug, true ) );

      for(TriggerEmulator::TrigEmulatorTool* tEmulator : trigEmulators){
	tEmulator->AddTrig("EMU_4j_3b",      
			   {hTTurnOn::L1ORAll_4j_3b}, 
			   {jetTurnOn::PF45BTag},{4},
			   {bTagTurnOn::CaloCSV},{3});

	tEmulator->AddTrig("EMU_2b",    
			   {jetTurnOn::L1100BTag,    jetTurnOn::PF100BTag}, {2, 2}, 
			   {bTagTurnOn::Calo100BTag, bTagTurnOn::CaloCSV2b100},{2, 2});
	
	tEmulator->AddTrig("EMU_2j_2j_3b", 
			   {hTTurnOn::L1ORAll_2j_2j_3b}, 
			   //{jetTurnOn::Calo30BTag,jetTurnOn::Calo90BTag,jetTurnOn::PF30BTag,jetTurnOn::PF90BTag},{4,2,4,2},
			   {jetTurnOn::Calo90BTag,jetTurnOn::PF30BTag,jetTurnOn::PF90BTag},{2,4,2},
			   {bTagTurnOn::CaloCSV},{3});
      }


      trigEmulators3b.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorData3b", nToys, "2016", debug, false) );
      trigEmulators3b.push_back( new TriggerEmulator::TrigEmulatorTool("trigEmulatorMC3b",   nToys, "2016", debug, true ) );

      for(TriggerEmulator::TrigEmulatorTool* tEmulator : trigEmulators3b){
	tEmulator->AddTrig("EMU_4j_3b",      
			   {hTTurnOn::L1ORAll_4j_3b}, 
			   {jetTurnOn::PF45BTag},{4},
			   {bTagTurnOn::CaloCSVloose},{3});

	tEmulator->AddTrig("EMU_2b",    
			   {jetTurnOn::L1100BTag,    jetTurnOn::PF100BTag}, {2, 2}, 
			   {bTagTurnOn::Calo100BTag, bTagTurnOn::CaloCSV2b100loose},{2, 2});
	
	tEmulator->AddTrig("EMU_2j_2j_3b", 
			   {hTTurnOn::L1ORAll_2j_2j_3b}, 
			   //{jetTurnOn::Calo30BTag,jetTurnOn::Calo90BTag,jetTurnOn::PF30BTag,jetTurnOn::PF90BTag},{4,2,4,2},
			   {jetTurnOn::Calo90BTag,jetTurnOn::PF30BTag,jetTurnOn::PF90BTag},{2,4,2},
			   {bTagTurnOn::CaloCSVloose},{3});
      }



    }// 2016

  }// calcWeights


  std::cout << "eventData::eventData() Initialize jets" << std::endl;
  treeJets  = new  jetData(    "Jet", tree, true, isMC, "", "", bjetSF, btagVariations, JECSyst);
  std::cout << "eventData::eventData() Initialize muons" << std::endl;
  treeMuons = new muonData(   "Muon", tree, true, isMC);
  std::cout << "eventData::eventData() Initialize elecs" << std::endl;
  treeElecs = new elecData(   "Electron", tree, true, isMC);
  std::cout << "eventData::eventData() Initialize TrigObj" << std::endl;
  //treeTrig  = new trigData("TrigObj", tree);
} 

void eventData::loadJetCombinatoricModel(std::string jcmName){
  useLoadedJCM = true;
  inputBranch(tree, ("pseudoTagWeight_"+jcmName  ).c_str(), inputPSTagWeight);
}

//Set bTagging and sorting function
void eventData::setTagger(std::string tagger, float tag){
  bTagger = tagger;
  bTag    = tag;
  if(bTagger == "deepB")
    sortTag = sortDeepB;
  if(bTagger == "CSVv2")
    sortTag = sortCSVv2;
  if(bTagger == "deepFlavB" || bTagger == "deepjet")
    sortTag = sortDeepFlavB;
}



void eventData::resetEvent(){
  if(debug) std::cout<<"Reset eventData"<<std::endl;
  if(looseSkim){
    selJetsLoosePt.clear();
    tagJetsLoosePt.clear();
  }
  canJets.clear();
  othJets.clear();
  allNotCanJets.clear(); nAllNotCanJets = 0;
  topQuarkBJets.clear();
  topQuarkWJets.clear();
  dijets .clear();
  views  .clear();
  views_passMDRs.clear();
  view_selected.reset();
  view_dR_min.reset();
  view_max_FvT_q_score.reset();
  view_max_SvB_q_score.reset();
  canVDijets.clear();
  close.reset();
  other.reset();
  appliedMDRs = false;
  m4j = -99;
  // ZZSB = false; ZZCR = false; 
  // ZHSB = false; ZHCR = false; 
  // HHSB = false; HHCR = false; 
  ZZSR = false;
  ZHSR = false;
  HHSR = false;
  SB = false; 
  // CR = false; 
  SR = false;
  leadStM = -99; sublStM = -99;
  passDijetMass = false;
  d01TruthMatch = 0;
  d23TruthMatch = 0;
  d02TruthMatch = 0;
  d13TruthMatch = 0;
  d03TruthMatch = 0;
  d12TruthMatch = 0;
  truthMatch = false;
  selectedViewTruthMatch = false;
  passMDRs = false;
  passXWt = false;
  passL1  = false;
  passHLT = false;
  //passDEtaBB = false;
  p4j    .SetPtEtaPhiM(0,0,0,0);
  canJet1_pt = -99;
  canJet3_pt = -99;
  aveAbsEta = -99; aveAbsEtaOth = -0.1; stNotCan = 0;
  dRjjClose = -99;
  dRjjOther = -99;
  dR0123 = -99; dR0213 = -99; dR0312 = -99;
  nPseudoTags = 0;
  pseudoTagWeight = 1;
  mcWeight = 1;
  mcPseudoTagWeight = 1;
  weight = 1;
  weightNoTrigger = 1;
  trigWeight = 1;
  bTagSF = 1;
  treeJets->resetSFs();
  nTrueBJets = 0;
  t.reset(); t0.reset(); t1.reset(); //t2.reset();
  xWt0 = 1e6; xWt1 = 1e6; xWt = 1e6; //xWt2=1e6;
  xWbW0 = 1e6; xWbW1 = 1e6; xWbW = 1e6; //xWt2=1e6;  
  xW = 1e6; xt=1e6; xbW=1e6;
  dRbW = 1e6;
  passTTCR = false;
  passTTCRe = false;
  passTTCRem = false;

  if(runKlBdt) BDT_kl = -99;

  for(const std::string& jcmName : jcmNames){
    pseudoTagWeightMap[jcmName]= 1.0;
    mcPseudoTagWeightMap[jcmName] = 1.0;;
  }
  
}



void eventData::update(long int e){
  if(debug){
    std::cout<<"Get Entry "<<e<<std::endl;
    std::cout<<tree->GetCurrentFile()->GetName()<<std::endl;
    tree->Show(e);
  }

  // if(printCurrentFile && tree->GetCurrentFile()->GetName() != currentFile){
  //   currentFile = tree->GetCurrentFile()->GetName();
  //   std::cout<< std::endl << "Loading: " << currentFile << std::endl;
  // }

  Long64_t loadStatus = tree->LoadTree(e);
  if(loadStatus<0){
   std::cout << "Error "<<loadStatus<<" getting event "<<e<<std::endl; 
   return;
  }

  tree->GetEntry(e);
  if(debug) std::cout<<"Got Entry "<<e<<std::endl;

  if(check_FvT_event){
    assert( event==ULong64_t(FvT_event) );
  }
  if(check_SvB_event){
    assert( event==ULong64_t(SvB_event) );
  }
  if(check_SvB_MA_event){
    assert( event==ULong64_t(SvB_MA_event) );
  }

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
  if(debug) std::cout << "Get Jets\n";
  //getJets(float ptMin = -1e6, float ptMax = 1e6, float etaMax = 1e6, bool clean = false, float tagMin = -1e6, std::string tagger = "CSVv2", bool antiTag = false, int puIdMin = 0);
  allJets = treeJets->getJets(20, 1e6, 1e6, false, -1e6, bTagger, false, puIdMin);

  if(debug) std::cout << "Get Muons\n";
  allMuons         = treeMuons->getMuons();
  muons_isoMed25   = treeMuons->getMuons(25, 2.4, 4, true);
  muons_isoMed40   = treeMuons->getMuons(40, 2.4, 4, true);
  nIsoMuons = muons_isoMed40.size();

  allElecs         = treeElecs->getElecs();
  elecs_isoMed25   = treeElecs->getElecs(25, 2.4, true);
  elecs_isoMed40   = treeElecs->getElecs(40, 2.4, true);
  nIsoElecs = elecs_isoMed40.size();



  buildEvent();

  //
  // Trigger 
  //    (TO DO. Only do emulation in the SR)
  //
  if(isMC && (calcTrigWeights || doTrigEmulation)){

    if(calcTrigWeights){

      if(fourTag){
	trigWeight_Data   = GetTrigEmulationWeight(trigEmulators.at(0));
	trigWeight_MC     = GetTrigEmulationWeight(trigEmulators.at(1));
      }else if(threeTag){
	trigWeight_Data   = GetTrigEmulationWeight(trigEmulators3b.at(0));
	trigWeight_MC     = GetTrigEmulationWeight(trigEmulators3b.at(1));


	//
	// SF to correct the 3b btag SFs
	//
	if(year == 2018){
	  trigWeight_Data *= 0.600;
	  trigWeight_MC   *= 0.600;
	}else if(year == 2017){
	  trigWeight_Data *= 0.558;
	  trigWeight_MC   *= 0.558;
	}else if(year == 2016){
	  trigWeight_Data *= 0.857;
	  trigWeight_MC *= 0.857;
	}
      }

    }
 
    trigWeight = useMCTurnOns ? trigWeight_MC : trigWeight_Data;
    if(useUnitTurnOns) trigWeight = 1.0;

    weight *= trigWeight;

    passL1  = trigWeight>0;
    passHLT = trigWeight>0;

  }else{
    for(auto &trigger: HLT_triggers){
      ///bool pass_seed = boost::accumulate(HLT_L1_seeds[trigger.first] | boost::adaptors::map_values, false, [](bool pass, bool *seed){return pass||*seed;});//std::logical_or<bool>());
      //passL1  = passL1  || pass_seed;
      //passHLT = passHLT || (trigger.second && pass_seed);
      passHLT = passHLT || (trigger.second);
    }

  }


  //
  //  Other weigths
  //
  for(float oWeight: otherWeights){
    if(debug) std::cout << "other weight is "<<  oWeight <<std::endl;
    weight *= oWeight;
  }
  

  //
  // For signal injection study / and mixed + 4b TTbar  dataset
  //

  //
  //  Determine if the mixed event is actuall from Data or MC
  //
  if(isDataMCMix){
    if(run > 2){
      mixedEventIsData = true;
    }else{
      mixedEventIsData = false;
      passHLT = true; // emulation weights already included in the skimming 
    }

  }

  //hack to get bTagSF normalization factor
  //passHLT=true;

  if(debug) std::cout<<"eventData updated\n";
  return;
}

void eventData::buildEvent(){

  //
  // Select Jets
  //
  if(looseSkim){
    selJetsLoosePt = treeJets->getJets(       allJets, jetPtMin-5, 1e6, jetEtaMax, doJetCleaning);
    tagJetsLoosePt = treeJets->getJets(selJetsLoosePt, jetPtMin-5, 1e6, jetEtaMax, doJetCleaning, bTag,   bTagger);
  }
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
    if(usePreCalcBTagSFs){
      bTagSF = inputBTagSF;
    }else{
      //for(auto &jet: selJets) bTagSF *= treeJets->getSF(jet->eta, jet->pt, jet->deepFlavB, jet->hadronFlavour);
      treeJets->updateSFs(selJets, debug);
      bTagSF = treeJets->m_btagSFs["central"];
    }

    if(debug) std::cout << "eventData buildEvent bTagSF = " << bTagSF << std::endl;
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

  threeTag = (nLooseTagJets == 3 && nSelJets >= 4);
  fourTag  = (nTagJets >= 4);
  //hack to get bTagSF normalization factor
  //fourTag = (nSelJets >= 4); threeTag = false;
  if(threeTag || fourTag){
    // if event passes basic cuts start doing higher level constructions
    chooseCanJets(); // need to do this before computePseudoTagWeight which uses s4j
    buildViews();
    if(fastSkim) return; // early exit when running fast skim to maximize event loop rate
    buildTops();
    #if SLC6 == 0 //Defined in ZZ4b/nTupleAnalysis/BuildFile.xml 
    run_SvB_ONNX(); // will only run if a model was initialized
    #endif
    //((sqrt(pow(xbW/2.5,2)+pow((xW-0.5)/2.5,2)) > 1)&(xW<0.5)) || ((sqrt(pow(xbW/2.5,2)+pow((xW-0.5)/4.0,2)) > 1)&(xW>=0.5)); //(t->xWbW > 2); //(t->xWt > 2) & !( (t->m>173)&(t->m<207) & (t->W->m>90)&(t->W->m<105) );
    passXWt = t->rWbW > 3;
    passTTCR   = (muons_isoMed40.size()>0) && (t->rWbW < 2);
    passTTCRe  = (elecs_isoMed40.size()>0) && (t->rWbW < 2);
    passTTCRem = (elecs_isoMed25.size()>0) && (muons_isoMed25.size()>0);
  }

  //nPSTJets = nLooseTagJets + nPseudoTags;
  nPSTJets = nTagJets; // if threeTag use nLooseTagJets + nPseudoTags
  if(threeTag && useJetCombinatoricModel) computePseudoTagWeight();
  if(threeTag && useLoadedJCM)            applyInputPseudoTagWeight();

  if(threeTag){
    for(const std::string& jcmName : jcmNames){
      computePseudoTagWeight(jcmName);
      //std::cout << "JCM for " << jcmName << " is " << pseudoTagWeightMap[jcmName] << std::endl;
    }
    nPSTJets = nLooseTagJets + nPseudoTags;
  }

  //allTrigJets = treeTrig->getTrigs(0,1e6,1);
  //std::cout << "L1 Jets size:: " << allTriggerJets.size() << std::endl;

  ht = 0;
  ht30 = 0;
  ht30_noMuon = 0;
  for(const jetPtr& jet: allJets){

    if(fabs(jet->eta) < 2.5){
      ht += jet->pt_wo_bRegCorr;
      if(jet->pt_wo_bRegCorr > 30){

	ht30 += jet->pt_wo_bRegCorr;

	bool failMuonOverlap = false;
	for(const muonPtr &isoMed25: muons_isoMed25){
	  if(jet->p.DeltaR(isoMed25->p) < 0.1) {
	    failMuonOverlap = true;
	    break;
	  }
	}

	if(!failMuonOverlap) ht30_noMuon += jet->pt_wo_bRegCorr;
      }


    }
    
  }

  
  if(treeTrig) {
    allTrigJets = treeTrig->getTrigs(0,1e6,1);
    selTrigJets = treeTrig->getTrigs(allTrigJets,30,2.5);

    L1ht = 0;
    L1ht30 = 0;
    HLTht = 0;
    HLTht30 = 0;
    HLTht30Calo = 0;
    HLTht30CaloAll = 0;
    HLTht30Calo2p6 = 0;
    for(auto &trigjet: allTrigJets){
      if(fabs(trigjet->eta) < 2.5){
    	L1ht += trigjet->l1pt;
    	HLTht += trigjet->pt;

    	if(trigjet->l1pt > 30){
    	  L1ht30 += trigjet->l1pt;
    	}

    	if(trigjet->pt > 30){
    	  HLTht30 += trigjet->pt;
    	}

    	if(trigjet->l2pt > 30){
    	  HLTht30Calo += trigjet->l2pt;
    	}

      }// Eta

      if(trigjet->l2pt > 30){
	HLTht30CaloAll += trigjet->l2pt;
	if(fabs(trigjet->eta) < 2.6){
	  HLTht30Calo2p6 += trigjet->l2pt;
	}
      }

    }
  }

  //
  //  Apply reweight to three tag data
  //
  if((doReweight && threeTag)){
    if(debug) cout << "applyReweight: event->FvT = " << FvT << endl;
    //event->FvTWeight = spline->Eval(event->FvT);
    //event->FvTWeight = event->FvT / (1-event->FvT);
    //event->weight  *= event->FvTWeight;
    reweight = FvT;
    //if     (event->reweight > 10) event->reweight = 10;
    //else if(event->reweight <  0) event->reweight =  0;
    weight *= reweight;
    weightNoTrigger *= reweight;
  }

  //
  //  Appply 4b reweight
  //
  if(fourTag){
    weight *= reweight4b;
    weightNoTrigger *= reweight4b;    
  }

  //
  //  Apply DvT Reweight
  //
  if(doDvTReweight){
    float reweightDvT =  DvT > 0 ? DvT : 0;
    //cout << "weight was " << weight; 
    weight *= reweightDvT;
    weightNoTrigger *= reweightDvT;  
    //cout << " weight now " << weight <<endl;
  }


  if(debug) std::cout<<"eventData buildEvent done\n";
  return;
}



int eventData::makeNewEvent(std::vector<nTupleAnalysis::jetPtr> new_allJets)
{
  if(debug) cout << "eventData::makeNewEvent eventWeight " << weight << endl;
  
  bool threeTag_old = (nLooseTagJets == 3 && nSelJets >= 4);
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
  looseTagJets.clear();
  tagJets.clear();
  antiTag.clear();
  resetEvent();
  if(debug) cout << "eventData::makeNewEvent  eventWeight after reset " << weight << endl;

  allJets = new_allJets;

  //
  // Undo any bjet regression that may have been done.
  //
  for(const jetPtr& jet: allJets){
    if(jet->AppliedBRegression()) {
      jet->undo_bRegression();
    }
  }

  buildEvent();

  for(auto &trigger: HLT_triggers){
    //bool pass_seed = boost::accumulate(HLT_L1_seeds[trigger.first] | boost::adaptors::map_values, false, [](bool pass, bool *seed){return pass||*seed;});//std::logical_or<bool>());
    //passL1  = passL1  || pass_seed;
    //passHLT = passHLT || (trigger.second && pass_seed);
    passHLT = passHLT || (trigger.second);

  }


  bool threeTag_new = (nLooseTagJets == 3 && nSelJets >= 4);
  bool fourTag_new = (nTagJets >= 4);

  bool diffTagJets = ((nTagJets - nTagJet_old) != 0);
  bool diffSelJets = ((nSelJets - nSelJet_old) != 0);
  bool diffAllJets = ((allJets.size() - nAllJet_old) != 0);


  if(diffTagJets || diffSelJets || diffAllJets){
    std::cout << "event is " << event << std::endl;
    std::cout << "ERROR : three tag_new " << threeTag_new << " vs " << threeTag_old
	      << " nTag_new=" << nTagJets << " vs " << nTagJet_old 
	      << " nSel_new=" <<  nSelJets << " vs " << nSelJet_old 
	      << " nAll_new=" <<  allJets.size() << " vs " << nAllJet_old << std::endl;
    return -1;
  }

  if(fourTag_old != fourTag_new) {
    std::cout << "ERROR : four tag_new " << fourTag_new << " vs " << fourTag_old 
	      << " nTag_new=" << nTagJets << " vs " << nTagJet_old 
	      << " nSel_new=" <<  nSelJets << " vs " << nSelJet_old 
	      << " nAll_new=" <<  allJets.size() << " vs " << nAllJet_old << std::endl;
    return -1;
  }
  

  if(threeTag_old != threeTag_new) {
    std::cout << "event is " << event << std::endl;
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
  for(uint i = 0; i < 3;        ++i) topQuarkBJets.push_back(selJets.at(i));
  for(uint i = 2; i < nSelJets; ++i) topQuarkWJets.push_back(selJets.at(i));
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
      if(jet->p.DeltaR(can->p)<0.1){ matched = true; break; }
    }
    if(matched) continue;
    allNotCanJets.push_back(jet);
    notCanJet_pt[i] = jet->pt; notCanJet_eta[i] = jet->eta; notCanJet_phi[i] = jet->phi; notCanJet_m[i] = jet->m; i+=1;
    stNotCan += jet->pt;
  }
  nAllNotCanJets = i;//allNotCanJets.size();

  //apply bjet pt regression to candidate jets
  for(auto &jet: canJets) {
    jet->bRegression();
  }

  //choose vector boson candidate dijets when evaluate kl categorization BDT output
  for(uint i = 0; i < nOthJets; ++ i){
    for(uint j = i + 1; j < nOthJets; ++j){
      auto othDijet = std::make_shared<dijet>(othJets.at(i), othJets.at(j));
      if (othDijet->m >= 65 && othDijet->m <= 105){ // vector boson mass window
        canVDijets.push_back(othDijet);
      }
    }
  }
  std::sort(canVDijets.begin(), canVDijets.end(), sortDijetPt);

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
  if(nAntiTag != (nSelJets-nLooseTagJets)) std::cout << "eventData::computePseudoTagWeight WARNING nAntiTag = " << nAntiTag << " != " << (nSelJets-nLooseTagJets) << " = (nSelJets-nLooseTagJets)" << std::endl;

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
  std::vector<float> nPseudoTagProb;
  float nPseudoTagProbSum = 0;
  for(uint i=0; i<=nAntiTag; i++){
    float Cnk = boost::math::binomial_coefficient<float>(nAntiTag, i);
    nPseudoTagProb.push_back( threeTightTagFraction * Cnk * pow(p, i) * pow((1-p), (nAntiTag - i)) ); //i pseudo tags and nAntiTag-i pseudo antiTags
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
  nPseudoTags = nAntiTag; // Inint at max, set lower below based on cum. probs
  float cummulativeProb = 0;
  random->SetSeed(event);
  float randomProb = random->Uniform(nPseudoTagProb[0], nPseudoTagProbSum);
  for(uint i=0; i<nAntiTag+1; i++){
    //keep track of the total pseudoTagProb for at least i pseudoTags
    cummulativeProb += nPseudoTagProb[i];

    //Wait until cummulativeProb > randomProb, if never max (set above) kicks in
    if(cummulativeProb <= randomProb) continue;
    //When cummulativeProb exceeds randomProb, we have found our pseudoTag selection

    //nPseudoTags+nLooseTagJets should model the true number of b-tags in the fourTag data
    nPseudoTags = i;
    return;
  }
  
  //std::cout << "Error: Did not find a valid pseudoTag assignment" << std::endl;
  return;
}


void eventData::applyInputPseudoTagWeight(){
  pseudoTagWeight = inputPSTagWeight;

  if(pseudoTagWeight < 1e-6) std::cout << "eventData::applyInputPseudoTagWeight WARNING pseudoTagWeight " << pseudoTagWeight << " nAntiTag " << nAntiTag << std::endl;

  // update the event weight
  if(debug) std::cout << "eventData::applyInputPseudoTagWeight pseudoTagWeight " << pseudoTagWeight << std::endl;
  weight *= pseudoTagWeight;

  weightNoTrigger *= pseudoTagWeight;

  // TO do store and load nPseudoTags 
  nPseudoTags = nAntiTag;
  
  //std::cout << "Error: Did not find a valid pseudoTag assignment" << std::endl;
  return;
}



void eventData::computePseudoTagWeight(std::string jcmName){
  if(nAntiTag != (nSelJets-nLooseTagJets)) std::cout << "eventData::computePseudoTagWeight WARNING nAntiTag = " << nAntiTag << " != " << (nSelJets-nLooseTagJets) << " = (nSelJets-nLooseTagJets)" << std::endl;

  float p; float e; float d;

  p = pseudoTagProbMap[jcmName];
  e = pairEnhancementMap[jcmName];
  d = pairEnhancementDecayMap[jcmName];

  //First compute the probability to have n pseudoTags where n \in {0, ..., nAntiTag Jets}
  //float nPseudoTagProb[nAntiTag+1];
  std::vector<float> nPseudoTagProb;
  float nPseudoTagProbSum = 0;
  for(uint i=0; i<=nAntiTag; i++){
    float Cnk = boost::math::binomial_coefficient<float>(nAntiTag, i);
    nPseudoTagProb.push_back( threeTightTagFractionMap[jcmName] * Cnk * pow(p, i) * pow((1-p), (nAntiTag - i)) ); //i pseudo tags and nAntiTag-i pseudo antiTags
    if((i%2)==1) nPseudoTagProb[i] *= 1 + e/pow(nAntiTag, d);//this helps fit but makes sum of prob != 1
    nPseudoTagProbSum += nPseudoTagProb[i];
  }

  //if( fabs(nPseudoTagProbSum - 1.0) > 0.00001) std::cout << "Error: nPseudoTagProbSum - 1 = " << nPseudoTagProbSum - 1.0 << std::endl;

  pseudoTagWeightMap[jcmName]= nPseudoTagProbSum - nPseudoTagProb[0];

  if(pseudoTagWeight < 1e-6) std::cout << "eventData::computePseudoTagWeight WARNING pseudoTagWeight " << pseudoTagWeightMap[jcmName] << " nAntiTag " << nAntiTag << " nPseudoTagProbSum " << nPseudoTagProbSum << std::endl;

  // update the event weight
  if(debug) std::cout << "eventData::computePseudoTagWeight pseudoTagWeight " << pseudoTagWeight << std::endl;
  return;
}


#if SLC6 == 0 //Defined in ZZ4b/nTupleAnalysis/BuildFile.xml 
void eventData::load_SvB_ONNX(std::string fileName){
  if(fileName=="") return;
  cout << "eventData::load_SvB_ONNX( " << fileName << " )" << endl;
  SvB_ONNX = new multiClassifierONNX(fileName);
}

void eventData::run_SvB_ONNX(){
  if(!SvB_ONNX) return;
  SvB_ONNX->run(this);
  if(debug) SvB_ONNX->dump();  
  this->SvB_pzz = SvB_ONNX->c_score[2];
  this->SvB_pzh = SvB_ONNX->c_score[3];
  this->SvB_phh = SvB_ONNX->c_score[4];
  this->SvB_ptt = SvB_ONNX->c_score[1];
  this->SvB_ps  = SvB_ONNX->c_score[2] + SvB_ONNX->c_score[3] + SvB_ONNX->c_score[4];

  this->SvB_q_score[0] = SvB_ONNX->q_score[0];
  this->SvB_q_score[1] = SvB_ONNX->q_score[1];
  this->SvB_q_score[2] = SvB_ONNX->q_score[2];
  // this->SvB_q_1234 = SvB_ONNX->q_score[0];
  // this->SvB_q_1324 = SvB_ONNX->q_score[1];
  // this->SvB_q_1423 = SvB_ONNX->q_score[2];
  
}
#endif



void eventData::buildViews(){
  if(debug) std::cout<<"buildViews()\n";
  //construct all dijets from the four canJets. 
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[1], false, truth)));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[2], canJets[3], false, truth)));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[2], false, truth)));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[3], false, truth)));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[0], canJets[3], false, truth)));
  dijets.push_back(std::make_shared<dijet>(dijet(canJets[1], canJets[2], false, truth)));

  d01TruthMatch = dijets[0]->truthMatch ? dijets[0]->truthMatch->pdgId : 0;
  d23TruthMatch = dijets[1]->truthMatch ? dijets[1]->truthMatch->pdgId : 0;
  d02TruthMatch = dijets[2]->truthMatch ? dijets[2]->truthMatch->pdgId : 0;
  d13TruthMatch = dijets[3]->truthMatch ? dijets[3]->truthMatch->pdgId : 0;
  d03TruthMatch = dijets[4]->truthMatch ? dijets[4]->truthMatch->pdgId : 0;
  d12TruthMatch = dijets[5]->truthMatch ? dijets[5]->truthMatch->pdgId : 0;

  // //Find dijet with smallest dR and other dijet
  // close = *std::min_element(dijets.begin(), dijets.end(), sortdR);
  // int closeIdx = std::distance(dijets.begin(), std::find(dijets.begin(), dijets.end(), close));
  // //Index of the dijet made from the other two jets is either the one before or one after because of how we constructed the dijets vector
  // //if closeIdx is even, add one, if closeIdx is odd, subtract one.
  // int otherIdx = (closeIdx%2)==0 ? closeIdx + 1 : closeIdx - 1; 
  // other = dijets[otherIdx];

  // //flat nTuple variables for neural network inputs
  // dRjjClose = close->dR;
  // dRjjOther = other->dR;

  views.push_back(std::make_shared<eventView>(eventView(dijets[0], dijets[1], FvT_q_score[0], SvB_q_score[0], SvB_MA_q_score[0])));
  views.push_back(std::make_shared<eventView>(eventView(dijets[2], dijets[3], FvT_q_score[1], SvB_q_score[1], SvB_MA_q_score[1])));
  views.push_back(std::make_shared<eventView>(eventView(dijets[4], dijets[5], FvT_q_score[2], SvB_q_score[2], SvB_MA_q_score[2])));

  dR0123 = views[0]->dRBB;
  dR0213 = views[1]->dRBB;
  dR0312 = views[2]->dRBB;

  view_max_FvT_q_score = *std::max_element(views.begin(), views.end(), comp_FvT_q_score);
  view_max_SvB_q_score = *std::max_element(views.begin(), views.end(), comp_SvB_q_score);
  view_dR_min = *std::min_element(views.begin(), views.end(), comp_dR_close);
  close = view_dR_min->close;
  other = view_dR_min->other;
  //flat nTuple variables for neural network inputs
  dRjjClose = close->dR;
  dRjjOther = other->dR;

  random->SetSeed(11*event+5);
  for(auto &view: views){
    view->random = random->Uniform(0,1); // random float for random sorting
    if(view->passDijetMass){ view->random += 1; passDijetMass = true; } // add one so that views passing dijet mass cut are at top of list after random sort
    if(view->passMDRs)     { view->random += 1; } // add one so that views passing dijet mass cut and MDRs are at top of list after random sort
    truthMatch = truthMatch || view->truthMatch; // check if there is a view which was truth matched to two massive boson decays
  }
  std::sort(views.begin(), views.end(), sortRandom); // put in random order for random view selection  
  for(auto &view: views){ views_passMDRs.push_back(view); }

  return;
}


bool failSBSR(std::shared_ptr<eventView> &view){ return !view->passDijetMass; }
bool failMDRs(std::shared_ptr<eventView> &view){ return !view->passMDRs; }

void eventData::applyMDRs(){
  appliedMDRs = true;
  views_passMDRs.erase(std::remove_if(views_passMDRs.begin(), views_passMDRs.end(), failSBSR), views_passMDRs.end()); // only consider views within SB outer boundary
  views_passMDRs.erase(std::remove_if(views_passMDRs.begin(), views_passMDRs.end(), failMDRs), views_passMDRs.end());
  passMDRs = views_passMDRs.size() > 0;

  if(passMDRs){
    view_selected = views_passMDRs[0];
    // HHSB = view_selected->HHSB; HHCR = view_selected->HHCR; 
    // ZHSB = view_selected->ZHSB; ZHCR = view_selected->ZHCR;
    // ZZSB = view_selected->ZZSB; ZZCR = view_selected->ZZCR; 
    HHSR = view_selected->HHSR;
    ZHSR = view_selected->ZHSR;
    ZZSR = view_selected->ZZSR;
    SB = view_selected->SB; 
    // CR = view_selected->CR; 
    SR = view_selected->SR;
    leadStM = view_selected->leadSt->m; sublStM = view_selected->sublSt->m;
    //passDEtaBB = view_selected->passDEtaBB;
    selectedViewTruthMatch = view_selected->truthMatch;
  // }else{
  //   ZHSB = false; ZHCR = false; ZHSR=false;
  //   ZZSB = false; ZZCR = false; ZZSR=false;
  //   SB   = false;   CR = false;   SR=false;
  //   leadStM = 0;  sublStM = 0;
  //   //passDEtaBB = false;
  //   selectedViewTruthMatch = false;
    if(runKlBdt && canVDijets.size() > 0){
      auto score = bdtModel->getBDTScore(this, view_selected);
      BDT_kl = score["BDT"];
    }
  }
  return;
}

void eventData::buildTops(){
  //All quadjet events will have well defined xWt0, a top candidate where all three jets are allowed to be candidate jets.
  for(auto &b: topQuarkBJets){
    for(auto &j: topQuarkWJets){
      if(b.get()==j.get()) continue; //require they are different jets
      if(b->deepFlavB < j->deepFlavB) continue; //don't consider W pairs where j is more b-like than b.
      for(auto &l: topQuarkWJets){
	if(b.get()==l.get()) continue; //require they are different jets
	if(j.get()==l.get()) continue; //require they are different jets
  	if(j->deepFlavB < l->deepFlavB) continue; //don't consider W pairs where l is more b-like than j.
  	trijet* thisTop = new trijet(b,j,l);
  	if(thisTop->xWbW < xWbW0){
  	  xWt0 = thisTop->xWt;
	  xWbW0= thisTop->xWbW;
	  dRbW = thisTop->dRbW;
	  t0.reset(thisTop);
  	  xWt = xWt0; // define global xWt in this case
	  xWbW= xWbW0;
	  xW = thisTop->W->xW;
	  xt = thisTop->xt;
	  xbW = thisTop->xbW;
	  t = t0;
  	}else{delete thisTop;}
      }
    }
  }
  if(nSelJets<5) return; 

  // for events with additional jets passing preselection criteria, make top candidates requiring at least one of the jets to be not a candidate jet. 
  // This is a way to use b-tagging information without creating a bias in performance between the three and four tag data.
  // This should be a higher quality top candidate because W bosons decays cannot produce b-quarks. 
  for(auto &b: topQuarkBJets){
    for(auto &j: topQuarkWJets){
      if(b.get()==j.get()) continue; //require they are different jets
      if(b->deepFlavB < j->deepFlavB) continue; //don't consider W pairs where j is more b-like than b.
      for(auto &l: othJets){
	if(b.get()==l.get()) continue; //require they are different jets
	if(j.get()==l.get()) continue; //require they are different jets
  	if(j->deepFlavB < l->deepFlavB) continue; //don't consider W pairs where l is more b-like than j.
  	trijet* thisTop = new trijet(b,j,l);
  	if(thisTop->xWbW < xWbW1){
  	  xWt1 = thisTop->xWt;
  	  xWbW1= thisTop->xWbW;
	  dRbW = thisTop->dRbW;
  	  t1.reset(thisTop);
  	  xWt = xWt1; // overwrite global best top candidate
  	  xWbW= xWbW1; // overwrite global best top candidate
	  xW = thisTop->W->xW;
	  xt = thisTop->xt;
	  xbW = thisTop->xbW;
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
	  // xW = thisTop->W->xW;
	  // xt = thisTop->xt;
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
  std::cout << "allMuons: " << allMuons.size() << " | isoMuons: " << muons_isoMed40.size() << std::endl;

  cout << "All Jets" << endl;
  for(auto& jet : allJets){
    std::cout << "\t " << jet->pt << " (" << jet->pt_wo_bRegCorr << ") " <<  jet->eta << " " << jet->phi << " " << jet->deepB  << " " << jet->deepFlavB << " " << (jet->pt - 40) << std::endl;
  }

  cout << "Sel Jets" << endl;
  for(auto& jet : selJets){
    std::cout << "\t " << jet->pt << " " << jet->eta << " " << jet->phi << " " << jet->deepB  << " " << jet->deepFlavB << std::endl;
  }

  cout << "Tag Jets" << endl;
  for(auto& jet : tagJets){
    std::cout << "\t " << jet->pt << " " << jet->eta << " " << jet->phi << " " << jet->deepB  << " " << jet->deepFlavB << std::endl;
  }


  return;
}

eventData::~eventData(){} 


float eventData::GetTrigEmulationWeight(TriggerEmulator::TrigEmulatorTool* tEmulator){

  vector<float> selJet_pts;
  for(const jetPtr& sJet : selJets){
    selJet_pts.push_back(sJet->pt_wo_bRegCorr);
  }

  //vector<float> tagJet_pts;
  //for(const jetPtr& tJet : tagJets){
  //  tagJet_pts.push_back(tJet->pt_wo_bRegCorr);
  //}

  vector<float> tagJet_pts;
  for(const jetPtr& cJet : canJets){
    tagJet_pts.push_back(cJet->pt_wo_bRegCorr);
  }


  return tEmulator->GetWeightOR(selJet_pts, tagJet_pts, ht30_noMuon);
}




bool eventData::pass4bEmulation(unsigned int offset, bool passAll)
{
  if(debug) cout << "bool eventData::pass4bEmulation("<<offset<<","<< passAll << ")" << endl;
  if(passAll)
    return true;
  

  random->SetSeed(7*event+13);
  float randNum = random->Uniform(0,1);

  //cout << "pseudoTagWeight " << pseudoTagWeight << " vs weight " << weight << " bTag SF x pseudoTagWeight " << bTagSF * pseudoTagWeight << endl;

  // For MC this weight only included the btagSF and the JCM (if given)
  float upperLimit = ((offset+1) * weight);
  float lowerLimit = ( offset    * weight);
  //if( upperLimit > 1)
  //cout << " ----------------- upperLimit is " << upperLimit << " offset+1 " << offset+1 << " pseudoTagWeight " << pseudoTagWeight << endl;

  while(upperLimit > 1){
    unsigned int alt_offset = random->Integer(10);
    upperLimit = ((alt_offset+1) * weight);
    lowerLimit = ( alt_offset    * weight);
    //cout << " \tupperLimit is now " << upperLimit << " alt_offset is " << alt_offset << endl;
  }

  if(debug){
    cout << "randNum > lowerLimit && randNum < upperLimit = " <<randNum<<" > "<<lowerLimit<<" && "<<randNum<<" < "<<upperLimit << endl;
    cout << "                                             = " << (randNum > lowerLimit && randNum < upperLimit) << endl;
  }

  if(randNum > lowerLimit && randNum < upperLimit){
    return true;
  }

  return false;
}

void eventData::setPSJetsAsTagJets()
{
  std::sort(selJets.begin(), selJets.end(), sortTag);
  
  unsigned int nPromotedBTags = 0;

  // start at 3 b/c first 3 jets should be btagged
  for(uint i = 3; i < nSelJets; ++i){
    jetPtr& selJetRef = selJets.at(i);
    
    bool isTagJet = find(tagJets.begin(), tagJets.end(), selJetRef) != tagJets.end();
    
    
    if(!isTagJet){

      // 
      //  Needed to preseve order of the non-tags jets in btag score 
      //    but dont want to incease them too much so they have a btag-score higher than a tagged jet
      //
      float bTagOffset = 0.001*(nPseudoTags-nPromotedBTags);

      //cout << "Btagging was " << selJetRef->deepFlavB << "  now " << bTag + bTagOffset << " ( " << bTagOffset << " )" <<endl;
      selJetRef->deepFlavB = bTag + bTagOffset;
      selJetRef->deepB     = bTag + bTagOffset;
      selJetRef->CSVv2     = bTag + bTagOffset;
      
      ++nPromotedBTags;
    }

    if(nPromotedBTags == nPseudoTags)
      break;

  }
    
  //assert(nPromotedBTags == nPseudoTags );
  //if(nPromotedBTags != nPseudoTags){
  //
  //  for(uint i = 0; i < nSelJets; ++i){
  //    jetPtr& selJetRef = selJets.at(i);
  //  
  //    bool isTagJet = find(tagJets.begin(), tagJets.end(), selJetRef) != tagJets.end();
  //  }
  //}
  
  std::sort(selJets.begin(), selJets.end(), sortPt); 
  return;
}



void eventData::setLooseAndPSJetsAsTagJets(bool debug)
{
  std::sort(selJets.begin(), selJets.end(), sortTag);
  if(debug) cout << " ------ " << endl;
  unsigned int nPromotedBTags = 0;

  int nLooseNotTight = nLooseTagJets - nTagJets;
  if(debug) cout << " nLooseNotTight " << nLooseNotTight << " nPseudoTag " << nPseudoTags << endl;
  for(uint i = 0; i < nSelJets; ++i){
    jetPtr& selJetRef = selJets.at(i);
    
    bool isTagJet = find(tagJets.begin(), tagJets.end(), selJetRef) != tagJets.end();
    bool isLooseTagJet = find(looseTagJets.begin(), looseTagJets.end(), selJetRef) != looseTagJets.end();
    
    if(debug) cout << "\t tag/looseTag " << isTagJet << " " << isLooseTagJet << endl;

    if(!isTagJet){

      // 
      //  Needed to preseve order of the non-tags jets in btag score 
      //    but dont want to incease them too much so they have a btag-score higher than a tagged jet
      //
      float bTagOffset = 0.001*((nPseudoTags+nLooseNotTight)-nPromotedBTags);


      if(debug) cout << "Btagging was " << selJetRef->deepFlavB << "  now " << bTag + bTagOffset << " ( " << bTagOffset << " )  isLooseTagJet " <<  isLooseTagJet <<  " nPseudoTags " << nPseudoTags << " nPromotedBTags " << nPromotedBTags << endl;
      selJetRef->deepFlavB = bTag + bTagOffset;
      selJetRef->deepB     = bTag + bTagOffset;
      selJetRef->CSVv2     = bTag + bTagOffset;
      
      ++nPromotedBTags;
    }

    if(nPromotedBTags == (nPseudoTags+nLooseNotTight))
      break;

  }
    
  //assert(nPromotedBTags == nPseudoTags );
  //if(nPromotedBTags != (nPseudoTags+nLooseNotTight)){
  //
  //  for(uint i = 0; i < nSelJets; ++i){
  //    jetPtr& selJetRef = selJets.at(i);
  //  
  //    bool isTagJet = find(tagJets.begin(), tagJets.end(), selJetRef) != tagJets.end();
  //  }
  //}
  
  std::sort(selJets.begin(), selJets.end(), sortPt); 
  return;
}



bool eventData::passPSDataFilter(bool invertW)
{
  random->SetSeed(17*event+19);
  float randNum = random->Uniform(0,1);

  if(randNum < weight){ // use weight here to include weight and btagSF
    if(invertW) return false;
    return true;
  }

  if(invertW) return true;
  return false;
}

// https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting
//  SF  data/POWHEG+Pythia8
float eventData::ttbarSF(float pt){

  float inputPt = pt;
  if(pt > 500) inputPt = 500;
  
  return exp(0.0615 - 0.0005*inputPt);
}
