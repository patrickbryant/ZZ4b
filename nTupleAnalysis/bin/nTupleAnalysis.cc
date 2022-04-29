#include <iostream>
#include <iomanip>
#include <TROOT.h>
#include <TFile.h>
#include "TSystem.h"
#include "TChain.h"

#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"

#include "DataFormats/FWLite/interface/InputSource.h"
#include "DataFormats/FWLite/interface/OutputFiles.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakePyBind11ParameterSets.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"

#include "ZZ4b/nTupleAnalysis/interface/analysis.h"

using namespace nTupleAnalysis;

int main(int argc, char * argv[]){
  std::cout << "int nTupleAnalysis::main(int argc, char * argv[])" << std::endl;
  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();
  
  // parse arguments
  if ( argc < 2 ) {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  //
  // get the python configuration
  //
  const edm::ParameterSet& process    = edm::cmspybind11::readPSetsFrom(argv[1], argc, argv)->getParameter<edm::ParameterSet>("process");

  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("nTupleAnalysis");
  bool debug = parameters.getParameter<bool>("debug");
  bool isMC  = parameters.getParameter<bool>("isMC");
  bool mcUnitWeight  = parameters.getParameter<bool>("mcUnitWeight");
  bool makePSDataFromMC  = parameters.getParameter<bool>("makePSDataFromMC");
  bool removePSDataFromMC  = parameters.getParameter<bool>("removePSDataFromMC");
  bool isDataMCMix  = parameters.getParameter<bool>("isDataMCMix");
  bool skip4b  = parameters.getParameter<bool>("skip4b");
  bool skip3b  = parameters.getParameter<bool>("skip3b");
  bool usePreCalcBTagSFs  = parameters.getParameter<bool>("usePreCalcBTagSFs");
  bool emulate4bFrom3b  = parameters.getParameter<bool>("emulate4bFrom3b");
  int  emulationOffset  = parameters.getParameter<int>("emulationOffset");
  bool blind = parameters.getParameter<bool>("blind");
  std::string histDetailLevel = parameters.getParameter<std::string>("histDetailLevel");
  bool doReweight = parameters.getParameter<bool>("doReweight");
  bool doDvTReweight = parameters.getParameter<bool>("doDvTReweight");
  bool doTTbarPtReweight = parameters.getParameter<bool>("doTTbarPtReweight");
  float lumi = parameters.getParameter<double>("lumi");
  float xs   = parameters.getParameter<double>("xs");
  float fourbkfactor   = parameters.getParameter<double>("fourbkfactor");
  std::string year = parameters.getParameter<std::string>("year");
  bool    doTrigEmulation = parameters.getParameter<bool>("doTrigEmulation");
  bool    calcTrigWeights = parameters.getParameter<bool>("calcTrigWeights");
  bool    useMCTurnOns    = parameters.getParameter<bool>("useMCTurnOns");
  bool    useUnitTurnOns    = parameters.getParameter<bool>("useUnitTurnOns");
  int         firstEvent = parameters.getParameter<int>("firstEvent");
  float       bTag    = parameters.getParameter<double>("bTag");
  std::string bTagger = parameters.getParameter<std::string>("bTagger");
  std::string bjetSF  = parameters.getParameter<std::string>("bjetSF");
  std::string btagVariations = parameters.getParameter<std::string>("btagVariations");
  std::string JECSyst = parameters.getParameter<std::string>("JECSyst");
  std::string friendFile = parameters.getParameter<std::string>("friendFile");
  bool looseSkim = parameters.getParameter<bool>("looseSkim");
  bool writeOutEventNumbers = parameters.getParameter<bool>("writeOutEventNumbers");
  std::string FvTName = parameters.getParameter<std::string>("FvTName");
  std::string reweight4bName = parameters.getParameter<std::string>("reweight4bName");
  std::string reweightDvTName = parameters.getParameter<std::string>("reweightDvTName");
  std::vector<std::string> friends          = parameters.getParameter<std::vector<std::string> >("friends");
  std::vector<std::string> otherWeights          = parameters.getParameter<std::vector<std::string> >("otherWeights");
  std::vector<std::string> inputWeightFiles = parameters.getParameter<std::vector<std::string> >("inputWeightFiles");
  std::vector<std::string> inputWeightFiles4b = parameters.getParameter<std::vector<std::string> >("inputWeightFiles4b");
  std::vector<std::string> inputWeightFilesDvT = parameters.getParameter<std::vector<std::string> >("inputWeightFilesDvT");
  std::string klBdtWeightFile = parameters.getParameter<std::string>("klBdtWeightFile");
  std::string klBdtMethods = parameters.getParameter<std::string>("klBdtMethods");
  bool runKlBdt = parameters.getParameter<bool>("runKlBdt");

  //lumiMask
  const edm::ParameterSet& inputs = process.getParameter<edm::ParameterSet>("inputs");   
  std::vector<edm::LuminosityBlockRange> lumiMask;
  if( !isMC && inputs.exists("lumisToProcess") ){
    std::vector<edm::LuminosityBlockRange> const & lumisTemp = inputs.getUntrackedParameter<std::vector<edm::LuminosityBlockRange> > ("lumisToProcess");
    lumiMask.resize( lumisTemp.size() );
    copy( lumisTemp.begin(), lumisTemp.end(), lumiMask.begin() );
  }
  if(debug) for(auto lumiID: lumiMask) std::cout<<"lumiID "<<lumiID<<std::endl;

  //picoAOD
  const edm::ParameterSet& picoAODParameters = process.getParameter<edm::ParameterSet>("picoAOD");
  //bool         usePicoAOD = picoAODParameters.getParameter<bool>("use");
  bool      createPicoAOD = picoAODParameters.getParameter<bool>("create");
  bool           fastSkim = picoAODParameters.getParameter<bool>("fastSkim");
  std::string picoAODFile = picoAODParameters.getParameter<std::string>("fileName");
  //fwlite::TFileService fst = fwlite::TFileService(picoAODFile);

  // hemiSphere Mixing
  const edm::ParameterSet& hSphereParameters = process.getParameter<edm::ParameterSet>("hSphereLib");
  bool      createHSphereLib = hSphereParameters.getParameter<bool>("create");
  bool      writePicoAODBeforeDiJetMass = hSphereParameters.getParameter<bool>("noMjjInPAOD");
  bool      loadHSphereLib   = hSphereParameters.getParameter<bool>("load");
  bool      useHemiWeights   = hSphereParameters.getParameter<bool>("useHemiWeights");
  double      mcHemiWeight   = hSphereParameters.getParameter<double>("mcHemiWeight");
  std::string hSphereLibFile = hSphereParameters.getParameter<std::string>("fileName");
  std::vector<std::string> hSphereLibFiles_3tag = hSphereParameters.getParameter<std::vector<std::string> >("inputHLibs_3tag");
  std::vector<std::string> hSphereLibFiles_4tag = hSphereParameters.getParameter<std::vector<std::string> >("inputHLibs_4tag");
  int       maxNHemis   = hSphereParameters.getParameter<int>("maxNHemis");
  //fwlite::TFileService fst = fwlite::TFileService(picoAODFile);


  //NANOAOD Input source
  fwlite::InputSource inputHandler(process); 

  //Init Events Tree and Runs Tree which contains info for MC weight calculation
  TChain* events     = new TChain("Events");
  TChain* runs       = new TChain("Runs");
  TChain* lumiBlocks = new TChain("LuminosityBlocks");
  for(unsigned int iFile=0; iFile<inputHandler.files().size(); ++iFile){
    std::cout << "           Input File: " << inputHandler.files()[iFile].c_str() << std::endl;
    int e = events    ->AddFile(inputHandler.files()[iFile].c_str());
    int r = runs      ->AddFile(inputHandler.files()[iFile].c_str());
    int l = lumiBlocks->AddFile(inputHandler.files()[iFile].c_str());
    if(e!=1 || r!=1 || l!=1){ std::cout << "ERROR" << std::endl; return 1;}
    if(debug){
      std::cout<<"Added to TChain"<<std::endl;
      events->Show(0);
    }
  }

  //
  //  Add an input file as a friend
  //
  if(friends.size()){
    for(std::string friendFile : friends){
      TChain* thisFriend = new TChain("Events");
      std::cout << "           Friend File: " << friendFile << std::endl;
      int e = thisFriend->AddFile(friendFile.c_str());
      if(e!=1){ std::cout << "ERROR" << std::endl; return 1;}
      thisFriend->SetName(friendFile.c_str());
      events->AddFriend(thisFriend);
    }
  }

  if(inputWeightFiles.size()){
    TChain* eventWeights     = new TChain("Events");
    for(std::string inputWeightFile : inputWeightFiles){
      std::cout << "           Input Weight File: " << inputWeightFile << std::endl;
      int e = eventWeights    ->AddFile(inputWeightFile.c_str());
      if(e!=1){ std::cout << "ERROR" << std::endl; return 1;}
    }
    eventWeights->SetName("EventsWeights");
    events->AddFriend(eventWeights);
  }


  if(inputWeightFiles4b.size()){
    TChain* eventWeights4b     = new TChain("Events");
    for(std::string inputWeightFile4b : inputWeightFiles4b){
      std::cout << "           Input 4b Weight File: " << inputWeightFile4b << std::endl;
      int e = eventWeights4b    ->AddFile(inputWeightFile4b.c_str());
      if(e!=1){ std::cout << "ERROR" << std::endl; return 1;}
    }
    eventWeights4b->SetName("Events4bWeights");
    events->AddFriend(eventWeights4b);
  }

  if(inputWeightFilesDvT.size()){
    TChain* eventWeightsDvT     = new TChain("Events");
    for(std::string inputWeightFileDvT : inputWeightFilesDvT){
      std::cout << "           Input DvT Weight File: " << inputWeightFileDvT << std::endl;
      int e = eventWeightsDvT    ->AddFile(inputWeightFileDvT.c_str());
      if(e!=1){ std::cout << "ERROR" << std::endl; return 1;}
    }
    eventWeightsDvT->SetName("EventsDvTWeights");
    events->AddFriend(eventWeightsDvT);
  }
  



  //Histogram output
  fwlite::OutputFiles histOutput(process);
  std::cout << "Event Loop Histograms: " << histOutput.file() << std::endl;
  fwlite::TFileService fsh = fwlite::TFileService(histOutput.file());


  //
  // Define analysis and run event loop
  //
  std::cout << "Initialize analysis" << std::endl;
  if(doTrigEmulation)
    std::cout << "\t emulating the trigger. " << std::endl;
  if(calcTrigWeights)
    std::cout << "\t calculating trigger weights. " << std::endl;
  if(useMCTurnOns)
    std::cout << "\t using MC Turn-ons. " << std::endl;
  if(useUnitTurnOns)
    std::cout << "\t using Unit Turn-ons. (ie:no trigger applied) " << std::endl;

  analysis a = analysis(events, runs, lumiBlocks, fsh, isMC, blind, year, histDetailLevel, 
			doReweight, debug, fastSkim, doTrigEmulation, calcTrigWeights, useMCTurnOns, useUnitTurnOns, isDataMCMix, usePreCalcBTagSFs, 
			bjetSF, btagVariations,
			JECSyst, friendFile,
			looseSkim, FvTName, reweight4bName, reweightDvTName, otherWeights,
			klBdtWeightFile, klBdtMethods, runKlBdt);
      
  a.event->setTagger(bTagger, bTag);
  a.makePSDataFromMC = makePSDataFromMC;
  a.removePSDataFromMC = removePSDataFromMC;
  a.mcUnitWeight = mcUnitWeight;
  a.skip4b = skip4b;
  a.skip3b = skip3b;

  for(std::string oWeight : otherWeights){
    std::cout << "Will weight events with : (" << oWeight << ")" << std::endl;
  }

  if(doDvTReweight){
    std::cout << "\t doDvTReweight = true " << std::endl;    
    a.event->doDvTReweight = true;
  }

  if(doTTbarPtReweight){
    std::cout << "\t doTTbarPtReweight = true " << std::endl;    
    a.event->doTTbarPtReweight = true;
  }

  if(isMC){
    a.lumi     = lumi;
    a.xs       = xs;
    a.fourbkfactor = fourbkfactor;
  }
  if(!isMC){
    a.lumiMask = lumiMask;
    std::string lumiData = parameters.getParameter<std::string>("lumiData");
    a.getLumiData(lumiData);
  }

  std::string jcmNameLoad          = parameters.getParameter<std::string>("jcmNameLoad");
  std::string jetCombinatoricModel = parameters.getParameter<std::string>("jetCombinatoricModel");

  if(jcmNameLoad != ""){
    a.loadJetCombinatoricModel(jcmNameLoad);
  }else{
    a.storeJetCombinatoricModel(jetCombinatoricModel);
  }

  std::vector<std::string> jcmFileList = parameters.getParameter<std::vector<std::string> >("jcmFileList");
  std::vector<std::string> jcmNameList = parameters.getParameter<std::vector<std::string> >("jcmNameList");

  unsigned int nJCMFile = jcmNameList.size();
  for(unsigned int iJCM = 0; iJCM<nJCMFile; ++iJCM){
    std::cout << "Will add JCM weights with name: " << jcmNameList.at(iJCM) << " from file " <<  jcmFileList.at(iJCM) << std::endl;
    a.storeJetCombinatoricModel(jcmNameList.at(iJCM),jcmFileList.at(iJCM));
  }


  a.emulate4bFrom3b = emulate4bFrom3b;
  a.emulationOffset = emulationOffset;
  if(emulate4bFrom3b){
    std::cout << "     Sub-sampling the 3b with offset: " << emulationOffset << std::endl;    
  }
  //std::string reweight = parameters.getParameter<std::string>("reweight");
  //a.storeReweight(reweight);

  std::string SvB_ONNX = parameters.getParameter<std::string>("SvB_ONNX");
  a.event->load_SvB_ONNX(SvB_ONNX);
  
  a.writeOutEventNumbers = writeOutEventNumbers;


  if(loadHSphereLib){
    std::cout << "     Loading hemi-sphere files... " << std::endl;
    std::cout << "     \t useHemiWeights set to " << useHemiWeights << std::endl;
    std::cout << "     \t mcHemiWeight set to " << mcHemiWeight << std::endl;
    a.loadHemisphereLibrary(hSphereLibFiles_3tag, hSphereLibFiles_4tag, fsh, maxNHemis, useHemiWeights, mcHemiWeight);
  }


  if(createPicoAOD){
    std::cout << "     Creating picoAOD: " << picoAODFile << std::endl;
    
    // If we do hemisphere mixing, dont copy orignal picoAOD output
    bool copyInputPicoAOD = !loadHSphereLib && !emulate4bFrom3b;
    std::cout << "     \t fastSkim: " << fastSkim << std::endl;
    std::cout << "     \t copy Input TTree structure for output picoAOD: " << copyInputPicoAOD << std::endl;
    a.createPicoAOD(picoAODFile, copyInputPicoAOD);
  }

  if(createHSphereLib){
    std::cout << "     Creating hemi-sphere file: " << hSphereLibFile << std::endl;
    a.createHemisphereLibrary(hSphereLibFile, fsh);
  }else if(writePicoAODBeforeDiJetMass){
    std::cout << "     Writting pico AODs before DiJetMass Cut " << std::endl;    
    a.writePicoAODBeforeDiJetMass = true;
  }

  // if(createPicoAOD && (loadHSphereLib || emulate4bFrom3b)){
  //   std::cout << "     Creating new PicoAOD Branches... " << std::endl;
  //   a.createPicoAODBranches();
  // }


  int maxEvents = inputHandler.maxEvents();
  a.eventLoop(maxEvents, firstEvent);

  if(createPicoAOD){
    std::cout << "      Created picoAOD: " << picoAODFile << std::endl;
    a.storePicoAOD();
  }

  if(createHSphereLib){
    std::cout << "     Created hemi-sphere file: " << hSphereLibFile << std::endl;
    a.storeHemiSphereFile();
  }


  return 0;
}
