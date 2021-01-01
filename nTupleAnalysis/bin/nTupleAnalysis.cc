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
#if SLC6 == 1 //Defined in ZZ4b/nTupleAnalysis/bin/BuildFile.xml
#include "nTupleAnalysis/baseClasses/interface/myParameterSetReader.h"
#else
#include "FWCore/PythonParameterSet/interface/MakePyBind11ParameterSets.h"
#endif 

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
#if SLC6 == 1
  const edm::ParameterSet& process    = edm::readPSetsFrom(argv[1], argc, argv)->getParameter<edm::ParameterSet>("process");
#else
  const edm::ParameterSet& process    = edm::cmspybind11::readPSetsFrom(argv[1], argc, argv)->getParameter<edm::ParameterSet>("process");
#endif 
  //std::shared_ptr<edm::ParameterSet> config = edm::readConfig(argv[1], argc, argv);
  //std::unique_ptr<edm::ParameterSet> config = edm::cmspybind11::readConfig(argv[1], argc, argv);
  //const edm::ParameterSet& process    = config->getParameter<edm::ParameterSet>("process");

  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("nTupleAnalysis");
  bool debug = parameters.getParameter<bool>("debug");
  bool isMC  = parameters.getParameter<bool>("isMC");
  bool mcUnitWeight  = parameters.getParameter<bool>("mcUnitWeight");
  bool makePSDataFromMC  = parameters.getParameter<bool>("makePSDataFromMC");
  bool removePSDataFromMC  = parameters.getParameter<bool>("removePSDataFromMC");
  bool isDataMCMix  = parameters.getParameter<bool>("isDataMCMix");
  bool skip4b  = parameters.getParameter<bool>("skip4b");
  bool skip3b  = parameters.getParameter<bool>("skip3b");
  bool is3bMixed  = parameters.getParameter<bool>("is3bMixed");
  bool emulate4bFrom3b  = parameters.getParameter<bool>("emulate4bFrom3b");
  int  emulationOffset  = parameters.getParameter<int>("emulationOffset");
  bool blind = parameters.getParameter<bool>("blind");
  int histogramming = parameters.getParameter<int>("histogramming");
  int histDetailLevel = parameters.getParameter<int>("histDetailLevel");
  bool doReweight = parameters.getParameter<bool>("doReweight");
  float lumi = parameters.getParameter<double>("lumi");
  float xs   = parameters.getParameter<double>("xs");
  float fourbkfactor   = parameters.getParameter<double>("fourbkfactor");
  std::string year = parameters.getParameter<std::string>("year");
  bool    doTrigEmulation = parameters.getParameter<bool>("doTrigEmulation");
  bool    doTrigStudy     = parameters.getParameter<bool>("doTrigStudy");
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
  std::vector<std::string> inputWeightFiles = parameters.getParameter<std::vector<std::string> >("inputWeightFiles");

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
  if(inputWeightFiles.size()){
    TChain* eventWeights     = new TChain("Events");
    for(std::string inputWeightFile : inputWeightFiles){
      std::cout << "           Input Weight File: " << inputWeightFile << std::endl;
      int e = eventWeights    ->AddFile(inputWeightFile.c_str());
      if(e!=1){ std::cout << "ERROR" << std::endl; return 1;}
    }
    events->AddFriend(eventWeights);
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
  analysis a = analysis(events, runs, lumiBlocks, fsh, isMC, blind, year, histogramming, histDetailLevel, 
			doReweight, debug, fastSkim, doTrigEmulation, doTrigStudy, isDataMCMix, is3bMixed, 
			bjetSF, btagVariations,
			JECSyst, friendFile,
			looseSkim, FvTName);
  a.event->setTagger(bTagger, bTag);
  a.makePSDataFromMC = makePSDataFromMC;
  a.removePSDataFromMC = removePSDataFromMC;
  a.mcUnitWeight = mcUnitWeight;
  a.skip4b = skip4b;
  a.skip3b = skip3b;

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

  #if SLC6 == 0
  std::string SvB_ONNX = parameters.getParameter<std::string>("SvB_ONNX");
  a.event->load_SvB_ONNX(SvB_ONNX);
  #endif
  
  a.writeOutEventNumbers = writeOutEventNumbers;

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

  if(loadHSphereLib){
    std::cout << "     Loading hemi-sphere files... " << std::endl;
    a.loadHemisphereLibrary(hSphereLibFiles_3tag, hSphereLibFiles_4tag, fsh, maxNHemis);
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
