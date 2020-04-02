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
//#include "nTupleAnalysis/baseClasses/interface/myParameterSetReader.h"
#include "FWCore/PythonParameterSet/interface/MakePyBind11ParameterSets.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"

#include "ZZ4b/nTupleAnalysis/interface/analysis.h"

using namespace nTupleAnalysis;

int main(int argc, char * argv[]){
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
  const edm::ParameterSet& process    = edm::readPSetsFrom(argv[1], argc, argv)->getParameter<edm::ParameterSet>("process");
  //std::shared_ptr<edm::ParameterSet> config = edm::readConfig(argv[1], argc, argv);
  //const edm::ParameterSet& process    = config->getParameter<edm::ParameterSet>("process");

  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("nTupleAnalysis");
  bool debug = parameters.getParameter<bool>("debug");
  bool isMC  = parameters.getParameter<bool>("isMC");
  bool mcUnitWeight  = parameters.getParameter<bool>("mcUnitWeight");
  bool isDataMCMix  = parameters.getParameter<bool>("isDataMCMix");
  bool skip4b  = parameters.getParameter<bool>("skip4b");
  bool skip3b  = parameters.getParameter<bool>("skip3b");
  bool emulate4bFrom3b  = parameters.getParameter<bool>("emulate4bFrom3b");
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
			doReweight, debug, fastSkim, doTrigEmulation, doTrigStudy, mcUnitWeight, isDataMCMix, skip4b, skip3b,
			bjetSF, btagVariations,
			JECSyst, friendFile,
			looseSkim);
  a.event->setTagger(bTagger, bTag);
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
  std::string jetCombinatoricModel = parameters.getParameter<std::string>("jetCombinatoricModel");
  a.storeJetCombinatoricModel(jetCombinatoricModel);
  a.emulate4bFrom3b = emulate4bFrom3b;
  //std::string reweight = parameters.getParameter<std::string>("reweight");
  //a.storeReweight(reweight);

  if(createPicoAOD){
    std::cout << "     Creating picoAOD: " << picoAODFile << std::endl;
    
    // If we do hemisphere mixing, dont copy orignal picoAOD output
    bool copyInputPicoAOD = !loadHSphereLib && !emulate4bFrom3b;
    std::cout << "     \t using fastSkim: " << fastSkim << std::endl;
    std::cout << "     \t copying Input picoAOD: " << copyInputPicoAOD << std::endl;
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

  if(createPicoAOD && (loadHSphereLib || emulate4bFrom3b)){
    std::cout << "     Creating new PicoAOD Branches... " << std::endl;
    a.createPicoAODBranches();
  }


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
