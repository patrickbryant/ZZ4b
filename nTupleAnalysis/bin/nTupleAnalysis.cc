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
#include "nTupleAnalysis/baseClasses/interface/myParameterSetReader.h"

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
  bool blind = parameters.getParameter<bool>("blind");
  int histogramming = parameters.getParameter<int>("histogramming");
  float lumi = parameters.getParameter<double>("lumi");
  float xs   = parameters.getParameter<double>("xs");
  std::string year = parameters.getParameter<std::string>("year");
  float       bTag    = parameters.getParameter<double>("bTag");
  std::string bTagger = parameters.getParameter<std::string>("bTagger");

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
  std::string picoAODFile = picoAODParameters.getParameter<std::string>("fileName");
  //fwlite::TFileService fst = fwlite::TFileService(picoAODFile);

  // hemiSphere Mixing
  const edm::ParameterSet& hSphereParameters = process.getParameter<edm::ParameterSet>("hSphereLib");
  bool      createHSphereLib = hSphereParameters.getParameter<bool>("create");
  bool      loadHSphereLib   = hSphereParameters.getParameter<bool>("load");
  std::string hSphereLibFile = hSphereParameters.getParameter<std::string>("fileName");
  std::vector<std::string> hSphereLibFiles_3tag = hSphereParameters.getParameter<std::vector<std::string> >("inputHLibs_3tag");
  std::vector<std::string> hSphereLibFiles_4tag = hSphereParameters.getParameter<std::vector<std::string> >("inputHLibs_4tag");
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
    if(debug) std::cout<<"Added to TChain"<<std::endl;
  }

  //Histogram output
  fwlite::OutputFiles histOutput(process);
  std::cout << "Event Loop Histograms: " << histOutput.file() << std::endl;
  fwlite::TFileService fsh = fwlite::TFileService(histOutput.file());


  //
  // Define analysis and run event loop
  //
  std::cout << "Initialize analysis" << std::endl;
  analysis a = analysis(events, runs, lumiBlocks, fsh, isMC, blind, year, histogramming, debug);
  a.event->setTagger(bTagger, bTag);
  if(isMC){
    a.lumi     = lumi;
    a.xs       = xs;
  }
  if(!isMC){
    a.lumiMask = lumiMask;
    std::string lumiData = parameters.getParameter<std::string>("lumiData");
    a.getLumiData(lumiData);
  }
  std::string jetCombinatoricModel = parameters.getParameter<std::string>("jetCombinatoricModel");
  a.storeJetCombinatoricModel(jetCombinatoricModel);
  bool doReweight = parameters.getParameter<bool>("doReweight");
  a.doReweight = doReweight;
  //std::string reweight = parameters.getParameter<std::string>("reweight");
  //a.storeReweight(reweight);

  if(createPicoAOD){
    std::cout << "     Creating picoAOD: " << picoAODFile << std::endl;
    a.createPicoAOD(picoAODFile);
    a.addDerivedQuantitiesToPicoAOD();
  }

  if(createHSphereLib){
    std::cout << "     Creating hemi-sphere file: " << hSphereLibFile << std::endl;
    a.createHemisphereLibrary(hSphereLibFile, fsh);
  }

  if(loadHSphereLib){
    std::cout << "     Loading hemi-sphere files... " << std::endl;
    a.loadHemisphereLibrary(hSphereLibFiles_3tag, hSphereLibFiles_4tag, fsh);
  }

  int maxEvents = inputHandler.maxEvents();
  a.eventLoop(maxEvents);

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
