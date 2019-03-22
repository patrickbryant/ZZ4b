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
#include "ZZ4b/NtupleAna/interface/myParameterSetReader.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"

#include "ZZ4b/NtupleAna/interface/analysis.h"

using namespace NtupleAna;

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

  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("procNtupleTest");
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
  bool         usePicoAOD = picoAODParameters.getParameter<bool>("use");
  bool      createPicoAOD = picoAODParameters.getParameter<bool>("create");
  std::string picoAODFile = picoAODParameters.getParameter<std::string>("fileName");
  //fwlite::TFileService fst = fwlite::TFileService(picoAODFile);

  //NANOAOD Input source
  fwlite::InputSource inputHandler(process); 

  //Init Events Tree and Runs Tree which contains info for MC weight calculation
  TChain* events     = new TChain("Events");
  TChain* runs       = new TChain("Runs");
  TChain* lumiBlocks = new TChain("LuminosityBlocks");
  if(usePicoAOD){
    std::cout << "        Using picoAOD: " << picoAODFile << std::endl;
    events    ->Add(picoAODFile.c_str());
    runs      ->Add(picoAODFile.c_str());
    lumiBlocks->Add(picoAODFile.c_str());
  }else{
    for(unsigned int iFile=0; iFile<inputHandler.files().size(); ++iFile){
      std::cout << "           Input File: " << inputHandler.files()[iFile].c_str() << std::endl;
      events    ->Add(inputHandler.files()[iFile].c_str());
      runs      ->Add(inputHandler.files()[iFile].c_str());
      lumiBlocks->Add(inputHandler.files()[iFile].c_str());
      if(debug) std::cout<<"Added to TChain"<<std::endl;
    }
  }

  //Histogram output
  fwlite::OutputFiles histOutput(process);
  std::cout << "Event Loop Histograms: " << histOutput.file() << std::endl;
  fwlite::TFileService fsh = fwlite::TFileService(histOutput.file());


  //
  // Define analysis and run event loop
  //
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

  if(createPicoAOD){
    std::cout << "     Creating picoAOD: " << picoAODFile << std::endl;
    a.createPicoAOD(picoAODFile);
    a.addDerivedQuantitiesToPicoAOD();
  }

  int maxEvents = inputHandler.maxEvents();
  a.eventLoop(maxEvents);

  if(createPicoAOD) 
    a.storePicoAOD();

  return 0;
}
