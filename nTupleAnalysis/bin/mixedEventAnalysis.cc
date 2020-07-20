#include <iostream>
#include <iomanip>
#include <TROOT.h>
#include <TFile.h>
#include "TSystem.h"
#include "TChain.h"

//#include "DataFormats/FWLite/interface/Event.h"
//#include "DataFormats/FWLite/interface/Handle.h"
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

#include "ZZ4b/nTupleAnalysis/interface/mixedAnalysis.h"

using namespace nTupleAnalysis;

int main(int argc, char * argv[]){
  std::cout << " In mixedEventAnalysis " << std::endl;

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();
  std::cout << " FWLiteEnabler " << std::endl;
  
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
  //const edm::ParameterSet& process    = config->getParameter<edm::ParameterSet>("process");
  //std::unique_ptr<edm::ParameterSet> config = edm::cmspybind11::readConfig(argv[1], argc, argv);
  //const edm::ParameterSet& process    = config->getParameter<edm::ParameterSet>("process");

  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("mixedEventAnalysis");
  bool debug = parameters.getParameter<bool>("debug");
  int  firstEvent = parameters.getParameter<int>("firstEvent");

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
  std::cout << " Histograms: " << histOutput.file() << std::endl;
  fwlite::TFileService fsh = fwlite::TFileService(histOutput.file());

  //
  // Define analysis and run event loop
  //
  std::cout << "Initialize analysis" << std::endl;
  mixedAnalysis a = mixedAnalysis(events, runs, lumiBlocks, fsh, debug);

  int maxEvents = inputHandler.maxEvents();
  a.eventLoop(maxEvents, firstEvent);

  std::cout << "Done " << std::endl;

  return 0;
}
