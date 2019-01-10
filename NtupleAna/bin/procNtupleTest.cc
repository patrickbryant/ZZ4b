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
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"

#include "ZZ4b/NtupleAna/interface/analysis.h"
//#include "ZZ4b/NtupleAna/interface/Helpers.h"

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

  if( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ){
    std::cout << " ERROR: ParametersSet 'process' is missing in your configuration file" << std::endl; exit(0);
  }

  // get the python configuration
  const edm::ParameterSet& process = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");
  fwlite::InputSource inputHandler(process); 
  fwlite::OutputFiles outputHandler(process);
  fwlite::TFileService fs = fwlite::TFileService(outputHandler.file());

  //
  // Get the config
  //
  // now get each parameter
  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("procNtupleTest");
  bool debug = parameters.getParameter<bool>("debug");

  //
  //  Init Tree
  //
  TChain* tree = new TChain("Events");
  for(unsigned int iFile=0; iFile<inputHandler.files().size(); ++iFile){
    // open input file (can be located on castor)
    std::cout << "inputFile is " << inputHandler.files()[iFile].c_str() << std::endl;

    tree->Add(inputHandler.files()[iFile].c_str());
    if(debug) std::cout<<"Added to TChain"<<std::endl;
  }

  //
  // Define analysis and run event loop
  //
  analysis a = analysis(tree, fs, debug);

  int maxEvents = inputHandler.maxEvents();
  float lumi = parameters.getParameter<double>("lumi");
  a.lumi = lumi;
  a.eventLoop(maxEvents);

  return 0;
}
