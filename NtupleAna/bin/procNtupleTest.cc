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

  //
  // get the python configuration
  //
  const edm::ParameterSet& process    = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");
  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("procNtupleTest");
  bool debug = parameters.getParameter<bool>("debug");
  bool isMC  = parameters.getParameter<bool>("isMC");
  float lumi = parameters.getParameter<double>("lumi");
  std::string year = parameters.getParameter<std::string>("year");

  //lumiMask
  const edm::ParameterSet& inputs = process.getParameter<edm::ParameterSet>("inputs");   
  std::vector<edm::LuminosityBlockRange> lumiMask;
  if( inputs.exists("lumisToProcess") ){
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
  TChain* events = new TChain("Events");
  TChain* runs   = new TChain("Runs");
  if(usePicoAOD){
    std::cout << "inputFile is " << picoAODFile << std::endl;
    events->Add(picoAODFile.c_str());
    if(isMC){
      runs->Add(picoAODFile.c_str());
    }
  }else{
    for(unsigned int iFile=0; iFile<inputHandler.files().size(); ++iFile){
      std::cout << "inputFile is " << inputHandler.files()[iFile].c_str() << std::endl;
      events->Add(inputHandler.files()[iFile].c_str());
      if(isMC){
	runs->Add(inputHandler.files()[iFile].c_str());
      }
      if(debug) std::cout<<"Added to TChain"<<std::endl;
    }
  }

  //Histogramming
  fwlite::OutputFiles histogramming(process);
  fwlite::TFileService fsh = fwlite::TFileService(histogramming.file());


  //
  // Define analysis and run event loop
  //
  analysis a = analysis(events, runs, fsh, isMC, year, debug);
  a.lumi     = lumi;
  a.lumiMask = lumiMask;

  if(createPicoAOD) a.createPicoAOD(picoAODFile);

  int maxEvents = inputHandler.maxEvents();
  a.eventLoop(maxEvents);

  if(createPicoAOD) a.storePicoAOD();

  return 0;
}
