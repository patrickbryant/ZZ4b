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

#include "ZZ4b/nTupleAnalysis/interface/hemiAnalysis.h"

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
  //const edm::ParameterSet& process    = edm::readPSetsFrom(argv[1], argc, argv)->getParameter<edm::ParameterSet>("process");
  //std::shared_ptr<edm::ParameterSet> config = edm::readConfig(argv[1], argc, argv);
  //const edm::ParameterSet& process    = config->getParameter<edm::ParameterSet>("process");
  std::unique_ptr<edm::ParameterSet> config = edm::cmspybind11::readConfig(argv[1], argc, argv);
  const edm::ParameterSet& process    = config->getParameter<edm::ParameterSet>("process");

  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("hemisphereAnalysis");
  bool debug = parameters.getParameter<bool>("debug");
  bool loadJetFourVecs = parameters.getParameter<bool>("loadJetFourVecs");

  // hemiSphere Mixing
  const edm::ParameterSet& hSphereParameters = process.getParameter<edm::ParameterSet>("hSphereLib");
  //bool      createHSphereLib = hSphereParameters.getParameter<bool>("create");
  //bool      loadHSphereLib   = hSphereParameters.getParameter<bool>("load");
  std::vector<std::string> hSphereLibFiles = hSphereParameters.getParameter<std::vector<std::string> >("fileNames");
  int       maxHemi   = hSphereParameters.getParameter<int>("maxHemi");


  //Histogram output
  fwlite::OutputFiles histOutput(process);
  std::cout << " Histograms: " << histOutput.file() << std::endl;
  fwlite::TFileService fsh = fwlite::TFileService(histOutput.file());


  //
  // Define analysis and run event loop
  //
  std::cout << "Initialize analysis" << std::endl;
  std::cout << "\t loading jet four vectors ?  " << loadJetFourVecs << std::endl;
  hemiAnalysis a = hemiAnalysis(hSphereLibFiles, fsh, debug, loadJetFourVecs);

  //if(createHSphereLib){
  //  std::cout << "     Creating hemi-sphere file: " << hSphereLibFile << std::endl;
  //  a.createHemisphereLibrary(hSphereLibFile, fsh);
  //}

  a.hemiLoop(maxHemi);

  //if(createHSphereLib){
  //  std::cout << "     Created hemi-sphere file: " << hSphereLibFile << std::endl;
  //  a.storeHemiSphereFile();
  //}
  std::cout << "Done " << std::endl;

  return 0;
}
