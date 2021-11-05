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

#include "ZZ4b/nTupleAnalysis/interface/tTbarAnalysis.h"

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

  const edm::ParameterSet& parameters = process.getParameter<edm::ParameterSet>("tTbarAnalysis");
  bool debug = parameters.getParameter<bool>("debug");
  bool isMC  = parameters.getParameter<bool>("isMC");
  std::string histDetailLevel = parameters.getParameter<std::string>("histDetailLevel");
  float lumi = parameters.getParameter<double>("lumi");
  float xs   = parameters.getParameter<double>("xs");
  float fourbkfactor   = parameters.getParameter<double>("fourbkfactor");
  std::string year = parameters.getParameter<std::string>("year");
  bool    doTrigEmulation = parameters.getParameter<bool>("doTrigEmulation");
  bool doTTbarPtReweight = parameters.getParameter<bool>("doTTbarPtReweight");
  int         firstEvent = parameters.getParameter<int>("firstEvent");
  float       bTag    = parameters.getParameter<double>("bTag");
  std::string bTagger = parameters.getParameter<std::string>("bTagger");
  std::string bjetSF  = parameters.getParameter<std::string>("bjetSF");
  std::string btagVariations = parameters.getParameter<std::string>("btagVariations");
  std::string JECSyst = parameters.getParameter<std::string>("JECSyst");
  std::string friendFile = parameters.getParameter<std::string>("friendFile");
  bool writeOutEventNumbers = parameters.getParameter<bool>("writeOutEventNumbers");

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
  tTbarAnalysis a = tTbarAnalysis(events, runs, lumiBlocks, fsh, isMC, year, histDetailLevel, 
				  debug, doTrigEmulation, bjetSF, btagVariations,
				  JECSyst, friendFile);
  a.event->setTagger(bTagger, bTag);


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

  
  a.writeOutEventNumbers = writeOutEventNumbers;

  if(createPicoAOD){
    std::cout << "     Creating picoAOD: " << picoAODFile << std::endl;
    a.createPicoAOD(picoAODFile, true);
  }


  int maxEvents = inputHandler.maxEvents();
  a.eventLoop(maxEvents, firstEvent);

  if(createPicoAOD){
    std::cout << "      Created picoAOD: " << picoAODFile << std::endl;
    a.storePicoAOD();
  }

  return 0;
}
