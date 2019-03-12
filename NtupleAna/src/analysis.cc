#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>

#include "ZZ4b/NtupleAna/interface/analysis.h"

using namespace NtupleAna;

analysis::analysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _isMC, bool _blind, std::string _year, int _histogramming, bool _debug){
  if(_debug) std::cout<<"In analysis constructor"<<std::endl;
  debug      = _debug;
  isMC       = _isMC;
  blind      = _blind;
  year       = _year;
  events     = _events;
  events->SetBranchStatus("*", 0);
  runs       = _runs;
  histogramming = _histogramming;

  //Calculate MC weight denominator
  if(isMC){
    runs->SetBranchStatus("*", 0);
    initBranch(runs, "genEventCount", &genEventCount);
    initBranch(runs, "genEventSumw",  &genEventSumw);
    initBranch(runs, "genEventSumw2", &genEventSumw2);
    for(int r = 0; r < runs->GetEntries(); r++){
      runs->GetEntry(r);
      mcEventCount += genEventCount;
      mcEventSumw  += genEventSumw;
      mcEventSumw2 += genEventSumw2;
    }
    std::cout << "mcEventCount " << mcEventCount << " | mcEventSumw " << mcEventSumw << std::endl;
  }

  lumiBlocks = _lumiBlocks;
  event      = new eventData(events, isMC, year, debug);
  treeEvents = events->GetEntries();
  cutflow    = new tagCutflowHists("cutflow", fs);

  // hists
  if(histogramming > 4        ) allEvents    = new eventHists("allEvents",  fs);
  if(histogramming > 3        ) passPreSel   = new   tagHists("passPreSel", fs, true, blind);
  if(histogramming > 2        ) passMDRs     = new   tagHists("passMDRs",   fs, true, blind);
  if(histogramming > 1        ) passMDCs     = new   tagHists("passMDCs",   fs, true, blind);
  if(histogramming > 0        ) passDEtaBB   = new   tagHists("passDEtaBB", fs, true, blind);
} 

void analysis::createPicoAOD(std::string fileName){
  writePicoAOD = true;
  picoAODFile = TFile::Open(fileName.c_str() , "RECREATE");
  picoAODEvents     = events    ->CloneTree(0);
  picoAODRuns       = runs      ->CloneTree();
  picoAODLumiBlocks = lumiBlocks->CloneTree();
}

void analysis::storePicoAOD(){
  picoAODFile->Write();
  picoAODFile->Close();
}

void analysis::monitor(long int e){
  //Monitor progress
  percent        = (e+1)*100/nEvents;
  duration       = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  eventRate      = (e+1)/duration;
  timeRemaining  = (nEvents-e)/eventRate;
  minutes = static_cast<int>(timeRemaining/60);
  seconds = static_cast<int>(timeRemaining - minutes*60);
  getrusage(who, &usage);
  usageMB = usage.ru_maxrss/1024;
  //print status and flush stdout so that status bar only uses one line
  if(isMC){
    fprintf(stdout, "\rProcessed: %8li of %li ( %2li%% | %.0f events/s | done in %02i:%02i | memory usage: %li MB)       ", 
	                          e+1, nEvents, percent,   eventRate,    minutes, seconds,                usageMB);
  }else{
    fprintf(stdout, "\rProcessed: %8li of %li ( %2li%% | %.0f events/s | done in %02i:%02i | memory usage: %li MB | LumiBlocks %i | Est. Lumi %.1f/fb )       ", 
 	                          e+1, nEvents, percent,   eventRate,    minutes, seconds,                usageMB,            nls,         intLumi/1000 );    
  }
  fflush(stdout);
}

int analysis::eventLoop(int maxEvents){

  //Set Number of events to process. Take manual maxEvents if maxEvents is > 0 and less than the total number of events in the input files. 
  nEvents = (maxEvents > 0 && maxEvents < treeEvents) ? maxEvents : treeEvents;
  
  std::cout << "\nProcess " << nEvents << " of " << treeEvents << " events.\n";

  start = std::clock();
  for(long int e = 0; e < nEvents; e++){

    event->update(e);
    processEvent();
    if(debug) event->dump();

    //periodically update status
    if( (e+1)%1000 == 0 || e+1==nEvents || debug) 
      monitor(e);

  }

  std::cout << std::endl;
  if(!isMC) std::cout << "Runs " << firstRun << "-" << lastRun << std::endl;

  minutes = static_cast<int>(duration/60);
  seconds = static_cast<int>(duration - minutes*60);
                                        
  fprintf(stdout,"---------------------------\nFinished eventLoop in %02i:%02i\n\n", minutes, seconds);
  return 0;
}

int analysis::processEvent(){
  if(isMC){
    event->weight = event->genWeight * (lumi * xs * kFactor / mcEventSumw);
  }
  cutflow->Fill(event, "all", true);

  //
  //if we are processing data, first apply lumiMask and trigger
  //
  if(!isMC){
    if(!passLumiMask()){
      if(debug) std::cout << "Fail lumiMask" << std::endl;
      return 0;
    }
    cutflow->Fill(event, "lumiMask", true);

    //keep track of total lumi
    countLumi();

    if(!event->passHLT){
      if(debug) std::cout << "Fail HLT: data" << std::endl;
      return 0;
    }
    cutflow->Fill(event, "HLT", true);
  }
  if(allEvents != NULL) allEvents->Fill(event);

  //
  // Preselection
  // 
  //bool jetMultiplicity = (event->selJets.size() >= 4);
  bool jetMultiplicity = (event->selJets.size() == 4);
  if(!jetMultiplicity){
    if(debug) std::cout << "Fail Jet Multiplicity" << std::endl;
    //event->dump();
    return 0;
  }
  cutflow->Fill(event, "jetMultiplicity", true);

  bool bTags = (event->threeTag || event->fourTag);
  if(!bTags){
    if(debug) std::cout << "Fail b-tag " << std::endl;
    return 0;
  }
  cutflow->Fill(event, "bTags");

  //
  // if event passes basic cuts start doing higher level constructions
  //
  event->chooseCanJets(); // Pick the jets for use in boson candidate construction
  event->buildViews(); // Build all possible diboson candidate pairings "views"
  // build trijet top quark candidates

  if(passPreSel != NULL) passPreSel->Fill(event, event->views);
  
  // Fill picoAOD
  if(writePicoAOD) picoAODEvents->Fill();


  //
  // Event View Requirements: Mass Dependent Requirements (MDRs) on event views
  //
  event->applyMDRs();
  if(!event->passMDRs){
    if(debug) std::cout << "Fail MDRs" << std::endl;
    return 0;
  }
  cutflow->Fill(event, "MDRs");

  if(passMDRs != NULL && event->passHLT) passMDRs->Fill(event, event->views);


  //
  // Event View Cuts: Mass Dependent Cuts (MDCs) on event view variables
  //
  if(!event->views[0]->passMDCs){
    if(debug) std::cout << "Fail MDCs" << std::endl;
    return 0;
  }
  cutflow->Fill(event, "MDCs");

  if(passMDCs != NULL && event->passHLT) passMDCs->Fill(event, event->views);


  if(!event->views[0]->passDEtaBB){
    if(debug) std::cout << "Fail dEtaBB" << std::endl;
    return 0;
  }
  cutflow->Fill(event, "dEtaBB");
  
  if(passDEtaBB != NULL && event->passHLT) passDEtaBB->Fill(event, event->views);

  return 0;
}

bool analysis::passLumiMask(){
  // if the lumiMask is empty, then no JSON file was provided so all
  // events should pass
  if(lumiMask.empty()) return true;


  //make lumiID run:lumiBlock
  edm::LuminosityBlockID lumiID(event->run, event->lumiBlock);

  //define function that checks if a lumiID is contained in a lumiBlockRange
  bool (*funcPtr) (edm::LuminosityBlockRange const &, edm::LuminosityBlockID const &) = &edm::contains;

  //Loop over the lumiMask and use funcPtr to check for a match
  std::vector< edm::LuminosityBlockRange >::const_iterator iter = std::find_if (lumiMask.begin(), lumiMask.end(), boost::bind(funcPtr, _1, lumiID) );

  return lumiMask.end() != iter; 
}

void analysis::getLumiData(std::string fileName){
  std::cout << "Getting integrated luminosity estimate per lumiBlock from: " << fileName << std::endl;
  brilCSV brilFile(fileName);
  lumiData = brilFile.GetData();
}

void analysis::countLumi(){
  if(event->lumiBlock != prevLumiBlock || event->run != prevRun){
    if(event->run != prevRun){
      if(event->run < firstRun) firstRun = event->run;
      if(event->run >  lastRun)  lastRun = event->run;
    }
    prevLumiBlock = event->lumiBlock;
    prevRun       = event->run;
    edm::LuminosityBlockID lumiID(event->run, event->lumiBlock);
    intLumi += lumiData[lumiID];//convert units to /fb
    //std::cout << lumiID << " " << lumiData[lumiID] << " " << intLumi << " \n";
    nls   += 1;
    nruns += 1;
  }
  return;
}

analysis::~analysis(){} 

