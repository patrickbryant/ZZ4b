#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>
#include <signal.h>

#include "ZZ4b/nTupleAnalysis/interface/tTbarAnalysis.h"
#include "nTupleAnalysis/baseClasses/interface/helpers.h"

using std::cout;  using std::endl;

using namespace nTupleAnalysis;

tTbarAnalysis::tTbarAnalysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _isMC, std::string _year, std::string histDetailLevel, 
			     bool _debug, 
			     std::string bjetSF, std::string btagVariations,
			     std::string JECSyst, std::string friendFile){
		   
  if(_debug) std::cout<<"In tTbarAnalysis constructor"<<std::endl;
  debug      = _debug;
  isMC       = _isMC;
  year       = _year;
  events     = _events;
  events->SetBranchStatus("*", 0);

  //keep branches needed for JEC Uncertainties
  if(isMC){
    events->SetBranchStatus("nGenJet"  , 1);
    events->SetBranchStatus( "GenJet_*", 1);
  }
  events->SetBranchStatus(   "MET*", 1);
  events->SetBranchStatus("RawMET*", 1);
  events->SetBranchStatus("fixedGridRhoFastjetAll", 1);
  events->SetBranchStatus("Jet_rawFactor", 1);
  events->SetBranchStatus("Jet_area", 1);
  events->SetBranchStatus("Jet_neEmEF", 1);
  events->SetBranchStatus("Jet_chEmEF", 1);

  if(JECSyst!=""){
    std::cout << "events->AddFriend(\"Friends\", "<<friendFile<<")" << " for JEC Systematic " << JECSyst << std::endl;
    events->AddFriend("Friends", friendFile.c_str());
  }

  runs       = _runs;

  //Calculate MC weight denominator
  if(isMC){
    if(debug) runs->Print();
    runs->SetBranchStatus("*", 0);
    Long64_t loadStatus = runs->LoadTree(0);
    if(loadStatus < 0){
      std::cout << "ERROR in loading tree for entry index: " << 0 << "; load status = " << loadStatus << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if(runs->FindBranch("genEventCount")){
      std::cout << "Runs has genEventCount" << std::endl;
      inputBranch(runs, "genEventCount", genEventCount);
      inputBranch(runs, "genEventSumw",  genEventSumw);
      inputBranch(runs, "genEventSumw2", genEventSumw2);
    }else{//for some presumably idiotic reason, NANOAODv6 added an underscore to these branch names...
      std::cout << "Runs has genEventCount_" << std::endl;
      inputBranch(runs, "genEventCount_", genEventCount);
      inputBranch(runs, "genEventSumw_",  genEventSumw);
      inputBranch(runs, "genEventSumw2_", genEventSumw2);      
    }
    for(int r = 0; r < runs->GetEntries(); r++){
      runs->GetEntry(r);
      mcEventCount += genEventCount;
      mcEventSumw  += genEventSumw;
      mcEventSumw2 += genEventSumw2;
    }
    cout << "mcEventCount " << mcEventCount << " | mcEventSumw " << mcEventSumw << endl;
  }

  lumiBlocks = _lumiBlocks;
  event      = new tTbarEventData(events, isMC, year, debug, bjetSF, btagVariations, JECSyst);
  treeEvents = events->GetEntries();
  cutflow    = new tTbarCutFlowHists("cutflow", fs, isMC, debug);
  cutflow->AddCut("lumiMask");
  cutflow->AddCut("HLT");
  cutflow->AddCut("lepMultiplicity");
  cutflow->AddCut("jetMultiplicity");
  cutflow->AddCut("bTags");
  cutflow->AddCut("1LSelection");
  cutflow->AddCut("2LSelection");
  cutflow->AddCut("1OR2LSelection");

  lumiCounts    = new lumiHists("lumiHists", fs, true, debug);
  
  if(nTupleAnalysis::findSubStr(histDetailLevel,"allEvents"))        allEvents           = new tTbarEventHists("allEvents",         fs, isMC, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passPreSel"))       passPreSel          = new tTbarEventHists("passPreSel",        fs, isMC, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passEMuSel"))       passEMuSel          = new tTbarEventHists("passEMuSel",        fs, isMC, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"AllMeT"))           passEMuSelAllMeT    = new tTbarEventHists("passEMuSelAllMeT",  fs, isMC, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"passMuSel"))        passMuSel           = new tTbarEventHists("passMuSel",         fs, isMC, histDetailLevel, debug);
  if(nTupleAnalysis::findSubStr(histDetailLevel,"AllMeT"))           passMuSelAllMeT     = new tTbarEventHists("passMuSelAllMeT",   fs, isMC, histDetailLevel, debug);


  if(allEvents)     std::cout << "Turning on allEvents Hists" << std::endl; 
  if(passPreSel)    std::cout << "Turning on passPreSel Hists" << std::endl; 
  if(passEMuSel)    std::cout << "Turning on passEMuSel Hists" << std::endl; 
  if(passMuSel)     std::cout << "Turning on passMuSel Hists" << std::endl; 
  if(passEMuSelAllMeT)    std::cout << "Turning on passEMuSelAllMeT Hists" << std::endl; 
  if(passMuSelAllMeT)     std::cout << "Turning on passMuSelAllMeT Hists" << std::endl; 
  ////if(passDijetMass) std::cout << "Turning on passDijetMass Hists" << std::endl; 
  //if(passMDRs)      std::cout << "Turning on passMDRs Hists" << std::endl; 
  //if(passSvB)       std::cout << "Turning on passSvB Hists" << std::endl; 
  //if(passMjjOth)    std::cout << "Turning on passMjjOth Hists" << std::endl; 
  //if(failrWbW2)     std::cout << "Turning on failrWbW2 Hists" << std::endl; 
  //if(passMuon)      std::cout << "Turning on passMuon Hists" << std::endl; 
  //if(passDvT05)     std::cout << "Turning on passDvT05 Hists" << std::endl; 


  histFile = &fs.file();
} 




void tTbarAnalysis::createPicoAOD(std::string fileName, bool copyInputPicoAOD){
  writePicoAOD = true;
  picoAODFile = TFile::Open(fileName.c_str() , "RECREATE");
  if(copyInputPicoAOD){
    //We are making a skim so we can directly clone the input TTree
    picoAODEvents = events->CloneTree(0);
  }
  addDerivedQuantitiesToPicoAOD();
  picoAODRuns       = runs      ->CloneTree();
  picoAODLumiBlocks = lumiBlocks->CloneTree();
}



void tTbarAnalysis::picoAODFillEvents(){
  if(debug) std::cout << "analysis::picoAODFillEvents()" << std::endl;
  if(alreadyFilled){
    if(debug) std::cout << "analysis::picoAODFillEvents() alreadyFilled" << std::endl;
    //std::cout << "ERROR: Filling picoAOD with same event twice" << std::endl;
    return;
  }
  alreadyFilled = true;
  //if(m4jPrevious == event->m4j) std::cout << "WARNING: previous event had identical m4j = " << m4jPrevious << std::endl;

  if(debug) std::cout << "picoAODEvents->Fill()" << std::endl;
  picoAODEvents->Fill();  
  if(debug) std::cout << "analysis::picoAODFillEvents() done" << std::endl;
  return;
}



void tTbarAnalysis::addDerivedQuantitiesToPicoAOD(){
  cout << "tTbarAnalysis::addDerivedQuantitiesToPicoAOD()" << endl;

  picoAODEvents->Branch("passHLT", &event->passHLT);
  //picoAODEvents->Branch("passDijetMass", &event->passDijetMass);
  //picoAODEvents->Branch("passXWt", &event->passXWt);
  //picoAODEvents->Branch("xW", &event->xW);
  //picoAODEvents->Branch("xt", &event->xt);
  //picoAODEvents->Branch("xWt", &event->xWt);
  //picoAODEvents->Branch("xbW", &event->xbW);
  //picoAODEvents->Branch("xWbW", &event->xWbW);
  picoAODEvents->Branch("nIsoMuons", &event->nIsoMuons);
  picoAODEvents->Branch("ttbarWeight", &event->ttbarWeight);
  cout << "analysis::addDerivedQuantitiesToPicoAOD() done" << endl;
  return;
}

void tTbarAnalysis::storePicoAOD(){
  picoAODFile->Write();
  picoAODFile->Close();
  return;
}



void tTbarAnalysis::monitor(long int e){
  //Monitor progress
  timeTotal = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  timeElapsed          = timeTotal - previousMonitorTime;
  eventsElapsed        =         e - previousMonitorEvent;
  if( timeElapsed < 1 && e+1!=nEvents) return;
  previousMonitorEvent = e;
  previousMonitorTime  = timeTotal;
  percent              = (e+1)*100/nEvents;
  eventRate            = eventRate ? 0.9*eventRate + 0.1*eventsElapsed/timeElapsed : eventsElapsed/timeElapsed; // Running average with 0.9 momentum
  timeRemaining        = (nEvents-e)/eventRate;
  //eventRate      = (e+1)/timeTotal;
  //timeRemaining  = (nEvents-e)/eventRate;
  hours   = static_cast<int>( timeRemaining/3600 );
  minutes = static_cast<int>( timeRemaining/60   )%60;
  seconds = static_cast<int>( timeRemaining      )%60;
  getrusage(who, &usage);
  usageMB = usage.ru_maxrss/1024;
  //print status and flush stdout so that status bar only uses one line
  if(isMC){
    fprintf(stdout, "\rProcessed: %9li of %9li ( %2li%% | %5.0f events/s | done in %02i:%02i:%02i | memory usage: %li MB)       ", 
	                          e+1, nEvents, percent,     eventRate,        hours, minutes, seconds,          usageMB);
  }else{
    fprintf(stdout, "\rProcessed: %9li of %9li ( %2li%% | %5.0f events/s | done in %02i:%02i:%02i | memory usage: %li MB | LumiBlocks %5i | Est. Lumi %5.2f/fb )       ", 
	                          e+1, nEvents, percent,     eventRate,        hours, minutes, seconds,          usageMB,             nls,          intLumi/1000 );    
  }
  fflush(stdout);
  return;
}

int tTbarAnalysis::eventLoop(int maxEvents, long int firstEvent){

  //Set Number of events to process. Take manual maxEvents if maxEvents is > 0 and less than the total number of events in the input files. 
  nEvents = (maxEvents > 0 && maxEvents < treeEvents) ? maxEvents : treeEvents;
  
  cout << "\nProcess " << (nEvents - firstEvent) << " of " << treeEvents << " events.\n";
  if(firstEvent){
    cout << " \t... starting with  " <<  firstEvent << " \n";
    previousMonitorEvent = firstEvent;
  }

  bool mixedEventWasData = false;

  start = std::clock();
  for(long int e = firstEvent; e < nEvents; e++){
    
    currentEvent = e;

    alreadyFilled = false;
    //m4jPrevious = event->m4j;

    event->update(e);    

    if(writeOutEventNumbers){
      passed_runs  .push_back(event->run);
      passed_events.push_back(event->event);
    }

    if(debug) cout << "processing event " << endl;    
    processEvent();
    if(debug) cout << "Done processing event " << endl;    
    if(debug) event->dump();
    if(debug) cout << "done " << endl;    

    //periodically update status
    monitor(e);
    if(debug) cout << "done loop " << endl;    
  }

  cout << endl;
  if(!isMC) cout << "Runs " << firstRun << "-" << lastRun << endl;

  eventRate = (nEvents)/timeTotal;

  hours   = static_cast<int>( timeTotal/3600 );
  minutes = static_cast<int>( timeTotal/60   )%60;
  seconds = static_cast<int>( timeTotal      )%60;
                                 
  if(isMC){
    fprintf(stdout,"---------------------------\nProcessed  %9li events in %02i:%02i:%02i (%5.0f events/s)",            nEvents, hours, minutes, seconds, eventRate);
  }else{
    fprintf(stdout,"---------------------------\nProcessed  %9li events in %02i:%02i:%02i (%5.0f events/s | %5.2f/fb)", nEvents, hours, minutes, seconds, eventRate, intLumi/1000);
  }
  return 0;
}

int tTbarAnalysis::processEvent(){
  if(debug) cout << "processEvent start" << endl;

  if(isMC){
    event->mcWeight = event->genWeight * (lumi * xs * kFactor / mcEventSumw);
    if(event->nTrueBJets>=4) event->mcWeight *= fourbkfactor;
    event->weight *= event->mcWeight;


    if(debug){
    std::cout << "Event: " << event->event << " Run: " << event->run << std::endl;
    std::cout << "event->genWeight * (lumi * xs * kFactor / mcEventSumw) = " << std::endl;;
      std::cout<< event->genWeight << " * (" << lumi << " * " << xs << " * " << kFactor << " / " << mcEventSumw << ") = " << event->mcWeight << std::endl;
      std::cout<< "\tweight  " << event->weight << std::endl;
      std::cout<< "\tbTagSF  " << event->bTagSF << std::endl;
      std::cout<< "\tfourbkfactor " << fourbkfactor << std::endl;
      std::cout<< "\tnTrueBJets " << event->nTrueBJets << std::endl;
      std::cout<< "\tmcWeight " << event->mcWeight << std::endl;
      }


    //
    //  If using unit MC weights
    //
    if(mcUnitWeight){
      event->mcWeight = 1.0;
      event->weight = 1.0;
    }

  }

  if(debug) cout << "cutflow->Fill(event, all, true)" << endl;
  cutflow->Fill("all", event);

  lumiCounts->Fill(event);


  //
  //if we are processing data, first apply lumiMask and trigger
  //
  if(!isMC){
    if(!passLumiMask()){
      if(debug) cout << "Fail lumiMask" << endl;
      return 0;
    }
    cutflow->Fill("lumiMask", event);

    //keep track of total lumi
    countLumi();

    if( (intLumi - lumiLastWrite) > 500){
      lumiCounts->FillLumiBlock((intLumi - lumiLastWrite));
      lumiLastWrite = intLumi;
    }

    if(!event->passHLT){
      if(debug) cout << "Fail HLT: data" << endl;
      return 0;
      cutflow->Fill("HLT", event);    }
  }else{
    if(currentEvent > 0 && (currentEvent % 10000) == 0) 
      lumiCounts->FillLumiBlock(1.0);
  }

  if(allEvents != NULL && event->passHLT) allEvents->Fill(event);
  //if(allEvents != NULL) allEvents->Fill(event);
  

  //
  // Pre-selection 
  //
  bool lepMultiplicity = ( event->nIsoLeps >= 1);
  bool jetMultiplicity = (event->selJets.size() >= 2);
  bool bTags = (event->twoTag);

  //
  // Preselection
  // 
  if(!lepMultiplicity){
    if(debug) cout << "Fail Lep Multiplicity" << endl;
    return 0;
  }
  cutflow->Fill("lepMultiplicity", event);


  if(!jetMultiplicity){
    if(debug) cout << "Fail Jet Multiplicity" << endl;
    return 0;
  }
  cutflow->Fill("jetMultiplicity", event);

  if(!bTags){
    if(debug) cout << "Fail b-tag " << endl;
    return 0;
  }
  cutflow->Fill("bTags", event);

  //
  //  1L Selection
  //
  bool pass1L = ((event->nIsoLeps == 1) && (event->selJets.size() >= 4));  
  if(pass1L) cutflow->Fill("1LSelection", event);
  
  // 
  //  2L Selection
  //
  bool pass2L = (event->nIsoLeps >= 2);
  if(pass2L) cutflow->Fill("2LSelection", event);

  //
  // OR 
  //
  if( !pass1L && !pass2L){
    if(debug) cout << "Fail Lepton Selection " << endl;
    return 0;
  }
  cutflow->Fill("1OR2LSelection", event);  

  if(passPreSel != NULL && event->passHLT) passPreSel->Fill(event);

  // Fill picoAOD
  if(writePicoAOD){//for regular picoAODs, keep them small by filling after dijetMass cut
    picoAODFillEvents();
  }

  if(pass2L && event->passHLT_2L){
    if(passEMuSelAllMeT != NULL) passEMuSelAllMeT->Fill(event);

    if(event->treeMET->pt > 45)
      if(passEMuSel != NULL) passEMuSel->Fill(event);
  }

  if(pass1L && event->passHLT_1L) {
    if(passMuSelAllMeT != NULL) passMuSelAllMeT ->Fill(event);    

    if(event->treeMET->pt > 45)
      if(passMuSel != NULL) passMuSel->Fill(event);

  }

  return 0;
}

bool tTbarAnalysis::passLumiMask(){
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

void tTbarAnalysis::getLumiData(std::string fileName){
  cout << "Getting integrated luminosity estimate per lumiBlock from: " << fileName << endl;
  brilCSV brilFile(fileName);
  lumiData = brilFile.GetData();
}

void tTbarAnalysis::countLumi(){
  if(event->lumiBlock != prevLumiBlock || event->run != prevRun){
    if(event->run != prevRun){
      if(event->run < firstRun) firstRun = event->run;
      if(event->run >  lastRun)  lastRun = event->run;
    }
    prevLumiBlock = event->lumiBlock;
    prevRun       = event->run;
    edm::LuminosityBlockID lumiID(event->run, event->lumiBlock);
    intLumi += lumiData[lumiID];//convert units to /fb
    if(debug){
      std::cout << lumiID << " " << lumiData[lumiID] << " " << intLumi << " \n";
    }
    nls   += 1;
    nruns += 1;
  }
  return;
}


tTbarAnalysis::~tTbarAnalysis(){
  if(writeOutEventNumbers){
    cout << "Writing out event Numbers" << endl;
    histFile->WriteObject(&passed_events, "passed_events"); 
    histFile->WriteObject(&passed_runs,   "passed_runs"); 
  }
} 

