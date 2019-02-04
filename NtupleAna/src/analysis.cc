#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <TROOT.h>

#include "ZZ4b/NtupleAna/interface/analysis.h"

using namespace NtupleAna;

analysis::analysis(TChain* e, TChain* r, fwlite::TFileService& fs, bool mc, bool d){
  std::cout<<"In analysis constructor"<<std::endl;
  debug      = d;
  isMC       = mc;
  events     = e;
  events->SetBranchStatus("*", 0);
  runs       = r;
  runs->SetBranchStatus("*", 0);
  initBranch(runs, "genEventCount", &genEventCount);
  initBranch(runs, "genEventSumw",  &genEventSumw);
  initBranch(runs, "genEventSumw2", &genEventSumw2);
  runs->GetEntry(0);
  event      = new eventData(events, debug);
  treeEvents = events->GetEntries();
  cutflow    = new cutflowHists("cutflow", fs);

  // hists
  allEvents    = new eventHists("allEvents",  fs);
  passPreSel   = new eventHists("passPreSel", fs, true);
} 

void analysis::createPicoAOD(std::string fileName){
  writePicoAOD = true;
  picoAODFile = TFile::Open(fileName.c_str() , "RECREATE");
  picoAODEvents = events->CloneTree(0);
  picoAODRuns   = runs  ->CloneTree();
}

void analysis::storePicoAOD(){
  picoAODFile->Write();
  picoAODFile->Close();
}

int analysis::eventLoop(int maxEvents){
  std::cout << " In eventLoop" << std::endl;
  nEvents = (maxEvents > 0 && maxEvents < treeEvents) ? maxEvents : treeEvents;
  
  std::cout << "Number of input events: " << treeEvents << std::endl;
  std::cout << "Will process " << nEvents << " events." << std::endl;

  std::clock_t start = std::clock();
  double duration;
  double eventRate;
  double timeRemaining;
  for(long int e = 0; e < nEvents; e++){

    event->update(e);
    processEvent();
    if(debug) event->dump();

    if( (e+1)%1000 == 0 || e+1==nEvents || debug){
      duration       = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
      eventRate      = (e+1)/duration;
      timeRemaining  = (nEvents-e)/eventRate;
      fprintf(stdout, "\r  Processed: %8li of %li (%2li%%, %.0f events/s, done in %.0fs)       ", e+1, nEvents, (e+1)*100/nEvents, eventRate, timeRemaining);
      fflush(stdout);
    }
  }

  std::cout<<std::endl<<"Exit eventLoop"<<std::endl;
  return 0;
}

int analysis::processEvent(){
  //     #initialize event and do truth level stuff before moving to reco (actual data analysis) stuff
  event->weight = lumi * kFactor * event->genWeight / genEventCount;

  cutflow->Fill("all", event->weight);
  allEvents->Fill(event);

  //
  // Preselection
  // 
  bool jetMultiplicity = (event->selJets.size() >= 4);
  if(!jetMultiplicity){
    if(debug) std::cout << "Fail Jet Multiplicity" << std::endl;
    //event->dump();
    return 0;
  }
  cutflow->Fill("jetMultiplicity", event->weight);

  bool bTags = (event->tagJets.size() >= 4);
  if(!bTags){
    if(debug) std::cout << "Fail b-tag " << std::endl;
    return 0;
  }
  cutflow->Fill("bTags", event->weight);

  //
  // if event passes basic cuts start doing higher level constructions
  //
  event->chooseCanJets(); // Pick the jets for use in boson candidate construction
  event->buildViews(); // Build all possible diboson candidate pairings "views"

  passPreSel->Fill(event, event->views);
  
  // Fill picoAOD
  if(writePicoAOD) picoAODEvents->Fill();

    //     self.thisEvent.buildTops(self.thisEvent.recoJets, [])
    //     self.passPreSel.Fill(self.thisEvent, self.thisEvent.weight)

    //     self.thisEvent.applyMDRs()
    //     if not self.thisEvent.views:
    //         if self.debug: print( "No Views Pass MDRs" )
    //         return
    //     self.cutflow.Fill("MDRs", self.thisEvent.weight)
    //     self.passMDRs.Fill(self.thisEvent, self.thisEvent.weight)

    //     if not self.thisEvent.views[0].passMDCs:
    //         if self.debug: print( "Fail MDCs" )
    //         return
    //     self.cutflow.Fill("MDCs", self.thisEvent.weight)
    //     self.passMDCs.Fill(self.thisEvent, self.thisEvent.weight)
        
    //     if not self.thisEvent.views[0].passHCdEta:
    //         if self.debug: print( "Fail HC dEta" )
    //         return
    //     self.cutflow.Fill("HCdEta", self.thisEvent.weight)
    //     self.passHCdEta.Fill(self.thisEvent, self.thisEvent.weight)
        
    //     if not self.thisEvent.passTopVeto:
    //         if self.debug: print( "Fail top veto" )
    //         return
    //     self.cutflow.Fill("topVeto", self.thisEvent.weight)
    //     self.passTopVeto.Fill(self.thisEvent, self.thisEvent.weight)
        
    //     if not self.thisEvent.views[0].ZZ:
    //         if self.debug: print( "Fail xZZ =",self.thisEvent.views[0].xZZ )
    //         return
    //     self.cutflow.Fill("xZZ", self.thisEvent.weight)
  return 0;
}

analysis::~analysis(){} 

