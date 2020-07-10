#include <iostream>
#include <iomanip>
#include <cstdio>
#include <TROOT.h>
#include <boost/bind.hpp>

#include "ZZ4b/nTupleAnalysis/interface/mixedAnalysis.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"


using namespace nTupleAnalysis;
using std::cout; using std::endl; 


mixedAnalysis::mixedAnalysis(TChain* _events, TChain* _runs, TChain* _lumiBlocks, fwlite::TFileService& fs, bool _debug){
  if(_debug) cout<<"In mixedAnalysis constructor"<<endl;
  debug      = _debug;
  events     = _events;
  events->SetBranchStatus("*", 0);


  // Turn on hemi branches

  runs       = _runs;
  lumiBlocks = _lumiBlocks;
  event      = new mixedEventData(events, debug);
  treeEvents = events->GetEntries();

  //dir = fs.mkdir();
  hHemiUsage        = fs.make<TH1F>("hemiUsage",       std::string("hemiUsage; Number of times hemi used; Entries").c_str(),  50,-0.5,49.5);
  hHemiPairUsage    = fs.make<TH1F>("hemiPairUsage",   std::string("hemiPairUsage; Number of times hemi pair used; Entries").c_str(),  50,-0.5,49.5);
  hHemiUsage_SR     = fs.make<TH1F>("hemiUsage_SR",    std::string("hemiUsage_SR; Number of times hemi used; Entries").c_str(),  50,-0.5,49.5);
  hHemiPairUsage_SR = fs.make<TH1F>("hemiPairUsage_SR",std::string("hemiPairUsage_SR; Number of times hemi pair used; Entries").c_str(),  50,-0.5,49.5);
}



int mixedAnalysis::eventLoop(int maxEvents, long int firstEvent){

  //Set Number of events to process. Take manual maxEvents if maxEvents is > 0 and less than the total number of events in the input files. 
  nEvents = (maxEvents > 0 && maxEvents < treeEvents) ? maxEvents : treeEvents;
  
  cout << "\nProcess " << (nEvents - firstEvent) << " of " << treeEvents << " events.\n";
  if(firstEvent)
    cout << " \t... starting with  " <<  firstEvent << " \n";

  start = std::clock();//2546000 //2546043
  for(long int e = firstEvent; e < nEvents; e++){

    event->update(e);    
    if(debug) cout << "processing event " << endl;    
    processEvent();
    if(debug) cout << "Done processing event " << endl;    
    if(debug) event->dump();
    if(debug) cout << "done " << endl;    

    //periodically update status
    if( (e+1)%10000 == 0 || e+1==nEvents || debug) 
      monitor(e);
    if(debug) cout << "done loop " << endl;    

  }
  cout << endl;

  eventRate = (nEvents)/duration;

  hours   = static_cast<int>( duration/3600 );
  minutes = static_cast<int>( duration/60   )%60;
  seconds = static_cast<int>( duration      )%60;
                                 
  fprintf(stdout,"---------------------------\nProcessed  %9li events in %02i:%02i:%02i (%5.0f events/s | %5.2f/fb)", nEvents, hours, minutes, seconds, eventRate, intLumi/1000);

  cout << " Analyzing hemispheres " << endl;

  std::multimap<unsigned int, HemiEventID> allHemisCounts_sorted = invert<HemiEventID>(allHemisCounts);

  for(const auto& mapItr: allHemisCounts_sorted){
    hHemiUsage->Fill(mapItr.first);
  }


  std::multimap<unsigned int, HemiEventID> allHemisCounts_sorted_SR = invert<HemiEventID>(allHemisCounts_SR);

  for(const auto& mapItr: allHemisCounts_sorted_SR){
    hHemiUsage_SR->Fill(mapItr.first);
  }


  std::multimap<unsigned int, HemiPairEventID> allHemiPairsCounts_sorted = invert<HemiPairEventID>(allHemiPairsCounts);

  for(const auto& mapItr: allHemiPairsCounts_sorted){
    hHemiPairUsage->Fill(mapItr.first);
  }


  std::multimap<unsigned int, HemiPairEventID> allHemiPairsCounts_sorted_SR = invert<HemiPairEventID>(allHemiPairsCounts_SR);

  for(const auto& mapItr: allHemiPairsCounts_sorted_SR){
    hHemiPairUsage_SR->Fill(mapItr.first);
  }



  return 0;
}

int mixedAnalysis::processEvent(){
  if(debug){
    cout << "Event: " << event->event << " Run: " << event->run << " fourTag " << event->fourTag << " (SB/CR/SR) " << event->SB << " / " << event->CR << " / " << event->SR  << endl;
    cout << "\t: H1: (r/e) " << event->h1_run << " / " << event->h1_event << " H2: (r/e) " << event->h2_run << " / " << event->h2_event << endl;
  }

  HemiEventID h1(event->h1_run, event->h1_event);
  HemiEventID h2(event->h2_run, event->h2_event);

  if(allHemisCounts.find(h1) == allHemisCounts.end())
    allHemisCounts.insert(std::make_pair(h1,0));

  allHemisCounts.at(h1) += 1;

  if(allHemisCounts.find(h2) == allHemisCounts.end())
    allHemisCounts.insert(std::make_pair(h2,0));

  allHemisCounts.at(h2) += 1;


  if(event->SR){
    if(allHemisCounts_SR.find(h1) == allHemisCounts_SR.end())
      allHemisCounts_SR.insert(std::make_pair(h1,0));

    allHemisCounts_SR.at(h1) += 1;


    if(allHemisCounts_SR.find(h2) == allHemisCounts_SR.end())
      allHemisCounts_SR.insert(std::make_pair(h2,0));

    allHemisCounts_SR.at(h2) += 1;

  }




  //
  //  smaller runNumber first
  //  if runNumbers the same smaller eventNumber first
  //
  HemiPairEventID mID(h1, h2);
  if(h2.first < h1.first)
    mID = HemiPairEventID(h2, h1);
  if((h1.first == h2.first) && (h2.second < h1.second))
    mID = HemiPairEventID(h2, h1);

  if(allHemiPairsCounts.find(mID) == allHemiPairsCounts.end())
    allHemiPairsCounts.insert(std::make_pair(mID,0));

  allHemiPairsCounts.at(mID) += 1;  


  if(event->SR){
    if(allHemiPairsCounts_SR.find(mID) == allHemiPairsCounts_SR.end())
      allHemiPairsCounts_SR.insert(std::make_pair(mID,0));

    allHemiPairsCounts_SR.at(mID) += 1;  
  }

    

  return 0;
}


mixedAnalysis::~mixedAnalysis(){} 


void mixedAnalysis::monitor(long int e){
  //Monitor progress
  percent        = (e+1)*100/nEvents;
  duration       = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  eventRate      = (e+1)/duration;
  timeRemaining  = (nEvents-e)/eventRate;
  hours   = static_cast<int>( timeRemaining/3600 );
  minutes = static_cast<int>( timeRemaining/60   )%60;
  seconds = static_cast<int>( timeRemaining      )%60;
  getrusage(who, &usage);
  usageMB = usage.ru_maxrss/1024;
  //print status and flush stdout so that status bar only uses one line
  fprintf(stdout, "\rProcessed: %9li of %9li ( %2li%% | %5.0f events/s | done in %02i:%02i:%02i | memory usage: %li MB | LumiBlocks %5i | Est. Lumi %5.2f/fb )       ", 
	                          e+1, nEvents, percent,     eventRate,        hours, minutes, seconds,          usageMB,             nls,          intLumi/1000 );    
  fflush(stdout);
}


