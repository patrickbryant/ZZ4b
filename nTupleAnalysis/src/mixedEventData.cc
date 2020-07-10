#include "ZZ4b/nTupleAnalysis/interface/mixedEventData.h"

using namespace nTupleAnalysis;

using std::cout; using std::endl; 
using std::vector;

mixedEventData::mixedEventData(TChain* t, bool d){
  std::cout << "mixedEventData::mixedEventData()" << std::endl;
  tree  = t;
  debug = d;

  //std::cout << "mixedEventData::mixedEventData() tree->Lookup(true)" << std::endl;
  //tree->Lookup(true);
  std::cout << "mixedEventData::mixedEventData() tree->LoadTree(0)" << std::endl;
  tree->LoadTree(0);
  inputBranch(tree, "run",             run);
  inputBranch(tree, "luminosityBlock", lumiBlock);
  inputBranch(tree, "event",           event);

  inputBranch(tree, "fourTag",           fourTag);
  inputBranch(tree, "SB",                SB);
  inputBranch(tree, "CR",                CR);
  inputBranch(tree, "SR",                SR);

  inputBranch(tree, "h1_run",                h1_run);
  inputBranch(tree, "h1_event",              h1_event);

  inputBranch(tree, "h2_run",                h2_run);
  inputBranch(tree, "h2_event",              h2_event);

}






void mixedEventData::update(long int e){

  if(debug){
    std::cout<<"Get Entry "<<e<<std::endl;
    std::cout<<tree->GetCurrentFile()->GetName()<<std::endl;
    tree->Show(e);
  }

  Long64_t loadStatus = tree->LoadTree(e);
  if(loadStatus<0){
   std::cout << "Error "<<loadStatus<<" getting event "<<e<<std::endl; 
   return;
  }

  tree->GetEntry(e);
  if(debug) std::cout<<"Got Entry "<<e<<std::endl;


  if(debug) std::cout<<"mixedEventData updated\n";
  return;
}


void mixedEventData::dump(){
  std::cout << "   Run: " << run    << std::endl;
  std::cout << " Event: " << event  << std::endl;  
  return;
}

mixedEventData::~mixedEventData(){} 

