#include "ZZ4b/NtupleAna/interface/viewHists.h"

using namespace NtupleAna;

viewHists::viewHists(std::string name, fwlite::TFileService& fs) {
  dir = fs.mkdir(name);

  //
  // Object Level
  //
  //canJets = new jetHists(name+"/canJets", fs, "Boson Candidate Jets");
    
  //
  // Event  Level
  //
  dHH = dir.make<TH1F>("dHH", (name+"/dHH; D_{HH}; Entries").c_str(), 40, 0, 200);
  xZZ = dir.make<TH1F>("xZZ", (name+"/xZZ; X_{ZZ}; Entries").c_str(), 50, 0, 5);
  mZZ = dir.make<TH1F>("mZZ", (name+"/mZZ; m_{ZZ} [GeV]; Entries").c_str(), 130, 100, 1400);
  mZH = dir.make<TH1F>("mZH", (name+"/mZH; m_{ZH} [GeV]; Entries").c_str(), 130, 100, 1400);
} 

void viewHists::Fill(eventData* event, eventView &view){
  //
  // Object Level
  //

  //
  // Event Level
  //
  dHH->Fill(view.dHH, event->weight);
  xZZ->Fill(view.xZZ, event->weight);
  mZZ->Fill(view.mZZ, event->weight);
  mZH->Fill(view.mZH, event->weight);

  return;
}

viewHists::~viewHists(){} 

