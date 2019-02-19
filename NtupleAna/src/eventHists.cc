#include "ZZ4b/NtupleAna/interface/eventHists.h"

using namespace NtupleAna;

eventHists::eventHists(std::string name, fwlite::TFileService& fs, bool _doViews, bool blind) {
  doViews = _doViews;
  dir = fs.mkdir(name);

  //
  // Object Level
  //
  nAllJets = dir.make<TH1F>("nAllJets", (name+"/nAllJets; Number of Jets (no selection); Entries").c_str(),  11,-0.5,10.5);
  nSelJets = dir.make<TH1F>("nSelJets", (name+"/nSelJets; Number of Selected Jets; Entries").c_str(),  11,-0.5,10.5);
  nTagJets = dir.make<TH1F>("nTagJets", (name+"/nTagJets; Number of Tagged Jets; Entries").c_str(),  11,-0.5,10.5);
  nCanJets = dir.make<TH1F>("nCanJets", (name+"/nCanJets; Number of Boson Candidate Jets; Entries").c_str(),  11,-0.5,10.5);
  allJets = new jetHists(name+"/allJets", fs, "All Jets");
  selJets = new jetHists(name+"/selJets", fs, "Selected Jets");
  tagJets = new jetHists(name+"/tagJets", fs, "Tagged Jets");
  canJets = new jetHists(name+"/canJets", fs, "Boson Candidate Jets");
    
  nAllMuons = dir.make<TH1F>("nAllMuons", (name+"/nAllMuons; Number of Muons (no selection); Entries").c_str(),  6,-0.5,5.5);
  nIsoMuons = dir.make<TH1F>("nIsoMuons", (name+"/nIsoMuons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
  allMuons = new muonHists(name+"/allMuons", fs, "All Muons");
  isoMuons = new muonHists(name+"/isoMuons", fs, "Prompt Muons");

  //
  // Event  Level
  //
  v4j = new vecHists(name+"/v4j", fs, "4j");
  if(doViews){
    allViews = new massRegionHists(name+"/allViews", fs, blind);
    mainView = new massRegionHists(name+"/mainView", fs, blind);
  }
} 

void eventHists::Fill(eventData* event){
  //
  // Object Level
  //
  nAllJets->Fill(event->allJets.size(), event->weight);
  nSelJets->Fill(event->selJets.size(), event->weight);
  nTagJets->Fill(event->tagJets.size(), event->weight);
  nCanJets->Fill(event->canJets.size(), event->weight);
  for(auto &jet: event->allJets) allJets->Fill(jet, event->weight);
  for(auto &jet: event->selJets) selJets->Fill(jet, event->weight);
  for(auto &jet: event->tagJets) tagJets->Fill(jet, event->weight);
  for(auto &jet: event->canJets) canJets->Fill(jet, event->weight);

  nAllMuons->Fill(event->allMuons.size(), event->weight);
  nIsoMuons->Fill(event->isoMuons.size(), event->weight);
  for(auto &muon: event->allMuons) allMuons->Fill(muon, event->weight);
  for(auto &muon: event->isoMuons) isoMuons->Fill(muon, event->weight);

  //
  // Event Level
  //
  v4j->Fill(event->p4j, event->weight);

  return;
}

void eventHists::Fill(eventData* event, std::vector<eventView*> &views){
  // Basic Fill
  this->Fill(event);

  // View Fills
  mainView->Fill(event, views[0]);
  for(auto &view: views) 
    allViews->Fill(event, view);

  return;
}

eventHists::~eventHists(){} 


/*
from hists import *
from massRegionHists import *
from truthHists import *
from particleHists import *

class eventHists:
    def __init__(self, outFile, directory, truth=False):
        self.thisDir = outFile.GetDirectory(directory)
        if '<ROOT.TDirectory object at 0x0>' == str(self.thisDir):
            self.thisDir = outFile.mkdir(directory)

        self.m4j = makeTH1F(self.thisDir, "m4j", directory+"_m4j; m_{4j} [GeV]; Entries", 220, 100, 1200)
        self.xWt = makeTH1F(self.thisDir, "xWt", directory+"_xWt; x_{Wt}; Entries", 50, 0 , 5)

        self.m4j_vs_nViews = makeTH2F(self.thisDir, "m4j_vs_nViews",
                                      directory+"_m4j_vs_nViews; m_{4j} [GeV]; # of event views; Entries",
                                      110,100,1200, 3,0.5,3.5  )
        
        outFile.mkdir(directory+"/allViews")
        self.allViews = massRegionHists(outFile, directory+"/allViews")
        outFile.mkdir(directory+"/mainView")
        self.mainView = massRegionHists(outFile, directory+"/mainView")

        self.truth = None
        if truth:
            outFile.mkdir(directory+"/truth")
            self.truth = truthHists(outFile, directory+"/truth")

    def Fill(self, event, weight=1, view=None):
        self.m4j.Fill(event.m4j, weight)
        self.xWt.Fill(event.xWt, weight)

        self.m4j_vs_nViews.Fill(event.m4j, len(event.views), weight)

        if type(view) == list:
            for v in view:
                self.allViews.Fill(v, weight, event)
            self.mainView.Fill(view[0], weight, event)
        elif view:
            for v in event.views:
                self.allViews.Fill(v, weight, event)
            self.mainView.Fill(view, weight, event)
        else:
            for v in event.views:
                self.allViews.Fill(v, weight, event)
            self.mainView.Fill(event.views[0], weight, event)

        if self.truth:
            self.truth.Fill(event, weight)
            
    def Write(self, outFile=None):
        self.thisDir.cd()
        self.m4j.Write()
        self.xWt.Write()

        self.m4j_vs_nViews.Write()

        self.allViews.Write()
        self.mainView.Write()

        if self.truth:
            self.truth.Write()
*/
