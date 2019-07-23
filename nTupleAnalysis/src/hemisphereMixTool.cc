
#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"

using namespace nTupleAnalysis;
using std::vector; using std::endl; using std::cout;


void hemisphere::write(hemisphereMixTool* hMixTool, int localPairIndex){

  hMixTool->clearBranches();

  hMixTool->m_Run   = Run;
  hMixTool->m_Event = Event;
  hMixTool->m_regionIdx = regionIdx;
  hMixTool->m_tAxis_x = thrustAxis.X();
  hMixTool->m_tAxis_y = thrustAxis.Y();
  hMixTool->m_sumPz        = sumPz;
  hMixTool->m_sumPt_T 	   = sumPt_T;
  hMixTool->m_sumPt_Ta	   = sumPt_Ta;
  hMixTool->m_combinedMass = combinedMass;
  hMixTool->m_NJets = (tagJets.size() + nonTagJets.size());
  hMixTool->m_NBJets = tagJets.size();
  
  if(hMixTool->m_debug) cout << " \t pairIdx " << hMixTool->hemiTree->GetEntries() + localPairIndex << endl;
  hMixTool->m_pairIdx = (hMixTool->hemiTree->GetEntries() + localPairIndex);
  

  for(const jetPtr& tagJet : tagJets){
    hMixTool->m_jet_pt        ->push_back(tagJet->pt);
    hMixTool->m_jet_eta       ->push_back(tagJet->eta);  
    hMixTool->m_jet_phi       ->push_back(tagJet->phi);  
    hMixTool->m_jet_m         ->push_back(tagJet->m);  
    hMixTool->m_jet_e         ->push_back(tagJet->e);  
    hMixTool->m_jet_bRegCorr  ->push_back(tagJet->bRegCorr);  
    hMixTool->m_jet_deepB     ->push_back(tagJet->deepB);  
    hMixTool->m_jet_CSVv2     ->push_back(tagJet->CSVv2);  
    hMixTool->m_jet_deepFlavB ->push_back(tagJet->deepFlavB);  
    hMixTool->m_jet_isTag     ->push_back(true);

  }

  for(const jetPtr& nonTagJet : nonTagJets){
    hMixTool->m_jet_pt        ->push_back(nonTagJet->pt);
    hMixTool->m_jet_eta       ->push_back(nonTagJet->eta);  
    hMixTool->m_jet_phi       ->push_back(nonTagJet->phi);  
    hMixTool->m_jet_m         ->push_back(nonTagJet->m);  
    hMixTool->m_jet_e         ->push_back(nonTagJet->e);  
    hMixTool->m_jet_bRegCorr  ->push_back(nonTagJet->bRegCorr);  
    hMixTool->m_jet_deepB     ->push_back(nonTagJet->deepB);  
    hMixTool->m_jet_CSVv2     ->push_back(nonTagJet->CSVv2);  
    hMixTool->m_jet_deepFlavB ->push_back(nonTagJet->deepFlavB);  
    hMixTool->m_jet_isTag     ->push_back(false);
  }


  hMixTool->hemiTree->Fill();

}


hemisphereMixTool::hemisphereMixTool(std::string name, std::string fileName, bool fCreateLibrary, fwlite::TFileService& fs, bool debug) {
  if(m_debug) cout << " hemisphereMixTool::In hemisphereMixTool " << name << endl;  
  m_name = name;
  m_debug = debug;
  
  createLibrary = fCreateLibrary;
  if(createLibrary){
    hemiFile = TFile::Open((fileName+"_"+name+".root").c_str() , "RECREATE");
    hemiTree = new TTree("hemiTree","Tree for hemishpere mixing");
  }else{
    hemiFile = TFile::Open((fileName).c_str() , "READ");
    if(m_debug) cout << " hemisphereMixTool::Read file " << hemiFile << endl;
    hemiTree = (TTree*)hemiFile->Get("hemiTree");
    if(m_debug) cout << " hemisphereMixTool::Got Tree " << hemiTree << endl;

  }

  initBranches();

  connectBranch<UInt_t>     (hemiTree, "runNumber",   &m_Run, "i");
  connectBranch<ULong64_t>  (hemiTree, "evtNumber",   &m_Event, "l");
  connectBranch<float>      (hemiTree, "tAxis_x",     &m_tAxis_x, "F");
  connectBranch<float>      (hemiTree, "tAxis_y",     &m_tAxis_y, "F");
  connectBranch<float>      (hemiTree, "sumPz",       &m_sumPz         , "F");
  connectBranch<float>      (hemiTree, "sumPt_T",     &m_sumPt_T      , "F");
  connectBranch<float>      (hemiTree, "sumPt_Ta",    &m_sumPt_Ta    , "F");
  connectBranch<float>      (hemiTree, "combinedMass",&m_combinedMass  , "F");
  connectBranch<UInt_t>     (hemiTree, "NJets",       &m_NJets, "i");  
  connectBranch<UInt_t>     (hemiTree, "NBJets",      &m_NBJets, "i");  
  connectBranch<UInt_t>     (hemiTree, "pairIdx",     &m_pairIdx, "i");  
  connectBranch<UInt_t>     (hemiTree, "regionIdx",   &m_regionIdx, "i");  


  connectVecBranch<float>(hemiTree,  "jet_pt",   &m_jet_pt);
  connectVecBranch<float>(hemiTree,  "jet_eta",  &m_jet_eta);
  connectVecBranch<float>(hemiTree,  "jet_phi",  &m_jet_phi);
  connectVecBranch<float>(hemiTree,  "jet_m",    &m_jet_m);
  connectVecBranch<float>(hemiTree,  "jet_e",    &m_jet_e);
  connectVecBranch<float>(hemiTree,  "jet_bRegCorr",    &m_jet_bRegCorr);
  connectVecBranch<float>(hemiTree,  "jet_deepB",    &m_jet_deepB);
  connectVecBranch<float>(hemiTree,  "jet_CSVv2",    &m_jet_CSVv2);
  connectVecBranch<float>(hemiTree,  "jet_deepFlavB",    &m_jet_deepFlavB);
  connectVecBranch<Bool_t>(hemiTree, "jet_isTag",    &m_jet_isTag);

  //
  // Create Histograms
  //
  dir = fs.mkdir("hMix_"+name);
  hNJets  = dir.make<TH1F>("hNJets",  (name+"/NJets;  ;Entries").c_str(),  10,-0.5,9.5);  
  hNBJets = dir.make<TH1F>("hNBJets", (name+"/NBJets; ;Entries").c_str(),  10,-0.5,9.5);  
  hPz     = dir.make<TH1F>("hPz",     (name+"/Pz; ;Entries").c_str(),     100,-1000,1000);  
  hSumPt_T = dir.make<TH1F>("hSumPt_T",     (name+"/SumPt_T; ;Entries").c_str(),     100,0,1000);  
  hSumPt_Ta = dir.make<TH1F>("hSumPt_Ta",     (name+"/SumPt_Ta; ;Entries").c_str(),     100,0,500);  
  hCombMass = dir.make<TH1F>("hCombMass",     (name+"/CombMass; ;Entries").c_str(),     100,0,500);  

  hdelta_NJets  = dir.make<TH1F>("hdelta_NJets",  (name+"/del_NJets;  ;Entries").c_str(),  19,-9.5,9.5);  
  hdelta_NBJets = dir.make<TH1F>("hdelta_NBJets", (name+"/del_NBJets; ;Entries").c_str(),  19,-9.5,9.5);  
  hdelta_Pz     = dir.make<TH1F>("hdeltaPz",      (name+"/del_Pz; ;Entries").c_str(),  100,-500,500);  
  hdelta_SumPt_T = dir.make<TH1F>("hdeltaSumPt_T",     (name+"/del_SumPt_T; ;Entries").c_str(),     100,-300,300);  
  hdelta_SumPt_Ta = dir.make<TH1F>("hdeltaSumPt_Ta",     (name+"/del_SumPt_Ta; ;Entries").c_str(),     100,-200,200);  
  hdelta_CombMass = dir.make<TH1F>("hdeltaCombMass",     (name+"/del_CombMass; ;Entries").c_str(),     100,-300,300);  

  //
  // json files for Event Displays
  //
  //
  makeEventDisplays = true;
  if(makeEventDisplays){
    eventDisplay = new EventDisplayData("Events");
    eventDisplay->addJetCollection("posHemiJets");
    eventDisplay->addJetCollection("negHemiJets");
    eventDisplay->addEventVar     ("thrustPhi");
  }

  if(!createLibrary)
    makeIndexing();

} 


void hemisphereMixTool::initBranches() {
  m_jet_pt        = new vector<float>();
  m_jet_eta       = new vector<float>();  
  m_jet_phi       = new vector<float>();  
  m_jet_m         = new vector<float>();  
  m_jet_e         = new vector<float>();  
  m_jet_bRegCorr  = new vector<float>();  
  m_jet_deepB     = new vector<float>();  
  m_jet_CSVv2     = new vector<float>();  
  m_jet_deepFlavB = new vector<float>();  
  m_jet_isTag     = new vector<Bool_t>(); 
}


void hemisphereMixTool::clearBranches() {
  m_jet_pt        ->clear();
  m_jet_eta       ->clear();
  m_jet_phi       ->clear();
  m_jet_m         ->clear();
  m_jet_e         ->clear();
  m_jet_bRegCorr  ->clear();
  m_jet_deepB     ->clear();
  m_jet_CSVv2     ->clear();
  m_jet_deepFlavB ->clear();
  m_jet_isTag     ->clear();

}

void hemisphereMixTool::addEvent(eventData* event){

  //
  //  Calculate Thrust Axis
  //
  m_thrustAxis = getThrustAxis(event);

  
  m_regionIdx = ( (event->ZHSB << 0) | (event->ZHCR << 1) | (event->ZHSR << 2) | 
		  (event->ZZSB << 3) | (event->ZZCR << 4) | (event->ZZSR << 5) |
		  (event->SB   << 6) | (event->CR   << 7) | (event->SR   << 8) );


  //
  //  Make Hemispheres
  //
  hemisphere posHemi(event->run, event->event, m_thrustAxis.X(), m_thrustAxis.Y(), m_regionIdx);
  hemisphere negHemi(event->run, event->event, m_thrustAxis.X(), m_thrustAxis.Y(), m_regionIdx);

  for(const jetPtr& thisJet : event->selJets){
    TVector2 thisJetPt2 = TVector2(thisJet->p.Px(),thisJet->p.Py());
    if( (thisJetPt2 * m_thrustAxis ) > 0) posHemi.addJet(thisJet, event->tagJets);
    else                                  negHemi.addJet(thisJet, event->tagJets);
  }


  if(makeEventDisplays){
    for(const jetPtr& jetData : posHemi.tagJets)    eventDisplay->AddJet("posHemiJets",jetData);
    for(const jetPtr& jetData : posHemi.nonTagJets) eventDisplay->AddJet("posHemiJets",jetData);
    for(const jetPtr& jetData : negHemi.tagJets)    eventDisplay->AddJet("negHemiJets",jetData);
    for(const jetPtr& jetData : negHemi.nonTagJets) eventDisplay->AddJet("negHemiJets",jetData);
  }


  //
  //  Fill some histograms
  //
  FillHists(posHemi, negHemi);

  //
  // write to output tree
  //
  if(m_debug) cout << " Write pos hemei " << endl;
  posHemi.write(this,  1);
  if(m_debug) cout << " Write neg hemei " << endl;
  negHemi.write(this, -1);

  if(makeEventDisplays){
    eventDisplay->AddEventVar("thrustPhi", m_thrustAxis.Phi());
    eventDisplay->NewEvent();
  }

  return;
}


TVector2 hemisphereMixTool::getThrustAxis(eventData* event){

  vector<TVector2> jetPts;
  for(const jetPtr& thisJet : event->selJets){
    jetPts.push_back(TVector2(thisJet->p.Px(),thisJet->p.Py()));
  }



  return calcThrust(jetPts);
}


//
// Following is From
// https://rivet.hepforge.org/code/1.8.2/a00677_source.html
//


// Do the full calculation
TVector2 hemisphereMixTool::calcThrust(const vector<TVector2>& jetPts) {

  // Make a vector of the three-momenta in the final state
  //double momentumSum(0.0);
  //for(const TVector2& p2 : jetPts) {
  //  momentumSum += p2.Mod();
  //}
  if(m_debug) cout <<  "Number of particles = " << jetPts.size() << endl;
  
  assert(jetPts.size() > 3);

  // Temporary variables for calcs
  TVector2 axis(0,0);
  double val = 0.;

  // Get thrust
  calcT(jetPts, val, axis);
  //if(m_debug) cout << "Mom sum = " << momentumSum << endl;
  //m_thrust = (val / momentumSum);
  // Make sure that thrust always points along the +ve x-axis.
  if (axis.X() < 0) axis = -1*axis;
  axis = axis.Unit();
  if(m_debug) cout << "Axis = " << axis.X() << " " << axis.Y() << endl;
  return axis;
}

// Do the general case thrust calculation
void hemisphereMixTool::calcT(const vector<TVector2>& momenta, double& t, TVector2& taxis) {
  // This function implements the iterative algorithm as described in the
  // Pythia manual. We take eight (four) different starting vectors
  // constructed from the four (three) leading particles to make sure that
  // we don't find a local maximum.
  vector<TVector2> p = momenta;
  assert(p.size() >= 3);
  unsigned int n = 4;

  vector<TVector2> tvec;
  vector<double> tval;

  for (int i = 0 ; i < 8; ++i) {
    // Create an initial vector from the leading four jets
    TVector2 foo(0,0);
    int sign = i;
    for (unsigned int k = 0 ; k < n ; ++k) {
      (sign % 2) == 1 ? foo += p[k] : foo -= p[k];
      sign /= 2;
    }
    foo=foo.Unit();

    // Iterate
    double diff=999.;
    while (diff>1e-5) {
      TVector2 foobar(0,0);
      for (unsigned int k=0 ; k<p.size() ; k++)
	(foo *p[k])>0 ? foobar+=p[k] : foobar-=p[k];
      diff=(foo-foobar.Unit()).Mod();
      foo=foobar.Unit();
    }

    // Calculate the thrust value for the vector we found
    t=0.;
    for (unsigned int k=0 ; k<p.size() ; k++)
      t+=fabs(foo*p[k]);

    // Store everything
    tval.push_back(t);
    tvec.push_back(foo);
  }

  // Pick the solution with the largest thrust
  t=0.;
  for (unsigned int i=0 ; i<tvec.size() ; i++)
    if (tval[i]>t){
      t=tval[i];
      taxis=tvec[i];
    }
}



hemisphereMixTool::~hemisphereMixTool(){} 

void hemisphereMixTool::FillHists(const hemisphere& posHemi, const hemisphere& negHemi ){
  FillHists(posHemi);
  FillHists(negHemi);

  hdelta_NJets    ->Fill( (posHemi.tagJets.size() + posHemi.nonTagJets.size())  -  (negHemi.tagJets.size() + negHemi.nonTagJets.size()));
  hdelta_NBJets   ->Fill(  posHemi.tagJets.size() -  negHemi.tagJets.size());
  hdelta_Pz       ->Fill( posHemi.sumPz        - negHemi.sumPz         );
  hdelta_SumPt_T  ->Fill( posHemi.sumPt_T      - negHemi.sumPt_T       );
  hdelta_SumPt_Ta ->Fill( posHemi.sumPt_Ta     - negHemi.sumPt_Ta      );
  hdelta_CombMass ->Fill( posHemi.combinedMass - negHemi.combinedMass  );

}


void hemisphereMixTool::FillHists(const hemisphere& hIn){
  hNJets ->Fill((hIn.tagJets.size() + hIn.nonTagJets.size()));
  hNBJets->Fill( hIn.tagJets.size() );
  hPz       ->Fill( hIn.sumPz);
  hSumPt_T  ->Fill( hIn.sumPt_T);
  hSumPt_Ta ->Fill( hIn.sumPt_Ta);
  hCombMass ->Fill( hIn.combinedMass);

}

void hemisphereMixTool::storeLibrary(){
  hemiFile->Write();
  hemiFile->Close();

  if(makeEventDisplays)
    eventDisplay->Write("EventDisplay_"+m_name+".txt");

  return;
}

hemisphere hemisphereMixTool::getHemi(unsigned int entry){
  hemiTree->GetEntry(entry);
  hemisphere outHemi = hemisphere(m_Run, m_Event, m_tAxis_x, m_tAxis_y, m_regionIdx);
  outHemi.sumPz = m_sumPz;
  outHemi.sumPt_T = m_sumPt_T;
  outHemi.sumPt_Ta = m_sumPt_Ta;
  outHemi.combinedMass = m_combinedMass;
  outHemi.NJets = m_NJets;
  outHemi.NBJets = m_NBJets;
  outHemi.pairIdx = m_pairIdx;
  outHemi.regionIdx = m_regionIdx;

  return outHemi;
}

void hemisphereMixTool::makeIndexing(){
  
  //(NJets, NBJets) vector<entries>
  
  unsigned int nHemis = hemiTree->GetEntries();
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    hemisphere thisHemi = getHemi(hemiIdx);
    int thisNJet  = thisHemi.NJets;
    int thisNBJet = thisHemi.NBJets;
    EventID thisEventID = { {thisNJet, thisNBJet} };
    if(m_EventIndex.find(thisEventID) == m_EventIndex.end()){
      cout << "Making index for (" << thisNJet << " / " << thisNBJet << " )" << endl;
      m_EventIndex.insert(std::make_pair(thisEventID, IndexVec()));
    }
    m_EventIndex[thisEventID].push_back(hemiIdx);
  }

}


hemisphereMixTool::IndexVec hemisphereMixTool::getHemiSphereIndices(const hemisphere& hIn){

  EventID thisEventID = { {int(hIn.NJets), int(hIn.NBJets)} };
  if(m_EventIndex.find(thisEventID) != m_EventIndex.end()){
    return m_EventIndex[thisEventID];
  }
  
  return IndexVec();
}
