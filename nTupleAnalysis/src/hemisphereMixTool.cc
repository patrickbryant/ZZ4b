
#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"
#include <boost/algorithm/string.hpp>


using namespace nTupleAnalysis;
using std::vector; using std::endl; using std::cout;


hemiDataHandler::hemiDataHandler(UInt_t nJetBin, UInt_t nBJetBin, bool createLibrary, std::string fileName, std::string name ) : 
  m_nJetBin(nJetBin), m_nBJetBin(nBJetBin), m_createLibrary(createLibrary)
{

  std::stringstream ss;
  ss << m_nJetBin << "_" << m_nBJetBin;
  m_EventIDPostFix = ss.str();


  if(m_createLibrary){
    hemiFile = TFile::Open((fileName+"_"+name+"_"+m_EventIDPostFix+".root").c_str() , "RECREATE");
    hemiTree = new TTree("hemiTree","Tree for hemishpere mixing");
  }else{
    hemiFile = TFile::Open((fileName).c_str() , "READ");
    //if(m_debug) cout << " hemisphereMixTool::Read file " << hemiFile << endl;
    hemiTree = (TTree*)hemiFile->Get("hemiTree");
    //if(m_debug) cout << " hemisphereMixTool::Got Tree " << hemiTree << endl;
  }

      
  initBranches();

  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "runNumber",   &m_Run, "i");
  connectBranch<ULong64_t>  (m_createLibrary, hemiTree, "evtNumber",   &m_Event, "l");
  connectBranch<float>      (m_createLibrary, hemiTree, "tAxis_x",     &m_tAxis_x, "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "tAxis_y",     &m_tAxis_y, "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "sumPz",       &m_sumPz         , "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "sumPt_T",     &m_sumPt_T      , "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "sumPt_Ta",    &m_sumPt_Ta    , "F");
  connectBranch<float>      (m_createLibrary, hemiTree, "combinedMass",&m_combinedMass  , "F");
  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "NJets",       &m_NJets, "i");  
  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "NBJets",      &m_NBJets, "i");  
  connectBranch<UInt_t>     (m_createLibrary, hemiTree, "pairIdx",     &m_pairIdx, "i");  


  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_pt",   &m_jet_pt);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_eta",  &m_jet_eta);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_phi",  &m_jet_phi);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_m",    &m_jet_m);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_e",    &m_jet_e);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_bRegCorr",    &m_jet_bRegCorr);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_deepB",    &m_jet_deepB);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_CSVv2",    &m_jet_CSVv2);
  connectVecBranch<float> (m_createLibrary, hemiTree,  "jet_deepFlavB",    &m_jet_deepFlavB);
  connectVecBranch<Bool_t>(m_createLibrary, hemiTree, "jet_isTag",    &m_jet_isTag);


}


void hemiDataHandler::initBranches() {
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


hemisphere hemiDataHandler::getHemi(unsigned int entry){
  if(debug) cout << "In hemiDataHandler::getHemi " << endl;
  hemiTree->GetEntry(entry);
  hemisphere outHemi = hemisphere(m_Run, m_Event, m_tAxis_x, m_tAxis_y);
  outHemi.sumPz = m_sumPz;
  outHemi.sumPt_T = m_sumPt_T;
  outHemi.sumPt_Ta = m_sumPt_Ta;
  outHemi.combinedMass = m_combinedMass;
  outHemi.NJets = m_NJets;
  outHemi.NBJets = m_NBJets;
  outHemi.pairIdx = m_pairIdx;
  if(debug) cout << "Leave hemiDataHandler::getHemi " << endl;
  return outHemi;
}

hemisphere hemiDataHandler::getHemiNearNeig(const hemisphere& hIn, unsigned int entry){
  if(debug) cout << "In hemiDataHandler::getHemiNearNeig " << endl;

  if(!m_kdTree){
    cout << "Warning no KDtree defined (" << hIn.NJets << " " << hIn.NBJets << " ) " << endl;
    return hemisphere(0,0,0,0);
  }

  //
  //  Local index of this event
  //
  // (porbably want to precompute these local distances)
  // Might not be needed now....
  //int localIndex = std::distance(m_EventIndex.begin(), find(m_EventIndex.begin(), m_EventIndex.end(), entry));

  //
  //  Get the nearest
  //
  int    nearestIdx[1];
  double nearestDist[1];
  m_kdTree->nnearest(entry,nearestIdx,nearestDist,1);

  //cout << "closest distance is " << nearestDist[0] << " (" << hIn.NJets << " " << hIn.NBJets << " ) " << localIndex << endl;

  if(nearestDist[0] > 10){
    cout << "ERROR: closest distance is " << nearestDist[0] << " (" << hIn.NJets << " " << hIn.NBJets << " ) " << entry << endl;
    return hemisphere(0,0,0,0);
  }

  //if(debug) cout << "Getting globalIdxOfMatch " << endl;  
  //long int globalIdxOfMatch = m_EventIndex.at(nearestIdx[0]);

  //return hemisphere(0,0,0,0);
  if(debug) cout << "Leave hemiDataHandler::getHemiNearNeig " << endl;
  return this->getHemi(nearestIdx[0]);
}


void hemiDataHandler::buildData(){
  this->calcVariance();

  m_hemiPoints.clear();

  //
  //  Sort by event index
  // 
  unsigned int nHemis = hemiTree->GetEntries();
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    hemisphere thisHemi = this->getHemi(hemiIdx);
    
    m_hemiPoints.push_back(hemiPoint(thisHemi.sumPz       /m_varV.x[0], 
				     thisHemi.sumPt_T     /m_varV.x[1], 
				     thisHemi.sumPt_Ta    /m_varV.x[2], 
				     thisHemi.combinedMass/m_varV.x[3])
			   );

  }

  //
  //  Make the kd-trees
  // 
  if(m_hemiPoints.size() < 3) return;
  cout << " Making KD tree " << m_hemiPoints.size() << endl;
  m_kdTree = new hemiKDtree(m_hemiPoints);
  cout << " Done " << endl;
}

void hemiDataHandler::calcVariance(){

  m_sumV  = hemiPoint(0,0,0,0);
  m_sumV2 = hemiPoint(0,0,0,0);
  m_nTot  = 0;
  m_varV  = hemiPoint(0,0,0,0);

  unsigned int nHemis = hemiTree->GetEntries();
  for(long int hemiIdx = 0; hemiIdx < nHemis; hemiIdx++){
    hemisphere thisHemi = this->getHemi(hemiIdx);
    
    m_nTot += 1;

    m_sumV.x[0] += thisHemi.sumPz;
    m_sumV.x[1] += thisHemi.sumPt_T;
    m_sumV.x[2] += thisHemi.sumPt_Ta;
    m_sumV.x[3] += thisHemi.combinedMass;

    m_sumV2.x[0] += (thisHemi.sumPz        * thisHemi.sumPz);       
    m_sumV2.x[1] += (thisHemi.sumPt_T      * thisHemi.sumPt_T);     
    m_sumV2.x[2] += (thisHemi.sumPt_Ta     * thisHemi.sumPt_Ta);    
    m_sumV2.x[3] += (thisHemi.combinedMass * thisHemi.combinedMass);
  }


  m_varV.x[0] = sqrt((m_sumV2.x[0] - (m_sumV.x[0]*m_sumV.x[0])/m_nTot)/m_nTot);
  m_varV.x[1] = sqrt((m_sumV2.x[1] - (m_sumV.x[1]*m_sumV.x[1])/m_nTot)/m_nTot);
  m_varV.x[2] = sqrt((m_sumV2.x[2] - (m_sumV.x[2]*m_sumV.x[2])/m_nTot)/m_nTot);
  m_varV.x[3] = sqrt((m_sumV2.x[3] - (m_sumV.x[3]*m_sumV.x[3])/m_nTot)/m_nTot);
  cout << " var for " << m_nJetBin << "/" << m_nBJetBin << ": " 
       << m_varV.x[0] << " / " 
       << m_varV.x[1] << " / " 
       << m_varV.x[2] << " / " 
       << m_varV.x[3] 
       << endl;
  

  return;
}



void hemisphere::write(hemisphereMixTool* hMixTool, int localPairIndex){

  UInt_t nJets = (tagJets.size() + nonTagJets.size());
  UInt_t nBJets = tagJets.size();
  hemisphereMixTool::EventID thisEventID = { {int(nJets), int(nBJets)} };

  hemiDataHandler* dataHandler = hMixTool->getDataHandler(thisEventID);

  dataHandler->clearBranches();

  dataHandler->m_Run   = Run;
  dataHandler->m_Event = Event;
  dataHandler->m_tAxis_x = thrustAxis.X();
  dataHandler->m_tAxis_y = thrustAxis.Y();
  dataHandler->m_sumPz        = sumPz;
  dataHandler->m_sumPt_T 	   = sumPt_T;
  dataHandler->m_sumPt_Ta	   = sumPt_Ta;
  dataHandler->m_combinedMass = combinedMass;
  dataHandler->m_NJets = nJets;
  dataHandler->m_NBJets = tagJets.size();
  
  if(hMixTool->m_debug) cout << " \t pairIdx " << dataHandler->hemiTree->GetEntries() + localPairIndex << endl;
  dataHandler->m_pairIdx = (dataHandler->hemiTree->GetEntries() + localPairIndex);
  

  for(const jetPtr& tagJet : tagJets){
    dataHandler->m_jet_pt        ->push_back(tagJet->pt);
    dataHandler->m_jet_eta       ->push_back(tagJet->eta);  
    dataHandler->m_jet_phi       ->push_back(tagJet->phi);  
    dataHandler->m_jet_m         ->push_back(tagJet->m);  
    dataHandler->m_jet_e         ->push_back(tagJet->e);  
    dataHandler->m_jet_bRegCorr  ->push_back(tagJet->bRegCorr);  
    dataHandler->m_jet_deepB     ->push_back(tagJet->deepB);  
    dataHandler->m_jet_CSVv2     ->push_back(tagJet->CSVv2);  
    dataHandler->m_jet_deepFlavB ->push_back(tagJet->deepFlavB);  
    dataHandler->m_jet_isTag     ->push_back(true);

  }

  for(const jetPtr& nonTagJet : nonTagJets){
    dataHandler->m_jet_pt        ->push_back(nonTagJet->pt);
    dataHandler->m_jet_eta       ->push_back(nonTagJet->eta);  
    dataHandler->m_jet_phi       ->push_back(nonTagJet->phi);  
    dataHandler->m_jet_m         ->push_back(nonTagJet->m);  
    dataHandler->m_jet_e         ->push_back(nonTagJet->e);  
    dataHandler->m_jet_bRegCorr  ->push_back(nonTagJet->bRegCorr);  
    dataHandler->m_jet_deepB     ->push_back(nonTagJet->deepB);  
    dataHandler->m_jet_CSVv2     ->push_back(nonTagJet->CSVv2);  
    dataHandler->m_jet_deepFlavB ->push_back(nonTagJet->deepFlavB);  
    dataHandler->m_jet_isTag     ->push_back(false);
  }


  dataHandler->hemiTree->Fill();

}


hemisphereMixTool::hemisphereMixTool(std::string name, std::string outputFile, std::vector<std::string> inputFiles, bool fCreateLibrary, fwlite::TFileService& fs, bool debug) {
  if(m_debug) cout << " hemisphereMixTool::In hemisphereMixTool " << name << endl;  
  m_name = name;
  m_outputFileName = outputFile;
  m_inputFileNames = inputFiles;
  m_debug = debug;
  createLibrary = fCreateLibrary;

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

  if(!createLibrary){
    makeIndexing();
  }

} 





void hemisphereMixTool::addEvent(eventData* event){

  //
  //  Calculate Thrust Axis
  //
  m_thrustAxis = getThrustAxis(event);

  //
  //  Make Hemispheres
  //
  hemisphere posHemi(event->run, event->event, m_thrustAxis.X(), m_thrustAxis.Y());
  hemisphere negHemi(event->run, event->event, m_thrustAxis.X(), m_thrustAxis.Y());

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

  for (std::pair<EventID, hemiDataHandler*> element : m_dataHandleIndex) {
    hemiDataHandler* thisHandler = element.second;
    thisHandler->hemiFile->Write();
    thisHandler->hemiFile->Close();
  }

  if(makeEventDisplays)
    eventDisplay->Write("EventDisplay_"+m_name+".txt");

  return;
}





void hemisphereMixTool::makeIndexing(){

  //
  //  Calculate the variances for each event Indx
  //
  // loop on the input files 
  for(std::string inFile : m_inputFileNames){

    cout << "Processing file: " << inFile << endl;
    std::vector<std::string> results;

    boost::algorithm::split(results, inFile, boost::algorithm::is_any_of("_."));
    
    //for(std::string res : results){
    //  cout << "fourn " << res  << endl;
    //}
    std::string nJetStr = results.at(results.size()-3);
    std::string nBJetStr = results.at(results.size()-2);
    std::stringstream ss;
    ss  << nJetStr << "_" << nBJetStr;
    std::string eventIdxStr = ss.str();
    cout << "Found res: " << eventIdxStr << endl;
    hemisphereMixTool::EventID thisEventID = { {std::stoi(nJetStr), std::stoi(nBJetStr)} };
    
    hemiDataHandler* thisDataHandler = getDataHandler(thisEventID,inFile);

    thisDataHandler->buildData();

  }// loop on files
}






hemiDataHandler* hemisphereMixTool::getDataHandler(EventID thisEventID, std::string inFileName){
  if(m_dataHandleIndex.find(thisEventID) == m_dataHandleIndex.end()){
    if(!createLibrary)
      m_dataHandleIndex.insert(std::make_pair(thisEventID, new hemiDataHandler(thisEventID.at(0), thisEventID.at(1), createLibrary, inFileName, m_name) ));
    else
      m_dataHandleIndex.insert(std::make_pair(thisEventID, new hemiDataHandler(thisEventID.at(0), thisEventID.at(1), createLibrary, m_outputFileName, m_name) ));
  }
  return m_dataHandleIndex[thisEventID];
}
