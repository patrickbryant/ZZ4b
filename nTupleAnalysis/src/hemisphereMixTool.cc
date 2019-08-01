


#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"
#include <boost/algorithm/string.hpp>


using namespace nTupleAnalysis;
using std::vector; using std::endl; using std::cout;



hemisphereMixTool::hemisphereMixTool(std::string name, std::string outputFile, std::vector<std::string> inputFiles, bool createLibrary, fwlite::TFileService& fs, bool debug, bool loadJetFourVecs, bool dualAccess) {
  if(m_debug) cout << " hemisphereMixTool::In hemisphereMixTool " << name << endl;  
  m_name = name;
  m_outputFileName = outputFile;
  m_inputFileNames = inputFiles;
  m_debug = debug;
  m_createLibrary = createLibrary;
  m_loadJetFourVecs = loadJetFourVecs;
  m_dualAccess = dualAccess;

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

  if(!m_createLibrary){
    makeIndexing();
  }

} 





void hemisphereMixTool::addEvent(eventData* event){

  //
  //  Calculate Thrust Axis
  //
  TVector2 thrustAxis = getThrustAxis(event);

  //
  //  Make Hemispheres
  //
  hemiPtr posHemi = std::make_shared<hemisphere>(hemisphere(event->run, event->event, thrustAxis.X(), thrustAxis.Y()));
  hemiPtr negHemi = std::make_shared<hemisphere>(hemisphere(event->run, event->event, thrustAxis.X(), thrustAxis.Y()));

  for(const jetPtr& thisJet : event->allJets){
    TVector2 thisJetPt2 = TVector2(thisJet->p.Px(),thisJet->p.Py());
    if( (thisJetPt2 * thrustAxis ) > 0) posHemi->addJet(thisJet, event->selJets, event->tagJets);
    else                                negHemi->addJet(thisJet, event->selJets, event->tagJets);
  }


  if(makeEventDisplays){
    for(const jetPtr& jetData : posHemi->tagJets)    eventDisplay->AddJet("posHemiJets",jetData);
    for(const jetPtr& jetData : posHemi->nonTagJets) eventDisplay->AddJet("posHemiJets",jetData);
    for(const jetPtr& jetData : negHemi->tagJets)    eventDisplay->AddJet("negHemiJets",jetData);
    for(const jetPtr& jetData : negHemi->nonTagJets) eventDisplay->AddJet("negHemiJets",jetData);
  }


  //
  //  Fill some histograms
  //
  FillHists(posHemi, negHemi);

  //
  // write to output tree
  //
  if(m_debug) cout << " Write pos hemei " << endl;
  posHemi->write(this,  1);
  if(m_debug) cout << " Write neg hemei " << endl;
  negHemi->write(this, -1);

  if(makeEventDisplays){
    eventDisplay->AddEventVar("thrustPhi", thrustAxis.Phi());
    eventDisplay->NewEvent();
  }

  return;
}

int hemisphereMixTool::makeArtificialEvent(eventData* event){

  //
  //  Calculate Thrust Axis
  //
  TVector2 thrustAxis = getThrustAxis(event);

  //
  //  Make Hemispheres
  //
  //std::make_shared<hemisphere>(hemisphere(m_Run, m_Event, m_tAxis_x, m_tAxis_y));
  hemiPtr posHemi = std::make_shared<hemisphere>(hemisphere(event->run, event->event, thrustAxis.X(), thrustAxis.Y()));
  hemiPtr negHemi = std::make_shared<hemisphere>(hemisphere(event->run, event->event, thrustAxis.X(), thrustAxis.Y()));

  for(const jetPtr& thisJet : event->allJets){
    TVector2 thisJetPt2 = TVector2(thisJet->p.Px(),thisJet->p.Py());
    if( (thisJetPt2 * thrustAxis ) > 0) posHemi->addJet(thisJet, event->selJets, event->tagJets);
    else                                negHemi->addJet(thisJet, event->selJets, event->tagJets);
  }

  //
  //  Get the Nearest Neighbor Hemis
  //

  // pos Hemi
  hemisphereMixTool::EventID posEventID = { {int(posHemi->tagJets.size()+posHemi->nonTagJets.size()), int(posHemi->tagJets.size()), int(posHemi->nonSelJets.size())} };
  hemiDataHandler* posDataHandle = getDataHandler(posEventID);

  // logic for making sure it makes sense to look up a new hemi
  if(!posDataHandle) return -1;
  if(!posDataHandle->m_isValid) return -2;

  // neg Hemi
  hemisphereMixTool::EventID negEventID = { {int(negHemi->tagJets.size()+negHemi->nonTagJets.size()), int(negHemi->tagJets.size()), int(negHemi->nonSelJets.size())  } };
  hemiDataHandler* negDataHandle = getDataHandler(negEventID);

  // logic for making sure it makes sense to look up a new hemi
  if(!negDataHandle) return -1;
  if(!negDataHandle->m_isValid) return -2;

  //get best matches
  hemiPtr posHemiBestMatch = posDataHandle->getHemiNearNeig(posHemi, true);
  hemiPtr negHemiBestMatch = negDataHandle->getHemiNearNeig(negHemi, true);

  std::vector<nTupleAnalysis::jetPtr> new_allJets;
  for(const nTupleAnalysis::jetPtr& pos_jet: posHemiBestMatch->allJets){
    new_allJets.push_back(pos_jet);
  }
  for(const nTupleAnalysis::jetPtr& neg_jet: negHemiBestMatch->allJets){
    new_allJets.push_back(neg_jet);
  }
  
  event->makeNewEvent(new_allJets);

  return 0;
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

void hemisphereMixTool::FillHists(const hemiPtr& posHemi, const hemiPtr& negHemi ){
  FillHists(posHemi);
  FillHists(negHemi);

  hdelta_NJets    ->Fill( (posHemi->tagJets.size() + posHemi->nonTagJets.size())  -  (negHemi->tagJets.size() + negHemi->nonTagJets.size()));
  hdelta_NBJets   ->Fill(  posHemi->tagJets.size() -  negHemi->tagJets.size());
  hdelta_Pz       ->Fill( posHemi->sumPz        - negHemi->sumPz         );
  hdelta_SumPt_T  ->Fill( posHemi->sumPt_T      - negHemi->sumPt_T       );
  hdelta_SumPt_Ta ->Fill( posHemi->sumPt_Ta     - negHemi->sumPt_Ta      );
  hdelta_CombMass ->Fill( posHemi->combinedMass - negHemi->combinedMass  );

}


void hemisphereMixTool::FillHists(const hemiPtr& hIn){
  hNJets ->Fill((hIn->tagJets.size() + hIn->nonTagJets.size()));
  hNBJets->Fill( hIn->tagJets.size() );
  hPz       ->Fill( hIn->sumPz);
  hSumPt_T  ->Fill( hIn->sumPt_T);
  hSumPt_Ta ->Fill( hIn->sumPt_Ta);
  hCombMass ->Fill( hIn->combinedMass);

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
    std::string nJetStr = results.at(results.size()-4);
    std::string nBJetStr = results.at(results.size()-3);
    std::string nNonSelJetStr = results.at(results.size()-2);
    std::stringstream ss;
    ss  << nJetStr << "_" << nBJetStr << "_" << nNonSelJetStr;
    std::string eventIdxStr = ss.str();
    cout << "Found res: " << eventIdxStr << endl;
    hemisphereMixTool::EventID thisEventID = { {std::stoi(nJetStr), std::stoi(nBJetStr), std::stoi(nNonSelJetStr)} };

    if(m_debug) cout << "Get Datahandler: " << endl;    
    hemiDataHandler* thisDataHandler = getDataHandler(thisEventID,inFile);
    
    if(m_debug) cout << "Building data: " << endl;
    thisDataHandler->buildData();

  }// loop on files
}






hemiDataHandler* hemisphereMixTool::getDataHandler(EventID thisEventID, std::string inputFile){
  if(m_dataHandleIndex.find(thisEventID) == m_dataHandleIndex.end()){
    if(!m_createLibrary){
      if(inputFile == ""){
	return nullptr;
      }else{
	if(m_debug) cout << "Making new hemiDataHandler: " << endl;
	m_dataHandleIndex.insert(std::make_pair(thisEventID, new hemiDataHandler(thisEventID.at(0), thisEventID.at(1), thisEventID.at(2), m_createLibrary, inputFile, m_name, m_loadJetFourVecs, m_dualAccess, m_debug) ));
      }
    }else{
      if(m_debug) cout << "hemisphereMixTool::getDataHandler making new dataHandler (filename: " << m_outputFileName << ")" << endl;
      m_dataHandleIndex.insert(std::make_pair(thisEventID, new hemiDataHandler(thisEventID.at(0), thisEventID.at(1), thisEventID.at(2), m_createLibrary, m_outputFileName, m_name, m_loadJetFourVecs, m_dualAccess, m_debug) ));
    }
  }
  return m_dataHandleIndex[thisEventID];
}
