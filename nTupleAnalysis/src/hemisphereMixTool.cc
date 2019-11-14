#include "ZZ4b/nTupleAnalysis/interface/hemisphereMixTool.h"
#include "ZZ4b/nTupleAnalysis/interface/hemiDataHandler.h"
#include "nTupleHelperTools/baseClasses/interface/thrust.h"
#include <boost/algorithm/string.hpp>


using namespace nTupleAnalysis;
using std::vector; using std::endl; using std::cout;



hemisphereMixTool::hemisphereMixTool(std::string name, std::string outputFile, std::vector<std::string> inputFiles, bool createLibrary, fwlite::TFileService& fs, int maxNHemis, bool debug, bool loadJetFourVecs, bool dualAccess) {
  if(m_debug) cout << " hemisphereMixTool::In hemisphereMixTool " << name << endl;  
  m_name = name;
  m_outputFileName = outputFile;
  m_inputFileNames = inputFiles;
  m_debug = debug;
  m_createLibrary = createLibrary;
  m_loadJetFourVecs = loadJetFourVecs;
  m_dualAccess = dualAccess;
  m_maxNHemis = maxNHemis;

  //
  // Create Histograms
  //
  dir = fs.mkdir("hMix_"+name);
  hHists = new hemiHists(name, dir);
  hSameEventCheck  = dir.make<TH1F>("hSameEvent",  (name+"/sameEvent;  ;Entries").c_str(),  2,-0.5,1.5);  
  hNHemisFetched   = dir.make<TH1F>("hNHemisFetched",  (name+"/NHemisFetched;  ;Entries").c_str(),  20,-0.5,19.5);  
  hCode            = dir.make<TH1F>("hCode",         (name+"/Code;  ;Entries").c_str(),  10,-0.5,9.5);  

  //
  // json files for Event Displays
  //
  //
  makeEventDisplays = false;
  if(makeEventDisplays){
    eventDisplay = new EventDisplayData("Events");
    eventDisplay->addJetCollection("posHemiJets");
    eventDisplay->addJetCollection("negHemiJets");
    eventDisplay->addEventVar     ("thrustPhi");
  }

  if(!m_createLibrary){
    makeIndexing();

    trigEmulator = new TriggerEmulator::TrigEmulatorTool("trigEmulator", 1, 1);
    trigEmulator->AddTrig("EMU_HT330_4j", "330ZH", {"75","60","45","40"}, {1,2,3,4});
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
  hHists->Fill(posHemi, nullptr);
  hHists->Fill(negHemi, nullptr);
  hHists->hDiffNN->Fill(posHemi, negHemi, nullptr);

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

  if(m_debug) cout << "In makeArtificialEvent " << endl;
  
  //
  //  Calculate Thrust Axis
  //
  TVector2 thrustAxis = getThrustAxis(event);
  
  //  std::cout << "Initial thrust " << thrustAxis.Phi() << std::endl;

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
  int posNJets  = posHemi->tagJets.size()+posHemi->nonTagJets.size();
  int posNBJets = posHemi->tagJets.size();
  int posNNonSelJets = posHemi->nonSelJets.size();
  EventID posEventID = { {int(posNJets), int(posNBJets), int(posNNonSelJets)} };
  hemiDataHandler* posDataHandle = getDataHandler(posEventID);

  // logic for making sure it makes sense to look up a new hemi
  if(!posDataHandle) {
    m_h1_matchCode = -1;
    m_h2_matchCode = -1;
    hCode->Fill(-1);
    return -1;
  }
  if(!posDataHandle->m_isValid){
    m_h1_matchCode = -2;
    m_h2_matchCode = -2;
    hCode->Fill(-2);
    return -2;
  }

  // neg Hemi
  int negNJets  = negHemi->tagJets.size()+negHemi->nonTagJets.size();
  int negNBJets = negHemi->tagJets.size();
  int negNNonSelJets = negHemi->nonSelJets.size();
  EventID negEventID = { {int(negNJets), int(negNBJets), int(negNNonSelJets)} };
  hemiDataHandler* negDataHandle = getDataHandler(negEventID);

  // logic for making sure it makes sense to look up a new hemi
  if(!negDataHandle) {
    m_h1_matchCode = -3;
    m_h2_matchCode = -3;
    hCode->Fill(-3);
    return -3;
  }
  if(!negDataHandle->m_isValid) {
    m_h1_matchCode = -4;
    m_h2_matchCode = -4;
    hCode->Fill(-4);
    return -4;
  }

  //
  // get best matches
  //
  if(m_debug) cout << "\t Getting best matches " << endl;
  double posMatchDistance = 1e6;
  double negMatchDistance = 1e6;
  hemiPtr posHemiBestMatch = nullptr;//posDataHandle->getHemiNearNeig(posHemi, posMatchDistance, true);
  hemiPtr negHemiBestMatch = nullptr;//negDataHandle->getHemiNearNeig(negHemi, negMatchDistance, true);

  bool passTrig = false;
  unsigned int nHemisFetched = 0;
  unsigned int nHemisFetched_pos = 0;
  unsigned int nHemisFetched_neg = 0;
  
  while(!passTrig && (nHemisFetched < 11)){

    //
    // Get the Matching Hemis
    //
    if(!nHemisFetched){
      if(m_debug) cout << "\t Got best matches " << endl;
      posHemiBestMatch = posDataHandle->getHemiNearNeig(posHemi, posMatchDistance, true);
      negHemiBestMatch = negDataHandle->getHemiNearNeig(negHemi, negMatchDistance, true);
      ++nHemisFetched_pos;
      ++nHemisFetched_neg;
      nHemisFetched += 2;
    
    }else{
      if(nHemisFetched_pos > nHemisFetched_neg){
	if(m_debug) cout << "\t getting kth neg " << nHemisFetched_neg << endl;
	negHemiBestMatch = negDataHandle->getHemiKthNearNeig(negHemi, nHemisFetched_neg, negMatchDistance, true);
	++nHemisFetched_neg;
	++nHemisFetched;
      }else{
	if(m_debug) cout << "\t getting kth pos " << nHemisFetched_pos << endl;
	posHemiBestMatch = posDataHandle->getHemiKthNearNeig(posHemi, nHemisFetched_pos, posMatchDistance, true);
	++nHemisFetched_pos;
	++nHemisFetched;
      }

    }

    if(m_debug) cout << "\t nHemisFetched " << nHemisFetched << " " << nHemisFetched_pos << "/" << nHemisFetched_neg << endl;

    //
    //  Rotate thrust axis to match
    //
    posHemiBestMatch->rotateTo(thrustAxis, true );
    negHemiBestMatch->rotateTo(thrustAxis, false);

    float posHemiVal =(thrustAxis * TVector2(posHemiBestMatch->combinedVec.Px(), posHemiBestMatch->combinedVec.Py()));
    float negHemiVal =(thrustAxis * TVector2(negHemiBestMatch->combinedVec.Px(), negHemiBestMatch->combinedVec.Py()));
    if((posHemiVal * negHemiVal ) > 0)
      cout << posHemiVal << " " << negHemiVal << endl;

    std::vector<nTupleAnalysis::jetPtr> new_allJets;
    for(const nTupleAnalysis::jetPtr& pos_jet: posHemiBestMatch->tagJets){
      new_allJets.push_back(pos_jet);
    }
    for(const nTupleAnalysis::jetPtr& pos_jet: posHemiBestMatch->nonTagJets){
      new_allJets.push_back(pos_jet);
    }
    for(const nTupleAnalysis::jetPtr& pos_jet: posHemiBestMatch->nonSelJets){
      new_allJets.push_back(pos_jet);
    }

    for(const nTupleAnalysis::jetPtr& neg_jet: negHemiBestMatch->tagJets){
      new_allJets.push_back(neg_jet);
    }
    for(const nTupleAnalysis::jetPtr& neg_jet: negHemiBestMatch->nonTagJets){
      new_allJets.push_back(neg_jet);
    }
    for(const nTupleAnalysis::jetPtr& neg_jet: negHemiBestMatch->nonSelJets){
      new_allJets.push_back(neg_jet);
    }
  
    //
    //  Make the new event and Debuging on Error
    //
    if(event->makeNewEvent(new_allJets) < 0){
      cout << " Old Event posHemi " << posNJets << " / " << posNBJets << " / " << posNNonSelJets << endl; 
      cout << " Old Event negHemi " << negNJets << " / " << negNBJets << " / " << negNNonSelJets << endl; 
      cout << " New Event posHemi " << posHemiBestMatch->NJets << " / " << posHemiBestMatch->NBJets << " / " << posHemiBestMatch->NNonSelJets << endl; 
      cout << " New Event negHemi " << negHemiBestMatch->NJets << " / " << negHemiBestMatch->NBJets << " / " << negHemiBestMatch->NNonSelJets << endl; 
    }

    //
    // Check trigger
    //
    if(false){

      //
      // Set Decision
      //
      vector<float> selJet_pts;
      for(const jetPtr& sJet : event->selJets){
	selJet_pts.push_back(sJet->pt_wo_bRegCorr);
      }

      vector<float> tagJet_pts;
      trigEmulator->SetDecisions(selJet_pts, tagJet_pts, event->ht30);

      //
      // Pass Trig
      //
      passTrig = trigEmulator->GetDecision("EMU_HT330_4j");
      if(m_debug) cout << "\t passTrig " << passTrig << endl;

    }else{
      passTrig = true;
    }
    
  }
  if(m_debug) cout << "End while loop " << endl;

  if(!passTrig){
    cout << "WARNING::Mixed hemisphere failed trigger " << endl;
    cout << " New Event posHemi " << posHemiBestMatch->NJets << " / " << posHemiBestMatch->NBJets << " / " << posHemiBestMatch->NNonSelJets << endl; 
    cout << " New Event negHemi " << negHemiBestMatch->NJets << " / " << negHemiBestMatch->NBJets << " / " << negHemiBestMatch->NNonSelJets << endl; 
    event->passHLT = false;
  }

  //
  //  Histograms
  //
  hHists->Fill(posHemi, posDataHandle);
  hHists->hDiffNN->Fill(posHemi, posHemiBestMatch, posDataHandle);

  hHists->Fill(negHemi, negDataHandle);
  hHists->hDiffNN->Fill(negHemi, negHemiBestMatch, negDataHandle);

  hNHemisFetched->Fill(nHemisFetched);

  if( (posHemiBestMatch->Event == negHemiBestMatch->Event) && (posHemiBestMatch->Run == negHemiBestMatch->Run) ){
    hSameEventCheck->Fill(1);
  }else{
    hSameEventCheck->Fill(0);
  }



  //
  //  pico AODS
  //
  m_h1_run                = posHemiBestMatch->Run;
  m_h1_event              = posHemiBestMatch->Event;
  m_h1_NJet               = posHemiBestMatch->NJets;
  m_h1_NBJet              = posHemiBestMatch->NBJets;
  m_h1_NNonSelJet         = posHemiBestMatch->NNonSelJets;
  m_h1_matchCode          = 0;
  m_h1_pz                 = posHemiBestMatch->sumPz;
  m_h1_pz_sig             = posHemiBestMatch->sumPz/posDataHandle->m_varV.x[0];
  m_h1_match_pz           = posHemi->sumPz;
  m_h1_sumpt_t            = posHemiBestMatch->sumPt_T;
  m_h1_sumpt_t_sig        = posHemiBestMatch->sumPt_T/posDataHandle->m_varV.x[1];
  m_h1_match_sumpt_t      = posHemi->sumPt_T;
  m_h1_sumpt_ta           = posHemiBestMatch->sumPt_Ta;
  m_h1_sumpt_ta_sig       = posHemiBestMatch->sumPt_Ta/posDataHandle->m_varV.x[2];
  m_h1_match_sumpt_ta     = posHemi->sumPt_Ta;
  m_h1_combinedMass       = posHemiBestMatch->combinedMass;
  m_h1_combinedMass_sig   = posHemiBestMatch->combinedMass/posDataHandle->m_varV.x[3];
  m_h1_match_combinedMass = posHemi->combinedMass;
  m_h1_match_dist         = posMatchDistance;


  m_h2_run                = negHemiBestMatch->Run;
  m_h2_event              = negHemiBestMatch->Event;
  m_h2_NJet               = negHemiBestMatch->NJets;
  m_h2_NBJet              = negHemiBestMatch->NBJets;
  m_h2_NNonSelJet         = negHemiBestMatch->NNonSelJets;
  m_h2_matchCode          = 0;
  m_h2_pz                 = negHemiBestMatch->sumPz;
  m_h2_pz_sig             = negHemiBestMatch->sumPz/negDataHandle->m_varV.x[0];
  m_h2_match_pz           = negHemi->sumPz;
  m_h2_sumpt_t            = negHemiBestMatch->sumPt_T;
  m_h2_sumpt_t_sig        = negHemiBestMatch->sumPt_T/negDataHandle->m_varV.x[1];
  m_h2_match_sumpt_t      = negHemi->sumPt_T;
  m_h2_sumpt_ta           = negHemiBestMatch->sumPt_Ta;
  m_h2_sumpt_ta_sig       = negHemiBestMatch->sumPt_Ta/negDataHandle->m_varV.x[2];
  m_h2_match_sumpt_ta     = negHemi->sumPt_Ta;
  m_h2_combinedMass       = negHemiBestMatch->combinedMass;
  m_h2_combinedMass_sig   = negHemiBestMatch->combinedMass/negDataHandle->m_varV.x[3];
  m_h2_match_combinedMass = negHemi->combinedMass;
  m_h2_match_dist         = negMatchDistance;

  //
  //  Debugging
  //


  //
  //  Calculate Thrust Axis
  //
  //TVector2 thrustAxisNewEvt = getThrustAxis(event);
  
  //float thrustdPhi = thrustAxisNewEvt.DeltaPhi(thrustAxis);
  //if(fabs(thrustdPhi) > 0.1)
  //  std::cout << "New thrust differnece is " << thrustdPhi << std::endl;

  
  //  std::cout << "Initial thrust " << thrustAxis.Phi() << std::endl;

//  //
//  //  Make Hemispheres
//  //
//  //std::make_shared<hemisphere>(hemisphere(m_Run, m_Event, m_tAxis_x, m_tAxis_y));
//  hemiPtr posHemiNew = std::make_shared<hemisphere>(hemisphere(event->run, event->event, thrustAxisNewEvt.X(), thrustAxisNewEvt.Y()));
//  hemiPtr negHemiNew = std::make_shared<hemisphere>(hemisphere(event->run, event->event, thrustAxisNewEvt.X(), thrustAxisNewEvt.Y()));
//
//  for(const jetPtr& thisJet : event->allJets){
//    TVector2 thisJetPt2 = TVector2(thisJet->p.Px(),thisJet->p.Py());
//    if( (thisJetPt2 * thrustAxis ) > 0) posHemiNew->addJet(thisJet, event->selJets, event->tagJets);
//    else                                negHemiNew->addJet(thisJet, event->selJets, event->tagJets);
//  }
//
//
//  std::cout << " posHemi sumPt_T " << posHemi->sumPt_T << " mass "  << posHemi->combinedMass << std::endl;
//  std::cout << " negHemi sumPt_T " << negHemi->sumPt_T << " mass "  << negHemi->combinedMass << std::endl;

  hCode->Fill(0);
  return 0;
}



TVector2 hemisphereMixTool::getThrustAxis(eventData* event){

  vector<TVector2> jetPts;
  for(const jetPtr& thisJet : event->allJets){
    jetPts.push_back(TVector2(thisJet->p.Px(),thisJet->p.Py()));
  }

  return nTupleHelperTools::calcThrust(jetPts, m_debug);
}


//
// Following is From
// https://rivet.hepforge.org/code/1.8.2/a00677_source.html
//





hemisphereMixTool::~hemisphereMixTool(){} 


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

    if(m_debug) cout << "Processing file: " << inFile << endl;
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
    if(m_debug) cout << "Found res: " << eventIdxStr << endl;
    EventID thisEventID = { {std::stoi(nJetStr), std::stoi(nBJetStr), std::stoi(nNonSelJetStr)} };

    if(m_debug) cout << "Get Datahandler: " << endl;    
    hemiDataHandler* thisDataHandler = getDataHandler(thisEventID,inFile);

    if(!thisDataHandler->m_isValid) continue;
    
    cout << "Processing file: " << inFile << endl;
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
	m_dataHandleIndex.insert(std::make_pair(thisEventID, new hemiDataHandler(thisEventID, m_createLibrary, inputFile, m_name, m_maxNHemis, m_loadJetFourVecs, m_dualAccess, m_debug) ));
      }
    }else{
      if(m_debug) cout << "hemisphereMixTool::getDataHandler making new dataHandler (filename: " << m_outputFileName << ")" << endl;
      m_dataHandleIndex.insert(std::make_pair(thisEventID, new hemiDataHandler(thisEventID, m_createLibrary, m_outputFileName, m_name, m_maxNHemis, m_loadJetFourVecs, m_dualAccess, m_debug) ));
    }
  }
  return m_dataHandleIndex[thisEventID];
}
