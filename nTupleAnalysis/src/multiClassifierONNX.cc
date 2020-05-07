//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/multiClassifierONNX.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

using namespace nTupleAnalysis;

multiClassifierONNX::multiClassifierONNX(std::string modelFile) {
  std::cout << "multiClassifierONNX( "<<modelFile<<" )" << std::endl;
  
  Ort::SessionOptions* session_options = new Ort::SessionOptions();
  session_options->SetIntraOpNumThreads(1);

  model = std::make_unique<cms::Ort::ONNXRuntime>(modelFile, session_options);

  input_names = {"J","O","D","Q"};
  output_names= {"output"};

  data.clear();

  // canJets
  data.emplace_back(12*4,1);

  // othJets
  data.emplace_back(12*5,1);
  
  // dijets
  data.emplace_back(6*2,1);
  
  // quadjets
  data.emplace_back(3*6,1);
  
  output = model->run(input_names, data, output_names, 1)[0];

  std::cout << "test output ";
  for(std::vector<float>::size_type i=0; i<output.size(); i++){
    std::cout << output[i] << " ";
    std::cout << std::endl;
  };

} 

void multiClassifierONNX::loadData(eventData* event){
  data.clear();
  output.clear();

  // canJets
  data.emplace_back(12*4,0);
  int j = 0;
  for(int i: canJetImageIndicies){
    data[0][   j] = event->canJets[i]->pt;
    data[0][12+j] = event->canJets[i]->eta;
    data[0][24+j] = event->canJets[i]->phi;
    data[0][36+j] = event->canJets[i]->m;
    j++;
  };

  // othJets
  data.emplace_back(12*5,0);
  for(uint i=0; i<event->nAllNotCanJets; i++){
    data[1][   i] = event->allNotCanJets[i]->pt;
    data[1][12+i] = event->allNotCanJets[i]->eta;
    data[1][24+i] = event->allNotCanJets[i]->phi;
    data[1][36+i] = event->allNotCanJets[i]->m;
    bool isSelJet = (event->allNotCanJets[i]->pt>40) & (fabs(event->allNotCanJets[i]->eta)<2.4);
    data[1][48+i] = isSelJet ? 1 : 0; 
  };
  for(uint i=event->nAllNotCanJets; i<12; i++){
    data[1][48+i] = -1;
  };

  // dijets
  data.emplace_back(6*2,0);
  for(int i=0; i<6; i++){
    data[2][  i] = event->dijets[i]->m;
    data[2][6+i] = event->dijets[i]->dR;
  };

  // self.quadjetAncillaryFeatures=['dR0123', 'dR0213', 'dR0312',
  // 	          		    'm4j',    'm4j',    'm4j',
  // 				    'xW',     'xW',     'xW',
  // 			  	    'xbW',    'xbW',    'xbW',
  // 				    'nSelJets', 'nSelJets', 'nSelJets',
  // 				    'year',   'year',   'year',
  //                                ]
  // quadjets
  data.emplace_back(3*6,0);
  data[3][0] = event->dR0123;
  data[3][1] = event->dR0213;
  data[3][2] = event->dR0312;
  for(int i=0; i<3; i++){
    data[3][ 3+i] = event->m4j;
    data[3][ 6+i] = event->xW;
    data[3][ 9+i] = event->xbW;
    data[3][12+i] = event->nSelJets;
    data[3][15+i] = event->year;
  };
  
}


multiClassifierONNX::~multiClassifierONNX(){} 

