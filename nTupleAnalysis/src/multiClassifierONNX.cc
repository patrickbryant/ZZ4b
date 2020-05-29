//#include "TChain.h"
#include "ZZ4b/nTupleAnalysis/interface/multiClassifierONNX.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

using namespace nTupleAnalysis;

multiClassifierONNX::multiClassifierONNX(std::string modelFile) {
  std::cout << "multiClassifierONNX( "<<modelFile<<" )" << std::endl;
  
  Ort::SessionOptions* session_options = new Ort::SessionOptions();
  session_options->SetIntraOpNumThreads(1);

  model = std::make_unique<cms::Ort::ONNXRuntime>(modelFile, session_options);

  model->getOutputNames();

  input_names = {"J","O","D","Q"};
  output_names= {"c_score", "q_score"};//model->getOutputNames();//{"Output", "q_score"};
  //for(auto name: output_names) std::cout << name << std::endl;

  this->clear();

  // canJets
  input.emplace_back(12*4,1);

  // othJets
  input.emplace_back(12*5,1);
  
  // dijets
  input.emplace_back(6*2,1);
  
  // quadjets
  input.emplace_back(3*6,1);

  this->run();
  this->dump();

} 

void multiClassifierONNX::clear(){
  input.clear();
  output.clear();
  c_score.clear();
  q_score.clear();
}

void multiClassifierONNX::loadInput(eventData* event){
  // canJets
  input.emplace_back(12*4,0);
  int j = 0;
  for(int i: canJetImageIndicies){
    input[0][   j] = event->canJets[i]->pt;
    input[0][12+j] = event->canJets[i]->eta;
    input[0][24+j] = event->canJets[i]->phi;
    input[0][36+j] = event->canJets[i]->m;
    j++;
  };

  // othJets
  input.emplace_back(12*5,0);
  for(uint i=0; i<event->nAllNotCanJets; i++){
    input[1][   i] = event->allNotCanJets[i]->pt;
    input[1][12+i] = event->allNotCanJets[i]->eta;
    input[1][24+i] = event->allNotCanJets[i]->phi;
    input[1][36+i] = event->allNotCanJets[i]->m;
    bool isSelJet = (event->allNotCanJets[i]->pt>40) & (fabs(event->allNotCanJets[i]->eta)<2.4);
    input[1][48+i] = isSelJet ? 1 : 0; 
  };
  for(uint i=event->nAllNotCanJets; i<12; i++){
    input[1][48+i] = -1;
  };

  // dijets
  input.emplace_back(6*2,0);
  for(int i=0; i<6; i++){
    input[2][  i] = event->dijets[i]->m;
    input[2][6+i] = event->dijets[i]->dR;
  };

  // self.quadjetAncillaryFeatures=['dR0123', 'dR0213', 'dR0312',
  // 	          		    'm4j',    'm4j',    'm4j',
  // 				    'xW',     'xW',     'xW',
  // 			  	    'xbW',    'xbW',    'xbW',
  // 				    'nSelJets', 'nSelJets', 'nSelJets',
  // 				    'year',   'year',   'year',
  //                                ]
  // quadjets
  input.emplace_back(3*6,0);
  input[3][0] = event->dR0123;
  input[3][1] = event->dR0213;
  input[3][2] = event->dR0312;
  for(int i=0; i<3; i++){
    input[3][ 3+i] = event->m4j;
    input[3][ 6+i] = event->xW;
    input[3][ 9+i] = event->xbW;
    input[3][12+i] = event->nSelJets;
    input[3][15+i] = event->year;
  };
  
}

void multiClassifierONNX::run(){
  output = model->run(input_names, input, output_names, 1);
  c_score = output[0];
  q_score = output[1];
}

void multiClassifierONNX::run(eventData* event){
  this->clear();
  this->loadInput(event);
  this->run();
}

void multiClassifierONNX::dump(){
  std::cout << "multiClassifierONNX::dump() inputs" << std::endl;
  for(std::vector<float>::size_type i=0; i<input.size(); i++){
    std::cout << input_names[i] << ": ";
    for(std::vector<float>::size_type j=0; j<input[i].size(); j++){
      std::cout << input[i][j] << " ";
    }
    std::cout << std::endl;
  };
  std::cout << "multiClassifierONNX::dump() outputs" << std::endl;
  for(std::vector<float>::size_type i=0; i<output.size(); i++){
    std::cout << output_names[i] << ": ";
    for(std::vector<float>::size_type j=0; j<output[i].size(); j++){
      std::cout << output[i][j] << " ";
    }
    std::cout << std::endl;
  };
}

multiClassifierONNX::~multiClassifierONNX(){} 

